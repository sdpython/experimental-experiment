import os
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from ._dort_cmd_common_models import (
    _create_configuration_for_benchmark_llama,
    _create_configuration_for_benchmark_mistral,
    _create_configuration_for_benchmark_phi,
)


def get_fused_aten_ops_dispatcher():
    """
    Returns a dispatcher with additional converting function to
    convert fused operators into ATen ops onnxruntime can call.
    """
    from ..torch_interpreter import Dispatcher

    def onnx_scaled_dot_product_efficient_attention(
        g: "GraphBuilder",  # noqa: F821
        sts: Dict[str, Any],
        outputs: List[str],
        query,
        key,
        value,
        attn_bias,
        compute_log_sumexp: bool,
        dropout_p: float,
        is_causal: bool,
        scale: float = 1.0,
        **kwargs,
    ):
        assert (
            len(outputs) == 4
        ), f"Unexpected number of outputs {outputs}{g.get_debug_msg()}"
        assert len(kwargs) == 0, (
            f"Unexpected kwargs {kwargs} in "
            f"onnx_scaled_dot_product_efficient_attention{g.get_debug_msg()}"
        )
        # itype = g.get_type(value)
        # dtype = tensor_dtype_to_np_dtype(itype)
        t_compute_log_sumexp = g.make_initializer(
            "", np.array(compute_log_sumexp, dtype=np.bool_)
        )
        t_dropout_p = g.make_initializer("", np.array(dropout_p, dtype=np.float32))
        t_is_causal = g.make_initializer("", np.array(is_causal, dtype=np.bool_))
        t_scale = g.make_initializer("", np.array(scale or 1.0, dtype=np.float32))
        output, log_sumexp, philox_seed, philox_offset = g.make_node(
            "ATen",
            [
                query,
                key,
                value,
                attn_bias or "",
                t_compute_log_sumexp,
                t_dropout_p,
                t_is_causal,
                t_scale,
            ],
            outputs=outputs,
            operator="_scaled_dot_product_efficient_attention",
            domain="org.pytorch.aten",
            name="scaled_dot_product_efficient_attention",
        )
        g.add_domain("org.pytorch.aten")
        return output, log_sumexp, philox_seed, philox_offset

    def onnx_scaled_dot_product_attention_backward(
        g: "GraphBuilder",  # noqa: F821
        sts: Dict[str, Any],
        outputs: List[str],
        grad,
        query,
        key,
        value,
        attn_bias,
        output,
        logsumexp,
        philox_seed,
        philox_offset,
        dropout_p,
        grad_input_mask,
        is_causal: bool,
        scale: float = 1.0,
        **kwargs,
    ):
        assert (
            len(outputs) == 4
        ), f"Unexpected number of outputs {outputs}{g.get_debug_msg()}"
        assert len(kwargs) == 0, (
            f"Unexpected kwargs {kwargs} in "
            f"onnx_scaled_dot_product_attention_backward{g.get_debug_msg()}"
        )
        t_scale = g.make_initializer("", np.array(scale or 1.0, dtype=np.float32))
        t_dropout_p = g.make_initializer("", np.array(dropout_p, dtype=np.float32))
        t_is_causal = g.make_initializer("", np.array(is_causal, dtype=np.bool_))
        t_grad_input_mask = g.make_initializer(
            "", np.array(grad_input_mask, dtype=np.int64)
        )
        # onnxruntime fails with type inference failed
        # Let's add some Cast even if not needed.
        dt = g.get_type(grad)
        helper = ",".join(map(str, [dt, dt, dt, dt]))
        node_name = f"scaled_dot_product_attention_backward[{helper}]"
        grad_query, grad_key, grad_value, grad_attn_bias = g.make_node(
            "ATen",
            [
                grad,
                query,
                key,
                value,
                attn_bias or "",
                output,
                logsumexp,
                philox_seed,
                philox_offset,
                t_dropout_p,
                t_grad_input_mask,
                t_is_causal,
                t_scale,
            ],
            outputs=outputs,
            operator="_scaled_dot_product_efficient_attention_backward",
            domain="org.pytorch.aten",
            name=node_name,
        )
        g.add_domain("org.pytorch.aten")
        return grad_query, grad_key, grad_value, grad_attn_bias

    dispatcher = Dispatcher(
        {
            "_scaled_dot_product_efficient_attention_default": onnx_scaled_dot_product_efficient_attention,
            "_scaled_dot_product_efficient_attention_backward_default": onnx_scaled_dot_product_attention_backward,
        }
    )
    return dispatcher


def create_compiled_model(
    model: Any,
    backend: str,
    target_opset: int,
    use_dynamic: bool = False,
    verbose: int = 0,
    enable_pattern: Union[str, List[str]] = "default",
    disable_pattern: Union[str, List[str]] = None,
    return_storage: bool = False,
    rename_inputs: bool = True,
    dump_prefix: Optional[str] = None,
    optimize: bool = True,
    use_fused_aten_ops: bool = False,
) -> Any:
    """
    Creates the compilrf model.

    :param model: module
    :param backend: kind of backend
    :param use_dynamic: use dynamic shape
    :param verbose: verbosity
    :param enable_pattern: to enable optimization pattern
    :param disable_pattern: to disable optimization pattern
    :param return_storage: return a container for the models,
        only works with backend *custom* and *debug*
    :param rename_inputs: rename inputs into ``input_{i}``
    :param dump_prefix: dumps the models (backend, custom and debug)
    :param optimize: enable optimizations
    :param use_fused_aten_ops: use fused opetor when converting the model,
        it only works the backend custom
    :return: compiled model
    """
    import torch
    from torch._dynamo.backends.common import aot_autograd
    from experimental_experiment.torch_models.training_helper import make_aot_ort
    from experimental_experiment.torch_dynamo import (
        get_decomposition_table,
        get_decomposition_table_dynamo,
        dynger_backend,
        onnx_custom_backend,
        onnx_debug_backend,
    )

    if use_fused_aten_ops and backend in {"ort", "custom", "backort", "plug"}:
        from onnxruntime.training.ortmodule.torch_cpp_extensions import aten_op_executor
        from onnxruntime.capi import _pybind_state as _C

        _C.register_aten_op_executor(
            str(aten_op_executor.is_tensor_argument_address()),
            str(aten_op_executor.execute_aten_operator_address()),
        )

        dispatcher = get_fused_aten_ops_dispatcher()
    else:
        dispatcher = None

    if backend == "ort":
        assert (
            not return_storage
        ), f"return_storage=True not implemented with backend={backend!r}"
        local_aot_ort, local_ort = make_aot_ort(
            dynamic=use_dynamic, rewrite=optimize, verbose=verbose
        )
        return torch.compile(model, backend=local_ort)

    if backend == "plug":
        assert (
            not return_storage
        ), f"return_storage=True not implemented with backend={backend!r}"
        os.environ["ONNXRT_CHANGE_REWRITER"] = "1"

        local_aot_ort, local_ort = make_aot_ort(
            dynamic=use_dynamic, rewrite=False, verbose=verbose
        )
        return torch.compile(model, backend=local_ort)

    if backend == "inductor":
        assert (
            not return_storage
        ), f"return_storage=True not implemented with backend={backend!r}"
        return torch.compile(model, backend="inductor", dynamic=use_dynamic)

    if backend == "eager":
        assert (
            not return_storage
        ), f"return_storage=True not implemented with backend={backend!r}"
        return model

    if backend == "custom":
        storage = {} if return_storage else None
        target_opset = target_opset
        aot_compiler = aot_autograd(
            fw_compiler=lambda *args, **kwargs: onnx_custom_backend(
                *args,
                target_opset=target_opset,
                verbose=verbose,
                enable_pattern=enable_pattern,
                disable_pattern=disable_pattern,
                storage=storage,
                rename_inputs=rename_inputs,
                dump_prefix=dump_prefix,
                optimize=optimize,
                dispatcher=dispatcher,
                **kwargs,
            ),
            decompositions=get_decomposition_table(),
        )
        cc = torch.compile(
            model, backend=aot_compiler, fullgraph=True, dynamic=use_dynamic
        )
        if return_storage:
            return cc, storage
        return cc

    if backend == "backort":
        storage = {} if return_storage else None
        target_opset = target_opset
        aot_compiler = aot_autograd(
            fw_compiler=lambda *args, **kwargs: onnx_custom_backend(
                *args,
                target_opset=target_opset,
                verbose=verbose,
                enable_pattern=enable_pattern,
                disable_pattern=disable_pattern,
                storage=storage,
                rename_inputs=rename_inputs,
                dump_prefix=dump_prefix,
                optimize=optimize,
                exporter="dynamo",
                dispatcher=dispatcher,
                **kwargs,
            ),
            decompositions=get_decomposition_table_dynamo(),
        )
        cc = torch.compile(
            model, backend=aot_compiler, fullgraph=True, dynamic=use_dynamic
        )
        if return_storage:
            return cc, storage
        return cc

    if backend == "debug":
        storage = {} if return_storage else None
        target_opset = target_opset
        aot_compiler = aot_autograd(
            fw_compiler=lambda *args, **kwargs: onnx_debug_backend(
                *args,
                target_opset=target_opset,
                backend="ref",
                enable_pattern=enable_pattern,
                disable_pattern=disable_pattern,
                storage=storage,
                rename_inputs=rename_inputs,
                verbose=verbose,
                dump_prefix=dump_prefix,
                optimize=optimize,
                dispatcher=dispatcher,
                **kwargs,
            ),
            decompositions=get_decomposition_table(),
        )
        cc = torch.compile(
            model, backend=aot_compiler, fullgraph=True, dynamic=use_dynamic
        )
        if return_storage:
            return cc, storage
        return cc

    if backend == "dynger":
        aot_compiler = aot_autograd(
            fw_compiler=lambda *args, **kwargs: dynger_backend(
                *args, verbose=verbose, optimize=optimize, **kwargs
            ),
            decompositions=get_decomposition_table(),
        )
        cc = torch.compile(
            model, backend=aot_compiler, fullgraph=True, dynamic=use_dynamic
        )
        if return_storage:
            return cc, None
        return cc

    raise ValueError(f"Unexpected backend={backend!r}.")


def dort_args(name: str, description: str):
    from experimental_experiment.args import get_parsed_args

    args = get_parsed_args(
        name,
        description=description,
        model=("llama", "model to measure, llama, mistral, phi, ..."),
        backend=(
            "ort",
            "'ort' or 'inductor' or 'eager', " "'plug', or 'custom', or 'backort'",
        ),
        device=("cpu", "'cpu' or 'cuda'"),
        num_hidden_layers=(1, "number of hidden layers"),
        warmup=5,
        repeat=5,
        mixed=(0, "mixed precision (based on autocast)"),
        export=("", "export the dynamo models"),
        dynamic=("0", "use dynamic shapes"),
        target_opset=(18, "opset to convert into, use with backend=custom"),
        config=("default", "default, medium, or small to test"),
        verbose=(0, "verbosity"),
        implementation=("eager", "eager or sdpa"),
        disable_pattern=("", "a list of optimization patterns to disable"),
        enable_pattern=("default", "list of optimization patterns to enable"),
        optimize=(1, "optimize the model"),
        with_mask=(1, "with or without mask, dynamo may fail with a mask"),
        expose="backend,repeat,warmup,device,num_hidden_layers,"
        "mixed,export,config,target_opset,dynamic,verbose,"
        "enable_pattern,disable_pattern,model,optimize,with_mask",
    )
    return args


def export_args(name: str, description: str):
    from experimental_experiment.args import get_parsed_args

    args = get_parsed_args(
        name,
        description=description,
        model=("llama", "model to measure, llama, mistral, phi, ..."),
        exporter=("custom", "script, dynamo, custom"),
        device=("cpu", "'cpu' or 'cuda'"),
        num_hidden_layers=(1, "number of hidden layers"),
        mixed=(0, "mixed precision (based on autocast)"),
        dynamic=("0", "use dynamic shapes"),
        target_opset=(18, "opset to convert into, use with backend=custom"),
        config=("default", "default, medium, or small to test"),
        verbose=(0, "verbosity"),
        ort=(1, "produce the model optimized by onnxruntime"),
        implementation=("eager", "eager or sdpa"),
        disable_pattern=("", "a list of optimization patterns to disable"),
        enable_pattern=("default", "list of optimization patterns to enable"),
        optimize=(1, "optimize the model"),
        with_mask=(1, "with or without mask, dynamo may fail with a mask"),
        expose="exporter,device,num_hidden_layers,ort,"
        "mixed,config,target_opset,dynamic,verbose,"
        "enable_pattern,disable_pattern,model,optimize,with_mask",
    )
    return args


def create_configuration_for_benchmark(
    model: str = "llama",
    config: str = "small",
    repeat: int = 5,
    warmup: int = 3,
    num_hidden_layers: int = 1,
    implementation: str = "eager",
    with_mask: bool = True,
) -> Dict[str, Union[str, int, List[Tuple[int, int]]]]:
    """
    Creates a model based on the given configuration.

    :param model: model name
    :param config: size of the model (small, medium, large)
    :param warmup: number of warmup steps
    :param repeat: number of repetition
    :param num_hidden_layers: number of hidden layers
    :param implementation: implementation
    :param with_mask: use a mask
    :return: dictionary
    """
    fcts = {
        "llama": _create_configuration_for_benchmark_llama,
        "mistral": _create_configuration_for_benchmark_mistral,
        "phi": _create_configuration_for_benchmark_phi,
    }
    assert model in fcts, f"Not implemented for model {model!r}, config={config}"
    return fcts[model](
        config=config,
        repeat=repeat,
        warmup=warmup,
        num_hidden_layers=num_hidden_layers,
        implementation=implementation,
        with_mask=with_mask,
    )


def create_model(
    model: str, config_dict: Dict[str, Union[int, str]]
) -> Tuple[Any, List[Tuple[Any, ...]]]:
    """
    Returns a model and a list of inputs.

    :param model: model name
    :param config_dict: configuration
    :return: model, list of inputs
    """

    if model == "llama":
        from ..torch_models.llama_helper import get_llama_model

        return get_llama_model(**config_dict)

    if model == "mistral":
        from ..torch_models.mistral_helper import get_mistral_model

        return get_mistral_model(**config_dict)

    if model == "phi":
        from ..torch_models.phi_helper import get_phi_model

        return get_phi_model(**config_dict)

    raise AssertionError(f"not implemented for model={model!r}")
