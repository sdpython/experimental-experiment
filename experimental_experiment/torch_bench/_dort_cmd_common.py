import os
from typing import Any, Dict, List, Tuple, Union


def _create_configuration_for_benchmark_llama(
    config: str = "small",
    repeat: int = 5,
    warmup: int = 3,
    num_hidden_layers: int = 1,
    implementation: str = "eager",
) -> Dict[str, Union[str, int, List[Tuple[int, int]]]]:
    """
    Creates a model based on the given configuration.

    :param config: size of the model (small, medium, large)
    :param warmup: number of warmup steps
    :param repeat: number of repetition
    :param num_hidden_layers: number of hidden layers
    :param implementation: implementation
    :return: dictionary
    """
    if config == "small":
        return dict(
            input_dims=[(2, 1024)] * (repeat + warmup),
            hidden_size=16,
            num_hidden_layers=num_hidden_layers,
            vocab_size=1024,
            intermediate_size=16,
            max_position_embeddings=1024,
            num_attention_heads=2,
            _attn_implementation=implementation,
        )
    if config == "medium":
        return dict(
            input_dims=[(2, 1024)] * (repeat + warmup),
            hidden_size=1024,
            num_hidden_layers=num_hidden_layers,
            vocab_size=1024,
            intermediate_size=1024,
            max_position_embeddings=1024,
            num_attention_heads=2,
            _attn_implementation=implementation,
        )
    if config in ("large", "default"):
        return dict(
            input_dims=[(2, 1024)] * (repeat + warmup),
            hidden_size=4096,
            num_hidden_layers=num_hidden_layers,
            vocab_size=32000,
            intermediate_size=11008,
            max_position_embeddings=2048,
            num_attention_heads=32,
            _attn_implementation=implementation,
        )
    raise ValueError(f"Unexpected value for config={config!r}.")


def _create_configuration_for_benchmark_mistral(
    config: str = "small",
    repeat: int = 5,
    warmup: int = 3,
    num_hidden_layers: int = 1,
    implementation: str = "eager",
) -> Dict[str, Union[str, int, List[Tuple[int, int]]]]:
    """
    Creates a model based on the given configuration.

    :param config: size of the model (small, medium, large)
    :param warmup: number of warmup steps
    :param repeat: number of repetition
    :param num_hidden_layers: number of hidden layers
    :param implementation: implementation
    :return: dictionary
    """
    if config == "small":
        return dict(
            input_dims=[(2, 1024)] * (repeat + warmup),
            hidden_size=16,
            num_hidden_layers=num_hidden_layers,
            vocab_size=1024,
            intermediate_size=16,
            max_position_embeddings=1024,
            num_attention_heads=2,
            _attn_implementation=implementation,
        )
    if config == "medium":
        return dict(
            input_dims=[(2, 1024)] * (repeat + warmup),
            hidden_size=1024,
            num_hidden_layers=num_hidden_layers,
            vocab_size=1024,
            intermediate_size=1024,
            max_position_embeddings=1024,
            num_attention_heads=2,
            _attn_implementation=implementation,
        )
    if config in ("large", "default"):
        return dict(
            input_dims=[(2, 1024)] * (repeat + warmup),
            hidden_size=4096,
            num_hidden_layers=num_hidden_layers,
            vocab_size=32000,
            intermediate_size=11008,
            max_position_embeddings=2048,
            num_attention_heads=32,
            _attn_implementation=implementation,
        )
    raise ValueError(f"Unexpected value for config={config!r}.")


def create_configuration_for_benchmark(
    model: str = "llama",
    config: str = "small",
    repeat: int = 5,
    warmup: int = 3,
    num_hidden_layers: int = 1,
    implementation: str = "eager",
) -> Dict[str, Union[str, int, List[Tuple[int, int]]]]:
    """
    Creates a model based on the given configuration.

    :param model: model name
    :param config: size of the model (small, medium, large)
    :param warmup: number of warmup steps
    :param repeat: number of repetition
    :param num_hidden_layers: number of hidden layers
    :param implementation: implementation
    :return: dictionary
    """
    if model == "llama":
        return _create_configuration_for_benchmark_llama(
            config=config,
            repeat=repeat,
            warmup=warmup,
            num_hidden_layers=num_hidden_layers,
            implementation=implementation,
        )
    if model == "mistral":
        return _create_configuration_for_benchmark_mistral(
            config=config,
            repeat=repeat,
            warmup=warmup,
            num_hidden_layers=num_hidden_layers,
            implementation=implementation,
        )
    raise ValueError(f"Not implemented for model={model!r}.")


def create_compiled_model(
    model: Any,
    backend: str,
    target_opset: int,
    use_dynamic: bool = False,
    verbose: int = 0,
    enable_pattern: Union[str, List[str]] = "default",
    disable_pattern: Union[str, List[str]] = None,
    return_storage: bool = False,
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
    :return: compiled model
    """
    import torch
    from torch._dynamo.backends.common import aot_autograd
    from experimental_experiment.torch_helper.training_helper import make_aot_ort
    from experimental_experiment.torch_dynamo import (
        get_decomposition_table,
        onnx_custom_backend,
        onnx_debug_backend,
    )

    if backend == "ort":
        assert (
            not return_storage
        ), f"return_storage=True not implemented with backend={backend!r}"
        local_aot_ort, local_ort = make_aot_ort(
            dynamic=use_dynamic, rewrite=True, verbose=verbose
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

    raise ValueError(f"Unexpected backend={backend!r}.")


def dort_args(name: str, description: str):
    from experimental_experiment.args import get_parsed_args

    args = get_parsed_args(
        name,
        description=description,
        model=("llama", "model to measure, llama, mistral"),
        backend=("ort", "'ort' or 'inductor' or 'eager', 'plug', or 'custom'"),
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
        expose="backend,repeat,warmup,device,num_hidden_layers,"
        "mixed,export,config,target_opset,dynamic,verbose,"
        "enable_pattern,disable_pattern,model",
    )
    return args
