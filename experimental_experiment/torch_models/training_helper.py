import os
from typing import Callable, List, Optional, Tuple, Union
from onnx.inliner import inline_local_functions


def make_aot_ort(
    dynamic: bool = False,
    rewrite: bool = True,
    rewrite_more: bool = False,
    aten_conversion_changes: Optional[List[Tuple[Callable, str]]] = None,
    verbose: int = 0,
    enable_pattern: Optional[Union[str, List[Union[str, type]]]] = "default",
    disable_pattern: Optional[Union[str, List[Union[str, type]]]] = None,
    processor: str = "CPU",
    ort_optimization_level: Optional[str] = None,
    order_algorithm: Optional[str] = None,
    dump_patterns: Optional[str] = None,
    dump_prefix: Optional[str] = None,
) -> tuple:
    """
    Creates a backend to train model with DORT.

    :param dynamic: enable dynamic shapes
    :param rewrite: rewrite the model after its conversion to onnx,
        it must be True, as it is no longer possible to disable that option
    :param rewrite_more: runs more optimization
    :param aten_conversion_changes: calls aten ops
    :param verbose: verbosity
    :param enable_pattern: optimization patterns to enable
    :param disable_pattern: optimization patterns to disable
    :param processor: optimization should be made for this processor
        or this list of processors (comma separated value)
    :param ort_optimization_level: onnxruntime optimization level
    :param order_algorithm: algorithm optimizing the order the onnx node,
        none by default
    :param dump_patterns: dump the applied patterns
    :param dump_prefix: prefix before saving the models
    :return: twice the same backend
    """
    assert not dump_prefix, f"dump_prefix={dump_prefix!r} not implemented"
    import onnxruntime
    from torch.onnx import (
        # OnnxRegistry,
        _OrtBackend as OrtBackend,
        _OrtBackendOptions as OrtBackendOptions,
        ExportOptions,
    )

    names = []
    onnx_registry = None
    assert aten_conversion_changes is None, (
        f"this option is not supported anymore but "
        f"aten_conversion_changes={aten_conversion_changes}"
    )

    ort_session_options = onnxruntime.SessionOptions()
    # ort_session_options.log_severity_level = 1
    if ort_optimization_level is not None:
        assert hasattr(onnxruntime.GraphOptimizationLevel, ort_optimization_level), (
            f"Unexpected value {ort_optimization_level!r} for GraphOptimizationLevel, "
            f"expecting one of the values in {dir(onnxruntime.GraphOptimizationLevel)}"
        )
        ort_session_options.graph_optimization_level = getattr(
            onnxruntime.GraphOptimizationLevel, ort_optimization_level
        )
        if ort_optimization_level == "ORT_DISABLE_ALL":
            ort_session_options.enable_mem_pattern = False
            ort_session_options.enable_mem_reuse = False
            ort_session_options.enable_cpu_mem_arena = False
            # ort_session_options.add_session_config_entry("set_denormal_as_zero", "1")
            ort_session_options.add_session_config_entry("disable_prepacking", "1")

    if enable_pattern and (
        "experimental" in enable_pattern or any("experimental" in s for s in enable_pattern)
    ):
        try:
            from onnx_extended.ortops.optim.cuda import get_ort_ext_libs

            register = True
        except ImportError:
            register = False

        if register:
            assert os.path.exists(
                get_ort_ext_libs()[0]
            ), f"Unable to find library {get_ort_ext_libs()[0]!r}."
            ort_session_options.register_custom_ops_library(get_ort_ext_libs()[0])

            from onnx_extended.ortops.optim.cpu import (
                get_ort_ext_libs as get_ort_ext_libs_cpu,
            )

            assert os.path.exists(
                get_ort_ext_libs()[0]
            ), f"Unable to find library {get_ort_ext_libs_cpu()[0]!r}."
            ort_session_options.register_custom_ops_library(get_ort_ext_libs_cpu()[0])

    assert rewrite, "It is no longer possible to disable rewriting."
    import packaging.version as pv
    from torch import __version__ as torch_version

    assert pv.Version(".".join(torch_version.split(".")[:2])) >= pv.Version(
        "2.3"
    ), f"This requires torch>=2.3 not {torch_version!r}"

    if onnx_registry is None:
        export_options = ExportOptions(dynamic_shapes=dynamic)  # noqa: F841
    else:
        if verbose:
            print(f"[make_aot_ort] enable {onnx_registry!r}")
        export_options = ExportOptions(  # noqa: F841
            dynamic_shapes=dynamic, onnx_registry=onnx_registry
        )  # noqa: F841

    if rewrite_more:

        def opt_f(
            *args,
            order_algorithm=order_algorithm,
            enable_pattern=enable_pattern,
            disable_pattern=disable_pattern,
            verbose=verbose,
            **kwargs,
        ):
            from ..xbuilder import GraphBuilder, OptimizationOptions
            from ..xoptim import get_pattern_list

            first_model_proto = args[0]

            next_model = inline_local_functions(first_model_proto)
            # next_model = optimize_model_proto_oxs(
            #     *args, verbose=verbose, onnx_shape_inference=False, **kwargs
            # )

            patterns = get_pattern_list(enable_pattern, disable_pattern, verbose=verbose)
            if order_algorithm is not None:
                from ..xoptim import OrderAlgorithm

                order_algorithm = getattr(OrderAlgorithm, order_algorithm.upper())

            gr = GraphBuilder(
                next_model,
                infer_shapes_options=True,
                optimization_options=OptimizationOptions(
                    patterns=patterns,
                    processor=processor,
                    order=order_algorithm,
                    dump_applied_patterns=dump_patterns,
                ),
                verbose=verbose,
            )
            model_proto = gr.to_onnx()

            del first_model_proto.graph.node[:]
            del first_model_proto.functions[:]
            del first_model_proto.graph.initializer[:]
            del first_model_proto.opset_import[:]
            first_model_proto.graph.node.extend(model_proto.graph.node)
            first_model_proto.functions.extend(model_proto.functions)
            first_model_proto.graph.initializer.extend(model_proto.graph.initializer)
            first_model_proto.opset_import.extend(model_proto.opset_import)

            return first_model_proto

    else:

        def opt_f(*args, **kwargs):
            first_model_proto = args[0]

            next_model = inline_local_functions(first_model_proto)
            # next_model = optimize_model_proto_oxs(
            #     *args, verbose=verbose, onnx_shape_inference=False, **kwargs
            # )

            del first_model_proto.graph.node[:]
            del first_model_proto.functions[:]
            del first_model_proto.graph.initializer[:]
            del first_model_proto.opset_import[:]
            first_model_proto.graph.node.extend(next_model.graph.node)
            first_model_proto.functions.extend(next_model.functions)
            first_model_proto.graph.initializer.extend(next_model.graph.initializer)
            first_model_proto.opset_import.extend(next_model.opset_import)

            return first_model_proto

    options = OrtBackendOptions(
        # export_options=export_options,  # not available anymore
        ort_session_options=ort_session_options,
        pre_ort_model_transforms=[opt_f],
    )

    ort_backend = OrtBackend(options=options)

    if names:
        for n in names:
            ort_backend._supported_ops._support_dict[n] = None

    return ort_backend, ort_backend


def train_loop(model, *args, loss_fn=None, optimizer=None):
    import torch

    if loss_fn is None:
        loss_fn = torch.nn.MSELoss()
    if optimizer is None:
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()

    # Compute prediction and loss
    pred = model(*args)
    if isinstance(pred, tuple):
        v = pred[0]
    elif hasattr(pred, "last_hidden_state"):
        v = pred.last_hidden_state
    else:
        v = pred
    loss = loss_fn(v, torch.ones_like(v))

    # Backpropagation
    loss.backward()
    optimizer.step()
    # skip that part to retrieve the gradients
    # optimizer.zero_grad()

    # returns the gradients
    res = tuple(p.grad for p in model.parameters() if p.grad is not None)
    assert len(res) > 0, f"No gradient, loss is {loss}"
    return res


def train_loop_mixed_precision(model, *args, loss_fn=None, optimizer=None):
    import torch

    if loss_fn is None:
        loss_fn = torch.nn.MSELoss()
    if optimizer is None:
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        # Set the model to training mode -
        # important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        model.train()

        # Compute prediction and loss
        pred = model(*args)
        if isinstance(pred, tuple):
            v = pred[0]
        elif hasattr(pred, "last_hidden_state"):
            v = pred.last_hidden_state
        else:
            v = pred
        loss = loss_fn(v, torch.ones_like(v))

        # Backpropagation
        loss.backward()
        optimizer.step()
        # skip that part to retrieve the gradients
        # optimizer.zero_grad()

    # returns the gradients
    res = tuple(p.grad for p in model.parameters() if p.grad is not None)
    assert len(res) > 0, f"No gradient, loss is {loss}"
    return res
