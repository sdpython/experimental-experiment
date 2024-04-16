import warnings
from typing import Callable, List, Optional, Tuple, Union


def make_aot_ort(
    dynamic: bool = False,
    rewrite: bool = "try",
    rewrite_more: bool = False,
    aten_conversion_changes: Optional[List[Tuple[Callable, str]]] = None,
    verbose: int = 0,
    enable_pattern: Optional[Union[str, List[Union[str, type]]]] = "default",
    disable_pattern: Optional[Union[str, List[Union[str, type]]]] = None,
    processor: str = "CPU",
) -> tuple:
    """
    Creates a backend to train model with DORT.

    :param dynamic: enable dynamic shapes
    :param rewrite: rewrite the model after its conversion to onnx
    :param rewrite_more: runs more optimization
    :param aten_conversion_changes: calls aten ops
    :param verbose: verbosity
    :param enable_pattern: optimization patterns to enable
    :param disable_pattern: optimization patterns to disable
    :param processor: optimization for this processor
    :return: twice the same backend
    """
    import onnxruntime
    from torch.onnx import (
        OnnxRegistry,
        _OrtBackend as OrtBackend,
        _OrtBackendOptions as OrtBackendOptions,
        ExportOptions,
    )

    names = []
    onnx_registry = None
    if aten_conversion_changes is not None:
        onnx_registry = OnnxRegistry()
        for fct, name in aten_conversion_changes:
            onnx_registry.register_op(
                function=fct, namespace="aten", op_name=name, overload="default"
            )
            names.append(f"torch.ops.aten.{name}.default")
            if verbose:
                print(f"[make_aot_ort] register {names[-1]!r}")

    ort_session_options = onnxruntime.SessionOptions()
    # ort_session_options.log_severity_level = 1

    if rewrite is True:
        # we switch to try if torch is not recent enough.
        import packaging.version as pv
        from torch import __version__ as torch_version

        if pv.Version(".".join(torch_version.split(".")[:2])) < pv.Version("2.3"):
            rewrite = "try"

    if rewrite == "try":
        import packaging.version as pv
        from torch import __version__ as torch_version

        if pv.Version(".".join(torch_version.split(".")[:2])) < pv.Version("2.3"):
            warnings.warn(
                f"option pre_ort_model_transforms not available in torch {torch_version}"
            )
            rewrite = False
            rewrite_more = False

    if onnx_registry is None:
        export_options = ExportOptions(dynamic_shapes=dynamic)
    else:
        if verbose:
            print(f"[make_aot_ort] enable {onnx_registry!r}")
        export_options = ExportOptions(
            dynamic_shapes=dynamic, onnx_registry=onnx_registry
        )

    if rewrite:
        from ..convert.convert_helper import optimize_model_proto

        if verbose:
            print("[make_aot_ort] enable rewriting")

        if rewrite_more:

            def opt_f(*args, **kwargs):
                from ..xbuilder import GraphBuilder, OptimizationOptions
                from ..xoptim import get_pattern_list

                first_model_proto = args[0]

                next_model = optimize_model_proto(
                    *args, verbose=verbose, onnx_shape_inference=False, **kwargs
                )

                patterns = get_pattern_list(
                    enable_pattern, disable_pattern, verbose=verbose
                )

                gr = GraphBuilder(
                    next_model,
                    infer_shapes=True,
                    optimization_options=OptimizationOptions(
                        patterns=patterns, processor=processor
                    ),
                    verbose=verbose,
                )
                model_proto = gr.to_onnx()

                del first_model_proto.graph.node[:]
                del first_model_proto.functions[:]
                del first_model_proto.graph.initializer[:]
                first_model_proto.graph.node.extend(model_proto.graph.node)
                first_model_proto.functions.extend(model_proto.functions)
                first_model_proto.graph.initializer.extend(
                    model_proto.graph.initializer
                )

                return first_model_proto

        else:
            opt_f = (  # noqa: E731
                lambda *args, v=verbose, **kwargs: optimize_model_proto(
                    *args, verbose=v, onnx_shape_inference=False, **kwargs
                )
            )

        options = OrtBackendOptions(
            export_options=export_options,
            ort_session_options=ort_session_options,
            pre_ort_model_transforms=[opt_f],
        )
    else:
        assert not rewrite_more, "rewrite_more must be False if rewrite is False"
        options = OrtBackendOptions(
            export_options=export_options,
            ort_session_options=ort_session_options,
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
