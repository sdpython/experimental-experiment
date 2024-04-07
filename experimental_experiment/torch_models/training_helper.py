import warnings
from typing import Callable, List, Optional, Tuple


def make_aot_ort(
    dynamic: bool = False,
    rewrite: bool = "try",
    aten_conversion_changes: Optional[List[Tuple[Callable, str]]] = None,
    verbose: int = 0,
):
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

        options = OrtBackendOptions(
            export_options=export_options,
            ort_session_options=ort_session_options,
            pre_ort_model_transforms=[
                lambda *args, v=verbose, **kwargs: optimize_model_proto(
                    *args, verbose=v, onnx_shape_inference=False, **kwargs
                )
            ],
        )
    else:
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
