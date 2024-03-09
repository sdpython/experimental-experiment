import warnings


def make_aot_ort(dynamic: bool = False, rewrite: bool = "try", verbose: int = 0):
    import onnxruntime
    from torch.onnx import (
        _OrtBackend as OrtBackend,
        _OrtBackendOptions as OrtBackendOptions,
        ExportOptions,
    )

    ort_session_options = onnxruntime.SessionOptions()
    # ort_session_options.log_severity_level = 1

    if rewrite is True:
        # we switch to try if torch is not recent enough.
        import packaging.version as pv
        from torch import __version__ as torch_version

        if pv.Version(torch_version) < pv.Version("2.3"):
            rewrite = "try"

    if rewrite == "try":
        import packaging.version as pv
        from torch import __version__ as torch_version

        if pv.Version(torch_version) < pv.Version("2.3"):
            warnings.warn("option pre_ort_model_transforms not available in torch {e}")
            rewrite = False
        else:
            try:
                import onnxrewriter  # noqa: F401
            except ImportError:
                warnings.warn("unable to rewrite a model with onnx-rewriter due to {e}")
                rewrite = False

    if rewrite:
        from ..convert.convert_helper import optimize_model_proto

        options = OrtBackendOptions(
            export_options=ExportOptions(dynamic_shapes=dynamic),
            ort_session_options=ort_session_options,
            pre_ort_model_transforms=[
                lambda *args, v=verbose, **kwargs: optimize_model_proto(
                    *args, verbose=v, **kwargs
                )
            ],
        )
    else:
        options = OrtBackendOptions(
            export_options=ExportOptions(dynamic_shapes=dynamic),
            ort_session_options=ort_session_options,
        )

    ort_backend = OrtBackend(options=options)
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
    v = pred[0] if isinstance(pred, tuple) else pred
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
