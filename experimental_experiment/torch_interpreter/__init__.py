from ._exceptions import FunctionNotFoundError
from .export_options import ExportOptions
from .onnx_export import (
    to_onnx,
    match_input_parameters,
    FunctionOptions,
    get_default_aten_as_function,
)
from .dispatcher import Dispatcher, ForceDispatcher

LOCAL_DOMAIN = "aten_local_function"
DEFAULT_TARGET_OPSET = 18


class TorchOpOverload:
    """
    The class is unused only to bypass a documentation warning.
    The alias ``TorchOpOverload`` refers to ``torch._ops.Overload``.
    """

    pass  # noqa: PIE790


def make_undefined_dimension(i: int) -> "torch.SymInt":  # noqa: F821
    """
    Uses for a custom op when a new dimension must be introduced to bypass
    some verficiation. The following function creates a dummy output
    with a dimension based on the content.

    .. code-block:: python

        def symbolic_shape(x, y):
            return torch.empty(
                x.shape[0],
                make_undefined_dimension(min(x.shape[1], y[0])),
            )

    """
    import torch

    t = torch.ones((i * 2,))
    t[:i] = 0
    res = torch.nonzero(t).shape[0]
    return res
