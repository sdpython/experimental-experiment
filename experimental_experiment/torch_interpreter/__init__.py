from ._exceptions import FunctionNotFoundError
from .export_options import ExportOptions
from .onnx_export import to_onnx, match_input_parameters
from .dispatcher import Dispatcher, ForceDispatcher

LOCAL_DOMAIN = "aten_local_function"


class TorchOpOverload:
    """
    The class is unused only to bypass a documentation warning.
    The alias ``TorchOpOverload`` refers to ``torch._ops.Overload``.
    """

    pass  # noqa: PIE790
