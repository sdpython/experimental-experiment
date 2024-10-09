from ._exceptions import FunctionNotFoundError
from .onnx_export import to_onnx
from .dispatcher import Dispatcher, ForceDispatcher

LOCAL_DOMAIN = "aten_local_function"
