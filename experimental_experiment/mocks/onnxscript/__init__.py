from .onnx_function import OnnxFunction, TracedOnnxFunction  # noqa: F401

# Added because onnxscript automatically imports these submodules.
from .onnx_opset import last_opset  # noqa: F401
from .tensor import last_tensor_opset  # noqa: F401
