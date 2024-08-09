import os
from onnx import ModelProto


def append_custom_libraries(
    onx: ModelProto,
    options: "onnxruntime.SessionOptions",  # noqa: F821
):
    """
    Appends libraries implementing custom kernels.
    The functions checks the opsets then add the necessary
    custom libraries to the options.

    :param onx: model proto
    :param options: onnxruntime.SessionOptions
    """
    domains = set(d.domain for d in onx.opset_import)
    if "onnx_extended.ortops.optim.cuda" in domains:
        from onnx_extended.ortops.optim.cuda import get_ort_ext_libs

        assert os.path.exists(
            get_ort_ext_libs()[0]
        ), f"Unable to find library {get_ort_ext_libs()[0]!r}."
        options.register_custom_ops_library(get_ort_ext_libs()[0])
    elif "onnx_extended.ortops.optim.cpu" in domains:
        from onnx_extended.ortops.optim.cpu import get_ort_ext_libs

        assert os.path.exists(
            get_ort_ext_libs()[0]
        ), f"Unable to find library {get_ort_ext_libs()[0]!r}."
        options.register_custom_ops_library(get_ort_ext_libs()[0])
