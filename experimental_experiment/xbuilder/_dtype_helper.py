from onnx import TensorProto
from onnx.helper import np_dtype_to_tensor_dtype


def onnx_dtype_to_torch_dtype(itype: int) -> "torch.dtype":  # noqa: F821
    import torch

    if itype == TensorProto.FLOAT:
        return torch.float32
    if itype == TensorProto.FLOAT16:
        return torch.float16
    if itype == TensorProto.BFLOAT16:
        return torch.bfloat16
    if itype == TensorProto.DOUBLE:
        return torch.float64
    if itype == TensorProto.INT32:
        return torch.int32
    if itype == TensorProto.INT64:
        return torch.int64
    if itype == TensorProto.UINT32:
        return torch.uint32
    if itype == TensorProto.UINT64:
        return torch.uint64
    if itype == TensorProto.BOOL:
        return torch.bool
    if itype == TensorProto.INT16:
        return torch.int16
    if itype == TensorProto.UINT16:
        return torch.uint16
    if itype == TensorProto.INT8:
        return torch.int16
    if itype == TensorProto.UINT8:
        return torch.uint16
    if itype == TensorProto.COMPLEX64:
        return torch.complex64
    if itype == TensorProto.COMPLEX128:
        return torch.complex128
    raise NotImplementedError(f"Unable to convert onnx type {itype} to torch.type.")


def torch_dtype_to_onnx_dtype(to: "torch.dtype") -> int:  # noqa: F821
    import torch

    if to == torch.float32:
        return TensorProto.FLOAT
    if to == torch.float16:
        return TensorProto.FLOAT16
    if to == torch.bfloat16:
        return TensorProto.BFLOAT16
    if to == torch.float64:
        return TensorProto.DOUBLE
    if to == torch.int64:
        return TensorProto.INT64
    if to == torch.int32:
        return TensorProto.INT32
    if to == torch.bool:
        return TensorProto.BOOL
    if to == torch.SymInt:
        return TensorProto.INT64
    if to == torch.SymFloat:
        return TensorProto.FLOAT
    if to == torch.complex64:
        return TensorProto.COMPLEX64
    if to == torch.complex128:
        return TensorProto.COMPLEX128
    raise NotImplementedError(f"Unable to convert torch dtype {to!r} to onnx dtype.")


def dtype_to_tensor_dtype(dt: "dtype") -> int:  # noqa: F821
    try:
        return np_dtype_to_tensor_dtype(dt)
    except (KeyError, TypeError):
        pass
    return torch_dtype_to_onnx_dtype(dt)


def string_to_elem_type(name: str) -> int:
    """
    Converts a string into an element type.
    INT64 -> TensorProto.INT64
    """
    assert hasattr(TensorProto, name), f"Unable to interpret type {name!r}"
    return getattr(TensorProto, name)
