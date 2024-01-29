from typing import Any, List, Sequence, Set, Tuple
import numpy as np
from onnx import TensorProto
from onnx.helper import np_dtype_to_tensor_dtype, tensor_dtype_to_np_dtype


def onnx_dtype_to_torch_dtype(itype: int) -> "torch.dtype":  # noqa: F821
    import torch

    if itype == TensorProto.FLOAT:
        return torch.float32
    if itype == TensorProto.FLOAT16:
        return torch.float16
    if itype == TensorProto.INT64:
        return torch.int64
    raise NotImplementedError(f"Unable to convert onnx type {itype} to torch.type.")


def torch_dtype_to_onnx_dtype(to: "torch.dtype") -> int:  # noqa: F821
    import torch

    if to == torch.float32:
        return TensorProto.FLOAT
    if to == torch.float64:
        return TensorProto.DOUBLE
    if to == torch.int64:
        return TensorProto.INT64
    if to == torch.bool:
        return TensorProto.BOOL
    raise NotImplementedError(f"Unable to convert torch dtype {to!r} to onnx dtype.")


def dtype_to_tensor_dtype(dt: "dtype") -> int:  # noqa: F821
    try:
        return np_dtype_to_tensor_dtype(dt)
    except KeyError:
        pass
    return torch_dtype_to_onnx_dtype(dt)


def broadcast_shape(sh1: Tuple[int, ...], sh2: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    Computes the shape for many broadcasting operators.
    """
    if len(sh1) == len(sh2):
        return tuple(max(i, j) for i, j in zip(sh1, sh2))
    shape = tuple(max(i, j) for i, j in zip(sh1, sh2))
    if len(sh1) > len(shape):
        return shape + sh1[len(shape) :]
    return shape + sh2[len(shape) :]


def set_shape_type_unary_op(
    g: "GraphBuilder", name: str, input_name: str  # noqa: F821
):
    """
    Sets the shape and type for an unary operator (abs, exp, ...).
    """
    g.set_type(name, g.get_type(input_name))
    if g.has_shape(input_name):
        g.set_shape(name, g.get_shape(input_name))
    elif g.has_rank(input_name):
        g.set_rank(name, g.get_rank(input_name))


def set_shape_type_binary_op(
    g: "GraphBuilder", name: str, *input_names: List[str], begin: int = 0  # noqa: F821
):
    """
    Sets the shape and type for a binary operator (add, mul, ...).
    """
    # type
    dtype = None
    for input_name in input_names[begin:]:
        if g.has_type(input_name):
            dtype = g.get_type(input_name)
            break
    assert dtype, f"Unable to guess type from {input_names}{g.get_debug_msg()}"
    g.set_type(name, dtype)

    # shape
    shape = None
    for input_name in input_names:
        if g.has_shape(input_name):
            if shape is None:
                shape = g.get_shape(input_name)
            else:
                shape = broadcast_shape(shape, g.get_shape(input_name))
        elif shape is not None:
            # one shape is missing
            shape = None
            break

    if shape is not None:
        g.set_shape(name, shape)
        return

    # rank otherwise
    rank = None
    for input_name in input_names:
        if g.has_rank(input_name):
            if rank is None:
                rank = g.get_rank(input_name)
            else:
                rank = max(rank, g.get_rank(input_name))
        elif rank is not None:
            # one shape is missing
            rank = None
            break
    if rank is not None:
        g.set_rank(name, rank)


def _get_input_type(g: "GraphBuilder", x: Any) -> int:  # noqa: F821
    if isinstance(x, int):
        return TensorProto.INT64
    if isinstance(x, float):
        return TensorProto.FLOAT
    if isinstance(x, str):
        return g.get_type(x)
    if isinstance(x, np.array):
        return np_dtype_to_tensor_dtype(x.dtype)
    # A torch tensor.
    if hasattr(x, "dtype"):
        return torch_dtype_to_onnx_dtype(x.dtype)
    raise RuntimeError(f"Unable to guess type from {type(x)}.")


def _get_compute_type(dtypes: Set[int]) -> int:
    order = [
        TensorProto.DOUBLE,
        TensorProto.FLOAT,
        TensorProto.FLOAT16,
        TensorProto.INT64,
        TensorProto.UINT64,
        TensorProto.INT32,
        TensorProto.UINT32,
        TensorProto.INT16,
        TensorProto.UINT16,
        TensorProto.INT8,
        TensorProto.UINT8,
    ]
    for t in order:
        if t in dtypes:
            return t
    raise RuntimeError(f"Unable to guess compute type {dtypes}.")


def _cast_inputs(g: "GraphBuilder", a: Any, itype: int) -> str:  # noqa: F821
    if isinstance(a, str):
        # a result
        return g.op.Cast(a, to=itype)
    if isinstance(a, (int, float)):
        a = np.array(a)
    if isinstance(a, np.ndarray):
        return g.make_initializer("", a.astype(tensor_dtype_to_np_dtype(itype)))
    raise RuntimeError(f"Unexpected type {type(a)}, itype={itype}.")


def prepare_inputs_homogeneous_operator(
    g: "GraphBuilder", *args: Sequence[str]  # noqa: F821
) -> Tuple[str]:
    """
    Cast any inputs to ensure all inputs share the same type.
    """
    dtypes_list = [_get_input_type(g, a) for a in args]
    dtypes = set(dtypes_list)
    only = _get_compute_type(set(dtypes))
    inputs = []
    for dt, a in zip(dtypes_list, args):
        if dt == only and isinstance(a, str):
            inputs.append(a)
            continue
        inputs.append(_cast_inputs(g, a, only))
    return tuple(inputs)


def _adjust_attributes_of_max_pool(
    expand_size: int,
    kernel_size: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
) -> Tuple[Sequence[int], Sequence[int], Sequence[int], Sequence[int]]:
    if isinstance(dilation, int):
        dilations = [dilation] * expand_size
    else:
        dilations = dilation

    if isinstance(kernel_size, int):
        kernel_shape = [kernel_size] * expand_size
    else:
        kernel_shape = kernel_size

    if isinstance(padding, int):
        pads = [padding] * expand_size * 2
    elif len(padding) == 1:
        pads = padding * expand_size * 2
    elif len(padding) == 2:
        # 2D padding
        pads = padding * 2
    elif len(padding) == 3:
        # 3D padding
        pads = padding * 2
    else:
        # When padding is already done for all dimensions,
        # we don't need to double it
        # eg: (1, 1, 1, 1, 1, 1)
        pads = padding

    if isinstance(stride, int):
        strides = [stride] * expand_size
    elif not stride:
        strides = kernel_shape
    else:
        strides = stride

    return (kernel_shape, strides, pads, dilations)
