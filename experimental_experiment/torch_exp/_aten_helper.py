from typing import Sequence, Tuple
from onnx import TensorProto


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


def torch_dtype_to_onnx_dtype(to: "torch.dtype") -> int:  # noqa: F821
    import torch

    if to == torch.float32:
        return TensorProto.FLOAT
    if to == torch.int64:
        return TensorProto.INT64
    raise NotImplementedError(f"Unable to convert torch dtype {to} to onnx dtype.")


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
    g: "GraphBuilder", name: str, input_name1: str, input_name2: str  # noqa: F821
):
    """
    Sets the shape and type for a binary operator (add, mul, ...).
    """
    dtype = g.get_type(input_name1) if g.has_type(input_name1) else None
    if not dtype:
        dtype = g.get_type(input_name2) if g.has_type(input_name2) else None
    assert dtype, f"{g.get_type(input_name1)}"
    g.set_type(name, dtype)
    if g.has_shape(input_name1) and g.has_shape(input_name2):
        g.set_shape(
            name, broadcast_shape(g.get_shape(input_name1), g.get_shape(input_name2))
        )
    elif g.has_rank(input_name1) and g.has_rank(input_name2):
        g.set_rank(name, max(g.get_rank(input_name1), g.get_rank(input_name2)))
