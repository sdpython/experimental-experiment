from typing import Any, Callable, List, Optional, Sequence, Set, Tuple
import numpy as np
from onnx import TensorProto, TensorShapeProto
from onnx.helper import np_dtype_to_tensor_dtype, tensor_dtype_to_np_dtype
from .annotations import STATIC_SHAPE, is_static_shape


def _nice_shape(shape: TensorShapeProto) -> str:
    els = []
    for sh in shape.dim:
        els.append(str(sh.dim_value) if sh.HasField("dim_value") else sh.dim_param)
    return "x".join(els)


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
    if to == torch.float16:
        return TensorProto.FLOAT16
    if to == torch.float64:
        return TensorProto.DOUBLE
    if to == torch.int64:
        return TensorProto.INT64
    if to == torch.int32:
        return TensorProto.INT32
    if to == torch.bool:
        return TensorProto.BOOL
    raise NotImplementedError(f"Unable to convert torch dtype {to!r} to onnx dtype.")


def dtype_to_tensor_dtype(dt: "dtype") -> int:  # noqa: F821
    try:
        return np_dtype_to_tensor_dtype(dt)
    except KeyError:
        pass
    return torch_dtype_to_onnx_dtype(dt)


def broadcast_shape(sh1: STATIC_SHAPE, sh2: STATIC_SHAPE) -> STATIC_SHAPE:
    """
    Computes the shape for many broadcasting operators.

    :param sh1: first shape
    :param sh2: second shape
    :return: resulting shape
    """
    assert is_static_shape(sh1), f"Unexpected sh1={sh1}"
    assert is_static_shape(sh2), f"Unexpected sh2={sh2}"
    if len(sh1) == len(sh2):
        return tuple(max(i, j) for i, j in zip(sh1, sh2))
    shape = tuple(max(i, j) for i, j in zip(sh1, sh2))
    if len(sh1) > len(shape):
        return shape + sh1[len(shape) :]
    return shape + sh2[len(shape) :]


def set_type_shape_reshape(
    g: "GraphBuilder",  # noqa: F821
    name: str,
    input_name: str,
    new_shape: Sequence[int],
):
    dtype = g.get_type(input_name)
    g.set_type(name, dtype)
    if isinstance(new_shape, str):
        if g.has_shape(new_shape):
            g.set_rank(name, len(g.get_shape(new_shape)))
    elif not is_static_shape(new_shape):
        g.set_rank(name, len(new_shape))
    elif min(new_shape) == -1:
        if g.has_shape(input_name):
            shape = list(g.get_shape(input_name))
            arg_size = np.prod([a for a in new_shape if a >= 0])
            size = np.prod(shape)
            index = new_shape.index(-1)
            if arg_size == 0:
                assert size == 0, f"Unable to reshape {shape} into {new_shape}"
                shape[index] = 1
            else:
                shape[index] = int(size // arg_size)
            g.set_shape(name, tuple(shape))
        else:
            g.set_rank(name, len(new_shape))
    else:
        g.set_shape(name, tuple(new_shape))


def set_type_shape_unary_op(
    g: "GraphBuilder",  # noqa: F821
    name: str,
    input_name: str,
    itype: Optional[int] = None,
):
    """
    Sets the shape and type for an unary operator (abs, exp, ...).
    """
    g.set_type(name, itype or g.get_type(input_name))
    if g.has_shape(input_name):
        g.set_shape(name, g.get_shape(input_name))
    elif g.has_rank(input_name):
        g.set_rank(name, g.get_rank(input_name))


def set_type_shape_binary_op(
    g: "GraphBuilder",  # noqa: F821
    name: str,
    *input_names: List[str],
    begin: int = 0,
    cmp_op: bool = False,
):
    """
    Sets the shape and type for a binary operator (add, mul, ...).
    """
    # type
    dtype = None
    if cmp_op:
        # operator comparing values
        g.set_type(name, TensorProto.BOOL)
    else:
        for input_name in input_names[begin:]:
            if g.has_type(input_name):
                dtype = g.get_type(input_name)
                break
        assert dtype, f"Unable to guess type from {input_names}{g.get_debug_msg()}"
        g.set_type(name, dtype)
        assert (
            dtype != TensorProto.BOOL
        ), f"dtype is bool, does not work with a binary operator{g.get_debug_msg()}"

    # shape
    shape = None
    for input_name in input_names:
        if g.has_shape(input_name):
            input_shape = g.get_shape(input_name)
            if None in input_shape:
                shape = None
                break
            shape = (
                input_shape if shape is None else broadcast_shape(shape, input_shape)
            )
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


def set_type_shape_reduce_op(
    g: "GraphBuilder",  # noqa: F821
    name: str,
    x: str,
    keepdim: int,
):
    assert keepdim in {0, 1}, f"keepdim={keepdim} must be in {{0, 1}}"
    g.set_type(name, g.get_type(x))
    g.set_rank(name, g.get_rank(x) + keepdim - 1)


def _get_input_type(
    g: "GraphBuilder", x: Any, python_default: bool  # noqa: F821
) -> int:
    if isinstance(x, int):
        return TensorProto.INT64 if python_default else None
    if isinstance(x, float):
        return TensorProto.FLOAT if python_default else None
    if isinstance(x, str):
        return g.get_type(x)
    if isinstance(x, np.ndarray):
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
    if TensorProto.BOOL in dtypes:
        return TensorProto.INT32
    raise RuntimeError(f"Unable to guess compute type {dtypes}.")


def _cast_inputs(
    g: "GraphBuilder", a: Any, itype: int, name: Optional[str] = None  # noqa: F821
) -> str:
    if isinstance(a, str):
        # a result
        res = g.op.Cast(a, to=itype, name=name)
        g.set_type(res, itype)
        if g.has_shape(a):
            g.set_shape(res, g.get_shape(a))
        else:
            g.set_rank(res, g.get_rank(a))
        return res
    if isinstance(a, (int, float)):
        a = np.array(a)
    if isinstance(a, np.ndarray):
        return g.make_initializer("", a.astype(tensor_dtype_to_np_dtype(itype)))
    raise RuntimeError(f"Unexpected type {type(a)}, itype={itype}.")


def prepare_inputs_homogeneous_operator(
    g: "GraphBuilder",  # noqa: F821
    *args: Sequence[str],
    f: Optional[Callable] = None,
    outputs: Optional[List[str]] = None,
    name: Optional[str] = None,
) -> Tuple[str, ...]:
    """
    Cast any inputs to ensure all inputs share the same type.
    """
    dtypes_list = [_get_input_type(g, a, python_default=False) for a in args]
    dtypes_list_not_none = [n for n in dtypes_list if n is not None]
    if not dtypes_list_not_none:
        # the type cannot be guessed from the input as it is only python types,
        # let's include them
        dtypes_list_not_none = [
            _get_input_type(g, a, python_default=True) for a in args
        ]
    dtypes = set(dtypes_list_not_none)
    only = _get_compute_type(set(dtypes))
    inputs = []
    for dt, a in zip(dtypes_list, args):
        if dt == only and isinstance(a, str):
            inputs.append(a)
            continue
        inputs.append(_cast_inputs(g, a, only, name=name))
    if f is None:
        return tuple(inputs)
    if inputs == args:
        # No cast.
        res = f(*inputs, outputs=outputs, name=name)
    else:
        res = g.op.Cast(
            f(*inputs, name=name), to=dtypes_list[0], outputs=outputs, name=name
        )
    return tuple([res, *inputs])


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
