from typing import Any, Dict, List, Sequence
import numpy as np
from onnx.helper import tensor_dtype_to_np_dtype
from ._aten_helper import (
    torch_dtype_to_onnx_dtype,
    set_shape_type_binary_op,
    set_shape_type_unary_op,
)
from .graph_builder import GraphBuilder

T = str


def aten_meth_bool(
    g: GraphBuilder, set_shape_type: bool, outputs: List[str], x: T
) -> T:
    import torch

    return aten_meth_to(g, set_shape_type, outputs, x, dtype=torch.bool)


def aten_meth_contiguous(
    g: GraphBuilder, set_shape_type: bool, outputs: List[str], x: T
) -> T:
    return g.make_node("Identity", [x], outputs, name="contiguous")


def aten_meth_expand(
    g: GraphBuilder, set_shape_type: bool, outputs: List[str], x: T, *dims: List[int]
) -> T:
    size = np.abs(np.array(dims, dtype=np.int64))
    res = g.op.Expand(x, size, outputs=outputs)
    if set_shape_type:
        g.set_type(res, g.get_type(x))
        g.set_shape(res, tuple(dims))
    return res


def aten_meth_masked_fill(
    g: GraphBuilder, set_shape_type: bool, outputs: List[str], x: T, mask: T, value: Any
) -> T:
    return aten_meth_masked_fill_(g, set_shape_type, outputs, x, mask, value)


def aten_meth_masked_fill_(
    g: GraphBuilder, set_shape_type: bool, outputs: List[str], x: T, mask: T, value: Any
) -> T:
    value_cast = g.op.CastLike(value, x, name="masked_fill")
    res = g.op.Where(mask, value_cast, x, name="masked_fill")
    if set_shape_type:
        g.set_type(res, g.get_type(x))
        g.set_type(value_cast, g.get_type(x))
        if isinstance(value, str):
            if g.has_shape(value):
                g.set_shape(value_cast, g.get_shape(value))
            elif g.has_rank(value):
                g.set_rank(value_cast, g.get_rank(value))
        elif isinstance(value, (int, float, bool)):
            g.set_shape(value_cast, tuple())
        elif hasattr(value, "shape"):
            g.set_shape(value_cast, value.shape)
        else:
            raise RuntimeError(f"Unable to guess shape from type {type(value)}")
        set_shape_type_binary_op(g, res, mask, value_cast, x, begin=1)

    return res


def aten_meth_mean(
    g: GraphBuilder,
    set_shape_type: bool,
    outputs: List[str],
    x: T,
    dim: T,
    keepdim: bool = False,
) -> T:
    if isinstance(dim, int):
        cst = g.make_initializer("", np.array([dim], dtype=np.int64))
    elif isinstance(dim, tuple):
        cst = g.make_initializer("", np.array(dim, dtype=np.int64))
    else:
        raise RuntimeError(f"Unexpected type {type(dim)} for dim.")
    res = g.make_node(
        "ReduceMean", [x, cst], outputs, keepdims=1 if keepdim else 0, name="mean"
    )
    # if set_shape_type:
    #    set_shape_type_unary_op(g, outputs[0], x)
    return res


def aten_meth_pow(
    g: GraphBuilder,
    set_shape_type: bool,
    outputs: List[str],
    x: T,
    exponent: T,
) -> T:
    assert isinstance(
        x, str
    ), f"Unexpected type {type(x)} (x={x!r}, exponent={exponent!r})"
    if isinstance(exponent, (int, float)):
        cst = g.make_initializer(
            "", np.array(exponent, dtype=tensor_dtype_to_np_dtype(g.get_type(x)))
        )
    elif isinstance(exponent, np.array):
        cst = g.make_initializer(
            "", exponent.as_type(tensor_dtype_to_np_dtype(g.get_type(x)))
        )
    elif isinstance(exponent, str):
        cst = exponent
    else:
        raise RuntimeError(f"Unexpected type {type(exponent)} for exponent.")
    res = g.make_node("Pow", [x, cst], outputs)
    if set_shape_type:
        set_shape_type_unary_op(g, outputs[0], x)
    return res


def aten_meth_reshape(
    g: GraphBuilder,
    set_shape_type: bool,
    outputs: List[str],
    input_name: T,
    *shape: List[int],
) -> T:
    cst = g.make_initializer("", np.array(shape, dtype=np.int64))
    res = g.make_node("Reshape", [input_name, cst], outputs)
    if set_shape_type:
        g.set_type(outputs[0], g.get_type(input_name))
        g.set_shape(outputs[0], tuple(shape))
    return res


def aten_meth_to(
    g: GraphBuilder,
    set_shape_type: bool,
    outputs: List[str],
    input_name: T,
    *args: List[Any],
    **kwargs: Dict[str, Any],
) -> T:
    import torch

    dtype = kwargs.get("dtype", None)
    device = kwargs.get("device", None)
    for a in args:
        if isinstance(a, torch.dtype):
            assert (
                dtype is None
            ), f"dtype is specified in args and kwargs {args}, {kwargs}"
            dtype = a
            continue
        if isinstance(a, torch.device):
            assert (
                device is None
            ), f"device is specified in args and kwargs {args}, {kwargs}"
            device = a
            continue
        raise NotImplementedError(f"Unexpected type for argument {type(a)}")
    assert (
        dtype is not None or device is not None
    ), "dtype or device cannot be None for method to"

    if dtype is None:
        return g.op.Identity(input_name, outputs=outputs, name="to")
    onnx_to = torch_dtype_to_onnx_dtype(dtype)

    res = g.make_node("Cast", [input_name], outputs, to=onnx_to, name="to")
    if set_shape_type:
        g.set_type(outputs[0], onnx_to)
        if g.has_shape(input_name):
            g.set_shape(outputs[0], g.get_shape(input_name))
        elif g.has_rank(input_name):
            g.set_rank(outputs[0], g.get_rank(input_name))
    return res


def aten_meth_transpose(
    g: GraphBuilder,
    set_shape_type: bool,
    outputs: List[str],
    input_name: T,
    dim0: int,
    dim1: int,
) -> T:
    perm = list(range(g.rank(input_name)))
    perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
    res = g.make_node("Transpose", [input_name], outputs, perm=perm)
    if set_shape_type:
        g.set_type(outputs[0], g.get_type(input_name))
        if g.has_shape(input_name):
            shape = list(g.get_shape(input_name))
            shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
            g.set_shape(outputs[0], tuple(shape))
        elif g.has_rank(input_name):
            g.set_rank(outputs[0], g.has_rank(input_name))
    return res


def aten_meth_unsqueeze(
    g: GraphBuilder,
    set_shape_type: bool,
    outputs: List[str],
    input_name: T,
    dim: int,
) -> T:
    new_name = g.unique_name(f"{input_name}_axes")
    g.make_initializer(new_name, np.array([dim], dtype=np.int64))
    res = g.make_node("Unsqueeze", [input_name, new_name], outputs)
    if set_shape_type:
        dtype = g.get_type(input_name)
        g.set_type(outputs[0], dtype)
        if g.has_shape(input_name):
            shape = list(g.get_shape(input_name))
            shape.insert(dim, 1)
            g.set_shape(outputs[0], tuple(shape))
        elif g.has_rank(input_name):
            g.set_rank(outputs[0], g.get_rank(input_name) + 1)
    return res


def aten_meth_view(
    g: GraphBuilder,
    set_shape_type: bool,
    outputs: List[str],
    input_name: T,
    *args: Sequence[int],
) -> T:
    new_shape_name = g.unique_name(f"{input_name}_view_shape")
    g.make_initializer(new_shape_name, np.array(args, dtype=np.int64))
    res = g.make_node("Reshape", [input_name, new_shape_name], outputs, name="view")
    if set_shape_type:
        dtype = g.get_type(input_name)
        g.set_shape(outputs[0], args[1:])
        g.set_type(outputs[0], dtype)
    return res
