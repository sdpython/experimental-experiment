from typing import List, Optional, Sequence
import numpy as np
from ._aten_helper import torch_dtype_to_onnx_dtype
from .graph_builder import GraphBuilder

T = str


def aten_meth_to(
    g: GraphBuilder,
    set_shape_type: bool,
    outputs: List[str],
    input_name: T,
    dtype: Optional["torch.dtype"] = None,  # noqa: F821
) -> T:
    assert dtype is not None, "dtype cannot be None for method to"
    onnx_to = torch_dtype_to_onnx_dtype(dtype)
    res = g.make_node("Cast", [input_name], outputs, to=onnx_to)
    if set_shape_type:
        shape = g.get_shape(input_name)
        g.set_shape(outputs[0], shape)
        g.set_type(outputs[0], onnx_to)
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
        dtype = g.get_type(input_name)
        shape = list(g.get_shape(input_name))
        shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
        g.set_shape(outputs[0], tuple(shape))
        g.set_type(outputs[0], dtype)
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
    res = g.make_node("Reshape", [input_name, new_shape_name], outputs)
    if set_shape_type:
        dtype = g.get_type(input_name)
        g.set_shape(outputs[0], args[1:])
        g.set_type(outputs[0], dtype)
    return res
