from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from ..helpers import tensor_dtype_to_np_dtype, torch_dtype_to_onnx_dtype
from ..xbuilder._shape_helper import all_int
from ..xbuilder.graph_builder import GraphBuilder
from ..xbuilder.shape_type_compute import (
    broadcast_shape,
    set_type_shape_unary_op,
    set_type_shape_reduce_op,
)


T = str


def prims_add(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    y: T,
    name="prims_add",
) -> T:
    "add"
    from ._aten_functions import aten_add

    return aten_add(g, sts, outputs, x, y, name=name)


def prims_amax(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    dim: Optional[int] = None,
    keepdim: bool = False,
    output_dtype: Optional["torch.dtype"] = None,  # noqa: F821
    name: str = "prims_amax",
) -> T:
    "reducemax"
    assert (
        output_dtype is None
    ), f"not implemented when output_dtype={output_dtype!r}{g.get_debug_msg()}"
    if dim is None:
        res = g.op.ReduceMaxAnyOpset(x, keepdims=1 if keepdim else 0, outputs=outputs)
    elif isinstance(dim, int):
        res = g.op.ReduceMaxAnyOpset(
            x,
            np.array([dim], dtype=np.int64),
            keepdims=1 if keepdim else 0,
            outputs=outputs,
        )
    elif isinstance(dim, list) and all_int(dim):
        res = g.op.ReduceMaxAnyOpset(
            x,
            np.array(dim, dtype=np.int64),
            keepdims=1 if keepdim else 0,
            outputs=outputs,
        )
    else:
        raise RuntimeError(f"Unexpected type {type(dim)} for dim")
    if not sts:
        set_type_shape_reduce_op(g, outputs[0], x, keepdim=keepdim)
    return res


def prims_broadcast_in_dim(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    a: T,
    shape: List[int],
    broadcast_dimensions: List[int],
) -> T:
    """
    broadcast

    ::

        s = list(shape)
        for broadcast_dimension in broadcast_dimensions:
            s[broadcast_dimension] = -1

        v = a
        for idx, x in enumerate(s):
            if x != -1:
                v = unsqueeze(v, idx)

        return expand(v, shape)
    """
    assert max(broadcast_dimensions) < len(shape), (
        f"Index out of boundary, shape={shape}, "
        f"broadcast_dimensions={broadcast_dimensions}{g.get_debug_msg()}"
    )
    s = list(shape)
    for broadcast_dimension in broadcast_dimensions:
        s[broadcast_dimension] = -1

    uns = []
    for idx, x in enumerate(s):
        if x != -1:
            uns.append(idx)

    unsqueezed = (
        g.op.UnsqueezeAnyOpset(a, np.array(uns, dtype=np.int64), name="broadcast_in_dim")
        if len(uns) > 0
        else a
    )
    res = g.op.Expand(
        unsqueezed,
        np.array(shape, dtype=np.int64),
        name="broadcast_in_dim",
        outputs=outputs,
    )

    if not sts:
        g.set_type(res, g.get_type(a))
        g.set_shape(res, shape)

    return res


def prims_cat(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    tensors: Tuple[T, ...],
    dim: int = 0,
    name: str = "prims_cat",
) -> T:
    "concat"
    from ._aten_functions import aten_cat

    return aten_cat(g, sts, outputs, tensors, dim=dim, name=name)


def prims_clone(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    memory_format: Optional[str] = None,
) -> T:
    "identity"
    from ._aten_functions import aten_clone

    return aten_clone(g, sts, outputs, x, memory_format=memory_format, name="prims_clone")


def prims_convert_element_type(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    dtype: "torch.dtype",  # noqa: F821
    name: str = "prims_convert_element_type",
) -> T:
    "cast"
    assert (
        dtype is not None
    ), f"dtype cannot be none for prims_convert_element_type{g.get_debug_msg()}"
    onnx_to = torch_dtype_to_onnx_dtype(dtype)
    if onnx_to == g.get_type(x):
        return g.op.Identity(x, outputs=outputs, name=name)
    res = g.make_node("Cast", [x], outputs, to=onnx_to, name=name)
    return res


def prims_collapse_view(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    start: int,
    end: int,
    name: str = "prims_collapse_view",
) -> T:
    "reshape"
    assert g.has_shape(
        x
    ), f"collapse_view not implemented if x has no shape{g.get_debug_msg()}"
    shape = g.get_shape(x)
    start = (start + len(shape)) % len(shape)
    end = (end + len(shape)) % len(shape)
    new_shape = []
    s = 1
    for i in range(len(shape)):
        if start <= i <= end:
            if i == start:
                new_shape.append(-1)
            s *= shape[i]
        else:
            new_shape.append(shape[i])
    ashape = np.array(new_shape, dtype=np.int64)
    res = g.op.Reshape(x, ashape, outputs=outputs, name=name)
    if not sts:
        g.set_type(res, g.get_type(x))
        ashape[ashape == -1] = s
        g.set_shape(res, tuple(ashape))
    return res


def prims_cos(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    name: str = "prims_cos",
) -> T:
    "cos"
    from ._aten_functions import aten_cos

    return aten_cos(g, sts, outputs, x, name=name)


def prims_div(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    y: T,
    name: str = "prims_div",
) -> T:
    "div"
    from ._aten_functions import aten_div

    return aten_div(g, sts, outputs, x, y, name=name)


def prims_empty_strided(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    size: T,
    stride: T,
    dtype: Optional["torch.dtype"] = None,  # noqa: F821
    layout=None,
    device: Optional["torch.device"] = None,  # noqa: F821
    requires_grad: bool = False,
    name: str = "prims_empty_strided",
) -> T:
    "constantofshape"
    # strided is unused.
    from ._aten_functions import aten_empty_strided

    return aten_empty_strided(
        g,
        sts,
        outputs,
        size,
        stride,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
        name=name,
    )


def prims_eq(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    y: T,
    name: str = "prims_eq",
) -> T:
    "equal"
    from ._aten_functions import aten_eq

    return aten_eq(g, sts, outputs, x, y, name=name)


def prims_exp(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    name: str = "prims_exp",
) -> T:
    "exp"
    from ._aten_functions import aten_exp

    return aten_exp(g, sts, outputs, x, name=name)


def prims_ge(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    y: T,
    name: str = "prims_ge",
) -> T:
    "less"
    from ._aten_functions import aten_ge

    return aten_ge(g, sts, outputs, x, y, name=name)


def prims_gt(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    y: T,
    name: str = "prims_gt",
) -> T:
    "greater"
    from ._aten_functions import aten_gt

    return aten_gt(g, sts, outputs, x, y, name=name)


def prims_iota(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    length: int,
    start: int = 0,
    step: int = 1,
    dtype: Optional["torch.dtype"] = None,  # noqa: F821
    device: Optional["torch.device"] = None,  # noqa: F821
    requires_grad: bool = False,
) -> T:
    "arange"
    assert isinstance(
        length, int
    ), f"not implemented when length={length!r}{g.get_debug_msg()}"
    assert isinstance(start, int), f"not implemented when start={start!r}{g.get_debug_msg()}"
    assert isinstance(step, int), f"not implemented when step={step!r}{g.get_debug_msg()}"
    end = start + length * step
    from ._aten_functions import aten_arange

    return aten_arange(
        g,
        sts,
        outputs,
        start,
        end,
        step,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
        name="prims_iota",
    )


def prims_lt(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    y: T,
    name: str = "prims_lt",
) -> T:
    "less"
    from ._aten_functions import aten_lt

    return aten_lt(g, sts, outputs, x, y, name=name)


def prims_mul(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    y: T,
    name: str = "prims_mul",
) -> T:
    "mul"
    from ._aten_functions import aten_mul

    return aten_mul(g, sts, outputs, x, y, name=name)


def prims_neg(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    name="prims_neg",
) -> T:
    "neg"
    from ._aten_functions import aten_neg

    return aten_neg(g, sts, outputs, x, name=name)


def prims_pow(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    exponent: T,
    name: str = "prims_pow",
) -> T:
    "pow"
    from ._aten_functions import aten_pow_Tensor_Tensor

    return aten_pow_Tensor_Tensor(g, sts, outputs, x, exponent, name=name)


def prims_rsqrt(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    name: str = "prims_rsqrt",
) -> T:
    "rqsrt"
    res = g.op.Reciprocal(g.op.Sqrt(x, name=name), name=name, outputs=outputs)
    if not sts:
        set_type_shape_unary_op(g, res, x)
    return res


def prims_sin(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    name: str = "prims_sin",
) -> T:
    "sim"
    from ._aten_functions import aten_sin

    return aten_sin(g, sts, outputs, x, name=name)


def prims_split_dim(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    dim: int,
    outer_length: int,
    name: str = "prims_split_dim",
):
    "split"
    assert len(outputs) == 1, f"Expecting 1 outputs but got {outputs}{g.get_debug_msg()}"
    assert g.has_shape(x), f"Not implemented when shape of {x!r} is unknown{g.get_debug_msg()}"
    shape = g.get_shape(x)
    shape_dim = shape[dim]
    assert isinstance(
        shape_dim, int
    ), f"Not implemented for a dynamic dimension {shape_dim}{g.get_debug_msg()}"
    assert shape_dim % outer_length == 0, (
        f"shape_dim={shape_dim} not a multiple of "
        f"outer_length={outer_length}{g.get_debug_msg()}"
    )

    inner_length = shape_dim // outer_length
    new_shape = shape[0:dim] + (outer_length, inner_length) + shape[dim + 1 :]
    res = g.op.Reshape(x, np.array(new_shape), outputs=outputs, name=name)
    if not sts:
        g.set_type(res, g.get_type(x))
        g.get_shape(res, tuple(new_shape))
    return res


def prims_sub(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    y: T,
    name: str = "prims_sub",
) -> T:
    "sub"
    from ._aten_functions import aten_sub

    return aten_sub(g, sts, outputs, x, y, name=name)


def prims_sum(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    dim: Optional[Union[int, List[int]]] = None,
    keepdim: bool = False,
    output_dtype: Optional["torch.dtype"] = None,  # noqa: F821
) -> T:
    "reducesum"
    from ._aten_functions import aten_sum

    return aten_sum(
        g, sts, outputs, x, dim, keepdim=keepdim, dtype=output_dtype, name="prims_sum"
    )


def prims_transpose(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    input_name: T,
    perm: List[int],
    name: str = "prims_transpose",
) -> T:
    "transpose"
    res = g.make_node("Transpose", [input_name], outputs, perm=list(perm), name=name)
    if not sts:
        g.set_type(outputs[0], g.get_type(input_name))
        if g.has_shape(input_name):
            shape = list(g.get_shape(input_name))
            new_shape = shape.copy()
            for i, p in enumerate(perm):
                new_shape[i] = shape[p]
            g.set_shape(outputs[0], tuple(new_shape))
        elif g.has_rank(input_name):
            g.set_rank(outputs[0], g.has_rank(input_name))
    return res


def prims_view_of(
    g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T
) -> T:
    "identity"
    return g.op.Identity(x, outputs=outputs, name="prims_view_of")


def prims_where(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    condition: T,
    x: T,
    other: T,
    name: str = "prims_where",
) -> T:
    "where"
    assert not (isinstance(x, (int, float)) and isinstance(other, (int, float))), (
        f"The two last arguments ({x}, {other}) are constant and cannot be used "
        f"to infer types{g.get_debug_msg}"
    )
    assert not isinstance(x, float) or not isinstance(other, float), (
        f"The output type cannot be guessed if the last two arguments are both floats, "
        f"x={x}, other={other}{g.get_debug_msg()}"
    )
    dtype = tensor_dtype_to_np_dtype(g.get_type(other if isinstance(other, str) else x))
    ax = x if isinstance(x, str) else np.array([x], dtype=dtype)
    aother = other if isinstance(other, str) else np.array([other], dtype=dtype)
    res = g.op.Where(condition, ax, aother, outputs=outputs, name=name)
    if not sts:
        g.set_type(res, g.get_type(other))
        if g.has_shape(condition) and g.has_shape(other):
            shape = broadcast_shape(
                g.get_shape(condition), g.get_shape(other), graph_builder=g
            )
            g.set_shape(res, shape)
        else:
            g.set_rank(max(g.get_rank(condition), g.get_rank(other)))
    return res
