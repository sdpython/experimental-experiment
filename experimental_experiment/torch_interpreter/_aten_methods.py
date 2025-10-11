from typing import Any, Dict, List, Optional, Sequence
import numpy as np
from onnx import TensorProto
from ..helpers import tensor_dtype_to_np_dtype
from ..xshape._shape_helper import all_int
from ..xbuilder.graph_builder import GraphBuilder
from ..xbuilder.shape_type_compute import (
    torch_dtype_to_onnx_dtype,
    set_type_shape_binary_op,
    set_type_shape_unary_op,
    set_type_shape_reduce_op,
    set_type_shape_reshape,
)
from ._aten_functions import (
    aten_clamp_max,
    aten_clamp_min,
    aten_cos,
    aten_expand,
    aten_eq,
    aten_repeat,
    aten_sin,
    aten_t,
)

T = str


def aten_meth_bool(g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T) -> T:
    "cast"
    import torch

    return aten_meth_to(g, sts, outputs, x, dtype=torch.bool)


def aten_meth_clamp_max(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    max_: T,
    name: str = "meth_clamp_max",
) -> T:
    """meth_clamp_max"""
    return aten_clamp_max(g, sts, outputs, x, max_, name=name)


def aten_meth_clamp_min(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    min_: T,
    name: str = "meth_clamp_min",
) -> T:
    """meth_clamp_min"""
    return aten_clamp_min(g, sts, outputs, x, min_, name=name)


def aten_meth_clone(
    g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T
) -> T:
    "identity"
    assert (
        x != outputs[0]
    ), f"Input and output are the same x={x!r}, outputs={outputs!r}{g.get_debug_msg()}"
    return g.make_node("Identity", [x], outputs, name=".clone")


def aten_meth_contiguous(
    g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T
) -> T:
    "identity"
    return g.make_node("Identity", [x], outputs, name=".contiguous")


def aten_meth_cos(g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T) -> T:
    "cos"
    return aten_cos(g, sts, outputs, x, name=".cos")


def aten_meth_cpu(g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T) -> T:
    "identity"
    return g.make_node("Identity", [x], outputs, name="cpu")


def aten_meth_detach(
    g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T
) -> T:
    "identity"
    return g.make_node("Identity", [x], outputs, name="detach")


def aten_meth_eq(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    y: T,
    name="meth_eq",
) -> T:
    "equal"
    return aten_eq(g, sts, outputs, x, y, name=name)


def aten_meth___eq__(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    y: T,
    name="meth__eq___",
) -> T:
    "equal"
    return aten_eq(g, sts, outputs, x, y, name=name)


def aten_meth_expand(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    *dims: List[int],
) -> T:
    "expand"
    return aten_expand(g, sts, outputs, x, dims, name=".expand")


def aten_meth_float(
    g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T
) -> T:
    "cast"
    import torch

    return aten_meth_to(g, sts, outputs, x, dtype=torch.float32)


def aten_meth_item(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    name: str = "aten_meth_item",
) -> T:
    "float(x)"
    if not g.has_shape(x):
        # Shape is unknown but using this operator means it is a number.
        # Let's unsqueeze
        res = g.op.Squeeze(x, outputs=outputs, name=name)
    else:
        assert g.get_shape(x) in (tuple(), (1,)), (
            f"Missing shape or unexpected shape for {x!r}: has_shape={g.has_shape(x)}, "
            f"has_rank={g.has_rank(x)}{g.get_debug_msg()}"
        )
        if g.has_shape() == (1,):
            res = g.op.SqueezeAnyOpset(x, g.ZERO, outputs=outputs, name=name)
        else:
            res = g.op.Identity(x, outputs=outputs, name=name)
    if not sts:
        if g.has_type(x):
            g.set_type(res, g.get_type(x))
        g.set_shape(res, tuple())
    return res


def aten_meth_expand_as(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    y: T,
    name: str = "aten_meth_expand_as",
) -> T:
    "expand_as"
    shape = g.op.Shape(y, name=name)
    res = g.op.Expand(x, shape, name=name, outputs=outputs)
    if not sts:
        if g.has_shape(y):
            g.set_shape(res, g.get_shape(y))
        elif g.has_rank(y):
            g.set_rank(res, g.get_rank(y))
        if g.has_type(x):
            g.set_type(res, g.get_type(x))
    return res


def aten_meth_masked_fill(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    mask: T,
    value: Any,
    name: str = "aten_meth_masked_fill",
) -> T:
    "masked_fill"
    if isinstance(value, float):
        itype = g.get_type(x)
        value_cast = g.make_initializer(
            "",
            np.array([value], dtype=tensor_dtype_to_np_dtype(itype)),
            source="aten_meth_masked_fill_",
        )
        set_shape_cast = False
    else:
        value_cast = g.op.CastLike(value, x, name=name)
        set_shape_cast = True
    res = g.op.Where(mask, value_cast, x, name=name)
    if not sts:
        g.set_type(res, g.get_type(x))
        if set_shape_cast:
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
        set_type_shape_binary_op(g, res, mask, value_cast, x, begin=1)
    return res


def aten_meth_masked_fill_(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    mask: T,
    value: Any,
) -> T:
    "masked"
    raise RuntimeError(
        "These calls should be removed from the fx graph as it is inplace modification "
        "(aten_meth_masked_fill_)."
    )


def aten_meth_mean(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    dim: T,
    keepdim: bool = False,
) -> T:
    "reducemean"
    if isinstance(dim, int):
        cst = g.make_initializer(
            "", np.array([dim], dtype=np.int64), source="aten_meth_mean.cst.1"
        )
    elif isinstance(dim, tuple):
        cst = g.make_initializer("", np.array(dim, dtype=np.int64), source="aten_meth_mean.cst.2")
    else:
        raise RuntimeError(f"Unexpected type {type(dim)} for dim.")
    res = g.op.ReduceMeanAnyOpset(
        x, cst, outputs=outputs, keepdims=1 if keepdim else 0, name=".mean"
    )
    if not sts:
        set_type_shape_reduce_op(
            g,
            outputs[0],
            x,
            keepdim=keepdim,
            axes=(dim,) if isinstance(dim, int) else dim,
        )
    return res


def aten_meth_numel(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    name: str = "meth_numel",
) -> T:
    "meth_numel"
    res = g.op.Size(x, outputs=outputs, name=name)
    if not sts:
        g.set_type(res, TensorProto.INT64)
        g.set_shape(res, tuple())
    return res


def aten_meth_pow(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    exponent: T,
) -> T:
    "pow"
    assert isinstance(x, str), f"Unexpected type {type(x)} (x={x!r}, exponent={exponent!r})"
    if isinstance(exponent, (int, float)):
        cst = g.make_initializer(
            "",
            np.array(
                exponent,
                dtype=tensor_dtype_to_np_dtype(g.get_type(x)),
            ),
            source="aten_meth_pow.exponent.scalar",
        )
    elif isinstance(exponent, np.array):
        cst = g.make_initializer(
            "",
            exponent.as_type(tensor_dtype_to_np_dtype(g.get_type(x))),
            source="aten_meth_pow.exponent.tensor",
        )
    elif isinstance(exponent, str):
        cst = exponent
    else:
        raise RuntimeError(f"Unexpected type {type(exponent)} for exponent.")
    res = g.make_node("Pow", [x, cst], outputs, name="meth_pow")
    if not sts:
        set_type_shape_unary_op(g, outputs[0], x)
    return res


def aten_meth_repeat(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    *repeats: List[int],
) -> T:
    "repeat"
    return aten_repeat(g, sts, outputs, x, repeats, name=".repeat")


def aten_meth_reshape(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    input_name: T,
    *shape: List[int],
    name: str = "reshape",
) -> T:
    "reshape"
    if isinstance(shape, tuple) and len(shape) == 1 and isinstance(shape[0], tuple):
        shape = shape[0]
    if all_int(shape):
        # static version
        cst = g.make_initializer(
            "", np.array(shape, dtype=np.int64), source="aten_meth_reshape.shape"
        )
        res = g.make_node("Reshape", [input_name, cst], outputs, name=name)
        if not sts:
            set_type_shape_reshape(g, res, input_name, shape)
        return res
    # dynamic version
    dyn_shape = g.make_shape_from_results(shape, name=name)
    res = g.make_node("Reshape", [input_name, dyn_shape], outputs, name=name)
    if not sts:
        set_type_shape_reshape(g, res, input_name, shape)
    return res


def aten_meth_sin(g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T) -> T:
    "sin"
    return aten_sin(g, sts, outputs, x, name=".sin")


def aten_meth_size(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    dim: Optional[int] = None,
    name: str = ".size",
) -> T:
    "size"
    if dim is None:
        res = g.op.Shape(x, name=f"{name}A", outputs=outputs)
        if not sts:
            g.set_type(res, TensorProto.INT64)
            if g.has_rank(x):
                g.set_shape(res, (g.get_rank(x),))
        return res

    s = g.op.Shape(x, name=name)
    d = g.op.Gather(s, np.array([dim], dtype=np.int64), name=f"{name}B")
    res = g.op.SqueezeAnyOpset(d, g.ZERO, name=f"{name}B", outputs=outputs)
    if not sts:
        g.set_type(res, TensorProto.INT64)
        g.set_shape(res, tuple())
    return res


def aten_meth_sum(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    axis: T,
    keepdim: bool = False,
    dim: Optional[int] = None,
) -> T:
    "reducesum"
    if axis is not None and isinstance(axis, int):
        axes = np.array([axis], dtype=np.int64)
    elif dim is not None and isinstance(dim, int):
        axes = np.array([dim], dtype=np.int64)
    else:
        raise AssertionError(
            f"Unexpected value for dim={dim!r} or axis={axis!r}{g.get_debug_msg()}"
        )
    res = g.op.ReduceSumAnyOpset(
        x, axes, outputs=outputs, keepdims=1 if keepdim else 0, name=".sum"
    )
    if not sts:
        set_type_shape_reduce_op(
            g,
            outputs[0],
            x,
            keepdim=keepdim,
            axes=tuple(map(int, axes)),
        )
    return res


def aten_meth_t(g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T) -> T:
    "transpose"
    return aten_t(g, sts, outputs, x, name=".t")


def aten_meth_to(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    input_name: T,
    *args: List[Any],
    name: str = ".to",
    **kwargs: Dict[str, Any],
) -> T:
    "cast"
    import torch

    dtype = kwargs.get("dtype", None)
    device = kwargs.get("device", None)
    for a in args:
        if isinstance(a, torch.dtype):
            assert dtype is None, f"dtype is specified in args and kwargs {args}, {kwargs}"
            dtype = a
            continue
        if isinstance(a, torch.device):
            assert device is None, f"device is specified in args and kwargs {args}, {kwargs}"
            device = a
            continue
        if isinstance(a, bool):  # copy, non_blocking
            continue
        raise NotImplementedError(
            f"Unexpected type for argument {type(a)}, iunput_name={input_name!r} "
            f"args={args}{g.get_debug_msg()}"
        )
    assert dtype is not None or device is not None, "dtype or device cannot be None for method to"

    if dtype is None:
        return g.op.Identity(input_name, outputs=outputs, name=name)
    onnx_to = torch_dtype_to_onnx_dtype(dtype)

    res = g.make_node("Cast", [input_name], outputs, to=onnx_to, name=name)
    if not sts:
        g.set_type(outputs[0], onnx_to)
        if g.has_shape(input_name):
            g.set_shape(outputs[0], g.get_shape(input_name))
        elif g.has_rank(input_name):
            g.set_rank(outputs[0], g.get_rank(input_name))
    return res


def aten_meth_transpose(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    input_name: T,
    dim0: int,
    dim1: int,
) -> T:
    "transpose"
    assert g.has_rank(input_name), f"{input_name!r} must have a rank{g.get_debug_msg}"
    perm = list(range(g.rank(input_name)))
    assert max(dim0, dim1) < len(perm), (
        f"aten_meth_transpose: unexpected perm={perm}, dim0={dim0}, dim1={dim1}, "
        f"input_name={input_name!r}, rank={g.rank(input_name)}"
        f"{g.get_debug_msg()}"
    )
    perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
    res = g.make_node("Transpose", [input_name], outputs, perm=perm, name="meth_transpose")
    if not sts:
        g.set_type(outputs[0], g.get_type(input_name))
        if g.has_shape(input_name):
            shape = list(g.get_shape(input_name))
            shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
            g.set_shape(outputs[0], tuple(shape))
        elif g.has_rank(input_name):
            g.set_rank(outputs[0], g.get_rank(input_name))
    return res


def aten_meth_unsqueeze(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    input_name: T,
    dim: int,
) -> T:
    "unsqueeze"
    new_name = g.unique_name(f"{input_name}_axes")
    g.make_initializer(
        new_name, np.array([dim], dtype=np.int64), source="aten_meth_unsqueeze.axis"
    )
    res = g.make_node("Unsqueeze", [input_name, new_name], outputs, name="meth_unsqueeze")
    if not sts:
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
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    input_name: T,
    *args: Sequence[int],
) -> T:
    "view"
    if isinstance(args, tuple) and len(args) == 1 and len(outputs) == 1:
        # Maybe the traced model is call this with a wrong signature.
        args = args[0]
    if all_int(args):
        # static shape
        new_shape_name = g.unique_name(f"{input_name}_view_shape")
        g.make_initializer(
            new_shape_name, np.array(args, dtype=np.int64), source="aten_meth_view.shape"
        )
        res = g.make_node("Reshape", [input_name, new_shape_name], outputs, name=".view")
        if not sts:
            set_type_shape_reshape(g, res, input_name, args)
        return res

    new_shape_name = g.make_shape_from_results(args, name=".view")
    res = g.make_node("Reshape", [input_name, new_shape_name], outputs, name=".view")
    if not sts:
        g.set_type(new_shape_name, TensorProto.INT64)
        g.set_shape(new_shape_name, (len(args),))
        set_type_shape_reshape(g, res, input_name, new_shape_name)
        assert g.get_rank(res) == len(args), f"error in set_type_shape_reshape args={args!r}"
    return res
