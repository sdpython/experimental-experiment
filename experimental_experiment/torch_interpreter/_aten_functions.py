"""
See https://pytorch.org/docs/stable/torch.compiler_ir.html
for the full list of aten functions.
"""

import math
import sys
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from onnx import TensorProto, ValueInfoProto
from onnx.helper import (
    tensor_dtype_to_np_dtype,
    make_graph,
    make_node,
    make_tensor_value_info,
)
from onnx.numpy_helper import from_array
from ..xbuilder._shape_helper import (
    all_float,
    all_int,
    all_int_or_float,
    is_static_dimension,
    is_static_shape,
    DYNAMIC_SHAPE,
)
from ..xbuilder._dtype_helper import (
    onnx_dtype_to_torch_dtype,
    torch_dtype_to_onnx_dtype,
)
from ..xbuilder.graph_builder import GraphBuilder
from ..xbuilder.shape_type_compute import (
    _adjust_attributes_of_max_pool,
    set_type_shape_unary_op,
    set_type_shape_binary_op,
    set_type_shape_reduce_op,
    set_type_shape_reshape,
    set_type_shape_matmul,
    prepare_inputs_homogeneous_operator,
)
from ._exceptions import FunctionNotFoundError


T = str


class Reduction(Enum):
    NONE = 0
    MEAN = 1
    SUM = 2


def aten__log_api_usage_once(
    g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], module_name: str
) -> T:
    "_log_api_usage_once: creates a dummy result."
    return g.make_node("Constant", [], value_ints=[1], name="_log_api_usage_once")


def aten_abs(g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T) -> T:
    "abs"
    res = g.make_node("Abs", [x], outputs, name="abs")
    if not sts:
        set_type_shape_unary_op(g, outputs[0], x)
    return res


def aten_acos(g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T) -> T:
    "acos"
    res = g.make_node("Acos", [x], outputs, name="acos")
    if not sts:
        set_type_shape_unary_op(g, outputs[0], x)
    return res


def aten_acosh(g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T) -> T:
    "acosh"
    res = g.make_node("Acosh", [x], outputs, name="acosh")
    if not sts:
        set_type_shape_unary_op(g, outputs[0], x)
    return res


def aten_add(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    y: T,
    name: str = "add",
) -> T:
    "add"
    res, x, y = prepare_inputs_homogeneous_operator(
        g, x, y, f=g.op.Add, name=name, outputs=outputs, sts=sts
    )
    if not sts:
        set_type_shape_binary_op(g, outputs[0], x, y)
    return res


def aten_add_Scalar(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    y: T,
    alpha: Optional[Any] = None,
) -> T:
    "add"
    assert alpha in (None, 1), f"alpha={alpha}, not implemented"
    x, y = prepare_inputs_homogeneous_operator(g, x, y)
    res = g.op.Add(x, y, outputs=outputs, name="add_Scalar")
    if not sts:
        set_type_shape_binary_op(g, outputs[0], x, y)
    return res


def aten_add_Tensor(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    y: T,
    alpha: Optional[Any] = None,
) -> T:
    "add"
    assert alpha in (None, 1), f"alpha={alpha}, not implemented"
    x, y = prepare_inputs_homogeneous_operator(g, x, y)
    res = g.op.Add(x, y, outputs=outputs, name="add_Tensor")
    if not sts:
        set_type_shape_binary_op(g, outputs[0], x, y)
    return res


def aten_addcmul(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    t1: T,
    t2: T,
    value: float = 1.0,
    name: str = "addcmul",
) -> T:
    "addcmul"
    itype = g.get_type(x)
    dtype = tensor_dtype_to_np_dtype(itype)
    cst = np.array([value], dtype=dtype)
    res = g.op.Add(
        x,
        g.op.Mul(g.op.Mul(t1, t2, name=name), cst, name=name),
        name=name,
        outputs=outputs,
    )
    return res


def aten_addmm(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    a: T,
    b: T,
    c: T,
    beta: float = 1.0,
    alpha: float = 1.0,
) -> T:
    "gemm"
    res = g.op.Gemm(
        b, c, a, alpha=float(alpha), beta=float(beta), outputs=outputs, name="addmm"
    )
    if not sts:
        g.set_type(res, g.get_type(b))
        g.set_rank(res, 2)
    return res


def aten_all(g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T) -> T:
    "all"
    res = g.op.Cast(
        g.op.ReduceMin(g.op.Cast(x, to=TensorProto.INT32, name="all"), name="all"),
        to=TensorProto.BOOL,
        outputs=outputs,
        name="all",
    )
    if not sts:
        g.set_type(res, TensorProto.BOOL)
        g.set_shape(res, tuple())
    return res


def aten_all_dim(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    dim: int,
    keepdim: bool = False,
    name: str = "all_dim",
) -> T:
    "all_dim"
    assert g.has_rank(x), f"{x!r} must have a rank{g.get_debug_msg()}"
    rkx = g.get_rank(x)
    if rkx == 0:
        res = g.op.Cast(x, to=TensorProto.BOOL, outputs=outputs, name=name)
        if not sts:
            g.get_type(res, TensorProto.BOOL)
            g.get_shape(res, tuple())
        return res

    res = g.op.Cast(
        g.op.ReduceMin(
            g.op.Cast(x, to=TensorProto.INT32, name=name),
            np.array([dim], dtype=np.int64),
            keepdims=1 if keepdim else 0,
            name=name,
        ),
        to=TensorProto.BOOL,
        outputs=outputs,
        name=name,
    )
    if not sts:
        g.set_type(res, TensorProto.BOOL)
        g.set_rank(res, rkx if keepdim else (rkx - 1))
    return res


def aten_alias(g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T) -> T:
    "identity"
    return g.make_node("Identity", [x], outputs, name="alias")


def aten_amax(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    dim: Optional[int] = None,
    keepdim: bool = False,
    output_dtype: Optional["torch.dtype"] = None,  # noqa: F821
    name: str = "aten_amax",
) -> T:
    "reducemax"
    from ._prims_functions import prims_amax

    return prims_amax(g, sts, outputs, x, dim, keepdim, output_dtype=output_dtype, name=name)


def aten_and(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    y: T,
    name="and",
) -> T:
    "and"
    res, x, y = prepare_inputs_homogeneous_operator(
        g, x, y, f=g.op.And, name=name, outputs=outputs, sts=sts
    )
    if not sts:
        set_type_shape_binary_op(g, outputs[0], x, y)
    return res


def aten_or(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    y: T,
    name="or",
) -> T:
    "or"
    res, x, y = prepare_inputs_homogeneous_operator(
        g, x, y, f=g.op.Or, name=name, outputs=outputs, sts=sts
    )
    if not sts:
        set_type_shape_binary_op(g, outputs[0], x, y)
    return res


def aten_and_(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    y: T,
    name="and",
) -> T:
    "and"
    return aten_and(g, sts, outputs, x, y, name="and_")


def aten_logical_and(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    y: T,
    name="and",
) -> T:
    "and"
    return aten_and(g, sts, outputs, x, y, name="logical_and")


def aten_logical_or(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    y: T,
    name="or",
) -> T:
    "or"
    return aten_or(g, sts, outputs, x, y, name="logical_or")


def aten_any(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    name: str = "any",
) -> T:
    """any"""

    if g.has_shape(x):
        shape = g.get_shape(x)
        if shape in (tuple(), (1,)):
            return g.op.Cast(x, to=TensorProto.BOOL, name=name, outputs=outputs)

    if g.main_opset >= 20:
        self_bool = g.op.Cast(x, to=TensorProto.BOOL, name=name)
        res = g.op.ReduceMax(self_bool, keepdims=0, name=name, outputs=outputs)
    else:
        self_int = g.op.Cast(x, to=TensorProto.INT32, name=name)
        res = g.op.Cast(
            g.op.ReduceMax(self_int, keepdims=0, name=name),
            to=TensorProto.BOOL,
            outputs=outputs,
        )

    if not sts:
        g.set_type(res, TensorProto.BOOL)
        g.set_shape(res, tuple())
    return res


def aten_any_dim(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    dim: int,
    keepdim: bool = False,
    name: str = "all_dim",
) -> T:
    "all_dim"
    assert g.has_rank(x), f"{x!r} must have a rank{g.get_debug_msg()}"
    rkx = g.get_rank(x)
    if rkx == 0:
        res = g.op.Cast(x, to=TensorProto.BOOL, outputs=outputs, name=name)
        if not sts:
            g.get_type(res, TensorProto.BOOL)
            g.get_shape(res, tuple())
        return res

    res = g.op.Cast(
        g.op.ReduceMax(
            g.op.Cast(x, to=TensorProto.INT32, name=name),
            np.array([dim], dtype=np.int64),
            keepdims=1 if keepdim else 0,
            name=name,
        ),
        to=TensorProto.BOOL,
        outputs=outputs,
        name=name,
    )
    if not sts:
        g.set_type(res, TensorProto.BOOL)
        g.set_rank(res, rkx if keepdim else (rkx - 1))
    return res


def aten_arange(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    start: Optional[int] = None,
    end: Optional[int] = None,
    step: int = 1,
    dtype: Optional["torch.dtype"] = None,  # noqa: F821
    layout=None,
    device: Optional["torch.device"] = None,  # noqa: F821
    pin_memory=None,
    name: str = "arange",
    requires_grad: bool = False,
) -> T:
    "arange"
    assert (
        layout is None
    ), f"arange not implemented for layout={layout!r} is not None{g.get_debug_msg()}"
    assert not pin_memory, f"arange not implemented for pin_memory=True{g.get_debug_msg()}"
    assert (
        not requires_grad
    ), f"arange not implemented when requires_grad is True{g.get_debug_msg()}"
    if start is not None and end is None:
        end = start
        start = 0

    if dtype is None:
        import torch
        from torch._prims_common import IntLike, FloatLike

        # coming from function arange in torch/_refs.__init__.py
        args = (start, end, step)
        dt = torch.int64
        for a in args:
            if isinstance(a, IntLike):
                continue
            if isinstance(a, FloatLike):
                dt = torch.float32
                continue
            if isinstance(a, str):
                it = g.get_type(a)
                if it != TensorProto.INT64:
                    dt = onnx_dtype_to_torch_dtype(it)
        if dt is None:
            dt = torch.get_default_dtype(dt)
        if dt is not None:
            dtype = torch_dtype_to_onnx_dtype(dt)

    if dtype is None:
        if isinstance(end, str):
            itype = g.get_type(end)
        elif isinstance(end, int):
            itype = TensorProto.INT64
        elif isinstance(end, float):
            itype = TensorProto.FLOAT
        else:
            itype = torch_dtype_to_onnx_dtype(type(end))
    elif isinstance(dtype, int):
        itype = dtype
    else:
        itype = torch_dtype_to_onnx_dtype(dtype)

    def _may_cast(a, it):
        assert g.has_rank(a), f"Missing rank for {a!r}{g.get_debug_msg()}"
        rk = g.get_rank(a)
        if rk == 1:
            # It must be a scalar.
            dt = g.get_type(a)
            a = g.op.SqueezeAnyOpset(a, np.array([0], dtype=np.int64), name=name)
            g.set_type(a, dt)
            g.set_shape(a, tuple())
        gi = g.get_type(a)
        if gi == it:
            return a
        return g.op.Cast(a, to=it, name=name)

    dtype = onnx_dtype_to_torch_dtype(itype)
    npdtype = tensor_dtype_to_np_dtype(itype)
    if step is None:
        step = 1
    assert start is not None, "start cannot be None"
    assert end is not None, "end cannot be None"
    assert step is not None, "step cannot be None"
    i_start = (
        _may_cast(start, itype) if isinstance(start, str) else np.array(start, dtype=npdtype)
    )
    i_end = _may_cast(end, itype) if isinstance(end, str) else np.array(end, dtype=npdtype)
    i_step = _may_cast(step, itype) if isinstance(step, str) else np.array(step, dtype=npdtype)

    res = g.op.Range(i_start, i_end, i_step, outputs=outputs, name=name)
    if not sts:
        g.set_type(res, itype)
        if isinstance(end, str) or isinstance(start, str) or isinstance(step, str):
            g.set_rank(res, 1)
        else:
            g.set_shape(res, ((end - start) // step,))
    return res


def aten_arange_start(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    start: Optional[int] = None,
    end: Optional[int] = None,
    dtype: Optional["torch.dtype"] = None,  # noqa: F821
    layout=None,
    device: Optional["torch.device"] = None,  # noqa: F821
    pin_memory=None,
) -> T:
    "arange"
    import torch

    assert layout in (
        None,
        torch.strided,
    ), f"arange not implemented for layout={layout!r}"
    assert not pin_memory, "arange not implemented for pin_memory=True"
    return aten_arange(g, sts, outputs, start, end, dtype=dtype)


def aten_arange_start_step(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    start: Optional[int] = None,
    end: Optional[int] = None,
    step: int = 1,
    dtype: Optional["torch.dtype"] = None,  # noqa: F821
    layout=None,
    device: Optional["torch.device"] = None,  # noqa: F821
    pin_memory=None,
) -> T:
    "arange"
    import torch

    assert layout in (
        None,
        torch.strided,
    ), f"arange not implemented for layout={layout!r} is not None"
    assert not pin_memory, "arange not implemented for pin_memory=True"
    return aten_arange(g, sts, outputs, start, end, step, dtype)


def aten_argmax(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    dim: Optional[int] = None,
    keepdim: bool = False,
) -> T:
    "argmax"
    if dim is None:
        xf = g.op.Reshape(x, np.array([-1], dtype=np.int64))
        res = g.op.SqueezeAnyOpset(
            g.op.ArgMax(xf, keepdims=(1 if keepdim else 0)),
            outputs=outputs,
            name="argmax",
        )
    elif isinstance(dim, int):
        res = g.op.ArgMax(
            x, axis=dim, keepdims=1 if keepdim else 0, outputs=outputs, name="argmax"
        )
    else:
        raise RuntimeError(f"Unexpected type {type(dim)} for dim")
    if not sts:
        g.set_type(res, TensorProto.INT64)
        if dim is None:
            g.set_shape(res, (1,))
        elif g.has_shape(x):
            sh = g.get_shape(x)
            g.set_shape(res, (sh[dim],))
        else:
            g.set_rank(res, 1)
    return res


def aten_as_strided(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    size: List[int],
    stride: List[int],
    storage_offset: Optional[int] = None,
) -> T:
    "as_strided"
    if storage_offset is None and min(stride) == max(stride) == 1 and g.has_shape(x):
        shape = g.get_shape(x)
        if np.prod(shape) == np.prod(size):
            return g.op.Reshape(x, np.array(size, dtype=np.int64), outputs=outputs)

    raise AssertionError(
        f"The implementation is still incorrect, x={x!r}, "
        f"shape={g.get_shape(x)}, size={size}, "
        f"stride={stride}, storage_offset={storage_offset}{g.get_debug_msg()}"
    )

    import torch
    from torch.fx.experimental.proxy_tensor import maybe_disable_fake_tensor_mode

    assert g.has_shape(x), f"not implemented when shape of x is not known{g.get_debug_msg()}"
    shape = g.get_shape(x)
    n = np.prod(shape)
    indices = np.arange(n).reshape(shape)
    with maybe_disable_fake_tensor_mode():
        tindices = torch.tensor(indices)
        try:
            strided = torch.as_strided(tindices, size, stride, storage_offset)
        except RuntimeError as e:
            raise RuntimeError(
                f"error with as_strided, x={x!r}, shape={shape!r}, n={n}, "
                f"storage_offset={storage_offset}, "
                f"size={size}, stride={stride}: {e}{g.get_debug_msg()}"
            ) from e
        np_strided = strided.detach().numpy().ravel()

    flat = g.op.Reshape(x, np.array([-1], dtype=np.int64))
    xflat = g.op.Gather(flat, np_strided.astype(np.int64))
    res = g.op.Reshape(xflat, np.array(size, dtype=np.int64), outputs=outputs)

    if not sts:
        g.set_type(res, g.get_type(x))
        g.set_shape(res, tuple(size))
    return res


def aten_asin(g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T) -> T:
    "asin"
    res = g.make_node("Asin", [x], outputs, name="asin")
    if not sts:
        set_type_shape_unary_op(g, outputs[0], x)
    return res


def aten_asinh(g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T) -> T:
    "asinh"
    res = g.make_node("Asinh", [x], outputs, name="asinh")
    if not sts:
        set_type_shape_unary_op(g, outputs[0], x)
    return res


def aten_atan(g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T) -> T:
    "atan"
    res = g.make_node("Atan", [x], outputs, name="atan")
    if not sts:
        set_type_shape_unary_op(g, outputs[0], x)
    return res


def aten_atanh(g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T) -> T:
    "atanh"
    res = g.make_node("Atanh", [x], outputs, name="atanh")
    if not sts:
        set_type_shape_unary_op(g, outputs[0], x)
    return res


def _adjust_attributes_of_avg_pool(
    expand_size: int,
    kernel_size: Union[Sequence[int], int],
    stride: Union[Sequence[int], int],
    padding: Union[Sequence[int], int],
) -> Tuple[Sequence[int], Sequence[int], Sequence[int]]:
    """Adjust attributes of avg_pool to match ONNX specification."""

    kernel_shape = [kernel_size] * expand_size if isinstance(kernel_size, int) else kernel_size

    if isinstance(padding, int):
        pads = [padding] * expand_size * 2
    elif len(padding) == 1:
        pads = padding * expand_size * 2
    elif len(padding) == 2:
        pads = padding * expand_size
    else:
        pads = padding * 2

    if isinstance(stride, int):
        strides = [stride] * expand_size
    elif not stride:
        strides = kernel_shape
    else:
        strides = stride

    return (kernel_shape, strides, pads)


def aten_avg_pool2d(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    kernel_size: Sequence[int] = (),
    stride: Sequence[int] = (),
    padding: Sequence[int] = (0, 0),
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: Optional[int] = None,
    name: str = "aten_avg_pool2d",
) -> T:
    "AveragePool"
    assert divisor_override is None, (
        f"avg_pool2d not implemented for divisor_override="
        f"{divisor_override}{g.get_debug_msg()}"
    )

    expand_size = 2

    kernel_shape, strides, pads = _adjust_attributes_of_avg_pool(
        expand_size, kernel_size, stride, padding
    )

    result = g.op.AveragePool(
        x,
        ceil_mode=1 if ceil_mode else 0,
        count_include_pad=1 if count_include_pad else 0,
        kernel_shape=kernel_shape,
        pads=pads,
        strides=strides,
        outputs=outputs,
        name=name,
    )

    if not sts:
        g.set_type(result, g.get_type(x))
        g.set_rank(result, g.get_rank(x))

    return result


def aten_avg_pool2d_backward(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    grad_output: T,
    x: T,
    kernel_size: Sequence[int] = (),
    stride: Sequence[int] = (),
    padding: Sequence[int] = (0, 0),
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: Optional[int] = None,
    **kwargs,
) -> T:
    "AveragePoolGrad (not a standard onnx operator)"
    assert divisor_override is None, (
        f"avg_pool2d_backward not implemented for divisor_override="
        f"{divisor_override}{g.get_debug_msg()}"
    )
    assert (
        not kwargs
    ), f"avg_pool2d_backward not implemented for kwargs={kwargs}{g.get_debug_msg()}"

    expand_size = 2

    kernel_shape, strides, pads = _adjust_attributes_of_avg_pool(
        expand_size, kernel_size, stride, padding
    )

    grad = g.make_node(
        "AveragePoolGrad",
        grad_output,
        auto_pad="NOTSET",
        ceil_mode=1 if ceil_mode else 0,
        count_include_pad=1 if count_include_pad else 0,
        kernel_shape=kernel_shape,
        pads=pads,
        strides=strides,
        outputs=outputs,
        # domain="com.microsoft",
        name="avg_pool2d_backward",
    )

    # It is an average, so x is not used to compute the gradient.
    # result = g.op.Add(x, grad, name="avg_pool2d_backward", outputs=outputs)
    if not sts:
        g.set_type(grad, g.get_type(x))
        g.set_rank(grad, g.get_rank(x))
    return grad


def aten_baddbmm(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    batch1: T,
    batch2: T,
    beta: Optional[T] = None,
    alpha: Optional[T] = None,
    name: str = "baddbmm",
) -> T:
    """baddbmm"""
    assert g.has_type(x), f"baddbmm: type is unknown for {x!r}{g.get_debug_msg()}"
    assert alpha is None or isinstance(
        alpha, (float, int)
    ), f"baddbmmnexpected type {type(alpha)} for alpha{g.get_debug_msg()}"
    assert beta is None or isinstance(
        beta, (float, int)
    ), f"baddbmmnexpected type {type(beta)} for alpha{g.get_debug_msg()}"
    dtype = tensor_dtype_to_np_dtype(g.get_type(x))
    batch_mul = g.op.MatMul(batch1, batch2, name=name)
    mul_a = (
        batch_mul
        if alpha is None
        else g.op.Mul(batch_mul, np.array([alpha], dtype=dtype), name=name)
    )
    mul_b = x if alpha is None else g.op.Mul(x, np.array([beta], dtype=dtype), name=name)
    res = g.op.Add(mul_a, mul_b, name=name)
    return res


def aten_bitwise_not(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    name: str = "bitwise_not",
) -> T:
    "bitwise not"
    if g.get_type(x) == TensorProto.BOOL:
        res = g.op.Not(x, outputs=outputs, name=name)
    else:
        res = g.op.BitwiseNot(x, outputs=outputs, name=name)
    if not sts:
        set_type_shape_unary_op(g, res, x)
    return res


def aten_adaptive_avg_pool1d(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    output_size: Tuple[int, ...],
    name="aten.adaptive_avg_pool1d",
):
    """adaptative AvgPool"""
    return _aten_adaptive_avg_poolnd(g, sts, outputs, x, output_size, d=1, name=name)


def aten_adaptive_avg_pool2d(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    output_size: Tuple[int, ...],
    name="aten.adaptive_avg_pool2d",
):
    """adaptative AvgPool"""
    return _aten_adaptive_avg_poolnd(g, sts, outputs, x, output_size, d=2, name=name)


def aten_adaptive_avg_pool3d(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    output_size: Tuple[int, ...],
    name="aten.adaptive_avg_pool3d",
):
    """adaptative AvgPool"""
    return _aten_adaptive_avg_poolnd(g, sts, outputs, x, output_size, d=3, name=name)


def _aten_adaptive_avg_poolnd(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    output_size: Tuple[int, ...],
    d: int,
    name="aten.adaptive_avg_poolnd",
):
    """adaptative AvgPool"""
    assert (
        len(output_size) == d
    ), f"Dimension mismatch between d={2} and output_size={output_size}{g.get_debug_msg()}"
    if output_size == [1] * len(output_size):
        res = g.op.GlobalAveragePool(x, outputs=outputs, name=name)
        if not sts:
            g.set_type(res, g.get_type(x))
            if g.has_shape(x):
                shape = g.get_shape(x)
                new_shape = (*shape[:-d], *output_size)
                g.set_shape(res, new_shape)
            elif g.has_rank(x):
                rk = g.get_rank(x)
                g.get_shape(res, rk)
        return res

    shape = g.get_shape(x) if g.has_shape(x) else None

    if (
        d == 2
        and shape is not None
        and is_static_shape(shape[-d:])
        and shape[-2] % output_size[-2] == 0
        and shape[-1] % output_size[-1] == 0
    ):
        # Optimization coming from pytorch decomposition
        stride = tuple(i // o for i, o in zip(shape[-2:], output_size))
        kernel = tuple(i - (o - 1) * s for i, o, s in zip(shape[-2:], output_size, stride))
        return aten_avg_pool2d(
            g, sts, outputs, x, kernel_size=kernel, stride=stride, name=name
        )

    raise AssertionError(
        f"_aten_adaptive_avg_poolnd (d={d}, output_size={output_size}, "
        f"input_shape={g.get_shape(x) if g.has_shape(x) else '?'}) "
        f"not yet implemented{g.get_debug_msg()}"
    )


def aten_bitwise_or(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    y: T,
    name: str = "bitwise_or",
) -> T:
    "bitwise or"
    if g.get_type(x) == TensorProto.BOOL and g.get_type(y) == TensorProto.BOOL:
        x, y = prepare_inputs_homogeneous_operator(g, x, y, name=name)
        res = g.op.Or(x, y, outputs=outputs, name=name)
        if not sts:
            set_type_shape_binary_op(g, outputs[0], x, y)
        return res

    x, y = prepare_inputs_homogeneous_operator(g, x, y, name=name)
    res = g.op.BitwiseOr(x, y, outputs=outputs, name=name)
    if not sts:
        set_type_shape_binary_op(g, outputs[0], x, y, cmp_op=True)
    return res


def aten_bitwise_or_Tensor(
    g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T, y: T
) -> T:
    "bitwise or"
    return aten_bitwise_or(g, sts, outputs, x, y, name="bitwise_or_Tensor")


def aten_bmm(
    g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T, y: T
) -> T:
    "bmm"
    assert g.get_type(x) == g.get_type(y), (
        f"type mismatch between {x!r}:{g.get_type(x)} and "
        f"{y!r}:{g.get_type(y)}{g.get_debug_msg()}"
    )
    res = g.op.MatMul(x, y, outputs=outputs, name="bmm")
    if not sts:
        set_type_shape_matmul(g, res, x, y)
    return res


def aten_cat(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    tensors: Tuple[T, ...],
    dim: int = 0,
    name="cat",
) -> T:
    "concat"
    assert len(tensors) > 0, f"No tensor to concat{g.get_debug_msg()}"
    if len(tensors) == 1:
        # Nothing to concatenate.
        return g.op.Identity(tensors[0], name=name, outputs=outputs)
    input_types = [(g.get_type(t) if isinstance(t, str) else None) for t in tensors]
    if len(set(input_types)) != 1:
        # Type conflicts: we use the output type
        itype = g.get_type_known(outputs[0])
        new_inputs = []
        for t, dt in zip(tensors, input_types):
            if dt == itype:
                new_inputs.append(t)
                continue
            new_inputs.append(g.op.Cast(t, to=itype, name=name))
        name += "c"
    else:
        new_inputs = tensors

    res = g.op.Concat(*new_inputs, axis=dim, outputs=outputs, name="cat")
    if not sts:
        dt0 = g.get_type(tensors[0])
        assert all(g.get_type(t) == dt0 for t in tensors)
        r0 = g.get_rank(tensors[0])
        assert all(g.get_rank(t) == r0 for t in tensors)
        g.set_type(outputs[0], dt0)
        g.set_rank(outputs[0], r0)
    return res


def aten_clamp(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    min: Optional[float] = None,
    max: Optional[float] = None,
    name: str = "clamp",
) -> T:
    """clip"""
    if min is None and max is None:
        return g.op.Identity(x, outputs=outputs, name=name)

    itype = g.get_type(x)
    dtype = tensor_dtype_to_np_dtype(itype)
    if max is None:
        res = g.op.Clip(x, np.array([min], dtype=dtype), outputs=outputs, name=name)
    elif min is None:
        res = g.op.Clip(x, None, np.array([max], dtype=dtype), outputs=outputs, name=name)
    else:
        res = g.op.Clip(
            x,
            np.array([min], dtype=dtype),
            np.array([max], dtype=dtype),
            outputs=outputs,
            name=name,
        )
    if not sts:
        set_type_shape_unary_op(g, res, x)
    return res


def aten_clamp_max(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    max_: T,
    name: str = "clamp_min",
) -> T:
    """clamp_min"""
    if isinstance(max_, (float, int)):
        assert g.has_type(x), f"Missing type for x={x!r}{g.get_debug_msg()}"
        dtype = tensor_dtype_to_np_dtype(g.get_type(x))
        max_value = np.array([max_], dtype=dtype)
        res = g.op.Clip(x, None, max_value, name=name, outputs=outputs)
    else:
        assert isinstance(max_, str), f"Unexpected type {type(max_)}{g.get_debug_msg()}"
        res = g.op.Max(x, max_, name=name, outputs=outputs)
    if not sts:
        set_type_shape_unary_op(g, res, x)
    return res


def aten_clamp_min(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    min_: T,
    name: str = "clamp_min",
) -> T:
    """clamp_min"""
    if isinstance(min_, (float, int)):
        assert g.has_type(x), f"Missing type for x={x!r}{g.get_debug_msg()}"
        dtype = tensor_dtype_to_np_dtype(g.get_type(x))
        min_value = np.array([min_], dtype=dtype)
        res = g.op.Clip(x, min_value, name=name, outputs=outputs)
    else:
        assert isinstance(min_, str), f"Unexpected type {type(min_)}{g.get_debug_msg()}"
        res = g.op.Max(x, min_, name=name, outputs=outputs)
    if not sts:
        set_type_shape_unary_op(g, res, x)
    return res


def aten_clamp_Tensor(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    min_t: Optional[T],
    max_t: Optional[T],
    name: str = "clamp_Tensor",
) -> T:
    """clip"""
    assert (
        min_t is not None or max_t is not None
    ), f"Not implemented yet when min_t or max_t is None{g.get_debug_msg()}"

    if min_t is None:
        if g.get_type(max_t) != g.get_type(x):
            max_t = g.op.Cast(max_t, to=g.get_type(x), name=name)
        res = g.op.Clip(x, max_t, outputs=outputs, name=name)
    elif max_t is None:
        if g.get_type(min_t) != g.get_type(x):
            min_t = g.op.Cast(min_t, to=g.get_type(x), name=name)
        res = g.op.Clip(x, None, min_t, outputs=outputs, name=name)
    else:
        if g.get_type(min_t) != g.get_type(x):
            min_t = g.op.Cast(min_t, to=g.get_type(x), name=name)
        if g.get_type(max_t) != g.get_type(x):
            max_t = g.op.Cast(max_t, to=g.get_type(x), name=name)
        res = g.op.Clip(x, min_t, max_t, outputs=outputs, name=name)
    if not sts:
        set_type_shape_unary_op(g, res, x)
    return res


def aten_clip(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    min: Optional[float] = None,
    max: Optional[float] = None,
    name: str = "clip",
) -> T:
    "clip"
    return aten_clamp(g, sts, outputs, x, min=min, max=max, name=name)


def aten_clone(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    memory_format: Optional[str] = None,
    name="clone",
) -> T:
    "identity"
    import torch

    assert (
        memory_format is None
        or memory_format == torch.contiguous_format
        or memory_format == torch.preserve_format
    ), f"Unexpected value for memory_format={memory_format!r}{g.get_debug_msg()}"
    return g.make_node("Identity", [x], outputs, name=name)


def aten_cond(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    cond: T,
    true_graph: str,
    false_graph: str,
    inputs: List[T],
    name="aten_cond",
) -> T:
    "cond"
    assert g.has_local_function(
        true_graph, g.local_domain
    ), f"Unable to find local function {true_graph!r}{g.get_debug_msg()}"
    assert g.has_local_function(
        false_graph, g.local_domain
    ), f"Unable to find local function {false_graph!r}{g.get_debug_msg()}"

    def mkv(name):
        value_info_proto = ValueInfoProto()
        value_info_proto.name = name
        return value_info_proto

    res = g.make_node(
        "If",
        [cond],
        outputs,
        name=name,
        then_branch=make_graph(
            [make_node(true_graph, inputs, outputs, domain=g.local_domain)],
            true_graph,
            [],
            [mkv(o) for o in outputs],
        ),
        else_branch=make_graph(
            [make_node(false_graph, inputs, outputs, domain=g.local_domain)],
            false_graph,
            [],
            [mkv(o) for o in outputs],
        ),
    )
    return res


def aten_convolution(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    input: T,
    weight: T,
    bias: T = None,
    stride: Sequence[int] = (1,),
    padding: Union[str, Sequence[int]] = (0, 0),
    dilation: Sequence[int] = (1,),
    transposed: bool = False,
    output_padding: Sequence[int] = (0,),
    groups: int = 1,
    auto_pad: str = "NOTSET",
    d: int = 0,
    name: str = "convolution",
) -> T:
    "conv"
    if (
        not transposed
        and output_padding
        and (min(output_padding) != 0 or max(output_padding) != 0)
    ):
        raise FunctionNotFoundError(
            f"aten_convolution does not support output_padding={output_padding}."
        )
    if isinstance(padding, str):
        assert padding == "same", (
            f"output should have the same dimensions but padding={padding}"
            f"{g.get_debug_msg()}"
        )
        assert d > 0, f"Dimension must be known d={d}{g.get_debug_msg()}"
        assert output_padding == (
            0,
        ), f"Not implemented when output_padding={output_padding!r}{g.get_debug_msg()}"
        assert g.has_shape(weight), (
            f"Not implemented when weight has no shape={output_padding!r}"
            f"{g.get_debug_msg()}"
        )
        shapew = g.get_shape(weight)
        assert (
            len(shapew) == 4
        ), f"Unexpected shape={shapew} for the weights{g.get_debug_msg()}"
        assert set(i % 2 for i in shapew[2:]) == {
            1
        }, f"Not implemented for even shape for the weight: {shapew}{g.get_debug_msg()}"
        padding = []
        for i in shapew[2:]:
            padding.extend([i // 2])

    if not isinstance(padding, Sequence):
        padding = (padding, padding)
    pads = [*padding, *padding]

    if not isinstance(dilation, Sequence):
        dilation = (dilation, dilation)
    dilations = list(dilation)

    if not isinstance(stride, Sequence):
        stride = (stride, stride)
    strides = list(stride)

    if bias is None:
        if g.main_opset >= 13:
            weight_dim_0 = g.op.Shape(weight, start=0, end=1, name=name)
        else:
            shape = g.op.Shape(weight, name=name)
            first_dim = g.op.Gather(shape, np.array([0], dtype=np.int64), name=name)
            weight_dim_0 = g.op.Reshape(first_dim, np.array([1], dtype=np.int64), name=name)
        dtype = tensor_dtype_to_np_dtype(g.get_type(input))
        if g.main_opset >= 9:
            bias = g.op.ConstantOfShape(
                weight_dim_0, value=from_array(np.array([0], dtype=dtype)), name=name
            )
        else:
            bias = g.op.Expand(np.array([0], dtype=dtype), weight_dim_0, name=name)

    # if Rank(input) != Rank(weight):
    #    input = op.UnsqueezeAnyOpset(input, op.Constant(value_ints=[0]))
    if transposed:
        res = g.make_node(
            "ConvTranspose",
            [input, weight, bias],
            outputs,
            strides=strides,
            pads=pads,
            group=groups,
            dilations=dilations,
            output_padding=output_padding,
            name=name,
        )
    else:
        res = g.make_node(
            "Conv",
            [input, weight, bias],
            outputs,
            strides=strides,
            pads=pads,
            group=groups,
            dilations=dilations,
            name=name,
        )
    if not sts:
        g.set_type(res, g.get_type(input))
        g.set_rank(res, g.get_rank(input))
    return res


def aten_conv1d(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    weight: T,
    bias: T = None,
    stride: Sequence[int] = (1,),
    padding: Union[str, Sequence[int]] = (0,),
    dilation: Sequence[int] = (1,),
    groups: int = 1,
    auto_pad: str = "NOTSET",
    name: str = "conv1d",
) -> T:
    "conv1d"
    return aten_convolution(
        g,
        sts,
        outputs,
        x,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        auto_pad=auto_pad,
        name=name,
        d=1,
    )


def aten_conv2d(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    weight: T,
    bias: T = None,
    stride: Sequence[int] = (1, 1),
    padding: Union[str, Sequence[int]] = (0, 0),
    dilation: Sequence[int] = (1, 1),
    groups: int = 1,
    auto_pad: str = "NOTSET",
    name: str = "conv2d",
) -> T:
    "conv2d"
    return aten_convolution(
        g,
        sts,
        outputs,
        x,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        auto_pad=auto_pad,
        name=name,
        d=2,
    )


def aten_conv3d(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    weight: T,
    bias: T = None,
    stride: Sequence[int] = (1, 1),
    padding: Union[str, Sequence[int]] = (0, 0),
    dilation: Sequence[int] = (1, 1),
    groups: int = 1,
    auto_pad: str = "NOTSET",
    name: str = "conv3d",
) -> T:
    "conv3d"
    return aten_convolution(
        g,
        sts,
        outputs,
        x,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        auto_pad=auto_pad,
        name=name,
        d=3,
    )


def aten_conv2d_padding(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    weight: T,
    bias: T = None,
    stride: Sequence[int] = (1, 1),
    padding: Union[str, Sequence[int]] = (0, 0),
    dilation: Sequence[int] = (1, 1),
    groups: int = 1,
    name: str = "conv2d_padding",
) -> T:
    "conv"
    return aten_convolution(
        g,
        sts,
        outputs,
        x,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        name=name,
        d=2,
    )


def _convolution(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x,
    weight,
    bias,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    benchmark,
    deterministic,
    cudnn_enabled,
    allow_tf32=None,
    name: str = "_convolution",
):
    assert g.has_shape(weight), f"weight={weight!r} has no shape{g.get_debug_msg()}"
    weight_size = g.get_shape(weight)
    kernel_shape = weight_size[2:]
    assert is_static_shape(
        tuple(kernel_shape)
    ), f"Not implemented for weight shape {weight_size}{g.get_debug_msg()}"

    args = [x, weight]

    add_bias = False
    if bias is not None and bias != "":
        if g.has_rank(bias) and g.get_rank(bias) == 1:
            # ONNX only supports 1D bias
            args.append(bias)
        else:
            add_bias = True

    kwargs = {
        "kernel_shape": list(weight_size[2:]),
        "strides": list(stride),
        # ONNX supports asymmetric padding, whereas PyTorch supports only
        # symmetric padding
        "pads": list(padding + padding),
        "dilations": list(dilation),
        "group": groups,
    }

    if any(o != 0 for o in output_padding):
        # ONNX supports both output_shape and output_padding. they are equivalent expressive.
        # output_padding is more straightforward, so we use it here.
        # output_shape = stride * (input_shape - 1) +
        #                output_padding + kernel_shape - padding * 2
        assert transposed, f"transposed not specified{g.get_debug_msg()}"
        assert len(stride) == len(
            output_padding
        ), f"Length mismath {len(stride)} != {len(output_padding)}{g.get_debug_msg()}"
        kwargs["output_padding"] = list(output_padding)

    n = (
        g.op.ConvTranspose(*args, name=name, **kwargs)
        if transposed
        else g.op.Conv(*args, name=name, **kwargs)
    )
    g.set_type(n, g.get_type(x))

    if add_bias:
        return g.op.Add(n, bias, outputs=outputs, name=name)
    return g.op.Identity(n, outputs=outputs, name=name)


def aten_conv_transpose2d_input(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    weight: T,
    bias: T,
    stride: List[int],
    padding: List[int],
    output_padding: List[int],
    groups: List[int],
    dilation: List[int],
    name: str = "conv_transpose2d_input",
) -> T:
    "conv_transpose2d"
    return _convolution(
        g,
        sts,
        outputs,
        x,
        weight,
        bias,
        stride,
        padding,
        dilation,
        True,
        output_padding,
        groups,
        None,
        None,
        None,
        None,
        name=name,
    )


def aten_conv_transpose3d_input(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    weight: T,
    bias: T,
    stride: List[int],
    padding: List[int],
    output_padding: List[int],
    groups: List[int],
    dilation: List[int],
    name: str = "conv_transpose3d_input",
) -> T:
    "conv_transpose3d"
    return _convolution(
        g,
        sts,
        outputs,
        x,
        weight,
        bias,
        stride,
        padding,
        dilation,
        True,
        output_padding,
        groups,
        None,
        None,
        None,
        None,
        name=name,
    )


def aten_copy(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    src: T,
    non_blocking: bool = False,
    name: str = "copy",
) -> T:
    "identity"
    assert not non_blocking, "copy implemented when non_blocking is True"
    if g.get_type(x) == g.get_type(src):
        return g.op.Identity(src, name=name)
    return g.op.CastLike(src, x, name=name)


def aten_copy_(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    src: T,
    non_blocking: bool = False,
) -> T:
    "identity"
    return aten_copy(g, sts, outputs, x, src, non_blocking, name="copy_")


def aten_cos(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    name: str = "cos",
) -> T:
    "cos"
    res = g.make_node("Cos", [x], outputs, name=name)
    if not sts:
        set_type_shape_unary_op(g, outputs[0], x)
    return res


def aten_cosh(g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T) -> T:
    "cosh"
    res = g.make_node("Cosh", [x], outputs, name="cosh")
    if not sts:
        set_type_shape_unary_op(g, outputs[0], x)
    return res


def aten_cross_entropy_loss(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    target: T,
    weight: Optional[T] = None,
    reduction: int = Reduction.MEAN.value,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> T:
    """cross_entropy_loss"""
    assert (
        label_smoothing == 0.0
    ), f"Unexpected value for label_smoothing={label_smoothing}{g.get_debug_msg()}"

    if reduction == Reduction.NONE.value:
        reduction_name = "none"
    elif reduction == Reduction.SUM.value:
        reduction_name = "sum"
    elif reduction == Reduction.MEAN.value:
        reduction_name = "mean"
    else:
        raise AssertionError(f"Unexpected value for reduction={reduction}{g.get_debug_msg()}")

    res = g.op.SoftmaxCrossEntropyLoss(
        x,
        target,
        weight,
        reduction=reduction_name,
        ignore_index=ignore_index,
        outputs=outputs,
    )

    if not sts:
        g.set_type(outputs[0], g.get_type(x))
        if len(outputs) > 1:
            g.set_type(outputs[1], g.get_type(x))

    return res


def aten_cumsum(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    dim: T,
    dtype: Optional["torch.dtype"] = None,  # noqa: F821
    name: str = "cumsum",
) -> T:
    """cumsum"""
    assert isinstance(dim, int), f"Not implemented for dim={dim!r}{g.get_debug_msg()}"

    if dtype is None:
        itype = g.get_type(x)
        if itype == TensorProto.INT32:
            # computation is done with INT64
            itype = TensorProto.INT64
            xi = g.op.Cast(x, to=itype, name=name)
        else:
            xi = x

    else:
        itype = dtype if isinstance(dtype, int) else torch_dtype_to_onnx_dtype(dtype)
        xi = g.op.Cast(x, to=itype, name=name)

    res = g.op.CumSum(xi, np.array([dim], dtype=np.int64), outputs=outputs, name=name)
    if not sts:
        set_type_shape_unary_op(g, res, x, itype=itype)
    return res


def aten_detach(g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T) -> T:
    "identity"
    return g.make_node("Identity", [x], outputs, name="detach")


def aten_div(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    y: T,
    name="div",
) -> T:
    "div"
    res, x, y = prepare_inputs_homogeneous_operator(
        g, x, y, f=g.op.Div, name=name, outputs=outputs, sts=sts
    )
    if not sts:
        set_type_shape_binary_op(g, outputs[0], x, y)
    return res


def aten_div_Scalar(
    g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T, y: T
) -> T:
    "div"
    return aten_div(g, sts, outputs, x, y, name="div_Scalar")


def aten_div_Tensor(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    y: T,
    alpha: Optional[Any] = None,
) -> T:
    "div"
    assert alpha in (None, 1), f"alpha={alpha}, not implemented"
    x, y = prepare_inputs_homogeneous_operator(g, x, y)
    res = g.op.Div(x, y, outputs=outputs, name="div_Tensor")
    if not sts:
        set_type_shape_binary_op(g, outputs[0], x, y)
    return res


def aten_div_Tensor_mode(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    y: T,
    rounding_mode: Optional[str] = None,
    name: str = "div_Tensor_mode",
) -> T:
    "div_Tensor_mode"
    if rounding_mode is None:
        return aten_div(g, sts, outputs, x, y, name=name)

    assert rounding_mode in {"trunc", "floor"}, (
        f"aten_div_Tensor_mode: nexpected value for round_mode={rounding_mode!r}"
        f"{g.get_debug_msg()}"
    )
    assert rounding_mode == "floor", (
        f"aten_div_Tensor_mode not yet implemented for "
        f"round_mode={rounding_mode!r}{g.get_debug_msg()}"
    )
    x, y = prepare_inputs_homogeneous_operator(g, x, y)
    return g.op.Floor(g.op.Div(x, y, name=name), name=name, outputs=outputs)


def aten_dropout(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    p: T = 0.5,  # float
    training: T = True,  # bool
) -> T:
    "dropout"
    if len(outputs) == 1:
        outputs = outputs.copy()
        outputs.append("")
    if isinstance(p, float):
        p = np.array(p, dtype=tensor_dtype_to_np_dtype(g.get_type(x)))
    if isinstance(training, bool):
        training = np.array(training, dtype=np.bool_)
    result, _ = g.op.Dropout(x, p, training, outputs=outputs)
    if not sts:
        set_type_shape_unary_op(g, outputs[0], x)
    return result


def aten_einsum(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    equation: str,
    tensors: Sequence[T],
    path: Optional[int] = None,
    name: str = "einsum",
) -> T:
    """einsum"""
    assert path is None, f"not implemented when path={path}{g.get_debug_msg()}"
    return g.op.Einsum(*tensors, equation=equation, outputs=outputs, name=name)


def aten_elu(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    alpha: float = 1.0,
    scale: float = 1.0,
    input_scale: int = 1,
    inplace: bool = False,
    name="elu",
) -> T:
    "elu"
    assert (
        input_scale == 1
    ), f"not implemented when input_scale={input_scale}{g.get_debug_msg()}"
    assert not inplace, f"inplace computation is not allowed with onnx{g.get_debug_msg()}"
    if scale == 1:
        res = g.op.Elu(x, alpha=float(alpha), name=name, outputs=outputs)
    else:
        nptype = tensor_dtype_to_np_dtype(g.get_type(x))
        elu = g.op.Elu(x, alpha=float(alpha), name=name)
        res = g.op.Mul(np.array([scale], dtype=nptype), elu, name=name, outputs=outputs)
        set_type_shape_unary_op(g, elu, x)
    if not sts:
        set_type_shape_unary_op(g, res, x)
    return res


def aten_embedding(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    weight: T,
    indices: T,
    padding_idx: Optional[int] = None,
    max_norm: Optional[int] = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
    name: str = "embedding",
) -> T:
    """
    embedding

    ``padding_idx`` is only used for training,
    see :func:`torch.nn.functional.embedding`.
    It is not taken into account.
    """
    if scale_grad_by_freq or sparse or max_norm is not None:
        exc = True
        if g.has_shape(weight):
            shape = g.get_shape(weight)
            if len(shape) > 0 and isinstance(shape[0], int) and shape[0] == padding_idx + 1:
                # padding_idx should probably be -1, shape are probably dynamic
                padding_idx = -1
                exc = False
        if exc:
            raise NotImplementedError(
                f"Not implemented when padding_idx={padding_idx}, or "
                f"scale_grad_by_freq={scale_grad_by_freq} or sparse={sparse} "
                f"or max_norm={max_norm} or norm_type={norm_type} "
                f"are different from the default values, "
                f"weight: {g.get_shape(weight) if g.has_shape(weight) else '?'}, "
                f"indices: {g.get_shape(indices) if g.has_shape(indices) else '?'}"
                f"{g.get_debug_msg()}"
            )
    if g.get_type(weight) == 7:
        # Sometimes it is switched
        indices, weight = weight, indices
        assert g.get_type(indices) == 7, (
            f"indices ({indices!r}) must be integer not {g.get_type(indices)}, "
            f"weight ({weight!r}) is {g.get_type(weight)} (switched)"
            f"{g.get_debug_msg()}"
        )
    else:
        assert g.get_type(indices) == 7, (
            f"indices ({indices!r}) must be integer not {g.get_type(indices)}, "
            f"weight ({weight!r}) is {g.get_type(weight)}"
            f"{g.get_debug_msg()}"
        )

    res = g.op.Gather(weight, indices, outputs=outputs, name=name)
    if not sts:
        g.set_type(res, g.get_type(weight))
        g.set_rank(res, g.get_rank(weight) + g.get_rank(indices) - 1)
    return res


def aten__embedding_bag(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    weight: T,
    indices: T,  # INT64
    offsets: T,  # INT64
    scale_grad_by_freq: bool = False,
    mode: int = 0,  # [0,1,2] indicate ["sum", "mean", "max"]
    sparse: bool = False,
    per_sample_weights: Optional[T] = None,
    include_last_offset: bool = False,
    padding_idx: Optional[int] = None,
    name: str = "_embedding_bag",
) -> Tuple[T, T, T, T]:
    "_embedding_bag"
    return aten_embedding_bag_padding_idx(
        g,
        sts,
        outputs,
        weight,
        indices,
        offsets,
        scale_grad_by_freq,
        mode=mode,
        sparse=sparse,
        per_sample_weights=per_sample_weights,
        include_last_offset=include_last_offset,
        padding_idx=padding_idx,
        name=name,
    )


def aten_embedding_bag_padding_idx(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    weight: T,
    indices: T,  # INT64
    offsets: T,  # INT64
    scale_grad_by_freq: bool = False,
    mode: int = 0,  # [0,1,2] indicate ["sum", "mean", "max"]
    sparse: bool = False,
    per_sample_weights: Optional[T] = None,
    include_last_offset: bool = False,
    padding_idx: Optional[int] = None,
    name: str = "embedding_bag.padding_idx",
) -> Tuple[T, T, T, T]:
    "embedding_bag.padding_idx"
    if padding_idx == -1:
        padding_idx = None
    assert g.has_type(weight), f"unknown type for weight={weight!r}{g.get_debug_msg()}"
    assert padding_idx is None, (
        f"Not implemented for padding_idx={padding_idx}, "
        f"include_last_offset={include_last_offset}{g.get_debug_msg()}"
    )
    # There is no sparse tensor in onnx so that should not matter.
    # assert not sparse, (
    #     f"aten_embedding_bag_padding_idx implemented when "
    #     f"sparse is True{g.get_debug_msg()}"
    # )
    assert not scale_grad_by_freq, (
        f"aten_embedding_bag_padding_idx implemented when "
        f"scale_grad_by_freq is True{g.get_debug_msg()}"
    )
    itype = g.get_type(weight)

    # Change padding_idx to positive value, -1 means the last index
    if padding_idx is not None and padding_idx < 0:
        assert g.has_shape(weight), f"unknown shape for weight={weight!r}{g.get_debug_msg()}"
        shape_weight = g.get_shape(weight)
        assert isinstance(
            shape_weight[0], int
        ), f"Not implemented for dynamic dimension in {shape_weight}{g.get_debug_msg()}"
        padding_idx = shape_weight[0] + padding_idx
        assert padding_idx >= 0, f"Unexpected padding_idx={padding_idx}{g.get_debug_msg()}"

    if mode == 0:  # sum
        assert (
            g.main_opset >= 13
        ), f"Not implemented for opset={g.main_opset} and mode={mode}{g.get_debug_msg()}"
        op_type = "ReduceSum"
    elif mode == 1:  # mean
        assert (
            g.main_opset >= 18
        ), f"Not implemented for opset={g.main_opset} and mode={mode}{g.get_debug_msg()}"
        op_type = "ReduceMean"
    elif mode == 2:
        assert (
            g.main_opset >= 18
        ), f"Not implemented for opset={g.main_opset} and mode={mode}{g.get_debug_msg()}"
        op_type = "ReduceMax"
    else:
        raise AssertionError(f"Unexpected value for mode={mode}{g.get_debug_msg()}")

    # loop_condition = g.op("Constant", value_t=torch.tensor(1))
    # loop_condition = g.op("Cast", loop_condition, to_i=_C_onnx.TensorProtoDataType.BOOL)
    # zero = g.op("Constant", value_t=torch.tensor([0]))
    loop_condition = g.make_initializer("", np.array(True, dtype=np.bool_))
    zero = g.make_initializer("", np.array([0], dtype=np.int64))
    one = g.make_initializer("", np.array([1], dtype=np.int64))
    very_end = g.make_initializer("", np.array([sys.maxsize], dtype=np.int64))

    # indices_len = _unsqueeze_helper(g,
    #   _size_helper(g, indices, g.op("Constant", value_t=torch.tensor(0))),[0],)
    indices_len = g.op.UnsqueezeAnyOpset(
        g.op.Gather(g.op.Shape(indices, name=name), np.array(0, dtype=np.int64), name=name),
        zero,
        name=name,
    )
    if not include_last_offset:
        offsets = g.op.Concat(offsets, indices_len, axis=0, name=name)

    # Offsets holds the starting index position of each bag.
    # So we create a list of the indices slices (determined by
    # offsets) and gather those indices in indices_row.
    # Then we use this subset of indices to gather from embeddings.
    # The embeddings output is a loop scan output,
    # so we can avoid creating a sequence and inserting elements in.
    offsets_starts = g.op.Slice(offsets, zero, very_end, zero, name=name)
    offsets_ends = g.op.Slice(offsets, one, very_end, zero, name=name)

    # loop_len = _size_helper(g, offsets_ends, g.op("Constant", value_t=torch.tensor(0)))
    loop_len = g.op.Gather(g.op.Shape(offsets_ends, name=name), zero, name=name)

    assert (
        g.main_opset >= 18
    ), f"Not implemented for opset={g.main_opset} and mode={mode}{g.get_debug_msg()}"

    if per_sample_weights is None:
        per_sample_nodes = [make_node("Identity", ["embeddings_0"], ["embeddings"])]
    else:
        per_sample_nodes = [
            make_node(
                "Slice",
                [per_sample_weights, "indices_start", "indices_end", "zero"],
                ["per_sample_weights_row_0"],
            ),
            make_node(
                "Unsqueeze",
                ["per_sample_weights_row_0", "one"],
                ["per_sample_weights_row"],
            ),
            make_node("Mul", [weight, "per_sample_weights_row"], ["embeddings"]),
        ]

    nodes = [
        make_node("Gather", [offsets_starts, "block_input_iter"], ["indices_starts"], axis=0),
        make_node("Gather", [offsets_ends, "block_input_iter"], ["indices_ends"], axis=0),
        make_node("Unsqueeze", ["indices_starts", "zero"], ["indices_start"]),
        make_node("Unsqueeze", ["indices_ends", "zero"], ["indices_end"]),
        make_node("Slice", [indices, "indices_start", "indices_end", zero], ["indices_row"]),
        make_node("Gather", [weight, "indices_row"], ["embeddings_0"], axis=0),
        *per_sample_nodes,
        make_node(op_type, ["embeddings", "zero"], ["reduced_embedings"], keepdims=0),
        make_node("Identity", ["cond"], ["cond_out"]),
    ]

    loop_body = make_graph(
        nodes,
        f"loop_body_{name}",
        [
            make_tensor_value_info("block_input_iter", TensorProto.INT64, []),
            make_tensor_value_info("cond", TensorProto.BOOL, []),
        ],
        [
            make_tensor_value_info("cond_out", TensorProto.BOOL, []),
            make_tensor_value_info("reduced_embedings", itype, None),
        ],
        [
            from_array(np.array([0], dtype=np.int64), name="zero"),
            from_array(np.array([1], dtype=np.int64), name="one"),
        ],
    )

    g.make_node(
        "Loop",
        [loop_len, loop_condition],
        [outputs[0]],
        body=loop_body,
        name=name,
    )

    if len(outputs) == 1:
        return outputs[0]

    offsets_shape = g.op.Shape(offsets, name=name)
    if op_type == "ReduceSum":
        offset2bag_shape = g.op.Shape(indices, name=name)
        bag_size_shape = (
            offsets_shape if include_last_offset else g.op.Sub(offsets_shape, one, name=name)
        )
        max_indices_shape = bag_size_shape
    elif op_type == "ReduceMean":
        offset2bag_shape = g.op.Shape(indices, start=0, end=1, name=name)
        bag_size_shape = g.op.Sub(offsets_shape, one, name=name)
        max_indices_shape = bag_size_shape
    elif op_type == "ReduceMax":
        offset2bag_shape = g.op.Shape(indices, start=0, end=1, name=name)
        bag_size_shape = g.op.Sub(offsets_shape, one, name=name)
        # shape = (bag_size.dim[0], weight.dim[1])
        dim_0 = g.op.Gather(bag_size_shape, zero, name=name)
        dim_1 = g.op.Shape(weight, start=1, end=2, name=name)
        max_indices_shape = g.op.Concat(dim_0, dim_1, axis=0, name=name)
    else:
        raise AssertionError(f"Unexpeted op_type={op_type!r}{g.get_debug_msg()}")

    offset2bag = g.op.ConstantOfShape(
        offset2bag_shape,
        value=from_array(np.array([0], dtype=np.int64)),
        name=name,
        outputs=[outputs[1]],
    )
    new_outputs = [outputs[0], offset2bag]
    if len(outputs) > 2:
        bag_size = g.op.ConstantOfShape(
            bag_size_shape,
            value=from_array(np.array([0], dtype=np.int64)),
            name=name,
            outputs=[outputs[2]],
        )
        new_outputs.append(bag_size)
    if len(outputs) > 3:
        max_indices = g.op.ConstantOfShape(
            max_indices_shape,
            value=from_array(np.array([0], dtype=np.int64)),
            name=name,
            outputs=[outputs[3]],
        )
        new_outputs.append(max_indices)
    return tuple(new_outputs)


def aten_empty_like(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    dtype: Optional["torch.dtype"] = None,  # noqa: F821
    layout=None,
    device: Optional["torch.device"] = None,  # noqa: F821
    pin_memory=None,
    memory_format=None,
) -> T:
    "constantofshape"
    return aten_full(
        g,
        sts,
        outputs,
        x,
        0,
        dtype=dtype or g.get_type(x),
        name="empty_like",
    )


def aten_empty_permuted(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    size: T,
    physical_layout: T,
    dtype: Optional["torch.dtype"] = None,  # noqa: F821
    layout=None,
    device: Optional["torch.device"] = None,  # noqa: F821
    requires_grad: bool = False,
    pin_memory: bool = False,
    name: str = "empty_permuted",
) -> T:
    "constantofshape"
    # strided is unused.
    assert list(physical_layout) == list(range(len(physical_layout))), (
        f"empty_permuted not implemented when physical_layout={physical_layout}, "
        f"size={size}{g.get_debug_msg()}"
    )
    return aten_zeros(
        g,
        sts,
        outputs,
        size,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        requires_grad=requires_grad,
        name=name,
    )


def aten_empty_strided(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    size: T,
    stride: T,
    dtype: Optional["torch.dtype"] = None,  # noqa: F821
    layout=None,
    device: Optional["torch.device"] = None,  # noqa: F821
    requires_grad: bool = False,
    pin_memory: bool = False,
    name: str = "empty_strided",
) -> T:
    "constantofshape"
    # strided is unused.
    return aten_zeros(
        g,
        sts,
        outputs,
        size,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        requires_grad=requires_grad,
        name=name,
    )


def aten__enter_autocast(
    g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], *args: List[Any]
) -> T:
    """
    Returns the function returns a dummy which will be removed
    after the graph is created.
    """
    assert all((not isinstance(x, str) or x in {"cpu", "cuda"}) for x in args), (
        f"The function should not take any tensors as input but types are "
        f"{[type(_) for _ in args]}: {args}{g.get_debug_msg()}"
    )
    return g.make_node("Constant", [], value_ints=[1], name="_enter_autocast")


def aten_erf(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    name: str = "erf",
) -> T:
    """erf"""

    res = g.op.Erf(x, outputs=outputs, name=name)
    if not sts:
        set_type_shape_unary_op(g, res, x)
    return res


def aten_eq(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    y: T,
    name="eq",
) -> T:
    "equal"
    x, y = prepare_inputs_homogeneous_operator(g, x, y)
    res = g.op.Equal(x, y, outputs=outputs, name=name)
    if not sts:
        set_type_shape_binary_op(g, outputs[0], x, y, cmp_op=True)
    return res


def aten_eq_Tensor(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    y: T,
    name="eq_Tensor",
) -> T:
    "equal"
    return aten_eq(g, sts, outputs, x, y, name=name)


def aten_eq_Scalar(
    g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T, y: T
) -> T:
    "equal"
    return aten_eq(g, sts, outputs, x, y)


def aten_exp(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    name: str = "exp",
) -> T:
    "exp"
    res = g.make_node("Exp", [x], outputs, name=name)
    if not sts:
        set_type_shape_unary_op(g, outputs[0], x)
    return res


def aten__exit_autocast(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    output_of_enter_auto_cast: T,
) -> T:
    """
    Returns the function returns a dummy which will be removed
    after the graph is created.
    """
    return g.make_node("Identity", [output_of_enter_auto_cast], name="_exit_autocast")


def aten_expand(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    sizes: Union[T, List[Union[int, str]]],
    implicit: bool = False,
    name: str = "expand",
) -> T:
    "expand"
    assert not implicit, f"Unexpected value for implicit={implicit!r}"

    if not isinstance(sizes, str) and all_int(sizes) and min(sizes) >= 0:
        # static sizes
        res = g.op.Expand(x, np.array(sizes, dtype=np.int64), outputs=outputs, name=name)
        if not sts:
            g.set_type(res, g.get_type(x))
            g.set_shape(res, tuple(sizes))
        return res

    if not isinstance(sizes, str) and g.has_shape(x):
        shape = g.get_shape(x)
        while len(shape) < len(sizes):
            shape = (1, *shape)
        assert len(shape) == len(
            sizes
        ), f"Unable to expand x with shape={shape} because sizes={sizes}{g.get_debug_msg()}"
        new_shape = []
        is_static = True
        shape_x = None
        for di, (a, b) in enumerate(zip(shape, sizes)):
            if b == -1:
                assert isinstance(b, int), (
                    f"Not implemented when the shape is not fully known, "
                    f"shape={shape} for x as sizes={sizes}{g.get_debug_msg()}"
                )
                if isinstance(a, int):
                    new_shape.append(a)
                else:
                    if (
                        a in g.dynamic_objects
                        and isinstance(g.dynamic_objects[a], str)
                        and g.has_name(g.dynamic_objects[a])
                    ):
                        new_shape.append(g.dynamic_objects[a])
                    else:
                        if shape_x is None:
                            shape_x = g.op.Shape(x, name=name)
                        ds = g.op.Gather(shape_x, np.array([di], dtype=np.int64), name=name)
                        new_shape.append(ds)
                        g.add_dynamic_object(a, ds)
                    is_static = False
            else:
                new_shape.append(b)
                is_static = False
        i_new_shape = (
            np.array(new_shape, dtype=np.int64)
            if is_static
            else g.make_shape_from_results(new_shape, name=f"{name}_neg")
        )
        res = g.op.Expand(x, i_new_shape, outputs=outputs, name=f"{name}_neg")
        if not sts:
            g.set_type(res, g.get_type(x))
            g.set_shape(res, tuple(new_shape))
        return res

    if isinstance(sizes, (list, tuple)):
        # A combination of static and dynamic dimensions.
        new_shape = g.make_shape_from_results(list(sizes), name=f"{name}_dyn")
    else:
        new_shape = sizes

    res = g.op.Expand(
        x,
        g.op.Abs(new_shape, name=f"{name}_dyn"),
        outputs=outputs,
        name=f"{name}_dyn",
    )
    if not sts:
        g.set_type(res, g.get_type(x))
        g.set_rank(res, len(sizes))
    return res


def aten_fill_Scalar(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    v: T,
    name: str = "fill_Scalar",
) -> T:
    "constantofshape"
    if g.has_shape(x) and is_static_shape(g.get_shape(x)):
        return aten_full(
            g,
            sts,
            outputs,
            g.get_shape(x),
            v,
            dtype=g.get_type(x),
            name=f"{name}_static",
        )
    return aten_full(
        g,
        sts,
        outputs,
        x,  # full like
        v,
        dtype=g.get_type(x),
        name=f"{name}_dynamic",
    )


def aten_fill_Tensor(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    v: T,
    name: str = "fill_Tensor",
) -> T:
    "constantofshape"
    assert g.get_type(x) == g.get_type(
        v
    ), f"Type mismatch {g.get_type(x)} != {g.get_type(v)}{g.get_debug_msg()}"
    shape = g.op.Shape(x, name=name)
    return g.op.Expand(v, shape, name=name, outputs=outputs)


def aten_flatten(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    start_dim: int = 1,
    end_dim: int = -1,
    name: str = "flatten",
) -> T:
    "flatten"
    if start_dim < 0:
        assert g.has_rank(
            x
        ), f"Current implementation requires rank of {x!r}{g.get_debug_msg()}"
        rk = g.get_rank(x)
        start_dim += rk
    if start_dim != 0:
        if g.has_rank(x):
            rk = g.get_rank(x)
            if end_dim == rk - 1:
                end_dim = -1
        if start_dim == 1 and end_dim == -1:
            shape = g.op.Shape(x, name=name)
            take = g.op.GatherElements(shape, np.array([0], dtype=np.int64), axis=0, name=name)
            resh = g.op.Concat(take, np.array([-1], dtype=np.int64), axis=0, name=name)
            return g.op.Reshape(x, resh, outputs=outputs, name=name)
        if end_dim == -1:
            shape = g.op.Shape(x, name=name)
            take = g.op.GatherElements(
                shape, np.arange(start_dim).astype(np.int64), axis=0, name=name
            )
            resh = g.op.Concat(take, np.array([-1], dtype=np.int64), axis=0, name=name)
            return g.op.Reshape(x, resh, outputs=outputs, name=name)

        # x='_onx_tile03', start_dim=3, end_dim=-1 not supported, GPTJForCausalLM
        raise NotImplementedError(
            f"x={x!r}, start_dim={start_dim}, end_dim={end_dim} "
            f"not supported{g.get_debug_msg()}"
        )
    if end_dim == -1:
        return g.make_node("Flatten", [x], outputs, name=name)
    res = g.make_node("Flatten", [x], outputs, to=end_dim)
    if not sts:
        g.set_type(res, g.get_type(x))
        if g.has_shape(x, full=True):
            g.set_shape(res, (int(np.prod(g.get_shape(x)))))
        else:
            g.set_rank(res, 1)
    return res


def aten_floor(g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T) -> T:
    """floor"""
    res = g.op.Floor(x, name="floor", outputs=outputs)
    if not sts:
        set_type_shape_unary_op(g, res, x)
    return res


def aten_floordiv(
    g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T, y: T
) -> T:
    """floor + div"""
    return aten_floor_divide(g, sts, outputs, x, y, name="floordiv")


def aten_floor_divide(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    y: T,
    name="floor_divide",
) -> T:
    """floor + div"""
    if isinstance(y, str) and isinstance(x, str):
        div = g.op.Div(x, y, name=name)
        itype = g.get_type(x)
        g.set_rank(div, max(g.get_rank(x), g.get_rank(y)))
    elif isinstance(x, str) and isinstance(y, int):
        itype = g.get_type(x)
        dtype = tensor_dtype_to_np_dtype(itype)
        div = g.op.Div(x, np.array([y], dtype=dtype), name=name)
        if g.has_shape(x):
            g.set_shape(div, g.get_shape(x))
        else:
            g.set_rank(div, g.get_rank(x))
    elif isinstance(x, int) and isinstance(y, str):
        itype = g.get_type(y)
        dtype = tensor_dtype_to_np_dtype(itype)
        div = g.op.Div(np.array([x], dtype=dtype), y, name=name)
        if g.has_shape(y):
            g.set_shape(div, g.get_shape(y))
        else:
            g.set_rank(div, g.get_rank(y))
    else:
        raise AssertionError(
            f"Unable to implement floordiv for types {[type(x), type(y)]}{g.get_debug_msg()}"
        )

    g.set_type(div, itype)
    if itype in {
        TensorProto.INT64,
        TensorProto.INT32,
        TensorProto.UINT64,
        TensorProto.UINT32,
    }:
        res = g.op.Identity(div, outputs=outputs, name=name)
    else:
        assert itype in {
            TensorProto.FLOAT,
            TensorProto.DOUBLE,
            TensorProto.FLOAT16,
            TensorProto.BFLOAT16,
        }, f"Unexpected itype={itype}{g.get_debug_msg()}"
        res = g.op.Floor(div, outputs=outputs, name=name)
    if not sts:
        g.set_type(res, itype)
        if g.has_shape(div):
            g.set_shape(res, g.get_shape(div))
        else:
            g.set_rank(res, g.get_rank(div))
    return res


def aten_full(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    size: T,
    fill_value: float,
    dtype: Optional["torch.dtype"] = None,  # noqa: F821
    layout=None,
    device: Optional["torch.device"] = None,  # noqa: F821
    pin_memory=None,
    requires_grad: bool = False,
    name: str = "full",
) -> T:
    "constantofshape"
    import torch

    assert layout in (None, torch.strided), (
        f"full not implemented for layout={layout!r} is not None, "
        f"size={size!r}, dtype={dtype}{g.get_debug_msg()}"
    )
    assert not requires_grad, "aten_full does not implement requires_grad"
    assert not pin_memory, "full not implemented for pin_memory=True"
    assert fill_value is None or isinstance(
        fill_value, (float, int)
    ), f"Unexpected type {type(fill_value)} for fill_value."

    new_shape = None

    if isinstance(size, tuple) and all_int(size):
        if len(size) == 0:
            tsize = None
        else:
            tsize = np.array(size, dtype=np.int64)
            new_shape = size
    elif isinstance(size, (list, tuple)):
        if len(size) == 0:
            tsize = None
        elif all_int(size):
            tsize = np.array(size, dtype=np.int64)
            new_shape = size
        else:
            tsize = g.make_shape_from_results(size, name=f"{name}_make_shape")
    elif isinstance(size, str):
        if g.has_shape(size) and is_static_shape(size):
            tsize = np.array(g.get_shape(size), dtype=np.int64)
        else:
            tsize = g.op.Shape(size, name=f"{name}_shape")
    else:
        raise RuntimeError(f"Unexpected type {type(size)} for size.")

    if dtype is None:
        if fill_value is None or isinstance(fill_value, float):
            value = np.array(fill_value, dtype=np.float32).reshape((1,))
            itype = TensorProto.FLOAT
        elif isinstance(fill_value, int):
            value = np.array(fill_value, dtype=np.int64).reshape((1,))
            itype = TensorProto.INT64
        else:
            itype = torch_dtype_to_onnx_dtype(type(fill_value))
            ntype = tensor_dtype_to_np_dtype(itype)
            value = np.array(fill_value, dtype=ntype).reshape((1,))
    else:
        itype = dtype if isinstance(dtype, int) else torch_dtype_to_onnx_dtype(dtype)
        ntype = tensor_dtype_to_np_dtype(itype)
        value = np.array(fill_value or 0, dtype=ntype).reshape((1,))

    if tsize is None:
        # A scalar
        v = from_array(value.squeeze())
        res = g.op.Constant(value=v, outputs=outputs, name=name)
        if not sts:
            g.set_type(res, itype)
            g.set_shape(res, tuple())
        return res

    res = g.op.ConstantOfShape(tsize, value=from_array(value), outputs=outputs, name=name)
    if not sts:
        g.set_type(res, itype)
        if new_shape:
            g.set_shape(res, new_shape)

    # size = op.Cast(size, to=INT64.dtype)
    # fill_value = op.Cast(fill_value, to=dtype)
    # return op.Expand(fill_value, size)
    return res


def aten_full_like(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    fill_value: T,
    dtype: Optional["torch.dtype"] = None,  # noqa: F821
    layout=None,
    device: Optional["torch.device"] = None,  # noqa: F821
    pin_memory=None,
    memory_format=None,
    name: str = "full_like",
) -> T:
    "constantofshape"
    import torch

    assert layout is None, f"empty_like not implemented for layout={layout!r} is not None"
    assert not pin_memory, "empty_like not implemented for pin_memory=True"
    assert (
        memory_format is None or memory_format == torch.preserve_format
    ), f"empty_like not implemented for memory_format={memory_format}"

    if g.has_shape(x) and is_static_shape(g.get_shape(x)):
        # simple case
        return aten_full(
            g,
            sts,
            outputs,
            g.get_shape(x),
            fill_value,
            dtype=dtype or g.get_type(x),
            name=name,
        )
    return aten_full(
        g,
        sts,
        outputs,
        g.op.Shape(x, name="full_like"),
        fill_value,
        dtype=dtype or g.get_type(x),
        name=name,
    )


def aten_FunctionCtx(
    g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], *args, **kwargs
):
    "not implemented"
    if len(args) == 0 and len(kwargs) == 0:
        return
    raise NotImplementedError(f"args={args}, kwargs={kwargs}")


def aten_gather(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    dim: int,
    index: T,
    sparse_grad: bool = False,
    name: str = "gather",
) -> T:
    """gather"""
    assert (
        not sparse_grad
    ), f"not implemented with sparse_grad={sparse_grad}{g.get_debug_msg()}"
    assert isinstance(index, str), f"not implemented with index={index}{g.get_debug_msg()}"
    assert g.has_shape(index), f"Missing shape for index={index}{g.get_debug_msg()}"

    shape = g.has_shape(index)
    if shape in (tuple(), (1,)):
        # index is a scalar
        raise AssertionError(f"shape={shape!r} a scalar{g.get_debug_msg()}")

    if g.get_type(index) != TensorProto.INT64:
        new_index = g.op.Cast(index, to=TensorProto.INT64, name=name)
    else:
        new_index = index

    res = g.op.GatherElements(x, new_index, axis=dim, name=name, outputs=outputs)

    # if g.is_constant(index)
    # if IsScalar(index):  # When (index) is empty, return (self)
    #    result = self
    # else:
    #    if IsScalar(self):  # Unsqueeze for GatherElements op
    #        self = op.Reshape(self, op.Constant(value_ints=[-1]))
    #    if op.Size(index) == 0:  # Return empty array
    #        result = op.CastLike(index, self)

    if not sts:
        g.set_type(res, g.get_type(x))
    return res


def aten_ge(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    y: T,
    name: str = "ge",
) -> T:
    "greater or equal"
    x, y = prepare_inputs_homogeneous_operator(g, x, y, name=name)
    res = g.op.GreaterOrEqual(x, y, outputs=outputs, name=name)
    if not sts:
        set_type_shape_binary_op(g, outputs[0], x, y, cmp_op=True)
    return res


def aten_ge_Scalar(
    g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T, y: T
) -> T:
    "greater or equal"
    return aten_ge(g, sts, outputs, x, y, name="ge_Scalar")


def aten_ge_Tensor(
    g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T, y: T
) -> T:
    "greater or equal"
    return aten_ge(g, sts, outputs, x, y, name="ge_Tensor")


def aten_gelu(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    approximate: str = "none",
    name: str = "gelu",
) -> T:
    """gelu"""

    if g.main_opset >= 20:
        res = g.op.Gelu(x, approximate=approximate, name=name, outputs=outputs)
        if not sts:
            set_type_shape_unary_op(g, res, x)
        return res

    if approximate == "none":
        # GELU(x) = 0.5 * x * [1 + ERF(x/sqrt(2)]
        dtype = tensor_dtype_to_np_dtype(g.get_type(x))
        inner = g.op.Div(x, np.array([1.4142135623730951], dtype=dtype), name=name)
        erf = g.op.Erf(inner, name=name)
        inner = g.op.Add(erf, np.array([1], dtype=dtype), name=name)
        inner = g.op.Mul(x, inner, name=name)
        res = g.op.Mul(np.array([0.5], dtype=dtype), inner, name=name, outputs=outputs)
        if not sts:
            set_type_shape_unary_op(g, res, x)
        return res

    if approximate == "tanh":
        # GELU(x) = 0.5 * x * {1 + Tanh[\sqrt(2/pi) * (x + 0.044715 * x^3)]}
        dtype = tensor_dtype_to_np_dtype(g.get_type(x))
        cubed = g.op.Pow(x, np.array([3], dtype=np.int64), name=name)
        inner = g.op.Mul(np.array([0.044715], dtype=dtype), cubed, name=name)
        inner = g.op.Add(x, inner, name=name)
        # Prefer explicit graph construction over precomputed constants for clarity.
        inner = g.op.Mul(np.array([(2 / math.pi) ** 0.5], dtype=dtype), inner, name=name)
        inner = g.op.Tanh(inner, name=name)
        inner = g.op.Add(inner, np.array([1], dtype=dtype), name=name)
        inner = g.op.Mul(x, inner, name=name)
        res = g.op.Mul(np.array([0.5], dtype=dtype), inner, outputs=outputs, name=name)
        if not sts:
            set_type_shape_unary_op(g, res, x)
        return res

    raise AssertionError(
        f"Unexpected value for approximate={approximate!r}{g.get_debug_msg()}"
    )


def aten_group_norm(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    num_groups: int,
    weight: Optional[T] = None,
    bias: Optional[T] = None,
    eps: float = 1e-05,
    cudnn_enabled: bool = True,
    name: str = "group_norm",
) -> T:
    "instance_normalization"
    assert g.has_rank(x), f"rank for {x!r} is unknown{g.get_debug_msg()}"
    assert g.has_shape(x), f"shape for {x!r} is unknown{g.get_debug_msg()}"
    shape_x = g.get_shape(x)
    assert len(shape_x) > 1, f"Unexpected shape {shape_x} for {x!r}{g.get_debug_msg()}"
    channel_size = shape_x[1]
    assert is_static_dimension(
        channel_size
    ), f"number of channels cannot be dynamic, shape={shape_x}{g.get_debug_msg()}"
    assert channel_size % num_groups == 0, (
        f"number of channels {channel_size} must a multiple of num_groups="
        f"{num_groups}{g.get_debug_msg()}"
    )
    input_rank = g.get_rank(x)

    # 0 in the shape list keeps dimension value unchanged.
    if is_static_dimension(shape_x[0]):
        new_shape = np.array([shape_x[0], num_groups, -1], dtype=np.int64)
        input_reshaped = g.op.Reshape(x, new_shape, name=name)
    else:
        raise AssertionError(
            f"Dynamic batch size for shape={shape_x} "
            f"is not implemented yet{g.get_debug_msg()}"
        )

    # C is always divisible by num_groups
    # Due to shape difference. we need to apply weight and bias after
    # instance norm computation and reshape
    itype = g.get_type(x)
    dtype = tensor_dtype_to_np_dtype(itype)
    weight_ = np.array([1] * num_groups, dtype=dtype)
    bias_ = np.array([0] * num_groups, dtype=dtype)

    norm_reshaped = g.op.InstanceNormalization(
        input_reshaped, weight_, bias_, epsilon=eps, name=name
    )
    g.set_type(norm_reshaped, itype)
    if is_static_shape(shape_x):
        shape_x_name = np.array(shape_x, dtype=np.int64)
    else:
        shape_x_name = g.op.Shape(x, name=name)
    norm = g.op.Reshape(norm_reshaped, shape_x_name, name=name)
    g.set_type(norm, itype)
    g.set_shape(norm, shape_x)

    np_axes = np.array(list(range(1, input_rank - 1)), dtype=np.int64)
    if weight is None:
        w = np.array([1], dtype=dtype)
    else:
        w = g.op.Unsqueeze(weight, np_axes, name=name)

    if bias is None:
        b = np.array([0], dtype=dtype)
    else:
        b = g.op.Unsqueeze(bias, np_axes, name=name)

    # Norm has shape [N, C, *] so we reshape weight and bias to [C, *]
    res = g.op.Add(g.op.Mul(norm, w, name=name), b, name=name, outputs=outputs)
    return res


def aten_gt(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    y: T,
    name: str = "gt",
) -> T:
    "greater"
    x, y = prepare_inputs_homogeneous_operator(g, x, y, name=name)
    res = g.op.Greater(x, y, outputs=outputs, name=name)
    if not sts:
        set_type_shape_binary_op(g, outputs[0], x, y, cmp_op=True)
    return res


def aten_gt_Scalar(
    g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T, y: T
) -> T:
    "greater"
    return aten_gt(g, sts, outputs, x, y, name="gt_Scalar")


def aten_gt_Tensor(
    g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T, y: T
) -> T:
    "greater"
    return aten_gt(g, sts, outputs, x, y, name="gt_Tensor")


def aten_hardsigmoid(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    name: str = "hardsigmoid",
) -> T:
    """hardsigmoid"""
    res = g.op.HardSigmoid(x, alpha=1.0 / 6, beta=0.5, name=name, outputs=outputs)
    if not sts:
        set_type_shape_unary_op(g, res, x)
    return res


def aten_hardswish(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    name: str = "hardswish",
) -> T:
    """hardswish"""
    res = g.op.HardSwish(x, name=name, outputs=outputs)
    if not sts:
        set_type_shape_unary_op(g, res, x)
    return res


def aten_hardtanh(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    min_val: float = -1.0,
    max_val: float = 1.0,
    name: str = "hardtanh",
) -> T:
    """hardtanh(Tensor self, Scalar min_val=-1, Scalar max_val=1) -> Tensor"""
    dtype = tensor_dtype_to_np_dtype(g.get_type(x))
    res = g.op.Clip(
        x,
        np.array(min_val, dtype=dtype),
        np.array(max_val, dtype=dtype),
        name=name,
        outputs=outputs,
    )
    if not sts:
        set_type_shape_unary_op(g, res, x)
    return res


def aten_hardtanh_backward(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    grad_output: T,
    x: T,
    min_val: float,
    max_val: float,
    name: str = "hardtanh_backward",
) -> T:
    """hardtanh_backward"""
    dtype = tensor_dtype_to_np_dtype(g.get_type(x))
    max_mask = g.op.Where(
        g.op.Greater(x, np.array([max_val], dtype=dtype), name=name),
        np.array([0.0], dtype=dtype),
        np.array([1.0], dtype=dtype),
        name=name,
    )
    min_mask = g.op.Where(
        g.op.Less(x, np.array([min_val], dtype=dtype), name=name),
        np.array([0.0], dtype=dtype),
        np.array([1.0], dtype=dtype),
        name=name,
    )
    res = g.op.Mul(
        g.op.Mul(grad_output, max_mask, name=name), min_mask, name=name, outputs=outputs
    )
    if not sts:
        set_type_shape_unary_op(g, res, x)
    return res


def _get_im2col_indices_along_dim(
    input_d: DYNAMIC_SHAPE,
    kernel_size_d: int,
    dilation_d: int,
    padding_d: int,
    stride_d: int,
):
    # Input is always 4-D (N, C, H, W)
    # Calculate indices of sliding blocks along spatial dimension
    # Slide kernel over input each dim d:
    # each dimension d ranges from 0 to input[d]+2xpadding[d]-dilation[d]x(kernel_size[d]-1)
    # with steps = stride

    if is_static_dimension(input_d):
        blocks_d = input_d + ((padding_d * 2) - (dilation_d * (kernel_size_d - 1)))

        # Stride kernel over input and find starting indices along dim d
        blocks_d_indices = np.array([list(range(0, blocks_d, stride_d))], dtype=np.int64)

        # Apply dilation on kernel and find its indices along dim d
        kernel_mask = np.array(
            [list(range(0, kernel_size_d * dilation_d, dilation_d))], dtype=np.int64
        ).T

        # Broadcast and add kernel staring positions (indices) with
        # kernel_grid along dim d, to get block indices along dim d
        block_mask = blocks_d_indices + kernel_mask
        return block_mask

    raise AssertionError(f"Not impelmented yet for dynamic shapes, input_d={input_d!r}")


def aten_im2col(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    kernel_size: Sequence[int],
    dilation: Sequence[int] = (1, 1),
    padding: Sequence[int] = (0, 0),
    stride: Sequence[int] = (1, 1),
    name: str = "im2col",
) -> T:
    """im2col"""
    if not isinstance(kernel_size, Sequence):
        kernel_size = (kernel_size, kernel_size)
    kernel_sizes = list(kernel_size)

    if not isinstance(dilation, Sequence):
        dilation = (dilation, dilation)
    dilations = list(dilation)

    if not isinstance(padding, Sequence):
        padding = (padding, padding)
    pads = list(padding)

    if isinstance(stride, int):
        stride = (stride, stride)
    strides = list(stride)

    stride_h, stride_w = strides[0], strides[1]
    padding_h, padding_w = pads[0], pads[1]
    dilation_h, dilation_w = dilations[0], dilations[1]
    kernel_h, kernel_w = kernel_sizes[0], kernel_sizes[1]

    if g.has_shape(x) and is_static_shape(g.get_shape(x)):
        input_shape = g.get_shape(x)
        input_h = input_shape[2]
        input_w = input_shape[3]

        blocks_row_indices = _get_im2col_indices_along_dim(
            input_h, kernel_h, dilation_h, padding_h, stride_h
        )
        blocks_col_indices = _get_im2col_indices_along_dim(
            input_w, kernel_w, dilation_w, padding_w, stride_w
        )

        batch_dim = input_shape[0]
        channel_dim = input_shape[1]
        channel_unfolded = channel_dim * kernel_h * kernel_w
        output_shape = np.array([batch_dim, channel_unfolded, -1], dtype=np.int64)

        padded_input = g.op.Pad(
            x,
            np.array([0, 0, padding_h, padding_w, 0, 0, padding_h, padding_w], dtype=np.int64),
            name=name,
        )
        g.set_type(padded_input, g.get_type(x))

        output = g.op.Gather(padded_input, blocks_row_indices, axis=2, name=name)
        g.set_type(output, g.get_type(x))
        output = g.op.Gather(output, blocks_col_indices, axis=4, name=name)
        g.set_type(output, g.get_type(x))
        output = g.op.Transpose(output, perm=[0, 1, 2, 4, 3, 5], name=name)
        g.set_type(output, g.get_type(x))
        return g.op.Reshape(output, output_shape, outputs=outputs, name=name)

    raise AssertionError(f"Not implemented with dynamic shape for {x!r}{g.get_debug_msg()}")


def aten_index_Tensor(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    indices: List[int],
    name: str = "index_Tensor",
) -> T:
    "[...,:, ...]"
    assert isinstance(indices, (list, tuple)), f"Unexpected type {type(indices)} for indices"
    if len(indices) == 1 and isinstance(indices[0], str):
        return aten_index_select(
            g, sts, outputs, x, dim=0, index=indices[0], name="index1_Tensor"
        )

    n_none = len([i for i in indices if i is None])
    if n_none == 0:
        # No none dimension, example:
        # indices = [A, B]
        # shape(X) = (4, 1024, 2)
        # shape(A) = (4,)
        # shape(B) = (B,)
        # X[A, B] = ...
        shapes = [g.get_shape(i) for i in indices]
        assert (
            len(set(shapes)) == 1
        ), f"aten_index is not implemented for shapes={shapes} (1){g.get_debug_msg()}"
        same_shape = shapes[0]
        assert (
            len(same_shape) == 1
        ), f"aten_index is not implemented for shapes={shapes} (2){g.get_debug_msg()}"
        reshaped = [
            g.op.Reshape(i, np.array([-1, 1], dtype=np.int64), name=name) for i in indices
        ]
        concat = g.op.Concat(*reshaped, axis=-1, name=name)
        res = g.op.GatherND(x, concat, batch_dims=0, outputs=outputs)
        if not sts:
            g.set_type(res, g.get_type(x))
        return res

    if n_none == 1 and indices[0] is None and len(indices) == 3:
        shapes = [g.get_shape(i) for i in indices if i is not None]
        assert (
            len(set(shapes)) == 1
        ), f"aten_index is not implemented for shapes={shapes} (1){g.get_debug_msg()}"
        same_shape = shapes[0]
        assert (
            len(same_shape) == 1
        ), f"aten_index is not implemented for shapes={shapes} (2){g.get_debug_msg()}"
        dim = g.op.Shape(x, start=1, end=2, name=name)
        flat_index = g.op.Add(g.op.Mul(indices[1], dim, name=name), indices[2], name=name)

        dimx1 = g.op.Shape(x, start=0, end=1, name=name)
        new_shapex = g.op.Concat(dimx1, np.array([-1], dtype=np.int64), name=name, axis=0)
        reshaped_x = g.op.Reshape(x, new_shapex, name=name)

        res = g.op.Gather(reshaped_x, flat_index, axis=1, outputs=outputs)
        if not sts:
            g.set_type(res, g.get_type(x))
        return res

    if n_none == len(indices) - 1:
        # only one dimension is not None, the others must be added
        position = min(i for i, v in enumerate(indices) if v is not None)
        index = indices[position]
        if isinstance(index, str):
            res = aten_index_select(
                g,
                sts,
                None,
                x,
                dim=position,
                index=index,
                name="index2_Tensor",
            )
            to_add = [i for i in range(len(indices)) if i != position]
            assert len(to_add) > 0, (
                f"Unexpected value for to_add={to_add}, "
                f"position={position}, indices={indices}"
            )
            return g.op.Identity(res, name="index2_Tensor", outputs=outputs)

    raise RuntimeError(
        f"aten_index_Tensor not implemented yet for indices={indices}, "
        f"n_none={n_none}{g.get_debug_msg()}"
    )


def aten_index_put(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    indices: List[T],
    values: T,
    accumulate: bool = False,
    name="aten_index_put",
) -> T:
    "M[..., :, ...] = ..."
    assert isinstance(indices, list), f"Unexpected type {type(indices)}{g.get_debug_msg()}"
    assert g.has_shape(x), f"Missing shape for {x!r}{g.get_debug_msg()}"

    if len(indices) == 1:
        name = f"{name}1"
        index = indices[0]  # tensor
        index_dtype = g.get_type(index)
        if index_dtype == TensorProto.BOOL:
            name += "b_"
            assert not accumulate, (
                f"accumulate is True but it does not make sense in that case"
                f"{g.get_debug_msg()}"
            )
            res = g.op.Where(index, values, x, outputs=outputs)
            if not sts:
                g.set_type(res, g.get_type(x))
                g.set_shape(res, g.get_shape(x))
            return res

        new_index = g.op.UnsqueezeAnyOpset(index, np.array([-1], dtype=np.int64), name=name)
        g.set_type(new_index, index_dtype)
        if g.has_shape(index):
            name += "s_"
            g.set_shape(new_index, (*g.get_shape(index), 1))
        else:
            name += "r_"
            g.set_rank(new_index, g.get_rank(index) + 1)

        if accumulate:
            assert g.main_opset >= 13, (
                f"index_put cannot be implemented for opset < 13 "
                f"because ScatterND does not support reduction"
                f"{g.get_debug_msg()}"
            )
            res = g.op.ScatterND(
                x, new_index, values, name=name, reduction="add", outputs=outputs
            )
        else:
            res = g.op.ScatterND(x, new_index, values, name=name, outputs=outputs)

        if not sts:
            g.set_type(res, g.get_type(x))
            g.set_shape(res, g.get_shape(x))
        return res

    def _make_range_or_cast(
        ind: Any, shape_x: DYNAMIC_SHAPE, static_shape: bool, dim: int, name: str
    ) -> Tuple[Any, bool]:
        if ind is not None:
            if g.get_type(ind) != TensorProto.INT64:
                return g.op.Cast(ind, to=TensorProto.INT64, name=name), False
            return ind, False
        if static_shape:
            return np.arange(shape_x[dim]).astype(np.int64), True
        return (
            g.op.Range(
                np.array([0], dtype=np.int64),
                g.op.Gather(shape_x, np.array([dim], dtype=np.int64), name=name),
                np.array([1], dtype=np.int64),
                name=name,
            ),
            True,
        )

    if len(indices) == 2:
        # copy[kv_index, indices] = update.transpose(1, 0)
        ind0, ind1 = indices
        if (
            (
                ind0 is None
                or (
                    g.has_rank(ind0)
                    and g.get_rank(ind0) == 1
                    and g.get_type(ind0) in {TensorProto.INT64, TensorProto.INT32}
                )
            )
            and (
                ind1 is None
                or (
                    g.has_rank(ind1)
                    and g.get_rank(ind1) == 1
                    and g.get_type(ind1) in {TensorProto.INT64, TensorProto.INT32}
                )
            )
            and g.has_rank(values)
            and g.has_rank(x)
        ):
            name = (
                f"{name}2i{'o' if ind0 is None else g.get_rank(ind0)}"
                f"i{'o' if ind1 is None else g.get_rank(ind1)}"
                f"x{g.get_rank(x)}_v{g.get_rank(values)}"
            )
            assert g.get_rank(x) == 2 and (
                g.get_rank(values) == 1 or ind0 is None or ind1 is None
            ), (
                f"No implementation for index_put when indices={indices}, "
                f"rk(x)={g.get_rank(x)}, rk(values)={g.get_rank(values)} "
                f"{g.get_debug_msg()}"
            )
            if g.has_shape(x) and is_static_shape(g.get_shape(x)):
                static_shape = True
                shape_x = np.array(g.get_shape(x), dtype=np.int64)
                n_cols = shape_x[1:2]
                size = np.prod(shape_x).astype(np.int64)
                arange_1d = np.arange(0, size).astype(np.int64)
            else:
                static_shape = False
                shape_x = g.op.Shape(x, name=name)
                n_cols = g.op.GatherElements(shape_x, np.array([1], dtype=np.int64), name=name)
                size = g.op.Size(x, name=name)
                arange_1d = g.op.Range(
                    np.array(0, dtype=np.int64),
                    size,
                    np.array(1, dtype=np.int64),
                    name=name,
                )

            ind0, do_expand0 = _make_range_or_cast(ind0, shape_x, static_shape, 0, name)
            ind1, do_expand1 = _make_range_or_cast(ind1, shape_x, static_shape, 1, name)

            if do_expand0 or do_expand1:
                if (
                    isinstance(ind0, np.ndarray)
                    or (g.has_shape(ind0) and is_static_shape(g.get_shape(ind0)))
                ) and (
                    isinstance(ind1, np.ndarray)
                    or (g.has_shape(ind1) and is_static_shape(g.get_shape(ind1)))
                ):
                    sh0 = ind0.shape if isinstance(ind0, np.ndarray) else g.get_shape(ind0)
                    sh1 = ind1.shape if isinstance(ind1, np.ndarray) else g.get_shape(ind1)
                    new_shape = np.hstack([sh0, sh1]).astype(np.int64)
                else:
                    new_shape = g.op.Concat(
                        g.op.Shape(ind0, name=name),
                        g.op.Shape(ind1, name=name),
                        axis=0,
                        name=name,
                    )
                expanded = g.op.Expand(values, new_shape, name=name)
                indices_2d = g.op.Add(
                    g.op.Mul(
                        g.op.Reshape(ind0, np.array([1, -1], dtype=np.int64), name=name),
                        n_cols,
                        name=name,
                    ),
                    g.op.Reshape(ind1, np.array([1, -1], dtype=np.int64), name=name),
                    name=name,
                )
            else:
                indices_2d = g.op.Add(g.op.Mul(ind0, n_cols, name=name), ind1, name=name)
                expanded = values

            indices_1d = g.op.GatherElements(
                arange_1d,
                g.op.Reshape(indices_2d, np.array([-1], dtype=np.int64), name=name),
                name=name,
            )

            expanded = g.op.Reshape(expanded, np.array([-1], dtype=np.int64), name=name)
            flat_x = g.op.Reshape(x, np.array([-1], dtype=np.int64), name=name)
            if accumulate:
                flat_up_x = g.op.ScatterElements(
                    flat_x, indices_1d, expanded, name=name, reduction="add"
                )
            else:
                flat_up_x = g.op.ScatterElements(flat_x, indices_1d, expanded, name=name)
            g.set_type(flat_up_x, g.get_type(x))

            res = g.op.Reshape(flat_up_x, shape_x, name=name, outputs=outputs)
            if not sts:
                set_type_shape_unary_op(g, res, x)
            return res

        raise AssertionError(
            f"No implementation for index_put when indices={indices}, "
            f"rk(x)={g.get_rank(x)}, rk(values)={g.get_rank(values)} "
            f"{g.get_debug_msg()}"
        )

    if len(indices) == 3:
        # copy[index, middle, indices] = update.transpose(1, 0)
        # index_put.default(args = (%clone, [%kv_index, %arange], %view), kwargs = {})
        ind0, ind1, ind2 = indices
        if (
            (
                ind0 is None
                or (
                    g.has_rank(ind0)
                    and g.get_rank(ind0) == 1
                    and g.get_type(ind0) in {TensorProto.INT64, TensorProto.INT32}
                )
            )
            and (
                ind1 is None
                or (
                    g.has_rank(ind1)
                    and g.get_rank(ind1) == 1
                    and g.get_type(ind1) in {TensorProto.INT64, TensorProto.INT32}
                )
            )
            and (
                ind2 is None
                or (
                    g.has_rank(ind2)
                    and g.get_rank(ind2) == 1
                    and g.get_type(ind2) in {TensorProto.INT64, TensorProto.INT32}
                )
            )
            and g.has_rank(values)
            and g.has_rank(x)
        ):
            name = (
                f"{name}3i{'o' if ind0 is None else g.get_rank(ind0)}"
                f"i{'o' if ind1 is None else g.get_rank(ind1)}"
                f"i{'o' if ind2 is None else g.get_rank(ind2)}"
                f"x{g.get_rank(x)}v{g.get_rank(values)}_"
            )
            assert g.get_rank(x) == 3 and (
                g.get_rank(values) == 1 or ind0 is None or ind1 is None or ind2 is None
            ), (
                f"No implementation for index_put when indices={indices}, "
                f"rk(x)={g.get_rank(x)}, rk(values)={g.get_rank(values)} "
                f"{g.get_debug_msg()}"
            )

            if g.has_shape(x) and is_static_shape(g.get_shape(x)):
                static_shape = True
                shape_x = np.array(g.get_shape(x), dtype=np.int64)
                stride_1 = np.prod(shape_x[1:3]).reshape((-1,)).astype(np.int64)
                stride_2 = shape_x[2:3]
                size = np.prod(shape_x).astype(np.int64)
                arange_1d = np.arange(0, size).reshape((-1,)).astype(np.int64)
            else:
                static_shape = False
                shape_x = g.op.Shape(x, name=name)
                stride_1 = g.op.ReduceProd(
                    g.op.GatherElements(
                        shape_x, np.array([1, 2], dtype=np.int64), name=name, keepdim=1
                    ),
                    name=name,
                    keepdim=1,
                )
                stride_2 = g.op.GatherElements(
                    shape_x, np.array([2], dtype=np.int64), name=name
                )
                size = g.op.Size(x, name=name)
                arange_1d = g.op.Range(
                    np.array(0, dtype=np.int64),
                    size,
                    np.array(1, dtype=np.int64),
                    name=name,
                )

            ind0, expanded0 = _make_range_or_cast(ind0, shape_x, static_shape, 0, name)
            ind1, expanded1 = _make_range_or_cast(ind1, shape_x, static_shape, 1, name)
            ind2, expanded2 = _make_range_or_cast(ind2, shape_x, static_shape, 2, name)

            if expanded0 or expanded1 or expanded2:
                ind0_ = g.op.Reshape(ind0, np.array([-1, 1, 1], dtype=np.int64), name=name)
                ind1_ = g.op.Reshape(ind1, np.array([1, -1, 1], dtype=np.int64), name=name)
                ind2_ = g.op.Reshape(ind2, np.array([1, 1, -1], dtype=np.int64), name=name)

                if (
                    (
                        isinstance(ind0, np.ndarray)
                        or (g.has_shape(ind0) and is_static_shape(g.get_shape(ind0)))
                    )
                    and (
                        isinstance(ind1, np.ndarray)
                        or (g.has_shape(ind1) and is_static_shape(g.get_shape(ind1)))
                    )
                    and (
                        isinstance(ind1, np.ndarray)
                        or (g.has_shape(ind2) and is_static_shape(g.get_shape(ind2)))
                    )
                ):
                    sh0 = ind0.shape if isinstance(ind0, np.ndarray) else g.get_shape(ind0)
                    sh1 = ind1.shape if isinstance(ind1, np.ndarray) else g.get_shape(ind1)
                    sh2 = ind2.shape if isinstance(ind2, np.ndarray) else g.get_shape(ind2)
                    new_shape = np.hstack([sh0, sh1, sh2]).astype(np.int64)
                else:
                    new_shape = g.op.Concat(
                        g.op.Shape(ind0, name=name),
                        g.op.Shape(ind1, name=name),
                        g.op.Shape(ind2, name=name),
                        axis=0,
                        name=name,
                    )
                expanded = g.op.Expand(values, new_shape, name=name)
                indices_3d = g.op.Add(
                    g.op.Add(
                        g.op.Mul(ind0_, stride_1, name=name),
                        g.op.Mul(ind1_, stride_2, name=name),
                        name=name,
                    ),
                    ind2_,
                    name=name,
                )
            else:
                expanded = values
                indices_3d = g.op.Add(
                    g.op.Add(
                        g.op.Mul(ind0, stride_1, name=name),
                        g.op.Mul(ind1, stride_2, name=name),
                        name=name,
                    ),
                    ind2,
                    name=name,
                )

            indices_1d = g.op.GatherElements(
                arange_1d,
                g.op.Reshape(indices_3d, np.array([-1], dtype=np.int64), name=name),
                name=name,
            )

            expanded = g.op.Reshape(expanded, np.array([-1], dtype=np.int64), name=name)

            flat_x = g.op.Reshape(x, np.array([-1], dtype=np.int64), name=name)
            if accumulate:
                flat_up_x = g.op.ScatterElements(
                    flat_x, indices_1d, expanded, name=name, reduction="add"
                )
            else:
                flat_up_x = g.op.ScatterElements(flat_x, indices_1d, expanded, name=name)

            g.set_type(flat_up_x, g.get_type(x))
            res = g.op.Reshape(flat_up_x, shape_x, name=name, outputs=outputs)
            if not sts:
                set_type_shape_unary_op(g, res, x)
            return res

        def _s(ind):
            if ind is None:
                return "-"
            if g.has_shape(ind):
                s = str(g.get_shape(ind)).replace(" ", "")
                return f"{s}:{g.get_type(ind)}"
            return str(ind)

        raise AssertionError(
            f"No implementation for index_put when indices={indices}, "
            f"rk(x)={g.get_rank(x)}, rk(values)={g.get_rank(values)} "
            f"ind0={_s(ind0)}, ind1={_s(ind1)}, ind2={_s(ind2)}, "
            f"INT64={TensorProto.INT64}, INT32={TensorProto.INT32}"
            f"{g.get_debug_msg()}"
        )

    raise AssertionError(
        f"No implementation for index_put when indices={indices}{g.get_debug_msg()}"
    )


def aten_instance_norm(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    weight: Optional[T] = None,
    bias: Optional[T] = None,
    running_mean: Optional[T] = None,
    running_var: Optional[T] = None,
    use_input_stats: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-05,
    cudnn_enabled: bool = False,
    name: str = "instance_norm",
) -> T:
    """instance_norm"""
    assert cudnn_enabled, "Not implemented when cudnn_enabled is True"

    itype = g.get_type(x)
    dtype = tensor_dtype_to_np_dtype(itype)

    if weight is None and g.has_shape(x):
        sh = g.get_shape(x)
        if is_static_dimension(sh[1]):
            weight = np.ones(sh[1:2], dtype=dtype)
    if weight is None:
        sh = g.op.Shape(x, start=1, end=2, name=name)
        weight = g.op.Expand(np.array([1], dtype=dtype), sh, name=name)

    if bias is None and g.has_shape(x):
        sh = g.get_shape(x)
        if is_static_dimension(sh[1]):
            bias = np.zeros(sh[1:2], dtype=dtype)
    if bias is None:
        sh = g.op.Shape(x, start=1, end=2, name=name)
        bias = g.op.Expand(np.array([1], dtype=dtype), sh, name=name)

    if use_input_stats:
        # If `use_input_stats` is set to True,
        # ignore 'running_mean' and 'running_var' and
        # compute using input statistics.
        # Otherwise, compute using the running statistics.
        return g.op.InstanceNormalization(
            x, weight, bias, epsilon=eps, name=name, outputs=outputs
        )

    assert (
        running_mean is not None and running_var is not None
    ), "running_mean and running_var must be provided when use_input_stats is False"

    batch_size = None
    if g.has_shape(x):
        sh = g.get_shape(x)
        if is_static_dimension(sh[0]):
            batch_size = np.array([sh[0]], dtype=np.int64)

    if batch_size is None:
        batch_size = g.op.Shape(input, start=0, end=1, name=name)

    bias = g.op.Tile(bias, batch_size, name=name)
    weight = g.op.Tile(weight, batch_size, name=name)
    n_running_mean = g.op.Tile(running_mean, batch_size, name=name)
    n_running_var = g.op.Tile(running_var, batch_size, name=name)

    bn_input = None
    shape_x = None
    if g.has_shape(x):
        sh = g.get_shape(x)
        if is_static_shape(sh[2:]):
            cst = np.array([1, -1, *sh[2:]], dtype=np.int64)
            bn_input = g.op.Reshape(x, cst, name=name)
            shape_x = np.array(sh, dtype=np.int64)
    if bn_input is None:
        sh2 = g.op.Concat([1, -1], g.op.Shape(input, start=2, name=name), axis=0, name=name)
        bn_input = g.op.Reshape(x, sh2, name=name)
        shape_x = g.op.Shape(x, name=name)

    assert (
        len(outputs) == 1
    ), f"Only one output can be requested but outputs={outputs}{g.get_debug_msg()}"
    bi_name = g.unique_name(f"{outputs[0]}_bn")
    g.make_node(
        "BatchNormalization",
        [bn_input, weight, bias, n_running_mean, n_running_var],
        [bi_name],
        epsilon=eps,
        momentum=1.0 - momentum,
        training_mode=0,
        name=name,
    )

    return g.op.Reshape(bi_name, shape_x, name=name, outputs=outputs)


def aten_isinf(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    name: str = "isinf",
) -> T:
    "isinf"
    if g.main_opset >= 20 or g.get_type(x) in {TensorProto.FLOAT, TensorProto.DOUBLE}:
        res = g.op.IsInf(x, outputs=outputs, name=name)
    else:
        # opset < 20, IsInf only supports float32, float64.
        res = g.op.IsInf(
            g.op.Cast(x, to=TensorProto.FLOAT, name=name), outputs=outputs, name=name
        )
    if not sts:
        set_type_shape_unary_op(g, res, x, itype=TensorProto.BOOL)
    return res


def aten_isnan(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    name: str = "isnan",
) -> T:
    "isnan"
    if g.main_opset >= 20 or g.get_type(x) in {TensorProto.FLOAT, TensorProto.DOUBLE}:
        res = g.op.IsNaN(x, outputs=outputs, name=name)
    else:
        # opset < 20, IsInf only supports float32, float64.
        res = g.op.IsNaN(
            g.op.Cast(x, to=TensorProto.FLOAT, name=name), outputs=outputs, name=name
        )
    if not sts:
        set_type_shape_unary_op(g, res, x, itype=TensorProto.BOOL)
    return res


def aten__unsafe_index_put(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    self: T,
    indices: List[T],
    values: T,
    accumulate: bool = False,
) -> T:
    "[...,:, ...]"
    return aten_index_put(
        g,
        sts,
        outputs,
        self,
        indices,
        values,
        accumulate,
        name="aten__unsafe_index_put",
    )


def aten_index_select(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    dim: int,
    index: T,
    name: str = "index_select",
) -> T:
    "[...,:, ...]"
    res = g.op.Gather(x, index, axis=dim, outputs=outputs, name=name)
    if not sts:
        g.set_type(res, g.get_type(x))
        if g.has_shape(x) and g.has_shape(index):
            shape = list(g.get_shape(x))
            index_shape = g.get_shape(index)
            shape[dim] = index_shape[0]
            g.set_shape(res, tuple(shape))
        else:
            g.set_rank(res, g.get_rank(x))
    return res


def aten_layer_norm(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    normalized_shape: Sequence[int],
    weight: Optional[T] = None,
    bias: Optional[T] = None,
    eps: float = 1e-05,
    cudnn_enable: bool = False,  # not used
    name="layer_norm",
) -> T:
    "layer_norm"
    axes = np.array([-i for i in range(len(normalized_shape), 0, -1)], dtype=np.int64)
    itype = g.get_type(x)
    dtype = tensor_dtype_to_np_dtype(itype)

    two_cst = np.array([2.0], dtype=dtype)
    eps_cst = np.array([eps], dtype=dtype)

    mean = g.op.ReduceMeanAnyOpset(x, axes, name=name, keepdims=1)
    numerator = g.op.Sub(x, mean, name=name)
    variance = g.op.ReduceMeanAnyOpset(
        g.op.Pow(numerator, two_cst, name=name), axes, keepdims=1, name=name
    )
    denominator = g.op.Sqrt(g.op.Add(variance, eps_cst, name=name), name=name)
    normalized = g.op.Div(numerator, denominator, name=name)

    if weight is not None:
        normalized = g.op.Mul(normalized, weight, name=name)
    if bias is not None:
        normalized = g.op.Add(normalized, bias, name=name)

    # rdenominator = g.op.Reciprocal(denominator)
    return normalized  # , mean, rdenominator


def aten_le(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    y: T,
    name: str = "le",
) -> T:
    "less or equal"
    x, y = prepare_inputs_homogeneous_operator(g, x, y, name=name)
    res = g.op.LessOrEqual(x, y, outputs=outputs, name=name)
    if not sts:
        set_type_shape_binary_op(g, outputs[0], x, y, cmp_op=True)
    return res


def aten_le_Scalar(
    g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T, y: T
) -> T:
    "less or equal"
    return aten_le(g, sts, outputs, x, y, name="le_Scalar")


def aten_le_Tensor(
    g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T, y: T
) -> T:
    "less or equal"
    return aten_le(g, sts, outputs, x, y, name="le_Tensor")


def aten_leaky_relu(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    a: T,
    negative_slope: float = 0.01,
    inplace: bool = False,
    name: str = "leaky_relu",
) -> T:
    "leaky relu"
    assert not inplace, f"inplace not implemented for leaky_relu{g.get_debug_msg()}"

    dtype = tensor_dtype_to_np_dtype(g.get_type(a))
    slope = np.array([negative_slope], dtype=dtype)
    res = g.op.Where(
        g.op.Greater(a, np.array([0], dtype=dtype), name=name),
        a,
        g.op.Mul(a, slope, name=name),
        outputs=outputs,
        name=name,
    )
    if not sts:
        set_type_shape_unary_op(g, res, a)
    return res


def aten_leaky_relu_backward(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    grad_output: T,
    x: T,
    negative_slope: float,
    self_is_result: bool,
    name="leaky_relu_backward",
) -> T:
    "leaky relu"
    dtype = tensor_dtype_to_np_dtype(g.get_type(grad_output))
    slope = np.array([negative_slope], dtype=dtype)
    res = g.op.Where(
        g.op.Greater(x, np.array([0], dtype=dtype), name=name),
        grad_output,
        g.op.Mul(grad_output, slope, name=name),
        outputs=outputs,
        name=name,
    )
    if not sts:
        set_type_shape_unary_op(g, res, x)
    return res


def aten_lift_fresh_copy(
    g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T
) -> T:
    "identity"
    return g.op.Identity(x, outputs=outputs, name="lift_fresh_copy")


# def _aten__linalg_det(
#     g: GraphBuilder,
#     sts: Optional[Dict[str, Any]],
#     outputs: List[str],
#     x: T,
#     name: str = "_linalg_det",
# ) -> T:
#     """_linalg_det expected 3 outputs, det, LU decomposition, pivots"""
#     res = g.op.Det(x, outputs=outputs, name=name)
#     if not sts:
#         g.set_type(res, g.get_type(x))
#         if g.has_shape(x):
#             shape = g.get_shape(x)
#             g.set_shape(res, shape[:-2])
#         elif g.has_rank(x):
#             g.set_rank(res, g.get_rank(x) - 2)
#     return res


def aten_linalg_vector_norm(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    ord: float = 2.0,
    dim: Optional[int] = None,
    keepdim: bool = False,
    dtype: Optional[int] = None,
    name: str = "linagl_vector_norm",
) -> T:
    """reduce *"""
    assert (
        dtype is None
    ), f"aten_linalg_vector_norm not implementd when dtype={dtype}{g.get_debug_msg()}"
    assert (
        g.has_rank(x) and g.get_rank(x) > 0
    ), f"Rank of {x!r} is unknown or null{g.get_debug_msg()}"
    assert isinstance(dim, list) and all_int(
        dim
    ), f"Unsupported value for dim={dim} (type is {type(dim)}){g.get_debug_msg()}"
    assert isinstance(ord, (float, int)), (
        f"aten_linalg_vector_norm not implemented for ord={ord} "
        f"(type is {type(ord)}{g.get_debug_msg()}"
    )

    adim = np.array(dim, dtype=np.int64)
    kd = 1 if keepdim else 0

    # ord = op.Cast(ord, to=FLOAT.dtype)  # Must be FLOAT, due to op.IsInf() needs FLOAT

    if np.isinf(ord) and ord > 0:
        res = g.op.ReduceMax(
            g.op.Abs(x, name=name), adim, keepdims=kd, name=name, outputs=outputs
        )
    elif np.isinf(ord) and ord < 0:
        res = g.op.ReduceMin(
            g.op.Abs(x, name=name), adim, keepdims=kd, name=name, outputs=outputs
        )
    elif ord == 0.0:
        raise AssertionError(
            f"aten_linalg_vector_norm not yet implemented for ord={ord}{g.get_debug_msg()}"
        )
        # self_bool = g.op.Cast(self, to=TensorProto.BOOL)
        # self_0_1 = op.CastLike(self_bool, self)
        # result = op.ReduceSum(self_0_1, dim, keepdims=keepdim)
    elif ord == 1.0:
        res = g.op.ReduceL1(x, adim, keepdims=kd, name=name, outputs=outputs)
    elif ord == 2.0:
        res = g.op.ReduceL2(x, adim, keepdims=kd, name=name, outputs=outputs)
    else:
        raise AssertionError(
            f"aten_linalg_vector_norm not yet implemented for ord={ord}{g.get_debug_msg()}"
        )
        # ord_float = op.CastLike(ord, self)
        # self_pow = op.Pow(self, ord_float)
        # result = op.Pow(op.ReduceSum(self_pow, dim,
        # keepdims=keepdim), op.Div(1.0, ord_float))

    return res


def aten_linear(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    weight: T,
    bias: T = None,
) -> T:
    "linear"
    weight_transposed = g.op.Transpose(weight, perm=[1, 0], name="linear")
    if bias:
        res = g.op.MatMul(x, weight_transposed)
        set_type_shape_matmul(g, res, x, weight_transposed)
        res = g.op.Add(res, bias, outputs=outputs)
    else:
        res = g.op.MatMul(x, weight_transposed, outputs=outputs, name="linear")
    if not sts:
        g.set_type(res, g.get_type(x))
        if g.has_shape(x) and g.has_shape(weight):
            shape_x = g.get_shape(x)
            shape_w = g.get_shape(weight)
            new_shape = (shape_x[0], shape_w[0])
            g.set_shape(res, new_shape)
        elif g.has_rank(x) and g.has_rank(weight):
            rkx = g.get_rank(x)
            rkw = g.get_rank(weight)
            if rkw == rkx:
                g.set_rank(res, rkw)
    return res


def aten_log(g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T) -> T:
    """log"""
    res = g.op.Log(x)
    if not sts:
        set_type_shape_unary_op(g, res, x)
    return res


def aten__log_softmax(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    dim: int = -1,
    unnamed: bool = False,
    dtype: Optional["torch.dtype"] = None,  # noqa: F821
    name: str = "_log_softmax",
) -> T:
    "logsoftmax"
    assert not unnamed, "Not implemented when the third parameter is False"
    if dtype is not None:
        itype = torch_dtype_to_onnx_dtype(dtype)
        xc = g.op.Cast(x, to=itype, name=name)
    else:
        itype = None
        xc = x
    res = g.op.LogSoftmax(xc, axis=dim, outputs=outputs, name=name)
    if not sts:
        set_type_shape_unary_op(g, res, xc, itype=itype)
    return res


def aten_log_softmax_int(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    dim: int = -1,
    unnamed: bool = False,
    dtype: Optional["torch.dtype"] = None,  # noqa: F821
    name: str = "log_softmax_int",
) -> T:
    "logsoftmax"
    return aten__log_softmax(
        g, sts, outputs, x, dim=dim, unnamed=unnamed, dtype=dtype, name=name
    )


def aten__log_softmax_backward_data(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    grad_output: T,
    output: T,
    dim: int,
    input_dtype: Optional["torch.dtype"] = None,  # noqa: F821
    name: str = "_log_softmax_backward_data",
):
    "logsoftmax backward"
    itype = None
    grad_outputc = grad_output

    vexp = g.op.Exp(output, name=name)
    red = g.op.ReduceSum(
        grad_outputc,
        np.array([dim], dtype=np.int64),
        keepdims=True,
        name=name,
    )
    vmul = g.op.Mul(vexp, red, name=name)

    if input_dtype is not None:
        itype = torch_dtype_to_onnx_dtype(input_dtype)
        sub = g.op.Sub(grad_outputc, vmul, name=name)
        res = g.op.Cast(sub, to=itype, name=name, outputs=outputs)
    else:
        res = g.op.Sub(grad_outputc, vmul, outputs=outputs, name=name)

    set_type_shape_unary_op(g, vexp, output)
    set_type_shape_unary_op(g, vmul, vexp)
    set_type_shape_reduce_op(g, red, grad_outputc, keepdim=1, axes=(dim,))
    if not sts:
        set_type_shape_unary_op(g, res, grad_outputc, itype=itype)
    return res


def aten_lt(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    y: T,
    name="lt",
) -> T:
    "less"
    x, y = prepare_inputs_homogeneous_operator(g, x, y)
    res = g.op.Less(x, y, outputs=outputs, name=name)
    if not sts:
        set_type_shape_binary_op(g, outputs[0], x, y, cmp_op=True)
    return res


def aten_lt_Scalar(
    g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T, y: T
) -> T:
    "less"
    return aten_lt(g, sts, outputs, x, y, name="lt_Scalar")


def aten_lt_Tensor(
    g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T, y: T
) -> T:
    "less"
    return aten_lt(g, sts, outputs, x, y, name="lt_Tensor")


def aten_matmul(
    g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T, y: T
) -> T:
    "matmul"
    res = g.op.MatMul(x, y, outputs=outputs)
    if not sts:
        set_type_shape_binary_op(g, outputs[0], x, y)
    return res


def aten_masked_fill_Scalar(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    mask: T,
    value: T,
    name="masked_fill_Scalar",
) -> T:
    "masked"
    dt = g.get_type(mask)
    if dt != TensorProto.BOOL:
        cmask = g.op.Cast(mask, to=TensorProto.BOOL, name=name)
    else:
        cmask = mask
    dtx = g.get_type(x)
    if isinstance(value, T):
        # A tensor then
        if g.get_type(value) != g.get_type(x):
            # We need to cast the constant into the same type as X
            # assuming x has the expected type.
            avalue = g.op.Cast(value, to=g.get_type(x), name=name)
        else:
            avalue = value
    else:
        avalue = np.array([value], dtype=tensor_dtype_to_np_dtype(dtx))
    res = g.op.Where(cmask, avalue, x, name=name)
    if not sts:
        g.set_type(res, dtx)
        if g.has_shape(mask):
            g.set_shape(res, g.get_shape(mask))
        else:
            g.set_rank(res, g.get_rank(mask))
    return res


def aten_masked_fill_Tensor(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    mask: T,
    value,
    name="masked_fill_Tensor",
) -> T:
    "masked"
    return aten_masked_fill_Scalar(g, sts, outputs, x, mask, value, name=name)


def aten_max(
    g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T, name: str = "max"
) -> T:
    """min"""

    res = g.op.ReduceMax(x, keepdims=0, name=name, outputs=outputs)
    if not sts:
        g.set_type(res, g.get_type(x))
        g.get_shape(res, tuple())
    return res


def aten_maximum(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    y: T,
    name: str = "max",
) -> T:
    """maximum"""

    res = g.op.Max(x, y, name=name, outputs=outputs)
    if not sts:
        set_type_shape_binary_op(g, res, x, y)
    return res


def aten_max_dim(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    dim: int,
    keepdim: bool = False,
    name: str = "max_dim",
) -> T:
    """maximum"""
    axes = np.array([dim], dtype=np.int64)
    res = g.op.ReduceMax(x, axes, name=name, outputs=outputs[:1], keepdims=1 if keepdim else 0)
    if not sts:
        set_type_shape_unary_op(g, res, x)
    if len(outputs) == 1:
        return res

    indices = g.op.ArgMax(
        x, axis=dim, keepdims=1 if keepdim else 0, name=name, outputs=outputs[1:]
    )
    if not sts:
        g.get_type(indices, TensorProto.INT64)
    return res, indices


def aten_max_other(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    y: T,
    name: str = "max_other",
) -> T:
    """maximum"""

    if g.has_type(x) and g.has_type(y) and g.get_type(x) == g.get_type(y):
        res = g.op.Max(x, y, name=name, outputs=outputs)
        if not sts:
            set_type_shape_binary_op(g, res, x, y)
        return res

    if g.has_type(x) and g.has_type(y):
        # types are different
        assert g.get_type_known(outputs[0]), (
            f"Type mismatch for {x!r} ({g.get_type(x)}) and {y!r} ({g.get_type(y)}), "
            f"output {outputs[0]!r} has no type{g.get_debug_msg()}"
        )
        itype = g.get_type_known(outputs[0])
        if itype == g.get_type(x):
            res = g.op.Max(x, g.op.Cast(y, to=itype, name=name), name=name, outputs=outputs)
        elif itype == g.get_type(y):
            res = g.op.Max(g.op.Cast(x, to=itype, name=name), y, name=name, outputs=outputs)
        else:
            res = g.op.Max(
                g.op.Cast(x, to=itype, name=name),
                g.op.Cast(y, to=itype, name=name),
                name=name,
                outputs=outputs,
            )
        if not sts:
            set_type_shape_binary_op(g, res, x, y, itype=itype)
        return res

    raise AssertionError(
        f"Unable to guess the output type for {x!r} (type={g.get_type(x)}) "
        f"and {y!r} (type={g.get_type(y)})"
    )


def _aten_max_pool_onnx(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    kernel_shape: Sequence[int],
    strides: Sequence[int],
    pads: Sequence[int],
    dilations: Sequence[int],
    ceil_mode: bool,
    unbatched_rank: int,
    name: str,
) -> T:
    "maxpool"
    # self_rank_is_unbatched_rank = Rank(self) == unbatched_rank
    # if self_rank_is_unbatched_rank:  # C,H,W -> N,C,H,W and N=1
    #     self = op.UnsqueezeAnyOpset(self, op.Constant(value_ints=[0]))

    pool_result, _ = g.op.MaxPool(
        x,
        ceil_mode=ceil_mode,
        dilations=dilations,
        kernel_shape=kernel_shape,
        pads=pads,
        strides=strides,
        name=name,
    )

    # if self_rank_is_unbatched_rank:
    #    pool_result = op.SqueezeAnyOpset(pool_result, op.Constant(value_ints=[0]))

    return pool_result


def aten_max_pool1d(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    kernel_size: Sequence[int],
    stride: Sequence[int] = (),
    padding: Sequence[int] = (0, 0),
    dilation: Sequence[int] = (1, 1),
    ceil_mode: bool = False,
    name: str = "max_pool1d",
) -> T:
    """max_pool1d"""

    expand_size = 1

    kernel_shape, strides, pads, dilations = _adjust_attributes_of_max_pool(
        expand_size, kernel_size, stride, padding, dilation
    )

    return _aten_max_pool_onnx(
        g,
        sts,
        outputs,
        x,
        kernel_shape,
        strides,
        pads,
        dilations,
        ceil_mode,
        2,
        name=name,
    )


def aten_max_pool2d(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    kernel_size: Sequence[int],
    stride: Sequence[int] = (),
    padding: Sequence[int] = (0, 0),
    dilation: Sequence[int] = (1, 1),
    ceil_mode: bool = False,
    name: str = "max_pool2d",
) -> T:
    "max_pool2d"
    expand_size = 2

    kernel_shape, strides, pads, dilations = _adjust_attributes_of_max_pool(
        expand_size, kernel_size, stride, padding, dilation
    )

    return _aten_max_pool_onnx(
        g,
        sts,
        outputs,
        x,
        kernel_shape,
        strides,
        pads,
        dilations,
        ceil_mode,
        3,
        name=name,
    )


def aten_max_pool3d(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    kernel_size: Sequence[int],
    stride: Sequence[int] = (),
    padding: Sequence[int] = (0, 0, 0),
    dilation: Sequence[int] = (1, 1, 1),
    ceil_mode: bool = False,
    name: str = "max_pool3d",
) -> T:
    """max_pool3d"""

    expand_size = 3

    kernel_shape, strides, pads, dilations = _adjust_attributes_of_max_pool(
        expand_size, kernel_size, stride, padding, dilation
    )

    return _aten_max_pool_onnx(
        g,
        sts,
        outputs,
        x,
        kernel_shape,
        strides,
        pads,
        dilations,
        ceil_mode,
        4,
        name=name,
    )


def _aten_max_pool_with_indices_onnx(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    kernel_size: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    ceil_mode: bool,
    unbatched_rank: int,
    n_dims_one: Sequence[int],
    n_dims_zero: Sequence[int],
    n_dims_axes: Sequence[int],
    name: str,
) -> Tuple[T, T]:
    "maxpool"
    if isinstance(ceil_mode, str):
        raise TypeError(f"Unexpected ceil_mode={ceil_mode}")
    is_unbatched_rank = g.rank(x) == unbatched_rank
    if is_unbatched_rank:
        x = g.op.UnsqueezeAnyOpset(x, axes=0)

    pool_result, indices = g.op.MaxPool(
        x,
        ceil_mode=ceil_mode,
        dilations=dilation,
        kernel_shape=kernel_size,
        pads=padding,
        strides=stride,
        name=name,
    )
    _, flatten_indices = g.op.MaxPool(
        x, dilations=dilation, kernel_shape=n_dims_one, strides=n_dims_one
    )

    ends = g.make_initializer("", np.array(n_dims_one, dtype=np.int64))
    starts = g.make_initializer("", np.array(n_dims_zero, dtype=np.int64))
    axes = g.make_initializer("", np.array(n_dims_axes, dtype=np.int64))

    delta = g.op.Slice(flatten_indices, starts, ends, axes, name=name)
    indices = g.op.Sub(indices, delta, name=name)

    if is_unbatched_rank:
        pool_result = g.op.SqueezeAnyOpset(
            pool_result, np.array([0], dtype=np.int64), name=name
        )
        indices = g.op.SqueezeAnyOpset(indices, np.array([0], dtype=np.int64), name=name)

    g.set_type(delta, g.get_type(flatten_indices))

    if outputs:
        if not isinstance(outputs, (tuple, list)):
            raise TypeError(
                f"Multiple outputs are expeted but type(outputs) is {type(outputs)}."
            )
        if len(outputs) != 2:
            raise ValueError(f"Multiple outputs are expeted but outputs is {outputs}.")
        return (
            g.op.Identity(pool_result, outputs=outputs[0], name=name),
            g.op.Identity(indices, outputs=outputs[1], name=name),
        )
    return pool_result, indices


def aten_max_pool2d_with_indices(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    kernel_size: Sequence[int],
    stride: Sequence[int] = (),
    padding: Sequence[int] = (0, 0),
    dilation: Sequence[int] = (1, 1),
    ceil_mode: bool = False,
) -> Tuple[T, T]:
    "maxpool"
    assert isinstance(padding, (tuple, list))
    assert isinstance(ceil_mode, (bool, int))
    expand_size = 2

    kernel_shape, strides, pads, dilations = _adjust_attributes_of_max_pool(
        expand_size, kernel_size, stride, padding, dilation
    )

    return _aten_max_pool_with_indices_onnx(
        g,
        sts,
        outputs,
        x,
        kernel_shape,
        strides,
        pads,
        dilations,
        ceil_mode,
        3,
        ([1] * expand_size),
        ([0] * expand_size),
        ([2 + i for i in range(expand_size)]),
        name="max_pool2d_with_indices",
    )


def aten_mean_dim(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    dim: Optional[Union[int, List[int]]] = None,
    keepdim: bool = False,
    dtype: Optional["torch.dtype"] = None,  # noqa: F821
) -> T:
    "reducemean"
    if dtype is not None:
        itype = torch_dtype_to_onnx_dtype(dtype)
        xc = g.op.Cast(x, to=itype)
    else:
        xc = x
    if dim is None:
        result = g.op.ReduceMeanAnyOpset(
            xc, keepdims=keepdim, outputs=outputs, name="mean_dim"
        )
    else:
        if isinstance(dim, int):
            adim = np.array([dim], dtype=np.int64)
        else:
            adim = np.array(dim, dtype=np.int64)
        result = g.op.ReduceMeanAnyOpset(
            xc, adim, keepdims=keepdim, outputs=outputs, name="mean_dim"
        )
    if not sts:
        set_type_shape_reduce_op(g, outputs[0], x, keepdim=keepdim)
    return result


def aten_mean(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    dtype: Optional["torch.dtype"] = None,  # noqa: F821
    name: str = "mean",
) -> T:
    """mean"""
    if dtype is not None:
        itype = torch_dtype_to_onnx_dtype(dtype)
        x = g.op.Cast(x, to=itype, name=name)
    res = g.op.ReduceMean(x, keepdims=0, name=name, outputs=outputs)
    if not sts:
        g.set_type(res, g.get_type(x) if dtype is None else itype)
        g.get_shape(res, tuple())
    return res


def aten_min(
    g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T, name: str = "min"
) -> T:
    """min"""

    res = g.op.ReduceMin(x, keepdims=0, name=name, outputs=outputs)
    if not sts:
        g.set_type(res, g.get_type(x))
        g.get_shape(res, tuple())
    return res


def aten_minimum(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    y: T,
    name: str = "min",
) -> T:
    """minimum"""

    res = g.op.Min(x, y, name=name, outputs=outputs)
    if not sts:
        set_type_shape_binary_op(g, res, x, y)
    return res


def aten_min_other(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    y: T,
    name: str = "min_other",
) -> T:
    """minimum"""

    res = g.op.Min(x, y, name=name, outputs=outputs)
    if not sts:
        set_type_shape_binary_op(g, res, x, y)
    return res


def aten_mm(
    g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T, y: T
) -> T:
    "matmul"
    res = g.op.MatMul(x, y, outputs=outputs, name="mm")
    if not sts:
        set_type_shape_matmul(g, res, x, y)
    return res


def aten_mod(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    y: T,
    name="mod",
) -> T:
    "mod"
    res, x, y = prepare_inputs_homogeneous_operator(
        g,
        x,
        y,
        f=g.op.Mod,
        name=name,
        outputs=outputs,
        sts=sts,
    )
    if not sts:
        set_type_shape_binary_op(g, res, x, y)
    return res


def aten_mul(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    y: T,
    name="mul",
) -> T:
    "mul"
    if (
        isinstance(x, str)
        and isinstance(y, str)
        and g.get_type(x) == TensorProto.BOOL
        and g.get_type(y) == TensorProto.BOOL
    ):
        res = g.op.And(x, y, name=f"{name}_and", outputs=outputs)
    else:
        res, x, y = prepare_inputs_homogeneous_operator(
            g,
            x,
            y,
            f=g.op.Mul,
            name=name,
            outputs=outputs,
            sts=sts,
        )
    if not sts:
        set_type_shape_binary_op(g, res, x, y)
    return res


def aten_imul(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    y: T,
    name="imul",
) -> T:
    "imul"
    return aten_mul(g, sts, outputs, x, g.op.CastLike(y, x, name=name), name=name)


def aten_mul_Scalar(
    g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T, y: T
) -> T:
    "mul"
    return aten_mul(g, sts, outputs, x, y, name="mul_Scalar")


def aten_mul_Tensor(
    g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T, y: T
) -> T:
    "mul"
    return aten_mul(g, sts, outputs, x, y, name="mul_Tensor")


def aten_multiply_Tensor(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    y: T,
    name="multiply_Tensor",
) -> T:
    "mul"
    return aten_mul(g, sts, outputs, x, y, name=name)


def aten_native_dropout(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    p: float,
    train: bool = False,
    name: str = "native_dropout",
):
    "dropout"
    if not train:
        assert len(outputs) == 1, f"train is False and outputs is {outputs}{g.get_debug_msg()}"
        return g.op.Identity(x, outputs=outputs, name=name)
    assert len(outputs) == 2, f"train is True and outputs is {outputs}{g.get_debug_msg()}"
    tp = g.make_initializer("", np.array(p, dtype=tensor_dtype_to_np_dtype(g.get_type(x))))
    tt = g.make_initializer("", np.array(train, dtype=np.bool_))
    g.make_node("Dropout", [x, tp, tt], outputs, name=name)
    if not sts:
        set_type_shape_unary_op(g, outputs[0], x)
        g.set_type(outputs[1], TensorProto.BOOL)
        if g.has_shape(x):
            g.set_shape(outputs[1], g.get_shape(x))
        else:
            g.set_rank(outputs[1], g.get_rank(x))
    return tuple(outputs)


def aten_native_layer_norm(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    normalized_shape: Tuple[int, ...],  # int64
    weight: Optional[T] = None,
    bias: Optional[T] = None,
    eps: float = 1e-05,
    name: str = "aten_native_layer_norm",
) -> Tuple[T, T, T]:
    "native_layer_norm"
    assert isinstance(normalized_shape, list) and all_int(normalized_shape), (
        f"aten_native_layer_norm not implemented for normalized_shape={normalized_shape}"
        f"{g.get_debug_msg()}"
    )
    start_axis = -len(normalized_shape)

    if weight is None:
        dtype = tensor_dtype_to_np_dtype(g.get_type(x))
        weight = g.op.ConstantOfShape(
            g.op.Shape(x, start=start_axis, name=name),
            value=from_array([1], dtype=dtype),
            name=name,
        )

    g.make_node(
        "LayerNormalization",
        [x, weight, bias or ""],
        outputs,
        axis=start_axis,
        epsilon=eps,
        name=name,
    )
    if not sts:
        g.set_type(outputs[0], g.get_type(x))
        g.get_type(outputs[1], TensorProto.FLOAT)
        g.get_type(outputs[2], TensorProto.FLOAT)
        if g.has_shape(x):
            g.set_shape(outputs[0], g.get_shape(x))
        else:
            g.set_rank(outputs[0], g.get_rank(x))
        g.set_shape(outputs[1], (1,))
        g.set_shape(outputs[2], (1,))

    return tuple(outputs)


def aten__native_batch_norm_legit_no_training(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    weight: Optional[T] = None,
    bias: Optional[T] = None,
    running_mean: Optional[T] = None,
    running_var: Optional[T] = None,
    momentum: float = 0.9,
    eps: float = 1e-05,
    name: str = "_native_batch_norm_legit_no_training",
) -> Tuple[T, T, T]:
    """batch normalization = aten__native_batch_norm with training=False"""
    return aten__native_batch_norm(
        g,
        sts,
        outputs,
        x,
        weight=weight,
        bias=bias,
        running_mean=running_mean,
        running_var=running_var,
        training=False,
        momentum=momentum,
        eps=eps,
        name=name,
        empty_mean_std=True,
    )


def aten__native_batch_norm_legit_no_stats(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    weight: Optional[T] = None,
    bias: Optional[T] = None,
    training: bool = False,
    momentum: float = 0.9,
    eps: float = 1e-05,
    name: str = "_native_batch_norm_legit_no_stats",
) -> Tuple[T, T, T]:
    """batch normalization = aten__native_batch_norm"""
    return aten__native_batch_norm(
        g,
        sts,
        outputs,
        x,
        weight=weight,
        bias=bias,
        training=training,
        momentum=momentum,
        eps=eps,
        name=name,
    )


def aten__native_batch_norm(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    weight: Optional[T] = None,
    bias: Optional[T] = None,
    running_mean: Optional[T] = None,
    running_var: Optional[T] = None,
    training: bool = False,
    momentum: float = 0.9,
    eps: float = 1e-05,
    name: str = "_native_batch_norm",
    empty_mean_std: bool = False,
) -> Tuple[T, T, T]:
    """batch normalization"""
    assert g.has_rank(x), f"{x!r} must have a known rank{g.get_debug_msg()}"
    assert g.has_type(x), f"{x!r} must have a known type{g.get_debug_msg()}"
    dtype = tensor_dtype_to_np_dtype(g.get_type(x))

    if weight is None:
        # default is 1
        cst = np.array([1], dtype=dtype)
        weight = g.op.Expand(cst, g.op.Shape(x, start=1, end=2, name=name), name=name)

    if bias is None:
        # default is 0
        cst = np.array([0], dtype=dtype)
        bias = g.op.Expand(cst, g.op.Shape(x, start=1, end=2, name=name), name=name)

    axes_py = list(range(g.get_rank(x)))
    axes_py.pop(1)
    axes_np = np.array(axes_py, dtype=np.int64)

    if running_mean is None:
        # default is mean
        running_mean = g.op.ReduceMeanAnyOpset(x, axes_np, name=name, keepdims=0)

    if running_var is None:
        # default is var
        mean = g.op.ReduceMeanAnyOpset(x, axes_np, name=name, keepdims=1)
        input_sub_mean = g.op.Sub(x, mean, name=name)
        sqr_input_sub_mean = g.op.Pow(input_sub_mean, np.array([2], dtype=np.int64), name=name)
        running_var = g.op.ReduceMean(sqr_input_sub_mean, axes_np, name=name, keepdims=0)

    assert len(outputs) == 3, (
        f"Unexpected number of outputs {outputs!r}, "
        f"training_mode={training}{g.get_debug_msg()}"
    )
    outs = (
        [outputs[0], g.unique_name(f"{name}_mean"), g.unique_name(f"{name}_var")]
        if training
        else outputs[:1]
    )
    batch_out = g.op.BatchNormalization(
        x,
        weight,
        bias,
        running_mean,
        running_var,
        epsilon=eps,
        momentum=1 - momentum,
        # training_mode=1 is not supported by onnxruntime
        # Training mode does not support BN opset 14 (or higher) yet.
        # An optimizer should probably take care of that.
        training_mode=0 if not training else 1,
        name=name,
        outputs=outs,
    )
    if training:
        norm, bmean, bvar = batch_out
        g.set_type(bmean, TensorProto.FLOAT)
        g.set_type(bvar, TensorProto.FLOAT)
    else:
        assert isinstance(
            batch_out, str
        ), f"Unexpected output for batch normalisation{g.get_debug_msg()}"
        norm = batch_out

    if empty_mean_std:
        assert g.has_shape(
            x
        ), f"Not implemented when shape for {x!r} is not known{g.get_debug_msg()}"
        shape = g.get_shape(x)
        assert (
            len(shape) >= 2
        ), f"Not implemented when shape of {x!r} is {shape}{g.get_debug_msg()}"
        assert is_static_dimension(
            shape[1]
        ), f"Not implemented when shape of {x!r} is {shape}{g.get_debug_msg()}"
        size = shape[1]
        running_mean_fp32 = g.op.ConstantOfShape(np.array([size], dtype=np.int64), name=name)
        invstd = g.op.Identity(running_mean_fp32, name=name)
    elif not training:
        # CUDA and CPU gives different shapes:
        # https://github.com/pytorch/pytorch/blob/a44f8894fa6d973693aab44a3dda079a168b05c1/torch/_decomp/decompositions.py#L1451-L1457
        # We use CUDA's output here
        invstd = g.op.Reciprocal(
            g.op.Sqrt(
                g.op.Add(running_var, np.array([eps], dtype=dtype), name=name),
                name=name,
            ),
            name=name,
        )
        # https://github.com/pytorch/pytorch/blob/a44f8894fa6d973693aab44a3dda079a168b05c1/torch/_decomp/decompositions.py#L1475
        running_mean_fp32 = g.op.Cast(running_mean, to=TensorProto.FLOAT, name=name)
        invstd = g.op.Cast(invstd, to=TensorProto.FLOAT, name=name)
    else:
        running_mean_fp32, invstd = bmean, bvar

    if not sts:
        set_type_shape_unary_op(g, norm, x)

    itype = g.get_type(x)
    m = g.op.Cast(running_mean_fp32, to=itype, name=name, outputs=outputs[1:2])
    s = g.op.Cast(invstd, to=itype, name=name, outputs=outputs[2:3])
    return norm, m, s


def aten_cudnn_batch_norm(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    weight: T,
    bias: Optional[T],
    running_mean: Optional[T],
    running_var: Optional[T],
    training: bool = False,
    momentum: float = 0.9,
    eps: float = 1e-05,
    name: str = "cudnn_batch_norm",
) -> Tuple[T, T, T]:
    """cudnn_batch_norm"""
    a, b, c = aten__native_batch_norm(
        g,
        sts,
        outputs[:3],
        x,
        weight,
        bias,
        running_mean=running_mean,
        running_var=running_var,
        training=training,
        momentum=momentum,
        eps=eps,
        name=name,
    )

    d = g.op.ConstantOfShape(
        np.array([0], dtype=np.int64),
        value=from_array(np.array([0], dtype=tensor_dtype_to_np_dtype(TensorProto.UINT8))),
        outputs=outputs[3:],
        name=name,
    )
    if training:
        # Cudnn return running mean and variance when training is True
        return a, b, c, d

    return a, b, c, d


def aten_col2im(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    output_size: List[int],
    kernel_size: List[int],
    dilation: Sequence[int] = (1, 1),
    padding: Sequence[int] = (0, 0),
    stride: Sequence[int] = (1, 1),
    name: str = "col2im",
) -> T:
    """col2im"""

    assert (
        isinstance(output_size, list) and len(output_size) == 2
    ), f"not supported for output_size={output_size}{g.get_debug_msg()}"
    assert (
        isinstance(kernel_size, list) and len(kernel_size) == 2
    ), f"not supported for kernel_size={kernel_size}{g.get_debug_msg()}"
    assert isinstance(dilation, (tuple, list)) and (
        len(dilation) == 2
    ), f"not supported for dilation={dilation}{g.get_debug_msg()}"
    assert isinstance(stride, (tuple, list)) and (
        len(stride) == 2
    ), f"not supported for stride={stride}{g.get_debug_msg()}"

    # The pads should be [w, x, y, z] for ONNX
    if len(padding) == 1:  # [w] -> [w, w, w, w]
        pads = padding * 4
    elif len(padding) == 2:  # [w, x] -> [w, x, w, x]
        pads = padding * 2
    else:  # assert len(padding) == 4, already [w, x, y, z]
        pads = padding

    # Only one ONNX op here so didn't write a private function
    return g.op.Col2Im(
        x,
        np.array(output_size, dtype=np.int64),
        np.array(kernel_size, dtype=np.int64),
        dilations=list(dilation),
        pads=list(pads),
        strides=list(stride),
        name=name,
        outputs=outputs,
    )


def aten_ne(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    y: T,
    name="ne",
) -> T:
    "not equal"
    x, y = prepare_inputs_homogeneous_operator(g, x, y)
    eq = g.op.Equal(x, y, name=name)
    res = g.op.Not(eq, name=name, outputs=outputs)
    if not sts:
        set_type_shape_binary_op(g, eq, x, y, cmp_op=True)
        set_type_shape_unary_op(g, res, eq)
    return res


def aten_ne_Scalar(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    y: T,
    name="ne_Scalar",
) -> T:
    "not equal"
    res = aten_ne(g, sts, outputs, x, y, name=name)
    if not sts:
        set_type_shape_binary_op(g, outputs[0], x, y, cmp_op=True)
    return res


def aten_ne_Tensor(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    y: T,
    name="ne_Tensor",
) -> T:
    "not equal"
    res = aten_ne(g, sts, outputs, x, y, name=name)
    if not sts:
        set_type_shape_binary_op(g, outputs[0], x, y, cmp_op=True)
    return res


def aten_neg(
    g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T, name="neg"
) -> T:
    "neg"
    res = g.make_node("Neg", [x], outputs, name=name)
    if not sts:
        set_type_shape_unary_op(g, res, x)
    return res


def aten_new_zeros(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    size: T,
    dtype: Optional["torch.dtype"] = None,  # noqa: F821
    layout=None,
    device: Optional["torch.device"] = None,  # noqa: F821
    pin_memory=None,
    name: str = "seros",
) -> T:
    "constantofshape"
    if dtype is None:
        dtype = onnx_dtype_to_torch_dtype(g.get_type(x))
    return aten_full(
        g,
        sts,
        outputs,
        size,
        fill_value=None,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        name=name,
    )


def aten_nll_loss_forward(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    self: T,
    target: T,
    weight: Optional[T] = None,
    reduction: int = 0,
    ignore_index: int = -1,
    name: str = "nll_loss_forward",
) -> Tuple[T, T]:
    """nll_loss_forward"""
    n_dims = g.get_rank(self)
    channel_dim = 1
    if n_dims < 2:
        channel_dim = 0

    if weight is not None:
        weight_shape = g.get_shape(weight)
        if n_dims > 1:
            shape = [1] * n_dims
            shape[channel_dim] = weight_shape[0]
            w = g.op.Reshape(weight, np.array(shape, dtype=np.int64), name=name)
        else:
            w = weight
        self = g.op.Mul(self, w, name=name)

    target_not = g.op.Not(
        g.op.Equal(target, np.array([ignore_index], dtype=np.int64), name=name),
        name=name,
    )
    safe_target = g.op.Where(
        target_not,
        target,
        np.array([0], dtype=tensor_dtype_to_np_dtype(g.get_type(target))),
        name=name,
    )
    g.set_type_shape_or_rank(safe_target, like=target)
    safe_target_ = g.op.UnsqueezeAnyOpset(
        safe_target, np.array([channel_dim], dtype=np.int64), name=name
    )

    # result = -torch.gather(self, channel_dim, safe_target_).squeeze(channel_dim)
    result_ = g.op.Neg(
        g.op.SqueezeAnyOpset(
            g.op.GatherElements(self, safe_target_, axis=channel_dim, name=name),
            np.array([channel_dim], dtype=np.int64),
            name=name,
        ),
        name=name,
    )

    # result = torch.where(target != ignore_index, result, 0)
    result = g.op.Where(
        target_not,
        result_,
        np.array([0], dtype=tensor_dtype_to_np_dtype(g.get_type(self))),
        name=name,
    )
    g.set_type_shape_or_rank(result, like=result_)

    if reduction is None and n_dims > 1:
        return g.op.Identity(result, name=name, outputs=outputs[:1]), g.op.Identity(
            np.array([0], dtype=tensor_dtype_to_np_dtype(g.get_type(result))),
            name=name,
            outputs=outputs[1:],
        )

    if weight is not None:
        # w = w.expand(self.shape)
        self_shape = g.op.Shape(self, name=name)
        w_ = g.op.Expand(w, self_shape, name=name)

        # wsum = torch.gather(w, channel_dim, safe_target_).squeeze(channel_dim)
        gathered = g.op.GatherElements(w_, safe_target_, axis=channel_dim, name=name)
        wsum = g.op.SqueezeAnyOpset(
            gathered,
            np.array([channel_dim], dtype=np.int64),
            name=name,
        )

        # wsum = torch.where(target != ignore_index, wsum, 0)
        wsum_ = g.op.Where(
            target_not,
            wsum,
            np.array([0], dtype=tensor_dtype_to_np_dtype(g.get_type(wsum))),
            name=name,
        )
        g.set_type_shape_or_rank(wsum_, like=wsum)
        total_weight = g.op.ReduceSumAnyOpset(
            wsum_, name=name, outputs=outputs[1:], keepdims=0
        )
    else:
        # total_weight = (target != ignore_index).sum().to(self)
        total_weight = g.op.ReduceSumAnyOpset(
            g.op.Cast(target_not, to=g.get_type(self), name=name),
            name=name,
            outputs=outputs[1:],
            keepdims=0,
        )

    if reduction == Reduction.SUM.value:
        final_result = g.op.ReduceSumAnyOpset(
            result, name=name, outputs=outputs[:1], keepdims=0
        )
    elif reduction == Reduction.MEAN.value:
        final_result = g.op.Div(
            g.op.ReduceSumAnyOpset(result, name=name, keepdims=0),
            total_weight,
            outputs=outputs[:1],
            name=name,
        )
    else:
        final_result = g.op.Identity(result, name=name, outputs=outputs[:1])

    return final_result, total_weight


def aten_not(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    name: str = "not",
) -> T:
    "not"
    res = g.make_node("Not", [x], outputs, name=name)
    if not sts:
        set_type_shape_unary_op(g, res, x)
    return res


def aten_logical_not(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    name="logical_not",
) -> T:
    "logical not"
    return aten_not(g, sts, outputs, x, name=name)


def aten_not_(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    name: str = "not",
) -> T:
    "not"
    return aten_not(g, sts, outputs, x, name="not_")


def aten_ones(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    size: T,
    dtype: Optional[int] = None,
    layout=None,
    device: Optional["torch.device"] = None,  # noqa: F821
    pin_memory=None,
    name: str = "ones",
) -> T:
    "constantofshape"
    import torch

    assert layout in (
        None,
        torch.strided,
    ), f"not implemented for layout={layout!r}{g.get_debug_msg()}"
    assert not pin_memory, "ones not implemented for pin_memory=True"
    new_shape = None
    if isinstance(size, (tuple, list)):
        if all_int(size) and min(size) > 0:
            isize = np.array(size, dtype=np.int64)
            new_shape = tuple(size)
        else:
            isize = g.make_shape_from_results(list(size), name=f"{name}_dyn")
            new_shape = tuple(size)
    elif isinstance(size, int):
        isize = np.array([size], dtype=np.int64)
        new_shape = (size,)
    elif isinstance(size, torch.Tensor):
        isize = size.detach().numpy().astype(np.int64)
        new_shape = tuple(isize)
    else:
        isize = g.op.Cast(size, to=TensorProto.INT64)
        new_shape = None
    if dtype is None:
        import torch

        dtype = torch.float32
    res = g.op.ConstantOfShape(
        isize,
        value=from_array(
            np.array([1], dtype=tensor_dtype_to_np_dtype(torch_dtype_to_onnx_dtype(dtype)))
        ),
        outputs=outputs,
        name=name,
    )
    if not sts:
        g.set_type(res, dtype)
        if new_shape:
            g.set_shape(res, new_shape)
    return res


def aten_new_ones(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    size: T,
    dtype: Optional["torch.dtype"] = None,  # noqa: F821
    layout=None,
    device: Optional["torch.device"] = None,  # noqa: F821
    pin_memory=None,
    name: str = "new_ones",
) -> T:
    """new_ones"""
    if dtype is None:
        dtype = onnx_dtype_to_torch_dtype(g.get_type(x))
    return aten_ones(
        g,
        sts,
        outputs,
        size,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        name=name,
    )


def aten_ones_like(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    dtype: Optional["torch.dtype"] = None,  # noqa: F821
    layout=None,
    device: Optional["torch.device"] = None,  # noqa: F821
    pin_memory=None,
    memory_format=None,
) -> T:
    "constantofshape"
    return aten_full_like(
        g,
        sts,
        outputs,
        x,
        1,
        dtype=dtype or g.get_type(x),
        name="ones_like",
    )


def aten_pad(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    pad: Union[T, Tuple[int, ...]],
    mode: str = "constant",
    value: Optional[float] = None,
    name: str = "pad",
    pad_is_right: bool = False,
) -> T:
    """pad"""
    assert mode in {
        "constant",
        "reflect",
    }, f"Not implemented for mode={mode!r}{g.get_debug_msg()}"
    assert g.has_rank(x), f"Not implemented when rank of {x!r} is missing{g.get_debug_msg()}"
    value = float(value or 0)

    rk = g.get_rank(x)
    if isinstance(pad, list):
        if len(pad) < rk * 2:
            pad = list(pad) + list((0,) * (rk * 2 - len(pad)))
        new_pad = pad[::2][::-1] + pad[1::2][::-1]
        new_pad = np.array(new_pad, dtype=np.int64)
    else:
        assert (
            pad_is_right
        ), f"not implemented if pad={pad!r} is coming from pytorch{g.get_debug_msg()}"
        new_pad = pad

    dtype = tensor_dtype_to_np_dtype(g.get_type(x))
    cst = np.array(value, dtype=dtype)
    res = g.op.Pad(x, new_pad, cst, name=name, outputs=outputs, mode=mode)
    if not sts:
        g.set_type(res, g.get_type(x))
        if g.has_rank(x):
            g.set_rank(res, g.get_rank(x))
    return res


def aten_constant_pad_nd(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    pad: Any,
    value: float = 0.0,
    name: str = "constant_pad_nd",
) -> T:
    """pad"""
    if all(isinstance(i, (int, str)) for i in pad):
        # We need to concatenate first.
        assert g.has_rank(
            x
        ), f"Not implemented when rank of {x!r} is missing{g.get_debug_msg()}"
        rk = g.get_rank(x)
        if len(pad) < rk * 2:
            pad = list(pad) + list((0,) * (rk * 2 - len(pad)))
        onnx_pad = pad[::2][::-1] + pad[1::2][::-1]
        new_pad = g.make_shape_from_results(onnx_pad, name=name)
        name = f"{name}_dyn"
        pad_is_right = True
    elif all_int(pad) and isinstance(value, (int, float)):
        new_pad = pad
        pad_is_right = False
    else:
        raise AssertionError(
            f"Not implemented for pad={pad!r}, value={value!r}{g.get_debug_msg()}"
        )
    if value == 0:
        return aten_pad(
            g, sts, outputs, x, new_pad, mode="constant", name=name, pad_is_right=pad_is_right
        )
    return aten_pad(
        g,
        sts,
        outputs,
        x,
        new_pad,
        mode="constant",
        name=name,
        value=value,
        pad_is_right=pad_is_right,
    )


def aten_reflection_pad2d(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    pad: Any,
    value: float = 0.0,
    name: str = "reflection_pad2d",
) -> T:
    """pad"""
    if all_int(pad) and isinstance(value, (int, float)):
        if value == 0:
            return aten_pad(g, sts, outputs, x, pad, mode="reflect", name=name)
        return aten_pad(g, sts, outputs, x, pad, mode="reflect", name=name, value=value)

    raise AssertionError(
        f"Not implemented for pad={pad!r}, value={value!r}{g.get_debug_msg()}"
    )


def aten_permute(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    dims: Sequence[int],
) -> T:
    "transpose"
    if not dims:
        return g.op.Transpose(x, outputs=outputs, name="permute")

    dims = [axis + len(dims) if axis < 0 else axis for axis in dims]
    return g.op.Transpose(x, perm=dims, outputs=outputs, name="permute")


def aten_polar(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    angle: T,
    name: str = "polar",
) -> T:
    """polar"""
    itype = g.get_type(x)
    ctype = TensorProto.COMPLEX128 if itype == TensorProto.DOUBLE else TensorProto.COMPLEX64
    j = np.array([1j], dtype=np.complex128 if itype == TensorProto.DOUBLE else np.complex64)
    real = g.op.Cast(g.op.Cos(angle, name=name), to=ctype, name=name)
    imag = g.op.Mul(g.op.Cast(g.op.Sin(angle, name=name), to=ctype, name=name), j, name=name)
    res = g.op.Mul(
        g.op.Cast(x, to=ctype, name=name),
        g.op.Add(real, imag, name=name),
        name=name,
        outputs=outputs,
    )
    if not sts:
        set_type_shape_binary_op(g, res, x, angle, itype=ctype)
    return res


def aten_pow_Scalar(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    exponent: T,
    name: str = "pow_Scalar",
) -> T:
    "pow"
    return aten_pow_Tensor_Tensor(g, sts, outputs, x, exponent, name=name)


def aten_pow_Tensor_Scalar(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    exponent: T,
    name: str = "pow_Tensor_Scalar",
) -> T:
    "pow"
    return aten_pow_Tensor_Tensor(g, sts, outputs, x, exponent, name=name)


def aten_pow_Tensor_Tensor(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    exponent: T,
    name: str = "pow_Tensor_Tensor",
) -> T:
    "pow"
    if isinstance(exponent, (int, float)):
        if exponent == 1:
            # The node is removed.
            return g.op.Identity(x, outputs=outputs)
        exponent = np.array([exponent])
    if isinstance(x, (int, float)):
        assert isinstance(exponent, str), (
            f"Unexpected type for exponent, type(x)={type(x)}, "
            f"type(exponent)={type(exponent)}{g.get_debug_msg()}"
        )
        itype = g.get_type(exponent)
        x = np.array([x], dtype=tensor_dtype_to_np_dtype(itype))
        res = g.op.Pow(x, exponent, outputs=outputs, name=name)
        if not sts:
            set_type_shape_unary_op(g, res, exponent)
        return res

    if isinstance(exponent, np.ndarray):
        if g.has_type(x):
            exponent = exponent.astype(tensor_dtype_to_np_dtype(g.get_type(x)))
        else:
            exponent = g.op.CastLike(exponent, x, name=name)
    else:
        assert isinstance(
            exponent, str
        ), f"unexpected type {type(exponent)} for exponent{g.get_debug_msg()}"
        assert g.get_type(x) == g.get_type(exponent), (
            f"type mismatch between {x!r} and {exponent!r}, "
            f"{g.get_type(x)} != {g.get_type(exponent)}{g.get_debug_msg()}"
        )
    res = g.op.Pow(x, exponent, outputs=outputs, name=name)
    if not sts:
        set_type_shape_unary_op(g, res, x)
    return res


def aten_prelu(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    a: T,
    slope: T,
    name: str = "prelu",
) -> T:
    "prelu"
    if g.has_rank(a) and g.is_constant(slope):
        cst = g.get_constant(slope, computed_value=True)
        if cst.size == 1:
            rc = g.get_rank(slope)
            r = g.get_rank(a)
            if r != rc:
                # onnxruntime is faster when the rank of the slope is the same
                # as the rank of the input even if it contains only one element
                slope = g.op.Reshape(slope, np.array((1,) * r, dtype=np.int64), name=name)
    res = g.op.PRelu(a, slope, name=name, outputs=outputs)
    if not sts:
        set_type_shape_unary_op(g, res, a)
    return res


def aten__prelu_kernel(
    g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T, weight: T
) -> T:
    "prelu"
    dtype = tensor_dtype_to_np_dtype(g.get_type(x))
    ge = g.op.Greater(x, np.array([0], dtype=dtype), name="_prelu_kernel")
    mu = g.op.Mul(x, weight, name="_prelu_kernel")
    res = g.op.Where(ge, x, mu, outputs=outputs, name="_prelu_kernel")
    set_type_shape_unary_op(g, ge, x, TensorProto.BOOL)
    set_type_shape_unary_op(g, mu, weight)
    if not sts:
        set_type_shape_unary_op(g, res, x)
    return res


def aten__prelu_kernel_backward(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    grad_output: T,
    x: T,
    weight: T,
) -> Tuple[T, T]:
    "prelu backward"
    dtype = tensor_dtype_to_np_dtype(g.get_type(x))
    zero = g.make_initializer("zero", np.array([0], dtype=dtype))
    xg0 = g.op.Greater(x, zero, name="_prelu_kernel_backward")
    mu1 = g.op.Mul(weight, grad_output, name="_prelu_kernel_backward")
    input_grad = g.op.Where(
        xg0,
        grad_output,
        mu1,
        name="_prelu_kernel_backward",
        outputs=outputs[:1],
    )
    mu2 = g.op.Mul(x, grad_output, name="_prelu_kernel_backward")
    weight_grad = g.op.Where(
        xg0, zero, mu2, name="_prelu_kernel_backward", outputs=outputs[1:]
    )
    set_type_shape_unary_op(g, xg0, x, TensorProto.BOOL)
    set_type_shape_unary_op(g, mu1, weight)
    set_type_shape_unary_op(g, mu2, x)
    if not sts:
        set_type_shape_unary_op(g, input_grad, x)
        set_type_shape_unary_op(g, weight_grad, weight)
    return input_grad, weight_grad


def aten_prod(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    dtype: Optional["torch.dtype"] = None,  # noqa: F821
    name: str = "prod",
) -> T:
    """min"""
    if dtype is not None:
        itype = torch_dtype_to_onnx_dtype(dtype)
        x = g.op.Cast(x, to=itype, name=name)
    res = g.op.ReduceProd(x, keepdims=0, name=name, outputs=outputs)
    if not sts:
        g.set_type(res, g.get_type(x) if dtype is None else itype)
        g.set_shape(res, tuple())
    return res


def aten_prod_dim_int(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    dim: int,
    keepdim: bool = False,
    dtype: Optional["torch.dtype"] = None,  # noqa: F821
    name="prod_dim_int",
) -> T:
    """reduce_prod"""

    if dtype is not None:
        itype = torch_dtype_to_onnx_dtype(dtype)
        x = g.op.Cast(x, to=itype, name=name)

    return g.op.ReduceProd(
        x,
        np.array([dim], dtype=np.int64),
        keepdims=1 if keepdim else 0,
        outputs=outputs,
        name=name,
    )


def aten_reciprocal(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    name: str = "reciprocal",
) -> T:
    """reciprocal"""
    res = g.op.Reciprocal(x, name=name, outputs=outputs)
    if not sts:
        set_type_shape_unary_op(g, res, x)
    return res


def aten_scatter_add(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    dim: int,
    index: T,
    src: T,
    name: str = "scatter_add",
) -> T:
    """scatter_add"""
    assert isinstance(
        dim, int
    ), f"scatter_add not implemented if type(dim)={type(dim)}{g.get_debug_msg()}"
    if g.main_opset < 13:
        # reduction is not available for opset < 13
        raise FunctionNotFoundError(
            f"reduction='add' not available in opset {g.main_opset}{g.get_debug_msg()}"
        )
    res = g.op.ScatterElements(
        x, index, src, axis=dim, reduction="add", name=name, outputs=outputs
    )
    if not sts:
        set_type_shape_unary_op(g, res, x)
    return res


def aten_scatter_reduce_two(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    dim: int,
    index: T,
    src: T,
    reduce: str,
    include_self: bool = True,
    name: str = "scatter_reduce_two",
):
    """scatter_reduce.two"""
    assert (
        reduce != "mean"
    ), f"scatter_reduce_two not implemented for reduce={reduce!r}{g.get_debug_msg()}"
    assert g.has_rank(x), f"rank of {x!r} is expected{g.get_debug_msg()}"
    assert not include_self, f"not implemented if include_self is True{g.get_debug_smg()}"
    reduce_mode = {"mean": "none", "sum": "add", "prod": "mul", "amin": "min", "amax": "max"}
    onnx_reduce = reduce_mode[reduce]
    is_scalar = g.get_rank(x) == 0
    if is_scalar:
        x = g.op.Reshape(x, np.array([-1], dtype=np.int64), name=name)
        index = g.op.Reshape(x, np.array([-1], dtype=np.int64), name=name)
        src = g.op.Reshape(src, np.array([-1], dtype=np.int64), name=name)
        result = g.op.ScatterElements(
            x, index, src, axis=dim, reduction=onnx_reduce, name=name
        )
        result = g.op.Squeeze(result, name=name, outputs=outputs)
    else:
        result = g.op.ScatterElements(
            x, index, src, axis=dim, reduction=onnx_reduce, name=name, outputs=outputs
        )
    return result


def aten_relu(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    inplace: bool = False,
) -> T:
    "relu"
    assert not inplace, f"inplace computation is not allowed with onnx{g.get_debug_msg()}"
    res = g.op.Relu(x, outputs=outputs)
    if not sts:
        set_type_shape_unary_op(g, outputs[0], x)
    return res


def aten_rrelu_with_noise_backward(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    grad_output: T,
    x: T,
    noise: T,
    lower: float,
    upper: float,
    training: bool,
    self_is_result: bool,
    name: str = "rrelu_with_noise_backward",
) -> T:
    "rrelu"
    assert not training, f"Not implemented if training is True{g.get_debug_msg()}"

    # if training and upper - lower > 1e-6:
    #     return grad_output.mul(noise)
    negative_slope = (lower + upper) / 2
    return aten_leaky_relu_backward(
        g, sts, outputs, grad_output, x, negative_slope, self_is_result, name=name
    )


def aten_remainder(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    other: T,
    name="remainder",
) -> T:
    """mod"""

    # a - a.div(b, rounding_mode="floor") * b
    if isinstance(other, (int, float)):
        dtype = tensor_dtype_to_np_dtype(g.get_type(x))
        res = g.op.Mod(x, np.array([other], dtype=dtype), name=name, outputs=outputs)
        if not sts:
            set_type_shape_unary_op(g, res, x)
        return res

    # rounded_quotient = g.op.Floor(g.op.Div(self, other))
    # return op.Sub(self, op.Mul(rounded_quotient, other))
    itype = g.get_type(x)
    if itype in {
        TensorProto.FLOAT,
        TensorProto.DOUBLE,
        TensorProto.FLOAT16,
        TensorProto.BFLOAT16,
    }:
        # This does not work for negative values.
        # res = g.op.Mod(x, other, name=name, outputs=outputs, fmod=1)

        rounded_quotient = g.op.Floor(g.op.Div(x, other, name=name), name=name)
        res = g.op.Sub(
            x, g.op.Mul(rounded_quotient, other, name=name), outputs=outputs, name=name
        )
    else:
        res = g.op.Mod(x, other, name=name, outputs=outputs)
    if not sts:
        set_type_shape_unary_op(g, res, x)
    return res


def aten_remainder_Scalar(
    g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T, other: T
) -> T:
    """mod"""
    return aten_remainder(g, sts, outputs, x, other, name="remainder_Scalar")


def aten_remainder_Tensor(
    g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T, other: T
) -> T:
    """mod"""
    return aten_remainder(g, sts, outputs, x, other, name="remainder_Tensor")


def aten_repeat(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    repeats: T,
    name: str = "repeat",
) -> T:
    "repeat"
    assert isinstance(repeats, (tuple, list))
    if all_int(repeats):
        if set(repeats) == {1}:
            # identity
            return g.op.Identity(x, name=name, outputs=outputs)
        irep = np.array(repeats, dtype=np.int64)
    elif g.is_dynamic_shape(repeats):
        # repeats is like a shape
        irep = g.make_shape_from_results(repeats)
    else:
        raise RuntimeError(f"repeat not implemented for repeats={repeats}{g.get_debug_msg()}")
    if g.get_rank(x) != len(repeats):
        assert g.get_rank(x) < len(repeats), (
            f"Unexpected rank {g.get_rank(x)} for x={x!r}, repeats={repeats}"
            f"{g.get_debug_msg()}"
        )
        expanded = g.op.UnsqueezeAnyOpset(
            x, np.arange(len(repeats) - g.get_rank(x)).astype(np.int64), name=name
        )
    else:
        expanded = x
    res = g.op.Tile(expanded, irep, name=name)
    if not sts:
        g.set_type(res, g.get_type(x))
        if g.has_shape(x) and all_int(repeats):
            shape = g.get_shape(x)
            if len(shape) == len(repeats):
                new_shape = np.array(shape) * irep
                g.set_shape(res, tuple(map(int, new_shape)))
            else:
                g.set_rank(res, len(repeats))
        else:
            g.set_rank(res, len(repeats))
    return res


def aten_repeat_interleave(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    repeats: List[int],
    dim: Optional[int] = None,
    output_size: Optional[Tuple[int, ...]] = None,
    name: str = "repeat_interleave",
) -> T:
    "repeat_interleave"
    assert isinstance(dim, int), f"dim={dim} is not an integer{g.get_debug_msg()}"
    assert (
        output_size is None
    ), f"Not implemented when output_size={output_size} is not None{g.get_debug_msg()}"
    assert g.has_rank(x), f"Rank for x={x!r} is needed{g.get_debug_msg()}"
    rkx = g.get_rank(x)

    if isinstance(dim, int) and isinstance(repeats, int):
        pos_dim = (dim + rkx) % rkx
        unsqueezed = g.op.UnsqueezeAnyOpset(
            x, np.array([pos_dim + 1], dtype=np.int64), name=name
        )
        onehot = np.ones((rkx + 1,), dtype=np.int64)
        onehot[pos_dim + 1] = repeats
        tiled = g.op.Tile(unsqueezed, onehot, name=name)

        if dim < -1:
            dim += rkx
        res = aten_flatten(
            g,
            sts,
            outputs,
            tiled,
            -2 if dim == -1 else dim,
            -1 if dim == -1 else (dim + 1),
            name=name,
        )
        if not sts:
            g.set_type(res, g.get_type(x))
            g.set_rank(res, rkx)
        return res

    raise NotImplementedError(
        f"Not Implemented for x={x!r}, repeats={repeats!r}, dim={dim!r}, "
        f"output_size={output_size!r}{g.get_debug_msg()}"
    )


def aten_repeat_interleave_self_int(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    repeats: List[int],
    dim: Optional[int] = None,
    output_size: Optional[Tuple[int, ...]] = None,
    name: str = "repeat_interleave_self_int",
) -> T:
    "repeat_interleave_self_int"
    return aten_repeat_interleave(g, sts, outputs, x, repeats, dim, output_size, name=name)


def aten_roll(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    shifts: List[int],
    dims: List[int],
    name: str = "roll",
) -> T:
    "roll"
    assert len(shifts) == len(
        dims
    ), f"Unexpected values for shifts={shifts} and dims={dims}{g.get_debug_msg()}"

    shape_x = g.get_shape(x) if g.has_shape(x) else None
    shape_xx = None

    result = x
    for i in range(len(shifts)):
        if shifts[i] == 0:
            continue

        shapes = []
        if shape_x is not None and is_static_dimension(shape_x[dims[i]]):
            end = np.array([shape_x[dims[i]]], dtype=np.int64)
        else:
            if shape_xx is None:
                shape_xx = g.op.Shape(x, name=name)
            end = g.op.Gather(shape_xx, np.array([dims[i]], dtype=np.int64), name=name)

        axis = np.array([dims[i]], dtype=np.int64)

        # first part
        shape = g.op.Slice(
            result, np.array([-shifts[i]], dtype=np.int64), end, axis, name=name
        )
        shapes.append(shape)
        # second part
        shape = g.op.Slice(
            result,
            np.array([0], dtype=np.int64),
            np.array([-shifts[i]], dtype=np.int64),
            axis,
            name=name,
        )
        shapes.append(shape)

        result = g.op.Concat(*shapes, axis=dims[i], name=name)
        g.set_type(result, g.get_type(x))
        if g.has_shape(x):
            g.set_shape(result, g.get_shape(x))

    return g.op.Identity(result, name=name, outputs=outputs)


def aten_round(g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T) -> T:
    "round"
    res = g.make_node("Round", [x], outputs, name="round")
    if not sts:
        set_type_shape_unary_op(g, outputs[0], x)
    return res


def aten_rsqrt(g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T) -> T:
    "rqsrt"
    ext = g.make_node("Sqrt", [x], name="rsqrt")
    res = g.op.Reciprocal(ext, outputs=outputs, name="rsqrt")
    if not sts:
        set_type_shape_unary_op(g, outputs[0], x)
    return res


def aten_rsub(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    y: T,
    alpha: float = 1,
) -> T:
    "rsub"
    assert alpha == 1, f"Not implemented with alpha={alpha}"
    return aten_sub(g, sts, outputs, y, x, name="rsub")


def aten_rsub_Scalar(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    y: T,
    alpha: float = 1,
) -> T:
    "rsub"
    assert alpha == 1, f"Not implemented with alpha={alpha}"
    return aten_sub(g, sts, outputs, y, x, name="rsub_Scalar")


def aten_scalar_tensor(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    s: float,
    dtype: Optional[int] = None,
    layout: str = "",
    device: Optional["torch.device"] = None,  # noqa: F821
    pin_memory=None,
) -> T:
    """constant"""
    import torch

    assert layout in (
        None,
        torch.strided,
    ), f"not implemented for layout={layout!r}{g.get_debug_msg()}"
    assert pin_memory in (
        None,
        False,
    ), f"not implemented for pin_memory={pin_memory!r} {g.get_debug_msg()}"
    assert isinstance(
        s, (float, int)
    ), f"not implemented for type(s)={type(s)!r}{g.get_debug_msg()}"
    if dtype is None:
        if g.has_type(outputs[0]):
            dtype = tensor_dtype_to_np_dtype(g.get_type(outputs[0]))
        else:
            np_dtype = np.float32  # if isinstance(s, float) else np.int64
    else:
        np_dtype = tensor_dtype_to_np_dtype(torch_dtype_to_onnx_dtype(dtype))
    return g.op.Identity(np.array(s, dtype=np_dtype), outputs=outputs)


def aten_select_int(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    dim: int,
    index: int,
) -> T:
    "gather"
    assert isinstance(dim, int), f"Unexpected type {type(dim)} for dim{g.get_debug_msg()}"
    assert isinstance(index, int), f"Unexpected type {type(index)} for dim{g.get_debug_msg()}"
    res = g.op.Gather(x, np.array(index, dtype=np.int64), axis=dim, outputs=outputs)
    if not sts:
        g.set_type(res, g.get_type(x))
        if g.has_shape(x):
            shape = g.get_shape(x)
            if dim < 0:
                dim += len(shape)
            assert dim < len(shape), f"shape is {shape}, dim is {dim}{g.get_debug_msg()}"
            new_shape = [s for i, s in enumerate(shape) if i != dim]
            g.set_shape(res, tuple(new_shape))
        else:
            g.get_rank(res, g.get_rank(x) - 1)
    return res


def aten_select_scatter(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    src: T,
    dim: int,
    index: int,
    name: str = "select_scatter",
) -> T:
    """scatter elements"""
    assert isinstance(
        index, int
    ), f"select_scatter not implemented for index={index!r}{g.get_debug_msg()}"

    # Change src rank to self rank according to dim
    # e.g. if self is [2,3,4], src is [2,4], dim=1, then update is [2,1,4]
    update = g.op.Unsqueeze(src, axes=dim, name=name)
    # Change index rank to the same as 'update' [2,1,4]
    indices = g.op.Expand(
        np.array([index], dtype=np.int64), g.op.Shape(update, name=name), name=name
    )
    res = g.op.ScatterElements(
        x, indices, update, axis=dim, reduction="none", name=name, outputs=outputs
    )
    if not sts:
        g.set_type(res, g.get_type(x))
    return res


def aten_selu(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    inplace: bool = False,
) -> T:
    "relu"
    assert not inplace, f"inplace computation is not allowed with onnx{g.get_debug_msg()}"
    res = g.op.Selu(x, outputs=outputs)
    if not sts:
        set_type_shape_unary_op(g, outputs[0], x)
    return res


def aten__set_grad_enabled(
    g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], enable: bool
) -> T:
    """
    Returns the function returns a dummy which will be removed
    after the graph is created.
    """
    assert isinstance(
        enable, bool
    ), f"Unexpected type for enable={enable!r}{g.get_debug_msg()}"
    return g.make_node("Constant", [], value_ints=[1], name="_set_grad_enabled")


def aten_setitem(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    indices: Tuple[Any, ...],
    values: T,
) -> T:
    "scatter"
    if (
        isinstance(indices, tuple)
        and len(indices) == 2
        and indices[0] is Ellipsis
        and isinstance(indices[1], slice)
    ):
        s = indices[1]
        return aten_slice_scatter(
            g,
            sts,
            outputs,
            x,
            values,
            dim=g.get_rank(x) - 1,
            start=s.start,
            end=s.stop,
            step=s.step,
            name="setitem",
        )

    # if not sts:
    #    g.set_type(res, g.get_type(x))
    #    if g.has_shape(x):
    #        g.set_shape(res, g.get_shape(x))
    #    else:
    #        g.set_rank(res, g.get_rank(x))
    # return res
    raise RuntimeError(f"setitem not implemented for indices={indices}{g.get_debug_msg()}")


def aten_slice_Tensor(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    dim: int = 0,
    start: int = 0,
    end: Optional[int] = None,
    step: Optional[int] = None,
) -> T:
    "slice"
    assert isinstance(dim, int), f"aten_slice_Tensor not implemented for dim={dim!r}"
    assert g.is_dynamic_dimension(
        start
    ), f"aten_slice_Tensor not implemented for start={start!r}{g.get_debug_msg()}"
    assert end is None or g.is_dynamic_dimension(
        end
    ), f"aten_slice_Tensor not implemented for end={end!r}{g.get_debug_msg()}"
    assert step is None or isinstance(step, int), f"Not implemented for step={step!r}"
    if end is None:
        end = start
        start = 0
    if start == 0 and end == 9223372036854775807 and step in {1, None}:
        # nothing to do
        return g.op.Identity(x, outputs=outputs)
    inputs = [
        g.get_dynamic_dimension(start),
        g.get_dynamic_dimension(end),
        np.array([dim], dtype=np.int64),
    ]
    if step is not None and step != 1:
        inputs.append(np.array([step], dtype=np.int64))
    res = g.op.Slice(x, *inputs, outputs=outputs, name="slice_Tensor")
    if not sts:
        g.set_type(res, g.get_type(x))
        if is_static_dimension(start) and is_static_dimension(end):
            shape = g.get_shape(x)
            new_shape = g._apply_slice_to_shape(
                shape, slice(start, end, step), axes=[dim], expand_axes=[]
            )
            g.set_shape(res, new_shape)
        else:
            g.set_rank(res, g.get_rank(x))
    return res


def aten_slice_backward(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    grad_output: T,
    input_sizes: List[int],
    dim: int,
    start: int,
    end: int,
    step: int,
    name: str = "slice_backward",
) -> T:
    "slice backward"
    assert step == 1, f"slice_backward not implemented for step={step}{g.get_debug_msg()}"

    itype = g.get_type(grad_output)
    value = from_array(np.array([0], dtype=tensor_dtype_to_np_dtype(itype)))

    inputs = []

    if g.has_shape(grad_output) and is_static_shape(g.get_shape(grad_output)):
        name_s = f"{name}_static"
        # static version
        shape = g.get_shape(grad_output)

        if isinstance(start, int):
            if start > 0:
                cst_shape = list(shape)
                cst_shape[dim] = start
                cst = g.op.ConstantOfShape(
                    np.array(cst_shape, dtype=np.int64), value=value, name=name_s
                )
                inputs.append(cst)
        else:
            name_s_ = f"{name_s}_2d"
            a_shape = g.op.Shape(grad_output, name=name_s_)
            cst_shape = g.op.ScatterElements(
                a_shape,
                np.array([dim], dtype=np.int64),
                g.op.Cast(start, to=TensorProto.INT64, name=name_s_),
                name=name_s_,
            )
            cst = g.op.ConstantOfShape(cst_shape, value=value, name=name_s_)
            inputs.append(cst)

        inputs.append(grad_output)

        if isinstance(end, int):
            if end < 9223372036854775807:
                assert isinstance(input_sizes[dim], int), (
                    f"Not implemented for input_sizes={input_sizes}, "
                    f"end={end}{g.get_debug_msg()}"
                )
                cst_shape = list(shape)
                cst_shape[dim] = input_sizes[dim] - end
                cst = g.op.ConstantOfShape(
                    np.array(cst_shape, dtype=np.int64), value=value, name=name_s
                )
                inputs.append(cst)
        else:
            name_s_ = f"{name_s}_2d"
            a_shape = g.op.Shape(grad_output, name=name_s_)
            cst_shape = g.op.ScatterElements(
                a_shape,
                np.array([dim], dtype=np.int64),
                g.op.Sub(
                    (
                        np.array([input_sizes[dim]], dtype=np.int64)
                        if isinstance(input_sizes[dim], int)
                        else g.op.Cast(input_sizes[dim], to=TensorProto.INT64, name=name_s_)
                    ),
                    g.op.Cast(end, to=TensorProto.INT64, name=name_s_),
                    name=name_s_,
                ),
                name=name_s_,
            )
            cst = g.op.ConstantOfShape(cst_shape, value=value, name=name_s_)
            inputs.append(cst)

    else:
        name_d = f"{name}_dynamic"
        # dynamic version
        shape = g.op.Shape(grad_output, name=name_d)

        if isinstance(start, str) or start > 0:
            the_start = np.array([start], dtype=np.int64) if isinstance(start, int) else start
            cst_shape = g.op.ScatterElements(
                shape,
                np.array([dim], dtype=np.int64),
                the_start,
                axis=0,
                name=name,
            )
            cst = g.op.ConstantOfShape(cst_shape, value=value, name=name_d)
            inputs.append(cst)

        inputs.append(grad_output)

        if isinstance(end, str) or end < 9223372036854775807:
            the_end = np.array([end], dtype=np.int64) if isinstance(end, int) else end
            new_dim = g.op.Sub(
                (
                    g.op.Cast(input_sizes[dim], to=TensorProto.INT64, name=name_d)
                    if isinstance(input_sizes[dim], str)
                    else np.array([input_sizes[dim]], dtype=np.int64)
                ),
                the_end,
                name=name_d,
            )
            cst_shape = g.op.ScatterElements(
                shape, np.array([dim], dtype=np.int64), new_dim, axis=0, name=name_d
            )
            cst = g.op.ConstantOfShape(cst_shape, value=value, name=name_d)
            inputs.append(cst)

    if len(inputs) > 1:
        res = g.op.Concat(*inputs, axis=dim, name=name, outputs=outputs)
    else:
        res = g.op.Identity(*inputs, outputs=outputs, name=name)

    if not sts:
        g.set_type(res, g.get_type(grad_output))
        g.set_shape(res, tuple(input_sizes))
    return res


def _aten_slice_scatter_static(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    src: T,
    dim: int = 0,
    start: Optional[int] = None,
    end: Optional[int] = None,
    step: Optional[int] = None,
    name="slice_scatter_static",
) -> T:
    "slice scatter"
    assert g.has_shape(
        x
    ), f"This implementation only works if shape of {x!r} is known{g.get_debug_msg()}"
    # step 1
    assert start is None or g.is_dynamic_dimension(
        start
    ), f"slice_scatter not implemented for start={start}{g.get_debug_msg()}"
    assert end is None or g.is_dynamic_dimension(
        end
    ), f"slice_scatter not implemented for end={end}{g.get_debug_msg()}"
    assert step is None or is_static_dimension(
        step
    ), f"slice_scatter not implemented for end={step}{g.get_debug_msg()}"
    shape = g.get_shape(x)
    dim_shape = shape[dim]

    assert is_static_dimension(
        dim_shape
    ), f"slice_scatter not implemented when shape={shape}{g.get_debug_msg()}"

    if g.has_shape(src):
        shape_src = g.get_shape(src)
        if shape_src == shape and start == 0 and end == 9223372036854775807:
            # It is identity.
            return g.op.Identity(src, outputs=outputs, name=name)

    index_1 = np.arange(0, dim_shape)
    if (start is None or isinstance(start, int)) and (end is None or isinstance(end, int)):
        if end is None:
            index_2 = index_1[start::step]
        else:
            index_2 = index_1[start or 0 : end : step]
        index_2 = index_2.copy()
    else:
        index_2 = g.op.Slice(
            index_1,
            g.get_dynamic_dimension(start),
            (
                np.array([dim_shape], dtype=np.int64)
                if end is None
                else g.get_dynamic_dimension(end)
            ),
            np.array([0], dtype=np.int64),
            np.array([step or 1], dtype=np.int64),
            name=name,
        )

    # step 2

    resh = g.op.Reshape(index_2, np.array([-1, 1], dtype=np.int64), name=name)
    if dim == 0:
        res = g.op.ScatterND(x, resh, src, outputs=outputs, name=name)
    else:
        perm = list(range(g.get_rank(x)))
        perm[0], perm[dim] = perm[dim], perm[0]
        res = g.op.Transpose(
            g.op.ScatterND(
                g.op.Transpose(x, perm=perm, name=name),
                resh,
                g.op.Transpose(src, perm=perm, name=name),
                name=name,
            ),
            perm=perm,
            outputs=outputs,
            name=name,
        )

    if not sts:
        g.set_type(res, g.get_type(x))
        g.set_shape(res, shape)
    return res


def _aten_slice_scatter_dynamic(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    src: T,
    dim: int = 0,
    start: Optional[int] = None,
    end: Optional[int] = None,
    step: Optional[int] = None,
    name="slice_scatter_dynamic",
) -> T:
    "slice scatter"
    # step 1
    assert start is None or g.is_dynamic_dimension(
        start
    ), f"slice_scatter not implemented for start={start}{g.get_debug_msg()}"
    assert end is None or g.is_dynamic_dimension(
        end
    ), f"slice_scatter not implemented for end={end}{g.get_debug_msg()}"
    assert step is None or is_static_dimension(
        step
    ), f"slice_scatter not implemented for step={step}{g.get_debug_msg()}"

    if (
        isinstance(start, int)
        and not isinstance(start, g.torch.SymInt)
        and start == 0
        and isinstance(end, int)
        and not isinstance(end, g.torch.SymInt)
        and end == 9223372036854775807
        and isinstance(step, int)
        and not isinstance(step, g.torch.SymInt)
        and step == 1
    ):
        # slice scatter on dimension
        if g.has_shape(x) and g.has_shape(src) and g.get_shape(x) == g.get_shape(src):
            # Identity
            res = g.op.Identity(src, name=name)
            if not sts:
                g.set_type(res, g.get_type(x))
                g.set_shape(res, g.set_shape(src))
            return res
        raise AssertionError(
            f"start={start}, end={end}, step={step} is not implemented yet "
            f"for slice_scatter{g.get_debug_msg()}"
        )

    shape = g.op.Shape(x, name=name)
    dim_shape = g.op.Gather(shape, np.array([dim], dtype=np.int64), name=name)

    index_1 = g.op.Range(
        np.array([0], dtype=np.int64),
        dim_shape,
        np.array([1], dtype=np.int64),
        name=name,
    )
    g.set_type(index_1, TensorProto.INT64)
    g.set_rank(index_1, 1)
    index_2 = g.op.Slice(
        index_1,
        g.get_dynamic_dimension(start),
        (dim_shape if end is None else g.get_dynamic_dimension(end)),
        np.array([0], dtype=np.int64),
        np.array([step or 1], dtype=np.int64),
        name=name,
    )

    # step 2

    resh = g.op.Reshape(index_2, np.array([-1, 1], dtype=np.int64), name=name)
    if dim == 0:
        res = g.op.ScatterND(x, resh, src, outputs=outputs, name=name)
    else:
        perm = list(range(g.get_rank(x)))
        perm[0], perm[dim] = perm[dim], perm[0]
        res = g.op.Transpose(
            g.op.ScatterND(
                g.op.Transpose(x, perm=perm, name=name),
                resh,
                g.op.Transpose(src, perm=perm, name=name),
                name=name,
            ),
            perm=perm,
            outputs=outputs,
            name=name,
        )

    if not sts:
        g.set_type(res, g.get_type(x))
        g.set_rank(res, g.get_rank(x))
    return res


def aten_slice_scatter(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    src: T,
    dim: int = 0,
    start: Optional[int] = None,
    end: Optional[int] = None,
    step: Optional[int] = None,
    name: Optional[str] = None,
) -> T:
    "slice scatter"
    if g.has_shape(x) and is_static_shape(g.get_shape(x)):
        return _aten_slice_scatter_static(
            g,
            sts,
            outputs,
            x,
            src,
            dim,
            start,
            end,
            step,
            name=name or "slice_scatter_static",
        )
    return _aten_slice_scatter_dynamic(
        g,
        sts,
        outputs,
        x,
        src,
        dim,
        start,
        end,
        step,
        name=name or "slice_scatter_dynamic",
    )


def aten_sigmoid(
    g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T
) -> T:
    "sigmoid"
    res = g.op.Sigmoid(x, outputs=outputs)
    if not sts:
        set_type_shape_unary_op(g, outputs[0], x)
    return res


def aten_sigmoid_backward(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    out_grad: T,
    y: T,
) -> T:
    """
    sigmoid backward

    See https://github.com/pytorch/pytorch/blob/main/torch/_decomp/decompositions.py#L108.
    conj_physical = identity for real number.

    ::

        return out_grad * (y * (1 - y)).conj_physical()
    """
    dtype = tensor_dtype_to_np_dtype(g.get_type(y))
    _1y = g.op.Sub(np.array([1], dtype=dtype), y, name="sigmoid_backward")
    y1y = g.op.Mul(y, _1y, name="sigmoid_backward")
    res = g.op.Mul(out_grad, y1y, outputs=outputs, name="sigmoid_backward")

    set_type_shape_unary_op(g, _1y, y)
    set_type_shape_unary_op(g, y1y, y)
    if not sts:
        set_type_shape_unary_op(g, res, y)
    return res


def aten_sign(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    name: str = "sign",
) -> T:
    """sign"""

    res = g.op.Sign(x, outputs=outputs, name=name)
    if not sts:
        set_type_shape_unary_op(g, res, x)
    return res


def aten_silu(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    inplace: bool = False,
) -> T:
    "silu"
    assert not inplace, f"inplace computation is not allowed with onnx{g.get_debug_msg()}"
    res = g.op.Mul(x, g.op.Sigmoid(x, name="silu"), outputs=outputs, name="silu")
    if not sts:
        set_type_shape_unary_op(g, res, x)
    return res


def aten_sin(
    g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T, name="sin"
) -> T:
    "sin"
    res = g.make_node("Sin", [x], outputs, name=name)
    if not sts:
        set_type_shape_unary_op(g, outputs[0], x)
    return res


def aten_sinh(g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T) -> T:
    "sinh"
    res = g.make_node("Sinh", [x], outputs, name="sinh")
    if not sts:
        set_type_shape_unary_op(g, outputs[0], x)
    return res


def aten_softmax(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    dim: int = -1,
    dtype: Optional["torch.dtype"] = None,  # noqa: F821
    name: str = "softmax",
) -> T:
    "softmax"
    if dtype is not None:
        itype = torch_dtype_to_onnx_dtype(dtype)
        xc = g.op.Cast(x, to=itype, name=name)
    else:
        itype = None
        xc = x
    res = g.op.Softmax(xc, axis=dim, outputs=outputs, name=name)
    if not sts:
        set_type_shape_unary_op(g, res, xc, itype=itype)
    return res


def aten_softmax_int(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    dim: int = -1,
    dtype: Optional["torch.dtype"] = None,  # noqa: F821
) -> T:
    "softmax"
    return aten_softmax(g, sts, outputs, x, dim, dtype, name="softmax_int")


def aten__softmax(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    dim: int = -1,
    half_to_float: bool = False,
) -> T:
    "softmax"
    cx = g.op.Cast(x, to=TensorProto.FLOAT) if half_to_float else x
    res = g.op.Softmax(cx, axis=dim, outputs=outputs, name="_softmax")
    if not sts:
        if half_to_float:
            set_type_shape_unary_op(g, res, x, itype=TensorProto.FLOAT)
        else:
            set_type_shape_unary_op(g, res, x)
    return res


def aten__softmax_backward_data(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    grad_output: T,
    y: T,
    dim: int,
    input_dtype: Optional["torch.dtype"] = None,  # noqa: F821
    name: str = "_softmax_backward_data",
) -> T:
    "softmax backward"
    itype = None
    grad_outputc = grad_output

    new_grad_output = g.op.Mul(grad_outputc, y, name=name)
    set_type_shape_unary_op(g, new_grad_output, grad_outputc)
    sums = g.op.ReduceSum(
        new_grad_output,
        np.array([dim], dtype=np.int64),
        keepdims=1,
        name=name,
    )
    set_type_shape_reduce_op(g, sums, new_grad_output, keepdim=1, axes=(dim,))
    temp = g.op.Mul(y, sums, name=name)
    set_type_shape_unary_op(g, temp, y)

    if input_dtype is not None:
        itype = torch_dtype_to_onnx_dtype(input_dtype)
        sub = g.op.Sub(new_grad_output, temp, name=name)
        res = g.op.Cast(sub, to=itype, name=name, outputs=outputs)
    else:
        res = g.op.Sub(new_grad_output, temp, outputs=outputs, name=name)

    if not sts:
        set_type_shape_unary_op(g, res, grad_outputc, itype=itype)
    return res


def aten_softplus(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    beta: float = 1.0,
    threshold: float = 20.0,
    name: str = "softplus",
):
    "softplus"
    assert isinstance(
        beta, (int, float)
    ), f"softplus not implemented for beta={beta!a}{g.get_debug_msg()}"
    assert isinstance(
        threshold, (int, float)
    ), f"softplus not implemented for threshold={threshold!a}{g.get_debug_msg()}"

    dtype = tensor_dtype_to_np_dtype(g.get_type(x))
    if beta != 1:
        bnp = np.array([beta], dtype=dtype)
        x = g.op.Mul(x, bnp, name=name)
    softplus = g.op.Softplus(x, name=name)
    if beta != 1:
        softplus = g.op.Div(softplus, bnp, name=name)
    res = g.op.Where(
        g.op.Greater(x, np.array([threshold], dtype=dtype), name=name),
        x,
        softplus,
        name=name,
        outputs=outputs,
    )
    if not sts:
        set_type_shape_unary_op(g, res, x)
    return res


def aten_split_Tensor(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    split_sizes: T,
    dim: int = 0,
    name: str = "split_Tensor",
) -> Tuple[T, ...]:
    "split_to_sequence or split"
    return aten_split_with_sizes(g, sts, outputs, x, split_sizes, dim, name=name)


def aten_split_with_sizes(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    split_sizes: T,
    dim: int = 0,
    name: str = "split_with_sizes",
    use_sequence: bool = False,
) -> Tuple[T, ...]:
    "split_to_sequence or split"
    if not use_sequence and isinstance(split_sizes, int):
        # Then torch means to split into equal chunk of this size
        assert g.has_shape(x), (
            f"Not implemented when split_sizes={split_sizes} "
            f"and x has not known shape "
            f"{g.get_debug_msg()}"
        )
        shape = g.get_shape(x)
        size = shape[dim]
        assert isinstance(size, int), (
            f"Not implemented when split_sizes={split_sizes} "
            f"and x has dynamic shape {shape} "
            f"{g.get_debug_msg()}"
        )
        n_splits = (
            (size // split_sizes) if size % split_sizes == 0 else (size // split_sizes + 1)
        )
        if len(outputs) == 1:
            o = outputs[0]
            outputs = [f"{o}#{i}" for i in range(n_splits)]
        res = g.make_node("Split", [x], outputs, axis=dim, num_outputs=n_splits, name=name)
        if not sts:
            new_shapes = []
            new_ranks = []
            shape = g.get_shape(x)
            new_shape = list(shape)
            for _ in range(n_splits):
                s = min(size - split_sizes, split_sizes)
                new_shape[dim] = s
                size -= split_sizes
                new_shapes.append(tuple(new_shape))
            t = g.get_type(x)
            for i, o in enumerate(res):
                g.set_type(o, t)
                if new_shapes:
                    g.get_shape(o, new_shape[i])
                else:
                    g.get_rank(o, new_ranks[i])
        return res

    assert isinstance(split_sizes, list) and all_int(split_sizes), (
        f"Not implemented when split_sizes ({split_sizes}) "
        f"is a constant{g.get_debug_msg()}"
    )
    assert isinstance(dim, int), f"dim={dim} is not an integer{g.get_debug_msg()}"
    assert all(d > 0 for d in split_sizes), (
        f"split_with_sizes only implemented when all sizes are positive "
        f"but split_sizes={split_sizes}{g.get_debug_msg()}"
    )
    assert len(outputs) in (1, len(split_sizes)), (
        f"Number of outputs is unexpected, outputs={outputs}, "
        f"split_sizes={split_sizes}{g.get_debug_msg()}"
    )
    init = g.make_initializer("", np.array(split_sizes, dtype=np.int64))
    if use_sequence:
        res = g.make_node("SplitToSequence", [x, init], outputs, axis=dim, name=name)
        if not sts:
            if g.has_shape(x):
                new_shapes = []
                shape = g.get_shape(x)
                new_shape = list(shape)
                for s in split_sizes:
                    new_shape[dim] = s
                    new_shapes.append(tuple(new_shape))
                g.set_sequence(res, g.get_type(x), shapes=new_shapes)
            else:
                r = g.get_rank(x)
                g.set_sequence(res, g.get_type(x), types=[r for o in split_sizes])
        return res

    # Split directly as tensors.
    if len(outputs) == 1:
        o = outputs[0]
        outputs = [f"{o}#{i}" for i, _ in enumerate(split_sizes)]
    res = g.make_node("Split", [x, init], outputs, axis=dim, name=name)
    if not sts:
        new_shapes = []
        new_ranks = []
        if g.has_shape(x):
            shape = g.get_shape(x)
            new_shape = list(shape)
            for s in split_sizes:
                new_shape[dim] = s
                new_shapes.append(tuple(new_shape))
        else:
            r = g.get_rank(x)
            new_ranks = [r for s in split_sizes]
        t = g.get_type(x)
        for i, o in enumerate(res):
            g.set_type(o, t)
            if new_shapes:
                g.get_shape(o, new_shape[i])
            else:
                g.get_rank(o, new_ranks[i])
    return res


def aten_sqrt(g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T) -> T:
    "sqrt"
    res = g.make_node("Sqrt", [x], name="sqrt")
    if not sts:
        set_type_shape_unary_op(g, outputs[0], x)
    return res


def aten__sym_sqrt(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    name: str = "_sym_sqrt",
) -> T:
    "symbolic sqrt"
    assert g.has_type(x), f"Missing type for {x!r}{g.get_debug_msg()}"
    itype = g.get_type(x)
    if itype == TensorProto.INT64:
        res = g.op.Sqrt(
            g.op.Cast(x, to=TensorProto.FLOAT, name=name),
            name=name,
            outputs=outputs,
        )
        if not sts:
            set_type_shape_unary_op(g, res, x, itype=TensorProto.FLOAT)
    else:
        assert (
            itype == TensorProto.FLOAT
        ), f"Unexpected type {itype} for {x!r}{g.get_debug_msg()}"
        res = g.op.Sqrt(x, name=name, outputs=outputs)
        if not sts:
            set_type_shape_unary_op(g, res, x)
    return res


def aten_squeeze(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    name="squeeze",
) -> T:
    "squeeze"
    return g.op.SqueezeAnyOpset(x, name=name)


def aten_squeeze_dim(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    dim: int,
    name="squeeze",
) -> T:
    "squeeze_dim"
    return g.op.SqueezeAnyOpset(x, np.array([dim], dtype=np.int64), name=name)


def aten_stack(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    tensors: Sequence[T],
    dim: int = 0,
    name: str = "stack",
) -> T:
    """concat"""
    new_tensors = []
    adim = g.make_initializer("", np.array([dim], dtype=np.int64))
    for t in tensors:
        r = g.op.UnsqueezeAnyOpset(t, adim, name=name)
        new_tensors.append(r)
    res = g.op.Concat(*new_tensors, axis=dim, outputs=outputs, name=name)
    if not sts:
        g.set_type(res, g.get_type(tensors[0]))
    return res


def aten_std_dim(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    dims: Sequence[int],
    correction: float,
    keepdim: bool = False,
    name: str = "std_dim",
) -> T:
    "std_dim"
    assert isinstance(dims, list) and all_int(
        dims
    ), f"Unexpected value for dims={dims!r}{g.get_debug_msg()}"
    cdims = np.array(dims, dtype=np.int64)

    mean = g.op.ReduceMeanAnyOpset(x, cdims, name=name)
    sub_mean = g.op.Sub(x, mean, name=name)
    sqr_mean = g.op.Mul(sub_mean, sub_mean, name=name)
    var = g.op.ReduceMeanAnyOpset(sqr_mean, cdims, keepdims=1 if keepdim else 0, name=name)

    if correction > 0:
        assert g.has_shape(
            x
        ), f"not implemented if shape of x={x!r} is missing{g.get_debug_msg()}"
        shape = g.get_shape(x)
        assert is_static_shape(
            shape
        ), f"not implemented for shape={shape!r} for x={x!r}{g.get_debug_msg()}"
        dim_size = np.array(shape)[cdims]
        itype = g.get_type(x)
        dtype = tensor_dtype_to_np_dtype(itype)
        numel = np.prod(dim_size).astype(dtype)
        mul = g.op.Mul(var, numel, name=name)
        sub = g.op.Sub(numel, np.array([correction], dtype=dtype), name=name)
        var = g.op.Div(mul, sub, name=name)

    res = g.op.Sqrt(var, name=name)
    if not sts:
        g.set_type(res, g.get_type(x))
    return res


def aten_sub(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    y: T,
    name="sub",
) -> T:
    "sub"
    x, y = prepare_inputs_homogeneous_operator(g, x, y)
    res = g.op.Sub(x, y, outputs=outputs, name=name)
    if not sts:
        set_type_shape_binary_op(g, res, x, y)
    return res


def aten_sub_Tensor(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    y: T,
    alpha: float,
) -> T:
    "sub"
    assert alpha == 1, f"sub_Tensor not implemented for alpha={alpha}{g.get_debug_msg()}"
    return aten_sub(g, sts, outputs, x, y, name="sub_Tensor")


def aten_sum(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    dim: Optional[Union[int, List[int]]] = None,
    keepdim: bool = False,
    dtype: Optional["torch.dtype"] = None,  # noqa: F821
    name="sum",
) -> T:
    "reducesum"
    assert g.has_type(x), f"sum: type of {x!r} must be known{g.get_debug_msg()}"
    if dtype is None and g.get_type(x) == TensorProto.BOOL:
        dtype = g.torch.int64
    if dtype is not None:
        itype = torch_dtype_to_onnx_dtype(dtype)
        xc = g.op.Cast(x, to=itype)
    else:

        xc = x
    if dim is None:
        result = g.op.ReduceSumAnyOpset(xc, keepdims=keepdim, outputs=outputs, name=name)
    else:
        if isinstance(dim, int):
            adim = np.array([dim], dtype=np.int64)
        else:
            adim = np.array(dim, dtype=np.int64)
        result = g.op.ReduceSumAnyOpset(xc, adim, keepdims=keepdim, outputs=outputs, name=name)
    if not sts:
        set_type_shape_reduce_op(g, outputs[0], x, keepdim=keepdim)
    return result


def aten_sum_dim_IntList(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    dim: Optional[Union[int, List[int]]],
    keepdim: bool,
    dtype: Optional["torch.dtype"] = None,  # noqa: F821
) -> T:
    "reducesum"
    if dtype is None:
        return aten_sum(g, sts, outputs, x, dim, keepdim)

    res = aten_sum(g, sts, None, x, dim, keepdim)
    itype = torch_dtype_to_onnx_dtype(dtype)
    result = g.op.Cast(res, to=itype, outputs=outputs, name="sum_dim_IntList")
    return result


def aten_sym_size_int(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    dim: int,
    name: str = "sym_size_int",
) -> T:
    """
    Shape + Gather
    """
    assert (
        g.main_opset >= 15
    ), f"aten_sym_size_int is not implemented for opset < 15{g.get_debug_msg()}"
    assert isinstance(dim, int), f"type(dim)={type(int)} must be an int{g.get_debug_msg()}"
    res = g.op.Shape(x, name=name, start=dim, end=dim + 1)
    if not sts:
        g.set_type(res, TensorProto.INT64)
        g.set_shape(
            res,
            tuple(
                1,
            ),
        )
    return res


def aten__to_copy(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    dtype: Optional["torch.dtype"] = None,  # noqa: F821
    layout=None,
    device: Optional["torch.device"] = None,  # noqa: F821
    pin_memory=None,
    non_blocking=False,
    memory_format=None,
) -> T:
    "identity"
    import torch

    assert layout in (
        None,
        torch.strided,
    ), f"_to_copy implemented with layout={layout!r}"
    # assert device is None, f"_to_copy implemented with device={device!r}"
    assert pin_memory is None, f"_to_copy implemented with pin_memory={pin_memory!r}"
    assert not non_blocking, f"_to_copy implemented with non_blocking={non_blocking!r}"
    assert memory_format is None, f"_to_copy implemented with memory_format={memory_format!r}"
    if dtype is None:
        return g.op.Identity(x, outputs=outputs, name="_to_copy")
    itype = torch_dtype_to_onnx_dtype(dtype)
    assert (
        isinstance(itype, int) and itype > 0
    ), f"Unexpected value for itype={itype}, dtype={dtype}"
    res = g.op.Cast(x, to=itype, outputs=outputs, name="_to_copy")
    if not sts:
        set_type_shape_unary_op(g, res, x, itype=itype)
    return res


def aten_t(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    name: str = "t",
) -> T:
    "transpose"
    res = g.op.Transpose(x, perm=[1, 0], outputs=outputs, name=name)
    if not sts:
        g.set_type(res, g.get_type(x))
        if g.has_shape(x):
            shape = list(g.get_shape(x))
            assert len(shape) == 2, f"Unexpected shape={shape}, should be 2D"
            shape[0], shape[1] = shape[1], shape[0]
            g.set_shape(res, tuple(shape))
        else:
            g.set_rank(res, g.get_rank(x))
    return res


def aten_tan(g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T) -> T:
    "tan"
    res = g.make_node("Tan", [x], outputs, name="tan")
    if not sts:
        set_type_shape_unary_op(g, outputs[0], x)
    return res


def aten_tanh(g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T) -> T:
    "tanh"
    res = g.make_node("Tanh", [x], outputs, name="tanh")
    if not sts:
        set_type_shape_unary_op(g, outputs[0], x)
    return res


def aten_tanh_backward(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    out_grad: T,
    y: T,
) -> T:
    "tanh backward"
    # out_grad * (1 - y * y)
    dtype = tensor_dtype_to_np_dtype(g.get_type(y))
    yy = g.op.Pow(y, np.array([2], dtype=np.int64), name="tanh_backward")
    _1yy = g.op.Sub(np.array([1], dtype=dtype), yy, name="tanh_backward")
    res = g.op.Mul(out_grad, _1yy, outputs=outputs, name="tanh_backward")

    set_type_shape_unary_op(g, yy, y)
    set_type_shape_unary_op(g, _1yy, y)
    if not sts:
        set_type_shape_unary_op(g, res, y)
    return res


def _aten_tensor_int1(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    input_name: T,
    indices: Tuple[Any, ...],
    axes: List[int],
    expand_axes: List[int],
    name: str = "_aten_tensor_int1",
) -> T:
    assert isinstance(axes, list), f"Unexpected type {type(axes)} for axes"
    assert all_int(axes), f"Expected only integer axis but got {axes}"
    assert all_int(indices), f"Expected only integer axis but got {indices}"
    assert len(axes) == 1, f"Length mismatch {len(axes)} != 1"

    # axes
    indices_name = g.unique_name(f"{outputs[0]}_indices")
    g.make_initializer(indices_name, np.array(indices, dtype=np.int64))

    res = g.make_node(
        "Gather",
        [input_name, indices_name],
        outputs=outputs,
        axis=axes[0],
        name=name,
    )

    if expand_axes:
        raise RuntimeError(f"Not implemented when expand_axes={expand_axes}.")
    if not sts:
        dtype = g.get_type(input_name)
        g.set_type(res, dtype)
        if g.has_shape(input_name):
            shape = g.get_shape(input_name)
            new_shape = g._apply_slice_to_shape(
                shape, indices, axes=axes, expand_axes=expand_axes
            )
            if g.has_shape(outputs[0]) and new_shape != g.get_shape(outputs[0]):
                raise RuntimeError(
                    f"Shape for node {res!r} is already set to "
                    f"{g.get_shape(res)} with type "
                    f"{g.get_type(res)} (expecting {dtype}) "
                    f"new_shape={new_shape}, shape={shape}, index_slice={indices}, "
                    f"axes={axes}, expand_axes={expand_axes}"
                    f"{g.get_debug_msg()}"
                )
            g.set_shape(res, new_shape)
        else:
            g.set_rank(res, g.get_rank(input_name))
    return res


def aten_tensor(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    indices: Optional[Tuple[Any, ...]] = None,
) -> T:
    "[..., :, ...]"
    if indices is None:
        # x is some data to convert into a Tensor
        if isinstance(x, list) and all_int_or_float(x):
            if all_int(x):
                cst = np.array(x, dtype=np.int64)
            elif all_float(x):
                cst = np.array(x, dtype=np.float32)
            else:
                raise RuntimeError(
                    f"Unable to convert to guess value dtype for x={x}{g.get_debug_msg()}"
                )

            return g.make_initializer(outputs[0], cst)
        raise RuntimeError(
            f"Unable to convert a value into a tensor, x={x}{g.get_debug_msg()}"
        )

    if isinstance(indices, tuple) and len(indices) == 1:
        if isinstance(indices[0], list) and all_int(indices[0]):
            return _aten_tensor_int1(g, sts, outputs, x, indices, [0], [])
    raise RuntimeError(f"Unable to handle getitem with indices={indices}{g.get_debug_msg()}")


def aten_threshold_backward(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    grad_output: T,
    x: T,
    threshold: float,
    name: str = "threshold_backward",
) -> T:
    "lessorequal"
    dtype = tensor_dtype_to_np_dtype(g.get_type(grad_output))
    le = g.op.LessOrEqual(x, np.array([threshold], dtype=dtype), name=name)
    res = g.op.Where(le, np.array([0], dtype=dtype), grad_output, outputs=outputs, name=name)
    set_type_shape_unary_op(g, le, x, TensorProto.BOOL)
    if not sts:
        set_type_shape_unary_op(g, res, x)
    return res


def aten_transpose(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    input_name: T,
    dim0: int,
    dim1: int,
) -> T:
    "transpose"
    perm = list(range(g.rank(input_name)))
    perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
    res = g.make_node("Transpose", [input_name], outputs, perm=perm, name="transpose")
    if not sts:
        g.set_type(outputs[0], g.get_type(input_name))
        if g.has_shape(input_name):
            shape = list(g.get_shape(input_name))
            shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
            g.set_shape(outputs[0], tuple(shape))
        elif g.has_rank(input_name):
            g.set_rank(outputs[0], g.has_rank(input_name))
    return res


def aten_transpose_int(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    input_name: T,
    dim0: int,
    dim1: int,
) -> T:
    "transpose"
    return aten_transpose(g, sts, outputs, input_name, dim0, dim1)


def aten_tril(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    diagonal: int = 0,
) -> T:
    """tril"""

    if diagonal == 0:
        res = g.op.Trilu(x, upper=0, outputs=outputs)
    else:
        res = g.op.Trilu(x, np.array(diagonal, dtype=np.int64), upper=0, outputs=outputs)
    if not sts:
        set_type_shape_unary_op(g, res, x)
    return res


def aten_truediv(
    g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T, y: T
) -> T:
    "truediv"
    x, y = prepare_inputs_homogeneous_operator(g, x, y)
    res = g.op.Div(x, y, outputs=outputs, name="truediv")
    if not sts:
        set_type_shape_binary_op(g, outputs[0], x, y)
    return res


def aten_triu(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    diagonal: Union[int, T] = 0,
) -> T:
    """trilu"""
    if isinstance(diagonal, int):
        k = np.array(diagonal, dtype=np.int64)
    elif isinstance(diagonal, str):
        k = diagonal
    elif isinstance(diagonal, g.torch.Tensor):
        assert tuple(diagonal.shape) in (
            tuple(),
            (1,),
        ), f"Unexpected for diagonal={diagonal}{g.get_debug_msg()}"
        value = int(
            diagonal.cpu().numpy() if len(diagonal.shape) == 0 else diagonal.cpu().numpy()[0]
        )
        k = np.array(value, dtype=np.int64)
    else:
        raise AssertionError(
            f"triu: unexpected type={type(diagonal)} for diagonal{g.get_debug_msg()}"
        )
    res = g.op.Trilu(x, k, upper=1, name="triu", outputs=outputs)
    if not sts:
        set_type_shape_unary_op(g, res, x)
    return res


def aten_type_as(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    other: int,
    name: str = "type_as",
) -> T:
    """castlike"""
    if g.get_type(x) == g.get_type(other):
        return g.op.Identity(x, name=name, outputs=outputs)
    # res = g.op.CastLike(x, other, name=name, outputs=outputs)
    res = g.op.Cast(x, to=g.get_type(other), name=name, outputs=outputs)
    if not sts:
        set_type_shape_unary_op(g, res, x, itype=g.get_type(other))
    return res


def aten_unbind_int(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    dim: int = 0,
    use_sequence: bool = False,
    name: str = "unbind",
) -> Tuple[T, ...]:
    """split"""
    assert not use_sequence, f"Not implemented for use_sequence={use_sequence}"
    assert g.has_shape(
        x
    ), f"Not implemented when the input {x!r} has no shape{g.get_debug_msg()}"
    shape = g.get_shape(x)
    assert isinstance(shape[dim], int), (
        f"Not implemented when shape on this axis is dynamic, "
        f"name={name!r}, shape={shape!r}"
    )

    if shape[dim] == 1:
        return g.op.Identity(x, outputs=outputs, name=name)

    if len(outputs) == 1:
        o = outputs[0]
        unbind_outputs = [f"{o}_u{i}" for i in range(shape[dim])]
        new_outputs = [g.unique_name(f"{o}#{i}") for i in range(shape[dim])]
    else:
        unbind_outputs = [g.unique_name(f"{o}{i}") for i, o in enumerate(outputs)]
        new_outputs = outputs

    g.make_node("Split", [x], unbind_outputs, axis=dim, num_outputs=shape[dim], name=name)
    dim_np = g.make_initializer("", np.array([dim], dtype=np.int64))
    for o, u in zip(new_outputs, unbind_outputs):
        g.make_node("Squeeze", [u, dim_np], [o], name=name)
    res = new_outputs
    if not sts:
        shape = g.get_shape(x)
        new_shape = list(shape)
        del new_shape[dim]
        t = g.get_type(x)
        for o in res:
            g.set_type(o, t)
            g.get_shape(o, new_shape)
    return res


def aten_unfold(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    dimension: int,
    size: int,
    step: int,
    name: str = "unfold",
) -> T:
    "unfold"
    assert g.has_shape(x), f"Not implemented when x={x!r} has no shape{g.get_debug_msg()}"
    shape = g.get_shape(x)
    sizedim = shape[dimension]
    assert is_static_dimension(sizedim), (
        f"Dynamic shape is not implemented for x={x!r} and shape={shape!r}"
        f"{g.get_debug_msg()}"
    )

    stack = [
        g.op.Slice(
            x,
            np.array([low], dtype=np.int64),
            np.array([hi], dtype=np.int64),
            np.array([dimension], dtype=np.int64),
            name=name,
        )
        for low, hi in zip(range(0, sizedim, step), range(size, sizedim + 1, step))
    ]
    rk = len(shape)
    perm = list(range(rk))
    perm.append(perm.pop(dimension))
    unsqueeze = [
        g.op.UnsqueezeAnyOpset(
            g.op.Transpose(t, perm=perm, name=name),
            np.array([dimension], dtype=np.int64),
            name=name,
        )
        for t in stack
    ]
    if len(unsqueeze) == 1:
        return g.op.Identity(unsqueeze[0], name=name, outputs=outputs)
    return g.op.Concat(*unsqueeze, axis=dimension, name=name, outputs=outputs)


def aten_unsqueeze(
    g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T, dim: int
) -> T:
    "unsqueeze"
    assert isinstance(dim, int), f"Not implemented for dim={dim!r}"
    res = g.op.UnsqueezeAnyOpset(x, np.array([dim], dtype=np.int64), outputs=outputs)
    if not sts:
        g.set_type(res, g.get_type(x))
        if g.has_shape(x):
            shape = list(g.get_shape(x))
            shape.insert(dim, 1)
            g.set_shape(res, tuple(shape))
        else:
            g.set_rank(res, g.get_rank(x) + 1)
    return res


def _aten_upsample_output_size(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    output_size: T,
    mode: str,
    coordinate_transformation_mode: str,
    d: int,
    name: str = "upsample_output_size",
) -> T:
    batch_channel = None
    if g.has_shape(x):
        shape = g.get_shape(x)
        if is_static_shape(shape):
            batch_channel = g.make_initializer("", np.array(shape[:2], dtype=np.int64))
    if batch_channel is None:
        batch_channel = g.op.Shape(x, start=0, end=2, name=name)
    if isinstance(output_size, (tuple, list)):
        assert is_static_shape(output_size), f"output_size={output_size} must be static"
        rsize = g.make_initializer("", np.array(output_size, dtype=np.int64))
    else:
        assert isinstance(
            output_size, str
        ), f"Unexpected type {type(output_size)} for output_size"
        rsize = output_size
    new_output_size = g.op.Concat(batch_channel, rsize, axis=0, name=name)
    res = g.op.Resize(
        x,
        None,
        None,
        new_output_size,
        mode=mode,
        coordinate_transformation_mode=coordinate_transformation_mode,
        nearest_mode="floor",
        outputs=outputs,
        name=name,
    )
    if not sts:
        g.set_type(res, g.get_type(x))
        g.set_rank(res, g.get_rank(x))
    return res


def aten_upsample_nearest2d(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    output_size: T,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
    name: str = "upsample_nearest2d",
) -> T:
    """resize"""
    assert output_size is not None, "Not implemented when size is None"
    assert scales_h is None, f"Not impelmented when scales_h={scales_h}"
    assert scales_w is None, f"Not impelmented when scales_h={scales_w}"

    return _aten_upsample_output_size(
        g,
        sts,
        outputs,
        x,
        output_size,
        mode="nearest",
        coordinate_transformation_mode="asymmetric",
        d=2,
        name=name,
    )


def aten_upsample_nearest3d(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    output_size: T,
    scales_d: Optional[float] = None,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
    name: str = "upsample_nearest3d",
) -> T:
    """resize"""
    assert output_size is not None, "Not implemented when size is None"
    assert scales_d is None, f"Not impelmented when scales_h={scales_h}"
    assert scales_h is None, f"Not impelmented when scales_h={scales_h}"
    assert scales_w is None, f"Not impelmented when scales_h={scales_w}"

    return _aten_upsample_output_size(
        g,
        sts,
        outputs,
        x,
        output_size,
        mode="nearest",
        coordinate_transformation_mode="asymmetric",
        d=3,
        name=name,
    )


def _upsample_compute_output_size(input_size, output_size, scale_factors):
    spatial_dimensions = len(input_size) - 2
    if output_size is not None:
        assert scale_factors is None, f"scale_factors={scale_factors}"
        assert len(output_size) == spatial_dimensions, f"output_size={output_size}"
        return output_size
    if scale_factors is not None:
        assert output_size is None, f"scale_factors={scale_factors}, output_size={output_size}"
        assert len(scale_factors) == spatial_dimensions, f"output_size={scale_factors}"
        output_size = []
        for i, s in enumerate(scale_factors):
            assert isinstance(
                s, (int, float)
            ), f"Not implemented when a shape is dynamic, scale_factors={scale_factors}"
            output_size.append(input_size[i + 2] * int(s))
        return output_size
    raise AssertionError("Either scale_factors or output_size must be specified.")


def aten_upsample_bicubic2d(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    output_size: T,
    align_corners: bool,
    scales_d: Optional[float] = None,
    scales_h: Optional[float] = None,
    name: str = "upsample_bicubic2d",
) -> T:
    """resize"""
    assert output_size is not None, "Not implemented when size is None"
    assert scales_d is None, f"Not impelmented when scales_h={scales_h}"
    assert scales_h is None, f"Not impelmented when scales_h={scales_h}"

    return _aten_upsample_output_size(
        g,
        sts,
        outputs,
        x,
        output_size,
        mode="cubic",
        coordinate_transformation_mode=(
            "align_corners" if align_corners else "pytorch_half_pixel"
        ),
        d=2,
        name=name,
    )


def aten_upsample_bilinear2d(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    output_size: T,
    align_corners: bool,
    scales_d: Optional[float] = None,
    scales_h: Optional[float] = None,
    name: str = "upsample_bilinear2d",
) -> T:
    """resize"""
    assert output_size is not None, "Not implemented when size is None"
    assert scales_d is None, f"Not impelmented when scales_h={scales_h}"
    assert scales_h is None, f"Not impelmented when scales_h={scales_h}"

    return _aten_upsample_output_size(
        g,
        sts,
        outputs,
        x,
        output_size,
        mode="linear",
        coordinate_transformation_mode=(
            "align_corners" if align_corners else "pytorch_half_pixel"
        ),
        d=2,
        name=name,
    )


def aten_upsample_bicubic2d_vec(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    output_size: T,
    align_corners: bool,
    scale_factors: Optional[Sequence[float]] = None,
    name: str = "upsample_bicubic2d_vec",
) -> T:
    """resize"""
    assert g.has_shape(x), f"Not implemented when {x!r} has no shape{g.get_debug_msg()}"
    osize = _upsample_compute_output_size(g.get_shape(x), output_size, scale_factors)
    # scales = (
    #     scale_factors if scale_factors else [None] * len(osize)
    # )
    scales = [None, None, None]
    return aten_upsample_bicubic2d(g, sts, outputs, x, osize, *scales, name=name)


def aten_upsample_bilinear2d_vec(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    output_size: T,
    align_corners: bool,
    scale_factors: Optional[Sequence[float]] = None,
    name: str = "upsample_bicubic2d_vec",
) -> T:
    """resize"""
    assert g.has_shape(x), f"Not implemented when {x!r} has no shape{g.get_debug_msg()}"
    osize = _upsample_compute_output_size(g.get_shape(x), output_size, scale_factors)
    # scales = (
    #     scale_factors if scale_factors else [None] * len(osize)
    # )
    scales = [None, None, None]
    return aten_upsample_bilinear2d(g, sts, outputs, x, osize, *scales, name=name)


def aten_upsample_nearest2d_vec(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    output_size: Optional[T] = None,
    scale_factors: Optional[List[int]] = None,
    name: str = "upsample_nearest2d_vec",
) -> T:
    "resize"
    assert g.has_shape(x), f"Not implemented when {x!r} has no shape{g.get_debug_msg()}"
    osize = _upsample_compute_output_size(g.get_shape(x), output_size, scale_factors)
    # scales = (
    #     scale_factors if scale_factors else [None] * len(osize)
    # )
    scales = [None, None]
    return aten_upsample_nearest2d(g, sts, outputs, x, osize, *scales, name=name)


def aten_upsample_nearest3d_vec(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    output_size: Optional[T] = None,
    scale_factors: Optional[List[int]] = None,
    name: str = "upsample_nearest3d_vec",
) -> T:
    "resize"
    assert g.has_shape(x), f"Not implemented when {x!r} has no shape{g.get_debug_msg()}"
    osize = _upsample_compute_output_size(g.get_shape(x), output_size, scale_factors)
    # scales = (
    #     scale_factors if scale_factors else [None] * len(osize)
    # )
    scales = [None, None, None]
    return aten_upsample_nearest3d(g, sts, outputs, x, osize, *scales, name=name)


def aten_upsample_trilinear3d(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    output_size: T,
    align_corners: bool,
    scales_d: Optional[float] = None,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
    name: str = "upsample_trilinear3d",
) -> T:
    """resize"""
    assert output_size is not None, "Not implemented when size is None"
    assert scales_d is None, f"Not impelmented when scales_h={scales_h}"
    assert scales_h is None, f"Not impelmented when scales_h={scales_h}"
    assert scales_w is None, f"Not impelmented when scales_h={scales_w}"

    return _aten_upsample_output_size(
        g,
        sts,
        outputs,
        x,
        output_size,
        mode="linear",
        coordinate_transformation_mode=(
            "align_corners" if align_corners else "pytorch_half_pixel"
        ),
        d=3,
        name=name,
    )


def aten_upsample_trilinear3d_vec(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    output_size: T,
    align_corners: bool,
    scale_factors: Optional[Sequence[float]] = None,
    name: str = "upsample_trilinear3d_vec",
) -> T:
    """resize"""
    assert g.has_shape(x), f"Not implemented when {x!r} has no shape{g.get_debug_msg()}"
    osize = _upsample_compute_output_size(g.get_shape(x), output_size, scale_factors)
    # scales = (
    #     scale_factors if scale_factors else [None] * len(osize)
    # )
    scales = [None, None, None]
    return aten_upsample_trilinear3d(g, sts, outputs, x, osize, *scales, name=name)


def aten_view(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    size: T,
    node_name: str = "view",
) -> T:
    "slice"
    if isinstance(size, (int, tuple, list)):
        asize = [size] if isinstance(size, int) else list(size)
        if is_static_shape(asize):
            asize = np.array(asize, dtype=np.int64)
            assert (
                len(asize.shape) == 1
            ), f"Unexpected shape for view, size={size}{g.get_debug_msg()}"
        elif g.is_dynamic_shape(asize):
            asize = g.make_shape_from_results(asize, name=node_name)
            # TODO: check that the shape is just a number
        else:
            raise RuntimeError(
                f"aten_view not implemented when size={size!r}{g.get_debug_msg()}"
            )
        res = g.op.Reshape(x, asize, outputs=outputs, name=node_name)
        if not sts:
            set_type_shape_reshape(g, res, x, asize)
        return res

    csize = g.op.Cast(size, to=TensorProto.INT64, name=node_name)
    res = g.op.Reshape(x, csize, outputs=outputs, name=node_name)
    if not sts:
        set_type_shape_reshape(g, res, x, size)
    return res


def aten__unsafe_view(
    g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T, size: T
) -> T:
    "slice"
    return aten_view(g, sts, outputs, x, size, node_name="_unsafe_view")


def aten_where(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    condition: T,
    x: T,
    other: T,
    name: str = "where",
) -> T:
    """where"""
    res = g.op.Where(condition, x, other, name=name, outputs=outputs)
    if not sts:
        set_type_shape_binary_op(g, res, condition, x, other, begin=1)
    return res


def aten_where_Scalar(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    condition: T,
    x: T,
    other: T,
    name: str = "where_Scalar",
) -> T:
    """
    where

    This function may introduce some type issues when 'x' and 'other' are both floats.
    Implicit cast may be done by torch. Checks what happens after this node.
    """
    if isinstance(x, float) and isinstance(other, float) and g.get_type_known(outputs[0]):
        itype = g.get_type_known(outputs[0])
        dtype = tensor_dtype_to_np_dtype(itype)
        res = g.op.Where(
            condition,
            np.array(x, dtype=dtype),
            np.array(other, dtype=dtype),
            name=name,
            outputs=outputs,
        )
        if not sts:
            g.set_type(res, itype)
            if g.has_shape(condition):
                g.set_shape(res, g.get_shape(condition))
            elif g.has_rank(condition):
                g.set_rank(res, g.get_rank(condition))
        return res

    assert isinstance(x, str) or isinstance(other, str), (
        f"aten_where not implemented when both constants are float, "
        f"x={x}, other={other}{g.get_debug_msg()}"
    )
    return aten_where(g, sts, outputs, condition, x, other, name=name)


def aten_where_self(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    condition: T,
    x: T,
    other: T,
) -> T:
    """where"""
    return aten_where(g, sts, outputs, condition, x, other, name="where_self")


def aten_wrap_with_autocast(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    device_type: str,
    dtype: Optional["torch.dtype"],  # noqa: F821
    enabled: bool,
    cache_enabled: Optional[bool],
    wrapped_func,
    *args: Sequence[T],
    **kwargs,
) -> T:
    "identity, calling a local function"
    # We don't need to check for the type, the module is converted as a function,
    # hopefully this will go smoothly. Maybe here, we should check the local
    # function is valid for the considered dtype.
    # dtype is mentioned in the doc_string and should remain unless the graph
    # is inlined.

    # assert dtype is None, f"Not implemented with dtype={dtype}{g.get_debug_msg()}"
    assert not enabled, f"Not implemented with dtype={enabled}{g.get_debug_msg()}"
    assert (
        not cache_enabled
    ), f"Not implemented with cache_enabled={cache_enabled}{g.get_debug_msg()}"
    assert g.has_local_function(
        wrapped_func, domain=g.local_domain
    ), f"No local function {wrapped_func!r}, domain={g.local_domain!r}\n{g.pretty_text()}"
    assert all(
        isinstance(_, str) for _ in args
    ), f"Unexpected input types args={args}{g.get_debug_msg()}"
    local_outputs = g.get_local_function_outputs(wrapped_func, domain=g.local_domain)
    if len(outputs) == len(local_outputs):
        return g.make_node(
            wrapped_func,
            args,
            outputs,
            name="wrap_with_autocast",
            domain=g.local_domain,
            doc_string=f"wrap_with_autocast(..., dtype={dtype})" if dtype is not None else "",
        )
    assert len(outputs) == 1, (
        f"Unexpected outputs={outputs} but local_outputs={local_outputs} "
        f"for function {wrapped_func!r}{g.get_debug_msg()}"
    )
    new_outputs = [f"{outputs[0]}#{i}" for i in range(len(local_outputs))]
    return g.make_node(
        wrapped_func,
        args,
        new_outputs,
        name="wrap_with_autocast",
        domain=g.local_domain,
    )


def aten_wrap_with_set_grad_enabled(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    enable_grad: bool,
    wrapped_func,
    *args: Sequence[T],
    **kwargs,
) -> T:
    "identity"
    assert (
        not enable_grad
    ), f"Not implemented with enable_grad={enable_grad}{g.get_debug_msg()}"
    assert g.has_local_function(
        wrapped_func, domain=g.local_domain
    ), f"No local function {wrapped_func!r}{g.get_debug_msg()}"
    assert all(
        isinstance(_, str) for _ in args
    ), f"Unexpected input types args={args}{g.get_debug_msg()}"
    local_outputs = g.get_local_function_outputs(wrapped_func, domain=g.local_domain)
    if len(outputs) == len(local_outputs):
        return g.make_node(
            wrapped_func,
            args,
            outputs,
            name="wrap_with_set_grad_enabled",
            domain=g.local_domain,
        )
    assert len(outputs) == 1, (
        f"Unexpected outputs={outputs} but local_outputs={local_outputs} "
        f"for function {wrapped_func!r}{g.get_debug_msg()}"
    )
    new_outputs = [f"{outputs[0]}#{i}" for i in range(len(local_outputs))]
    return g.make_node(
        wrapped_func,
        args,
        new_outputs,
        name="wrap_with_set_grad_enabled",
        domain=g.local_domain,
    )


def aten_zero(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    dtype: Optional["torch.dtype"] = None,  # noqa: F821
    layout=None,
    device: Optional["torch.device"] = None,  # noqa: F821
    pin_memory=None,
    memory_format: Optional[str] = None,
    name: str = "zero",
) -> T:
    "constantofshape"
    return aten_zeros_like(
        g,
        sts,
        outputs,
        x,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        name=name,
    )


def aten_zeros(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    size: T,
    dtype: Optional["torch.dtype"] = None,  # noqa: F821
    layout=None,
    device: Optional["torch.device"] = None,  # noqa: F821
    pin_memory=None,
    requires_grad: bool = False,
    name: str = "zeros",
) -> T:
    "constantofshape"
    return aten_full(
        g,
        sts,
        outputs,
        size,
        fill_value=None,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        requires_grad=requires_grad,
        name=name,
    )


def aten_zeros_like(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    dtype: Optional["torch.dtype"] = None,  # noqa: F821
    layout=None,
    device: Optional["torch.device"] = None,  # noqa: F821
    pin_memory=None,
    memory_format: Optional[str] = None,
    name: str = "zeros_like",
) -> T:
    "constantofshape"
    assert (
        memory_format is None
    ), f"unexpected value for memory_format={memory_format}{g.get_debug_msg()}"
    return aten_full_like(
        g,
        sts,
        outputs,
        x,
        fill_value=None,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        name=name,
    )
