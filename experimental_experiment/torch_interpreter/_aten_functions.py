from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from onnx import TensorProto
from onnx.helper import tensor_dtype_to_np_dtype, make_tensor
from onnx.numpy_helper import from_array
from ..xbuilder.shape_helper import (
    all_float,
    all_int,
    all_int_or_float,
    is_static_dimension,
    is_static_shape,
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


def aten_abs(
    g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T
) -> T:
    "abs"
    res = g.make_node("Abs", [x], outputs, name="abs")
    if not sts:
        set_type_shape_unary_op(g, outputs[0], x)
    return res


def aten_acos(
    g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T
) -> T:
    "acos"
    res = g.make_node("Acos", [x], outputs, name="acos")
    if not sts:
        set_type_shape_unary_op(g, outputs[0], x)
    return res


def aten_acosh(
    g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T
) -> T:
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


def aten_alias(
    g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T
) -> T:
    "identity"
    return g.make_node("Identity", [x], outputs, name="alias")


def aten_all(
    g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T
) -> T:
    "cast"
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

    return prims_amax(
        g, sts, outputs, x, dim, keepdim, output_dtype=output_dtype, name=name
    )


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
    assert (
        not pin_memory
    ), f"arange not implemented for pin_memory=True{g.get_debug_msg()}"
    assert (
        not requires_grad
    ), f"arange not implemented when requires_grad is True{g.get_debug_msg()}"
    if start is not None and end is None:
        end = start
        start = 0

    if dtype is None:
        import torch
        from torch._prims_common import IntLike

        # coming from function arange in torch/_refs.__init__.py
        args = (start, end, step)
        dt = torch.int64
        for a in args:
            if isinstance(a, IntLike):
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
        _may_cast(start, itype)
        if isinstance(start, str)
        else np.array(start, dtype=npdtype)
    )
    i_end = (
        _may_cast(end, itype) if isinstance(end, str) else np.array(end, dtype=npdtype)
    )
    i_step = (
        _may_cast(step, itype)
        if isinstance(step, str)
        else np.array(step, dtype=npdtype)
    )

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
    assert False, f"The implementation is still incorrect{g.get_debug_msg()}"

    import torch
    from torch.fx.experimental.proxy_tensor import maybe_disable_fake_tensor_mode

    assert g.has_shape(
        x
    ), f"not implemented when shape of x is not known{g.get_debug_msg()}"
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


def aten_asin(
    g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T
) -> T:
    "asin"
    res = g.make_node("Asin", [x], outputs, name="asin")
    if not sts:
        set_type_shape_unary_op(g, outputs[0], x)
    return res


def aten_asinh(
    g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T
) -> T:
    "asinh"
    res = g.make_node("Asinh", [x], outputs, name="asinh")
    if not sts:
        set_type_shape_unary_op(g, outputs[0], x)
    return res


def aten_atan(
    g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T
) -> T:
    "atan"
    res = g.make_node("Atan", [x], outputs, name="atan")
    if not sts:
        set_type_shape_unary_op(g, outputs[0], x)
    return res


def aten_atanh(
    g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T
) -> T:
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

    kernel_shape = (
        [kernel_size] * expand_size if isinstance(kernel_size, int) else kernel_size
    )

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
        name="avg_pool2d",
    )

    if sts:
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
    assert not kwargs, (
        f"avg_pool2d_backward not implemented for kwargs="
        f"{kwargs}{g.get_debug_msg()}"
    )

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
        # domain="com.microsoft",
        name="avg_pool2d_backward",
    )
    g.set_type(grad, g.get_type(x))
    g.set_rank(grad, g.get_rank(x))

    result = g.op.Add(x, grad, name="avg_pool2d_backward", outputs=outputs)
    if sts:
        g.set_type(result, g.get_type(x))
        g.set_rank(result, g.get_rank(x))
    return result


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
    assert len(tensors) > 1, f"Not enough tensors to concat{g.get_debug_msg()}"
    res = g.op.Concat(*tensors, axis=dim, outputs=outputs, name="cat")
    if not sts:
        dt0 = g.get_type(tensors[0])
        assert all(map(lambda t: g.get_type(t) == dt0, tensors))
        r0 = g.get_rank(tensors[0])
        assert all(map(lambda t: g.get_rank(t) == r0, tensors))
        g.set_type(outputs[0], dt0)
        g.set_rank(outputs[0], r0)
    return res


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


def aten_convolution(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    input: T,
    weight: T,
    bias: T = None,
    stride: Sequence[int] = (1,),
    padding: Sequence[int] = (0,),
    dilation: Sequence[int] = (1,),
    transposed: bool = False,
    output_padding: Sequence[int] = (0,),
    groups: int = 1,
) -> T:
    "conv"
    if transposed:
        raise FunctionNotFoundError(
            f"aten_convolution does not support transposed={transposed}."
        )
    if output_padding and (min(output_padding) != 0 or max(output_padding) != 0):
        raise FunctionNotFoundError(
            f"aten_convolution does not support output_padding={output_padding}."
        )
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
            weight_dim_0 = g.make_node("Shape", [weight], start=0, end=1, name="conv")
        elif g.main_opset >= 13:
            shape = g.op.Shape(weight, name="conv")
            weight_dim_0 = g.op.Slice(
                shape,
                np.array([0], dtype=np.int64),
                np.array([1], dtype=np.int64),
                np.array([0], dtype=np.int64),
                name="conv",
            )
        else:
            shape = g.op.Shape(weight, name="conv")
            weight_dim_0 = g.op.Slice(
                shape, axes=[0], starts=[0], ends=[1], name="conv"
            )
        cst1 = g.make_initializer("", np.array([1], dtype=np.int64))
        bias_shape = g.make_node("Expand", [weight_dim_0, cst1], name="conv")
        dtype = tensor_dtype_to_np_dtype(g.get_type(input))
        bias = g.op.Expand(np.array([0.0], dtype=dtype), bias_shape, name="conv")

    # if Rank(input) != Rank(weight):
    #    input = op.UnsqueezeAnyOpset(input, op.Constant(value_ints=[0]))

    res = g.make_node(
        "Conv",
        [input, weight, bias],
        outputs,
        strides=strides,
        pads=pads,
        group=groups,
        dilations=dilations,
        name="conv",
    )
    if not sts:
        g.set_type(res, g.get_type(input))
        g.set_rank(res, g.get_rank(input))
    return res


def aten_conv2d(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    input: T,
    weight: T,
    bias: T = None,
    stride: Sequence[int] = (1, 1),
    padding: Sequence[int] = (0, 0),
    dilation: Sequence[int] = (1, 1),
    groups: int = 1,
) -> T:
    "conv"
    return aten_convolution(
        g,
        sts,
        outputs,
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
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


def aten_cosh(
    g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T
) -> T:
    "cosh"
    res = g.make_node("Cosh", [x], outputs, name="cosh")
    if not sts:
        set_type_shape_unary_op(g, outputs[0], x)
    return res


def aten_detach(
    g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T
) -> T:
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
        p = np.array(p, dtype=np.float64)
    if isinstance(training, bool):
        training = np.array(training, dtype=np.bool_)
    result, _ = g.op.Dropout(x, p, training, outputs=outputs)
    if not sts:
        set_type_shape_unary_op(g, outputs[0], x)
    return result


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
    assert (
        not inplace
    ), f"inplace computation is not allowed with onnx{g.get_debug_msg()}"
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
) -> T:
    "embedding"
    if (
        (padding_idx is not None and padding_idx >= 0)
        or scale_grad_by_freq
        or sparse
        or max_norm is not None
    ):
        raise NotImplementedError(
            f"Not implemented when padding_idx={padding_idx}, or "
            f"scale_grad_by_freq={scale_grad_by_freq} or sparse={sparse} "
            f"or max_norm={max_norm} or norm_type={norm_type} "
            f"are different from the default values."
        )
    assert g.get_type(indices) == 7, (
        f"indices be integer not {g.get_type(indices)}, "
        f"weight is {g.get_type(weight)}"
        f"{g.get_debug_msg()}"
    )

    res = g.op.Gather(weight, indices, outputs=outputs, name="embedding")
    if not sts:
        g.set_type(res, g.get_type(weight))
        g.set_rank(res, g.get_rank(weight) + g.get_rank(indices) - 1)
    return res


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
    assert all(map(lambda x: not isinstance(x, str) or x in {"cpu", "cuda"}, args)), (
        f"The function should not take any tensors as input but types are "
        f"{[type(_) for _ in args]}: {args}{g.get_debug_msg()}"
    )
    return g.make_node("Constant", [], value_floats=[0], name="_enter_autocast")


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
        res = g.op.Expand(
            x, np.array(sizes, dtype=np.int64), outputs=outputs, name=name
        )
        if not sts:
            g.set_type(res, g.get_type(x))
            g.set_shape(res, tuple(sizes))
        return res

    if not isinstance(sizes, str) and g.has_shape(x):
        shape = g.get_shape(x)
        assert len(shape) == len(
            sizes
        ), f"Unexpected shape={shape} for x as sizes={sizes}{g.get_debug_msg()}"
        new_shape = []
        is_static = True
        for a, b in zip(shape, sizes):
            if b == -1:
                assert isinstance(b, int), (
                    f"Not implemented when the shape is not fully known, "
                    f"shape={shape} for x as sizes={sizes}{g.get_debug_msg()}"
                )
                new_shape.append(a)
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
    g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T, v: T
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
            name="fill_Scalar",
        )
    raise RuntimeError(
        f"fill is not implemented when shape is not fully known{g.get_debug_msg()}"
    )


def aten_flatten(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    start_dim: int = 1,
    end_dim: int = -1,
) -> T:
    "flatten"
    if start_dim != 0:
        if start_dim == 1 and end_dim == -1:
            shape = g.op.Shape(x, name="flatten")
            take = g.op.GatherElements(
                shape, np.array([0], dtype=np.int64), axis=0, name="flatten"
            )
            resh = g.op.Concat(
                take, np.array([-1], dtype=np.int64), axis=0, name="flatten"
            )
            return g.op.Reshape(x, resh, outputs=outputs, name="flatten")
        raise NotImplementedError(
            f"start_dim={start_dim}, end_dim={end_dim} not supported."
        )
    if end_dim == -1:
        return g.make_node("Flatten", [x], outputs, name="flatten")
    res = g.make_node("Flatten", [x], outputs, to=end_dim)
    if not sts:
        g.set_type(res, g.get_type(x))
        if g.has_shape(x, full=True):
            g.set_shape(res, (int(np.prod(g.get_shape(x)))))
        else:
            g.set_rank(res, 1)
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
        tsize = np.array(size, dtype=np.int64)
        new_shape = size
    elif isinstance(size, (list, tuple)):
        if all_int(size):
            tsize = np.array(size, dtype=np.int64)
            new_shape = size
        else:
            tsize = g.make_shape_from_results(size, name=name)
    elif isinstance(size, str):
        if g.has_shape(size) and is_static_shape(size):
            tsize = np.array(g.get_shape(size), dtype=np.int64)
        else:
            tsize = g.op.Shape(size, name=name)
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

    res = g.op.ConstantOfShape(
        tsize, value=from_array(value), outputs=outputs, name="name"
    )
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

    assert (
        layout is None
    ), f"empty_like not implemented for layout={layout!r} is not None"
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


def aten_ge_Tensor(
    g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T, y: T
) -> T:
    "greater or equal"
    return aten_ge(g, sts, outputs, x, y, name="ge_Tensor")


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


def aten_gt_Tensor(
    g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T, y: T
) -> T:
    "greater"
    return aten_gt(g, sts, outputs, x, y, name="gt_Tensor")


def aten_index_Tensor(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    indices: List[int],
) -> T:
    "[...,:, ...]"
    assert isinstance(
        indices, (list, tuple)
    ), f"Unexpected type {type(indices)} for indices"
    if len(indices) == 1 and isinstance(indices[0], str):
        return aten_index_select(
            g, sts, outputs, x, dim=0, index=indices[0], name="index1_Tensor"
        )
    n_none = len(list(i for i in indices if i is None))
    if n_none == len(indices) - 1:
        # only one dimension is not None, the other must be added
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
            to_add = list(i for i in range(len(indices)) if i != position)
            assert (
                len(to_add) > 0
            ), f"Unexpected value for to_add={to_add}, position={position}, indices={indices}"
            # res = g.op.UnsqueezeAnyOpset(
            #     temp, np.array(to_add, dtype=np.int64), outputs=outputs
            # )
            # if not sts:
            #     g.set_type(res, g.get_type(x))
            #     if g.has_shape(temp):
            #         shape = list(g.get_shape(temp))
            #         for i in to_add:
            #            shape.insert(i, 1)
            #         g.set_shape(res, tuple(shape))
            #     else:
            #         g.set_rank(res, g.get_rank(temp) + 2)
            return res

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
    "[...,:, ...]"
    assert isinstance(
        indices, list
    ), f"Unexpected type {type(indices)}{g.get_debug_msg()}"
    assert (
        len(indices) == 1
    ), f"Not implementeded for indices={indices}{g.get_debug_msg()}"
    assert g.has_shape(x), f"Missing shape for {x!r}{g.get_debug_msg()}"

    index = indices[0]  # tensor
    new_index = g.op.UnsqueezeAnyOpset(index, np.array([-1], dtype=np.int64), name=name)
    g.set_type(new_index, g.get_type(index))
    if g.has_shape(index):
        g.set_shape(new_index, g.get_shape(index) + (1,))
    else:
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
        else:
            g.set_rank(res, 2)
    return res


def aten__log_softmax(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    dim: int = -1,
    unnamed: bool = False,
    dtype: Optional["torch.dtype"] = None,  # noqa: F821
) -> T:
    "logsoftmax"
    assert not unnamed, "Not implemented when the third parameter is False"
    if dtype is not None:
        itype = torch_dtype_to_onnx_dtype(dtype)
        xc = g.op.Cast(x, to=itype, name="log_softmax")
    else:
        itype = None
        xc = x
    res = g.op.LogSoftmax(xc, axis=dim, outputs=outputs)
    if not sts:
        set_type_shape_unary_op(g, res, xc, itype=itype)
    return res


def aten__log_softmax_backward_data(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    grad_output: T,
    output: T,
    dim: int,
    input_dtype: Optional["torch.dtype"] = None,  # noqa: F821
):
    "logsoftmax backward"
    if input_dtype is not None:
        itype = torch_dtype_to_onnx_dtype(input_dtype)
        grad_outputc = g.op.Cast(
            grad_output, to=itype, name="log_softmax_backward_data"
        )
        set_type_shape_unary_op(g, grad_outputc, grad_output, itype)
    else:
        itype = None
        grad_outputc = grad_output

    vexp = g.op.Exp(output, name="log_softmax_backward_data")
    red = g.op.ReduceSum(
        grad_outputc,
        np.array([dim], dtype=np.int64),
        keepdims=True,
        name="log_softmax_backward_data",
    )
    vmul = g.op.Mul(vexp, red, name="log_softmax_backward_data")
    res = g.op.Sub(
        grad_outputc, vmul, outputs=outputs, name="log_softmax_backward_data"
    )

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
    value,
    name="masked_fill_Scalar",
) -> T:
    "masked"
    dt = g.get_type(mask)
    if dt != TensorProto.BOOL:
        cmask = g.op.Cast(mask, to=TensorProto.BOOL, name=name)
    else:
        cmask = mask
    dtx = g.get_type(x)
    avalue = np.array([value], dtype=tensor_dtype_to_np_dtype(dtx))
    res = g.op.Where(cmask, avalue, x, name=name)
    if not sts:
        g.set_type(res, dtx)
        if g.has_shape(mask):
            g.set_shape(res, g.get_shape(mask))
        else:
            g.set_rank(res, g.get_rank(mask))
    return res


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
) -> T:
    "maxpool"
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
        name="max_pool2d",
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
        indices = g.op.SqueezeAnyOpset(
            indices, np.array([0], dtype=np.int64), name=name
        )

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


def aten_mm(
    g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T, y: T
) -> T:
    "matmul"
    res = g.op.MatMul(x, y, outputs=outputs, name="mm")
    if not sts:
        set_type_shape_matmul(g, res, x, y)
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
    if g.get_type(x) == TensorProto.BOOL and g.get_type(y) == TensorProto.BOOL:
        res = g.op.And(x, y, name="mul_and", outputs=outputs)
    else:
        res, x, y = prepare_inputs_homogeneous_operator(
            g,
            x,
            y,
            f=g.op.Mul,
            name="mul",
            outputs=outputs,
            sts=sts,
        )
    if not sts:
        set_type_shape_binary_op(g, res, x, y)
    return res


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
        assert (
            len(outputs) == 1
        ), f"train is False and outputs is {outputs}{g.get_debug_msg()}"
        return g.op.Identity(x, outputs=outputs, name=name)
    assert (
        len(outputs) == 2
    ), f"train is True and outputs is {outputs}{g.get_debug_msg()}"
    tp = g.make_initializer(
        "", np.array(p, dtype=tensor_dtype_to_np_dtype(g.get_type(x)))
    )
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
    normalized_shape: T,  # int64
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
    dtype: int = None,
    layout=None,
    device: Optional["torch.device"] = None,  # noqa: F821
    pin_memory=None,
    name: str = "ones",
) -> T:
    "constantofshape"
    import torch

    assert layout is None, f"ones not implemented for layout={layout!r} is not None"
    assert not pin_memory, "ones not implemented for pin_memory=True"
    new_shape = None
    if isinstance(size, list):
        isize = np.array(size, dtype=np.int64)
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
        dtype = TensorProto.FLOAT
    res = g.op.ConstantOfShape(
        isize,
        value=from_array(np.array([1], dtype=tensor_dtype_to_np_dtype(dtype))),
        outputs=outputs,
        name=name,
    )
    if not sts:
        g.set_type(res, dtype)
        if new_shape:
            g.set_shape(res, new_shape)
    return res


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
        set_type_shape_unary_op(g, outputs[0], x)
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


def aten_relu(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    inplace: bool = False,
) -> T:
    "relu"
    assert (
        not inplace
    ), f"inplace computation is not allowed with onnx{g.get_debug_msg()}"
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
        irep = np.array(repeats, dtype=np.int64)
    elif g.is_dynamic_shape(repeats):
        # repeats is like a shape
        irep = g.make_shape_from_results(repeats)
    else:
        raise RuntimeError(
            f"repeat not implemented for repeats={repeats}{g.get_debug_msg()}"
        )
    if g.get_rank(x) != len(repeats):
        expanded = g.op.Expand(
            x, np.array((1,) * len(repeats), dtype=np.int64), name=name
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


def aten_rsqrt(
    g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T
) -> T:
    "rqsrt"
    ext = g.make_node("Sqrt", [x], name="rsqrt")
    res = g.op.Reciprocal(ext, outputs=outputs, name="rsqrt")
    if not sts:
        set_type_shape_unary_op(g, outputs[0], x)
    return res


def aten_round(
    g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T
) -> T:
    "round"
    res = g.make_node("Round", [x], outputs, name="round")
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


def _attention_scale(g: GraphBuilder, query: T, name: str = "_attention_scale") -> T:
    if g.has_shape(query):
        shape = g.get_shape(query)
        last = shape[-1]
        if isinstance(last, int):
            scale = 1.0 / (float(last) ** 0.5)
            return np.array([scale], dtype=tensor_dtype_to_np_dtype(g.get_type(query)))

    shape = g.op.Shape(query, name=name)
    last = g.op.Gather(shape, np.array([-1], dtype=np.int64), name=name)
    itype = g.get_type(query)
    clast = g.op.Cast(itype, to=itype, name=name)
    return g.op.Reciprocal(g.op.Sqrt(clast, name=name), name=name)


def _causal_attention_mask(
    g: GraphBuilder, query: T, key: T, name: str = "_causal_attention_mask"
) -> T:
    itype = g.get_type(query)
    dtype = tensor_dtype_to_np_dtype(itype)
    attn_mask = None
    if g.has_shape(query) and g.has_shape(key):
        shape_query, shape_key = g.get_shape(query), g.get_shape(key)
        if isinstance(shape_query[-2], int) and isinstance(shape_key[-2], int):
            shape = (shape_query[-2], shape_key[-2])
            attn_mask = g.op.ConstantOfShape(
                np.array(shape, dtype=np.int64),
                value=from_array(np.array([1], dtype=dtype)),
                name=name,
            )

    if attn_mask is None:
        # dynamic path
        shape_query = g.op.Shape(query, name=name)
        shape_key = g.op.Shape(key, name=name)
        dquery = g.op.Gather(shape_query, np.array([-2], dtype=np.int64), name=name)
        dkey = g.op.Gather(shape_key, np.array([-2], dtype=np.int64), name=name)
        size = g.op.Concat(dquery, dkey, axis=0)
        attn_mask = g.op.ConstantOfShape(
            size, value=from_array([1], dtype=dtype), name=name
        )

    tri_attn_mask = g.op.Trilu(attn_mask, upper=0, name=name)

    new_attn_mask = g.op.Where(
        g.op.Equal(tri_attn_mask, np.array([0], dtype=dtype), name=name),
        np.array([-float("inf")], dtype=dtype),
        np.array([0], dtype=dtype),
        name=name,
    )
    return new_attn_mask


def aten_scaled_dot_product_attention(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    query: T,
    key: T,
    value: T,
    attn_mask: Optional[T] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[T] = None,
    name: str = "aten_scaled_dot_product_attention",
):
    "scaled_dot_product_attention"
    assert (not is_causal) or (
        (is_causal and attn_mask is None)
    ), f"is_causal and attn_mask cannot be set at the same time{g.get_debug_msg()}"

    if scale is None:
        scale = _attention_scale(g, query)

    if is_causal:
        attn_mask = _causal_attention_mask(g, query, key)

    key_transposed_axes = list(range(g.get_rank(key)))
    key_transposed_axes[-1], key_transposed_axes[-2] = (
        key_transposed_axes[-2],
        key_transposed_axes[-1],
    )
    key_transposed = g.op.Transpose(key, perm=key_transposed_axes, name=name)

    sc = g.op.Sqrt(scale, name=name)
    query_scaled = g.op.Mul(query, sc, name=name)
    key_transposed_scaled = g.op.Mul(key_transposed, sc)
    mul_qk = g.op.MatMul(query_scaled, key_transposed_scaled, name=name)

    itype = g.get_type(query)
    dtype = tensor_dtype_to_np_dtype(itype)

    if attn_mask is None:
        mul_qk_add = mul_qk
    elif g.get_type(attn_mask) == TensorProto.BOOL:
        attn_mask = g.op.Where(
            attn_mask,
            np.array([0.0], dtype=dtype),
            np.array([-float("inf")], dtype=dtype),
            name=name,
        )
        mul_qk_add = g.op.Add(mul_qk, attn_mask, name=name)
    else:
        mul_qk_add = g.op.Add(mul_qk, attn_mask, name=name)

    attn_weight = g.op.Softmax(mul_qk_add, axis=-1)

    if dropout_p != 0:
        attn_weight = g.op.Dropout(
            attn_weight, np.array([dropout_p], dtype=dtype), name=name
        )[0]

    return g.op.MatMul(attn_weight, value, name=name, outputs=outputs)


def _aten__scaled_dot_product_flash_attention_fillin_empty_outputs(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    query: T,
    name: str = "_scaled_dot_product_flash_attention_fillin_empty_outputs",
) -> Tuple[T, T, T, T]:

    query_first_three_dims = g.op.Slice(
        g.op.Shape(query, name=name),
        g.op.Constant(value_ints=[0], name=name),
        g.op.Constant(value_ints=[3], name=name),
        name=name,
    )
    logsumexp = g.op.Expand(
        np.array([0], dtype=tensor_dtype_to_np_dtype(g.get_type(query))),
        query_first_three_dims,
        name=name,
        outputs=[outputs[0]],
    )

    empty_tensor_int = g.op.Cast(
        g.op.ConstantOfShape(
            g.op.Constant(
                value=make_tensor("Empty_INTS", TensorProto.INT64, [0], []), name=name
            ),
            name=name,
        ),
        to=TensorProto.INT64,
        name=name,
        outputs=[outputs[1]],
    )
    empty_tensor_float = g.op.ConstantOfShape(
        g.op.Constant(
            value=make_tensor("Empty_FLOATS", TensorProto.INT64, [0], []), name=name
        ),
        name=name,
        outputs=[outputs[2]],
    )
    empty_int = g.op.Constant(value_int=0, name=name, outputs=[outputs[3]])

    return logsumexp, empty_tensor_int, empty_int, empty_tensor_float


def aten__scaled_dot_product_flash_attention_for_cpu(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    query: T,
    key: T,
    value: T,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    attn_mask: Optional[T] = None,
    scale: Optional[float] = None,
    return_debug_mask: bool = False,
    name: str = "_scaled_dot_product_flash_attention_for_cpu_default",
) -> Tuple[T, T, T, T, T, T, T, T, T]:
    """_scaled_dot_product_flash_attention"""
    assert not return_debug_mask, "Not implemented when return_debug_mask is false."
    result = aten_scaled_dot_product_attention(
        g,
        sts,
        [outputs[0]],
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        name="_scaled_dot_product_flash_attention_for_cpu_default",
    )
    assert isinstance(result, str), f"Unexpected type {type(result)}{g.get_debug_msg()}"

    # The followings are not comsumed by the graph on llama 3 at least.
    if len(outputs) == 2:
        # only need 2
        query_first_three_dims = g.op.Slice(
            g.op.Shape(query, name=name),
            g.op.Constant(value_ints=[0], name=name),
            g.op.Constant(value_ints=[3], name=name),
            name=name,
        )
        logsumexp = g.op.Expand(
            np.array([0], dtype=tensor_dtype_to_np_dtype(g.get_type(query))),
            query_first_three_dims,
            name=name,
            outputs=[outputs[1]],
        )
        return result, logsumexp

    assert len(outputs) == 8, (
        f"Unexpected number of outputs {len(outputs)}, "
        f"outputs={outputs}{g.get_debug_msg()}"
    )
    (
        logsumexp,
        empty_tensor_int,
        empty_int,
        empty_tensor_float,
    ) = _aten__scaled_dot_product_flash_attention_fillin_empty_outputs(
        g, sts, [outputs[1], outputs[3], outputs[4], outputs[8]], query, name=name
    )

    empty_tensor_int2 = g.op.Identity(empty_tensor_int, name=name)
    empty_int2 = g.op.Identity(empty_int, name=name)
    empty_tensor_int2 = g.op.Identity(empty_tensor_int, name=name)

    return (
        result,  # 0
        logsumexp,  # 1
        empty_tensor_int,  # 2
        empty_tensor_int2,  # 3
        empty_int,  # 4
        empty_int2,  # 5
        empty_tensor_int,  # 6
        empty_tensor_int2,  # 7
        empty_tensor_float,  # 8
    )


def aten_select_int(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    dim: int,
    index: int,
) -> T:
    "gather"
    assert isinstance(
        dim, int
    ), f"Unexpected type {type(dim)} for dim{g.get_debug_msg()}"
    assert isinstance(
        index, int
    ), f"Unexpected type {type(index)} for dim{g.get_debug_msg()}"
    res = g.op.Gather(x, np.array(index, dtype=np.int64), axis=dim, outputs=outputs)
    if not sts:
        g.set_type(res, g.get_type(x))
        if g.has_shape(x):
            shape = g.get_shape(x)
            if dim < 0:
                dim += len(shape)
            assert dim < len(
                shape
            ), f"shape is {shape}, dim is {dim}{g.get_debug_msg()}"
            new_shape = [s for i, s in enumerate(shape) if i != dim]
            g.set_shape(res, tuple(new_shape))
        else:
            g.get_rank(res, g.get_rank(x) - 1)
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
    return g.make_node("Constant", [], value_floats=[0], name="_set_grad_enabled")


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
    raise RuntimeError(
        f"setitem not implemented for indices={indices}{g.get_debug_msg()}"
    )


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
    assert (
        step == 1
    ), f"slice_backward not implemented for step={step}{g.get_debug_msg()}"

    itype = g.get_type(grad_output)
    value = from_array(np.array([0], dtype=tensor_dtype_to_np_dtype(itype)))

    inputs = []

    if g.has_shape(grad_output) and is_static_shape(g.get_shape(grad_output)):
        name_s = f"{name}_static"
        # static version
        shape = g.get_shape(grad_output)

        if start > 0:
            cst_shape = list(shape)
            cst_shape[dim] = start
            cst = g.op.ConstantOfShape(
                np.array(cst_shape, dtype=np.int64), value=value, name=name_s
            )
            inputs.append(cst)

        inputs.append(grad_output)

        if end < 9223372036854775807:
            cst_shape = list(shape)
            cst_shape[dim] = input_sizes[dim] - shape[dim] - start
            cst = g.op.ConstantOfShape(
                np.array(cst_shape, dtype=np.int64), value=value, name=name_s
            )
            inputs.append(cst)

    else:
        name_d = f"{name}_dynamic"
        # dynamic version
        shape = g.op.Shape(grad_output, name=name_d)

        if start > 0:
            cst_shape = g.op.ScatterElements(
                shape,
                np.array([dim], dtype=np.int64),
                np.array([start], dtype=np.int64),
                axis=0,
                name=name,
            )
            cst = g.op.ConstantOfShape(cst_shape, value=value, name=name_d)
            inputs.append(cst)

        inputs.append(grad_output)

        if end < 9223372036854775807:
            shape_dim = g.op.Gather(shape, np.array([dim], dtype=np.int64), name=name_d)
            new_dim = g.op.Sub(
                np.array([input_sizes[dim] - start], dtype=np.int64), shape_dim
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

    index_1 = np.arange(0, dim_shape)
    if (start is None or isinstance(start, int)) and (
        end is None or isinstance(end, int)
    ):
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
    ), f"slice_scatter not implemented for end={step}{g.get_debug_msg()}"

    shape = g.op.Shape(x, name=name)
    dim_shape = g.op.Gather(shape, np.array([dim], dtype=np.int64), name=name)

    index_1 = g.op.Range(
        np.array([0], dtype=np.int64),
        dim_shape,
        np.array([1], dtype=np.int64),
        name=name,
    )
    index_2 = g.op.Slice(
        index_1,
        g.get_dynamic_dimension(start),
        (dim_shape if end is None else g.get_dynamic_dimension(end)),
        np.array([0], dtype=np.int64),
        np.array([step or 1], dtype=np.int64),
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


def aten_silu(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    inplace: bool = False,
) -> T:
    "silu"
    assert (
        not inplace
    ), f"inplace computation is not allowed with onnx{g.get_debug_msg()}"
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


def aten_sinh(
    g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T
) -> T:
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
    assert not half_to_float, f"Unexpected value for half_to_float={half_to_float!r}"
    res = g.op.Softmax(x, axis=dim, outputs=outputs, name="_softmax")
    if not sts:
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
) -> T:
    "softmax backward"
    if input_dtype is not None:
        itype = torch_dtype_to_onnx_dtype(input_dtype)
        grad_outputc = g.op.Cast(
            grad_output, to=itype, name="log_softmax_backward_data"
        )
        set_type_shape_unary_op(g, grad_outputc, grad_output, itype=itype)
    else:
        itype = None
        grad_outputc = grad_output

    new_grad_output = g.op.Mul(grad_outputc, y)
    set_type_shape_unary_op(g, new_grad_output, grad_outputc)
    sums = g.op.ReduceSum(
        new_grad_output,
        np.array([dim], dtype=np.int64),
        keepdims=1,
        name="softmax_backward_data",
    )
    set_type_shape_reduce_op(g, sums, new_grad_output, keepdim=1, axes=(dim,))
    temp = g.op.Mul(y, sums, name="softmax_backward_data")
    set_type_shape_unary_op(g, temp, y)
    res = g.op.Sub(new_grad_output, temp, outputs=outputs, name="softmax_backward_data")
    if not sts:
        set_type_shape_unary_op(g, res, grad_outputc, itype=itype)
    return res


def aten_split_with_sizes(
    g: GraphBuilder,
    sts: Optional[Dict[str, Any]],
    outputs: List[str],
    x: T,
    split_sizes: T,
    dim: int = 0,
    name: str = "split_with_sizes",
    use_sequence: bool = False,
) -> T:
    "split_to_sequence or split"
    assert isinstance(split_sizes, list) and all_int(
        split_sizes
    ), f"Implemented when split_sizes ({split_sizes}) is a constant{g.get_debug_msg()}"
    assert isinstance(dim, int), f"dim={dim} is not an integer{g.get_debug_msg()}"
    assert all(map(lambda d: d > 0, split_sizes)), (
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


def aten_sqrt(
    g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T
) -> T:
    "sqrt"
    res = g.make_node("Sqrt", [x], name="sqrt")
    if not sts:
        set_type_shape_unary_op(g, outputs[0], x)
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
    assert (
        alpha == 1
    ), f"sub_Tensor not implemented for alpha={alpha}{g.get_debug_msg()}"
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
    if dtype is not None:
        itype = torch_dtype_to_onnx_dtype(dtype)
        xc = g.op.Cast(x, to=itype)
    else:
        xc = x
    if dim is None:
        result = g.op.ReduceSumAnyOpset(
            xc, keepdims=keepdim, outputs=outputs, name=name
        )
    else:
        if isinstance(dim, int):
            adim = np.array([dim], dtype=np.int64)
        else:
            adim = np.array(dim, dtype=np.int64)
        result = g.op.ReduceSumAnyOpset(
            xc, adim, keepdims=keepdim, outputs=outputs, name=name
        )
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
    shape = g.op.Shape(x, name=name)
    return g.op.Gather(
        shape, np.array([dim], dtype=np.int64), name=name, outputs=outputs
    )


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
    assert (
        memory_format is None
    ), f"_to_copy implemented with memory_format={memory_format!r}"
    if dtype is None:
        return g.op.Identity(x, outputs=outputs, name="_to_copy")
    itype = torch_dtype_to_onnx_dtype(dtype)
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


def aten_tan(
    g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T
) -> T:
    "tan"
    res = g.make_node("Tan", [x], outputs, name="tan")
    if not sts:
        set_type_shape_unary_op(g, outputs[0], x)
    return res


def aten_tanh(
    g: GraphBuilder, sts: Optional[Dict[str, Any]], outputs: List[str], x: T
) -> T:
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
                    f"Unable to convert to guess value dtype "
                    f"for x={x}{g.get_debug_msg()}"
                )

            return g.make_initializer(outputs[0], cst)
        raise RuntimeError(
            f"Unable to convert a value into a tensor, x={x}{g.get_debug_msg()}"
        )

    if isinstance(indices, tuple) and len(indices) == 1:
        if isinstance(indices[0], list) and all_int(indices[0]):
            return _aten_tensor_int1(g, sts, outputs, x, indices, [0], [])
    raise RuntimeError(
        f"Unable to handle getitem with indices={indices}{g.get_debug_msg()}"
    )


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
    res = g.op.Where(
        le, np.array([0], dtype=dtype), grad_output, outputs=outputs, name=name
    )
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
        res = g.op.Trilu(
            x, np.array(diagonal, dtype=np.int64), upper=0, outputs=outputs
        )
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
    diagonal: int = 0,
) -> T:
    """trilu"""
    res = g.op.Trilu(x, diagonal, upper=1, name="triu")
    if not sts:
        set_type_shape_unary_op(g, res, x)
    return res


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
