"""
`sts` is an alias for `set_type_shape`.
"""

from typing import Any, List, Optional, Sequence, Tuple, Union
import numpy as np
from onnx import TensorProto
from onnx.helper import tensor_dtype_to_np_dtype
from onnx.numpy_helper import from_array
from .annotations import (
    all_float,
    all_int,
    all_int_or_float,
    is_static_dimension,
    is_static_shape,
)
from ._exceptions import FunctionNotFoundError
from ._aten_helper import (
    _adjust_attributes_of_max_pool,
    set_type_shape_unary_op,
    set_type_shape_binary_op,
    set_type_shape_reduce_op,
    set_type_shape_reshape,
    onnx_dtype_to_torch_dtype,
    prepare_inputs_homogeneous_operator,
    torch_dtype_to_onnx_dtype,
)
from .graph_builder import GraphBuilder


T = str


def aten_abs(g: GraphBuilder, sts: bool, outputs: List[str], x: T) -> T:
    res = g.make_node("Abs", [x], outputs)
    if sts:
        set_type_shape_unary_op(g, outputs[0], x)
    return res


def aten_acos(g: GraphBuilder, sts: bool, outputs: List[str], x: T) -> T:
    res = g.make_node("Acos", [x], outputs)
    if sts:
        set_type_shape_unary_op(g, outputs[0], x)
    return res


def aten_acosh(g: GraphBuilder, sts: bool, outputs: List[str], x: T) -> T:
    res = g.make_node("Acosh", [x], outputs)
    if sts:
        set_type_shape_unary_op(g, outputs[0], x)
    return res


def aten_add(g: GraphBuilder, sts: bool, outputs: List[str], x: T, y: T) -> T:
    res, x, y = prepare_inputs_homogeneous_operator(
        g, x, y, f=g.op.Add, name="add", outputs=outputs
    )
    if sts:
        set_type_shape_binary_op(g, outputs[0], x, y)
    return res


def aten_add_Scalar(
    g: GraphBuilder,
    sts: bool,
    outputs: List[str],
    x: T,
    y: T,
    alpha: Optional[Any] = None,
) -> T:
    assert alpha in (None, 1), f"alpha={alpha}, not implemented"
    x, y = prepare_inputs_homogeneous_operator(g, x, y)
    res = g.op.Add(x, y, outputs=outputs, name="add_Scalar")
    if sts:
        set_type_shape_binary_op(g, outputs[0], x, y)
    return res


def aten_add_Tensor(
    g: GraphBuilder,
    sts: bool,
    outputs: List[str],
    x: T,
    y: T,
    alpha: Optional[Any] = None,
) -> T:
    assert alpha in (None, 1), f"alpha={alpha}, not implemented"
    x, y = prepare_inputs_homogeneous_operator(g, x, y)
    res = g.op.Add(x, y, outputs=outputs, name="add_Tensor")
    if sts:
        set_type_shape_binary_op(g, outputs[0], x, y)
    return res


def aten_and(
    g: GraphBuilder, sts: bool, outputs: List[str], x: T, y: T, name="and"
) -> T:
    res, x, y = prepare_inputs_homogeneous_operator(
        g, x, y, f=g.op.And, name=name, outputs=outputs
    )
    if sts:
        set_type_shape_binary_op(g, outputs[0], x, y)
    return res


def aten_and_(
    g: GraphBuilder, sts: bool, outputs: List[str], x: T, y: T, name="and"
) -> T:
    return aten_and(g, sts, outputs, x, y, name="and_")


def aten_addmm(
    g: GraphBuilder,
    sts: bool,
    outputs: List[str],
    a: T,
    b: T,
    c: T,
    beta: float = 1.0,
    alpha: float = 1.0,
) -> T:
    return g.op.Gemm(
        b, c, a, alpha=float(alpha), beta=float(beta), outputs=outputs, name="addmm"
    )


def aten_alias(g: GraphBuilder, sts: bool, outputs: List[str], x: T) -> T:
    return g.make_node("Identity", [x], outputs, name="alias")


def aten_all(g: GraphBuilder, sts: bool, outputs: List[str], x: T) -> T:
    res = g.op.Cast(
        g.op.ReduceMin(g.op.Cast(x, to=TensorProto.INT32, name="all"), name="all"),
        to=TensorProto.BOOL,
        outputs=outputs,
        name="all",
    )
    if sts:
        g.set_type(res, TensorProto.BOOL)
        g.set_shape(res, tuple())
    return res


def aten_arange(
    g: GraphBuilder,
    sts: bool,
    outputs: List[str],
    start: Optional[int] = None,
    end: Optional[int] = None,
    step: int = 1,
    dtype: Optional["torch.dtype"] = None,  # noqa: F821
    layout=None,
    device=None,
    pin_memory=None,
    name: str = "arange",
) -> T:
    assert layout is None, f"arange not implemented for layout={layout!r} is not None"
    assert not pin_memory, "arange not implemented for pin_memory=True"
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
    if sts:
        g.set_type(res, itype)
        if isinstance(end, str) or isinstance(start, str) or isinstance(step, str):
            g.set_rank(res, 1)
        else:
            g.set_shape(res, ((end - start) // step,))
    return res


def aten_arange_start(
    g: GraphBuilder,
    sts: bool,
    outputs: List[str],
    start: Optional[int] = None,
    end: Optional[int] = None,
    dtype: Optional["torch.dtype"] = None,  # noqa: F821
    layout=None,
    device=None,
    pin_memory=None,
) -> T:
    import torch

    assert layout in (
        None,
        torch.strided,
    ), f"arange not implemented for layout={layout!r}"
    assert not pin_memory, "arange not implemented for pin_memory=True"
    return aten_arange(g, sts, outputs, start, end, dtype=dtype)


def aten_arange_start_step(
    g: GraphBuilder,
    sts: bool,
    outputs: List[str],
    start: Optional[int] = None,
    end: Optional[int] = None,
    step: int = 1,
    dtype: Optional["torch.dtype"] = None,  # noqa: F821
    layout=None,
    device=None,
    pin_memory=None,
) -> T:
    assert layout is None, f"arange not implemented for layout={layout!r} is not None"
    assert not pin_memory, "arange not implemented for pin_memory=True"
    return aten_arange(g, sts, outputs, start, end, step, dtype)


def aten_argmax(
    g: GraphBuilder,
    sts: bool,
    outputs: List[str],
    x: T,
    dim: Optional[int] = None,
    keepdim: bool = False,
) -> T:
    if dim is None:
        xf = g.op.Reshape(x, np.array([-1], dtype=np.int64))
        res = g.op.Squeeze(
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
    if sts:
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
    sts: bool,
    outputs: List[str],
    x: T,
    size: List[int],
    stride: List[int],
    storage_offset: Optional[int] = None,
) -> T:
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
        strided = torch.as_strided(tindices, size, stride, storage_offset)
        np_strided = strided.detach().numpy().ravel()

    flat = g.op.Reshape(x, np.array([-1], dtype=np.int64))
    xflat = g.op.Gather(flat, np_strided.astype(np.int64))
    res = g.op.Reshape(xflat, np.array(size, dtype=np.int64), outputs=outputs)

    if sts:
        g.set_type(res, g.get_type(x))
        g.set_shape(res, tuple(size))
    return res


def aten_asin(g: GraphBuilder, sts: bool, outputs: List[str], x: T) -> T:
    res = g.make_node("Asin", [x], outputs)
    if sts:
        set_type_shape_unary_op(g, outputs[0], x)
    return res


def aten_asinh(g: GraphBuilder, sts: bool, outputs: List[str], x: T) -> T:
    res = g.make_node("Asinh", [x], outputs)
    if sts:
        set_type_shape_unary_op(g, outputs[0], x)
    return res


def aten_atan(g: GraphBuilder, sts: bool, outputs: List[str], x: T) -> T:
    res = g.make_node("Atan", [x], outputs)
    if sts:
        set_type_shape_unary_op(g, outputs[0], x)
    return res


def aten_atanh(g: GraphBuilder, sts: bool, outputs: List[str], x: T) -> T:
    res = g.make_node("Atanh", [x], outputs)
    if sts:
        set_type_shape_unary_op(g, outputs[0], x)
    return res


def aten_bmm(g: GraphBuilder, sts: bool, outputs: List[str], x: T, y: T) -> T:
    assert g.get_type(x) == g.get_type(y), (
        f"type mismatch between {x!r}:{g.get_type(x)} and "
        f"{y!r}:{g.get_type(y)}{g.get_debug_msg()}"
    )
    return g.op.MatMul(x, y, outputs=outputs, name="bmm")


def aten_cat(
    g: GraphBuilder, sts: bool, outputs: List[str], tensors: Tuple[T, ...], dim: int = 0
) -> T:
    assert len(tensors) > 0, f"No tensor to concat{g.get_debug_msg()}"
    res = g.op.Concat(*tensors, axis=dim, outputs=outputs, name="cat")
    if sts:
        dt0 = g.get_type(tensors[0])
        assert all(map(lambda t: g.get_type(t) == dt0, tensors))
        r0 = g.get_rank(tensors[0])
        assert all(map(lambda t: g.get_rank(t) == r0, tensors))
        g.set_type(outputs[0], dt0)
        g.set_rank(outputs[0], r0)
    return res


def aten_clone(
    g: GraphBuilder,
    sts: bool,
    outputs: List[str],
    x: T,
    memory_format: Optional[str] = None,
    name="clone",
) -> T:
    import torch

    assert (
        memory_format is None
        or memory_format == torch.contiguous_format
        or memory_format == torch.preserve_format
    ), f"Unexpected value for memory_format={memory_format!r}{g.get_debug_msg()}"
    return g.make_node("Identity", [x], outputs, name=name)


def aten_convolution(
    g: GraphBuilder,
    sts: bool,
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
        cst1 = g.make_node("Constant", [], value_ints=[1], name="conv")
        bias_shape = g.make_node("Expand", [weight_dim_0, cst1], name="conv")
        dtype = tensor_dtype_to_np_dtype(g.get_type(input))
        bias = g.op.Expand(np.array([0.0], dtype=dtype), bias_shape, name="conv")

    # if Rank(input) != Rank(weight):
    #    input = op.Unsqueeze(input, op.Constant(value_ints=[0]))

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
    if sts:
        g.set_type(res, g.get_type(input))
        g.set_rank(res, g.get_rank(input))
    return res


def aten_conv2d(
    g: GraphBuilder,
    sts: bool,
    outputs: List[str],
    input: T,
    weight: T,
    bias: T = None,
    stride: Sequence[int] = (1, 1),
    padding: Sequence[int] = (0, 0),
    dilation: Sequence[int] = (1, 1),
    groups: int = 1,
) -> T:
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
    sts: bool,
    outputs: List[str],
    x: T,
    src: T,
    non_blocking: bool = False,
) -> T:
    assert not non_blocking, "copy implemented when non_blocking is True"
    if g.get_type(x) == g.get_type(src):
        return g.op.Identity(src, name="copy")
    return g.op.CastLike(src, x, name="copy")


def aten_cos(g: GraphBuilder, sts: bool, outputs: List[str], x: T) -> T:
    res = g.make_node("Cos", [x], outputs)
    if sts:
        set_type_shape_unary_op(g, outputs[0], x)
    return res


def aten_cosh(g: GraphBuilder, sts: bool, outputs: List[str], x: T) -> T:
    res = g.make_node("Cosh", [x], outputs)
    if sts:
        set_type_shape_unary_op(g, outputs[0], x)
    return res


def aten_detach(g: GraphBuilder, sts: bool, outputs: List[str], x: T) -> T:
    return g.make_node("Identity", [x], outputs, name="detach")


def aten_div(
    g: GraphBuilder, sts: bool, outputs: List[str], x: T, y: T, name="div"
) -> T:
    res, x, y = prepare_inputs_homogeneous_operator(
        g, x, y, f=g.op.Div, name=name, outputs=outputs
    )
    if sts:
        set_type_shape_binary_op(g, outputs[0], x, y)
    return res


def aten_div_Scalar(g: GraphBuilder, sts: bool, outputs: List[str], x: T, y: T) -> T:
    return aten_div(g, sts, outputs, x, y, name="div_Scalar")


def aten_div_Tensor(
    g: GraphBuilder,
    sts: bool,
    outputs: List[str],
    x: T,
    y: T,
    alpha: Optional[Any] = None,
) -> T:
    assert alpha in (None, 1), f"alpha={alpha}, not implemented"
    x, y = prepare_inputs_homogeneous_operator(g, x, y)
    res = g.op.Div(x, y, outputs=outputs, name="div_Tensor")
    if sts:
        set_type_shape_binary_op(g, outputs[0], x, y)
    return res


def aten_dropout(
    g: GraphBuilder,
    sts: bool,
    outputs: List[str],
    x: T,
    p: T = 0.5,  # float
    training: T = True,  # bool
) -> T:
    if len(outputs) == 1:
        outputs = outputs.copy()
        outputs.append("")
    if isinstance(p, float):
        p = np.array(p, dtype=np.float64)
    if isinstance(training, bool):
        training = np.array(training, dtype=np.bool_)
    result, _ = g.op.Dropout(x, p, training, outputs=outputs)
    if sts:
        set_type_shape_unary_op(g, outputs[0], x)
    return result


def aten_embedding(
    g: GraphBuilder,
    sts: bool,
    outputs: List[str],
    weight: T,
    indices: T,
    padding_idx: Optional[int] = None,
    max_norm: Optional[int] = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> T:
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
    if sts:
        g.set_type(res, g.get_type(weight))
        g.set_rank(res, g.get_rank(weight) + g.get_rank(indices) - 1)
    return res


def aten_embedding_dense_backward(
    g: GraphBuilder,
    sts: bool,
    outputs: List[str],
    grad_output: T,
    indices: T,
    num_weights: int,
    padding_idx: int,
    scale_grad_by_freq: bool,
) -> T:
    """
    def _unsqueeze_to_dim(x: Tensor, dim: int) -> Tensor:
        for _ in range(dim - x.dim()):
            x = x.unsqueeze(-1)
        return x

    def embedding_dense_backward(
        grad_output: Tensor,
        indices: Tensor,
        num_weights: int,
        padding_idx: int,
        scale_grad_by_freq: bool,
    ):
        computation_dtype, result_dtype = utils.elementwise_dtypes(
            grad_output, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
        )
        grad_output = grad_output.to(computation_dtype)
        indices = _maybe_convert_to_dtype(indices, torch.long)  # type: ignore[assignment]
        if scale_grad_by_freq:
            counts = indices.new_zeros((num_weights,))
            ones = torch.ones_like(indices)
            counts = aten._unsafe_index_put(counts, [indices], ones, accumulate=True)
            grad_weights_scale = counts[indices]
            grad_output = grad_output / grad_weights_scale.unsqueeze(-1)

        mask = _unsqueeze_to_dim(indices == padding_idx, grad_output.ndim)
        grad = grad_output.masked_fill(mask, 0)
        grad_weight = grad_output.new_zeros(
            (num_weights,) + grad_output.shape[indices.ndim :]
        )
        return aten._unsafe_index_put(grad_weight, [indices], grad, accumulate=True).to(
            result_dtype
        )
    """
    assert (
        not scale_grad_by_freq
    ), f"scale_grad_by_freq=True not implemented{g.get_debug_msg()}"
    assert g.has_shape(grad_output), f"missing shape for grad_output{g.get_debug_msg()}"
    assert g.has_shape(indices), f"missing shape for indices{g.get_debug_msg()}"
    assert is_static_shape(
        g.get_shape(grad_output)
    ), f"unknown shape for grad_output{g.get_debug_msg()}"

    # if scale_grad_by_freq:
    #     counts = indices.new_zeros((num_weights,))
    #     ones = torch.ones_like(indices)
    #     counts = aten._unsafe_index_put(counts, [indices], ones, accumulate=True)
    #     grad_weights_scale = counts[indices]
    #     grad_output = grad_output / grad_weights_scale.unsqueeze(-1)

    # mask = _unsqueeze_to_dim(indices == padding_idx, grad_output.ndim)
    shape_indices = list(g.get_shape(indices))
    ndim = g.get_rank(grad_output)
    rank_indices = g.get_rank(indices)
    shape_indices.extend([1] * (ndim - rank_indices))
    assert ndim == len(shape_indices), (
        f"New shape for indices is wrong shape_indices="
        f"{shape_indices}, expected rank={ndim}"
    )
    mask = g.op.Reshape(
        g.op.Equal(
            indices,
            np.array([padding_idx], dtype=np.int64),
            name="embedding_dense_backward",
        ),
        np.array(shape_indices, dtype=np.int64),
        name="embedding_dense_backward",
    )
    g.set_type(mask, TensorProto.BOOL)
    g.set_rank(mask, len(shape_indices))

    # grad = grad_output.masked_fill(mask, 0)
    grad = aten_masked_fill_Scalar(
        g,
        sts,
        None,
        grad_output,
        mask,
        0,
        name="embedding_dense_backward_masked_fill",
    )

    shape_output = g.get_shape(grad_output)
    new_shape = (num_weights,) + shape_output[rank_indices:]
    grad_weight = g.op.ConstantOfShape(
        np.array(new_shape, dtype=np.int64), name="embedding_dense_backward"
    )
    indices_reshaped = g.op.UnsqueezeAnyOpset(
        indices, np.array([0], dtype=np.int64), name="embedding_dense_backward"
    )
    res = g.op.ScatterElements(
        grad_weight,
        indices_reshaped,
        grad,
        outputs=outputs,
        name="embedding_dense_backward",
    )
    if sts:
        g.set_type(res, g.get_type(grad_output))
        g.set_shape(res, new_shape)
    assert res is None, (
        "aten_embedding_dense_backward is not correctly implemented, "
        "use get_decomposition_table."
    )
    return res


def aten_empty_like(
    g: GraphBuilder,
    sts: bool,
    outputs: List[str],
    x: T,
    dtype: Optional["torch.dtype"] = None,  # noqa: F821
    layout=None,
    device=None,
    pin_memory=None,
    memory_format=None,
) -> T:
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
            0,
            dtype=dtype or g.get_type(x),
            name="empty_like",
        )
    raise RuntimeError(
        f"empty_like is not implemented when shape is not fully known "
        f"for {x!r}{g.get_debug_msg()}"
    )


def aten_eq(g: GraphBuilder, sts: bool, outputs: List[str], x: T, y: T) -> T:
    x, y = prepare_inputs_homogeneous_operator(g, x, y)
    res = g.op.Equal(x, y, outputs=outputs)
    if sts:
        set_type_shape_binary_op(g, outputs[0], x, y, cmp_op=True)
    return res


def aten_eq_Scalar(g: GraphBuilder, sts: bool, outputs: List[str], x: T, y: T) -> T:
    return aten_eq(g, sts, outputs, x, y)


def aten_expand(
    g: GraphBuilder,
    sts: bool,
    outputs: List[str],
    x: T,
    sizes: Union[T, List[Union[int, str]]],
    implicit: bool = False,
    name: str = "expand",
) -> T:
    assert not implicit, f"Unexpected value for implicit={implicit!r}"

    if not isinstance(sizes, str) and all_int(sizes) and min(sizes) >= 0:
        # static sizes
        res = g.op.Expand(
            x, np.array(sizes, dtype=np.int64), outputs=outputs, name=name
        )
        if sts:
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
        if sts:
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
    if sts:
        g.set_type(res, g.get_type(x))
        g.set_rank(res, len(sizes))
    return res


def aten_fill_Scalar(g: GraphBuilder, sts: bool, outputs: List[str], x: T, v: T) -> T:
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
    sts: bool,
    outputs: List[str],
    x: T,
    start_dim: int = 1,
    end_dim: int = -1,
) -> T:
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
        return g.make_node("Flatten", [x], outputs)
    res = g.make_node("Flatten", [x], outputs, to=end_dim)
    if sts:
        g.set_type(res, g.get_type(x))
        if g.has_shape(x, full=True):
            g.set_shape(res, (int(np.prod(g.get_shape(x)))))
        else:
            g.set_rank(res, 1)
    return res


def aten_full(
    g: GraphBuilder,
    sts: bool,
    outputs: List[str],
    size: T,
    fill_value: float,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    name: str = "full",
) -> T:
    import torch

    assert layout in (None, torch.strided), (
        f"full not implemented for layout={layout!r} is not None, "
        f"size={size!r}, dtype={dtype}{g.get_debug_msg()}"
    )
    assert not pin_memory, "full not implemented for pin_memory=True"
    assert fill_value is None or isinstance(
        fill_value, (float, int)
    ), f"Unexpected type {type(fill_value)} for fill_value."

    new_shape = None

    if isinstance(size, tuple):
        assert all(
            map(lambda x: isinstance(x, int), size)
        ), f"Unexpected values for size={size}-{[type(s) for s in size]}"
        tsize = np.array(size, dtype=np.int64)
        new_shape = size
    elif isinstance(size, list):
        assert all(
            map(lambda x: isinstance(x, int), size)
        ), f"Unexpected values for size={size}-{[type(s) for s in size]}"
        tsize = np.array(size, dtype=np.int64)
        new_shape = size
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
    if sts:
        g.set_type(res, itype)
        if new_shape:
            g.set_shape(res, new_shape)

    # size = op.Cast(size, to=INT64.dtype)
    # fill_value = op.Cast(fill_value, to=dtype)
    # return op.Expand(fill_value, size)
    return res


def aten_FunctionCtx(g: GraphBuilder, sts: bool, outputs: List[str], *args, **kwargs):
    if len(args) == 0 and len(kwargs) == 0:
        return
    raise NotImplementedError(f"args={args}, kwargs={kwargs}")


def aten_gt(g: GraphBuilder, sts: bool, outputs: List[str], x: T, y: T) -> T:
    x, y = prepare_inputs_homogeneous_operator(g, x, y)
    res = g.op.Greater(x, y, outputs=outputs)
    if sts:
        set_type_shape_binary_op(g, outputs[0], x, y, cmp_op=True)
    return res


def aten_index_Tensor(
    g: GraphBuilder, sts: bool, outputs: List[str], x: T, indices: List[int]
) -> T:
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
            # if sts:
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
    sts: bool,
    outputs: List[str],
    self: T,
    indices: List[T],
    values: T,
    accumulate: bool = False,
    name="aten_index_put",
) -> T:
    assert isinstance(
        indices, list
    ), f"Unexpected type {type(indices)}{g.get_debug_msg()}"
    assert (
        len(indices) == 1
    ), f"Not implementeded for indices={indices}{g.get_debug_msg()}"
    assert g.has_shape(self), f"Missing shape for {self!r}{g.get_debug_msg()}"

    index = indices[0]  # tensor
    new_index = g.op.UnsqueezeAnyOpset(index, np.array([-1], dtype=np.int64), name=name)

    shape_self = g.get_shape(self)

    if accumulate:
        zeros = g.op.ConstantOfShape(
            np.array(shape_self, dtype=np.int64),
            value=from_array(
                np.array([0], dtype=tensor_dtype_to_np_dtype(g.get_type(values)))
            ),
            name=name,
        )
        result = g.op.ScatterND(zeros, new_index, values, name=name, reduction="add")
        res = g.op.Add(result, self, name=name)
    else:
        res = g.op.ScatterND(self, new_index, values, name=name)

    if sts:
        g.set_type(res, g.get_type(self))
        g.set_shape(res, g.get_shape(self))
    return result


def aten__unsafe_index_put(
    g: GraphBuilder,
    sts: bool,
    outputs: List[str],
    self: T,
    indices: List[T],
    values: T,
    accumulate: bool = False,
) -> T:
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
    sts: bool,
    outputs: List[str],
    x: T,
    dim: int,
    index: T,
    name: str = "index_select",
) -> T:
    res = g.op.Gather(x, index, axis=dim, outputs=outputs, name=name)
    if sts:
        g.set_type(res, g.get_type(x))
        if g.has_shape(x) and g.has_shape(index):
            shape = list(g.get_shape(x))
            index_shape = g.get_shape(index)
            shape[dim] = index_shape[0]
            g.set_shape(res, tuple(shape))
        else:
            g.set_rank(res, g.get_rank(x))
    return res


def aten_linear(
    g: GraphBuilder, sts: bool, outputs: List[str], x: T, weight: T, bias: T = None
) -> T:
    weight_transposed = g.op.Transpose(weight, perm=[1, 0], name="linear")
    if bias:
        res = g.op.MatMul(x, weight_transposed)
        res = g.op.Add(res, bias, outputs=outputs)
    else:
        res = g.op.MatMul(x, weight_transposed, outputs=outputs, name="linear")
    if sts:
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
    sts: bool,
    outputs: List[str],
    x: T,
    dim: int = -1,
    unnamed: bool = False,
    dtype: Optional["torch.dtype"] = None,  # noqa: F821
) -> T:
    assert not unnamed, "Not implemented when the third parameter is False"
    if dtype is not None:
        itype = torch_dtype_to_onnx_dtype(dtype)
        xc = g.op.Cast(x, to=itype, name="log_softmax")
    else:
        itype = None
        xc = x
    res = g.op.LogSoftmax(xc, axis=dim, outputs=outputs)
    if sts:
        set_type_shape_unary_op(g, res, xc, itype=itype)
    return res


def aten__log_softmax_backward_data(
    g: GraphBuilder,
    sts: bool,
    outputs: List[str],
    grad_output: T,
    output: T,
    dim: int,
    input_dtype: Optional["torch.dtype"] = None,  # noqa: F821
):
    if input_dtype is not None:
        itype = torch_dtype_to_onnx_dtype(input_dtype)
        grad_outputc = g.op.Cast(
            grad_output, to=itype, name="log_softmax_backward_data"
        )
    else:
        itype = None
        grad_outputc = grad_output
    res = g.op.Sub(
        grad_output,
        g.op.Mul(
            g.op.Exp(output, name="log_softmax_backward_data"),
            g.op.ReduceSum(
                grad_output,
                np.array([dim], dtype=np.int64),
                keepdims=True,
                name="log_softmax_backward_data",
            ),
            name="log_softmax_backward_data",
        ),
        outputs=outputs,
        name="log_softmax_backward_data",
    )
    if sts:
        set_type_shape_unary_op(g, res, grad_outputc, itype=itype)
    return res


def aten_lt(g: GraphBuilder, sts: bool, outputs: List[str], x: T, y: T, name="lt") -> T:
    x, y = prepare_inputs_homogeneous_operator(g, x, y)
    res = g.op.Less(x, y, outputs=outputs, name=name)
    if sts:
        set_type_shape_binary_op(g, outputs[0], x, y, cmp_op=True)
    return res


def aten_lt_Tensor(g: GraphBuilder, sts: bool, outputs: List[str], x: T, y: T) -> T:
    return aten_lt(g, sts, outputs, x, y, name="lt_Tensor")


def aten_matmul(g: GraphBuilder, sts: bool, outputs: List[str], x: T, y: T) -> T:
    res = g.op.MatMul(x, y, outputs=outputs)
    if sts:
        set_type_shape_binary_op(g, outputs[0], x, y)
    return res


def aten_masked_fill_Scalar(
    g: GraphBuilder,
    sts: bool,
    outputs: List[str],
    x: T,
    mask: T,
    value,
    name="masked_fill_Scalar",
) -> T:
    dt = g.get_type(mask)
    if dt != TensorProto.BOOL:
        cmask = g.op.Cast(mask, to=TensorProto.BOOL, name=name)
    else:
        cmask = mask
    dtx = g.get_type(x)
    avalue = np.array([value], dtype=tensor_dtype_to_np_dtype(dtx))
    res = g.op.Where(cmask, avalue, x, name=name)
    if sts:
        g.set_type(res, dtx)
        if g.has_shape(mask):
            g.set_shape(res, g.get_shape(mask))
        else:
            g.set_rank(res, g.get_rank(mask))
    return res


def _aten_max_pool_onnx(
    g: GraphBuilder,
    sts: bool,
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
    # self_rank_is_unbatched_rank = Rank(self) == unbatched_rank
    # if self_rank_is_unbatched_rank:  # C,H,W -> N,C,H,W and N=1
    #     self = op.Unsqueeze(self, op.Constant(value_ints=[0]))

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
    #    pool_result = op.Squeeze(pool_result, op.Constant(value_ints=[0]))

    return pool_result


def aten_max_pool2d(
    g: GraphBuilder,
    sts: bool,
    outputs: List[str],
    x: T,
    kernel_size: Sequence[int],
    stride: Sequence[int] = (),
    padding: Sequence[int] = (0, 0),
    dilation: Sequence[int] = (1, 1),
    ceil_mode: bool = False,
) -> T:
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
    sts: bool,
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

    ends = g.op.Constant(value_ints=n_dims_one, name=name)
    starts = g.op.Constant(value_ints=n_dims_zero, name=name)
    axes = g.op.Constant(value_ints=n_dims_axes, name=name)

    delta = g.op.Slice(flatten_indices, starts, ends, axes, name=name)
    indices = g.op.Sub(indices, delta, name=name)

    if is_unbatched_rank:
        pool_result = g.op.Squeeze(
            pool_result, g.op.Constant(value_ints=[0]), name=name
        )
        indices = g.op.Squeeze(indices, g.op.Constant(value_ints=[0]), name=name)

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
    sts: bool,
    outputs: List[str],
    x: T,
    kernel_size: Sequence[int],
    stride: Sequence[int] = (),
    padding: Sequence[int] = (0, 0),
    dilation: Sequence[int] = (1, 1),
    ceil_mode: bool = False,
) -> Tuple[T, T]:
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
    sts: bool,
    outputs: List[str],
    x: T,
    dim: Optional[Union[int, List[int]]] = None,
    keepdim: bool = False,
    dtype=None,
) -> T:
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
    if sts:
        set_type_shape_reduce_op(g, outputs[0], x, keepdim=keepdim)
    return result


def aten_mm(g: GraphBuilder, sts: bool, outputs: List[str], x: T, y: T) -> T:
    return g.op.MatMul(x, y, outputs=outputs, name="mm")


def aten_mul(
    g: GraphBuilder, sts: bool, outputs: List[str], x: T, y: T, name="mul"
) -> T:
    if g.get_type(x) == TensorProto.BOOL and g.get_type(y) == TensorProto.BOOL:
        res = g.op.And(x, y, name="mul_and", outputs=outputs)
    else:
        res, x, y = prepare_inputs_homogeneous_operator(
            g, x, y, f=g.op.Mul, name="mul", outputs=outputs
        )
    if sts:
        set_type_shape_binary_op(g, res, x, y)
    return res


def aten_mul_Scalar(g: GraphBuilder, sts: bool, outputs: List[str], x: T, y: T) -> T:
    return aten_mul(g, sts, outputs, x, y, name="mul_Scalar")


def aten_mul_Tensor(g: GraphBuilder, sts: bool, outputs: List[str], x: T, y: T) -> T:
    return aten_mul(g, sts, outputs, x, y, name="mul_Tensor")


def aten_neg(g: GraphBuilder, sts: bool, outputs: List[str], x: T) -> T:
    res = g.make_node("Neg", [x], outputs)
    if sts:
        set_type_shape_unary_op(g, res, x)
    return res


def aten_new_zeros(
    g: GraphBuilder,
    sts: bool,
    outputs: List[str],
    x: T,
    size: T,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    name: str = "seros",
) -> T:
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
    g: GraphBuilder, sts: bool, outputs: List[str], x: T, name: str = "not"
) -> T:
    res = g.make_node("Not", [x], outputs, name=name)
    if sts:
        set_type_shape_unary_op(g, res, x)
    return res


def aten_not_(
    g: GraphBuilder, sts: bool, outputs: List[str], x: T, name: str = "not"
) -> T:
    return aten_not(g, sts, outputs, x, name="not_")


def aten_ones(
    g: GraphBuilder,
    sts: bool,
    outputs: List[str],
    size: T,
    dtype: int = TensorProto.FLOAT,
    layout=None,
    device=None,
    pin_memory=None,
) -> T:
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
        name="ones",
    )
    if sts:
        g.set_type(res, dtype)
        if new_shape:
            g.set_shape(res, new_shape)
    return res


def aten_permute(
    g: GraphBuilder, sts: bool, outputs: List[str], x: T, dims: Sequence[int]
) -> T:
    if not dims:
        return g.op.Transpose(x, outputs=outputs, name="permute")

    dims = [axis + len(dims) if axis < 0 else axis for axis in dims]
    return g.op.Transpose(x, perm=dims, outputs=outputs, name="permute")


def aten_pow_Tensor_Scalar(
    g: GraphBuilder, sts: bool, outputs: List[str], x: T, exponent: T
) -> T:
    if isinstance(exponent, (int, float)):
        if exponent == 1:
            # The node is removed.
            return g.op.Identity(x, outputs=outputs)
        exponent = np.array([exponent])
    if isinstance(exponent, np.ndarray):
        if g.has_type(x):
            exponent = exponent.astype(tensor_dtype_to_np_dtype(g.get_type(x)))
        else:
            exponent = g.op.CastLike(exponent, x)
    res = g.op.Pow(x, exponent, outputs=outputs)
    if sts:
        set_type_shape_unary_op(g, outputs[0], x)
    return res


def aten_relu(g: GraphBuilder, sts: bool, outputs: List[str], x: T) -> T:
    return g.op.Relu(x, outputs=outputs)


def aten_repeat(
    g: GraphBuilder,
    sts: bool,
    outputs: List[str],
    x: T,
    repeats: T,
    name: str = "repeat",
) -> T:
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
    if sts:
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


def aten_rsqrt(g: GraphBuilder, sts: bool, outputs: List[str], x: T) -> T:
    ext = g.make_node("Sqrt", [x], name="rsqrt")
    res = g.make_node("Reciprocal", ext, outputs, name="rsqrt")
    if sts:
        set_type_shape_unary_op(g, outputs[0], x)
    return res


def aten_rsub(
    g: GraphBuilder,
    sts: bool,
    outputs: List[str],
    x: T,
    y: T,
    alpha: float = 1,
) -> T:
    assert alpha == 1, f"Not implemented with alpha={alpha}"
    return aten_sub(g, sts, outputs, y, x)


def aten_rsub_Scalar(
    g: GraphBuilder,
    sts: bool,
    outputs: List[str],
    x: T,
    y: T,
    alpha: float = 1,
) -> T:
    assert alpha == 1, f"Not implemented with alpha={alpha}"
    return aten_sub(g, sts, outputs, y, x)


def aten_setitem(
    g: GraphBuilder,
    sts: bool,
    outputs: List[str],
    x: T,
    indices: Tuple[Any, ...],
    values: T,
) -> T:
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

    # if sts:
    #    g.set_type(res, g.get_type(x))
    #    if g.has_shape(x):
    #        g.set_shape(res, g.get_shape(x))
    #    else:
    #        g.set_rank(res, g.get_rank(x))
    # return res
    raise RuntimeError(
        f"setitem not implemented for indices={indices}{g.get_debug_msg()}"
    )


def aten_sigmoid(g: GraphBuilder, sts: bool, outputs: List[str], x: T) -> T:
    res = g.op.Sigmoid(x, outputs=outputs)
    if sts:
        set_type_shape_unary_op(g, outputs[0], x)
    return res


def aten_sigmoid_backward(
    g: GraphBuilder, sts: bool, outputs: List[str], out_grad: T, y: T
) -> T:
    """
    See https://github.com/pytorch/pytorch/blob/main/torch/_decomp/decompositions.py#L108.
    conj_physical = identity for real number.

    ::

        return out_grad * (y * (1 - y)).conj_physical()
    """
    dtype = tensor_dtype_to_np_dtype(g.get_type(y))
    _1y = g.op.Sub(np.array([1], dtype=dtype), y, name="sigmoid_backward")
    y1y = g.op.Mul(y, _1y, name="sigmoid_backward")
    res = g.op.Mul(out_grad, y1y, outputs=outputs, name="sigmoid_backward")
    if sts:
        set_type_shape_unary_op(g, outputs[0], y)
    return res


def aten_silu(
    g: GraphBuilder, sts: bool, outputs: List[str], x: T, inplace: bool = False
) -> T:
    assert (
        not inplace
    ), f"inplace computation is not allowed with onnx{g.get_debug_msg()}"
    res = g.op.Mul(x, g.op.Sigmoid(x, name="silu"), outputs=outputs, name="silu")
    if sts:
        set_type_shape_unary_op(g, res, x)
    return res


def aten_sin(g: GraphBuilder, sts: bool, outputs: List[str], x: T) -> T:
    res = g.make_node("Sin", [x], outputs)
    if sts:
        set_type_shape_unary_op(g, outputs[0], x)
    return res


def aten_sinh(g: GraphBuilder, sts: bool, outputs: List[str], x: T) -> T:
    res = g.make_node("Sinh", [x], outputs)
    if sts:
        set_type_shape_unary_op(g, outputs[0], x)
    return res


def aten_slice_Tensor(
    g: GraphBuilder,
    sts: bool,
    outputs: List[str],
    x: T,
    dim: int = 0,
    start: int = 0,
    end: Optional[int] = None,
    step: Optional[int] = None,
) -> T:
    assert isinstance(dim, int), f"aten_slice_Tensor not implemented for dim={dim!r}"
    assert g.is_dynamic_dimension(start), (
        f"aten_slice_Tensor not implemented for start={start!r}" f"{g.get_debug_msg()}"
    )
    assert end is None or g.is_dynamic_dimension(end), (
        f"aten_slice_Tensor not implemented for end={end!r}" f"{g.get_debug_msg()}"
    )
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
    if sts:
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
    sts: bool,
    outputs: List[str],
    grad_output: T,
    input_sizes: List[int],
    dim: int,
    start: int,
    end: int,
    step: int,
    name: str = "slice_backward",
) -> T:
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

    res = g.op.Concat(*inputs, axis=dim, name=name)

    if sts:
        g.set_type(res, g.get_type(grad_output))
        g.set_shape(res, tuple(input_sizes))
    return res


def _aten_slice_scatter_static(
    g: GraphBuilder,
    sts: bool,
    outputs: List[str],
    x: T,
    src: T,
    dim: int = 0,
    start: Optional[int] = None,
    end: Optional[int] = None,
    step: Optional[int] = None,
    name="slice_scatter_static",
) -> T:
    assert g.has_shape(
        x
    ), f"This implementation only works if shape of {x!r} is known{g.get_debug_msg()}"
    # step 1
    assert start is None or g.is_dynamic_dimension(start), (
        f"slice_scatter not implemented for start={start}" f"{g.get_debug_msg()}"
    )
    assert end is None or g.is_dynamic_dimension(end), (
        f"slice_scatter not implemented for end={end}" f"{g.get_debug_msg()}"
    )
    assert step is None or is_static_dimension(step), (
        f"slice_scatter not implemented for end={step}" f"{g.get_debug_msg()}"
    )
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

    v = (0 if dim < 0 else len(shape)) - dim
    if v > 1:
        r = tuple(np.arange(1, v, 1))
        if isinstance(index_2, str):
            # dynamic shapes
            index_base = g.op.Expand(
                g.op.UnsqueezeAnyOpset(index_2, np.array([1], dtype=np.int64)),
                np.array(r, dtype=np.int64),
            )
        else:
            index_base = np.expand_dims(index_2, r)
    else:
        index_base = index_2

    # Step 3: Expand the indices.
    shape_expand = g.op.ScatterElements(
        np.array(shape, dtype=np.int64),
        np.array([dim], dtype=np.int64),
        np.array([1], dtype=np.int64),
        name=name,
    )
    indices = g.op.Expand(index_base, shape_expand, name=name)

    # Step 4: final ScatterElements.
    res = g.op.ScatterElements(x, indices, src, axis=dim, name=name)
    if sts:
        g.set_type(res, g.get_type(x))
        g.set_shape(res, shape)
    return res


def _aten_slice_scatter_dynamic(
    g: GraphBuilder,
    sts: bool,
    outputs: List[str],
    x: T,
    src: T,
    dim: int = 0,
    start: Optional[int] = None,
    end: Optional[int] = None,
    step: Optional[int] = None,
    name="slice_scatter_dynamic",
) -> T:
    # step 1
    assert start is None or g.is_dynamic_dimension(start), (
        f"slice_scatter not implemented for start={start}" f"{g.get_debug_msg()}"
    )
    assert end is None or g.is_dynamic_dimension(end), (
        f"slice_scatter not implemented for end={end}" f"{g.get_debug_msg()}"
    )
    assert step is None or is_static_dimension(step), (
        f"slice_scatter not implemented for end={step}" f"{g.get_debug_msg()}"
    )

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

    v = (0 if dim < 0 else g.get_rank(x)) - dim
    if v > 1:
        r = tuple(np.arange(1, v, 1))
        index_base = g.op.Expand(
            g.op.UnsqueezeAnyOpset(index_2, np.array([1], dtype=np.int64)),
            np.array(r, dtype=np.int64),
        )
    else:
        index_base = index_2
    # Step 3: Expand the indices.
    shape_expand = g.op.ScatterElements(
        shape,
        np.array([dim], dtype=np.int64),
        np.array([1], dtype=np.int64),
        name=name,
    )
    indices = g.op.Expand(index_base, shape_expand, name=name)

    # Step 4: final step
    res = g.op.ScatterElements(x, indices, src, axis=dim, name=name)
    if sts:
        g.set_type(res, g.get_type(x))
        g.set_rank(res, g.get_rank(x))
    return res


def aten_slice_scatter(
    g: GraphBuilder,
    sts: bool,
    outputs: List[str],
    x: T,
    src: T,
    dim: int = 0,
    start: Optional[int] = None,
    end: Optional[int] = None,
    step: Optional[int] = None,
    name: Optional[str] = None,
) -> T:

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


def aten_softmax(
    g: GraphBuilder,
    sts: bool,
    outputs: List[str],
    x: T,
    dim: int = -1,
    dtype: Optional["torch.dtype"] = None,  # noqa: F821
) -> T:
    if dtype is not None:
        itype = torch_dtype_to_onnx_dtype(dtype)
        xc = g.op.Cast(x, to=itype, name="softmax")
    else:
        itype = None
        xc = x
    res = g.op.Softmax(xc, axis=dim, outputs=outputs)
    if sts:
        set_type_shape_unary_op(g, res, xc, itype=itype)
    return res


def aten__softmax(
    g: GraphBuilder,
    sts: bool,
    outputs: List[str],
    x: T,
    dim: int = -1,
    half_to_float: bool = False,
) -> T:
    assert not half_to_float, f"Unexpected value for half_to_float={half_to_float!r}"
    res = g.op.Softmax(x, axis=dim, outputs=outputs, name="_softmax")
    if sts:
        set_type_shape_unary_op(g, res, x)
    return res


def aten__softmax_backward_data(
    g: GraphBuilder,
    sts: bool,
    outputs: List[str],
    grad_output: T,
    y: T,
    dim: int,
    input_dtype: Optional["torch.dtype"] = None,  # noqa: F821
) -> T:
    if input_dtype is not None:
        itype = torch_dtype_to_onnx_dtype(input_dtype)
        grad_outputc = g.op.Cast(
            grad_output, to=itype, name="log_softmax_backward_data"
        )
    else:
        itype = None
        grad_outputc = grad_output

    new_grad_output = g.op.Mul(grad_outputc, y)
    sums = g.op.ReduceSum(
        new_grad_output,
        np.array([dim], dtype=np.int64),
        keepdims=1,
        name="softmax_backward_data",
    )
    temp = g.op.Mul(y, sums, name="softmax_backward_data")
    res = g.op.Sub(new_grad_output, temp, outputs=outputs, name="softmax_backward_data")
    if sts:
        set_type_shape_unary_op(g, res, grad_outputc, itype=itype)
    return res


def aten_sub(
    g: GraphBuilder, sts: bool, outputs: List[str], x: T, y: T, name="sub"
) -> T:
    x, y = prepare_inputs_homogeneous_operator(g, x, y)
    res = g.op.Sub(x, y, outputs=outputs, name=name)
    if sts:
        set_type_shape_binary_op(g, outputs[0], x, y)
    return res


def aten_sub_Tensor(
    g: GraphBuilder, sts: bool, outputs: List[str], x: T, y: T, alpha: float
) -> T:
    assert (
        alpha == 1
    ), f"sub_Tensor not implemented for alpha={alpha}{g.get_debug_msg()}"
    return aten_sub(g, sts, outputs, x, y, name="sub_Tensor")


def aten_sum(
    g: GraphBuilder,
    sts: bool,
    outputs: List[str],
    x: T,
    dim: Optional[Union[int, List[int]]] = None,
    keepdim: bool = False,
    dtype=None,
) -> T:
    if dtype is not None:
        itype = torch_dtype_to_onnx_dtype(dtype)
        xc = g.op.Cast(x, to=itype)
    else:
        xc = x
    if dim is None:
        result = g.op.ReduceSumAnyOpset(
            xc, keepdims=keepdim, outputs=outputs, name="sum"
        )
    else:
        if isinstance(dim, int):
            adim = np.array([dim], dtype=np.int64)
        else:
            adim = np.array(dim, dtype=np.int64)
        result = g.op.ReduceSumAnyOpset(
            xc, adim, keepdims=keepdim, outputs=outputs, name="sum"
        )
    if sts:
        set_type_shape_reduce_op(g, outputs[0], x, keepdim=keepdim)
    return result


def aten_sum_dim_IntList(
    g: GraphBuilder,
    sts: bool,
    outputs: List[str],
    x: T,
    dim: Optional[Union[int, List[int]]],
    keepdim: bool,
    dtype: Optional["torch.dtype"] = None,  # noqa: F821
) -> T:
    if dtype is None:
        return aten_sum(g, sts, outputs, x, dim, keepdim)

    res = aten_sum(g, sts, None, x, dim, keepdim)
    itype = torch_dtype_to_onnx_dtype(dtype)
    result = g.op.Cast(res, to=itype, outputs=outputs, name="sum_dim_IntList")
    return result


def aten__to_copy(
    g: GraphBuilder,
    sts: bool,
    outputs: List[str],
    x: T,
    dtype: Optional["torch.dtype"] = None,  # noqa: F821
    layout=None,
    device=None,
    pin_memory=None,
    non_blocking=False,
    memory_format=None,
) -> T:
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
    if sts:
        set_type_shape_unary_op(g, res, x, itype=itype)
    return res


def aten_t(g: GraphBuilder, sts: bool, outputs: List[str], x: T, name: str = "t") -> T:
    res = g.op.Transpose(x, perm=[1, 0], outputs=outputs, name=name)
    if sts:
        g.set_type(res, g.get_type(x))
        if g.has_shape(x):
            shape = list(g.get_shape(x))
            assert len(shape) == 2, f"Unexpected shape={shape}, should be 2D"
            shape[0], shape[1] = shape[1], shape[0]
            g.set_shape(res, tuple(shape))
        else:
            g.set_rank(res, g.get_rank(x))
    return res


def aten_tan(g: GraphBuilder, sts: bool, outputs: List[str], x: T) -> T:
    res = g.make_node("Tan", [x], outputs)
    if sts:
        set_type_shape_unary_op(g, outputs[0], x)
    return res


def aten_tanh(g: GraphBuilder, sts: bool, outputs: List[str], x: T) -> T:
    res = g.make_node("Tanh", [x], outputs)
    if sts:
        set_type_shape_unary_op(g, outputs[0], x)
    return res


def aten_tanh_backward(
    g: GraphBuilder, sts: bool, outputs: List[str], out_grad: T, y: T
) -> T:
    # out_grad * (1 - y * y)
    dtype = tensor_dtype_to_np_dtype(g.get_type(y))
    yy = g.op.Pow(y, np.array([2], dtype=np.int64), name="tanh_backward")
    _1yy = g.op.Sub(np.array([1], dtype=dtype), yy, name="tanh_backward")
    res = g.op.Mul(out_grad, _1yy, outputs=outputs, name="tanh_backward")
    if sts:
        set_type_shape_unary_op(g, outputs[0], y)
    return res


def _aten_tensor_int1(
    g: GraphBuilder,
    sts: bool,
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
    if sts:
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
    sts: bool,
    outputs: List[str],
    x: T,
    indices: Optional[Tuple[Any, ...]] = None,
) -> T:
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


def aten_transpose(
    g: GraphBuilder, sts: bool, outputs: List[str], input_name: T, dim0: int, dim1: int
) -> T:
    perm = list(range(g.rank(input_name)))
    perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
    res = g.make_node("Transpose", [input_name], outputs, perm=perm)
    if sts:
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
    sts: bool,
    outputs: List[str],
    input_name: T,
    dim0: int,
    dim1: int,
) -> T:
    return aten_transpose(g, sts, outputs, input_name, dim0, dim1)


def aten_truediv(g: GraphBuilder, sts: bool, outputs: List[str], x: T, y: T) -> T:
    x, y = prepare_inputs_homogeneous_operator(g, x, y)
    res = g.op.Div(x, y, outputs=outputs, name="truediv")
    if sts:
        set_type_shape_binary_op(g, outputs[0], x, y)
    return res


def aten_unsqueeze(g: GraphBuilder, sts: bool, outputs: List[str], x: T, dim: int) -> T:
    assert isinstance(dim, int), f"Not implemented for dim={dim!r}"
    res = g.op.UnsqueezeAnyOpset(x, np.array([dim], dtype=np.int64), outputs=outputs)
    if sts:
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
    sts: bool,
    outputs: List[str],
    x: T,
    size: T,
    node_name: str = "view",
) -> T:
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
        if sts:
            set_type_shape_reshape(g, res, x, asize)
        return res

    csize = g.op.Cast(size, to=TensorProto.INT64, name=node_name)
    res = g.op.Reshape(x, csize, outputs=outputs, name=node_name)
    if sts:
        set_type_shape_reshape(g, res, x, size)
    return res


def aten__unsafe_view(
    g: GraphBuilder, sts: bool, outputs: List[str], x: T, size: T
) -> T:
    return aten_view(g, sts, outputs, x, size, node_name="_unsafe_view")


def aten_zeros(
    g: GraphBuilder,
    sts: bool,
    outputs: List[str],
    size: T,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    name: str = "seros",
) -> T:
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


def prims_broadcast_in_dim(
    g: GraphBuilder,
    sts: bool,
    outputs: List[str],
    a: T,
    shape: List[int],
    broadcast_dimensions: List[int],
) -> T:
    """
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
            uns.append(idx + len(uns))

    unsqueezed = g.op.UnsqueezeAnyOpset(
        a, np.array(uns, dtype=np.int64), name="broadcast_in_dim"
    )
    res = g.op.Expand(
        unsqueezed,
        np.array(shape, dtype=np.int64),
        name="broadcast_in_dim",
        outputs=outputs,
    )

    if sts:
        g.set_type(res, g.get_type(a))
        g.set_shape(res, shape)

    return res


def prims_clone(
    g: GraphBuilder,
    sts: bool,
    outputs: List[str],
    x: T,
    memory_format: Optional[str] = None,
) -> T:
    return aten_clone(
        g, sts, outputs, x, memory_format=memory_format, name="prims_clone"
    )


def prims_transpose(
    g: GraphBuilder, sts: bool, outputs: List[str], input_name: T, perm: List[int]
) -> T:
    res = g.make_node("Transpose", [input_name], outputs, perm=list(perm))
    if sts:
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
