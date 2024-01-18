from typing import List, Optional, Sequence, Tuple
import numpy as np
from onnx import TensorProto
from onnx.helper import tensor_dtype_to_np_dtype
from .graph_builder import GraphBuilder
from ._aten_helper import (
    _adjust_attributes_of_max_pool,
    set_shape_type_unary_op,
    set_shape_type_binary_op,
    onnx_dtype_to_torch_dtype,
    torch_dtype_to_onnx_dtype,
)


T = str


def aten_abs(g: GraphBuilder, set_shape_type: bool, outputs: List[str], x: T) -> T:
    res = g.make_node("Abs", [x], outputs)
    if set_shape_type:
        set_shape_type_unary_op(g, outputs[0], x)
    return res


def aten_add(
    g: GraphBuilder, set_shape_type: bool, outputs: List[str], x: T, y: T
) -> T:
    res = g.op.Add(x, y, outputs=outputs)
    if set_shape_type:
        set_shape_type_binary_op(g, outputs[0], x, y)
    return res


def aten_addmm(
    g: GraphBuilder,
    set_shape_type: bool,
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


def aten_all(g: GraphBuilder, set_shape_type: bool, outputs: List[str], x: T) -> T:
    res = g.op.Cast(
        g.op.ReduceMin(g.op.Cast(x, to=TensorProto.INT32, name="all"), name="all"),
        to=TensorProto.BOOL,
        outputs=outputs,
        name="all",
    )
    if set_shape_type:
        g.set_type(res, TensorProto.BOOL)
        g.set_shape(res, tuple())
    return res


def aten_arange(
    g: GraphBuilder,
    set_shape_type: bool,
    outputs: List[str],
    start: Optional[int] = None,
    end: Optional[int] = None,
    step: int = 1,
    dtype: Optional["torch.dtype"] = None,  # noqa: F821
) -> T:
    if start is not None and end is None:
        end = start
        start = 0
    if dtype is None:
        if isinstance(end, str):
            itype = g.get_type(end)
        else:
            itype = torch_dtype_to_onnx_dtype(type(end))
    else:
        itype = torch_dtype_to_onnx_dtype(dtype)
    dtype = onnx_dtype_to_torch_dtype(itype)
    npdtype = tensor_dtype_to_np_dtype(itype)
    if step is None:
        step = 1
    assert start is not None, "start cannot be None"
    assert end is not None, "end cannot be None"
    assert step is not None, "step cannot be None"
    if isinstance(start, str):
        i_start = start
    else:
        i_start = np.array([start], dtype=npdtype)
    if isinstance(end, str):
        i_end = end
    else:
        i_end = np.array([end], dtype=npdtype)
    if isinstance(step, str):
        i_step = step
    else:
        i_step = np.array([step], dtype=npdtype)
    res = g.op.Range(i_start, i_end, i_step, outputs=outputs)
    if set_shape_type:
        g.set_type(res, itype)
        if isinstance(end, str) or isinstance(start, str) or isinstance(step, str):
            g.set_rank(res, 1)
        else:
            g.set_shape(res, ((end - start) // step,))
    return res


def aten_cat(
    g: GraphBuilder,
    set_shape_type: bool,
    outputs: List[str],
    tensors: Tuple[T, ...],
    dim: int = 0,
) -> T:
    res = g.op.Concat(*tensors, axis=dim, outputs=outputs)
    if set_shape_type:
        dt0 = g.get_type(tensors[0])
        assert all(map(lambda t: g.get_type(t) == dt0, tensors))
        r0 = g.get_rank(tensors[0])
        assert all(map(lambda t: g.get_rank(t) == r0, tensors))
        g.set_type(outputs[0], dt0)
        g.set_rank(outputs[0], r0)
    return res


def aten_convolution(
    g: GraphBuilder,
    set_shape_type: bool,
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
        raise NotImplementedError(
            f"aten_convolution does not support transposed={transposed}."
        )
    if output_padding and (min(output_padding) != 0 or max(output_padding) != 0):
        raise NotImplementedError(
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
        weight_dim_0 = g.make_node("Shape", [weight], start=0, end=1)
        bias_shape = g.make_node(
            "Expand", [weight_dim_0], g.make_node("Constant", value_ints=[1])
        )
        zero = g.make_node("CastLike", [np.array([0.0]), input])
        bias = g.make_node("Expand", [zero, bias_shape])

    # if Rank(input) != Rank(weight):
    #    input = op.Unsqueeze(input, op.Constant(value_ints=[0]))

    return g.make_node(
        "Conv",
        [input, weight, bias],
        outputs,
        strides=strides,
        pads=pads,
        group=groups,
        dilations=dilations,
    )


def aten_conv2d(
    g: GraphBuilder,
    set_shape_type: bool,
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
        outputs,
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


def aten_dropout(
    g: GraphBuilder,
    set_shape_type: bool,
    outputs: List[str],
    x: T,
    p: T = 0.5,  # float
    train: T = False,  # bool
) -> T:
    """dropout(Tensor input, float p, bool train) -> Tensor"""

    # if IsScalar(input):
    #     input = op.Reshape(input, op.Constant(value_ints=[-1]))
    #     result, _ = op.Dropout(input, p, train)
    #     result = op.Squeeze(result)
    # else:
    #     result, _ = op.Dropout(x, p, train)

    if len(outputs) == 1:
        outputs = outputs.copy()
        outputs.append("")
    if isinstance(p, float):
        p = np.array(p, dtype=np.float64)
    if isinstance(train, bool):
        train = np.array(train, dtype=np.bool_)
    result, _ = g.op.Dropout(x, p, train, outputs=outputs)
    if set_shape_type:
        set_shape_type_unary_op(g, outputs[0], x)
    return result


def aten_embedding(
    g: GraphBuilder,
    set_shape_type: bool,
    outputs: List[str],
    weight: T,
    indices: T,
    padding_idx: Optional[int] = None,
    max_norm: Optional[int] = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> T:
    if padding_idx is not None or scale_grad_by_freq or sparse or max_norm is not None:
        raise NotImplementedError(
            f"Not implemented when padding_idx={padding_idx}, or "
            f"scale_grad_by_freq={scale_grad_by_freq} or sparse={sparse} "
            f"or max_norm={max_norm} or norm_type={norm_type} "
            f"are different from the default values."
        )
    res = g.op.Gather(weight, indices, outputs=outputs)
    if set_shape_type:
        g.set_type(res, g.get_type(weight))
    return res


def aten_eq(g: GraphBuilder, set_shape_type: bool, outputs: List[str], x: T, y: T) -> T:
    res = g.op.Equal(x, y, outputs=outputs)
    if set_shape_type:
        set_shape_type_binary_op(g, outputs[0], x, y)
    return res


def aten_flatten(
    g: GraphBuilder,
    set_shape_type: bool,
    outputs: List[str],
    x: T,
    start_dim: int = 1,
    end_dim: int = -1,
) -> T:
    if start_dim != 0:
        if start_dim == 1 and end_dim == -1:
            shape = g.op.Shape(x)
            take = g.op.GatherElements(shape, np.array([0], dtype=np.int64), axis=0)
            resh = g.op.Concat(take, np.array([-1], dtype=np.int64), axis=0)
            return g.op.Reshape(x, resh, outputs=outputs)
        raise NotImplementedError(
            f"start_dim={start_dim}, end_dim={end_dim} not supported."
        )
    if end_dim == -1:
        return g.make_node("Flatten", [x], outputs)
    return g.make_node("Flatten", [x], outputs, to=end_dim)


def aten_linear(
    g: GraphBuilder,
    set_shape_type: bool,
    outputs: List[str],
    input: T,
    weight: T,
    bias: T = None,
) -> T:
    weight_transposed = g.op.Transpose(weight, perm=[1, 0])
    if bias:
        res = g.op.MatMul(input, weight_transposed)
        return g.op.Add(res, bias, outputs=outputs)
    return g.op.MatMul(input, weight_transposed, outputs=outputs)


def aten_matmul(
    g: GraphBuilder, set_shape_type: bool, outputs: List[str], x: T, y: T
) -> T:
    res = g.op.MatMul(x, y, outputs=outputs)
    if set_shape_type:
        set_shape_type_binary_op(g, outputs[0], x, y)
    return res


def _aten_max_pool_onnx(
    g: GraphBuilder,
    set_shape_type: bool,
    outputs: List[str],
    x: T,
    kernel_shape: Sequence[int],
    strides: Sequence[int],
    pads: Sequence[int],
    dilations: Sequence[int],
    ceil_mode: bool,
    unbatched_rank: int,
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
    )

    # if self_rank_is_unbatched_rank:
    #    pool_result = op.Squeeze(pool_result, op.Constant(value_ints=[0]))

    return pool_result


def aten_max_pool2d(
    g: GraphBuilder,
    set_shape_type: bool,
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
        g, outputs, x, kernel_shape, strides, pads, dilations, ceil_mode, 3
    )


def _aten_max_pool_with_indices_onnx(
    g: GraphBuilder,
    set_shape_type: bool,
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
) -> Tuple[T, T]:
    if isinstance(ceil_mode, str):
        raise TypeError(f"Unexpected ceil_mode={ceil_mode}")
    is_unbatched_rank = g.rank(x) == unbatched_rank
    if is_unbatched_rank:
        x = g.op.Unsqueeze(x, axes=0)

    pool_result, indices = g.op.MaxPool(
        x,
        ceil_mode=ceil_mode,
        dilations=dilation,
        kernel_shape=kernel_size,
        pads=padding,
        strides=stride,
    )
    _, flatten_indices = g.op.MaxPool(
        x, dilations=dilation, kernel_shape=n_dims_one, strides=n_dims_one
    )

    ends = g.op.Constant(value_ints=n_dims_one)
    starts = g.op.Constant(value_ints=n_dims_zero)
    axes = g.op.Constant(value_ints=n_dims_axes)

    delta = g.op.Slice(flatten_indices, starts, ends, axes)
    indices = g.op.Sub(indices, delta)

    if is_unbatched_rank:
        pool_result = g.op.Squeeze(pool_result, g.op.Constant(value_ints=[0]))
        indices = g.op.Squeeze(indices, g.op.Constant(value_ints=[0]))

    if outputs:
        if not isinstance(outputs, (tuple, list)):
            raise TypeError(
                f"Multiple outputs are expeted but type(outputs) is {type(outputs)}."
            )
        if len(outputs) != 2:
            raise ValueError(f"Multiple outputs are expeted but outputs is {outputs}.")
        return (
            g.op.Identity(pool_result, outputs=outputs[0]),
            g.op.Identity(indices, outputs=outputs[1]),
        )
    return pool_result, indices


def aten_max_pool2d_with_indices(
    g: GraphBuilder,
    set_shape_type: bool,
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
    )


def aten_mul(
    g: GraphBuilder, set_shape_type: bool, outputs: List[str], x: T, y: T
) -> T:
    res = g.op.Mul(x, y, outputs=outputs)
    if set_shape_type:
        set_shape_type_binary_op(g, outputs[0], x, y)
    return res


def aten_neg(g: GraphBuilder, set_shape_type: bool, outputs: List[str], x: T) -> T:
    res = g.make_node("Neg", [x], outputs)
    if set_shape_type:
        g.set_type(outputs[0], g.get_type(x))
        g.set_shape(outputs[0], g.get_shape(x))
    return res


def aten_permute(
    g: GraphBuilder, set_shape_type: bool, outputs: List[str], x: T, dims: Sequence[int]
) -> T:
    if not dims:
        return g.op.Transpose(x, outputs=outputs, name="permute")

    dims = [axis + len(dims) if axis < 0 else axis for axis in dims]
    return g.op.Transpose(x, perm=dims, outputs=outputs, name="permute")


def aten_relu(g: GraphBuilder, set_shape_type: bool, outputs: List[str], x: T) -> T:
    return g.op.Relu(x, outputs=outputs)


def aten_rsqrt(
    g: GraphBuilder,
    set_shape_type: bool,
    outputs: List[str],
    x: T,
) -> T:
    ext = g.make_node("Sqrt", [x], name="rsqrt")
    res = g.make_node("Reciprocal", ext, outputs, name="rsqrt")
    if set_shape_type:
        set_shape_type_unary_op(g, outputs[0], x)
    return res


def aten_sigmoid(g: GraphBuilder, set_shape_type: bool, outputs: List[str], x: T) -> T:
    res = g.op.Sigmoid(x, outputs=outputs)
    if set_shape_type:
        set_shape_type_unary_op(g, outputs[0], x)
    return res


def aten_silu(g: GraphBuilder, set_shape_type: bool, outputs: List[str], x: T) -> T:
    return g.op.Mul(x, g.op.Sigmoid(x, name="silu"), outputs=outputs, name="silu")


def aten_softmax(
    g: GraphBuilder, set_shape_type: bool, outputs: List[str], x: T, dim: int = -1
) -> T:
    res = g.op.Softmax(x, axis=dim, outputs=outputs)
    if set_shape_type:
        set_shape_type_unary_op(g, outputs[0], x)
    return res


def aten_t(g: GraphBuilder, set_shape_type: bool, outputs: List[str], x: T) -> T:
    return g.op.Transpose(x, perm=[1, 0], outputs=outputs, name="t")


def aten_truediv(
    g: GraphBuilder, set_shape_type: bool, outputs: List[str], x: T, y: T
) -> T:
    res = g.op.Div(x, y, outputs=outputs, name="truediv")
    if set_shape_type:
        set_shape_type_binary_op(g, outputs[0], x, y)
    return res


def aten_view(
    g: GraphBuilder, set_shape_type: bool, outputs: List[str], x: T, size: T
) -> T:
    if isinstance(size, (int, tuple, list)):
        size = [size] if isinstance(size, int) else list(size)
        size = np.array(size, dtype=np.int64)
        shape = g.make_initializer("", size)
        return g.op.Reshape(x, shape, outputs=outputs, name="view")
    size = g.op.Cast(size, to=TensorProto.INT64, name="view")
    return g.op.Reshape(x, size, outputs=outputs, name="view")
