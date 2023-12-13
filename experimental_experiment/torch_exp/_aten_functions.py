from typing import List, Sequence, Tuple
import numpy as np
from onnx import TensorProto
from .graph_builder import GraphBuilder
from ._aten_helper import _adjust_attributes_of_max_pool


T = str


def aten_abs(g: GraphBuilder, outputs: List[str], x: T) -> T:
    return g.make_node("Abs", [x], outputs)


def aten_addmm(
    g: GraphBuilder,
    outputs: List[str],
    a: T,
    b: T,
    c: T,
    beta: float = 1.0,
    alpha: float = 1.0,
) -> T:
    return g.op.Gemm(b, c, a, alpha=float(alpha), beta=float(beta), outputs=outputs)


def aten_convolution(
    g: GraphBuilder,
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


def aten_flatten(
    g: GraphBuilder, outputs: List[str], x: T, start_dim: int = 1, end_dim: int = -1
) -> T:
    if start_dim != 0:
        if start_dim == 1 and end_dim == -1:
            shape = g.op.Shape(x)
            take = g.op.GatherElements(shape, np.array([0], dtype=np.int64), axis=1)
            resh = g.op.Concat(take, np.array([-1], dtype=np.int64))
            return g.op.Reshape(x, resh, outputs=outputs)
        raise NotImplementedError(
            f"start_dim={start_dim}, end_dim={end_dim} not supported."
        )
    if end_dim == -1:
        return g.make_node("Flatten", [x], outputs)
    return g.make_node("Flatten", [x], outputs, to=end_dim)


def aten_linear(
    g: GraphBuilder, outputs: List[str], input: T, weight: T, bias: T = None
) -> T:
    weight_transposed = g.op.Transpose(weight, perm=[1, 0])
    if bias:
        res = g.op.MatMul(input, weight_transposed)
        return g.op.Add(res, bias, outputs=outputs)
    return g.op.MatMul(input, weight_transposed, outputs=outputs)


def _aten_max_pool_onnx(
    g: GraphBuilder,
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


def aten_permute(g: GraphBuilder, outputs: List[str], x: T, dims: Sequence[int]) -> T:
    if not dims:
        return g.op.Transpose(x, outputs=outputs)

    dims = [axis + len(dims) if axis < 0 else axis for axis in dims]
    return g.op.Transpose(x, perm=dims, outputs=outputs)


def aten_relu(g: GraphBuilder, outputs: List[str], x: T) -> T:
    return g.op.Relu(x, outputs=outputs)


def aten_t(g: GraphBuilder, outputs: List[str], x: T) -> T:
    return g.op.Transpose(x, perm=[1, 0], outputs=outputs)


def aten_view(g: GraphBuilder, outputs: List[str], x: T, size: T) -> T:
    if isinstance(size, (int, tuple, list)):
        size = [size] if isinstance(size, int) else list(size)
        size = np.array(size, dtype=np.int64)
        shape = g.make_initializer("", size)
        return g.op.Reshape(x, shape, outputs=outputs)
    size = g.op.Cast(size, to=TensorProto.INT64)
    return g.op.Reshape(x, size, outputs=outputs)
