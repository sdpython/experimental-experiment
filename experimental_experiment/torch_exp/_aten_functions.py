from typing import Any, List, Optional, Sequence, Tuple, Union
import numpy as np
from onnx import TensorProto
from onnx.helper import tensor_dtype_to_np_dtype
from onnx.numpy_helper import from_array
from ._exceptions import FunctionNotFoundError
from ._aten_helper import (
    _adjust_attributes_of_max_pool,
    set_shape_type_unary_op,
    set_shape_type_binary_op,
    set_shape_type_reduce_op,
    onnx_dtype_to_torch_dtype,
    prepare_inputs_homogeneous_operator,
    torch_dtype_to_onnx_dtype,
)
from .graph_builder import GraphBuilder


T = str


def aten_abs(g: GraphBuilder, set_shape_type: bool, outputs: List[str], x: T) -> T:
    res = g.make_node("Abs", [x], outputs)
    if set_shape_type:
        set_shape_type_unary_op(g, outputs[0], x)
    return res


def aten_acos(g: GraphBuilder, set_shape_type: bool, outputs: List[str], x: T) -> T:
    res = g.make_node("Acos", [x], outputs)
    if set_shape_type:
        set_shape_type_unary_op(g, outputs[0], x)
    return res


def aten_acosh(g: GraphBuilder, set_shape_type: bool, outputs: List[str], x: T) -> T:
    res = g.make_node("Acosh", [x], outputs)
    if set_shape_type:
        set_shape_type_unary_op(g, outputs[0], x)
    return res


def aten_add(
    g: GraphBuilder, set_shape_type: bool, outputs: List[str], x: T, y: T
) -> T:
    x, y = prepare_inputs_homogeneous_operator(g, x, y)
    res = g.op.Add(x, y, outputs=outputs)
    if set_shape_type:
        set_shape_type_binary_op(g, outputs[0], x, y)
    return res


def aten_add_Scalar(
    g: GraphBuilder,
    set_shape_type: bool,
    outputs: List[str],
    x: T,
    y: T,
    alpha: Optional[Any] = None,
) -> T:
    assert alpha in (None, 1), f"alpha={alpha}, not implemented"
    x, y = prepare_inputs_homogeneous_operator(g, x, y)
    res = g.op.Add(x, y, outputs=outputs, name="add_Scalar")
    if set_shape_type:
        set_shape_type_binary_op(g, outputs[0], x, y)
    return res


def aten_add_Tensor(
    g: GraphBuilder,
    set_shape_type: bool,
    outputs: List[str],
    x: T,
    y: T,
    alpha: Optional[Any] = None,
) -> T:
    assert alpha in (None, 1), f"alpha={alpha}, not implemented"
    x, y = prepare_inputs_homogeneous_operator(g, x, y)
    res = g.op.Add(x, y, outputs=outputs, name="add_Tensor")
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
    layout=None,
    device=None,
    pin_memory=None,
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
        if all(isinstance(arg, IntLike) for arg in args):
            dt = torch.int64
        else:
            dt = torch.get_default_dtype()
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
        i_start = np.array(start, dtype=npdtype)
    if isinstance(end, str):
        i_end = end
    else:
        i_end = np.array(end, dtype=npdtype)
    if isinstance(step, str):
        i_step = step
    else:
        i_step = np.array(step, dtype=npdtype)
    res = g.op.Range(i_start, i_end, i_step, outputs=outputs, name="arange")
    if set_shape_type:
        g.set_type(res, itype)
        if isinstance(end, str) or isinstance(start, str) or isinstance(step, str):
            g.set_rank(res, 1)
        else:
            g.set_shape(res, ((end - start) // step,))
    return res


def aten_arange_start(
    g: GraphBuilder,
    set_shape_type: bool,
    outputs: List[str],
    start: Optional[int] = None,
    end: Optional[int] = None,
    dtype: Optional["torch.dtype"] = None,  # noqa: F821
    layout=None,
    device=None,
    pin_memory=None,
) -> T:
    assert layout is None, f"arange not implemented for layout={layout!r} is not None"
    assert not pin_memory, "arange not implemented for pin_memory=True"
    return aten_arange(g, set_shape_type, outputs, start, end, dtype=dtype)


def aten_arange_start_step(
    g: GraphBuilder,
    set_shape_type: bool,
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
    return aten_arange(g, set_shape_type, outputs, start, end, step, dtype)


def aten_argmax(
    g: GraphBuilder,
    set_shape_type: bool,
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
    if set_shape_type:
        g.set_type(res, TensorProto.INT64)
        if dim is None:
            g.set_shape(res, (1,))
        elif g.has_shape(x):
            sh = g.get_shape(x)
            g.set_shape(res, (sh[dim],))
        else:
            g.set_rank(res, 1)
    return res


def aten_asin(g: GraphBuilder, set_shape_type: bool, outputs: List[str], x: T) -> T:
    res = g.make_node("Asin", [x], outputs)
    if set_shape_type:
        set_shape_type_unary_op(g, outputs[0], x)
    return res


def aten_asinh(g: GraphBuilder, set_shape_type: bool, outputs: List[str], x: T) -> T:
    res = g.make_node("Asinh", [x], outputs)
    if set_shape_type:
        set_shape_type_unary_op(g, outputs[0], x)
    return res


def aten_atan(g: GraphBuilder, set_shape_type: bool, outputs: List[str], x: T) -> T:
    res = g.make_node("Atan", [x], outputs)
    if set_shape_type:
        set_shape_type_unary_op(g, outputs[0], x)
    return res


def aten_atanh(g: GraphBuilder, set_shape_type: bool, outputs: List[str], x: T) -> T:
    res = g.make_node("Atanh", [x], outputs)
    if set_shape_type:
        set_shape_type_unary_op(g, outputs[0], x)
    return res


def aten_bmm(
    g: GraphBuilder, set_shape_type: bool, outputs: List[str], x: T, y: T
) -> T:
    return g.op.MatMul(x, y, outputs=outputs, name="bmm")


def aten_cat(
    g: GraphBuilder,
    set_shape_type: bool,
    outputs: List[str],
    tensors: Tuple[T, ...],
    dim: int = 0,
) -> T:
    res = g.op.Concat(*tensors, axis=dim, outputs=outputs, name="cat")
    if set_shape_type:
        dt0 = g.get_type(tensors[0])
        assert all(map(lambda t: g.get_type(t) == dt0, tensors))
        r0 = g.get_rank(tensors[0])
        assert all(map(lambda t: g.get_rank(t) == r0, tensors))
        g.set_type(outputs[0], dt0)
        g.set_rank(outputs[0], r0)
    return res


def aten_clone(
    g: GraphBuilder,
    set_shape_type: bool,
    outputs: List[str],
    x: T,
    memory_format: Optional[str] = None,
) -> T:
    import torch

    assert (
        memory_format is None or memory_format == torch.contiguous_format
    ), f"Unexpected value for memory_format={memory_format!r}{g.get_debug_msg()}"
    return g.make_node("Identity", [x], outputs, name="clone")


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
    if set_shape_type:
        g.set_type(res, g.get_type(input))
        g.set_rank(res, g.get_rank(input))
    return res


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
        set_shape_type,
        outputs,
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


def aten_cos(g: GraphBuilder, set_shape_type: bool, outputs: List[str], x: T) -> T:
    res = g.make_node("Cos", [x], outputs)
    if set_shape_type:
        set_shape_type_unary_op(g, outputs[0], x)
    return res


def aten_cosh(g: GraphBuilder, set_shape_type: bool, outputs: List[str], x: T) -> T:
    res = g.make_node("Cosh", [x], outputs)
    if set_shape_type:
        set_shape_type_unary_op(g, outputs[0], x)
    return res


def aten_detach(g: GraphBuilder, set_shape_type: bool, outputs: List[str], x: T) -> T:
    return g.make_node("Identity", [x], outputs, name="detach")


def aten_div(
    g: GraphBuilder, set_shape_type: bool, outputs: List[str], x: T, y: T
) -> T:
    x, y = prepare_inputs_homogeneous_operator(g, x, y)
    res = g.op.Div(x, y, outputs=outputs)
    if set_shape_type:
        set_shape_type_binary_op(g, outputs[0], x, y)
    return res


def aten_div_Tensor(
    g: GraphBuilder,
    set_shape_type: bool,
    outputs: List[str],
    x: T,
    y: T,
    alpha: Optional[Any] = None,
) -> T:
    assert alpha in (None, 1), f"alpha={alpha}, not implemented"
    x, y = prepare_inputs_homogeneous_operator(g, x, y)
    res = g.op.Div(x, y, outputs=outputs, name="div_Tensor")
    if set_shape_type:
        set_shape_type_binary_op(g, outputs[0], x, y)
    return res


def aten_dropout(
    g: GraphBuilder,
    set_shape_type: bool,
    outputs: List[str],
    x: T,
    p: T = 0.5,  # float
    training: T = True,  # bool
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
    if isinstance(training, bool):
        training = np.array(training, dtype=np.bool_)
    result, _ = g.op.Dropout(x, p, training, outputs=outputs)
    if set_shape_type:
        set_shape_type_unary_op(g, outputs[0], x)
    return result


def aten_embedding(
    g: GraphBuilder,
    set_shape_type: bool,
    outputs: List[str],
    indices: T,
    weight: T,
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
    res = g.op.Gather(weight, indices, outputs=outputs, name="embedding")
    if set_shape_type:
        g.set_type(res, g.get_type(weight))
        g.set_rank(res, g.get_rank(weight) + g.get_rank(indices) - 1)
    return res


def aten_eq(g: GraphBuilder, set_shape_type: bool, outputs: List[str], x: T, y: T) -> T:
    x, y = prepare_inputs_homogeneous_operator(g, x, y)
    res = g.op.Equal(x, y, outputs=outputs)
    if set_shape_type:
        set_shape_type_binary_op(g, outputs[0], x, y, cmp_op=True)
    return res


def aten_expand(
    g: GraphBuilder,
    set_shape_type: bool,
    outputs: List[str],
    x: T,
    sizes: List[int],
    implicit: bool = False,
) -> T:
    assert not implicit, f"Unexpected value for implicit={implicit!r}"
    res = g.op.Expand(x, np.array(sizes, dtype=np.int64), outputs=outputs)
    if set_shape_type:
        g.set_type(res, g.get_type(x))
        g.set_shape(res, tuple(sizes))
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
    return g.make_node("Flatten", [x], outputs, to=end_dim)


def aten_full(
    g: GraphBuilder,
    set_shape_type: bool,
    outputs: List[str],
    size: T,
    fill_value: float,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
) -> T:
    assert layout is None, f"full not implemented for layout={layout!r} is not None"
    assert not pin_memory, "full not implemented for pin_memory=True"
    assert isinstance(
        fill_value, (float, int)
    ), f"Unexpected type {type(fill_value)} for fill_value."
    new_shape = None

    if isinstance(size, tuple):
        assert all(
            map(lambda x: isinstance(x, int), size)
        ), f"Unexpected values for size={size}"
        tsize = np.array(size, dtype=np.int64)
        new_shape = size
    elif isinstance(size, list):
        assert all(
            map(lambda x: isinstance(x, int), size)
        ), f"Unexpected values for size={size}"
        tsize = np.array(size, dtype=np.int64)
        new_shape = size
    else:
        raise RuntimeError(f"Unexpected type {type(size)} for size.")

    if dtype is None:
        if isinstance(fill_value, int):
            value = np.array(fill_value, dtype=np.int64).reshape((1,))
            itype = TensorProto.INT64
        elif isinstance(fill_value, float):
            value = np.array(fill_value, dtype=np.float32).reshape((1,))
            itype = TensorProto.FLOAT
        else:
            itype = torch_dtype_to_onnx_dtype(type(fill_value))
            ntype = tensor_dtype_to_np_dtype(itype)
            value = np.array(fill_value, dtype=ntype).reshape((1,))
    else:
        itype = torch_dtype_to_onnx_dtype(dtype)
        ntype = tensor_dtype_to_np_dtype(itype)
        value = np.array(fill_value, dtype=ntype).reshape((1,))

    res = g.op.ConstantOfShape(
        tsize, value=from_array(value), outputs=outputs, name="full"
    )
    if set_shape_type:
        g.set_type(res, itype)
        if new_shape:
            g.set_shape(res, new_shape)

    # size = op.Cast(size, to=INT64.dtype)
    # fill_value = op.Cast(fill_value, to=dtype)
    # return op.Expand(fill_value, size)
    return res


def aten_FunctionCtx(
    g: GraphBuilder, set_shape_type: bool, outputs: List[str], *args, **kwargs
):
    if len(args) == 0 and len(kwargs) == 0:
        return
    raise NotImplementedError(f"args={args}, kwargs={kwargs}")


def aten_gt(g: GraphBuilder, set_shape_type: bool, outputs: List[str], x: T, y: T) -> T:
    x, y = prepare_inputs_homogeneous_operator(g, x, y)
    res = g.op.Greater(x, y, outputs=outputs)
    if set_shape_type:
        set_shape_type_binary_op(g, outputs[0], x, y, cmp_op=True)
    return res


def aten_index_Tensor(
    g: GraphBuilder, set_shape_type: bool, outputs: List[str], x: T, indices: List[int]
) -> T:
    assert isinstance(
        indices, (list, tuple)
    ), f"Unexpected type {type(indices)} for indices"
    indices_rank = 0
    indices_last = 0
    if len(indices) == 1 and isinstance(indices[0], str):
        new_indices = g.op.Reshape(
            indices[0], np.array([-1, 1], dtype=np.int64), name="index"
        )
        res = g.op.GatherND(x, new_indices, outputs=outputs, name="index")
        indices_rank = g.get_rank(indices[0])
        indices_last = g.get_shape(indices[0])[-1]
    else:
        raise RuntimeError(
            f"aten_indices implemented yet for indices={indices}{g.get_debug_msg()}"
        )

    if set_shape_type:
        g.set_type(res, g.get_type(x))
        g.get_rank(res, g.get_rank(x) + indices_rank + indices_last - 1)
    return res


def aten_index_select(
    g: GraphBuilder, set_shape_type: bool, outputs: List[str], x: T, dim: int, index: T
) -> T:
    res = g.op.Gather(x, index, axis=dim, outputs=outputs)
    if set_shape_type:
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
    g: GraphBuilder,
    set_shape_type: bool,
    outputs: List[str],
    x: T,
    weight: T,
    bias: T = None,
) -> T:
    weight_transposed = g.op.Transpose(weight, perm=[1, 0], name="linear")
    if bias:
        res = g.op.MatMul(x, weight_transposed)
        res = g.op.Add(res, bias, outputs=outputs)
    else:
        res = g.op.MatMul(x, weight_transposed, outputs=outputs, name="linear")
    if set_shape_type:
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
    set_shape_type: bool,
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
    if set_shape_type:
        set_shape_type_unary_op(g, res, xc, itype=itype)
    return res


def aten__log_softmax_backward_data(
    g: GraphBuilder,
    set_shape_type: bool,
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
    if set_shape_type:
        set_shape_type_unary_op(g, res, grad_outputc, itype=itype)
    return res


def aten_lt(
    g: GraphBuilder, set_shape_type: bool, outputs: List[str], x: T, y: T, name="lt"
) -> T:
    x, y = prepare_inputs_homogeneous_operator(g, x, y)
    res = g.op.Less(x, y, outputs=outputs, name=name)
    if set_shape_type:
        set_shape_type_binary_op(g, outputs[0], x, y, cmp_op=True)
    return res


def aten_lt_Tensor(
    g: GraphBuilder, set_shape_type: bool, outputs: List[str], x: T, y: T
) -> T:
    return aten_lt(g, set_shape_type, outputs, x, y, name="lt_Tensor")


def aten_matmul(
    g: GraphBuilder, set_shape_type: bool, outputs: List[str], x: T, y: T
) -> T:
    res = g.op.MatMul(x, y, outputs=outputs)
    if set_shape_type:
        set_shape_type_binary_op(g, outputs[0], x, y)
    return res


def aten_masked_fill_Scalar(
    g: GraphBuilder, set_shape_type: bool, outputs: List[str], x: T, mask: T, value
) -> T:
    dt = g.get_type(mask)
    if dt != TensorProto.BOOL:
        cmask = g.op.Cast(mask, to=TensorProto.BOOL, name="masked_fill_Scalar")
    else:
        cmask = mask
    dtx = g.get_type(x)
    avalue = np.array([value], dtype=tensor_dtype_to_np_dtype(dtx))
    res = g.op.Where(cmask, avalue, x)
    if set_shape_type:
        g.set_type(res, dtx)
        if g.has_shape(mask):
            g.set_shape(res, g.get_shape(mask))
        else:
            g.set_rank(res, g.get_rank(mask))
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
        g,
        set_shape_type,
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
    name: str,
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
        set_shape_type,
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
    set_shape_type: bool,
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
    if set_shape_type:
        set_shape_type_reduce_op(g, outputs[0], x, keepdim=keepdim)
    return result


def aten_mm(g: GraphBuilder, set_shape_type: bool, outputs: List[str], x: T, y: T) -> T:
    return g.op.MatMul(x, y, outputs=outputs, name="mm")


def aten_mul(
    g: GraphBuilder, set_shape_type: bool, outputs: List[str], x: T, y: T
) -> T:
    x, y = prepare_inputs_homogeneous_operator(g, x, y)
    res = g.op.Mul(x, y, outputs=outputs)
    if set_shape_type:
        set_shape_type_binary_op(g, outputs[0], x, y)
    return res


def aten_mul_Tensor(
    g: GraphBuilder, set_shape_type: bool, outputs: List[str], x: T, y: T
) -> T:
    res = g.op.Mul(x, y, outputs=outputs, name="mul_Tensor")
    if set_shape_type:
        set_shape_type_binary_op(g, outputs[0], x, y)
    return res


def aten_neg(g: GraphBuilder, set_shape_type: bool, outputs: List[str], x: T) -> T:
    res = g.make_node("Neg", [x], outputs)
    if set_shape_type:
        g.set_type(outputs[0], g.get_type(x))
        g.set_shape(outputs[0], g.get_shape(x))
    return res


def aten_ones(
    g: GraphBuilder,
    set_shape_type: bool,
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
    if set_shape_type:
        g.set_type(res, dtype)
        if new_shape:
            g.set_shape(res, new_shape)
    return res


def aten_permute(
    g: GraphBuilder, set_shape_type: bool, outputs: List[str], x: T, dims: Sequence[int]
) -> T:
    if not dims:
        return g.op.Transpose(x, outputs=outputs, name="permute")

    dims = [axis + len(dims) if axis < 0 else axis for axis in dims]
    return g.op.Transpose(x, perm=dims, outputs=outputs, name="permute")


def aten_pow_Tensor_Scalar(
    g: GraphBuilder, set_shape_type: bool, outputs: List[str], x: T, exponent: T
) -> T:
    if isinstance(exponent, (int, float)):
        exponent = np.array([exponent])
    if isinstance(exponent, np.ndarray):
        if g.has_type(x):
            exponent = exponent.astype(tensor_dtype_to_np_dtype(g.get_type(x)))
        else:
            exponent = g.op.CastLike(exponent, x)
    res = g.op.Pow(x, exponent, outputs=outputs)
    if set_shape_type:
        set_shape_type_unary_op(g, outputs[0], x)
    return res


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


def aten_rsub(
    g: GraphBuilder,
    set_shape_type: bool,
    outputs: List[str],
    x: T,
    y: T,
    alpha: float = 1,
) -> T:
    assert alpha == 1, f"Not implemented with alpha={alpha}"
    return aten_sub(g, set_shape_type, outputs, y, x)


def aten_rsub_Scalar(
    g: GraphBuilder,
    set_shape_type: bool,
    outputs: List[str],
    x: T,
    y: T,
    alpha: float = 1,
) -> T:
    assert alpha == 1, f"Not implemented with alpha={alpha}"
    return aten_sub(g, set_shape_type, outputs, y, x)


def aten_sigmoid(g: GraphBuilder, set_shape_type: bool, outputs: List[str], x: T) -> T:
    res = g.op.Sigmoid(x, outputs=outputs)
    if set_shape_type:
        set_shape_type_unary_op(g, outputs[0], x)
    return res


def aten_sigmoid_backward(
    g: GraphBuilder, set_shape_type: bool, outputs: List[str], out_grad: T, y: T
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
    if set_shape_type:
        set_shape_type_unary_op(g, outputs[0], y)
    return res


def aten_silu(
    g: GraphBuilder,
    set_shape_type: bool,
    outputs: List[str],
    x: T,
    inplace: bool = False,
) -> T:
    assert (
        not inplace
    ), f"inplace computation is not allowed with onnx{g.get_debug_msg()}"
    res = g.op.Mul(x, g.op.Sigmoid(x, name="silu"), outputs=outputs, name="silu")
    if set_shape_type:
        set_shape_type_unary_op(g, res, x)
    return res


def aten_sin(g: GraphBuilder, set_shape_type: bool, outputs: List[str], x: T) -> T:
    res = g.make_node("Sin", [x], outputs)
    if set_shape_type:
        set_shape_type_unary_op(g, outputs[0], x)
    return res


def aten_sinh(g: GraphBuilder, set_shape_type: bool, outputs: List[str], x: T) -> T:
    res = g.make_node("Sinh", [x], outputs)
    if set_shape_type:
        set_shape_type_unary_op(g, outputs[0], x)
    return res


def aten_slice_Tensor(
    g: GraphBuilder,
    set_shape_type: bool,
    outputs: List[str],
    x: T,
    dim: int = 0,
    start: int = 0,
    end: Optional[int] = None,
    step: Optional[int] = None,
) -> T:
    assert isinstance(dim, int), f"Not implemented for dim={dim!r}"
    assert isinstance(start, int), f"Not implemented for start={start!r}"
    assert end is None or isinstance(end, int), f"Not implemented for end={end!r}"
    assert step is None or isinstance(step, int), f"Not implemented for step={step!r}"
    if end is None:
        end = start
        start = 0
    if start == 0 and end == 9223372036854775807 and step in {1, None}:
        # nothing to do
        return g.op.Identity(x, outputs=outputs)
    inputs = [
        np.array([start], dtype=np.int64),
        np.array([end], dtype=np.int64),
        np.array([dim], dtype=np.int64),
    ]
    if step is not None and step != 1:
        inputs.append(np.array([step], dtype=np.int64))
    res = g.op.Slice(x, *inputs, outputs=outputs)
    if set_shape_type:
        dtype = g.get_type(inputs[0])
        shape = g.get_shape(inputs[0])
        new_shape = g._apply_slice_to_shape(
            shape, slice(start, end, step), axes=[dim], expand_axes=[]
        )
        g.set_shape(res, new_shape)
        g.set_type(res, dtype)
    return res


def aten_softmax(
    g: GraphBuilder,
    set_shape_type: bool,
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
    if set_shape_type:
        set_shape_type_unary_op(g, res, xc, itype=itype)
    return res


def aten__softmax(
    g: GraphBuilder,
    set_shape_type: bool,
    outputs: List[str],
    x: T,
    dim: int = -1,
    half_to_float: bool = False,
) -> T:
    assert not half_to_float, f"Unexpected value for half_to_float={half_to_float!r}"
    res = g.op.Softmax(x, axis=dim, outputs=outputs, name="_softmax")
    if set_shape_type:
        set_shape_type_unary_op(g, res, x)
    return res


def aten_sub(
    g: GraphBuilder, set_shape_type: bool, outputs: List[str], x: T, y: T
) -> T:
    x, y = prepare_inputs_homogeneous_operator(g, x, y)
    res = g.op.Sub(x, y, outputs=outputs)
    if set_shape_type:
        set_shape_type_binary_op(g, outputs[0], x, y)
    return res


def aten_sum(
    g: GraphBuilder,
    set_shape_type: bool,
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
    if set_shape_type:
        set_shape_type_reduce_op(g, outputs[0], x, keepdim=keepdim)
    return result


def aten_sum_dim_IntList(
    g: GraphBuilder,
    set_shape_type: bool,
    outputs: List[str],
    x: T,
    dim: Optional[Union[int, List[int]]],
    keepdim: bool,
    dtype: Optional["torch.dtype"] = None,  # noqa: F821
) -> T:
    if dtype is None:
        return aten_sum(g, set_shape_type, outputs, x, dim, keepdim)

    res = aten_sum(g, set_shape_type, None, x, dim, keepdim)
    itype = torch_dtype_to_onnx_dtype(dtype)
    result = g.op.Cast(res, to=itype, outputs=outputs, name="sum_dim_IntList")
    return result


def aten__to_copy(
    g: GraphBuilder,
    set_shape_type: bool,
    outputs: List[str],
    x: T,
    dtype: Optional["torch.dtype"] = None,  # noqa: F821
    layout=None,
    device=None,
    pin_memory=None,
    non_blocking=False,
    memory_format=None,
) -> T:
    assert layout is None, f"_to_copy implemented with layout={layout!r}"
    assert device is None, f"_to_copy implemented with device={device!r}"
    assert pin_memory is None, f"_to_copy implemented with pin_memory={pin_memory!r}"
    assert not non_blocking, f"_to_copy implemented with non_blocking={non_blocking!r}"
    assert (
        memory_format is None
    ), f"_to_copy implemented with memory_format={memory_format!r}"
    if dtype is None:
        return g.op.Identity(x, outputs=outputs, name="_to_copy")
    itype = torch_dtype_to_onnx_dtype(dtype)
    return g.op.Cast(x, to=itype, outputs=outputs, name="_to_copy")


def aten_t(g: GraphBuilder, set_shape_type: bool, outputs: List[str], x: T) -> T:
    return g.op.Transpose(x, perm=[1, 0], outputs=outputs, name="t")


def _aten_tensor_int1(
    g: GraphBuilder,
    set_shape_type: bool,
    outputs: List[str],
    input_name: T,
    indices: Tuple[Any, ...],
    axes: List[int],
    expand_axes: List[int],
) -> T:
    assert isinstance(axes, list), f"Unexpected type {type(axes)} for axes"
    assert all(
        map(lambda i: isinstance(i, int), axes)
    ), f"Expected only integer axis but got {axes}"
    assert all(
        map(lambda i: isinstance(i, int), indices)
    ), f"Expected only integer axis but got {indices}"
    assert len(axes) == 1, f"Length mismatch {len(axes)} != 1"

    # axes
    indices_name = g.unique_name(f"{outputs[0]}_indices")
    g.make_initializer(indices_name, np.array(indices, dtype=np.int64))

    res = g.make_node(
        "Gather",
        [input_name, indices_name],
        outputs=outputs,
        axis=axes[0],
        name="getitem_int1",
        set_shape_type=True,
    )

    if expand_axes:
        raise RuntimeError(f"Not implemented when expand_axes={expand_axes}.")
    if set_shape_type:
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
    set_shape_type: bool,
    outputs: List[str],
    x: T,
    indices: Optional[Tuple[Any, ...]] = None,
) -> T:
    if indices is None:
        # x is some data to convert into a Tensor
        if isinstance(x, list) and all(map(lambda e: isinstance(e, (int, float)), x)):
            if all(map(lambda e: isinstance(e, int), x)):
                cst = np.array(x, dtype=np.int64)
            elif all(map(lambda e: isinstance(e, float), x)):
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
        if isinstance(indices[0], list) and all(
            map(lambda i: isinstance(i, int), indices[0])
        ):
            return _aten_tensor_int1(g, set_shape_type, outputs, x, indices, [0], [])
    raise RuntimeError(
        f"Unable to handle getitem with indices={indices}{g.get_debug_msg()}"
    )


def aten_transpose(
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


def aten_transpose_int(
    g: GraphBuilder,
    set_shape_type: bool,
    outputs: List[str],
    input_name: T,
    dim0: int,
    dim1: int,
) -> T:
    return aten_transpose(g, set_shape_type, outputs, input_name, dim0, dim1)


def aten_truediv(
    g: GraphBuilder, set_shape_type: bool, outputs: List[str], x: T, y: T
) -> T:
    x, y = prepare_inputs_homogeneous_operator(g, x, y)
    res = g.op.Div(x, y, outputs=outputs, name="truediv")
    if set_shape_type:
        set_shape_type_binary_op(g, outputs[0], x, y)
    return res


def aten_unsqueeze(
    g: GraphBuilder, set_shape_type: bool, outputs: List[str], x: T, dim: int
) -> T:
    assert isinstance(dim, int), f"Not implemented for dim={dim!r}"
    res = g.op.Unsqueeze(x, np.array([dim], dtype=np.int64), outputs=outputs)
    if set_shape_type:
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
    set_shape_type: bool,
    outputs: List[str],
    x: T,
    size: T,
    node_name: str = "view",
) -> T:
    if isinstance(size, (int, tuple, list)):
        asize = [size] if isinstance(size, int) else list(size)
        asize = np.array(asize, dtype=np.int64)
        assert (
            len(asize.shape) == 1
        ), f"Unexpected shape for view, size={size}{g.get_debug_msg()}"
        return g.op.Reshape(x, asize, outputs=outputs, name=node_name)
    size = g.op.Cast(size, to=TensorProto.INT64, name=node_name)
    return g.op.Reshape(x, size, outputs=outputs, name=node_name)


def aten__unsafe_view(
    g: GraphBuilder, set_shape_type: bool, outputs: List[str], x: T, size: T
) -> T:
    return aten_view(g, set_shape_type, outputs, x, size, node_name="_unsafe_view")
