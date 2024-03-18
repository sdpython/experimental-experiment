from typing import Any, Callable, List, Optional, Sequence, Set, Tuple
import numpy as np
from onnx import NodeProto, TensorProto
from onnx.helper import np_dtype_to_tensor_dtype, tensor_dtype_to_np_dtype
from ..xbuilder.shape_helper import STATIC_SHAPE, is_static_shape, all_int
from ..xbuilder._dtype_helper import dtype_to_tensor_dtype, torch_dtype_to_onnx_dtype


def broadcast_shape(sh1: STATIC_SHAPE, sh2: STATIC_SHAPE) -> STATIC_SHAPE:
    """
    Computes the shape for many broadcasting operators.

    :param sh1: first shape
    :param sh2: second shape
    :return: resulting shape
    """
    if sh1 == sh2:
        return sh1
    if len(sh1) == 0:
        return sh2
    if len(sh2) == 0:
        return sh1
    if sh1 == (1,) and len(sh2) >= 1:
        return sh2
    if sh2 == (1,) and len(sh1) >= 1:
        return sh1
    if len(sh1) < len(sh2):
        sh1 = (1,) * (len(sh2) - len(sh1)) + sh1
    elif len(sh1) > len(sh2):
        sh2 = (1,) * (len(sh1) - len(sh2)) + sh2
    new_shape = []
    for a, b in zip(sh1, sh2):
        if isinstance(a, int):
            if isinstance(b, int):
                d = max(a, b)
            elif a == 1:
                d = b
            else:
                d = None
        else:
            if isinstance(b, int):
                if b == 1:
                    d = a
                else:
                    d = None
            else:
                d = None
        if d is None:
            raise RuntimeError(
                f"Not implemented for sh1={sh1}, sh2={sh2}, a={a}, b={b}, "
                f"type(a)={type(a)}, type(b)={type(b)}"
            )
        new_shape.append(d)
    return tuple(new_shape)


def set_type_shape_reshape(
    g: "GraphBuilder",  # noqa: F821
    name: str,
    input_name: str,
    new_shape: Sequence[int],
):
    g.set_type(name, g.get_type(input_name))
    if isinstance(new_shape, str):
        if g.has_shape(new_shape):
            sh = g.get_shape(new_shape)
            assert len(sh) == 1, f"Unexpected value {sh} for shape={new_shape!r}"
            g.set_rank(name, sh[0])
    elif not is_static_shape(new_shape):
        g.set_rank(name, len(new_shape))
    elif min(new_shape) == -1:
        if g.has_shape(input_name):
            shape = list(g.get_shape(input_name))
            arg_size = np.prod([a for a in new_shape if a >= 0])
            size = np.prod(shape)
            index = new_shape.index(-1)
            if arg_size == 0:
                assert size == 0, f"Unable to reshape {shape} into {new_shape}"
                shape[index] = 1
            else:
                shape[index] = int(size // arg_size)
            g.set_shape(name, tuple(shape))
        else:
            g.set_rank(name, len(new_shape))
    else:
        g.set_shape(name, tuple(new_shape))


def set_type_shape_unary_op(
    g: "GraphBuilder",  # noqa: F821
    name: str,
    input_name: str,
    itype: Optional[int] = None,
):
    """
    Sets the shape and type for an unary operator (abs, exp, ...).
    """
    g.set_type(name, itype or g.get_type(input_name))
    if g.has_shape(input_name):
        g.set_shape(name, g.get_shape(input_name))
    elif g.has_rank(input_name):
        g.set_rank(name, g.get_rank(input_name))


def set_type_shape_binary_op(
    g: "GraphBuilder",  # noqa: F821
    name: str,
    *input_names: List[str],
    begin: int = 0,
    cmp_op: bool = False,
):
    """
    Sets the shape and type for a binary operator (add, mul, ...).
    """
    # type
    dtype = None
    if cmp_op:
        # operator comparing values
        g.set_type(name, TensorProto.BOOL)
    else:
        for input_name in input_names[begin:]:
            if g.has_type(input_name):
                dtype = g.get_type(input_name)
                break
        assert dtype, f"Unable to guess type from {input_names}{g.get_debug_msg()}"
        g.set_type(name, dtype)

    # shape
    shape = None
    for input_name in input_names:
        if g.has_shape(input_name):
            input_shape = g.get_shape(input_name)
            if None in input_shape:
                shape = None
                break
            shape = (
                input_shape if shape is None else broadcast_shape(shape, input_shape)
            )
        else:
            # one shape is missing
            shape = None
            break

    if shape is not None:
        g.set_shape(name, shape)
        return

    # rank otherwise
    rank = None
    for input_name in input_names:
        if g.has_rank(input_name):
            if rank is None:
                rank = g.get_rank(input_name)
            else:
                rank = max(rank, g.get_rank(input_name))
        elif rank is not None:
            # one shape is missing
            rank = None
            break
    if rank is not None:
        g.set_rank(name, rank)


def set_type_shape_matmul(g: "GraphBuilder", name: str, x: str, y: str):  # noqa: F821
    g.set_type(name, g.get_type(x))
    if g.has_shape(x) and g.has_shape(y):
        sh1 = g.get_shape(x)
        sh2 = g.get_shape(y)
        if len(sh1) >= 2 and len(sh2) >= 2 and len(sh1) != len(sh2):
            if len(sh1) < len(sh2):
                sh1 = (1,) * (len(sh2) - len(sh1)) + sh1
            else:
                sh2 = (1,) * (len(sh1) - len(sh2)) + sh2
        assert len(sh1) == len(
            sh2
        ), f"not implemented when shapes are {sh1} and {sh2}{g.get_debug_msg()}"
        new_shape = []
        for a, b in zip(sh1[:-2], sh2[:-2]):
            if all_int((a, b)) or a == b:
                new_shape.append(max(a, b))
            elif a == 1:
                new_shape.append(b)
            elif b == 1:
                new_shape.append(a)
            else:
                # unable to decide, falls back to rank
                g.set_rank(name, max(g.get_rank(x), g.get_rank(y)))
                return

        new_shape.append(sh1[-2])
        new_shape.append(sh2[-1])
        g.set_shape(name, tuple(new_shape))
        return
    if g.has_rank(x) and g.has_rank(y):
        g.set_rank(name, max(g.get_rank(x), g.get_rank(y)))


def set_type_shape_gemm(
    g: "GraphBuilder", name: str, x: str, y: str, transA: int, transB: int  # noqa: F821
):
    if transA == 0 and transB == 0:
        return set_type_shape_matmul(g, name, x, y)
    g.set_type(name, g.get_type(x))
    if g.has_shape(x) and g.has_shape(y):
        sh1 = g.get_shape(x)
        sh2 = g.get_shape(y)
        assert len(sh1) == len(
            sh2
        ), f"not implemented when shapes are {sh1} and {sh2}{g.get_debug_msg()}"
        new_shape = (sh1[-1] if transA else sh1[-2], sh2[-2] if transB else sh2[-1])
        g.set_shape(name, new_shape)
    else:
        g.set_rank(name, max(g.get_rank(x), g.get_rank(y)))


def set_type_shape_reduce_op(
    g: "GraphBuilder",  # noqa: F821
    name: str,
    x: str,
    keepdim: int,
    axes: Optional[Tuple[int]] = None,
):
    assert keepdim in {0, 1}, f"keepdim={keepdim} must be in {{0, 1}}"
    g.set_type(name, g.get_type(x))
    if axes is None:
        g.set_rank(name, int(keepdim))
    elif not g.has_shape(x):
        g.set_rank(name, g.get_rank(x) - int(keepdim) * len(axes))
    else:
        shape = list(g.get_shape(x))
        for d in axes:
            assert d < len(shape), (
                f"shape mismatch for a reduce op shape={shape}, "
                f"axes={axes}{g.get_debug_msg()}"
            )
            shape[d] = 1 if keepdim else None
        shape = tuple(_ for _ in shape if _ is not None)
        g.set_shape(name, shape)


def _get_input_type(
    g: "GraphBuilder", x: Any, python_default: bool  # noqa: F821
) -> int:
    if isinstance(x, int):
        if x is True or x is False:
            return TensorProto.BOOL
        return TensorProto.INT64 if python_default else None
    if isinstance(x, float):
        return TensorProto.FLOAT if python_default else None
    if isinstance(x, str):
        return g.get_type(x)
    if isinstance(x, np.ndarray):
        return np_dtype_to_tensor_dtype(x.dtype)
    # A torch tensor.
    if hasattr(x, "dtype"):
        return torch_dtype_to_onnx_dtype(x.dtype)
    raise RuntimeError(f"Unable to guess type from {type(x)}.")


def _get_compute_type(dtypes: Set[int]) -> int:
    order = [
        TensorProto.DOUBLE,
        TensorProto.FLOAT,
        TensorProto.FLOAT16,
        TensorProto.INT64,
        TensorProto.UINT64,
        TensorProto.INT32,
        TensorProto.UINT32,
        TensorProto.INT16,
        TensorProto.UINT16,
        TensorProto.INT8,
        TensorProto.UINT8,
    ]
    for t in order:
        if t in dtypes:
            return t
    if TensorProto.BOOL in dtypes:
        return TensorProto.INT32
    raise RuntimeError(f"Unable to guess compute type {dtypes}.")


def _cast_inputs(
    g: "GraphBuilder", a: Any, itype: int, name: Optional[str] = None  # noqa: F821
) -> str:
    if isinstance(a, str):
        # a result
        res = g.op.Cast(a, to=itype, name=name)
        g.set_type(res, itype)
        if g.has_shape(a):
            g.set_shape(res, g.get_shape(a))
        else:
            g.set_rank(res, g.get_rank(a))
        return res
    if isinstance(a, (int, float)):
        if a is True or a is False:
            a = np.array(a, dtype=np.bool_)
        else:
            a = np.array(a)
    if isinstance(a, np.ndarray):
        return g.make_initializer("", a.astype(tensor_dtype_to_np_dtype(itype)))
    raise RuntimeError(f"Unexpected type {type(a)}, itype={itype}.")


def prepare_inputs_homogeneous_operator(
    g: "GraphBuilder",  # noqa: F821
    *args: Sequence[str],
    f: Optional[Callable] = None,
    outputs: Optional[List[str]] = None,
    name: Optional[str] = None,
    sts: Optional[Any] = None,
) -> Tuple[str, ...]:
    """
    Cast any inputs to ensure all inputs share the same type.
    """
    dtypes_list = [_get_input_type(g, a, python_default=False) for a in args]
    dtypes_list_not_none = [n for n in dtypes_list if n is not None]
    if not dtypes_list_not_none:
        # the type cannot be guessed from the input as it is only python types,
        # let's include them
        dtypes_list_not_none = [
            _get_input_type(g, a, python_default=True) for a in args
        ]
    dtypes = set(dtypes_list_not_none)
    if len(dtypes) == 1:
        only = list(dtypes)[0]
    else:
        only = _get_compute_type(set(dtypes))
    inputs = []
    for dt, a in zip(dtypes_list, args):
        if dt == only and isinstance(a, str):
            inputs.append(a)
            continue
        inputs.append(_cast_inputs(g, a, only, name=name))
    if f is None:
        return tuple(inputs)
    if tuple(inputs) == tuple(args):
        # No cast.
        res = f(*inputs, outputs=outputs, name=name)
    else:
        assert dtype_to_tensor_dtype, (
            f"Unable to determine the type to Cast back into "
            f"dtypes_list={dtypes_list}, only={only}{g.get_debug_msg()}"
        )
        if sts and sts.get("dtype", None) is not None:
            itype = torch_dtype_to_onnx_dtype(sts["dtype"])
        else:
            assert len(set(dtypes_list_not_none)) == 1, (
                f"Too many choices for the output type, sts={sts} "
                f"dtypes_list={dtypes_list}, "
                f"dtypes_list_not_none={dtypes_list_not_none}{g.get_debug_msg()}"
            )
            itype = dtypes_list_not_none[0]
        tr = f(*inputs, name=name)
        set_type_shape_binary_op(g, tr, *inputs)
        res = g.op.Cast(tr, to=itype, outputs=outputs, name=name)
        if outputs is None:
            set_type_shape_unary_op(g, res, tr, itype=dtypes_list_not_none[0])
    return tuple([res, *inputs])


def _adjust_attributes_of_max_pool(
    expand_size: int,
    kernel_size: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
) -> Tuple[Sequence[int], Sequence[int], Sequence[int], Sequence[int]]:
    if isinstance(dilation, int):
        dilations = [dilation] * expand_size
    else:
        dilations = dilation

    if isinstance(kernel_size, int):
        kernel_shape = [kernel_size] * expand_size
    else:
        kernel_shape = kernel_size

    if isinstance(padding, int):
        pads = [padding] * expand_size * 2
    elif len(padding) == 1:
        pads = padding * expand_size * 2
    elif len(padding) == 2:
        # 2D padding
        pads = padding * 2
    elif len(padding) == 3:
        # 3D padding
        pads = padding * 2
    else:
        # When padding is already done for all dimensions,
        # we don't need to double it
        # eg: (1, 1, 1, 1, 1, 1)
        pads = padding

    if isinstance(stride, int):
        strides = [stride] * expand_size
    elif not stride:
        strides = kernel_shape
    else:
        strides = stride

    return (kernel_shape, strides, pads, dilations)


def _set_shape_type_op_any_reshape(self: "GraphBuilder", node: NodeProto):  # noqa: F821
    k = node.output[0]
    self.set_type(k, self.get_type(node.input[0]))
    shape_set = False
    if self.is_constant(node.input[1]):
        cst = tuple(
            self.get_constant(node.input[1], computed_value=True, as_shape=True)
        )
        if all_int(cst):
            if -1 not in cst:
                self.set_shape(k, cst)
                shape_set = True
            elif all_int(cst) and self.has_shape(node.input[0]):
                sh = self.get_shape(node.input[0])
                new_shape = self._apply_reshape_to_shape(sh, cst)
                if new_shape is not None:
                    self.set_shape(k, new_shape)
                    shape_set = True
    if not shape_set:
        if self.has_shape(node.input[1]):
            rk = self.get_shape(node.input[1])
            self.set_rank(k, rk[0])


def _set_shape_type_op_any_reduce(self: "GraphBuilder", node: NodeProto):  # noqa: F821
    keepdim = self.get_attribute(node, "keepdims", exc=False)
    axes = self.get_attribute(node, "axes", exc=False)
    if axes is None:
        if len(node.input) == 2:
            assert self.is_constant(node.input[1]), (
                f"axes from node {node.op_type}, name={node.name!r} is not a constant, "
                f"the new shape cannot be infered{self.get_debug_msg()}"
            )
            cst = self.get_constant(node.input[1])
            assert isinstance(cst, np.ndarray), (
                f"Unexpected type {type(cst)} for {node.input[1]!r}, "
                f"unable to set type and shape for node {node.op_type} "
                f"with name={node.name!r}{self.get_debug_msg()}"
            )
            iaxes = (int(cst),) if len(cst.shape) == 0 else tuple(int(i) for i in cst)
        else:
            iaxes = None
    else:
        iaxes = tuple(axes.ints)

    set_type_shape_reduce_op(
        self,
        node.output[0],
        node.input[0],
        keepdim=None if keepdim is None else keepdim.i,
        axes=iaxes,
    )


def _set_shape_type_op_any_matmul(self: "GraphBuilder", node: NodeProto):  # noqa: F821
    set_type_shape_matmul(self, node.output[0], *node.input)


def _set_shape_type_op_any_gemm(self: "GraphBuilder", node: NodeProto):  # noqa: F821
    transA = self.get_attribute(node, "transA", exc=False)
    transB = self.get_attribute(node, "transB", exc=False)
    set_type_shape_gemm(
        self,
        node.output[0],
        *node.input[:2],
        transA=0 if transA is None else transA.i,
        transB=0 if transB is None else transB.i,
    )


def _set_shape_type_op_any_cast(self: "GraphBuilder", node: NodeProto):  # noqa: F821
    set_type_shape_unary_op(
        self,
        node.output[0],
        node.input[0],
        itype=self.get_attribute(node, "to").i,
    )


def _set_shape_type_op_any_sign(self: "GraphBuilder", node: NodeProto):  # noqa: F821
    set_type_shape_unary_op(
        self, node.output[0], node.input[0], itype=TensorProto.INT64
    )


def _set_shape_type_op_any_castlike(
    self: "GraphBuilder", node: NodeProto  # noqa: F821
):
    set_type_shape_unary_op(
        self, node.output[0], node.input[0], itype=self.get_type(node.input[1])
    )


def _set_shape_type_op_any_maxpool(self: "GraphBuilder", node: NodeProto):  # noqa: F821
    self.set_type(node.output[0], self.get_type(node.input[0]))
    if len(node.output) > 1:
        self.set_type(node.output[1], TensorProto.INT64)


def _set_shape_type_op_any_gather_elements(
    self: "GraphBuilder", node: NodeProto  # noqa: F821
):
    self.set_type(node.output[0], self.get_type(node.input[0]))
    if self.has_shape(node.input[0]) and self.has_shape(node.input[1]):
        shape = self.get_shape(node.input[0])
        att_axis = self.get_attribute(node, "axis", exc=False)
        axis = 0 if att_axis is None else att_axis.i
        i_shape = self.get_shape(node.input[1])
        new_shape = list(shape)
        new_shape[axis] = i_shape[axis]
        self.set_shape(node.output[0], tuple(new_shape))
    else:
        self.set_rank(node.output[0], self.get_rank(node.input[0]))


def _set_shape_type_op_any_concat(self: "GraphBuilder", node: NodeProto):  # noqa: F821
    self.set_type(node.output[0], self.get_type(node.input[0]))
    if all(map(lambda s: self.has_shape(s), node.input)):
        axis = self.get_attribute(node, "axis").i
        shapes = list(self.get_shape(i) for i in node.input)
        new_shape = list(shapes[0])
        dims = [sh[axis] for sh in shapes]
        if all_int(dims):
            new_shape[axis] = sum(dims)
        else:
            new_shape[axis] = "+".join(map(str, dims))
        self.set_shape(node.output[0], tuple(new_shape))
    else:
        ranks = list(self.get_rank(i) for i in node.input)
        assert (
            len(set(ranks)) == 1
        ), f"Unexpected ranks={ranks} for node {node.op_type!r}{self.get_debug_msg()}"
        self.set_rank(node.output[0], ranks[0])


_set_shape_type_op_any_known = {
    "Cast": _set_shape_type_op_any_cast,
    "Concat": _set_shape_type_op_any_concat,
    "GatherElements": _set_shape_type_op_any_gather_elements,
    "Gemm": _set_shape_type_op_any_gemm,
    "MatMul": _set_shape_type_op_any_matmul,
    "MaxPool": _set_shape_type_op_any_maxpool,
    "Reshape": _set_shape_type_op_any_reshape,
    "Sign": _set_shape_type_op_any_sign,
}


def set_shape_type_op_any(self: "GraphBuilder", node: NodeProto):  # noqa: F821
    """
    Sets the shape and type if it can.
    """
    if node.op_type.startswith("Reduce"):
        _set_shape_type_op_any_reduce(self, node)
    elif node.op_type in _set_shape_type_op_any_known:
        _set_shape_type_op_any_known[node.op_type](self, node)
    elif node.op_type in self._op_type_element_wise_cmp_types:
        set_type_shape_binary_op(self, node.output[0], *node.input, cmp_op=True)
    elif node.op_type in self._op_type_element_wise_types:
        set_type_shape_binary_op(self, node.output[0], *node.input)
    elif node.op_type in self._op_type_unary_like:
        set_type_shape_unary_op(self, node.output[0], node.input[0])
