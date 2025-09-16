from typing import Any, Callable, List, Optional, Sequence, Set, Tuple
import numpy as np
from onnx import NodeProto, TensorProto
from ..helpers import (
    tensor_dtype_to_np_dtype,
    np_dtype_to_tensor_dtype,
    dtype_to_tensor_dtype,
    torch_dtype_to_onnx_dtype,
)
from ..xbuilder._shape_helper import (
    DYNAMIC_SHAPE,
    is_static_shape,
    all_int,
    all_int_or_str,
)


def broadcast_shape(
    sh1: DYNAMIC_SHAPE,
    sh2: DYNAMIC_SHAPE,
    graph_builder: Optional["GraphBuilder"] = None,  # noqa: F821
) -> DYNAMIC_SHAPE:
    """
    Computes the shape for many broadcasting operators.
    This function should be used while converting the graph into ONNX
    because it assumes the broadcast is possible and adds the necessary constraints
    on the dynamic in the GraphBuilder shapes to make it work.

    :param sh1: first shape
    :param sh2: second shape
    :param graph_builder: if not None, the function register
        any constraint which might appear while applying the broadcast
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
                # We have two indications, let's take the most strict one.
                d = a
                if graph_builder:
                    graph_builder.register_constraint_dimension(b, a)
        elif isinstance(b, int):
            # a is str
            if b == 1:
                d = a
            elif b != 1:
                # a is not int, it is str
                d = b
                if graph_builder:
                    graph_builder.register_constraint_dimension(a, b)
        else:
            # both str
            if a == b:
                d = a
            else:
                d = a
                if graph_builder:
                    graph_builder.register_constraint_dimension(a, b)
        if d is None:
            raise RuntimeError(
                f"Not implemented for sh1={sh1}, sh2={sh2}, a={a}, b={b}, "
                f"type(a)={type(a)}, type(b)={type(b)}, a={a}, b={b}"
            )
        new_shape.append(d)
    return tuple(new_shape)


def set_type_shape_reshape(
    g: "GraphBuilder",  # noqa: F821
    name: str,
    input_name: str,
    new_shape: Sequence[int],
):
    "Sets the output shape for node type Reshape"
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
            new_shape = list(new_shape)
            if arg_size == 0:
                assert size == 0, f"Unable to reshape {shape} into {new_shape}"
                new_shape[index] = 1
            else:
                new_shape[index] = int(size // arg_size)
            g.set_shape(name, tuple(new_shape))
        else:
            g.set_rank(name, len(new_shape))
    else:
        g.set_shape(name, tuple(new_shape))


def set_type_shape_unary_op(
    g: "GraphBuilder",  # noqa: F821
    name: str,
    input_name: str,
    itype: Optional[int] = None,
) -> bool:
    """Sets the shape and type for an unary operator (abs, exp, ...)."""
    if not itype and not g.has_type(input_name):
        return False
    g.set_type(name, itype or g.get_type(input_name))
    if g.has_shape(input_name):
        g.set_shape(name, g.get_shape(input_name), allow_zero=True)
        return True
    if g.has_rank(input_name):
        g.set_rank(name, g.get_rank(input_name))
        return True
    return False


def set_type_shape_unary_op_abs(
    g: "GraphBuilder",  # noqa: F821
    name: str,
    input_name: str,
    itype: Optional[int] = None,
) -> bool:
    """Sets the shape and type for an unary operator (abs, exp, ...)."""
    if not itype and not g.has_type(input_name):
        return False
    if not itype:
        itype = g.get_type(input_name)
    if itype in (TensorProto.COMPLEX64, TensorProto.COMPLEX128):
        if itype == TensorProto.COMPLEX64:
            rtype = TensorProto.FLOAT
        elif itype == TensorProto.COMPLEX128:
            rtype = TensorProto.DOUBLE
        else:
            raise AssertionError(
                f"Unexpected type {itype} for {input_name!r}{g.get_debug_msg()}"
            )

        g.set_type(name, rtype)
        if g.has_shape(input_name):
            shape = g.get_shape(input_name)
            g.set_shape(name, shape)
            return True
        if g.has_rank(input_name):
            g.set_rank(name, g.get_rank(input_name))
            return True
        return False

    g.set_type(name, itype)
    if g.has_shape(input_name):
        g.set_shape(name, g.get_shape(input_name))
        return True
    if g.has_rank(input_name):
        g.set_rank(name, g.get_rank(input_name))
        return True
    return False


def set_type_shape_binary_op(
    g: "GraphBuilder",  # noqa: F821
    name: str,
    *input_names: List[str],
    begin: int = 0,
    cmp_op: bool = False,
    itype: Optional[int] = None,
) -> bool:
    """Sets the shape and type for a binary operator (add, mul, ...)."""
    # type
    dtype = None
    if itype:
        g.set_type(name, itype)
    elif cmp_op:
        # operator comparing values
        g.set_type(name, TensorProto.BOOL)
    else:
        for input_name in input_names[begin:]:
            if g.has_type(input_name):
                dtype = g.get_type(input_name)
                break
        if not dtype and g.as_function:
            return False
        assert (
            dtype
        ), f"Unable to guess type for {name!r} from {input_names}{g.get_debug_msg()}"
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
                input_shape
                if shape is None
                else broadcast_shape(shape, input_shape, graph_builder=g)
            )
        else:
            # one shape is missing
            shape = None
            break

    if shape is not None:
        g.set_shape(name, shape)
        return True

    # rank otherwise
    rank = None
    for input_name in input_names:
        if g.has_rank(input_name):
            rank = (
                g.get_rank(input_name) if rank is None else max(rank, g.get_rank(input_name))
            )
            continue
        if rank is not None:
            rank = None
        # one shape is missing
        break

    if rank is not None:
        g.set_rank(name, rank)
        return True
    return False


def set_type_shape_matmul(g: "GraphBuilder", name: str, x: str, y: str) -> bool:  # noqa: F821
    "Sets the output shape for node type MatMul."
    if not g.has_type(x):
        return False
    g.set_type(name, g.get_type(x))
    if g.has_shape(x) and g.has_shape(y):
        sh1 = g.get_shape(x)
        sh2 = g.get_shape(y)
        if len(sh1) >= 2 and len(sh2) >= 2 and len(sh1) != len(sh2):
            if len(sh1) < len(sh2):
                sh1 = (1,) * (len(sh2) - len(sh1)) + sh1
            else:
                sh2 = (1,) * (len(sh1) - len(sh2)) + sh2
        assert len(sh1) == len(sh2), (
            f"not implemented when shapes are {sh1} ({x!r}) and {sh2} ({y!r})"
            f"{g.get_debug_msg()}"
        )
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
                return True

        new_shape.append(sh1[-2])
        new_shape.append(sh2[-1])
        g.set_shape(name, tuple(new_shape))
        return True
    if g.has_rank(x) and g.has_rank(y):
        g.set_rank(name, max(g.get_rank(x), g.get_rank(y)))
        return True
    return False


def set_type_shape_gemm(
    g: "GraphBuilder",  # noqa: F821
    name: str,
    x: str,
    y: str,
    transA: int,
    transB: int,
):
    "Sets the output shape for node type Gemm."
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
    elif g.has_rank(x) and g.has_rank(y):
        g.set_rank(name, max(g.get_rank(x), g.get_rank(y)))


def set_type_shape_reduce_op(
    g: "GraphBuilder",  # noqa: F821
    name: str,
    x: str,
    keepdim: int,
    axes: Optional[Tuple[int]] = None,
):
    "Sets the output shape for any Reduce type."
    assert keepdim in {None, 0, 1}, f"keepdim={keepdim!r} must be in {{0, 1}}"
    if keepdim is None:
        keepdim = 1
    g.set_type(name, g.get_type(x))
    if axes is None:
        g.set_shape(name, ((1,) * g.get_rank(x)) if keepdim else tuple())
    elif not g.has_shape(x):
        if g.has_rank(x):
            g.set_rank(name, g.get_rank(x) - (1 - int(keepdim)) * len(axes))
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
    g: "GraphBuilder",  # noqa: F821
    x: Any,
    python_default: bool,
) -> int:
    if isinstance(x, int):
        if x is True or x is False:
            return TensorProto.BOOL
        return TensorProto.INT64 if python_default else None
    if isinstance(x, float):
        return TensorProto.FLOAT if python_default else None
    if isinstance(x, complex):
        return TensorProto.COMPLEX64 if python_default else None
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
        TensorProto.COMPLEX128,
        TensorProto.DOUBLE,
        TensorProto.COMPLEX64,
        TensorProto.FLOAT,
        TensorProto.FLOAT16,
        TensorProto.BFLOAT16,
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
    g: "GraphBuilder",  # noqa: F821
    a: Any,
    itype: int,
    name: Optional[str] = None,
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
    if isinstance(a, (int, float, complex)):
        if a is True or a is False:
            a = np.array(a, dtype=np.bool_)
        else:
            a = np.array(a)
    if isinstance(a, np.ndarray):
        new_dtype = tensor_dtype_to_np_dtype(itype)
        na = a.astype(new_dtype)
        return g.make_initializer(
            "",
            na,
            source=(
                f"shape_type_compute._cast_inputs.1({name})"
                if name
                else "shape_type_compute._cast_inputs.0"
            ),
        )
    raise RuntimeError(f"Unexpected type {type(a)}, itype={itype}.")


def prepare_inputs_homogeneous_operator(
    g: "GraphBuilder",  # noqa: F821
    *args: Sequence[str],
    f: Optional[Callable] = None,
    outputs: Optional[List[str]] = None,
    name: Optional[str] = None,
    sts: Optional[Any] = None,
    check_shape: bool = True,
    op_type: Optional[str] = None,
    use_left: bool = False,
    force_type: Optional[int] = None,
) -> Tuple[str, ...]:
    """
    Casts any inputs to ensure all inputs share the same type.

    op_type can be specified to bypass some cases with ambiguities such as
    a float multiplied with an integer.

    :param g: GraphBuilder
    :param args: operator arguments
    :param f: function calling the operator
    :param outputs: output names
    :param sts: known shapes and types
    :param check_shape: extra verification for shapes
    :param op_type: operator type (onnx name)
    :param use_left: if the operator is in one inplaced modification
        then the type of the left side
    :param force_type: if not None, choose this type and cast the inputs
    :return: new inputs
    """
    dtypes_list = [_get_input_type(g, a, python_default=False) for a in args]
    dtypes_list_not_none = [n for n in dtypes_list if n not in (0, None)]
    if not dtypes_list_not_none:
        # the type cannot be guessed from the input as it is only python types,
        # let's include them
        dtypes_list_not_none = [_get_input_type(g, a, python_default=True) for a in args]
    dtypes = set(dtypes_list_not_none)
    if force_type is not None:
        only = force_type
    elif len(dtypes) == 1:
        only = list(dtypes)[0]  # noqa: RUF015
    elif use_left and dtypes_list[0]:
        only = dtypes_list[0]
    else:
        only = _get_compute_type(set(dtypes))
    assert only > 0, (
        f"Unexpected element type={only}, op_type={op_type!r}, "
        f"dtypes_list={dtypes_list}, dtypes_list_not_none={dtypes_list_not_none}, "
        f"name={name!r}, args={args}{g.get_debug_msg()}"
    )
    inputs = []
    for dt, a in zip(dtypes_list, args):
        if dt == only and isinstance(a, str):
            inputs.append(a)
            continue
        inputs.append(_cast_inputs(g, a, only, name=name))

    if check_shape:
        # Checks that one input is not a scalar without no dimension.
        shapes = []
        new_inputs = []
        for i in inputs:
            if g.has_shape(i):
                shape = g.get_shape(i)
                shapes.append(shape)
                if len(shape) == 0:
                    new_inputs.append(
                        g.op.Reshape(i, np.array([1], dtype=np.int64), name=name)
                    )
                    continue
            elif g.has_rank(i) and g.get_rank(i) == 0:
                new_inputs.append(g.op.Reshape(i, np.array([1], dtype=np.int64), name=name))
                continue
            new_inputs.append(i)
        if set(shapes) != {tuple()}:
            inputs = new_inputs

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
            itype = None
            set_itypes = set(dtypes_list_not_none)
            if op_type in {"Mul", "Div", "Add", "Sub"} and len(set_itypes) > 1:
                if set_itypes == {TensorProto.FLOAT, TensorProto.INT64}:
                    itype = TensorProto.FLOAT
                elif set_itypes == {TensorProto.FLOAT, TensorProto.DOUBLE} and len(args) == 2:
                    # This is usually the multiplication by a constant.
                    # This is not efficient but that what's the exported program expects.
                    itype = TensorProto.DOUBLE
            assert itype or len(set_itypes) == 1, (
                f"Too many choices for the output type, sts={sts}, itype={itype!r}, "
                f"dtypes_list={dtypes_list}, set_itypes={set_itypes}, name={name!r}, "
                f"dtypes_list_not_none={dtypes_list_not_none}, op_type={op_type!r}, "
                f"\nargs={args}\noutputs={outputs}{g.get_debug_msg()}"
            )
            if not itype:
                itype = dtypes_list_not_none[0]
        tr = f(*inputs, name=name)
        r = set_type_shape_binary_op(g, tr, *inputs)
        assert r or not g._debug_shape_missing, f"Unable to compute shape for node {op_type}."
        res = g.op.Cast(tr, to=itype, outputs=outputs, name=name)
        if outputs is None:
            r = set_type_shape_unary_op(g, res, tr, itype=dtypes_list_not_none[0])
            assert (
                r or not g._debug_shape_missing
            ), f"Unable to compute shape for node {op_type}."
    return (res, *inputs)


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


########################################
# Implementation for the main algorithm.
########################################


def _set_shape_type_op_any_batch_normalization(
    self: "GraphBuilder", node: NodeProto  # noqa: F821
):
    "Sets the output shape for node type BatchNormalization."
    set_type_shape_unary_op(self, node.output[0], node.input[0])
    if len(node.output) > 1:
        set_type_shape_unary_op(self, node.output[1], node.input[1])
    if len(node.output) > 2:
        set_type_shape_unary_op(self, node.output[2], node.input[2])


def _set_shape_type_op_any_layer_normalization(
    self: "GraphBuilder", node: NodeProto  # noqa: F821
):
    "Sets the output shape for node type LayerNormalization."
    set_type_shape_unary_op(self, node.output[0], node.input[0])
    if len(node.output) > 1:
        set_type_shape_unary_op(self, node.output[1], node.input[1])
    if len(node.output) > 2:
        set_type_shape_unary_op(self, node.output[2], node.input[2])


def _set_shape_type_op_any_cast(self: "GraphBuilder", node: NodeProto):  # noqa: F821
    "Sets the output shape for node type Cast."
    set_type_shape_unary_op(
        self,
        node.output[0],
        node.input[0],
        itype=self.get_attribute(node, "to").i,
    )


def _set_shape_type_op_any_rotary_embedding(
    self: "GraphBuilder", node: NodeProto  # noqa: F821
):
    "Sets the output shape for node type Cast."
    set_type_shape_unary_op(self, node.output[0], node.input[0])


def _set_shape_type_op_any_castlike(self: "GraphBuilder", node: NodeProto):  # noqa: F821
    "Sets the output shape for node type CastLike."
    set_type_shape_unary_op(
        self, node.output[0], node.input[0], itype=self.get_type(node.input[1])
    )


def _set_shape_type_op_any_concat(self: "GraphBuilder", node: NodeProto):  # noqa: F821
    "Sets the output shape for node type Concat."
    if self.has_type(node.input[0]):
        self.set_type(node.output[0], self.get_type(node.input[0]))
    if all(self.has_shape(s) for s in node.input):
        axis = self.get_attribute(node, "axis").i
        shapes = [self.get_shape(i) for i in node.input]
        new_shape = list(shapes[0])
        assert shapes and axis < min(len(sh) for sh in shapes), (
            f"axis={axis}, higher than a shape in {shapes}, "
            f"node={self.pretty_node(node)}{self.get_debug_msg()}"
        )
        assert all(
            axis < len(sh) for sh in shapes
        ), f"Unexpected shape in {shapes}, axis={axis}"
        dims = [sh[axis] for sh in shapes]
        if all_int(dims):
            new_shape[axis] = sum(dims)
        else:
            new_shape[axis] = "+".join(map(str, dims))
        self.set_shape(node.output[0], tuple(new_shape))
    elif all(map(self.has_rank, node.input)):
        ranks = [self.get_rank(i) for i in node.input]
        assert (
            len(set(ranks)) == 1
        ), f"Unexpected ranks={ranks} for node {node.op_type!r}{self.get_debug_msg()}"
        self.set_rank(node.output[0], ranks[0])
    else:
        assert not self._debug_shape_missing, (
            f"Unable to compute shape for node: "
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
        )


def _set_shape_type_op_any_conv_max_pool(self: "GraphBuilder", node: NodeProto):  # noqa: F821
    """
    Sets the output shape for node types Conv, MaxPool.

    This function defines the following functions::

        conf_f1(d,s,stride) = s - (stride if d % stride == 0 else d % stride) // 2
        conf_f2(d,s,stride) = (
            s - (stride if d % stride == 0 else d % stride)) // 2 + stride % 2
        )
        conv_f3(d,s,stride,ceil_mode,p) = ... (see the code)
    """
    if not self.has_type(node.input[0]):
        assert not self._debug_shape_missing, (
            f"Unable to compute shape for node: "
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
        )
        return
    self.set_type(node.output[0], self.get_type(node.input[0]))
    if len(node.output) > 1:
        self.set_type(node.output[1], TensorProto.INT64)

    if not self.has_shape(node.input[0]) or (
        len(node.input) > 1 and not self.has_shape(node.input[1])
    ):
        assert not self._debug_shape_missing, (
            f"Unable to compute shape for node: "
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
        )
        if self.has_rank(node.input[0]):
            self.set_rank(node.output[0], self.get_rank(node.input[0]))
        return

    input_shape = self.get_shape(node.input[0])
    assert len(input_shape) >= 2, (
        f"Input tensor {node.input[0]!r} must have at least 2 dimensions for node "
        f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
    )

    n_input_dims = len(input_shape) - 2

    dilations = self.get_attribute_with_default(node, "dilations", [1] * n_input_dims)
    assert len(dilations) == n_input_dims, (
        f"Mismatch with dilations={dilations}, "
        f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
    )

    strides = self.get_attribute_with_default(node, "strides", [1] * n_input_dims)
    assert len(strides) == n_input_dims, (
        f"Mismatch with strides={strides}, "
        f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
    )

    # Gestion de kernel_shape
    kernel_shape = self.get_attribute_with_default(node, "kernel_shape", None)
    if kernel_shape:
        assert len(kernel_shape) == n_input_dims, (
            f"Mismatch with strides={kernel_shape}, "
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
        )
    if not kernel_shape:
        shape_w = self.get_shape(node.input[1])
        kernel_shape = shape_w[2:]
        assert all_int(kernel_shape), (
            f"kernel_shape is not provided and its shape is unknown "
            f"for sure kernel_shape={kernel_shape}, "
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
        )

    effective_kernel_shape = [(k - 1) * d + 1 for k, d in zip(kernel_shape, dilations)]

    pads = self.get_attribute_with_default(node, "pads", [0] * (n_input_dims * 2))
    assert len(pads) == n_input_dims * 2, (
        f"Mismatch with pads={pads}, "
        f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
    )

    auto_pad_attr = self.get_attribute_with_default(node, "auto_pad", "NOTSET")
    if auto_pad_attr and auto_pad_attr != "VALID":
        for i in range(n_input_dims):
            stride = strides[i]
            if stride > 1:
                input_dim = input_shape[2 + i]
                if isinstance(input_dim, str):
                    if stride == 1:
                        residual = 0
                    else:
                        residual = None
                else:
                    residual = input_dim % stride

                if residual is not None:
                    total_pad = (
                        (effective_kernel_shape[i] - stride)
                        if residual == 0
                        else (effective_kernel_shape[i] - residual)
                    )
                    total_pad = max(total_pad, 0)
                    half_pad_small = total_pad // 2
                    half_pad_big = total_pad - half_pad_small
                    if auto_pad_attr == "SAME_UPPER":
                        pads[i] = half_pad_small
                        pads[i + n_input_dims] = half_pad_big
                    elif auto_pad_attr == "SAME_LOWER":
                        pads[i] = half_pad_big
                        pads[i + n_input_dims] = half_pad_small
                else:
                    # conf_f1=(d,s,stride) = (
                    #   s - (stride if d % stride == 0 else d % stride)) // 2
                    # )
                    pads[i] = f"conf_f1({input_dim};{effective_kernel_shape[i]};{stride})"
                    # conf_f2=(d,s,stride) = (
                    #   s - (stride if d % stride == 0 else d % stride)) // 2 + stride % 2
                    # )
                    pads[i + n_input_dims] = (
                        f"conf_f2({input_dim};{effective_kernel_shape[i]};{stride})"
                    )

    require_kernel_shape = node.op_type in {"MaxPool"}
    output_shape = []
    output_shape.append(input_shape[0])
    if require_kernel_shape:
        output_shape.append(input_shape[1])
    else:
        w_shape = self.get_shape(node.input[1])
        output_shape.append(w_shape[0])

    for i in range(len(kernel_shape)):
        ceil_mode = self.get_attribute_with_default(node, "ceil_mode", 0)
        input_size = input_shape[2 + i]
        if isinstance(pads[i], int):
            if isinstance(input_size, int):
                effective_input_size = input_size + pads[i] + pads[i + len(kernel_shape)]
                output_size = (
                    (
                        effective_input_size
                        - effective_kernel_shape[i]
                        + (strides[i] - 1 if ceil_mode else 0)
                    )
                    // strides[i]
                ) + 1
                if ceil_mode and (output_size - 1) * strides[i] >= input_size + pads[i]:
                    output_size -= 1
                output_shape.append(output_size)
                continue

        # conv_f3(d,s,stride,ceil_mode,p) = (
        #       d + (stride if d % stride == 0 else d % stride) +
        #       (stride - 1) * (ceil_mode == 1)
        #   ) // stride + 1 + ...
        output_size = (
            f"conf_f3({input_size};{effective_kernel_shape[i]};{strides[i]};{ceil_mode})"
        )
        output_shape.append(output_size)

    self.set_shape(node.output[0], tuple(output_shape))

    # Gestion de la deuxième sortie pour MaxPool
    if node.op_type == "MaxPool" and len(node.output) > 1:
        second_output_shape = []
        second_output_shape.extend(output_shape)
        self.set_shape(node.output[1], tuple(second_output_shape))


def _set_shape_type_op_any_gather(self: "GraphBuilder", node: NodeProto):  # noqa: F821
    "Sets the output shape for node type Gather."
    if self.has_type(node.input[0]):
        self.set_type(node.output[0], self.get_type(node.input[0]))
    if self.has_shape(node.input[0]) and self.has_shape(node.input[1]):
        sh1 = self.get_shape(node.input[0])
        sh2 = self.get_shape(node.input[1])
        att = self.get_attribute(node, "axis", exc=False)
        axis = 0 if att is None else att.i
        if len(sh2) == 0:
            new_shape = tuple(s for i, s in enumerate(sh1) if i != axis)
            self.set_shape(node.output[0], new_shape)
        elif len(sh1) == len(sh2) == 2 and axis == 0:
            new_shape = (*sh2, sh1[-1])
            self.set_shape(node.output[0], new_shape)
        else:
            self.set_rank(node.output[0], len(sh1) + len(sh2) - 1)
    elif self.has_rank(node.input[0]) and self.has_rank(node.input[1]):
        rk1 = self.get_rank(node.input[0])
        rk2 = self.get_rank(node.input[1])
        self.set_rank(node.output[0], rk1 + rk2 - 1)
    else:
        assert not self._debug_shape_missing, (
            f"Unable to compute shape for node: "
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
        )


def _set_shape_type_op_any_gather_elements(
    self: "GraphBuilder", node: NodeProto  # noqa: F821
):
    "Sets the output shape for node type GatherElements."
    if self.has_type(node.input[0]):
        self.set_type(node.output[0], self.get_type(node.input[0]))
    if self.has_shape(node.input[0]) and self.has_shape(node.input[1]):
        shape = self.get_shape(node.input[0])
        att_axis = self.get_attribute(node, "axis", exc=False)
        axis = 0 if att_axis is None else att_axis.i
        i_shape = self.get_shape(node.input[1])
        new_shape = list(shape)
        new_shape[axis] = i_shape[axis]
        self.set_shape(node.output[0], tuple(new_shape))
    elif self.has_rank(node.input[0]):
        self.set_rank(node.output[0], self.get_rank(node.input[0]))
    else:
        assert not self._debug_shape_missing, (
            f"Unable to compute shape for node: "
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
        )


def _set_shape_type_op_any_gemm(self: "GraphBuilder", node: NodeProto):  # noqa: F821
    "Sets the output shape for node type Gemm."
    transA = self.get_attribute(node, "transA", exc=False)
    transB = self.get_attribute(node, "transB", exc=False)
    assert len(node.input) >= 2, (
        f"Unexpected number of input {node.input} for node "
        f"{node.op_type} name {node.name!r}"
    )
    set_type_shape_gemm(
        self,
        node.output[0],
        *node.input[:2],
        transA=0 if transA is None else transA.i,
        transB=0 if transB is None else transB.i,
    )


def _set_shape_type_op_any_matmul(self: "GraphBuilder", node: NodeProto):  # noqa: F821
    "Sets the output shape for node type MatMul."
    r = set_type_shape_matmul(self, node.output[0], *node.input)
    assert r or not self._debug_shape_missing, (
        f"Unable to compute shape for node: "
        f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
    )


def _set_shape_type_op_any_non_zero(self: "GraphBuilder", node: NodeProto):  # noqa: F821
    "Sets the output shape for node type NonZro."
    self.set_type(node.output[0], TensorProto.INT64)
    if self.has_rank(node.input[0]):
        self.set_shape(
            node.output[0],
            (self.get_rank(node.input[0]), self.unique_dimension_name("NEWDIM_nonzero")),
        )
        return
    assert not self._debug_shape_missing, (
        f"Unable to compute shape for node: "
        f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
    )


def _set_shape_type_op_any_pad(self: "GraphBuilder", node: NodeProto):  # noqa: F821
    "Sets the output shape for node type Pad."
    if self.has_type(node.input[0]):
        self.set_type(node.output[0], self.get_type(node.input[0]))
    else:
        assert not self._debug_shape_missing, (
            f"Unable to compute shape for node: "
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
        )

    if self.has_shape(node.input[0]) and self.is_constant(node.input[1]):
        pads = self.compute_constant(node.input[1])[0]
        assert pads is not None or not self._debug_shape_missing, (
            f"Unable to evaluate pad={node.input[1]!r}: "
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
        )
        if pads is None:
            return
        pads = pads.tolist()
        if len(node.input) > 3 and node.input[3]:
            axes = self.compute_constant(node.input[1])[0]
            assert axes is not None or not self._debug_shape_missing, (
                f"Unable to evaluate axes={node.input[1]!r}: "
                f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
            )
            axes = axes.tolist()
        else:
            axes = list(range(len(pads) // 2))

        shape = self.get_shape(node.input[0])
        new_shape = list(shape)
        for i in range(len(axes)):
            a = axes[i]
            d = shape[a]
            p1, p2 = pads[i], pads[i + len(axes)]
            new_shape[a] = (d + p1 + p2) if isinstance(d, int) else f"{d}+{p1+p2}"
        self.set_shape(node.output[0], tuple(new_shape))
        return
    if self.has_rank(node.input[0]):
        self.set_rank(node.input[0], self.get_rank(node.input[0]))
        return
    assert not self._debug_shape_missing, (
        f"Unable to compute shape for node: "
        f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
    )


def _set_shape_type_op_any_range(self: "GraphBuilder", node: NodeProto):  # noqa: F821
    "Sets the output shape for for node type Range."
    types = [self.get_type(i) for i in node.input if self.has_type(i)]
    assert types and len(set(types)) == 1, (
        f"Mixed type for node {self.pretty_node(node)}, types={types}, "
        f"unable to set shape and types."
    )
    self.set_type(node.output[0], types[0])
    self.set_rank(node.output[0], 1)


def _set_shape_type_op_any_reduce(self: "GraphBuilder", node: NodeProto):  # noqa: F821
    "Sets the output shape for Reduce node type."
    keepdim = self.get_attribute(node, "keepdims", exc=False)
    axes = self.get_attribute(node, "axes", exc=False)
    if axes is None:
        if len(node.input) == 2:
            assert self.is_constant(node.input[1]), (
                f"axes {node.input[1]!r} from node {node.op_type}, "
                f"name={node.name!r} is not a constant, "
                f"the new shape cannot be infered{self.get_debug_msg()}"
            )
            cst = self.get_constant(node.input[1])
            if isinstance(cst, NodeProto) and self.is_constant(cst.output[0]):
                cst = self.get_constant(node.input[1], computed_value=True)
            assert isinstance(cst, (np.ndarray, self.torch.Tensor)), (
                f"Unexpected type {type(cst)} for {node.input[1]!r}, "
                f"unable to set type and shape for node {node.op_type} "
                f"with name={node.name!r}{self.get_debug_msg()}"
            )
            if isinstance(cst, self.torch.Tensor):
                cst = cst.cpu()
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


def _set_shape_type_op_any_reshape(self: "GraphBuilder", node: NodeProto):  # noqa: F821
    "Sets the output shape for node type Reshape."
    k = node.output[0]
    if self.has_type(node.input[0]):
        self.set_type(k, self.get_type(node.input[0]))
    shape_set = False
    value = None
    if self.is_constant(node.input[1]):
        value = self.get_constant(node.input[1], computed_value=True, as_shape=True, exc=False)
    if value is None:
        value = self.value_as_shape(node.input[1])
    if value is not None:
        cst = tuple(value)
        if all_int_or_str(cst):
            if -1 not in cst and 0 not in cst:
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
        else:
            assert not self._debug_shape_missing, (
                f"Unable to compute shape for node: "
                f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
            )


def _set_shape_type_op_any_expand(self: "GraphBuilder", node: NodeProto):  # noqa: F821
    "Sets the output shape for node type Reshape."
    k = node.output[0]
    if self.has_type(node.input[0]):
        self.set_type(k, self.get_type(node.input[0]))
    shape_set = False
    value = None
    if self.is_constant(node.input[1]):
        value = self.get_constant(node.input[1], computed_value=True, as_shape=True, exc=False)
    if value is None:
        value = self.value_as_shape(node.input[1])
    if value is not None:
        cst = tuple(value)
        if all_int_or_str(cst):
            if -1 not in cst and 1 not in cst and 0 not in cst:
                self.set_shape(k, cst)
                shape_set = True
            elif all_int(cst) and self.has_shape(node.input[0]):
                sh = self.get_shape(node.input[0])
                new_shape = self._apply_expand_to_shape(sh, cst)
                if new_shape is not None:
                    self.set_shape(k, new_shape)
                    shape_set = True

    if not shape_set:
        if self.has_shape(node.input[1]):
            rk = self.get_shape(node.input[1])
            self.set_rank(k, rk[0])
        else:
            assert not self._debug_shape_missing, (
                f"Unable to compute shape for node: "
                f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
            )


def _set_shape_type_op_any_sign(self: "GraphBuilder", node: NodeProto):  # noqa: F821
    "Sets the output shape for node type Sign."
    set_type_shape_unary_op(self, node.output[0], node.input[0])


def _set_shape_type_op_any_slice(self: "GraphBuilder", node: NodeProto):  # noqa: F821
    "Sets the output shape for node type Slice."
    self.set_type(node.output[0], self.get_type(node.input[0]))
    if self.has_rank(node.input[0]):
        self.set_rank(node.output[0], self.get_rank(node.input[0]))
    else:
        assert not self._debug_shape_missing, (
            f"Unable to compute shape for node: "
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
        )


def _set_shape_type_op_any_split(self: "GraphBuilder", node: NodeProto):  # noqa: F821
    "Sets the output shape for node type Split."
    num_outputs = self.get_attribute(node, "num_outputs", exc=False)
    assert num_outputs is None or num_outputs.i == len(
        node.output
    ), f"Unexpected number of outputs (should be {num_outputs.i}) for node {node}"
    if not self.has_type(node.input[0]):
        # the main type is missing, cannot continue
        assert not self._debug_shape_missing, (
            f"Unable to compute shape for node: "
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
        )
        return
    dtype = self.get_type(node.input[0])
    for o in node.output:
        self.set_type(o, dtype)
    if (
        self.has_shape(node.input[0])
        and len(node.input) > 1
        and self.is_constant(node.input[1])
    ):
        splits = list(self.get_constant(node.input[1]))
        assert len(splits) == len(
            node.output
        ), f"Unexpected number of outputs, output={node.output} splits={splits}"
        att = self.get_attribute(node, "axis", exc=False)
        axis = 0 if att is None else att.i

        sh = list(self.get_shape(node.input[0]))
        for i, o in enumerate(node.output):
            sh[axis] = int(splits[i])
            self.set_shape(o, tuple(sh), allow_zero=True)
    elif self.has_rank(node.input[0]):
        rank = self.get_rank(node.input[0])
        for o in node.output:
            self.set_rank(o, rank)
    else:
        assert not self._debug_shape_missing, (
            f"Unable to compute shape for node: "
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
        )


def _set_shape_type_op_any_scatternd(self: "GraphBuilder", node: NodeProto):  # noqa: F821
    "Sets the output shape for node type ScatterND."
    if not self.has_type(node.input[0]):
        # the main type is missing, cannot continue
        assert not self._debug_shape_missing, (
            f"Unable to compute shape for node: "
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
        )
        return
    dtype = self.get_type(node.input[0])
    self.set_type(node.output[0], dtype)
    if self.has_shape(node.input[0]):
        self.set_shape(node.output[0], self.get_shape(node.input[0]))
    elif self.has_rank(node.input[0]):
        self.set_rank(node.output[0], self.get_rank(node.input[0]))
    else:
        assert not self._debug_shape_missing, (
            f"Unable to compute shape for node: "
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
        )


def _set_shape_type_op_any_transpose(self: "GraphBuilder", node: NodeProto):  # noqa: F821
    "Sets the output shape for node type Transpose."
    if not self.has_type(node.input[0]):
        # the main type is missing, cannot continue
        assert not self._debug_shape_missing, (
            f"Unable to compute shape for node: "
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
        )
        return
    dtype = self.get_type(node.input[0])
    self.set_type(node.output[0], dtype)
    if self.has_shape(node.input[0]):
        perm = list(self.get_attribute(node, "perm").ints)
        shape = self.get_shape(node.input[0])
        assert len(perm) == len(shape), (
            f"Mismatch between perm={perm} and shape={shape}, "
            f"for op {node.op_type!r} and name={node.name!r}"
            f"{self.get_debug_msg()}"
        )
        new_shape = list(range(len(perm)))
        for i, p in enumerate(perm):
            new_shape[i] = shape[p]
        self.set_shape(node.output[0], tuple(new_shape))
    elif self.has_rank(node.input[0]):
        self.set_rank(node.output[0], self.get_rank(node.input[0]))
    else:
        assert not self._debug_shape_missing, (
            f"Unable to compute shape for node: "
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
        )


def _set_shape_type_op_any_tile(self: "GraphBuilder", node: NodeProto):  # noqa: F821
    "Sets the output shape for node type Tile."
    self.set_type(node.output[0], self.get_type(node.input[0]))
    if self.has_rank(node.input[0]):
        self.set_rank(node.output[0], self.get_rank(node.input[0]))
    else:
        assert not self._debug_shape_missing, (
            f"Unable to compute shape for node: "
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
        )


def _set_shape_type_op_any_topk(self: "GraphBuilder", node: NodeProto):  # noqa: F821
    "Sets the output shape for node type TopK."
    is_scalar = self.is_constant(node.input[1])
    if is_scalar and self.has_shape(node.input[0]):
        att = self.get_attribute(node, "axis", exc=False)
        axis = att.i if att is not None else -1
        shape = list(self.get_shape(node.input[0]))
        k = self.get_constant(node.input[1], computed_value=True)
        ki = int(k) if k.shape == tuple() else int(k[0])
        shape[axis] = ki
        shape = tuple(shape)
    else:
        shape = None

    if node.output[0]:
        self.set_type(node.output[0], self.get_type(node.input[0]))
        if shape is not None:
            self.set_shape(node.output[0], shape)
        elif self.has_rank(node.input[0]):
            self.set_rank(node.output[0], self.get_rank(node.input[0]))
        else:
            assert not self._debug_shape_missing, (
                f"Unable to compute shape for node: "
                f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
            )
    if node.output[1]:
        self.set_type(node.output[1], TensorProto.INT64)
        if shape is not None:
            self.set_shape(node.output[1], shape)
        elif self.has_rank(node.input[0]):
            self.set_rank(node.output[1], self.get_rank(node.input[0]))
        else:
            assert not self._debug_shape_missing, (
                f"Unable to compute shape for node: "
                f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
            )


def _set_shape_type_op_any_unsqueeze(self: "GraphBuilder", node: NodeProto):  # noqa: F821
    "Sets the output shape for node type Unsqueeze."
    if not self.has_type(node.input[0]):
        # the main type is missing, cannot continue
        assert not self._debug_shape_missing, (
            f"Unable to compute shape for node: "
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
        )
        return
    dtype = self.get_type(node.input[0])
    self.set_type(node.output[0], dtype)
    if self.has_shape(node.input[0]):
        if len(node.input) == 1:
            c = self.get_attribute(node, "axes")
            cst = np.array(c.ints, dtype=np.int64)
        else:
            assert self.is_constant(node.input[1]), (
                f"axes {node.input[1]!r} from node {node.op_type}, "
                f"name={node.name!r} is not a constant, "
                f"the new shape cannot be infered{self.get_debug_msg()}"
            )
            cst = self.get_constant(node.input[1])
            if isinstance(cst, NodeProto) and cst.op_type in (
                "Constant",
                "Identity",
                "ConstantOfShape",
            ):
                cst = self.get_constant(node.input[1], computed_value=True)

        if isinstance(cst, np.ndarray):
            iaxes = (int(cst),) if len(cst.shape) == 0 else tuple(int(i) for i in cst)
            shape = list(self.get_shape(node.input[0]))
            for i in iaxes:
                shape.insert((i + len(shape) + 1) if i < 0 else i, 1)
            self.set_shape(node.output[0], tuple(shape))
        elif isinstance(cst, self.torch.Tensor):
            with self.maybe_disable_fake_tensor_mode():
                iaxes = (int(cst),) if len(cst.shape) == 0 else tuple(int(i) for i in cst)
                shape = list(self.get_shape(node.input[0]))
                for i in iaxes:
                    shape.insert((i + len(shape) + 1) if i < 0 else i, 1)
                self.set_shape(node.output[0], tuple(shape))
        else:
            raise AssertionError(
                f"Unexpected type {type(cst)} for {node.input[1]!r}, "
                f"unable to set type and shape for node {node.op_type} "
                f"with name={node.name!r}{self.get_debug_msg()}"
            )
    elif self.has_rank(node.input[0]) and self.is_constant(node.input[1]):
        cst = self.get_constant(node.input[1], computed_value=True)
        self.set_rank(node.output[0], self.get_rank(node.input[0]) + cst.size)
    else:
        assert not self._debug_shape_missing, (
            f"Unable to compute shape for node: "
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
        )


def _set_shape_type_op_any_squeeze(self: "GraphBuilder", node: NodeProto):  # noqa: F821
    "Sets the output shape for node type Squeeze."
    if not self.has_type(node.input[0]):
        # the main type is missing, cannot continue
        assert not self._debug_shape_missing, (
            f"Unable to compute shape for node: "
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
        )
        return
    dtype = self.get_type(node.input[0])
    self.set_type(node.output[0], dtype)
    if len(node.input) == 1 and not node.attribute:
        # No axes specified.
        if self.has_shape(node.input[0]):
            shape_x = self.get_shape(node.input[0])
            if all_int(shape_x):
                new_shape = tuple(s for s in shape_x if s != 1)
                self.set_shape(node.output[0], new_shape)
        # In other cases, we cannot really determine the new shape for sure.
    elif self.has_shape(node.input[0]):
        if len(node.input) == 1:
            c = self.get_attribute(node, "axes")
            cst = np.array(c.ints, dtype=np.int64)
        else:
            assert self.is_constant(node.input[1]), (
                f"axes from node {node.op_type}, "
                f"name={node.name!r} is not a constant, "
                f"the new shape cannot be infered{self.get_debug_msg()}"
            )
            cst = self.get_constant(node.input[1])
            if isinstance(cst, NodeProto) and cst.op_type in (
                "Constant",
                "Identity",
                "ConstantOfShape",
            ):
                cst = self.get_constant(node.input[1], computed_value=True)
        if isinstance(cst, np.ndarray):
            iaxes = set((int(cst),) if len(cst.shape) == 0 else tuple(int(i) for i in cst))
            shape = list(self.get_shape(node.input[0]))
            iaxes = set((i + len(shape)) % len(shape) for i in iaxes)  # for negative value
            new_shape = tuple(s for i, s in enumerate(shape) if i not in iaxes)
            self.set_shape(node.output[0], new_shape)
        elif isinstance(cst, self.torch.Tensor):
            with self.maybe_disable_fake_tensor_mode():
                iaxes = set((int(cst),) if len(cst.shape) == 0 else tuple(int(i) for i in cst))
                shape = list(self.get_shape(node.input[0]))
                iaxes = set((i + len(shape)) % len(shape) for i in iaxes)  # for negative value
                new_shape = tuple(s for i, s in enumerate(shape) if i not in iaxes)
                self.set_shape(node.output[0], new_shape)
        else:
            raise AssertionError(
                f"Unexpected type {type(cst)} for {node.input[1]!r}, "
                f"unable to set type and shape for node {node.op_type} "
                f"with name={node.name!r}{self.get_debug_msg()}"
            )
    elif self.has_rank(node.input[0]) and self.is_constant(node.input[1]):
        cst = self.get_constant(node.input[1], computed_value=True)
        self.set_rank(node.output[0], self.get_rank(node.input[0]) - cst.size)
    else:
        assert not self._debug_shape_missing, (
            f"Unable to compute shape for node: "
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
        )


def _set_shape_type_op_any_where(self: "GraphBuilder", node: NodeProto):  # noqa: F821
    "Sets the output shape for node type Where."
    if not self.has_type(node.input[2]):
        assert not self._debug_shape_missing, (
            f"Unable to compute shape for node: "
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
        )
        return
    self.set_type(node.output[0], self.get_type(node.input[2]))
    if (
        self.has_shape(node.input[0])
        and self.has_shape(node.input[1])
        and self.has_shape(node.input[2])
    ):
        sh1 = broadcast_shape(
            self.get_shape(node.input[0]),
            self.get_shape(node.input[1]),
            graph_builder=self,
        )
        sh = broadcast_shape(sh1, self.get_shape(node.input[2]), graph_builder=self)
        self.set_shape(node.output[0], sh)
    elif all(self.has_rank(i) for i in node.input):
        self.set_rank(node.output[0], max(self.get_rank(i) for i in node.input))
    else:
        assert not self._debug_shape_missing, (
            f"Unable to compute shape for node: "
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
        )


def _set_shape_type_op_any_unary(
    self: "GraphBuilder",  # noqa: F821
    node: NodeProto,
    itype: Optional[int] = None,
):
    "Sets the output shape for any unary type."
    return set_type_shape_unary_op(self, node.output[0], node.input[0], itype=itype)


_set_shape_type_op_any_known = {
    "BatchNormalization": _set_shape_type_op_any_batch_normalization,
    "Cast": _set_shape_type_op_any_cast,
    "Concat": _set_shape_type_op_any_concat,
    "Conv": _set_shape_type_op_any_conv_max_pool,
    "Expand": _set_shape_type_op_any_expand,
    "Gather": _set_shape_type_op_any_gather,
    "GatherElements": _set_shape_type_op_any_gather_elements,
    "Gemm": _set_shape_type_op_any_gemm,
    "IsInf": lambda *args: _set_shape_type_op_any_unary(*args, itype=TensorProto.BOOL),
    "IsNaN": lambda *args: _set_shape_type_op_any_unary(*args, itype=TensorProto.BOOL),
    "LayerNormalization": _set_shape_type_op_any_layer_normalization,
    "MatMul": _set_shape_type_op_any_matmul,
    "MaxPool": _set_shape_type_op_any_conv_max_pool,
    "NonZero": _set_shape_type_op_any_non_zero,
    "Pad": _set_shape_type_op_any_pad,
    "Range": _set_shape_type_op_any_range,
    "Reshape": _set_shape_type_op_any_reshape,
    "RotaryEmbedding": _set_shape_type_op_any_rotary_embedding,
    "ScatterND": _set_shape_type_op_any_scatternd,
    "Sign": _set_shape_type_op_any_sign,
    "Slice": _set_shape_type_op_any_slice,
    "Split": _set_shape_type_op_any_split,
    "Squeeze": _set_shape_type_op_any_squeeze,
    "Tile": _set_shape_type_op_any_tile,
    "TopK": _set_shape_type_op_any_topk,
    "Transpose": _set_shape_type_op_any_transpose,
    "Unsqueeze": _set_shape_type_op_any_unsqueeze,
    "Where": _set_shape_type_op_any_where,
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
        r = set_type_shape_binary_op(self, node.output[0], *node.input, cmp_op=True)
        assert r or not self._debug_shape_missing, (
            f"No function to compute shape for node: "
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
        )
    elif node.op_type in self._op_type_element_wise_types:
        r = set_type_shape_binary_op(self, node.output[0], *node.input)
        assert r or not self._debug_shape_missing, (
            f"No function to compute shape for node: "
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
        )
    elif node.op_type in {"DequantizeLinear", "DynamicQuantizeLinear"}:
        raise AssertionError(
            f"set_shape_type_op_any not implemented for "
            f"{node.op_type!r}{self.get_debug_msg()}"
        )
    elif node.op_type in {"CastLike"}:
        r = set_type_shape_binary_op(
            self, node.output[0], node.input[0], itype=self.get_type(node.input[1])
        )
        assert r or not self._debug_shape_missing, (
            f"No function to compute shape for node: "
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
        )
    elif node.op_type in {"Pow"}:
        r = set_type_shape_binary_op(self, node.output[0], *node.input)
        assert r or not self._debug_shape_missing, (
            f"No function to compute shape for node: "
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
        )
    elif node.op_type in self._op_type_unary_like:
        if node.op_type == "Abs":
            r = set_type_shape_unary_op_abs(self, node.output[0], node.input[0])
        else:
            r = set_type_shape_unary_op(self, node.output[0], node.input[0])
        assert r or not self._debug_shape_missing, (
            f"No function to compute shape for node: "
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
        )
    elif node.op_type in {"ScatterElements", "ScatterND"}:
        r = set_type_shape_unary_op(self, node.output[0], node.input[0])
        assert r or not self._debug_shape_missing, (
            f"No function to compute shape for node: "
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
        )
    elif node.op_type not in {"Constant", "ConstantOfShape", "Identity", "Reshape", "Shape"}:
        # Some nodes are handled when the node is created such as Identity.
        assert not self._debug_shape_missing, (
            f"No function to compute shape for node: "
            f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
        )


def set_type_shape_fused_matmul(self: "GraphBuilder", node: NodeProto):  # noqa: F821
    name = node.output[0]
    x, y = node.input[:2]
    transA = self.get_attribute(node, "transA", exc=False)
    transA = transA.i if transA else 0
    transB = self.get_attribute(node, "transB", exc=False)
    transB = transB.i if transB else 0
    if transA == 0 and transB == 0:
        return set_type_shape_matmul(self, name, x, y)
    if self.has_type(x):
        self.set_type(name, self.get_type(x))
    elif self.has_type(y):
        self.set_type(name, self.get_type(y))
    if self.has_shape(x) and self.has_shape(y):
        sh1 = self.get_shape(x)
        sh2 = self.get_shape(y)
        if len(sh1) != len(sh2):
            if len(sh1) < len(sh2):
                sh1 = ((1,) * (len(sh2) - len(sh1))) + sh1
            else:
                sh2 = ((1,) * (len(sh1) - len(sh2))) + sh2
        prefix = (
            broadcast_shape(sh1[:-2], sh2[:-2], graph_builder=self)
            if len(sh1) > 2
            else tuple()
        )
        new_shape = (sh1[-1] if transA else sh1[-2], sh2[-2] if transB else sh2[-1])
        self.set_shape(name, prefix + new_shape)
        self.set_shape(name, prefix + new_shape)
    elif self.has_rank(x) and self.has_rank(y):
        self.set_rank(name, max(self.get_rank(x), self.get_rank(y)))


_set_shape_type_op_any_custom = {
    "FusedMatMul": set_type_shape_fused_matmul,
}


def set_type_shape_tree_ensemble(self: "GraphBuilder", node: NodeProto):  # noqa: F821
    self.set_type(node.output[0], self.get_type(node.input[0]))
    n_targets = self.get_attribute(node, "n_targets", exc=False)
    assert n_targets is not None, (
        f"Unable to extract the dimension of the output for node type "
        f"{node.op_type!r} and name={node.name!r}"
    )
    if self.has_shape(node.input[0]):
        shape = self.get_shape(node.input[0])
        self.set_shape(node.output[0], (shape[0], n_targets.i))
    else:
        self.set_rank(node.output[0], 2)


def set_shape_type_custom(self: "GraphBuilder", node: NodeProto):  # noqa: F821
    """
    Sets the shape and type if it can.
    """
    if node.domain == "ai.onnx.ml":
        if node.op_type == "TreeEnsembleRegressor":
            set_type_shape_tree_ensemble(self, node)
        return
    if node.op_type in {"ReplaceZero", "NegXplus1"}:
        set_type_shape_unary_op(self, node.output[0], node.input[0])
        return
    if node.op_type in _set_shape_type_op_any_custom:
        _set_shape_type_op_any_custom[node.op_type](self, node)
        return
    assert not self._debug_shape_missing, (
        f"No function to compute shape for node: "
        f"{self.pretty_node(node, shape=True)}{self.get_debug_msg()}"
    )
