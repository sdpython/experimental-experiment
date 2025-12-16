from typing import Optional, Sequence, Tuple, Union
from onnx import FunctionProto, NodeProto, TensorProto


_i1_o1_node_types = {
    "Acos",
    "Acosh",
    "Asin",
    "Asinh",
    "Atan",
    "Atanh",
    "Cos",
    "Cosh",
    "Exp",
    "Expand",
    "FastGelu",
    "Gather",
    "Gelu",
    "LogSoftmax",
    "Neg",
    "Reciprocal",
    "ReduceMean",
    "ReduceSum",
    "Relu",
    "Reshape",
    "Sigmoid",
    "Sin",
    "Sinh",
    "Slice",
    "Softmax",
    "Sqrt",
    "Tan",
    "Tanh",
    "Tile",
    "Transpose",
    "Unsqueeze",
}

_in_o1_node_types = {
    "Add",
    "Concat",
    "Div",
    "FusedMatMul",
    "Gemm",
    "MatMul",
    "Mul",
    "Squeeze",
    "Sub",
    "Unsqueeze",
}


def infer_types(
    node: Union[FunctionProto, NodeProto],
    input_types: Sequence[int],
    output_name: Optional[str] = None,
    exc: bool = True,
) -> Union[int, Tuple[int, ...]]:
    """
    Tries to infer the type of an output or all outputs.

    :param node: NodeProto
    :param input_types: type of the elements of the input tensors
    :param output_name: type for the desired output or
        all types if all are needed
    :param exc: raise an exception if type cannot be infered
    :return: tuple of types or output type
    """
    if isinstance(node, FunctionProto):
        assert (
            output_name is None
        ), f"output_name must be None if proto is a FunctionProto but output_name={output_name!r}"
        return _infer_types_function(node, input_types, exc=exc)
    return _infer_types_node(node, input_types, output_name, exc=exc)


def _infer_types_function(
    proto: FunctionProto,
    input_types: Sequence[int],
    exc: bool = True,
) -> Tuple[int, ...]:
    """
    Tries to infer the type of an output or all outputs.

    :param proto: FunctionProto
    :param input_types: type of the elements of the input tensors
    :param output_name: type for the desired output or
        all types if all are needed
    :param exc: raise an exception if type cannot be infered
    :return: tuple of types or output type
    """
    current = dict(zip(proto.input, input_types))
    for node in proto.node:
        out = _infer_types_node(node, [current[i] for i in node.input], None, exc=exc)
        current.update(dict(zip(node.output, out)))
    return [current[n] for n in proto.output]


def _infer_types_node(
    node: NodeProto,
    input_types: Sequence[int],
    output_name: Optional[str],
    exc: bool = True,
) -> Union[int, Tuple[int, ...]]:
    """
    Tries to infer the type of an output or all outputs.

    :param node: NodeProto
    :param input_types: type of the elements of the input tensors
    :param output_name: type for the desired output or
        all types if all are needed
    :param exc: raise an exception if type cannot be infered
    :return: tuple of types or output type
    """
    if node.op_type in _i1_o1_node_types:
        all_types = _infer_type_i1_o1(node, input_types)
    elif node.op_type in _in_o1_node_types:
        all_types = _infer_type_in_o1(node, input_types)
    elif node.op_type in _dict_type_inference:
        all_types = _dict_type_inference[node.op_type](node, input_types)
    else:
        all_types = None

    if not all_types:
        if exc:
            raise RuntimeError(
                f"Unable to infer type for node type {node.op_type!r}, node is {node}."
            )
        return 0

    if output_name:
        assert (
            len(node.output) == 1
        ), f"Unexpected number of outputs {node.output} for node type {node.op_type!r}"
        assert (
            node.output[0] == output_name
        ), f"Output {output_name!r} not in node.output {node.output}"
        return all_types[0]
    return all_types


def _raise_exc(node: NodeProto, input_types: Sequence[int]):
    raise RuntimeError(
        f"Unable to guess output type for node type {node.op_type!r}, "
        f"input_types={input_types}, node={node}"
    )


def _infer_type_i1_o1(node: NodeProto, input_types: Sequence[int]) -> Tuple[int]:
    """The node has one output and its type is the same as the first input type."""
    assert (
        len(node.output) == 1
    ), f"Unexpected number of outputs {node.output} for node type {node.op_type!r}"
    assert len(input_types) >= 1, (
        f"Unexpected number of inputs {len(input_types)} "
        f"for node type {node.op_type!r}, node is {node}"
    )
    return (input_types[0],)


def _infer_type_in_o1(node: NodeProto, input_types: Sequence[int]) -> Tuple[int]:
    """The node has one output and its type is the same as all inputs."""
    assert (
        len(node.output) == 1
    ), f"Unexpected number of outputs {node.output} for node type {node.op_type!r}"
    assert len(input_types) >= 1, (
        f"Unexpected number of inputs {len(input_types)} "
        f"for node type {node.op_type!r}, node is {node}"
    )
    dist = set(i for i in input_types if i != 0)
    if not dist:
        return (0,)
    assert len(dist) == 1, (
        f"Type mismatch for node type {node.op_type!r}, "
        f"input_types={input_types} in node {node}"
    )
    return (max(input_types),)


def _infer_type_cast(node: NodeProto, input_types: Sequence[int]) -> Tuple[int]:
    """Returns the output type for a node Cast."""
    for att in node.attribute:
        if att.name == "to":
            return (att.i,)
    _raise_exc(node, input_types)


def _infer_type_constant(node: NodeProto, input_types: Sequence[int]) -> Tuple[int]:
    """Returns the output type for a node Constant."""
    for att in node.attribute:
        if att.name in ("value_int", "value_ints"):
            return (TensorProto.INT64,)
        if att.name in ("value_float", "value_floats"):
            return (TensorProto.FLOAT,)
        if att.name in ("value",):
            return (att.t.data_type,)
    _raise_exc(node, input_types)


def _infer_type_cast_like(node: NodeProto, input_types: Sequence[int]) -> Tuple[int]:
    """Returns the output type for a node CastLike."""
    assert len(input_types) == 2, f"Missing input types {input_types}"
    return (input_types[1],)


def _infer_type_constant_of_shape(node: NodeProto, input_types: Sequence[int]) -> Tuple[int]:
    """Returns the output type for a node Cast."""
    if len(node.attribute) == 0:
        return (TensorProto.FLOAT,)
    value = node.attribute[0]
    return (value.data_type,)


def _infer_type_eye_like(node: NodeProto, input_types: Sequence[int]) -> Tuple[int]:
    """Returns the output type for a node CastLike."""
    for att in node.attribute:
        if att.name == "dtype":
            return (att.i,)
    return (input_types[0],)


def _infer_type_pow(node: NodeProto, input_types: Sequence[int]) -> Tuple[int]:
    """Returns the output type for a node Where."""
    raise AssertionError(f"Not implemented yet for node={node} and input_types={input_types}")


def _infer_type_range(node: NodeProto, input_types: Sequence[int]) -> Tuple[int]:
    """Returns the output type for a node Cast."""
    if len(node.input) == 3:
        # starts, ends, axis
        return (max(input_types[:2]),)
    _raise_exc(node, input_types)


def _infer_type_shape_or_size(node: NodeProto, input_types: Sequence[int]) -> Tuple[int]:
    """Returns the output type for a node Shape."""
    return (TensorProto.INT64,)


def _infer_type_split(node: NodeProto, input_types: Sequence[int]) -> Tuple[int]:
    """Returns the output type for a node Split."""
    return [input_types[0] for _ in node.output]


def _infer_type_where(node: NodeProto, input_types: Sequence[int]) -> Tuple[int]:
    """Returns the output type for a node Where."""
    return (max(input_types[1:]),)


_dict_type_inference = {
    "Cast": _infer_type_cast,
    "CastLike": _infer_type_cast_like,
    "Constant": _infer_type_constant,
    "ConstantOfShape": _infer_type_constant_of_shape,
    "EyeLike": _infer_type_eye_like,
    "Pow": _infer_type_pow,
    "Range": _infer_type_range,
    "Shape": _infer_type_shape_or_size,
    "Size": _infer_type_shape_or_size,
    "Split": _infer_type_split,
    "Where": _infer_type_where,
}
