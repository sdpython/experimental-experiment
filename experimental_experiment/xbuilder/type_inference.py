from typing import Optional, Sequence, Tuple, Union
from onnx import NodeProto, TensorProto


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
    "Gather",
    "Neg",
    "Pow",
    "Reciprocal",
    "ReduceMean",
    "ReduceSum",
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
    "Gemm",
    "MatMul",
    "Mul",
    "Sub",
}


def infer_types(
    node: NodeProto, input_types: Sequence[int], output_name: Optional[str]
) -> Union[int, Tuple[int]]:
    """
    Tries to infer the type of an output or all outputs.

    :param node: NodeProto
    :param input_types: type of the elements of the input tensors
    :param output_name: type for the desired output or all types if all are needed
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
        raise RuntimeError(
            f"Unable to infer type for node type {node.op_type!r}, node is {node}."
        )

    if output_name:
        assert len(node.output) == 1, (
            f"Unexpected number of outputs {node.output} "
            f"for node type {node.op_type!r}"
        )
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
    """
    The node has one output and its type is the same as the first input type.
    """
    assert (
        len(node.output) == 1
    ), f"Unexpected number of outputs {node.output} for node type {node.op_type!r}"
    assert len(input_types) >= 1, (
        f"Unexpected number of inputs {len(input_types)} "
        f"for node type {node.op_type!r}, node is {node}"
    )
    return (input_types[0],)


def _infer_type_in_o1(node: NodeProto, input_types: Sequence[int]) -> Tuple[int]:
    """
    The node has one output and its type is the same as all inputs.
    """
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
    """
    Returns the output type for a node Cast.
    """
    for att in node.attribute:
        if att.name == "to":
            return (att.i,)
    _raise_exc(node, input_types)


def _infer_type_cast_like(node: NodeProto, input_types: Sequence[int]) -> Tuple[int]:
    """
    Returns the output type for a node CastLike.
    """
    assert len(input_types) == 2, f"Missing input types {input_types}"
    return (input_types[1],)


def _infer_type_constant_of_shape(
    node: NodeProto, input_types: Sequence[int]
) -> Tuple[int]:
    """
    Returns the output type for a node Cast.
    """
    if len(node.attribute) == 0:
        return (TensorProto.FLOAT,)
    value = node.attribute[0]
    return (value.data_type,)


def _infer_type_eye_like(node: NodeProto, input_types: Sequence[int]) -> Tuple[int]:
    """
    Returns the output type for a node CastLike.
    """
    for att in node.attribute:
        if att.name == "dtype":
            return (att.i,)
    return (input_types[0],)


def _infer_type_range(node: NodeProto, input_types: Sequence[int]) -> Tuple[int]:
    """
    Returns the output type for a node Cast.
    """
    if len(node.input) == 3:
        # starts, ends, axis
        return (max(input_types[:2]),)
    _raise_exc(node, input_types)


def _infer_type_where(node: NodeProto, input_types: Sequence[int]) -> Tuple[int]:
    """
    Returns the output type for a node Where.
    """
    return (max(input_types[1:]),)


_dict_type_inference = {
    "Cast": _infer_type_cast,
    "CastLike": _infer_type_cast_like,
    "ConstantOfShape": _infer_type_constant_of_shape,
    "EyeLike": _infer_type_eye_like,
    "Range": _infer_type_range,
    "Where": _infer_type_where,
}
