import onnx
import onnx.helper as oh
from typing import Iterator, Optional, Set


def element_wise_binary_op_types() -> Set[str]:
    """
    Returns the list of element-wise operators.

    .. runpython::
        :showcode:

        import pprint
        from experimental_experiment.xshape._onnx_helper import (
            element_wise_binary_op_types,
        )
        pprint.pprint(element_wise_binary_op_types())
    """
    return {
        "Add",
        "And",
        "Div",
        "Mul",
        "Mod",
        "Or",
        "Sub",
        "Xor",
    }


def element_wise_op_cmp_types() -> Set[str]:
    """
    Returns the list of element-wise operators
    doing comparisons.

    .. runpython::
        :showcode:

        import pprint
        from experimental_experiment.xshape._onnx_helper import element_wise_op_cmp_types
        pprint.pprint(element_wise_op_cmp_types())
    """
    return {
        "Equal",
        "Greater",
        "GreaterOrEqual",
        "Less",
        "LessOrEqual",
    }


def unary_like_op_types() -> Set[str]:
    """
    Returns the list of unary *like* operators.
    They do not change the shape. They may change the type.

    .. runpython::
        :showcode:

        import pprint
        from experimental_experiment.xshape._onnx_helper import unary_like_op_types
        pprint.pprint(unary_like_op_types())
    """
    return {
        "Abs",
        "Acos",
        "Acosh",
        "Asin",
        "Asinh",
        "Atan",
        "Atanh",
        "BitShift",
        "Cast",
        "CastLike",
        "Ceil",
        "Celu",
        "Clip",
        "Cos",
        "Cosh",
        "DequantizeLinear",
        "DynamicQuantizeLinear",
        "Elu",
        "Erf",
        "Exp",
        "IsInf",
        "Log",
        "LogSoftmax",
        "Neg",
        "Not",
        "PRelu",
        "QuantizeLinear",
        "Reciprocal",
        "Relu",
        "Round",
        "Selu",
        "Sigmoid",
        "Sign",
        "Sin",
        "Sinh",
        "Softmax",
        "SoftmaxCrossEntropyLoss",
        "Softplus",
        "Softsign",
        "Sqrt",
        "Tan",
        "Tanh",
        "ThresholdRelu",
    }


def str_tensor_proto_type() -> str:
    """
    Returns the following string:

    .. runpython::
        :showcode:

        from experimental_experiment.xshape._onnx_helper import str_tensor_proto_type

        print(str_tensor_proto_type())
    """
    mapping = [
        (getattr(onnx.TensorProto, att), att)
        for att in dir(onnx.TensorProto)
        if att.upper() == att and isinstance(getattr(onnx.TensorProto, att), int)
    ]
    mapping.sort()
    return ", ".join(f"{k}:{v}" for k, v in mapping)


def enumerate_subgraphs(graph: onnx.GraphProto) -> Iterator[onnx.GraphProto]:
    """
    Enumerates all inputs from a node including all the hidden inputs
    from subgraphs.
    """
    yield graph
    for node in graph.node:
        if node.op_type[0] in "LSI" and node.op_type in {"Loop", "Scan", "If", "SequenceMap"}:
            for att in node.attribute:
                if att.type == onnx.AttributeProto.GRAPH:
                    yield from enumerate_subgraphs(att.g)


def _rewrite_info(info: onnx.ValueInfoProto):
    shape = []
    for i, dim in enumerate(info.type.tensor_type.shape.dim):
        if dim.dim_param:
            shape.append(dim.dim_param)
        else:
            name = f"dim{i}_{info.name}"
            shape.append(name)
    return oh.make_tensor_value_info(info.name, info.type.tensor_type.elem_type, shape)


def overwrite_shape_in_model_proto(
    model: onnx.ModelProto, n_in: Optional[int] = None
) -> onnx.ModelProto:
    """
    Removes inferred shapes. Overwrites input shapes to make them all dynamic.
    ``n_in`` indicates the number of inputs for which the shape must be rewritten.
    """
    assert isinstance(model, onnx.ModelProto), f"Unexpected type {type(model)} for model."
    for subgraph in enumerate_subgraphs(model.graph):
        new_info = [
            _rewrite_info(inp) if n_in is None or i < n_in else inp
            for i, inp in enumerate(subgraph.input)
        ]
        del subgraph.input[:]
        subgraph.input.extend(new_info)
        new_info = [_rewrite_info(i) for i in subgraph.output]
        del subgraph.output[:]
        subgraph.output.extend(new_info)
    return model
