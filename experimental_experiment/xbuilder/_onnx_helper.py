from typing import Dict, Iterator, Optional, Set, Tuple, Union
import numpy as np
from onnx import (
    AttributeProto,
    FunctionProto,
    GraphProto,
    ModelProto,
    NodeProto,
    TensorShapeProto,
)
from onnx.defs import onnx_opset_version, get_all_schemas_with_history


def _default_OPSET_TO_IR_VERSION() -> Dict[int, int]:
    """
    Returns the dictionary mapping the main opset
    to the corresponding `ir_version`.
    """
    return {
        1: 3,
        2: 3,
        3: 3,
        4: 3,
        5: 3,
        6: 3,
        7: 3,
        8: 4,
        9: 4,
        10: 5,
        11: 6,
        12: 7,
        13: 7,
        14: 7,
        15: 8,
        16: 8,
        17: 8,
        18: 8,
        19: 9,
        20: 9,
        21: 10,
        22: 10,
        23: 11,
    }


OPSET_ML_TO_OPSET: Dict[int, int] = {1: 11, 2: 15, 3: 18}


def enumerate_subgraphs(
    node: NodeProto, recursive: bool = True
) -> Iterator[Tuple[Tuple[NodeProto, str, GraphProto], ...]]:
    """
    Returns the subgraphs inside a graph.
    """
    for att in node.attribute:
        if att.type == AttributeProto.GRAPH and att.g:
            this = node, att.name, att.g
            yield this

            for no in att.g.node:
                for tu in enumerate_subgraphs(no):
                    yield this + tu


_history = None


def _nice_shape(shape: TensorShapeProto) -> str:
    els = []
    for sh in shape.dim:
        els.append(str(sh.dim_value) if sh.HasField("dim_value") else sh.dim_param)
    return "x".join(els)


def compatible_opsets(domain: str, op_type: str, current: int, new_version: int) -> bool:
    """
    Tells if two opset version for a particular operator type
    means the same version of it.

    :param domain: domain, only `ai.onnx` and `ai.onnx.ml` are checked.
    :param op_type: operator type
    :param current: current domain version
    :param new_version: new version
    :return: result
    """
    global _history
    if _history is None:
        res = {}
        for schema in get_all_schemas_with_history():
            domain = schema.domain
            version = schema.since_version
            name = schema.name
            if domain not in res:
                res[domain] = {}
            if name not in res[domain]:
                res[domain][name] = {}
            res[domain][name][version] = schema
        _history = res

    assert domain in _history, f"Unable to find domain {domain!r} in {list(sorted(_history))}."
    assert op_type in _history[domain], (
        f"Unable to find op_type {op_type!r}, domain={domain!r} "
        f"in {list(sorted(_history[domain]))}"
    )
    hist = _history[domain][op_type]
    version = list(sorted(hist))  # noqa: C413
    pos = np.searchsorted(version, current, side="right") - 1
    assert pos >= 0, (
        f"Available version for {op_type!r} from {domain!r}, "
        f"incompatible version is {current}"
    )
    if pos < len(version) - 1:
        a, b = version[pos], version[pos + 1]
        return a <= new_version < b
    return new_version >= version[pos]


def _get_default_opset_for_domain(domain: str, main_opset: Optional[int] = None) -> int:
    """
    Returns the associated for a domain given the main opset.

    :param domain: domain
    :param main_opset: opset for the main domain
    :return: version
    """
    if main_opset is None:
        main_opset = choose_consistent_domain_opset("")
    if domain == "":
        return main_opset
    if domain == "ai.onnx.ml":
        if main_opset >= 18:
            return 3
        if main_opset >= 6:
            return 2
        return 1
    if domain == "ai.onnx.training":
        return 1
    return None


def choose_consistent_domain_opset(
    domain: str, opsets: Optional[Dict[str, int]] = None
) -> int:
    """
    Chooses a compatible opset for a particular domain given
    this existing one. Only works for `ai.onnx.ml`,
    otherwise return 1.

    :param domain: new domain
    :param opsets: existing opsets
    :return: version
    """
    opsets = opsets or {}
    if domain in opsets:
        return opsets[domain]
    if domain == "":
        assert "ai.onnx.ml" not in opsets, (
            "If ai.onnx.ml is part of your model, "
            "your should add a version for the main opset as well."
        )
        return onnx_opset_version() - 2
    if domain != "ai.onnx.ml":
        return 1
    return _get_default_opset_for_domain(domain)


def element_wise_binary_op_types() -> Set[str]:
    """
    Returns the list of element-wise operators.

    .. runpython::
        :showcode:

        import pprint
        from experimental_experiment.xbuilder._onnx_helper import (
            element_wise_binary_op_types,
        )
        pprint.pprint(element_wise_binary_op_types())
    """
    return {
        "Add",
        "Div",
        "Mul",
        "Sub",
        "And",
        "Or",
        "Mod",
        "Xor",
    }


def element_wise_op_cmp_types() -> Set[str]:
    """
    Returns the list of element-wise operators
    doing comparisons.

    .. runpython::
        :showcode:

        import pprint
        from experimental_experiment.xbuilder._onnx_helper import element_wise_op_cmp_types
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
        from experimental_experiment.xbuilder._onnx_helper import unary_like_op_types
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


def same_function_proto(
    f1: FunctionProto, f2: FunctionProto, verbose: int = 0
) -> Union[str, bool]:
    """
    Compares two functions and tells if they are equal.

    :param f1: first function
    :param f2: second function
    :param verbose: to know why the comparison failed,
        the function returns a string in that case or True
    :return: comparison

    They may have different names.
    """
    if len(f1.input) != len(f2.input):
        return "different number of inputs" if verbose else False
    if len(f1.output) != len(f2.output):
        return "different number of outputs" if verbose else False
    if len(f1.node) != len(f2.node):
        return "different number of nodes" if verbose else False
    if len(f1.attribute) != len(f2.attribute):
        return "different number of attributes" if verbose else False
    if len(f1.attribute_proto) != len(f2.attribute_proto):
        return "different number of attributes (2)" if verbose else False
    if list(f1.attribute) != list(f2.attribute):
        return "different attribute names" if verbose else False
    if [a.SerializeToString() for a in f1.attribute_proto] != [
        a.SerializeToString() for a in f2.attribute_proto
    ]:
        return "different attribute protos" if verbose else False
    mapped = dict(zip(f1.input, f2.input))
    for i, (n1, n2) in enumerate(zip(f1.node, f2.node)):
        if n1.op_type != n2.op_type:
            return (
                f"different node type at position {i} - {n1.op_type} != {n2.op_type}"
                if verbose
                else False
            )
        if len(n1.input) != len(n2.input):
            return f"different number of inputs at node {i}" if verbose else False
        if len(n1.output) != len(n2.output):
            return f"different number of outputs at node {i}" if verbose else False
        if len(n1.attribute) != len(n2.attribute):
            return f"different number of attributes at node {i}" if verbose else False
        n2_input = [mapped[i] for i in n1.input]
        if list(n2.input) != n2_input:
            return (
                f"different input names at node {i}, {n1.input}, {n2.input} != {n2_input}"
                if verbose
                else False
            )
        for a1, a2 in zip(n1.attribute, n2.attribute):
            # The test should be improved for subgraphs.
            if a1.SerializeToString() != a2.SerializeToString():
                return f"different attribute {a1.name!r}" if verbose else False
        mapped.update(dict(zip(n1.output, n2.output)))

    f2_output = [mapped[i] for i in f1.output]
    if list(f2.output) != f2_output:
        return (
            f"different output names, {f1.output}, {f2.output} != {f2_output}"
            if verbose
            else False
        )
    return True


def clean_shapes(proto: Union[GraphProto, ModelProto]):
    """
    Cleans all shapes inplace.
    """
    if isinstance(proto, ModelProto):
        clean_shapes(proto.graph)
        return
    del proto.value_info[:]
    for node in proto.node:
        if node.op_type not in {"Scan", "If", "Loop"}:
            continue
        for att in node.attribute:
            if att.type == AttributeProto.GRAPH:
                clean_shapes(att.g)
