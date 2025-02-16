from typing import List, Union
import onnx
from ..xbuilder import GraphBuilder


def unfused_nodes(
    onx: Union[GraphBuilder, onnx.ModelProto], input_names: List[str], output_names: List[str]
) -> List[onnx.NodeProto]:
    """
    Extracts from a model all the nodes starting from ``input_names`` and
    producing ``output_names``.

    This function does not handle subgraphs.

    :param onx: model
    :param input_names: list of inputs to consider
    :param output_names: list of outputs to consider
    :return: list of involved nodes
    """
    if isinstance(onx, onnx.ModelProto):
        return unfused_nodes(GraphBuilder(onx), input_names, output_names)
    assert isinstance(onx, GraphBuilder), f"Unexpected type {type(onx)} for onx."
    in_names = set(input_names)
    nodes = []
    needed = set(output_names)
    for node in onx.nodes[::-1]:
        so = set(node.output)
        if so & needed:
            nodes.append(node)
            needed |= set(i for i in node.input if i not in in_names)
            needed -= needed & set(node.output)
        if not needed:
            break
    return list(reversed(nodes))
