from typing import Iterator, Tuple
from onnx import AttributeProto, GraphProto, NodeProto


def enumerate_subgraphs(
    node: NodeProto, recursive: bool = True
) -> Iterator[Tuple[Tuple[NodeProto, str, GraphProto], ...]]:
    """
    Returns the subgraphs inside a graphs.
    """
    for att in node.attribute:
        if att.type == AttributeProto.GRAPH and att.g:
            this = node, att.name, att.g
            yield this

            for no in att.g.node:
                for tu in enumerate_subgraphs(no):
                    yield this + tu
