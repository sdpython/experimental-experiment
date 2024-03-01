from typing import List, Optional
from onnx import NodeProto
from ..annotations import all_int
from .patterns_api import MatchResult, PatternOptimization


class ExpandPattern(PatternOptimization):
    """
    Checks that a Expand is really needed.
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Expand" or node.domain != "":
            return None
        if not g.has_shape(node.input[0]):
            return None
        shape = g.get_shape(node.input[0])
        if not all_int(shape):
            return None
        if not g.is_constant(node.input[1]):
            # It may be a symbolic shape.
            return None
        new_shape = tuple(g.get_computed_constant(node.input[1]))
        if shape != new_shape:
            return

        def apply(g: "GraphBuilder", node: NodeProto) -> List[NodeProto]:  # noqa: F821
            new_node = g.make_node(
                "Identity",
                node.input,
                node.output,
                name=f"{self.__class__.__name__}--{node.name}",
                doc_string=node.doc_string,
            )
            return [new_node]

        return MatchResult(self, [node], apply)
