from typing import List, Optional
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization


class FastGeluPattern(PatternOptimization):
    """
    Replaces Gelu by FastGelu.
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Gelu" or node.domain != "":
            return self.none()
        return MatchResult(self, [node], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        gelu_node: NodeProto,
    ) -> List[NodeProto]:
        return [
            g.make_node(
                "FastGelu",
                gelu_node.input,
                gelu_node.output,
                domain="com.microsoft",
                doc_string=gelu_node.doc_string,
                name=f"{self.__class__.__name__}--{dropout_node.name}",
            )
        ]
