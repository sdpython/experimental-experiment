import inspect
from typing import List, Optional
from onnx import NodeProto
from ..patterns.patterns_api import MatchResult, PatternOptimization


class AddReductionScatterND(PatternOptimization):
    """
    Replaces ConstantOfShape + ScatterND with ScatterNDOfShape (com.domain).
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "ScatterND" or node.domain != "":
            return self.none()

        if g.get_attribute(node, "reduction", exc=False) is not None:
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [node], self.apply, insert_at=node)

    @classmethod
    def apply(cls, g: "GraphBuilder", node: NodeProto) -> List[NodeProto]:  # noqa: F821
        new_node = g.make_node(
            "ScatterND",
            node.input,
            node.output,
            reduction="add",
            name=f"{cls.__name__}--{node.name}",
            doc_string=node.doc_string,
        )
        return [new_node]
