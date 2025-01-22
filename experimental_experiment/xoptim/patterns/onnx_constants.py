from typing import List, Optional
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization


class ConstantToInitiliazerPattern(PatternOptimization):
    """Replaces a node Constant by an initializer and a node Identity."""

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Constant" or node.domain != "":
            return self.none()
        return MatchResult(self, [node], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node: NodeProto,
    ) -> List[NodeProto]:
        cst = g.get_computed_constant(node.output[0])
        assert (
            cst is not None
        ), f"Node {g.pretty_node(cst)} is a constant, it must be possible to evaluate it."
        init = g.make_initializer(f"{node.output[0]}_cst2init", cst)
        return [
            g.make_node(
                "Identity", [init], node.output, name=f"{self.__class__.__name__}--{node.name}"
            )
        ]
