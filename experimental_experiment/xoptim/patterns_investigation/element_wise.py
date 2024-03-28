from typing import List, Optional
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization


class BinaryInvestigation(PatternOptimization):
    """
    Looks into
    """

    _ops = {"Add", "Div", "Mul", "Sub"}

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type not in self._ops:
            return self.none()
        left = g.node_before(node.input[0])
        right = g.node_before(node.input[1])
        if left is None and right is None:
            return self.none()

        nodes = [node, left, right]
        if self.verbose:
            print(f"[{self.__class__.__name__}] {self.report(g, *nodes)}")
        return self.none()

    @classmethod
    def _str(cls, g, node):
        if node.op_type in cls._ops:
            sh1 = g.get_shape(node.input[0]) if g.has_shape(node.input[0]) else ("?",)
            sh2 = g.get_shape(node.input[1]) if g.has_shape(node.input[1]) else ("?",)
            if len(sh1) == 0:
                sh1 = (1,)
            if len(sh2) == 0:
                sh2 = (1,)
            sh1 = "x".join(map(str, sh1))
            sh2 = "x".join(map(str, sh2))
            return f"{node.op_type}({sh1}, {sh2})"

        return f"{node.op_type}(...)"

    @classmethod
    def report(
        cls,
        g: "GraphBuilder",  # noqa: F821
        node: NodeProto,
        left: Optional[NodeProto],
        right: Optional[NodeProto],
    ):
        rows = [cls._str(g, node)]
        rows.append(f"[{cls._str(g, left)}]" if left is not None else "[?]")
        rows.append(f"[{cls._str(g, right)}]" if right is not None else "[?]")
        return " --- ".join(rows)
