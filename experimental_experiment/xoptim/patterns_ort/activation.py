from typing import List, Optional
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization
from ..patterns.onnx_functions import GeluPattern


class GeluOrtPattern(GeluPattern):
    """
    Detects the decomposed version of Gelu with Tanh

    .. math::

        y = \\frac{x}{2} \\left(1 + \\tanh\\left(\\sqrt{\\frac{2}{\\pi}}
        (x + 0.044715 * x^3)\\rigth)\\rigth)
    """

    def __init__(
        self,
        verbose: int = 0,
        priority: int = 0,
        min_opset: int = 1,
        domain: str = "com.microsoft",
    ):
        super(GeluOrtPattern, self).__init__(verbose, priority, min_opset=min_opset)
        self.domain = domain


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
        if node.op_type != "Gelu" or node.domain not in ("", "com.microsoft"):
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
                name=f"{self.__class__.__name__}--{gelu_node.name}",
            )
        ]
