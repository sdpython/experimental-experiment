import inspect
from typing import List, Optional
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization
from ..patterns.onnx_functions import GeluPattern


class BiasGeluPattern(PatternOptimization):
    """
    Replaces by ``y = BiasGelu(x, B)``::

        t = x + B
        y = t ( Erf(1 / t) + 1)
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Erf" or node.domain != "":
            return self.none()
        if g.is_used_more_than_once(node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)

        div = g.node_before(node.input[0])
        if (
            not g.is_constant_scalar(div.input[1])
            or g.get_constant_scalar(div.input[1]) != 1.4140625
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        add = g.node_before(div.input[0])
        if add.op_type != "Add" or add.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant(add.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        add1_nexts = g.next_nodes(add.output[0])
        if len(add1_nexts) != 2:
            return self.none(node, inspect.currentframe().f_lineno)

        add_next = g.next_nodes(node.output[0])
        if len(add_next) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        add_1 = add_next[0]
        if add_1.op_type != "Add" or add_1.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if (
            not g.is_constant_scalar(add_1.input[1])
            or g.get_constant_scalar(add_1.input[1]) != 1
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        muls = g.next_nodes(add_1.output[0])
        if len(muls) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        mul = muls[0]
        if mul.op_type != "Mul" or mul.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if set(mul.input) != {add.output[0], add_1.output[0]}:
            return self.none(node, inspect.currentframe().f_lineno)

        halfs = g.next_nodes(mul.output[0])
        if len(halfs) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        half = halfs[0]
        if half.op_type != "Mul" or half.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        index = 1 if half.input[0] == mul.output[0] else 0
        if (
            not g.is_constant_scalar(half.input[index])
            or g.get_constant_scalar(half.input[index]) != 0.5
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(
            self, [add, div, node, add_1, mul, half], self.apply, insert_at=node
        )

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        add_node: NodeProto,
        div_node: NodeProto,
        erf_node: NodeProto,
        add_1_node: NodeProto,
        mul_node: NodeProto,
        half_node: NodeProto,
    ) -> List[NodeProto]:
        return [
            g.make_node(
                "BiasGelu",
                add_node.input,
                half_node.output,
                domain="com.microsoft",
                doc_string=erf_node.doc_string,
                name=f"{self.__class__.__name__}--{erf_node.name}",
            )
        ]


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
        super().__init__(verbose, priority, min_opset=min_opset)
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
