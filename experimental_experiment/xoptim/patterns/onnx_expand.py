from typing import List, Optional
from onnx import NodeProto
from ...xbuilder.shape_helper import all_int
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


class ExpandBroadcastPattern(PatternOptimization):
    """
    Checks that a Expand is really needed before an element wise operator.
    The objective is to save one allocation and let the next operator
    do the expansion by broadcasting one input.
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

        if g.is_used_more_than_once(node.output[0]):
            # More than one output, not handled right now.
            return None

        next_nodes = g.next_nodes(node.output[0])
        assert (
            len(next_nodes) == 1
        ), "The previous test should have cleared out this case."
        next_node = next_nodes[0]

        if (
            next_node.op_type
            not in {
                "Add",
                "Div",
                "Mul",
                "Sub",
                "And",
                "Or",
                "Mod",
                "Equal",
                "Greater",
                "GreaterOrEqual",
                "Less",
                "LessOrEqual",
                "Xor",
            }
            or next_node.domain != ""
        ):
            # Not an element wise operator.
            return None

        if next_node.input[0] == node.output[0]:
            other = next_node.input[1]
        else:
            other = next_node.input[0]

        if not g.has_shape(other):
            return None

        other_shape = g.get_shape(other)
        if new_shape != other_shape:
            # Expand does not expand to the shape of the other element.
            return None
        if len(shape) != len(other_shape):
            # Different ranks.
            return None
        for a, b in zip(shape, other_shape):
            if not (a == b or a == 1 or b == 1):
                return None

        def apply(
            g: "GraphBuilder", node: NodeProto, next_node: NodeProto  # noqa: F821
        ) -> List[NodeProto]:
            if next_node.input[0] == node.output[0]:
                inputs = [node.input[0], next_node.input[1]]
            else:
                inputs = [next_node.input[0], node.input[0]]
            return [
                g.make_node(
                    next_node.op_type,
                    inputs,
                    next_node.output,
                    name=f"{self.__class__.__name__}--{node.name}",
                    doc_string=next_node.doc_string,
                )
            ]

        return MatchResult(self, [node, next_node], apply, insert_at=next_node)
