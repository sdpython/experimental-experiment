import inspect
from typing import List, Optional
from onnx import NodeProto
from ...xbuilder._onnx_helper import element_wise_binary_op_types, unary_like_op_types
from ...xbuilder._shape_helper import all_int
from ..patterns_api import MatchResult, PatternOptimization


class ExpandPattern(PatternOptimization):
    """
    Checks that a Expand is really needed.
    """

    def __init__(self, verbose: int = 0, priority: int = 0):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Expand" or node.domain != "":
            return self.none()
        if not g.has_shape(node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        shape = g.get_shape(node.input[0])
        if not all_int(shape):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant(node.input[1]):
            # It may be a symbolic shape.
            return self.none(node, inspect.currentframe().f_lineno)
        new_shape = tuple(g.get_computed_constant(node.input[1]))
        if shape != new_shape:
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [node], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node: NodeProto,
    ) -> List[NodeProto]:
        new_node = g.make_node(
            "Identity",
            node.input,
            node.output,
            name=f"{self.__class__.__name__}--{node.name}",
            doc_string=node.doc_string,
        )
        return [new_node]


class ExpandBroadcastPattern(PatternOptimization):
    """
    Checks that a Expand is really needed before an element wise operator.
    The objective is to save one allocation and let the next operator
    do the expansion by broadcasting one input.
    """

    _op_types = element_wise_binary_op_types()

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Expand" or node.domain != "":
            return self.none()
        if not g.has_shape(node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        shape = g.get_shape(node.input[0])
        if not all_int(shape):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant(node.input[1]):
            # It may be a symbolic shape.
            return self.none(node, inspect.currentframe().f_lineno)
        new_shape = tuple(g.get_computed_constant(node.input[1]))

        if g.is_used_more_than_once(node.output[0]):
            # More than one output, not handled right now.
            return self.none(node, inspect.currentframe().f_lineno)

        next_nodes = g.next_nodes(node.output[0])
        assert (
            len(next_nodes) == 1
        ), "The previous test should have cleared out this case."
        next_node = next_nodes[0]

        if next_node.op_type not in self._op_types or next_node.domain != "":
            # Not an element wise operator.
            return self.none(node, inspect.currentframe().f_lineno)

        if next_node.input[0] == node.output[0]:
            other = next_node.input[1]
        else:
            other = next_node.input[0]

        if not g.has_shape(other):
            return self.none(node, inspect.currentframe().f_lineno)

        other_shape = g.get_shape(other)
        if new_shape != other_shape:
            # Expand does not expand to the shape of the other element.
            return self.none(node, inspect.currentframe().f_lineno)
        if len(shape) != len(other_shape):
            # Different ranks.
            return self.none(node, inspect.currentframe().f_lineno)
        for a, b in zip(shape, other_shape):
            if not (a == b or a == 1 or b == 1):
                return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [node, next_node], self.apply, insert_at=next_node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node: NodeProto,
        next_node: NodeProto,
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


class ExpandSwapPattern(PatternOptimization):
    """
    Tries to move a node Expand forward in the graph.
    Expand + Exp can be changed into Exp + Expand.
    Then Exp applies on a tensor of a smaller or equal size.
    """

    _op_types = unary_like_op_types()
    _other_types = {"NegXplus1", "ReplaceZero", "Pow"}

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Expand" or node.domain != "":
            return self.none()
        if not g.has_shape(node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)

        assert g.is_used(node.output[0]), (
            f"The match should not even begin, {node.output[0]!r} "
            f"is not used among {node.output} and type={node.op_type!r}"
        )
        if g.is_used_more_than_once(node.output[0]):
            # More than one output so it probably must be done.
            return self.none(node, inspect.currentframe().f_lineno)

        next_nodes = g.next_nodes(node.output[0])
        assert (
            len(next_nodes) == 1
        ), "The previous test should have cleared out this case."
        next_node = next_nodes[0]

        if next_node.op_type not in self._other_types and (
            next_node.op_type not in self._op_types or next_node.domain != ""
        ):
            # Not an unary wise operator.
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [node, next_node], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node: NodeProto,
        next_node: NodeProto,
    ) -> List[NodeProto]:
        # We need to create a new name for the intermediate results.
        # The optimizer cannot reuse an existing name if the new result
        # has a different shape.
        new_name = g.unique_name(f"{self.__class__.__name__}_{node.input[0]}")
        unary = g.make_node(
            next_node.op_type,
            [node.input[0], *next_node.input[1:]],
            [new_name],
            name=f"{self.__class__.__name__}--{node.name}",
            domain=next_node.domain,
            doc_string=next_node.doc_string,
        )
        unary.attribute.extend(next_node.attribute)
        expand = g.make_node(
            node.op_type,  # Expand
            [new_name, node.input[1]],
            [next_node.output[0]],
            name=f"{self.__class__.__name__}--{node.name}",
            doc_string=node.doc_string,
        )
        return [unary, expand]
