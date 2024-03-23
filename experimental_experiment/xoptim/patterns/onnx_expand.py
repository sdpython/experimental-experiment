import inspect
from typing import List, Optional
from onnx import NodeProto
from ...xbuilder._onnx_helper import element_wise_binary_op_types, unary_like_op_types
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

        def apply(g: "GraphBuilder", node: NodeProto) -> List[NodeProto]:  # noqa: F821
            new_node = g.make_node(
                "Identity",
                node.input,
                node.output,
                name=f"{self.__class__.__name__}--{node.name}",
                doc_string=node.doc_string,
            )
            return [new_node]

        return MatchResult(self, [node], apply, insert_at=node)


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

    @classmethod
    def apply(
        cls, g: "GraphBuilder", node: NodeProto, next_node: NodeProto  # noqa: F821
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
                name=f"{cls.__name__}--{node.name}",
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

        if g.is_used_more_than_once(node.output[0]):
            # More than one output so it probably must be done.
            return self.none(node, inspect.currentframe().f_lineno)

        next_nodes = g.next_nodes(node.output[0])
        assert (
            len(next_nodes) == 1
        ), "The previous test should have cleared out this case."
        next_node = next_nodes[0]

        if next_node.op_type not in self._op_types or next_node.domain != "":
            # Not an unary wise operator.
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [node, next_node], self.apply, insert_at=node)

    @classmethod
    def apply(
        cls, g: "GraphBuilder", node: NodeProto, next_node: NodeProto  # noqa: F821
    ) -> List[NodeProto]:
        # We need to create a new name for the intermediate results.
        # The optimizer cannot reuse an existing name if the new result
        # has a different shape.
        new_name = g.unique_name(f"{cls.__class__.__name__}_{node.input[0]}")
        unary = g.make_node(
            next_node.op_type,
            [node.input[0], *next_node.input[1:]],
            [new_name],
            name=f"{cls.__name__}--{node.name}",
            doc_string=next_node.doc_string,
        )
        unary.attribute.extend(next_node.attribute)
        expand = g.make_node(
            node.op_type,  # Expand
            [new_name, node.input[1]],
            [next_node.output[0]],
            name=f"{cls.__name__}--{node.name}",
            doc_string=node.doc_string,
        )
        return [unary, expand]
