import inspect
from typing import List, Optional, Tuple, Union
import numpy as np
from onnx import NodeProto
from ...xbuilder._onnx_helper import (
    element_wise_binary_op_types,
    element_wise_op_cmp_types,
    unary_like_op_types,
)
from ...xbuilder._shape_helper import all_int, DYNAMIC_SHAPE
from ..patterns_api import MatchResult, PatternOptimization


class ExpandPattern(PatternOptimization):
    """Checks that a Expand is really needed."""

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
        value = g.get_computed_constant(node.input[1])
        if value is None:
            return self.none(node, inspect.currentframe().f_lineno)
        with g.builder.maybe_disable_fake_tensor_mode():
            new_shape = tuple(int(i) for i in value)
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

    _op_types = element_wise_binary_op_types() | element_wise_op_cmp_types()

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
        value = g.get_computed_constant(node.input[1])
        if value is None:
            return self.none(node, inspect.currentframe().f_lineno)
        with g.builder.maybe_disable_fake_tensor_mode():
            new_shape = tuple(int(i) for i in value)

        if g.is_used_more_than_once(node.output[0]):
            # More than one output, not handled right now.
            return self.none(node, inspect.currentframe().f_lineno)

        next_nodes = g.next_nodes(node.output[0])
        assert len(next_nodes) == 1, "The previous test should have cleared out this case."
        next_node = next_nodes[0]

        if next_node.op_type not in self._op_types or next_node.domain != "":
            # Not an element wise operator.
            return self.none(node, inspect.currentframe().f_lineno)

        other = next_node.input[1 if next_node.input[0] == node.output[0] else 0]

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


class ShapeBasedExpandBroadcastPattern(PatternOptimization):
    """
    Similar to
    :class:`experimental_experiment.xoptim.patterns.onnx_expand.BroadcastPattern`,
    but it allows dynamic shapes as well. It does not look into the second
    argument of Expand, it just infers than an expand is not needed for
    a binary operator following just after.
    """

    _op_types = element_wise_binary_op_types() | element_wise_op_cmp_types()

    @classmethod
    def _is_compatible_shapes_for_expand(
        cls, shape_left: DYNAMIC_SHAPE, shape_right: DYNAMIC_SHAPE, output_shape: DYNAMIC_SHAPE
    ) -> bool:
        """
        Checks that the binary operations of the two input shapes returns the output_shape.
        Then no Expand node is needed.
        """
        # Align shapes
        if len(shape_left) < len(shape_right):
            shape_left = (1,) * (len(shape_right) - len(shape_left)) + shape_left
        elif len(shape_left) > len(shape_right):
            shape_right = (1,) * (len(shape_left) - len(shape_right)) + shape_right

        for left, right, out in zip(shape_left, shape_right, output_shape):
            if isinstance(left, int):
                if isinstance(right, int):
                    # static right
                    if left == 1:
                        if right != out:
                            return False
                    elif right == 1:
                        if left != out:
                            return False
                    else:
                        if left != right or left != out or right != out:
                            return False
                else:
                    # dynamic right
                    if left == 1:
                        if right != out:
                            return False
                    else:
                        if left != right or left != out or right != out:
                            return False
            else:
                # dynamic left
                if isinstance(right, int):
                    # static right
                    if right == 1:
                        if left != out:
                            return False
                    else:
                        if left != right or left != out or right != out:
                            return False
                else:
                    # dynamic right
                    if left != right or left != out or right != out:
                        return False
        return True

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type not in self._op_types or node.domain != "":
            return self.none()
        if (
            not g.has_shape(node.output[0])
            or not g.has_shape(node.input[0])
            or not g.has_shape(node.input[1])
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        node_left = g.node_before(node.input[0])
        node_right = g.node_before(node.input[1])
        before = [
            None if n is None or n.op_type != "Expand" else n for n in [node_left, node_right]
        ]
        if before == [None, None]:
            return self.none(node, inspect.currentframe().f_lineno)

        # At least one expand.
        node_left, node_right = before
        shape_left = g.get_shape_renamed(
            node.input[0] if node_left is None else node_left.input[0]
        )
        shape_right = g.get_shape_renamed(
            node.input[1] if node_right is None else node_right.input[0]
        )
        if self._is_compatible_shapes_for_expand(
            shape_left, shape_right, g.get_shape_renamed(node.output[0])
        ):
            return MatchResult(self, [node_left, node_right, node], self.apply)
        return self.none(node, inspect.currentframe().f_lineno)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        expand_left: NodeProto,
        expand_right: NodeProto,
        binary_node: NodeProto,
    ) -> List[NodeProto]:
        nodes = []
        if expand_left is not None and g.is_used_more_than_once(expand_left.output[0]):
            nodes.append(expand_left)
        if expand_right is not None and g.is_used_more_than_once(expand_right.output[0]):
            nodes.append(expand_right)
        assert (
            not binary_node.attribute
        ), f"Binary operator should not have any attribute, binary_node={binary_node}"
        return [
            *nodes,
            g.make_node(
                binary_node.op_type,
                [
                    binary_node.input[0] if expand_left is None else expand_left.input[0],
                    binary_node.input[1] if expand_right is None else expand_right.input[0],
                ],
                binary_node.output,
                name=f"{self.__class__.__name__}--{binary_node.name}",
                doc_string=binary_node.doc_string,
            ),
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
        assert len(next_nodes) == 1, "The previous test should have cleared out this case."
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


class ShapeBasedStaticExpandPattern(PatternOptimization):
    """
    Compares input and output shapes to tell if the expand
    can uses a constant as a second input.
    """

    def __init__(self, verbose: int = 0, priority: int = 0):
        super().__init__(verbose, priority)

    @classmethod
    def _find_expand_shape(
        cls, sh1: Tuple[Union[str, int], ...], sh2: Tuple[Union[str, int], ...]
    ) -> Tuple[int, ...]:
        expand_shape = []
        for s1, s2 in zip(sh1, sh2):
            if s1 == s2:
                expand_shape.append(1)
                continue
            if not isinstance(s1, int) or not isinstance(s2, int):
                return None
            expand_shape.append(s2)
        return tuple(expand_shape)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Expand" or node.domain != "":
            return self.none()
        if g.is_constant(node.input[1]):
            # already done
            return self.none(node, inspect.currentframe().f_lineno)

        if not g.has_shape(node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.has_shape(node.output[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        sh1 = g.get_shape_renamed(node.input[0])
        sh2 = g.get_shape_renamed(node.output[0])
        assert len(sh1) == len(sh2), (
            f"Ranks disagree: shape({node.input[0]})={sh1} and shape({node.output[0]})={sh2}, "
            f"not renamed shapes {self.get_shape(node.input[0])} and "
            f"{self.get_shape(node.output[0])}"
        )
        expand_shape = self._find_expand_shape(sh1, sh2)
        if expand_shape is None:
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, [node], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        reshape: NodeProto,
    ) -> List[NodeProto]:
        expand_shape = self._find_expand_shape(
            g.get_shape_renamed(reshape.input[0]), g.get_shape_renamed(reshape.output[0])
        )
        new_shape = g.make_initializer(
            "",
            np.array(expand_shape, dtype=np.int64),
            source="StaticExpandPattern.m1",
        )
        return [
            g.make_node(
                "Expand",
                [reshape.input[0], new_shape],
                reshape.output,
                name=f"{self.__class__.__name__}--{reshape.name}",
                doc_string=reshape.doc_string,
            )
        ]
