import inspect
from typing import List, Optional
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization


class _common:
    @classmethod
    def _same_shape(
        cls, g: "GraphBuilderPatternOptimization", name1: str, name2: str  # noqa: F821
    ) -> bool:
        if not g.has_shape(name1) or not g.has_shape(name2):
            return False
        return g.get_shape(name1) == g.get_shape(name2)


class AddAddMulMulPattern(PatternOptimization, _common):
    """
    Replaces Add + Add by AddAdd or Mul + Mul by MulMul
    if they operate on the same shape.
    """

    def __init__(self, verbose: int = 0, priority: int = 3):
        super(AddAddMulMulPattern, self).__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if not g.has_processor("CUDA"):
            return self.none()
        if node.op_type not in {"Add", "Mul"} or node.domain != "":
            return self.none()

        if not self._same_shape(g, *node.input):
            return self.none(node, inspect.currentframe().f_lineno)

        node_left = g.node_before(node.input[0])
        if (
            node_left is not None
            and not g.is_used_more_than_once(node.input[0])
            and node_left.op_type == node.op_type
            and self._same_shape(g, *node_left.input)
        ):
            return MatchResult(
                self, [node_left, None, node], self.apply, insert_at=node
            )

        node_right = g.node_before(node.input[1])
        if (
            node_right is not None
            and not g.is_used_more_than_once(node.input[1])
            and node_right.op_type == node.op_type
            and self._same_shape(g, *node_right.input)
        ):
            return MatchResult(
                self, [None, node_right, node], self.apply, insert_at=node
            )

        return self.none(node, inspect.currentframe().f_lineno)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node_left: NodeProto,
        node_right: NodeProto,
        node: NodeProto,
    ) -> List[NodeProto]:
        assert (
            node_left is not None or node_right is not None
        ), "node_left and node_right cannot be both None"
        if node_left is None:
            # node_right
            new_node = g.make_node(
                node.op_type * 2,
                [node.input[0], *node_right.input],
                node.output,
                domain="onnx_extended.ortops.optim.cuda",
                name=f"{self.__class__.__name__}--{node.name}",
            )
        else:
            new_node = g.make_node(
                node.op_type * 2,
                [
                    *node_left.input,
                    node.input[1],
                ],
                node.output,
                domain="onnx_extended.ortops.optim.cuda",
                name=f"{self.__class__.__name__}--{node.name}",
            )
        return [new_node]


class AddMulPattern(PatternOptimization, _common):
    """
    Replaces Add + Mul by AddMul or Mul + Add by MulAdd
    if they operate on the same shape.
    """

    def __init__(self, verbose: int = 0, priority: int = 3):
        super(AddMulPattern, self).__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if not g.has_processor("CUDA"):
            return self.none()

        if node.op_type not in {"Add", "Mul"} or node.domain != "":
            return self.none()

        if not self._same_shape(g, *node.input):
            return self.none(node, inspect.currentframe().f_lineno)

        node_left = g.node_before(node.input[0])
        if (
            node_left is not None
            and not g.is_used_more_than_once(node.input[0])
            and node_left.op_type in {"Add", "Mul"}
            and node_left.op_type != node.op_type
            and self._same_shape(g, *node_left.input)
        ):
            return MatchResult(
                self, [node_left, None, node], self.apply, insert_at=node
            )

        node_right = g.node_before(node.input[1])
        if (
            node_right is not None
            and not g.is_used_more_than_once(node.input[1])
            and node_right.op_type in {"Add", "Mul"}
            and node_right.op_type != node.op_type
            and self._same_shape(g, *node_right.input)
        ):
            return MatchResult(
                self, [None, node_right, node], self.apply, insert_at=node
            )

        return self.none(node, inspect.currentframe().f_lineno)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node_left: NodeProto,
        node_right: NodeProto,
        node: NodeProto,
    ) -> List[NodeProto]:
        assert (
            node_left is not None or node_right is not None
        ), "node_left and node_right cannot be both None"
        if node_left is None:
            # node_right
            new_node = g.make_node(
                f"{node_right.op_type}{node.op_type}",
                [node.input[0], *node_right.input],
                node.output,
                domain="onnx_extended.ortops.optim.cuda",
                name=f"{self.__class__.__name__}--{node.name}",
            )
        else:
            new_node = g.make_node(
                f"{node_left.op_type}{node.op_type}",
                [
                    *node_left.input,
                    node.input[1],
                ],
                node.output,
                domain="onnx_extended.ortops.optim.cuda",
                name=f"{self.__class__.__name__}--{node.name}",
            )
        return [new_node]


class MulSigmoidPattern(PatternOptimization):
    """
    Replaces Mul + Sigmoid by MulSigmoid
    if they operate on the same input.
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if not g.has_processor("CUDA"):
            return self.none()
        if node.op_type not in {"Sigmoid"} or node.domain != "":
            return self.none()

        if g.is_used_more_than_once(node.output[0]):
            return self.none(node, inspect.currentframe().f_lineno)

        next_node = g.next_node(node.output[0])
        if next_node.op_type != "Mul" or node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        index = list(next_node.input).index(node.output[0])
        other_index = 1 - index
        if next_node.input[other_index] != node.input[0]:
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [node, next_node], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node_sigmoid: NodeProto,
        node_mul: NodeProto,
    ) -> List[NodeProto]:
        new_node = g.make_node(
            "MulSigmoid",
            node_sigmoid.input,
            node_mul.output,
            domain="onnx_extended.ortops.optim.cuda",
            name=f"{self.__class__.__name__}--{node_sigmoid.name}",
        )
        return [new_node]


class NegXplus1Pattern(PatternOptimization):
    """
    Replaces 1 - X by NegXplus1
    if they operate on the same input.
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if not g.has_processor("CUDA"):
            return self.none()
        if node.op_type not in {"Sub"} or node.domain != "":
            return self.none()

        if not g.is_constant(node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant_scalar(node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        cst = g.get_constant_scalar(node.input[0])
        if cst != 1:
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [node], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node: NodeProto,
    ) -> List[NodeProto]:
        new_node = g.make_node(
            "NegXplus1",
            node.input[1:],
            node.output,
            domain="onnx_extended.ortops.optim.cuda",
            name=f"{self.__class__.__name__}--{node.name}",
        )
        return [new_node]
