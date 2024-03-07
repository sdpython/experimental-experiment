import inspect
from typing import List, Optional
import numpy as np
from onnx import NodeProto
from .patterns_api import MatchResult, PatternOptimization


class MulMulMulScalarPattern(PatternOptimization):
    """
    Replaces the sequence {Div | Mul} and  {Div | Mul} + {Div | Mul} with {Div | Mul} Mul.
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type not in {"Div", "Mul"} or node.domain != "":
            return self.none()
        if g.is_used_more_than_once(node.input[0]) or g.is_used_more_than_once(
            node.input[1]
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        node_left = g.node_before(node.input[0])
        if (
            node_left is None
            or node_left.op_type not in {"Div", "Mul"}
            or node.domain != ""
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        node_right = g.node_before(node.input[1])
        if (
            node_right is None
            or node_right.op_type not in {"Div", "Mul"}
            or node.domain != ""
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        # checking for the constant (right)
        if not g.is_constant(node_left.input[1]) or not g.is_constant(
            node_right.input[1]
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        cst_left = g.get_computed_constant(node_left.input[1])
        cst_right = g.get_computed_constant(node_right.input[1])
        if cst_left.shape not in {tuple(), (1,)} or cst_right.shape not in {
            tuple(),
            (1,),
        }:
            return self.none(node, inspect.currentframe().f_lineno)

        nodes = [node, node_left, node_right]

        return MatchResult(self, nodes, self.apply)

    @classmethod
    def apply(
        cls,
        g: "GraphBuilder",  # noqa: F821
        node: NodeProto,
        node_left: NodeProto,
        node_right: NodeProto,
    ) -> List[NodeProto]:

        new_node = g.make_node(
            node.op_type,
            [node_left.input[0], node_right.input[0]],
            [g.unique_name(f"{cls.__class__.__name__}--{node.output[0]}")],
            name=f"{cls.__name__}--{node.name}",
        )
        cst_left = g.get_computed_constant(node_left.input[1])
        cst_right = g.get_computed_constant(node_right.input[1])
        if node_left.op_type == "Div":
            cst_left = np.reciprocal(cst_left)
        if node_right.op_type == "Div":
            cst_right = np.reciprocal(cst_right)

        if not isinstance(cst_left, np.ndarray):
            cst_left = np.array(cst_left)
        if not isinstance(cst_right, np.ndarray):
            cst_right = np.array(cst_right)
        assert (
            cst_left.dtype == cst_right.dtype
        ), f"Type mismatch left is {cst_left.dtype}, right is {cst_right.dtype}"
        new_value = cst_left * cst_right
        if not isinstance(new_value, np.ndarray):
            new_value = np.array(new_value)
        new_cst = g.make_initializer("", new_value)

        new_node2 = g.make_node(
            "Mul",
            [new_node.output[0], new_cst],
            node.output,
            name=f"{cls.__name__}--{node.name}-Cst",
        )

        return [new_node, new_node2]
