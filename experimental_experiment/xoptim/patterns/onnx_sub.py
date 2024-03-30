import inspect
from typing import List, Optional
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization


class Sub1MulPattern(PatternOptimization):
    """
    Replaces the sequence `(1 - X) x Y`  by `Y - X x Y` to avoid the creation
    of a constant in the graph. `x` means element wise multiplication.
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Mul" or node.domain != "":
            return self.none()

        node_left = g.node_before(node.input[0])
        node_right = g.node_before(node.input[1])
        op_left = None if node_left is None else node_left.op_type
        op_right = None if node_right is None else node_right.op_type

        if op_left != "Sub" and op_right != "Sub":
            return self.none(node, inspect.currentframe().f_lineno)

        if (op_left == "Sub" and g.is_used_more_than_once(node.input[0])) or (
            op_right == "Sub" and g.is_used_more_than_once(node.input[1])
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        cst_left, cst_right = None, None

        if op_left == "Sub" and g.is_constant(node_left.input[0]):
            cst_min, cst_max = g.get_computed_constant(
                node_left.input[0], ["min", "max"]
            )
            if cst_min == cst_max == 1:
                cst_left = cst_min

        if op_right == "Sub" and g.is_constant(node_right.input[0]):
            cst_min, cst_max = g.get_computed_constant(
                node_right.input[0], ["min", "max"]
            )
            if cst_min == cst_max == 1:
                cst_right = cst_min

        if cst_left is None and cst_right is None:
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

        cst_left = None
        if (
            node_left is not None
            and node_left.op_type == "Sub"
            and g.is_constant(node_left.input[0])
        ):
            cst = g.get_computed_constant(node_left.input[0])
            if cst.min() == cst.max() == 1:
                cst_left = cst

        if cst_left is not None:
            # rewrite `(1 - X) x Y` into `Y - X x Y`
            mul_node = g.make_node(
                "Mul",
                [node_left.input[1], node.input[1]],
                [g.unique_name(f"{cls.__class__.__name__}--{node.output[0]}")],
                name=f"{cls.__name__}--{node.name}",
            )
            sub_node = g.make_node(
                "Sub",
                [node.input[1], mul_node.output[0]],
                node.output,
                name=f"{cls.__name__}--{node.name}",
                doc_string=node.doc_string,
            )
            keep_node = node_right
        else:
            # rewrite `Y x (1 - X)` into `Y - (Y - X)`
            mul_node = g.make_node(
                "Mul",
                [node.input[0], node_right.input[1]],
                [g.unique_name(f"{cls.__class__.__name__}--{node.output[0]}")],
                name=f"{cls.__name__}--{node.name}",
            )
            sub_node = g.make_node(
                "Sub",
                [node.input[0], mul_node.output[0]],
                node.output,
                name=f"{cls.__name__}--{node.name}",
                doc_string=node.doc_string,
            )
            keep_node = node_left

        if keep_node is None:
            return [mul_node, sub_node]
        return [keep_node, mul_node, sub_node]
