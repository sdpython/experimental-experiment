import inspect
from typing import List, Optional
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization


class SoftmaxGradPattern(PatternOptimization):
    """
    Replaces the sequence Mul, ReduceSum, Mul, Sub by SoftmaxGrad
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "ReduceSum" or node.domain != "":
            return self.none()

        axis = g.get_constant_or_attribute(node, "axes", input_index=1, cvt=tuple)
        assert isinstance(axis, tuple), f"unexpected type {type(axis)} for axis"
        if len(axis) != 1:
            return self.none(node, inspect.currentframe().f_lineno)

        mul_node = g.node_before(node.input[0])
        if mul_node.op_type != "Mul" or mul_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        next_mul_node = g.next_node(node.output[0])
        if next_mul_node.op_type != "Mul" or next_mul_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        sub_node = g.next_node(next_mul_node.output[0])
        if sub_node.op_type != "Sub" or sub_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        next_nodes = g.next_nodes(mul_node.output[0])
        if len(next_nodes) != 2:
            return self.none(node, inspect.currentframe().f_lineno)
        if {id(next_nodes[0]), id(next_nodes[1])} != {id(sub_node), id(node)}:
            return self.none(node, inspect.currentframe().f_lineno)

        if g.is_used_more_than_once(
            next_mul_node.output[0]
        ) or g.is_used_more_than_once(node.output[0]):
            return self.none(node, inspect.currentframe().f_lineno)

        nodes = [mul_node, node, next_mul_node, sub_node]
        return MatchResult(self, nodes, self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        mul_node: NodeProto,
        reduce_node: NodeProto,
        next_mul_node: NodeProto,
        sub_node: NodeProto,
    ) -> List[NodeProto]:

        axis = g.get_constant_or_attribute(
            reduce_node, "axes", input_index=1, cvt=tuple
        )

        grad = g.make_node(
            "SoftmaxGrad",
            mul_node.input,
            sub_node.output,
            axis=int(axis[0]),
            name=f"{self.__class__.__name__}",
            doc_string=sub_node.doc_string,
            domain="com.microsoft",
        )

        return [grad]
