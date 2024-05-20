import inspect
from typing import List, Optional
from onnx import NodeProto, TensorProto
from ..patterns_api import MatchResult, PatternOptimization


class ReplaceZeroPattern(PatternOptimization):
    """
    Replaces Where(bool(X), value, X) into ReplaceZero(X, by=by).
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if not g.has_processor("CUDA"):
            return self.none()
        if node.op_type != "Where" or node.domain != "":
            return self.none()

        if not g.is_constant(node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant_scalar(node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)

        cast_node = g.node_before(node.input[0])
        if g.is_used_more_than_once(node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        if cast_node.op_type != "Cast" or node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        to = g.get_attribute(cast_node, "to").i
        if to != TensorProto.BOOL:
            return self.none(node, inspect.currentframe().f_lineno)

        if node.input[2] != cast_node.input[0]:
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [cast_node, node], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        cast_node: NodeProto,
        where_node: NodeProto,
    ) -> List[NodeProto]:
        cst = g.get_constant_scalar(where_node.input[1])
        new_node = g.make_node(
            "ReplaceZero",
            cast_node.input,
            where_node.output,
            by=cst,
            equal=False,
            name=f"{self.__class__.__name__}--{where_node.name}",
            domain="onnx_extended.ortops.optim.cuda",
        )
        return [new_node]
