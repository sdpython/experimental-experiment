import inspect
from typing import List, Optional
import numpy as np
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization


class SimpleRotaryPattern(PatternOptimization):
    """
    Replaces ConstantOfShape + ScatterND with ScatterNDOfShape (com.domain).
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if not g.has_processor("CUDA"):
            return self.none()
        if node.op_type != "Split" or node.domain != "":
            return self.none()

        if not g.has_rank(node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)

        axis = g.get_attribute(node, "axis", exc=False)
        if axis is None:
            axis = 0
        else:
            axis = axis.i
        rk = g.get_rank(node.input[0])
        if axis < 0:
            axis += rk
        if axis != rk - 1:
            return self.none(node, inspect.currentframe().f_lineno)

        cst = g.get_computed_constant(node.input[1])
        if cst.dtype != np.int64 or cst.shape != (2,) or cst[0] != cst[1]:
            return self.none(node, inspect.currentframe().f_lineno)

        if g.is_used_more_than_once(node.output[0]) or g.is_used_more_than_once(
            node.output[1]
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        left_node = g.next_node(node.output[0])
        right_node = g.next_node(node.output[1])
        if left_node.op_type != "Neg" and right_node.op_type != "Neg":
            return self.none(node, inspect.currentframe().f_lineno)
        if left_node.op_type != "Concat" and right_node.op_type != "Concat":
            return self.none(node, inspect.currentframe().f_lineno)

        if left_node.op_type == "Neg":
            inputs = [node.output[1], left_node.output[0]]
            neg_node = left_node
            concat_node = right_node
        else:
            inputs = [right_node.output[0], node.output[0]]
            neg_node = right_node
            concat_node = left_node
        if inputs != list(concat_node.input):
            return self.none(node, inspect.currentframe().f_lineno)

        if g.is_used_more_than_once(neg_node.output[0]):
            return self.none(node, inspect.currentframe().f_lineno)

        axis_ = g.get_attribute(concat_node, "axis", exc=False).i
        if axis_ < 0:
            axis_ += rk
        if axis != axis_:
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(
            self, [node, neg_node, concat_node], self.apply, insert_at=concat_node
        )

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        split_node: NodeProto,
        neg_node: NodeProto,
        concat_node: NodeProto,
    ) -> List[NodeProto]:
        side = "right" if neg_node.input[0] == split_node.output[1] else "left"
        new_node = g.make_node(
            "Rotary",
            split_node.input,
            concat_node.output,
            side=side,
            name=f"{self.__class__.__name__}--{neg_node.name}",
            domain="onnx_extended.ortops.optim.cuda",
        )
        return [new_node]
