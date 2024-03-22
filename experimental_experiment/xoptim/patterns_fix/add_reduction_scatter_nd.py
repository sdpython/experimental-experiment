import inspect
from typing import List, Optional
from onnx import NodeProto
from onnx.numpy_helper import to_array
from ..patterns.patterns_api import MatchResult, PatternOptimization


class AddReductionScatterND(PatternOptimization):
    """
    Replaces ConstantOfShape + ScatterND with ScatterNDOfShape (com.domain).
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "ScatterND" or node.domain != "":
            return self.none()
        if g.get_attribute(node, "reduction", exc=False) is not None:
            return self.none(node, inspect.currentframe().f_lineno)
        if g.is_used_more_than_once(node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)

        before = g.node_before(node.input[0])
        if before.op_type != "ConstantOfShape" or before.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        att = g.get_attribute(before, "value", exc=False)
        if att is None:
            value = 0
        else:
            t = to_array(att.t)
            value = t[0] if t.shape == (1,) else t
        if value != 0:
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [node], self.apply, insert_at=node)

    @classmethod
    def apply(cls, g: "GraphBuilder", node: NodeProto) -> List[NodeProto]:  # noqa: F821
        new_node = g.make_node(
            "ScatterND",
            node.input,
            node.output,
            reduction="add",
            name=f"{cls.__name__}--{node.name}",
            doc_string=node.doc_string,
        )
        return [new_node]
