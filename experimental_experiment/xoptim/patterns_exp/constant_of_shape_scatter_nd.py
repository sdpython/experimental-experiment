import inspect
from typing import List, Optional
from onnx import NodeProto
from onnx.numpy_helper import to_array
from ..patterns_api import MatchResult, PatternOptimization


class ConstantOfShapeScatterNDPattern(PatternOptimization):
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

        if not g.has_type(node.input[2]):
            itype = g.try_infer_type(node.input[2])
            if itype == 0:
                return self.none(node, inspect.currentframe().f_lineno)
        else:
            itype = g.get_type(node.input[2])

        node_before = g.node_before(node.input[0])
        if node_before.op_type != "ConstantOfShape" or node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        att = g.get_attribute(node_before, "value", False)
        if att is not None:
            arr = to_array(att.t)
            if arr[0] != 0:
                return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [node_before, node], self.apply, insert_at=node)

    def apply(
        self, g: "GraphBuilder", node_before: NodeProto, node: NodeProto  # noqa: F821
    ) -> List[NodeProto]:
        new_node = g.make_node(
            "ScatterNDOfShape",
            [node_before.input[0], *node.input[1:]],
            node.output,
            name=f"{self.__class__.__name__}--{node.name}",
            domain="onnx_extended.ortops.optim.cuda",
        )
        return [new_node]
