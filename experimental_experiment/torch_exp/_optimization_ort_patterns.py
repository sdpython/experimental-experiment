from typing import List, Optional
from onnx import NodeProto
from onnx.numpy_helper import to_array
from .optimization_patterns_api import MatchResult, PatternOptimization


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
            return None

        if not g.has_type(node.input[2]):
            itype = g.try_infer_type(node.input[2])
            if itype == 0:
                return None
        else:
            itype = g.get_type(node.input[2])

        node_before = g.node_before(node.input[0])
        if node_before.op_type != "ConstantOfShape" or node.domain != "":
            return None

        att = g.get_attribute(node_before, "value", False)
        if att is not None:
            arr = to_array(att.t)
            if arr[0] != 0:
                return None

        def apply(
            g: "GraphBuilder", node_before: NodeProto, node: NodeProto  # noqa: F821
        ) -> List[NodeProto]:
            new_node = g.make_node(
                "ScatterNDOfShape",
                [node_before.input[0], *node.input[1:]],
                node.output,
                name=f"{self.__class__.__name__}--{node.name}",
                domain="com.microsoft",
            )
            return [new_node]

        return MatchResult(self, [node_before, node], apply)
