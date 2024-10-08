import inspect
from typing import List, Optional
from onnx import NodeProto
from onnx.numpy_helper import to_array
from ..patterns_api import MatchResult, PatternOptimization


class GatherGradPattern(PatternOptimization):
    """
    Replaces ConstantOfShape + ScatterND with GatherGrad (com.domain).
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "ScatterND" or node.domain != "":
            return self.none()

        reduction = g.get_attribute(node, "reduction")
        if reduction is None or reduction.s != b"add":
            return self.none(node, inspect.currentframe().f_lineno)

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
        self,
        g: "GraphBuilder",  # noqa: F821
        node_before: NodeProto,
        node: NodeProto,
    ) -> List[NodeProto]:
        new_node = g.make_node(
            "GatherGrad",
            [node_before.input[0], *node.input[1:]],
            node.output,
            name=f"{self.__class__.__name__}--{node.name}",
            domain="com.microsoft",
        )
        for att in node.attribute:
            if att.name != "reduction":
                new_node.append(att)
        return [new_node]
