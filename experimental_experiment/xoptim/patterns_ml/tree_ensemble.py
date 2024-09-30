import inspect
from typing import List, Optional
import onnx.numpy_helper as onh
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization


class TreeEnsembleRegressorMulPattern(PatternOptimization):
    """
    Replaces TreeEnsembleRegressor + Mul(., scalar) with TreeEnsembleRegressor.
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "TreeEnsembleRegressor" or node.domain != "ai.onnx.ml":
            return self.none()

        next_nodes = g.next_nodes(node.output[0])
        if len(next_nodes) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        if next_nodes[0].op_type != "Mul" or next_nodes[0].domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant_scalar(next_nodes[0].input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, [node, next_nodes[0]], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        tree_node: NodeProto,
        mul_node: NodeProto,
    ) -> List[NodeProto]:
        cst = g.get_constant_scalar(mul_node.input[1])
        names = {"target_weights", "target_weights_as_tensor"}
        weights = None
        atts = []
        for att in tree_node.attribute:
            if att.name in names:
                assert weights is None, f"Both {names} can be set at the same time."
                weights = att
            else:
                atts.append(att)
        if att.name == "target_weights":
            kwargs = {att.name: [float(f * cst) for f in att.floats]}
        else:
            value = onh.to_array(att.t)
            kwargs = {att.name: onh.from_array(value * cst, name=att.name)}

        new_tree = g.make_node(
            tree_node.op_type,
            tree_node.input,
            mul_node.output,
            name=f"{self.__class__.__name__}--{tree_node.name}",
            domain=tree_node.domain,
            **kwargs,
        )
        new_tree.attribute.extend(atts)
        return [new_tree]
