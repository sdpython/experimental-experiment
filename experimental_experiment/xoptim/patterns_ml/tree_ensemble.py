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


class TreeEnsembleRegressorConcatPattern(PatternOptimization):
    """
    Replaces multiple TreeEnsembleRegressor + Concat(., axis=1)
    with one TreeEnsembleRegressor.
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
        concat_node = next_nodes[0]
        if concat_node.op_type != "Concat" or concat_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        axis = g.get_attribute(concat_node, "axis", exc=False)
        if axis is None or axis.i != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        trees = []
        for treeo in concat_node.input:
            t = g.node_before(treeo)
            if t.op_type != "TreeEnsembleRegressor" or t.domain != "ai.onnx.ml":
                return self.none(node, inspect.currentframe().f_lineno)
            n_targets = g.get_attribute(t, "n_targets", exc=False)
            if n_targets is None or n_targets.i != 1:
                # It could be implemented in that case as well.
                return self.none(node, inspect.currentframe().f_lineno)
            trees.append(t)

        return MatchResult(self, [concat_node, *trees], self.apply, insert_at=concat_node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        concat_node: NodeProto,
        *trees: NodeProto,
    ) -> List[NodeProto]:
        assert False, "Not implemented error"
