import inspect
from typing import Any, Dict, List, Optional
import numpy as np
import onnx.numpy_helper as onh
from onnx import AttributeProto, NodeProto
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
    with one TreeEnsembleRegressor. All trees must have only one target
    (it can be extended to multiple) and is assigned a distinct
    dimension. The aggregation must be SUM.
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
        if axis is None or axis.i not in (0, 1):
            return self.none(node, inspect.currentframe().f_lineno)

        trees = []
        post_transform = None
        inputs = None
        base_values_none = None

        for treeo in concat_node.input:
            t = g.node_before(treeo)
            if t.op_type != "TreeEnsembleRegressor" or t.domain != "ai.onnx.ml":
                return self.none(node, inspect.currentframe().f_lineno)

            if inputs is None:
                inputs = list(t.input)
            elif inputs != list(t.input):
                # not the same input
                return self.none(node, inspect.currentframe().f_lineno)

            n_targets = g.get_attribute(t, "n_targets", exc=False)
            if n_targets is None or n_targets.i != 1:
                # It could be implemented in that case as well.
                return self.none(node, inspect.currentframe().f_lineno)

            # only SUM is allowed
            agg = g.get_attribute(t, "aggregate_function", exc=False)
            if agg is not None and agg.s != b"SUM":
                return self.none(node, inspect.currentframe().f_lineno)

            # one unique post_transform is allowed
            post = g.get_attribute(t, "post_transform", exc=False)
            if post is None:
                if post_transform is not None and post_transform != post:
                    return self.none(node, inspect.currentframe().f_lineno)
                post_transform = b"NONE"
            elif post_transform is None:
                post_transform = post.s
            elif post_transform != post.s:
                return self.none(node, inspect.currentframe().f_lineno)

            # specific rule for base_values: all none or all filled
            bb = (
                g.get_attribute(t, "base_values", exc=False) is not None
                or g.get_attribute(t, "base_values_as_tensor", exc=False) is not None
            )
            if base_values_none is None:
                base_values_none = bb
            elif bb != base_values_none:
                return self.none(node, inspect.currentframe().f_lineno)

            trees.append(t)

        return MatchResult(self, [concat_node, *trees], self.apply, insert_at=concat_node)

    @classmethod
    def get_attribute_value(
        cls, g: "GraphBuilder", node: NodeProto, name: str, exc: bool = True  # noqa: F821
    ) -> Any:
        att = g.get_attribute(node, name, exc=exc)
        if not exc and att is None:
            return None
        if att.type == AttributeProto.INTS:
            return att.ints
        if att.type == AttributeProto.FLOATS:
            return att.floats
        if att.type == AttributeProto.STRING:
            return att.s.decode("ascii")
        if att.type == AttributeProto.STRINGS:
            return [t.decode("ascii") for t in att.strings]
        if att.type == AttributeProto.TENSOR:
            return onh.to_array(att.t)
        raise AssertionError(f"Unexpected attribute type {att.type} in {att}")

    @classmethod
    def _first_tree_id(
        cls, g: "GraphBuilder", trees: List[NodeProto]  # noqa: F821
    ) -> Dict[int, int]:
        res = {}
        current = 0
        for i, t in enumerate(trees):
            res[i] = current
            nodes_treeids = cls.get_attribute_value(g, t, "nodes_treeids")
            target_treeids = cls.get_attribute_value(g, t, "target_treeids")
            max_id = max(max(nodes_treeids), max(target_treeids)) + 1
            current += max_id
        return res

    @classmethod
    def _merge(
        cls,
        g: "GraphBuilder",  # noqa: F821
        trees: List[NodeProto],
        name: str,
        as_tensor: Optional[str] = None,
        increment=False,
        unique=False,
        first_tree_id: Optional[Dict[int, int]] = None,
    ) -> Any:
        if as_tensor is None:
            if unique:
                return cls.get_attribute_value(g, trees[0], name)
            collected = [cls.get_attribute_value(g, t, name) for t in trees]
            if not increment:
                assert (
                    first_tree_id is None
                ), "increment is False but first_tree_id is not None"
                merged = []
                for c in collected:
                    merged.extend(c)
                return merged

            # all attribute value are necessarily integers
            # nodes_treeids, target_treeids, target_ids
            assert first_tree_id is not None, "increment is True but first_tree_id is None"
            merged = []
            for i, value in enumerate(collected):
                first = first_tree_id[i]
                for v in value:
                    merged.append(v + first)
            return merged

        assert not increment, "as_tensor is true, increment is true, incompatible"
        assert not unique, "as_tensor is true, unique is true, incompatible"
        assert first_tree_id is None, "increment is False but first_tree_id is not None"

        collected = []
        for tree in trees:
            val = g.get_attribute(tree, name, exc=False)
            if val is not None:
                collected.append(np.array(val.floats, dtype=np.float32))
                continue

            collected.append(cls.get_attribute_value(g, tree, as_tensor, exc=False))

        if any(c is None for c in collected):
            return None
        merged = np.hstack(collected)
        return onh.from_array(merged, name=as_tensor)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        concat_node: NodeProto,
        *trees: NodeProto,
    ) -> List[NodeProto]:

        first_tree_id = self._first_tree_id(g, trees)
        axis = g.get_attribute(concat_node, "axis").i

        new_atts = dict(
            aggregate_function="SUM",
            base_values_as_tensor=self._merge(
                g, trees, "base_values", "base_values_as_tensor"
            ),
            n_targets=len(trees),
            nodes_falsenodeids=self._merge(g, trees, "nodes_falsenodeids"),
            nodes_featureids=self._merge(g, trees, "nodes_featureids"),
            nodes_hitrates_as_tensor=self._merge(
                g, trees, "nodes_hitrates", "nodes_hitrates_as_tensor"
            ),
            nodes_missing_value_tracks_true=self._merge(
                g, trees, "nodes_missing_value_tracks_true"
            ),
            nodes_modes=self._merge(g, trees, "nodes_modes"),
            nodes_nodeids=self._merge(g, trees, "nodes_nodeids"),
            nodes_treeids=self._merge(
                g, trees, "nodes_treeids", increment=True, first_tree_id=first_tree_id
            ),
            nodes_truenodeids=self._merge(g, trees, "nodes_truenodeids"),
            nodes_values=self._merge(g, trees, "nodes_values", "nodes_values_as_tensor"),
            post_transform=self._merge(g, trees, "post_transform", unique=True),
            target_ids=self._merge(
                g,
                trees,
                "target_ids",
                increment=True,
                first_tree_id=dict(enumerate(range(len(trees)))),
            ),
            target_nodeids=self._merge(g, trees, "target_nodeids"),
            target_treeids=self._merge(
                g, trees, "target_treeids", increment=True, first_tree_id=first_tree_id
            ),
            target_weights=self._merge(
                g, trees, "target_weights", "target_weights_as_tensor"
            ),
        )

        outputs = (
            concat_node.output
            if axis == 1
            else [g.unique_name(f"{self.__class__.__name__}_{concat_node.output[0]}")]
        )

        new_tree = g.make_node(
            trees[0].op_type,
            trees[0].input,
            outputs,
            name=f"{self.__class__.__name__}--{trees[0].name}",
            domain=trees[0].domain,
            **new_atts,
        )
        if axis == 1:
            return [new_tree]

        transpose_output = g.unique_name(
            f"{self.__class__.__name__}_{concat_node.output[0]}T"
        )
        transpose = g.make_node(
            "Transpose",
            outputs,
            [transpose_output],
            perm=[1, 0],
            name=f"{self.__class__.__name__}--{trees[0].name}",
        )
        new_shape = g.make_initializer("", np.array([-1, 1], dtype=np.int64))
        reshape = g.make_node(
            "Reshape",
            [transpose_output, new_shape],
            concat_node.output,
            name=f"{self.__class__.__name__}--{trees[0].name}",
        )
        return [new_tree, transpose, reshape]
