import inspect
from typing import List, Optional, Tuple
import numpy as np
from onnx import NodeProto, TensorProto
from ...helpers import make_idn
from ...xbuilder import FunctionOptions, GraphBuilder
from ...xbuilder._shape_helper import STATIC_SHAPE
from ..patterns_api import MatchResult, PatternOptimization


class RotaryConcatPartPattern(PatternOptimization):
    """
    Optimizes the following pattern

    .. plot::

        import numpy as np
        from onnx import TensorProto
        from onnx_array_api.light_api import start
        from onnx_array_api.plotting.graphviz_helper import plot_dot

        def mk(shape):
            return np.array(shape, dtype=np.int64)

        model = (
            start(opset=18, ir_version=9)
            .cst(mk([2, 2, 1024, 256]), "shape")
            .cst(mk([0]), "c0")
            .cst(mk([256]), "c256")
            .cst(mk([512]), "c512")
            .cst(mk([3]), "c3")
            .vin("X", TensorProto.FLOAT, ("a", "b", "c", "d"))
            .bring("shape")
            .ConstantOfShape()
            .rename("C1")
            .bring("shape")
            .ConstantOfShape()
            .rename("C2")
            .bring("X", "c256", "c512", "c3")
            .Slice()
            .rename("S1")
            .bring("C1", "S1")
            .Concat(axis=3)
            .rename("P1")
            .bring("X", "c0", "c256", "c3")
            .Slice()
            .Neg()
            .rename("S2")
            .bring("C1", "S2")
            .Concat(axis=3)
            .rename("P2")
            .bring("P1", "P2")
            .Add()
            .rename("Y")
            .vout(TensorProto.FLOAT, ("a", "b", "c", "d"))
            .to_onnx()
        )
        ax = plot_dot(model)
        ax.set_title("Dummy graph")
        plt.show()
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Add" or node.domain != "":
            return self.none()
        left, right = g.node_before(node.input[0]), g.node_before(node.input[1])
        if None in (left, right):
            return self.none()
        if "Concat" in (left.op_type, right.op_type):
            return self.match_concat(g, node, matched)
        if "Transpose" in (left.op_type, right.op_type):
            return self.match_transpose(g, node, matched)
        return self.none(node, inspect.currentframe().f_lineno)

    def match_concat(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Add" or node.domain != "":
            return self.none()
        if g.is_used_more_than_once(node.input[0]) or g.is_used_more_than_once(node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)

        concat_left, concat_right = (
            g.node_before(node.input[0]),
            g.node_before(node.input[1]),
        )
        if concat_left is None or concat_right is None:
            return self.none(node, inspect.currentframe().f_lineno)
        if len(concat_left.input) != 2 or len(concat_right.input) != 2:
            return self.none(node, inspect.currentframe().f_lineno)
        axis1 = g.get_axis(concat_left, 0)
        axis2 = g.get_axis(concat_right, 0)
        if axis1 != axis2:
            return self.none(node, inspect.currentframe().f_lineno)

        # checking every result has shapes
        all_inputs = list(concat_left.input) + list(concat_right.input)
        if any(not g.has_shape(a) for a in all_inputs):
            return self.none(node, inspect.currentframe().f_lineno)

        concat_left_before = [g.node_before(i) for i in concat_left.input]
        concat_right_before = [g.node_before(i) for i in concat_right.input]
        if None in concat_left_before or None in concat_right_before:
            return self.none(node, inspect.currentframe().f_lineno)

        type_left = [n.op_type for n in concat_left_before]
        type_right = [n.op_type for n in concat_right_before]
        if "ConstantOfShape" not in type_left or "ConstantOfShape" not in type_right:
            return self.none(node, inspect.currentframe().f_lineno)
        if type_left.index("ConstantOfShape") == type_right.index("ConstantOfShape"):
            return self.none(node, inspect.currentframe().f_lineno)

        cst_left = next(n for n in concat_left_before if n.op_type == "ConstantOfShape")
        cst_right = next(n for n in concat_right_before if n.op_type == "ConstantOfShape")

        tl = [n for n in concat_right_before if n.op_type == "Neg"]
        tr = [n for n in concat_left_before if n.op_type == "Neg"]
        if tl:
            neg_left = None
            neg_right = tl[0]

            slice_left = [n for n in concat_left_before if n.op_type == "Slice"]
            split_left = [n for n in concat_left_before if n.op_type == "Split"]
            if (len(slice_left) == 0 and len(split_left) == 0) or (
                len(slice_left) > 0 and len(split_left) > 0
            ):
                return self.none(node, inspect.currentframe().f_lineno)
            slice_left = slice_left[0] if slice_left else None
            split_left = split_left[0] if split_left else None

            right_ = g.node_before(neg_right.input[0])
            slice_right = None if right_.op_type == "Split" else right_
            split_right = right_ if right_.op_type == "Split" else None
        elif tr:
            neg_left = next(n for n in concat_left_before if n.op_type == "Neg")
            neg_right = None

            left_ = g.node_before(neg_left.input[0])
            slice_left = None if left_.op_type == "Split" else left_
            split_left = left_ if left_.op_type == "Split" else None

            slice_right = [n for n in concat_right_before if n.op_type == "Slice"]
            split_right = [n for n in concat_right_before if n.op_type == "Split"]
            if (len(slice_right) == 0 and len(split_right) == 0) or (
                len(slice_right) > 0 and len(split_right) > 0
            ):
                return self.none(node, inspect.currentframe().f_lineno)
            slice_right = slice_right[0] if slice_right else None
            split_right = split_right[0] if split_right else None
        else:
            return self.none(node, inspect.currentframe().f_lineno)

        if (
            (slice_left is None and slice_right is not None)
            or (slice_left is not None and slice_right is None)
            or (split_left is None and split_right is not None)
            or (split_left is not None and split_right is None)
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        is_slice = slice_left is not None
        if is_slice:
            if slice_left.input[0] != slice_right.input[0]:
                return self.none(node, inspect.currentframe().f_lineno)

            slice_left_def = [g.get_computed_constant(i) for i in slice_left.input[1:]]
            slice_right_def = [g.get_computed_constant(i) for i in slice_right.input[1:]]
            if len(slice_left_def) != 3 or len(slice_right_def) != 3:
                return self.none(node, inspect.currentframe().f_lineno)
            if slice_left_def[2].tolist() != slice_right_def[2].tolist():
                # axis are different
                return self.none(node, inspect.currentframe().f_lineno)
            lengths = {len(v) for v in slice_left_def} | {len(v) for v in slice_right_def}
            if lengths != {1}:
                # more than one axis
                return self.none(node, inspect.currentframe().f_lineno)

            axis = slice_left_def[2][0]
            dim_left = slice_left_def[1][0] - slice_left_def[0][0]
            dim_right = slice_right_def[1][0] - slice_right_def[0][0]

            shape_left = g.get_computed_constant(cst_left.input[0])
            shape_right = g.get_computed_constant(cst_right.input[0])
            cdim_left = shape_left[axis]
            cdim_right = shape_right[axis]

            if dim_left != cdim_right or dim_right != cdim_left:
                return self.none(node, inspect.currentframe().f_lineno)
        else:
            if make_idn(split_left) != make_idn(split_right):
                # not the same split
                return self.none(node, inspect.currentframe().f_lineno)
            if len(split_left.output) != 2:
                return self.none(node, inspect.currentframe().f_lineno)
            axis = g.get_axis(split_left, 0)
            axis1 = g.get_axis(concat_left, 0)
            axis2 = g.get_axis(concat_right, 0)
            if len({axis, axis1, axis2}) != 1:
                return self.none(node, inspect.currentframe().f_lineno)

        # last check about size.
        axis = g.get_axis(concat_left, 0)
        all_inputs = list(concat_left.input) + list(concat_right.input)
        shapes = [g.get_shape(x) for x in all_inputs]
        dims = [s[axis] for s in shapes]
        # We know that dims[0] + dims[1] == dims[2] + dims[3], otherwise,
        # the addition next to Concat would not be possible.
        # We need now that dims[1] + dims[2] == dims[0] + dims[3],
        # then dims[1] - dims[0] == dims[3] - dims[2],
        # then (1) + (2) ==> dims[1] = dims[3]
        idims = set(d for d in dims if isinstance(d, int))
        sdims = set(d for d in dims if isinstance(d, str))
        if len(idims) > 1 or len(sdims) > 2:
            return self.none(node, inspect.currentframe().f_lineno)

        nodes = [
            cst_left,
            split_left,
            slice_left,
            neg_left,
            concat_left,
            cst_right,
            slice_right,
            neg_right,
            concat_right,
            node,
        ]

        return MatchResult(self, nodes, self.apply_concat)

    def apply_concat(
        self,
        g: "GraphBuilder",  # noqa: F821
        cst_left: NodeProto,
        split: NodeProto,
        slice_left: NodeProto,
        neg_left: Optional[NodeProto],
        concat_left: NodeProto,
        cst_right: NodeProto,
        slice_right: NodeProto,
        neg_right: Optional[NodeProto],
        concat_right: NodeProto,
        node: NodeProto,
    ) -> List[NodeProto]:
        is_split = split is not None
        if is_split:
            axis = g.get_attribute(split, "axis").i
        else:
            axis = g.get_computed_constant(slice_left.input[3])[0]

        if neg_left is None:
            neg_out = neg_right.output[0]
            pos = list(concat_right.input).index(neg_out)
            concat_inputs = (
                [
                    neg_right.output[0],
                    split.output[0] if is_split else slice_left.output[0],
                ]
                if pos == 0
                else [
                    split.output[0] if is_split else slice_left.output[0],
                    neg_right.output[0],
                ]
            )
            neg = neg_right
        else:
            neg_out = neg_left.output[0]
            pos = list(concat_left.input).index(neg_out)
            concat_inputs = (
                [
                    neg_left.output[0],
                    split.output[1] if is_split else slice_right.output[0],
                ]
                if pos == 0
                else [
                    split.output[1] if is_split else slice_right.output[0],
                    neg_left.output[0],
                ]
            )
            neg = neg_left

        concat = g.make_node(
            "Concat",
            concat_inputs,
            node.output,
            axis=int(axis),
            doc_string=node.doc_string,
            name=f"{self.__class__.__name__}--{node.name}",
        )

        # We still keep the constant in case other node use it.
        # The algorithm removing the unused nodes will decide
        # whether or not to keep it.

        if is_split:
            assert slice_left is None and slice_right is None
            return [cst_left, split, neg, concat]

        return [cst_left, slice_left, slice_right, neg, concat]

    def match_transpose(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Add" or node.domain != "":
            return self.none()
        if g.is_used_more_than_once(node.input[0]) or g.is_used_more_than_once(node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)

        transpose_left, transpose_right = (
            g.node_before(node.input[0]),
            g.node_before(node.input[1]),
        )
        if transpose_left.op_type != "Transpose" or transpose_right.op_type != "Transpose":
            return self.none(node, inspect.currentframe().f_lineno)

        perm_left = list(g.get_attribute(transpose_left, "perm").ints)
        perm_right = list(g.get_attribute(transpose_right, "perm").ints)
        if perm_left != perm_right:
            return self.none(node, inspect.currentframe().f_lineno)

        scatter_left, scatter_right = (
            g.node_before(transpose_left.input[0]),
            g.node_before(transpose_right.input[0]),
        )
        if scatter_left is None or scatter_right is None:
            return self.none(node, inspect.currentframe().f_lineno)

        if scatter_left.op_type != "ScatterND" or scatter_right.op_type != "ScatterND":
            return self.none(node, inspect.currentframe().f_lineno)

        tr_data_left, tr_data_right = (
            g.node_before(scatter_left.input[0]),
            g.node_before(scatter_right.input[0]),
        )
        if (
            tr_data_left is None
            or tr_data_left.op_type != "Transpose"
            or tr_data_right is None
            or tr_data_right.op_type != "Transpose"
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        if (
            list(g.get_attribute(tr_data_left, "perm").ints) != perm_left
            or list(g.get_attribute(tr_data_right, "perm").ints) != perm_right
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        tr_update_left, tr_update_right = (
            g.node_before(scatter_left.input[2]),
            g.node_before(scatter_right.input[2]),
        )
        if (
            list(g.get_attribute(tr_update_left, "perm").ints) != perm_left
            or list(g.get_attribute(tr_update_right, "perm").ints) != perm_right
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        nodes = [
            scatter_left,
            scatter_right,
            transpose_left,
            transpose_right,
            tr_data_left,
            tr_data_right,
            tr_update_left,
            tr_update_right,
        ]

        allowed = (scatter_left.input[0], scatter_right.input[0])
        if any(
            (node.output[0] not in allowed and g.is_used_more_than_once(node.output[0]))
            for node in nodes
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        cst_left = g.node_before(tr_data_left.input[0])
        cst_right = g.node_before(tr_data_right.input[0])
        if cst_left.op_type != "ConstantOfShape" or cst_right.op_type != "ConstantOfShape":
            return self.none(node, inspect.currentframe().f_lineno)

        slice_left = g.node_before(tr_update_left.input[0])
        slice_right = g.node_before(tr_update_right.input[0])
        if slice_left.op_type not in (
            "Slice",
            "Neg",
            "Split",
        ) or slice_right.op_type not in (
            "Slice",
            "Neg",
            "Split",
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        if slice_left.op_type == "Neg" and g.node_before(slice_left.input[0]).op_type not in (
            "Slice",
            "Split",
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        if slice_right.op_type == "Neg" and g.node_before(slice_right.input[0]).op_type not in (
            "Slice",
            "Split",
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        if slice_left.op_type == "Neg":
            neg_left = slice_left
            neg_right = None
            slice_left = g.node_before(neg_left.input[0])
        else:
            neg_left = None
            neg_right = slice_right
            slice_right = g.node_before(neg_right.input[0])

        nodes2 = [
            cst_left,
            slice_left,
            neg_left,
            cst_right,
            slice_right,
            neg_right,
            node,
        ]

        if any(
            (
                node is not None
                and node.op_type not in {"Constant", "ConstantOfShape"}
                and g.is_used_more_than_once(node.output[0])
            )
            for node in nodes2
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        use_split = False
        if slice_left.op_type == "Split" == slice_right.op_type:
            if make_idn(slice_left) != make_idn(slice_right):
                return self.none(node, inspect.currentframe().f_lineno)
            use_split = True

        # Checking shapes and indices
        if not g.has_shape(scatter_left.input[0]) or not g.has_shape(scatter_right.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        shape_left = g.get_shape(scatter_left.input[0])
        shape_right = g.get_shape(scatter_right.input[0])
        if shape_left != shape_right:
            return self.none(node, inspect.currentframe().f_lineno)

        indices_left = g.get_computed_constant(scatter_left.input[1])
        indices_right = g.get_computed_constant(scatter_right.input[1])
        if (
            len(indices_left.shape) != 2
            or indices_left.shape[1] != 1
            or len(indices_right.shape) != 2
            or indices_right.shape[1] != 1
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        ind_left = indices_left.ravel().tolist()
        ind_right = indices_right.ravel().tolist()
        if ind_left[0] == 0:
            ind = ind_left + ind_right
        else:
            ind = ind_right + ind_left

        if ind != list(range(shape_left[0])):
            return self.none(node, inspect.currentframe().f_lineno)

        # Slices
        if not use_split:
            if slice_left.input[0] != slice_right.input[0]:
                return self.none(node, inspect.currentframe().f_lineno)

            slice_left_def = [g.get_computed_constant(i) for i in slice_left.input[1:]]
            slice_right_def = [g.get_computed_constant(i) for i in slice_right.input[1:]]
            if len(slice_left_def) != 4 or len(slice_right_def) != 4:
                return self.none(node, inspect.currentframe().f_lineno)
            if slice_left_def[2].tolist() != slice_right_def[2].tolist():
                return self.none(node, inspect.currentframe().f_lineno)
            if slice_left_def[3].tolist() != [1] or slice_right_def[3].tolist() != [1]:
                return self.none(node, inspect.currentframe().f_lineno)
            lengths = {len(v) for v in slice_left_def} | {len(v) for v in slice_right_def}
            if lengths != {1}:
                # more than one axis
                return self.none(node, inspect.currentframe().f_lineno)
            axis = slice_left_def[2][0]
            dim_left = slice_left_def[1][0] - slice_left_def[0][0]
            dim_right = slice_right_def[1][0] - slice_right_def[0][0]

            shape_left = g.get_computed_constant(cst_left.input[0])
            shape_right = g.get_computed_constant(cst_right.input[0])
            cdim_left = shape_left[axis]
            cdim_right = shape_right[axis]

            if dim_right + dim_left != cdim_right or cdim_right != cdim_left:
                return self.none(node, inspect.currentframe().f_lineno)
        else:
            split_node = slice_left
            if not g.is_constant(split_node.input[1]):
                return self.none(node, inspect.currentframe().f_lineno)
            cst = g.get_computed_constant(split_node.input[1])
            if cst.min() != cst.max():
                return self.none(node, inspect.currentframe().f_lineno)

        nodes = [
            cst_left,
            slice_left,
            neg_left,
            cst_right,
            slice_right,
            neg_right,
            scatter_left,
            scatter_right,
            transpose_left,
            transpose_right,
            tr_data_left,
            tr_data_right,
            tr_update_left,
            tr_update_right,
            node,
        ]

        return MatchResult(self, nodes, self.apply_transpose)

    def apply_transpose(
        self,
        g: "GraphBuilder",  # noqa: F821
        cst_left: NodeProto,
        slice_left: NodeProto,
        neg_left: NodeProto,
        cst_right: NodeProto,
        slice_right: NodeProto,
        neg_right: NodeProto,
        scatter_left: NodeProto,
        scatter_right: NodeProto,
        transpose_left: NodeProto,
        transpose_right: NodeProto,
        tr_data_left: NodeProto,
        tr_data_right: NodeProto,
        tr_update_left: NodeProto,
        tr_update_right: NodeProto,
        node: NodeProto,
    ) -> List[NodeProto]:
        keep = []
        if cst_left is not None and g.is_used_more_than_once(cst_left.output[0]):
            keep.append(cst_left)
        if (
            cst_right is not None
            and g.is_used_more_than_once(cst_right.output[0])
            and (cst_left is None or make_idn(cst_left) != make_idn(cst_right))
        ):
            keep.append(cst_right)

        if slice_left.op_type == "Split" == slice_right.op_type:
            split = slice_left
            axis = g.get_attribute(split, "axis").i
        else:
            slice_left_def = [g.get_computed_constant(i) for i in slice_left.input[1:]]
            slice_right_def = [g.get_computed_constant(i) for i in slice_right.input[1:]]
            axis = slice_left_def[2][0]
            dim_left = slice_left_def[1][0] - slice_left_def[0][0]
            dim_right = slice_right_def[1][0] - slice_right_def[0][0]

            splits = g.make_initializer(
                "",
                np.array([dim_left, dim_right], dtype=np.int64),
                source="RotaryConcatPartPattern.apply.splits",
            )

            split = g.make_node(
                "Split",
                [slice_left.input[0], splits],
                [
                    g.unique_name(f"{self.__class__.__name__}--{node.output[0]}"),
                    g.unique_name(f"{self.__class__.__name__}--{node.output[0]}"),
                ],
                axis=int(axis),
                name=f"{self.__class__.__name__}--{node.name}",
            )

        if neg_left is None:
            neg = g.make_node(
                "Neg",
                [split.output[1]],
                [g.unique_name(f"{self.__class__.__name__}--{node.output[0]}")],
                name=f"{self.__class__.__name__}--{node.name}",
            )
            concat_inputs = [split.output[0], neg.output[0]]
        else:
            neg = g.make_node(
                "Neg",
                [split.output[0]],
                [g.unique_name(f"{self.__class__.__name__}--{node.output[0]}")],
                name=f"{self.__class__.__name__}--{node.name}",
            )
            concat_inputs = [neg.output[0], split.output[1]]

        concat = g.make_node(
            "Concat",
            concat_inputs,
            node.output,
            axis=int(axis),
            doc_string=node.doc_string,
            name=f"{self.__class__.__name__}--{node.name}",
        )

        # restoring nodes used more than once
        other_nodes = []
        if scatter_left is not None and g.is_used_more_than_once(scatter_left.input[0]):
            if tr_data_left is not None:
                other_nodes.append(tr_data_left)
            if cst_left is not None:
                other_nodes.append(cst_left)
        if scatter_right is not None and g.is_used_more_than_once(scatter_right.input[0]):
            if tr_data_right is not None and make_idn(tr_data_right) != make_idn(tr_data_left):
                other_nodes.append(tr_data_right)
            if cst_right is not None and make_idn(cst_right) != make_idn(cst_left):
                other_nodes.append(cst_right)
        if other_nodes:
            return [*reversed(other_nodes), split, neg, concat]
        return [*keep, split, neg, concat]


class FunctionHalfRotaryEmbeddingPattern(PatternOptimization):
    """
    Fuses nodes matching half RotaryEmbedding(23) into a local function.

    .. runpython::
        :showcode:

        from experimental_experiment.xbuilder import GraphBuilder
        from experimental_experiment.xoptim import GraphBuilderPatternOptimization
        from experimental_experiment.xoptim.patterns import (
            FunctionHalfRotaryEmbeddingPattern,
        )

        pat = FunctionHalfRotaryEmbeddingPattern()
        g = GraphBuilderPatternOptimization(GraphBuilder(18))
        print(pat._pattern_to_string(g))
    """

    _operator_name = "HalfRotaryEmbedding"
    _domain_name = "intermediate"

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Split" or node.domain != "" or len(node.output) != 2:
            return self.none()
        split_node = node
        rk = None if not g.has_rank(node.input[0]) else g.get_rank(node.input[0])
        if rk is None or rk != 4:
            # 3 should be allowed as well. Let's see when it happens.
            return self.none(node, inspect.currentframe().f_lineno)

        # checks split is the expected split
        axis = g.get_attribute(node, "axis", exc=False)
        axis = 0 if axis is None else axis.i
        rk = g.get_rank(node.input[0])
        if axis < 0:
            axis += rk
        if axis != rk - 1:
            return self.none(node, inspect.currentframe().f_lineno)

        if len(node.input) == 2:
            cst = g.get_computed_constant(node.input[1])
            if cst.dtype != np.int64 or cst.shape != (2,) or cst[0] != cst[1]:
                return self.none(node, inspect.currentframe().f_lineno)
        else:
            att = g.get_attribute(node, "num_outputs", exc=False)
            if att is None or att.i != 2:
                return self.none(node, inspect.currentframe().f_lineno)

        # checks what follows
        node_after = g.next_nodes(node.output[1])
        if len(node_after) != 1 or node_after[0].op_type != "Neg":
            return self.none(node, inspect.currentframe().f_lineno)
        neg_node = node_after[0]
        node_after = g.next_nodes(neg_node.output[0])
        if len(node_after) != 1 or node_after[0].op_type != "Concat":
            return self.none(node, inspect.currentframe().f_lineno)
        concat_node = node_after[0]
        if set(concat_node.input) != {split_node.output[0], neg_node.output[0]}:
            return self.none(node, inspect.currentframe().f_lineno)
        axis = g.get_attribute(concat_node, "axis").i
        if axis != -1 and (rk is None or axis != rk - 1):
            return self.none(node, inspect.currentframe().f_lineno)

        node_after = g.next_nodes(concat_node.output[0])
        if len(node_after) != 1 or node_after[0].op_type != "Mul":
            return self.none(node, inspect.currentframe().f_lineno)
        mul1_node = node_after[0]
        node_after = g.next_nodes(split_node.input[0])
        if len(node_after) != 2:
            return self.none(node, inspect.currentframe().f_lineno)
        mul2_node = node_after[0] if node_after[0].op_type == "Mul" else node_after[1]
        if mul2_node.op_type != "Mul":
            return self.none(node, inspect.currentframe().f_lineno)
        node_after1 = g.next_nodes(mul1_node.output[0])
        node_after2 = g.next_nodes(mul2_node.output[0])
        if (
            len(node_after1) != 1
            or len(node_after2) != 1
            or node_after1[0].op_type != node_after2[0].op_type
            or node_after1[0].op_type != "Add"
            or make_idn(node_after1[0]) != make_idn(node_after2[0])
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(
            self,
            [split_node, neg_node, concat_node, mul1_node, mul2_node, node_after1[0]],
            self.apply,
            insert_at=node_after1[0],
        )

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        split_node: NodeProto,
        neg_node: NodeProto,
        concat_node: NodeProto,
        mul1_node: NodeProto,
        mul2_node: NodeProto,
        add_node: NodeProto,
    ) -> List[NodeProto]:
        names = (set(mul1_node.input) | set(mul2_node.input)) - {
            split_node.input[0],
            concat_node.output[0],
        }
        assert len(names) == 2, f"Inconsistency names={names}, it should have 2 names"
        lnames = list(names)
        index = 0 if lnames[0] in mul1_node.input else 1
        lnames = [lnames[index], lnames[1 - index]]
        cos_cache, sin_cache = lnames if concat_node.input[0] in mul1_node.input else lnames[::-1]

        rotary_nodes = [
            g.make_node(
                self._operator_name,
                [split_node.input[0], cos_cache, sin_cache],
                [add_node.output[0]],
                name=f"{self.__class__.__name__}--{split_node.name}",
                domain=self._domain_name,
            )
        ]
        nodes_to_return = rotary_nodes

        # Creates the local function
        if not g.builder.has_local_function(self._operator_name, domain=self._domain_name):
            self._add_local_function(g.builder)
        return nodes_to_return

    @classmethod
    def _add_local_function(cls, g: GraphBuilder):
        lg = GraphBuilder(g.main_opset, as_function=True)
        lg.make_tensor_input("X")
        lg.make_tensor_input("cos_cache")
        lg.make_tensor_input("sin_cache")

        left, right = lg.op.Split("X", num_outputs=2, axis=-1, name=cls.__name__)
        right_neg = lg.op.Neg(right, name=cls.__name__)
        conc = lg.op.Concat(right_neg, left, axis=-1, name=cls.__name__)
        lg.op.Add(
            lg.op.Mul("X", "cos_cache", name=cls.__name__),
            lg.op.Mul(conc, "sin_cache", name=cls.__name__),
            outputs=["Y"],
        )
        lg.make_tensor_output("Y")

        function_options = FunctionOptions(
            export_as_function=True, name=cls._operator_name, domain=cls._domain_name
        )
        g.make_local_function(lg, function_options=function_options)
        assert g.has_local_function(cls._operator_name, domain=cls._domain_name), (
            f"The function {cls._domain_name}.{cls._operator_name} "
            f"was not added to the builder."
        )


class RotaryEmbeddingPattern(PatternOptimization):
    """Fuses nodes matching RotaryEmbedding(23)."""

    _operator_name = FunctionHalfRotaryEmbeddingPattern._operator_name
    _domain_name = FunctionHalfRotaryEmbeddingPattern._domain_name

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if (
            g.main_opset < 23
            or node.op_type != self._operator_name
            or node.domain != self._domain_name
        ):
            # Not ready in opset 23.
            return self.none()
        if not g.has_rank(node.input[0]) or g.get_rank(node.input[0]) != 4:
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.has_shape(node.input[1]) or not g.has_shape(node.input[2]):
            return self.none(node, inspect.currentframe().f_lineno)
        shape_cos = g.get_shape(node.input[1])
        shape_sin = g.get_shape(node.input[2])
        if shape_cos != shape_sin:
            return self.none(node, inspect.currentframe().f_lineno)
        if len(shape_cos) != 4:
            return self.none(node, inspect.currentframe().f_lineno)
        if shape_cos[1] != 1 or shape_sin[1] != 1:
            return self.none(node, inspect.currentframe().f_lineno)

        concat_cos = g.node_before(node.input[1])
        if concat_cos is None or concat_cos.op_type != "Concat" or concat_cos.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if concat_cos.input[0] != concat_cos.input[1]:
            return self.none(node, inspect.currentframe().f_lineno)
        if g.get_attribute(concat_cos, "axis").i != -1:
            return self.none(node, inspect.currentframe().f_lineno)

        concat_sin = g.node_before(node.input[2])
        if concat_sin is None or concat_sin.op_type != "Concat" or concat_sin.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if concat_sin.input[0] != concat_sin.input[1]:
            return self.none(node, inspect.currentframe().f_lineno)
        if g.get_attribute(concat_sin, "axis").i != -1:
            return self.none(node, inspect.currentframe().f_lineno)

        if g.is_used_more_than_once(node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        split_node = g.node_before(node.input[0])
        if split_node is None or split_node.op_type != "Split" or split_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant(split_node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        cst = g.get_computed_constant(split_node.input[1])
        if cst.shape != (2,):
            return self.none(node, inspect.currentframe().f_lineno)

        next_nodes = g.next_nodes(node.output[0])
        if len(next_nodes) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        concat_node = next_nodes[0]
        if concat_node.op_type != "Concat" or concat_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if split_node.output[1] != concat_node.input[1]:
            return self.none(node, inspect.currentframe().f_lineno)
        axis = g.get_attribute(concat_node, "axis").i
        if axis != -1:
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(
            self,
            [concat_cos, concat_sin, split_node, node, concat_node],
            self.apply,
            insert_at=None if g.is_used_more_than_once(concat_cos.output[0]) else concat_node,
        )

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        concat_cos: NodeProto,
        concat_sin: NodeProto,
        split_node: NodeProto,
        half_node: NodeProto,
        concat_node: NodeProto,
    ) -> List[NodeProto]:

        if split_node is None:
            rotary_dim = None
            shape = g.get_shape(half_node.input[0])
            main_input = half_node.input[0]
            main_output = half_node.output[0]
        else:
            cst = g.get_computed_constant(split_node.input[1])
            rotary_dim = int(cst[0])
            shape = g.get_shape(split_node.input[0])
            main_input = split_node.input[0]
            main_output = concat_node.output[0]
        assert isinstance(shape[1], int), f"Number of heads is not fixed, shape(X)={shape}"
        num_heads = shape[1]

        rotary_nodes = []
        if g.is_used_more_than_once(concat_cos.output[0]):
            rotary_nodes.append(concat_cos)
        if g.is_used_more_than_once(concat_sin.output[0]):
            rotary_nodes.append(concat_sin)

        cos_name = g.unique_name(f"{self.__class__.__name__}--{half_node.input[1]}")
        sin_name = g.unique_name(f"{self.__class__.__name__}--{half_node.input[2]}")
        cos_expanded = g.unique_name(f"{self.__class__.__name__}--{half_node.input[1]}")
        sin_expanded = g.unique_name(f"{self.__class__.__name__}--{half_node.input[2]}")
        batch_name = g.unique_name(f"{self.__class__.__name__}--{half_node.input[0]}--dim")
        shape_name = g.unique_name(f"{self.__class__.__name__}--{half_node.input[0]}::Shape")
        one = g.make_initializer("", g.ONE, source=f"{self.__class__.__name__}.1")
        ones = g.make_initializer(
            "", np.array([1, 1], dtype=np.int64), source=f"{self.__class__.__name__}.11"
        )
        rotary_nodes.extend(
            [
                g._make_node("Shape", [split_node.input[0]], [batch_name], start=0, end=1),
                g._make_node("Concat", [batch_name, ones], [shape_name], axis=0),
                g._make_node("Squeeze", [concat_cos.input[0], one], [cos_name]),
                g._make_node("Squeeze", [concat_sin.input[0], one], [sin_name]),
                g._make_node("Expand", [cos_name, shape_name], [cos_expanded]),
                g._make_node("Expand", [sin_name, shape_name], [sin_expanded]),
                g._make_node(
                    "RotaryEmbedding",
                    [main_input, cos_expanded, sin_expanded],
                    [main_output],
                    rotary_embedding_dim=rotary_dim,
                    num_heads=num_heads,
                ),
            ]
        )
        for node in rotary_nodes:
            if not node.name:
                node.name = g.builder.unique_node_name(
                    f"{self.__class__.__name__}--{half_node.name}"
                )
        return rotary_nodes


class FunctionCausalMaskPattern(PatternOptimization):
    """
    Fuses nodes matching CausalMask into a local function.

    .. runpython::
        :showcode:

        from experimental_experiment.xbuilder import GraphBuilder
        from experimental_experiment.xoptim import GraphBuilderPatternOptimization
        from experimental_experiment.xoptim.patterns import (
            FunctionCausalMaskPattern,
        )

        pat = FunctionCausalMaskPattern()
        g = GraphBuilderPatternOptimization(GraphBuilder(18))
        print(pat._pattern_to_string(g))
    """

    _operator_names = ["CausalMask", "ShiftedCausalMask"]
    _domain_name = "intermediate"

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type not in {"LessOrEqual", "Greater"} or node.domain != "":
            return self.none()

        sq1 = g.node_before(node.input[0])
        sq2 = g.node_before(node.input[1])
        if sq1 is None or sq1.op_type != "Unsqueeze" or len(sq1.input) != 2:
            return self.none(node, inspect.currentframe().f_lineno)
        if sq2 is None or sq2.op_type not in {"Unsqueeze", "Sub"} or len(sq2.input) != 2:
            return self.none(node, inspect.currentframe().f_lineno)
        if sq2.op_type == "Sub":
            sub2 = sq2
            sq2 = g.node_before(sub2.input[0])
            if sq2 is None or sq2.op_type != "Unsqueeze" or len(sq2.input) != 2:
                return self.none(node, inspect.currentframe().f_lineno)
        else:
            sub2 = None
        if not g.is_constant(sq1.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant(sq2.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)

        axes1 = g.get_computed_constant(sq1.input[1])
        axes2 = g.get_computed_constant(sq2.input[1])
        if tuple(axes1) != (0, 1, 2):
            return self.none(node, inspect.currentframe().f_lineno)
        if tuple(axes2) != (0, 1, 3):
            return self.none(node, inspect.currentframe().f_lineno)

        rg1 = g.node_before(sq1.input[0])
        rg2 = g.node_before(sq2.input[0])
        if rg1 is None or rg1.op_type != "Range" or len(rg1.input) != 3:
            return self.none(node, inspect.currentframe().f_lineno)
        if rg2 is None or rg2.op_type != "Range" or len(rg1.input) != 3:
            return self.none(node, inspect.currentframe().f_lineno)

        if rg1.input[1] != rg2.input[1]:
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant(rg1.input[2]) and g.get_constant_scalar(rg1.input[2]) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant(rg1.input[0]) and g.get_constant_scalar(rg1.input[2]) != 0:
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant(rg2.input[2]) and g.get_constant_scalar(rg2.input[2]) != 1:
            return self.none(node, inspect.currentframe().f_lineno)

        # rg1 (0, d, 1)
        # rg2 (c, d, 1)

        dsq1 = g.node_before(rg2.input[0])
        dsq2 = g.node_before(rg2.input[1])
        if dsq1 is None or dsq1.op_type != "Squeeze" or len(dsq1.input) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        if dsq2 is None or dsq2.op_type != "Squeeze" or len(dsq2.input) != 1:
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [dsq1, dsq2, rg1, rg2, sq1, sq2, sub2, node], self.apply)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        dim_squeeze1: NodeProto,
        dim_squeeze2: NodeProto,
        range1: NodeProto,
        range2: NodeProto,
        rg_unsqueeze1: NodeProto,
        rg_unsqueeze2: NodeProto,
        sub2: NodeProto,
        less_or_equal: NodeProto,
    ) -> List[NodeProto]:
        version = 0 if sub2 is None else 1
        operator_name = self._operator_names[version]
        nodes_to_return = []
        if (
            g.is_used_more_than_once(dim_squeeze1.output[0])
            or g.is_used_more_than_once(range1.output[0])
            or g.is_used_more_than_once(rg_unsqueeze1.output[0])
        ):
            nodes_to_return.append(dim_squeeze1)
        if (
            g.is_used_more_than_once(dim_squeeze2.output[0])
            or g.is_used_more_than_once(range2.output[0])
            or g.is_used_more_than_once(rg_unsqueeze2.output[0])
        ):
            nodes_to_return.append(dim_squeeze2)
        if g.is_used_more_than_once(range1.output[0]) or g.is_used_more_than_once(
            rg_unsqueeze1.output[0]
        ):
            nodes_to_return.append(range1)
        if g.is_used_more_than_once(range2.output[0]) or g.is_used_more_than_once(
            rg_unsqueeze2.output[0]
        ):
            nodes_to_return.append(range2)
        if g.is_used_more_than_once(rg_unsqueeze1.output[0]):
            nodes_to_return.append(rg_unsqueeze1)
        if sub2 is not None and g.is_used_more_than_once(sub2.output[0]):
            nodes_to_return.append(sub2)
        if g.is_used_more_than_once(rg_unsqueeze2.output[0]):
            nodes_to_return.append(rg_unsqueeze2)

        # The matching checks the output of the other nodes are not used more than once.
        sub_inputs = [sub2.input[1]] if sub2 is not None else []
        nodes_to_return.append(
            g.make_node(
                operator_name,
                [dim_squeeze1.input[0], dim_squeeze2.input[0], *sub_inputs],
                [less_or_equal.output[0]],
                name=f"{self.__class__.__name__}--{less_or_equal.name}",
                domain=self._domain_name,
            )
        )

        # Creates the local function
        if not g.builder.has_local_function(operator_name, domain=self._domain_name):
            self._add_local_function(g.builder, version)
        return nodes_to_return

    @classmethod
    def _add_local_function(cls, g: GraphBuilder, version: int):
        lg = GraphBuilder(g.main_opset, as_function=True)
        lg.make_tensor_input("A")
        lg.make_tensor_input("B")
        if version == 1:
            lg.make_tensor_input("shift")

        sA = lg.op.Squeeze("A", name=cls.__name__)
        sB = lg.op.Squeeze("B", name=cls.__name__)

        rg1 = lg.op.Range(lg.ZERO_NO_DIM, sB, lg.ONE_NO_DIM, name=cls.__name__)
        rg2 = lg.op.Range(sA, sB, lg.ONE_NO_DIM, name=cls.__name__)

        unsq1 = lg.op.Unsqueeze(rg1, np.array([0, 1, 2], dtype=np.int64), name=cls.__name__)
        unsq2 = lg.op.Unsqueeze(rg2, np.array([0, 1, 3], dtype=np.int64), name=cls.__name__)
        if version == 0:
            lg.op.LessOrEqual(unsq1, unsq2, outputs=["mask"], name=cls.__name__)
        else:
            sub = lg.op.Sub(unsq2, "shift", name=cls.__name__)
            lg.op.Greater(unsq1, sub, outputs=["mask"], name=cls.__name__)

        lg.make_tensor_output("mask")

        operator_name = cls._operator_names[version]
        function_options = FunctionOptions(
            export_as_function=True,
            name=operator_name,
            domain=cls._domain_name,
            move_initializer_to_constant=True,
        )
        g.make_local_function(lg, function_options=function_options)
        assert g.has_local_function(
            operator_name, domain=cls._domain_name
        ), f"The function {cls._domain_name}.{operator_name} was not added to the builder."


class FunctionCausalMaskMulAddPattern(PatternOptimization):
    """
    Fuses nodes matching CausalMask into a local function.

    .. runpython::
        :showcode:

        from experimental_experiment.xbuilder import GraphBuilder
        from experimental_experiment.xoptim import GraphBuilderPatternOptimization
        from experimental_experiment.xoptim.patterns import (
            FunctionCausalMaskMulAddPattern,
        )

        pat = FunctionCausalMaskMulAddPattern()
        g = GraphBuilderPatternOptimization(GraphBuilder(18))
        print(pat._pattern_to_string(g))
    """

    _operator_name = "CausalMaskMulAdd"
    _domain_name = "intermediate"

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Add" or node.domain != "":
            return self.none()

        sq1 = g.node_before(node.input[0])
        mul = g.node_before(node.input[1])
        if sq1 is None or sq1.op_type != "Unsqueeze" or len(sq1.input) != 2:
            return self.none(node, inspect.currentframe().f_lineno)
        if sq1 is None or g.is_used_more_than_once(sq1.output[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        if mul is None or g.is_used_more_than_once(mul.output[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant(sq1.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)

        sq2 = g.node_before(mul.input[0])
        if sq2 is None or sq2.op_type != "Unsqueeze" or len(sq2.input) != 2:
            return self.none(node, inspect.currentframe().f_lineno)
        if g.is_used_more_than_once(sq2.output[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant(sq2.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)

        axes1 = g.get_computed_constant(sq1.input[1])
        axes2 = g.get_computed_constant(sq2.input[1])
        if tuple(axes1) != (0, 1, 2):
            return self.none(node, inspect.currentframe().f_lineno)
        if tuple(axes2) != (1, 2, 3):
            return self.none(node, inspect.currentframe().f_lineno)

        rg1 = g.node_before(sq1.input[0])
        rg2 = g.node_before(sq2.input[0])
        if g.is_used_more_than_once(rg1.output[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        if g.is_used_more_than_once(rg2.output[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        if rg1 is None or rg1.op_type != "Range" or len(rg1.input) != 3:
            return self.none(node, inspect.currentframe().f_lineno)
        if rg2 is None or rg2.op_type != "Range" or len(rg1.input) != 3:
            return self.none(node, inspect.currentframe().f_lineno)

        if not g.is_constant(rg1.input[2]) and g.get_constant_scalar(rg1.input[2]) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant(rg1.input[0]) and g.get_constant_scalar(rg1.input[2]) != 0:
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant(rg2.input[0]) and g.get_constant_scalar(rg2.input[0]) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant(rg2.input[2]) and g.get_constant_scalar(rg2.input[2]) != 1:
            return self.none(node, inspect.currentframe().f_lineno)

        # rg1 (0, d, 1)
        # rg2 (c, d, 1)

        dsq1 = g.node_before(rg1.input[1])
        dsq2 = g.node_before(rg2.input[1])
        if dsq1 is None or dsq1.op_type != "Squeeze" or len(dsq1.input) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        if dsq2 is None or dsq2.op_type != "Squeeze" or len(dsq2.input) != 1:
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [dsq1, dsq2, rg1, rg2, sq1, sq2, mul, node], self.apply)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        dim_squeeze1: NodeProto,
        dim_squeeze2: NodeProto,
        range1: NodeProto,
        range2: NodeProto,
        rg_unsqueeze1: NodeProto,
        rg_unsqueeze2: NodeProto,
        mul: NodeProto,
        add: NodeProto,
    ) -> List[NodeProto]:
        nodes_to_return = []
        if g.is_used_more_than_once(dim_squeeze1.output[0]):
            nodes_to_return.append(dim_squeeze1)
        if g.is_used_more_than_once(dim_squeeze2.output[0]):
            nodes_to_return.append(dim_squeeze2)

        # The matching checks the output of the other nodes are not used more than once.
        nodes_to_return.append(
            g.make_node(
                self._operator_name,
                [dim_squeeze1.input[0], dim_squeeze2.input[0], mul.input[1]],
                [add.output[0]],
                name=f"{self.__class__.__name__}--{add.name}",
                domain=self._domain_name,
            )
        )

        # Creates the local function
        if not g.builder.has_local_function(self._operator_name, domain=self._domain_name):
            self._add_local_function(g.builder)
        return nodes_to_return

    @classmethod
    def _add_local_function(cls, g: GraphBuilder):
        lg = GraphBuilder(g.main_opset, as_function=True)
        lg.make_tensor_input("A")
        lg.make_tensor_input("B")
        lg.make_tensor_input("C")

        sA = lg.op.Squeeze("A", name=cls.__name__)
        sB = lg.op.Squeeze("B", name=cls.__name__)

        rg1 = lg.op.Range(lg.ZERO_NO_DIM, sA, lg.ONE_NO_DIM, name=cls.__name__)
        rg2 = lg.op.Range(lg.ZERO_NO_DIM, sB, lg.ONE_NO_DIM, name=cls.__name__)

        unsq1 = lg.op.Unsqueeze(rg1, np.array([0, 1, 2], dtype=np.int64), name=cls.__name__)
        unsq2 = lg.op.Unsqueeze(rg2, np.array([1, 2, 3], dtype=np.int64), name=cls.__name__)

        mul = lg.op.Mul(unsq2, "C", name=cls.__name__)
        lg.op.Add(mul, unsq1, outputs=["mask"], name=cls.__name__)

        lg.make_tensor_output("mask")

        function_options = FunctionOptions(
            export_as_function=True,
            name=cls._operator_name,
            domain=cls._domain_name,
            move_initializer_to_constant=True,
        )
        g.make_local_function(lg, function_options=function_options)
        assert g.has_local_function(cls._operator_name, domain=cls._domain_name), (
            f"The function {cls._domain_name}.{cls._operator_name} "
            f"was not added to the builder."
        )


class FunctionCosSinCachePattern(PatternOptimization):
    """
    Fuses nodes to simplify the creation of cos/sin caches in LLM.

    .. runpython::
        :showcode:

        from experimental_experiment.xbuilder import GraphBuilder
        from experimental_experiment.xoptim import GraphBuilderPatternOptimization
        from experimental_experiment.xoptim.patterns import (
            FunctionCosSinCachePattern,
        )

        pat = FunctionCosSinCachePattern()
        g = GraphBuilderPatternOptimization(GraphBuilder(18))
        print(pat._pattern_to_string(g))
    """

    _operator_name = "CosSinCache"
    _domain_name = "intermediate"

    def _match_branch(
        self, g: "GraphBuilderPatternOptimization", node: NodeProto  # noqa: F821
    ) -> Optional[Tuple[NodeProto, NodeProto]]:
        next_nodes = g.next_nodes(node.output[0])
        if len(next_nodes) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        next_node = next_nodes[0]
        if next_node.op_type == "Cast" and next_node.domain == "":
            cast_node = next_node
        else:
            cast_node = None
        return cast_node

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Cos" or node.domain != "":
            return self.none()

        cos = node
        cos_sin = g.next_nodes(cos.input[0])
        if len(cos_sin) != 2:
            return self.none(node, inspect.currentframe().f_lineno)
        sin = cos_sin[1 if cos.output[0] == cos_sin[0].output[0] else 0]

        # what follows
        cos_cast = self._match_branch(g, cos)
        sin_cast = self._match_branch(g, sin)

        # what is before
        mul_node = g.node_before(cos.input[0])
        if (
            mul_node is None
            or g.is_used_more_than_once(mul_node.input[0])
            or mul_node.op_type != "Mul"
            or mul_node.domain != ""
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        reshape_node = g.node_before(mul_node.input[1])
        if (
            reshape_node is None
            or g.is_used_more_than_once(reshape_node.input[0])
            or reshape_node.op_type != "Reshape"
            or reshape_node.domain != ""
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant(reshape_node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        cst = g.get_computed_constant(reshape_node.input[1])
        if tuple(cst) != (0, -1, 1):
            return self.none(node, inspect.currentframe().f_lineno)
        if g.is_used_more_than_once(reshape_node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)

        cast_node = g.node_before(reshape_node.input[0])
        if cast_node is None:
            return self.none(node, inspect.currentframe().f_lineno)
        if g.is_used_more_than_once(cast_node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        unsq_node = g.node_before(cast_node.input[0])
        if unsq_node is None or unsq_node.op_type != "Unsqueeze" or unsq_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant(unsq_node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        cst_position_ids = g.get_computed_constant(unsq_node.input[1])
        if tuple(cst_position_ids) not in ((0, 1), (1,)):
            # unsqueeze before position ids
            # (0, 1) -> input is a scalar
            # (1, ) -> input is a tensor with position_ids
            return self.none(node, inspect.currentframe().f_lineno)

        range_node = g.node_before(unsq_node.input[0])
        if range_node is None:
            if cst_position_ids != (1,):
                return self.none(node, inspect.currentframe().f_lineno)
        elif (
            g.is_used_more_than_once(range_node.input[0])
            or g.is_used_more_than_once(range_node.input[1])
            or g.is_used_more_than_once(unsq_node.input[0])
            or range_node.op_type != "Range"
            or range_node.domain != ""
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        if range_node is not None and (
            not g.is_constant(range_node.input[2])
            or g.get_constant_scalar(range_node.input[2]) != 1
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        if range_node:
            dim_squeeze1 = g.node_before(range_node.input[0])
            if (
                dim_squeeze1 is None
                or dim_squeeze1.op_type != "Squeeze"
                or dim_squeeze1.domain != ""
                or len(dim_squeeze1.input) != 1
            ):
                return self.none(node, inspect.currentframe().f_lineno)
            dim_squeeze2 = g.node_before(range_node.input[1])
            if (
                dim_squeeze2 is None
                or dim_squeeze2.op_type != "Squeeze"
                or dim_squeeze2.domain != ""
                or len(dim_squeeze2.input) != 1
            ):
                return self.none(node, inspect.currentframe().f_lineno)
        else:
            dim_squeeze1 = None
            dim_squeeze2 = None

        if (sin_cast is None) != (cos_cast is None):
            # cast issue
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(
            self,
            [
                dim_squeeze1,
                dim_squeeze2,
                range_node,
                unsq_node,
                cast_node,
                reshape_node,
                mul_node,
                cos,
                cos_cast,
                sin,
                sin_cast,
            ],
            self.apply,
        )

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        dim_squeeze1: NodeProto,
        dim_squeeze2: NodeProto,
        range_node: NodeProto,
        unsq_node: NodeProto,
        cast_node: NodeProto,
        reshape_node: NodeProto,
        mul_node: NodeProto,
        cos: NodeProto,
        cos_cast: Optional[NodeProto],
        sin: NodeProto,
        sin_cast: Optional[NodeProto],
    ) -> List[NodeProto]:
        # Builds the name of the local function.
        to = None if (cos_cast is None or sin_cast is None) else g.get_attribute(cos_cast, "to").i
        cst_position_ids = tuple(g.get_computed_constant(unsq_node.input[1]))
        name = self._operator_name if to is None else f"{self._operator_name}_to{to}"
        if cst_position_ids != (0, 1):
            suffix = "".join(map(str, cst_position_ids))
            name = f"{name}_p{suffix}"
            assert (
                range_node is None and dim_squeeze1 is None and dim_squeeze2 is None
            ), "position_ids comes from the input not from a Range node"

        # The matching checks the output of the other nodes are not used more than once.
        keep_nodes = []
        if g.is_used_more_than_once(reshape_node.output[0]):
            keep_nodes.extend(
                [
                    n
                    for n in [
                        dim_squeeze1,
                        dim_squeeze2,
                        range_node,
                        unsq_node,
                        cast_node,
                        reshape_node,
                    ]
                    if n
                ]
            )
        nodes_to_return = [
            *keep_nodes,
            g.make_node(
                name,
                (
                    [dim_squeeze1.input[0], dim_squeeze2.input[0], mul_node.input[0]]
                    if range_node
                    else [unsq_node.input[0], mul_node.input[0]]
                ),
                [
                    cos_cast.output[0] if to is not None else cos.output[0],
                    sin_cast.output[0] if to is not None else sin.output[0],
                ],
                name=f"{self.__class__.__name__}--{mul_node.name}",
                domain=self._domain_name,
            ),
        ]

        # Creates the local function
        if not g.builder.has_local_function(name, domain=self._domain_name):
            self._add_local_function(
                g.builder,
                name=name,
                to=to,
                cst_position_ids=cst_position_ids,
                has_range_node=range_node is not None,
            )
        return nodes_to_return

    @classmethod
    def _add_local_function(
        cls,
        g: GraphBuilder,
        name: str,
        to: Optional[int] = None,
        cst_position_ids: Optional[STATIC_SHAPE] = None,
        has_range_node: bool = True,
    ):
        lg = GraphBuilder(g.main_opset, as_function=True)
        lg.make_tensor_input("dim1" if has_range_node else "position_ids")

        if has_range_node:
            lg.make_tensor_input("dim2")
            sA = lg.op.Squeeze("dim1", name=name)
            sB = lg.op.Squeeze("dim2", name=name)
            rg = lg.op.Range(sA, sB, lg.ONE_NO_DIM, name=name)
        else:
            rg = "position_ids"

        lg.make_tensor_input("weights")

        unsq = lg.op.Unsqueeze(rg, np.array(cst_position_ids, dtype=np.int64), name=name)
        cast = lg.op.Cast(unsq, to=TensorProto.FLOAT, name=name)
        resh = lg.op.Reshape(cast, np.array([0, -1, 1], dtype=np.int64), name=name)
        mul = lg.op.Mul("weights", resh, name=name)

        if to is not None:
            cos = lg.op.Cos(mul, name=name)
            sin = lg.op.Sin(mul, name=name)
            cos = lg.op.Cast(cos, to=to, name=name, outputs=["cos"])
            sin = lg.op.Cast(sin, to=to, name=name, outputs=["sin"])
        else:
            cos = lg.op.Cos(mul, name=name, outputs=["cos"])
            sin = lg.op.Sin(mul, name=name, outputs=["sin"])

        lg.make_tensor_output("cos")
        lg.make_tensor_output("sin")

        function_options = FunctionOptions(
            export_as_function=True,
            name=name,
            domain=cls._domain_name,
            move_initializer_to_constant=True,
        )
        g.make_local_function(lg, function_options=function_options)
        assert g.has_local_function(
            name, domain=cls._domain_name
        ), f"The function {cls._domain_name}.{name} was not added to the builder."
