import inspect
from typing import List, Optional
import numpy as np
from onnx import NodeProto
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
            if id(split_left) != id(split_right):
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

        if slice_right.op_type == "Neg" and g.node_before(
            slice_right.input[0]
        ).op_type not in ("Slice", "Split"):
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
            if id(slice_left) != id(slice_right):
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
    ):
        keep = []
        if cst_left is not None and g.is_used_more_than_once(cst_left.output[0]):
            keep.append(cst_left)
        if (
            cst_right is not None
            and g.is_used_more_than_once(cst_right.output[0])
            and (cst_left is None or id(cst_left) != id(cst_right))
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
            if tr_data_right is not None and id(tr_data_right) != id(tr_data_left):
                other_nodes.append(tr_data_right)
            if cst_right is not None and id(cst_right) != id(cst_left):
                other_nodes.append(cst_right)
        if other_nodes:
            return [*reversed(other_nodes), split, neg, concat]
        return [*keep, split, neg, concat]
