import inspect
from typing import List, Optional
from onnx import NodeProto
from .patterns_api import MatchResult, PatternOptimization


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
        if "ScatterElements" in (left.op_type, right.op_type):
            return self.match_scatter(g, node, matched)
        return self.none(node, inspect.currentframe().f_lineno)

    def match_concat(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Add" or node.domain != "":
            return self.none()
        if g.is_used_more_than_once(node.input[0]) or g.is_used_more_than_once(
            node.input[1]
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        concat_left, concat_right = g.node_before(node.input[0]), g.node_before(
            node.input[1]
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
        if any(map(lambda x: not g.has_shape(x), all_inputs)):
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

        cst_left = [n for n in concat_left_before if n.op_type == "ConstantOfShape"][0]
        cst_right = [n for n in concat_right_before if n.op_type == "ConstantOfShape"][
            0
        ]

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
            neg_left = [n for n in concat_left_before if n.op_type == "Neg"][0]
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
            slice_right_def = [
                g.get_computed_constant(i) for i in slice_right.input[1:]
            ]
            if len(slice_left_def) != 3 or len(slice_right_def) != 3:
                return self.none(node, inspect.currentframe().f_lineno)
            if slice_left_def[2].tolist() != slice_right_def[2].tolist():
                # axis are different
                return self.none(node, inspect.currentframe().f_lineno)
            lengths = {len(v) for v in slice_left_def} | {
                len(v) for v in slice_right_def
            }
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
            if len(set([axis, axis1, axis2])) != 1:
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

    @classmethod
    def apply_concat(
        cls,
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
            name=f"{cls.__name__}--{node.name}",
        )

        # We still keep the constant in case other node use it.
        # The algorithm removing the unused nodes will decide
        # whether or not to keep it.

        if is_split:
            assert slice_left is None and slice_right is None
            return [cst_left, split, neg, concat]

        return [cst_left, slice_left, slice_right, neg, concat]

    def match_scatter(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Add" or node.domain != "":
            return self.none()
        if g.is_used_more_than_once(node.input[0]) or g.is_used_more_than_once(
            node.input[1]
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        scatter_left, scatter_right = g.node_before(node.input[0]), g.node_before(
            node.input[1]
        )
        if scatter_left is None or scatter_right is None:
            return self.none(node, inspect.currentframe().f_lineno)

        cst_left = g.node_before(scatter_left.input[0])
        cst_right = g.node_before(scatter_right.input[0])
        if (
            cst_left.op_type != "ConstantOfShape"
            or cst_right.op_type != "ConstantOfShape"
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        slice_left = g.node_before(scatter_left.input[2])
        slice_right = g.node_before(scatter_right.input[2])
        if slice_left.op_type not in ("Slice", "Neg") or slice_right.op_type not in (
            "Slice",
            "Neg",
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        if (
            slice_left.op_type == "Neg"
            and g.node_before(slice_left.input[0]).op_type != "Slice"
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        if (
            slice_right.op_type == "Neg"
            and g.node_before(slice_right.input[0]).op_type != "Slice"
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

        nodes = [
            cst_left,
            slice_left,
            neg_left,
            cst_right,
            slice_right,
            neg_right,
            node,
        ]

        # still returning None for the time being
        if nodes:
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, nodes, self.apply_scatter)
