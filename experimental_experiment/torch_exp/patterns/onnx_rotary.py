from typing import List, Optional
from onnx import NodeProto
from .patterns_api import MatchResult, PatternOptimization


class RotaryConcatPartPattern(PatternOptimization):
    """
    Optimizes the following sequence.

    .. runpython::

        import numpy as np
        from onnx import TensorProto
        from onnx_array_api.light_api import start
        from onnx_array_api.plotting.text_plot import onnx_simple_text_plot

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
        print(onnx_simple_text_plot(model))
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Add" or node.domain != "":
            return None
        if g.is_used_more_than_once(node.input[0]) or g.is_used_more_than_once(
            node.input[1]
        ):
            return None

        concat_left, concat_right = g.node_before(node.input[0]), g.node_before(
            node.input[1]
        )
        if concat_left is None or concat_right is None:
            return None
        if len(concat_left.input) != 2 or len(concat_right.input) != 2:
            return None

        concat_left_before = [g.node_before(i) for i in concat_left.input]
        concat_right_before = [g.node_before(i) for i in concat_right.input]
        if None in concat_left_before or None in concat_right_before:
            return None

        type_left = [n.op_type for n in concat_left_before]
        type_right = [n.op_type for n in concat_right_before]
        if "ConstantOfShape" not in type_left or "ConstantOfShape" not in type_right:
            return None
        if type_left.index("ConstantOfShape") == type_right.index("ConstantOfShape"):
            return None

        cst_left = [n for n in concat_left_before if n.op_type == "ConstantOfShape"][0]
        cst_right = [n for n in concat_right_before if n.op_type == "ConstantOfShape"][
            0
        ]
        if g.is_used_more_than_once(cst_left.output[0]) or g.is_used_more_than_once(
            cst_right.output[0]
        ):
            return None

        tl = [n for n in concat_right_before if n.op_type == "Neg"]
        if tl:
            neg_left = None
            neg_right = tl[0]
            slice_left = [n for n in concat_left_before if n.op_type == "Slice"][0]
            slice_right = g.node_before(neg_right.input[0])
        else:
            neg_left = [n for n in concat_left_before if n.op_type == "Neg"][0]
            neg_right = None
            slice_left = g.node_before(neg_left.input[0])
            slice_right = [n for n in concat_right_before if n.op_type == "Slice"][0]

        if slice_left.input[0] != slice_right.input[0]:
            return None

        slice_left_def = [g.get_computed_constant(i) for i in slice_left.input[1:]]
        slice_right_def = [g.get_computed_constant(i) for i in slice_right.input[1:]]
        if len(slice_left_def) != 3 or len(slice_right_def) != 3:
            return None
        if slice_left_def[2].tolist() != slice_right_def[2].tolist():
            # axis are different
            return None
        lengths = {len(v) for v in slice_left_def} | {len(v) for v in slice_right_def}
        if lengths != {1}:
            # more than one axis
            return None

        axis = slice_left_def[2][0]
        dim_left = slice_left_def[1][0] - slice_left_def[0][0]
        dim_right = slice_right_def[1][0] - slice_right_def[0][0]

        shape_left = g.get_computed_constant(cst_left.input[0])
        shape_right = g.get_computed_constant(cst_right.input[0])
        cdim_left = shape_left[axis]
        cdim_right = shape_right[axis]

        if dim_left != cdim_right or dim_right != cdim_left:
            return None

        nodes = [
            cst_left,
            slice_left,
            neg_left,
            concat_left,
            cst_right,
            slice_right,
            neg_right,
            concat_right,
            node,
        ]

        def apply(
            g: "GraphBuilder",  # noqa: F821
            cst_left: NodeProto,
            slice_left: NodeProto,
            neg_left: Optional[NodeProto],
            concat_left: NodeProto,
            cst_right: NodeProto,
            slice_right: NodeProto,
            neg_right: Optional[NodeProto],
            concat_right: NodeProto,
            node: NodeProto,
        ) -> List[NodeProto]:
            axis = g.get_computed_constant(slice_left.input[3])[0]
            if neg_left is None:
                neg_out = neg_right.output[0]
                pos = list(concat_right.input).index(neg_out)
                concat_inputs = (
                    [neg_right.output[0], slice_left.output[0]]
                    if pos == 0
                    else [slice_left.output[0], neg_right.output[0]]
                )
                neg = neg_right
            else:
                neg_out = neg_left.output[0]
                pos = list(concat_left.input).index(neg_out)
                concat_inputs = (
                    [neg_left.output[0], slice_right.output[0]]
                    if pos == 0
                    else [slice_right.output[0], neg_left.output[0]]
                )
                neg = neg_left
            concat = g.make_node(
                "Concat",
                concat_inputs,
                node.output,
                axis=int(axis),
                doc_string=node.doc_string,
            )
            return [slice_left, slice_right, neg, concat]

        return MatchResult(self, nodes, apply)
