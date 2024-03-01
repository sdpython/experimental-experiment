from typing import List, Optional, Tuple
import numpy as np
from onnx import NodeProto
from .annotations import all_int, compatible_shapes, compatible_dimensions
from .optimization_patterns_api import MatchResult, PatternOptimization


class CastPattern(PatternOptimization):
    """
    Checks that a Cast is really needed.
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Cast" or node.domain != "":
            return None
        if not g.has_type(node.input[0]):
            itype = g.try_infer_type(node.input[0])
            if itype == 0:
                return None
        else:
            itype = g.get_type(node.input[0])
        att = g.get_attribute(node, "to")
        if att.i != itype:
            return None

        def apply(g: "GraphBuilder", node: NodeProto) -> List[NodeProto]:  # noqa: F821
            new_node = g.make_node(
                "Identity",
                node.input,
                node.output,
                name=f"{self.__class__.__name__}--{node.name}",
                doc_string=node.doc_string,
            )
            return [new_node]

        return MatchResult(self, [node], apply)


class ExpandPattern(PatternOptimization):
    """
    Checks that a Expand is really needed.
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Expand" or node.domain != "":
            return None
        if not g.has_shape(node.input[0]):
            return None
        shape = g.get_shape(node.input[0])
        if not all_int(shape):
            return None
        if not g.is_constant(node.input[1]):
            # It may be a symbolic shape.
            return None
        new_shape = tuple(g.get_computed_constant(node.input[1]))
        if shape != new_shape:
            return

        def apply(g: "GraphBuilder", node: NodeProto) -> List[NodeProto]:  # noqa: F821
            new_node = g.make_node(
                "Identity",
                node.input,
                node.output,
                name=f"{self.__class__.__name__}--{node.name}",
                doc_string=node.doc_string,
            )
            return [new_node]

        return MatchResult(self, [node], apply)


class MulMulMulPattern(PatternOptimization):
    """
    Replaces the sequence {Div | Mul} and  {Div | Mul} + {Div | Mul} with {Div | Mul} Mul.
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type not in {"Div", "Mul"} or node.domain != "":
            return None
        if g.is_used_more_than_once(node.input[0]) or g.is_used_more_than_once(
            node.input[1]
        ):
            return None
        node_left = g.node_before(node.input[0])
        if (
            node_left is None
            or node_left.op_type not in {"Div", "Mul"}
            or node.domain != ""
        ):
            return None
        node_right = g.node_before(node.input[1])
        if (
            node_right is None
            or node_right.op_type not in {"Div", "Mul"}
            or node.domain != ""
        ):
            return None

        # checking for the constant (right)
        if not g.is_constant(node_left.input[1]) or not g.is_constant(
            node_right.input[1]
        ):
            return None
        cst_left = g.get_computed_constant(node_left.input[1])
        cst_right = g.get_computed_constant(node_right.input[1])
        if cst_left.shape not in {tuple(), (1,)} or cst_right.shape not in {
            tuple(),
            (1,),
        }:
            return None

        nodes = [node, node_left, node_right]

        def apply(
            g: "GraphBuilder",  # noqa: F821
            node: NodeProto,
            node_left: NodeProto,
            node_right: NodeProto,
        ) -> List[NodeProto]:

            new_node = g.make_node(
                node.op_type,
                [node_left.input[0], node_right.input[0]],
                [g.unique_name(f"{self.__class__.__name__}--{node.output[0]}")],
                name=f"{self.__class__.__name__}--{node.name}",
            )
            cst_left = g.get_computed_constant(node_left.input[1])
            cst_right = g.get_computed_constant(node_right.input[1])
            if node_left.op_type == "Div":
                cst_left = np.reciprocal(cst_left)
            if node_right.op_type == "Div":
                cst_right = np.reciprocal(cst_right)

            if not isinstance(cst_left, np.ndarray):
                cst_left = np.array(cst_left)
            if not isinstance(cst_right, np.ndarray):
                cst_right = np.array(cst_right)
            assert (
                cst_left.dtype == cst_right.dtype
            ), f"Type mismatch left is {cst_left.dtype}, right is {cst_right.dtype}"
            new_value = cst_left * cst_right
            if not isinstance(new_value, np.ndarray):
                new_value = np.array(new_value)
            new_cst = g.make_initializer("", new_value)

            new_node2 = g.make_node(
                "Mul",
                [new_node.output[0], new_cst],
                node.output,
                name=f"{self.__class__.__name__}--{node.name}-Cst",
            )

            return [new_node, new_node2]

        return MatchResult(self, nodes, apply)


class ReshapeMatMulReshapePattern(PatternOptimization):
    """
    Replaces the sequence Reshape, Matmul, Reshape by Matmul.
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "MatMul" or node.domain != "":
            return None
        if g.is_used_more_than_once(node.output[0]):
            return None

        next_nodes = g.next_nodes(node.output[0])
        if len(next_nodes) == 0:
            return None
        next_node = next_nodes[0]
        if next_node.op_type != "Reshape" or node.domain != "":
            return None

        node_before_left = g.node_before(node.input[0])
        node_before_right = g.node_before(node.input[1])
        if node_before_left is None or node_before_right is None:
            return None
        if (
            node_before_left.op_type != "Reshape"
            or node_before_left.domain != ""
            or node_before_right.op_type != "Reshape"
            or node_before_right.domain != ""
        ):
            return None

        # condition on shapes
        if not g.is_constant(node_before_left.input[1]):
            return
        shape_left = tuple(
            int(i) for i in g.get_computed_constant(node_before_left.input[1])
        )
        if not g.is_constant(node_before_right.input[1]):
            return
        shape_right = tuple(
            int(i) for i in g.get_computed_constant(node_before_right.input[1])
        )
        if not g.is_constant(next_node.input[1]):
            return
        shape_final = tuple(int(i) for i in g.get_computed_constant(next_node.input[1]))
        if len(shape_final) < 4:
            return None
        ndim = len(shape_final)
        if len(shape_left) != 3 or len(shape_right) != 3:
            return None

        mshape_left = g.get_shape(node_before_left.input[0])
        mshape_right = g.get_shape(node_before_right.input[0])
        if len(mshape_left) != ndim or len(mshape_right) != ndim:
            return None
        if (
            not compatible_shapes(mshape_left[-2:], shape_left[-2:])
            or not compatible_shapes(mshape_right[-2:], shape_right[-2:])
            or not compatible_dimensions(
                mshape_left[-1], shape_left[-1], mshape_right[-2], shape_right[-2]
            )
        ):
            return None

        # At this stage, both Reshape before MatMul reduces the rank by 1
        # without changing the two last dimensions
        # and the Reshape after restores it. They can safely be removed.
        if g.verbose > 3:
            print(
                f"[ReshapeMatMulReshapePattern] compatible shapes: mshape_left={mshape_left} "
                f"shape_left={shape_left} | mshape_left={mshape_right} shape_left={shape_right}"
            )

        def apply(
            g: "GraphBuilder",  # noqa: F821
            node_before_left: NodeProto,
            node_before_right: NodeProto,
            node: NodeProto,
            next_node: NodeProto,
        ) -> List[NodeProto]:
            new_node = g.make_node(
                "MatMul",
                [node_before_left.input[0], node_before_right.input[0]],
                next_node.output,
                name=f"{self.__class__.__name__}--{node.name}",
                doc_string=next_node.doc_string,
            )
            res = [new_node]
            if g.is_used_more_than_once(node_before_left.output[0]):
                res.append(node_before_left)
            if g.is_used_more_than_once(node_before_right.output[0]):
                res.append(node_before_right)
            return res

        return MatchResult(
            self,
            [node_before_left, node_before_right, node, next_node],
            apply,
            insert_at=node,
        )


class ReshapeReshapePattern(PatternOptimization):
    """
    Replaces the sequence Reshape, Reshape by Reshape.
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Reshape" or node.domain != "":
            return None
        if g.is_used_more_than_once(node.output[0]):
            return None
        next_nodes = g.next_nodes(node.output[0])
        if len(next_nodes) != 1:
            return None
        next_node = next_nodes[0]
        if next_node.op_type != "Reshape" or node.domain != "":
            return None
        if next_node.input[0] != node.output[0]:
            return None

        def apply(
            g: "GraphBuilder", node: NodeProto, next_node: NodeProto  # noqa: F821
        ) -> List[NodeProto]:
            new_node = g.make_node(
                "Reshape",
                [node.input[0], next_node.input[1]],
                next_node.output,
                name=f"{self.__class__.__name__}--{node.name}",
                doc_string=next_node.doc_string,
            )
            return [new_node]

        return MatchResult(self, [node, next_node], apply)


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


class Sub1MulPattern(PatternOptimization):
    """
    Replaces the sequence `(1 - X) x Y`  by `Y - X x Y` to avoid the creation
    of a constant in the graph. `x` means element wise multiplication.
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Mul" or node.domain != "":
            return None
        if g.is_used_more_than_once(node.input[0]) or g.is_used_more_than_once(
            node.input[1]
        ):
            return None
        node_left = g.node_before(node.input[0])
        node_right = g.node_before(node.input[1])
        op_left = None if node_left is None else node_left.op_type
        op_right = None if node_right is None else node_right.op_type
        if op_left != "Sub" and op_right != "Sub":
            return None
        cst_left, cst_right = None, None
        if op_left == "Sub" and g.is_constant(node_left.input[0]):
            cst = g.get_computed_constant(node_left.input[0])
            if cst.min() == cst.max() == 1:
                cst_left = cst

        if op_right == "Sub" and g.is_constant(node_right.input[0]):
            cst = g.get_computed_constant(node_right.input[0])
            if cst.min() == cst.max() == 1:
                cst_right = cst

        if cst_left is None and cst_right is None:
            print(g.builder._known_shapes)
            return None

        nodes = [node, node_left, node_right]

        def apply(
            g: "GraphBuilder",  # noqa: F821
            node: NodeProto,
            node_left: NodeProto,
            node_right: NodeProto,
        ) -> List[NodeProto]:

            cst_left = None
            if op_left == "Sub" and g.is_constant(node_left.input[0]):
                cst = g.get_computed_constant(node_left.input[0])
                if cst.min() == cst.max() == 1:
                    cst_left = cst

            if cst_left is not None:
                # rewrite `(1 - X) x Y` into `Y - X x Y`
                mul_node = g.make_node(
                    "Mul",
                    [node_left.input[1], node.input[1]],
                    [g.unique_name(f"{self.__class__.__name__}--{node.output[0]}")],
                    name=f"{self.__class__.__name__}--{node.name}",
                )
                sub_node = g.make_node(
                    "Sub",
                    [node.input[1], mul_node.output[0]],
                    node.output,
                    name=f"{self.__class__.__name__}--{node.name}",
                    doc_string=node.doc_string,
                )
                keep_node = node_right
            else:
                # rewrite `Y x (1 - X)` into `Y - (Y - X)`
                mul_node = g.make_node(
                    "Mul",
                    [node.input[0], node_right.input[1]],
                    [g.unique_name(f"{self.__class__.__name__}--{node.output[0]}")],
                    name=f"{self.__class__.__name__}--{node.name}",
                )
                sub_node = g.make_node(
                    "Sub",
                    [node.input[0], mul_node.output[0]],
                    node.output,
                    name=f"{self.__class__.__name__}--{node.name}",
                    doc_string=node.doc_string,
                )
                keep_node = node_left

            return [keep_node, mul_node, sub_node]

        return MatchResult(self, nodes, apply)


class TransposeMatMulPattern(PatternOptimization):
    """
    Replaces the sequence Transpose, Matmul or Gemm into Gemm
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type not in {"MatMul", "Gemm"} or node.domain != "":
            return None
        if not g.has_rank(node.input[0]) or not g.has_rank(node.input[1]):
            return None
        if g.get_rank(node.input[0]) != 2 or g.get_rank(node.input[1]) != 2:
            return None

        nodes_before = [g.node_before(node.input[0]), g.node_before(node.input[1])]
        ns = [
            (
                n
                if n is not None and n.op_type == "Transpose" and n.domain == ""
                else None
            )
            for n in nodes_before
        ]
        if len([_ for _ in ns if _ is not None]) == 0:
            return None

        for n in ns:
            if n is None:
                continue
            perm = tuple(g.get_attribute(n, "perm").ints)
            if perm != (1, 0):
                # unexpected transpose
                return None

        # At this stage, one or two inputs are transposed before being used.
        # MatMul or Gemm are operating on 2D tensors.
        nodes = [*ns, node]

        def apply(
            g: "GraphBuilder",  # noqa: F821
            node_before_left: Optional[NodeProto],
            node_before_right: Optional[NodeProto],
            nodes: NodeProto,
        ) -> List[NodeProto]:

            inputs = [
                (
                    node.input[0]
                    if node_before_left is None
                    else node_before_left.input[0]
                ),
                (
                    node.input[1]
                    if node_before_right is None
                    else node_before_right.input[0]
                ),
                *node.input[2:],
            ]

            transA = 0 if node_before_left is None else 1
            transB = 0 if node_before_right is None else 1
            keep = []
            for att in node.attribute:
                if att.name in {"alpha", "beta"}:
                    keep.append(att)
                elif att.name == "transA":
                    transA = (att.i + transA) % 2
                elif att.name == "transB":
                    transB = (att.i + transB) % 2
                else:
                    raise NotImplementedError(
                        f"Unexpected attribute {att.name!r}={att} for node={node}"
                    )

            new_node = g.make_node(
                "Gemm",
                inputs,
                node.output,
                name=f"{self.__class__.__name__}--{node.name}",
                transA=transA,
                transB=transB,
                doc_string=node.doc_string,
            )
            new_node.attribute.extend(keep)
            res = [new_node]
            if node_before_left is not None and g.is_used_more_than_once(
                node_before_left.output[0]
            ):
                res.append(node_before_left)
            if node_before_right is not None and g.is_used_more_than_once(
                node_before_right.output[0]
            ):
                res.append(node_before_right)
            return res

        return MatchResult(self, nodes, apply, insert_at=node)


class TransposeTransposePattern(PatternOptimization):
    """
    Removes two consecutive transpose if the second one put the tensor in origin shape.
    """

    @classmethod
    def apply_transpose(cls, perm: Tuple[int, ...], on: List[int]) -> List[int]:
        assert len(perm) == len(on), "length mismatch"
        res = [None for i in on]
        for i, p in enumerate(perm):
            res[i] = p
        return res

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Transpose" or node.domain != "":
            return None
        next_nodes = g.next_nodes(node.output[0])
        next_node = None
        for n in next_nodes:
            if n.op_type == "Transpose":
                next_node = n
        if next_node is None:
            return None

        # Three consecutive transpose are not expected but let's continue
        # as if it could be possible.
        nodes = [node, next_node]
        perms = [tuple(g.get_attribute(n, "perm").ints) for n in nodes]
        lens = [len(p) for p in perms]
        assert min(lens) == max(lens), (
            f"Consecutive Transpose should apply on tensors with "
            f"the same rank but perms={perms}."
        )
        on = list(range(lens[0]))
        first = on.copy()
        for p in perms:
            self.apply_transpose(p, on)
        if on != first:
            return None

        def apply(
            g: "GraphBuilder", node: NodeProto, next_node: NodeProto  # noqa: F821
        ) -> List[NodeProto]:
            new_nodes = [
                g.make_node(
                    "Identity",
                    [node.input[0]],
                    next_node.output,
                    name=f"{self.__class__.__name__}--{node.name}",
                    doc_string=next_node.doc_string,
                )
            ]
            if g.is_used_more_than_once(node.output[0]):
                new_nodes.append(node)
            return new_nodes

        return MatchResult(self, [node, next_node], apply)


class UnsqueezeUnsqueezePattern(PatternOptimization):
    """
    Replaces the sequence Unsqueeze, Unsqueeze by Unsqueeze.
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Unsqueeze" or node.domain != "":
            return None
        if g.is_used_more_than_once(node.output[0]):
            return None
        next_nodes = g.next_nodes(node.output[0])
        if len(next_nodes) != 1:
            return None
        next_node = next_nodes[0]
        if next_node.op_type != "Unsqueeze" or node.domain != "":
            return None
        if next_node.input[0] != node.output[0]:
            return None

        def apply(
            g: "GraphBuilder", node: NodeProto, next_node: NodeProto  # noqa: F821
        ) -> List[NodeProto]:
            axis1 = g.get_constant_or_attribute(node, "axis", 1)
            axis2 = g.get_constant_or_attribute(next_node, "axis", 1)
            new_axis = g.make_initializer("", np.hstack([axis1, axis2]))
            new_node = g.make_node(
                "Unsqueeze",
                [node.input[0], new_axis],
                next_node.output,
                name=f"{self.__class__.__name__}--{node.name}",
                doc_string=next_node.doc_string,
            )
            return [new_node]

        return MatchResult(self, [node, next_node], apply)
