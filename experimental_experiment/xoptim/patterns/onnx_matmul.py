from typing import List, Optional
from onnx import NodeProto
from ...xbuilder.annotations import compatible_shapes, compatible_dimensions
from .patterns_api import MatchResult, PatternOptimization


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
