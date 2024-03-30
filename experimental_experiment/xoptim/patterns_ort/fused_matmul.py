import inspect
from typing import List, Optional
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization


class FusedMatMulPattern(PatternOptimization):
    """
    Replaces the sequence Transpose, Matmul into FusedMatMul.
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if (node.op_type != "MatMul" or node.domain != "") and (
            node.op_type != "FusedMatMul" or node.domain != "com.microsoft"
        ):
            return self.none()

        if node.op_type == "FusedMatMul":
            transA = g.get_attribute(node, "transA", exc=False) or 0
            transB = g.get_attribute(node, "transB", exc=False) or 0
            if transA != transB:
                # one side is already transposed.
                return self.none(node, inspect.currentframe().f_lineno)

        if not g.has_rank(node.input[0]) or not g.has_rank(node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        if g.get_rank(node.input[0]) < 2 or g.get_rank(node.input[1]) < 2:
            return self.none(node, inspect.currentframe().f_lineno)
        if g.get_rank(node.input[0]) <= 2 and g.get_rank(node.input[1]) <= 2:
            # Regular Gemm.
            return self.none(node, inspect.currentframe().f_lineno)

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
            return self.none(node, inspect.currentframe().f_lineno)

        for n in ns:
            if n is None:
                continue
            perm = list(g.get_attribute(n, "perm").ints)
            expecting = list(range(len(perm)))
            expecting[-2], expecting[-1] = expecting[-1], expecting[-2]
            if perm != expecting:
                # unexpected transpose
                return self.none(node, inspect.currentframe().f_lineno)

        # At this stage, one or two inputs are transposed before being used.
        # MatMul or Gemm are operating on 2D tensors.
        nodes = [*ns, node]
        if nodes[0] is not None and nodes[1] is not None:
            # Both are available, we only transpose one.
            nodes[0] = None
        if not g.is_used_more_than_once(node.output[0]):
            next_node = g.next_node(node.output[0])
            if (
                next_node.op_type in {"Div", "Mul"}
                and next_node.domain == ""
                and g.is_constant_scalar(next_node.input[1])
            ):
                # The node can be fused with matmul
                nodes.append(next_node)

        return MatchResult(self, nodes, self.apply)

    @classmethod
    def apply(
        cls,
        g: "GraphBuilder",  # noqa: F821
        node_before_left: Optional[NodeProto],
        node_before_right: Optional[NodeProto],
        node: NodeProto,
        scale: Optional[NodeProto] = None,
    ) -> List[NodeProto]:

        inputs = [
            (node.input[0] if node_before_left is None else node_before_left.input[0]),
            (
                node.input[1]
                if node_before_right is None
                else node_before_right.input[0]
            ),
            *node.input[2:],
        ]

        transA = 0 if node_before_left is None else 1
        transB = 0 if node_before_right is None else 1
        transBatchA = 0
        transBatchB = 0
        keep = []
        for att in node.attribute:
            if att.name in {"alpha", "beta"}:
                keep.append(att)
            elif att.name == "transA":
                transA = (att.i + transA) % 2
            elif att.name == "transB":
                transB = (att.i + transB) % 2
            elif att.name == "transBatchA":
                transBatchA = att.i
            elif att.name == "transBatchB":
                transBatchB = att.i
            else:
                raise NotImplementedError(
                    f"Unexpected attribute {att.name!r}={att} for node={node}"
                )

        kwargs = dict(
            transA=transA,
            transB=transB,
            transBatchA=transBatchA,
            transBatchB=transBatchB,
        )

        if scale is not None:
            # Let's include the scale as well
            cst = g.get_computed_constant(scale.input[1])
            value = float(cst[0] if cst.shape == (1,) else cst)
            assert scale.op_type in {
                "Div",
                "Mul",
            }, f"Match did not check next_node type {scale.op_type!r}"
            alpha = value if scale.op_type == "Mul" else (1.0 / value)
            kwargs["alpha"] = alpha
            output = scale.output[0]
        else:
            output = node.output[0]

        new_node = g.make_node(
            "FusedMatMul",
            inputs,
            [output],
            name=f"{cls.__name__}--{node.name}",
            doc_string=node.doc_string,
            domain="com.microsoft",
            **kwargs,
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
