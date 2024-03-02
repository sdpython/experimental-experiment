from typing import List, Optional
from onnx import NodeProto
from ..patterns.patterns_api import MatchResult, PatternOptimization


class FusedMatMulPattern(PatternOptimization):
    """
    Replaces the sequence Transpose, Matmul or Gemm into Gemm
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
            return None
        if not g.has_rank(node.input[0]) or not g.has_rank(node.input[1]):
            return None
        if g.get_rank(node.input[0]) < 2 or g.get_rank(node.input[1]) < 2:
            return None
        if g.get_rank(node.input[0]) <= 2 and g.get_rank(node.input[1]) <= 2:
            # Regular Gemm.
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
            perm = list(g.get_attribute(n, "perm").ints)
            expecting = list(range(len(perm)))
            expecting[-2], expecting[-1] = expecting[-1], expecting[-2]
            if perm != expecting:
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

            new_node = g.make_node(
                "FusedMatMul",
                inputs,
                node.output,
                name=f"{self.__class__.__name__}--{node.name}",
                transA=transA,
                transB=transB,
                transBatchA=transBatchA,
                transBatchB=transBatchB,
                doc_string=node.doc_string,
                domain="com.microsoft",
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
