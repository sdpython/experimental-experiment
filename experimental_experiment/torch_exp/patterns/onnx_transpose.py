from typing import List, Optional, Tuple
from onnx import NodeProto
from .patterns_api import MatchResult, PatternOptimization


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
