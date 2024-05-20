import inspect
from typing import List, Optional, Tuple, Union
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization


class TransposeTransposePattern(PatternOptimization):
    """
    Removes two consecutive transpose if the second one put the tensor in origin shape.
    """

    def __init__(self, verbose: int = 0, priority: int = 0):
        super(TransposeTransposePattern, self).__init__(verbose, priority)

    @classmethod
    def apply_transpose(
        cls, perm: Tuple[int, ...], on: List[Union[int, str]]
    ) -> List[Union[int, str]]:
        assert len(perm) == len(on), "length mismatch"
        res = [None for i in on]
        for i, p in enumerate(perm):
            res[i] = on[p]
        return res

    @classmethod
    def apply_transposes(
        cls, perms: List[Tuple[int, ...]], on: Optional[List[Union[int, str]]] = None
    ) -> List[Union[int, str]]:
        if on is None:
            on = list(range(len(perms[0])))
        for p in perms:
            on = cls.apply_transpose(p, on)
        return on

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Transpose" or node.domain != "":
            return self.none()
        next_nodes = g.next_nodes(node.output[0])
        next_node = None
        for n in next_nodes:
            if n.op_type == "Transpose":
                next_node = n
        if next_node is None:
            return self.none(node, inspect.currentframe().f_lineno)

        # Three consecutive transpose are not expected but let's continue
        # as if it could be possible.
        nodes = [node, next_node]
        perms = [tuple(g.get_attribute(n, "perm").ints) for n in nodes]
        lens = [len(p) for p in perms]
        assert min(lens) == max(lens), (
            f"Consecutive Transpose should apply on tensors with "
            f"the same rank but perms={perms}."
        )
        first = list(range(lens[0]))
        last = self.apply_transposes(perms)
        if last != first and g.is_used_more_than_once(node.output[0]):
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [node, next_node], self.apply)

    def apply(
        self, g: "GraphBuilder", node: NodeProto, next_node: NodeProto  # noqa: F821
    ) -> List[NodeProto]:

        perms = [tuple(g.get_attribute(n, "perm").ints) for n in [node, next_node]]
        first = list(range(len(perms[0])))
        last = self.apply_transposes(perms)
        if first == last:
            new_node = g.make_node(
                "Identity",
                [node.input[0]],
                next_node.output,
                name=f"{self.__class__.__name__}--{node.name}",
                doc_string=next_node.doc_string,
            )
        else:
            new_node = g.make_node(
                "Transpose",
                [node.input[0]],
                next_node.output,
                perm=last,
                name=f"{self.__class__.__name__}--{node.name}",
                doc_string=next_node.doc_string,
            )
        new_nodes = [new_node]
        if g.is_used_more_than_once(node.output[0]):
            new_nodes.append(node)
        return new_nodes
