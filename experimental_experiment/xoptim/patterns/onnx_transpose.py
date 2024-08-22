import inspect
from typing import List, Optional, Tuple, Union
import numpy as np
from onnx import NodeProto
from ...xbuilder._shape_helper import is_static_shape
from ..patterns_api import MatchResult, PatternOptimization


class TransposeTransposePattern(PatternOptimization):
    """
    Removes two consecutive transpose if the second one put the tensor in origin shape.
    """

    def __init__(self, verbose: int = 0, priority: int = 0):
        super().__init__(verbose, priority)

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
        self,
        g: "GraphBuilder",  # noqa: F821
        node: NodeProto,
        next_node: NodeProto,
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


class TransposeReshapeTransposePattern(PatternOptimization):
    """
    Swaps Reshapes and Transpose in a sequence such as this one:

    ::

        input is 32x4x14x4x14x128

        Transpose(., perm=[0, 1, 3, 2, 4, 5])
        Reshape(., 32x56x56x128)
        Transpose(., perm=[0, 3, 1, 2])

    By:

    ::

        Transpose(., perm=[0, 1, 3, 2, 4, 5])
        Transpose(., perm=[0, 5, 1, 2, 3, 4])
        Reshape(., 32x128x56x56)
    """

    def __init__(self, verbose: int = 0, priority: int = 0):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Transpose" or node.domain != "":
            return self.none()

        next_nodes = g.next_nodes(node.output[0])
        if len(next_nodes) != 1:
            return self.none(node, inspect.currentframe().f_lineno)

        reshape = next_nodes[0]
        if reshape.op_type != "Reshape" or reshape.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant(reshape.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)

        next_nodes = g.next_nodes(reshape.output[0])
        if len(next_nodes) != 1:
            return self.none(node, inspect.currentframe().f_lineno)

        transpose = next_nodes[0]
        if transpose.op_type != "Transpose" or transpose.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        resh_tr = self._new_shape_perm(g, node, reshape, transpose)

        if resh_tr is None:
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [node, reshape, transpose], self.apply)

    def _new_shape_perm(
        self,
        g: "GraphBulder",  # noqa: F821
        t1_node: NodeProto,
        reshape_node: NodeProto,
        t2_node: NodeProto,
    ) -> Optional[Tuple[Tuple[int, ...], List[int]]]:
        p1 = list(g.get_attribute(t1_node, "perm").ints)
        p2 = list(g.get_attribute(t2_node, "perm").ints)
        if len(p2) > len(p1):
            return None
        shape = g.get_computed_constant(reshape_node.input[1]).tolist()
        if not is_static_shape(shape):
            return None
        raise AssertionError("Not implemented yet.")

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        t1_node: NodeProto,
        reshape_node: NodeProto,
        t2_node: NodeProto,
    ) -> List[NodeProto]:
        new_perm, new_shape = self._new_shape_perm(g, t1_node, reshape_node, t2_node)
        new_name = g.unique_name(f"{self.__class__.__name__}_{t1_node.output[0]}")
        new_shape_name = g.make_initializer("", np.array(new_shape, dtype=np.int64))
        return [
            t1_node,
            g.make_node(
                "Transpose",
                [t1_node.output[0]],
                [new_name],
                perm=new_perm,
                name=f"{self.__class__.__name__}--{t2_node.name}",
                doc_string=t2_node.doc_string,
            ),
            g.make_node(
                "Reshape",
                [new_name, new_shape_name],
                t2_node.output,
                perm=new_perm,
                name=f"{self.__class__.__name__}--{reshape_node.name}",
                doc_string=reshape_node.doc_string,
            ),
        ]
