import inspect
from typing import List, Optional, Tuple, Union
import numpy as np
from onnx import NodeProto
from ...xshape._shape_helper import is_static_shape
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
    Swaps Reshape and Transpose in a sequence such as this one:

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

    def _align_shape(
        self, shape: Tuple[int, ...], new_shape: Tuple[int, ...]
    ) -> Optional[List[Tuple[Tuple[int, ...], Tuple[int, ...]]]]:
        mapped: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []
        i, j = 0, 0
        while i < len(shape) and j < len(new_shape):
            if shape[i] == new_shape[j]:
                mapped.append(((i,), (j,)))
                i += 1
                j += 1
                continue

            ii, jj = [i], [j]
            s1 = shape[i]
            s2 = new_shape[j]
            while s1 != s2 and i < len(shape) and j < len(new_shape):
                if s1 < s2:
                    i += 1
                    assert i < len(shape), f"Unxpected index i={i}, shape={shape}"
                    s1 *= shape[i]
                    ii.append(i)
                else:
                    j += 1
                    assert j < len(new_shape), f"Unxpected index i={j}, shape={new_shape}"
                    s2 *= new_shape[j]
                    jj.append(j)

            if min(len(ii), len(jj)) != 1:
                return None

            mapped.append((tuple(ii), tuple(jj)))
            i += 1
            j += 1

        if i != len(shape) or j != len(new_shape):
            return None
        return mapped

    def _new_shape_perm(
        self,
        g: "GraphBulder",  # noqa: F821
        t1_node: NodeProto,
        reshape_node: NodeProto,
        t2_node: NodeProto,
    ) -> Optional[Tuple[Tuple[int, ...], List[int], bool]]:
        p1 = list(g.get_attribute(t1_node, "perm").ints)
        p2 = list(g.get_attribute(t2_node, "perm").ints)
        new_shape = g.get_computed_constant(reshape_node.input[1]).tolist()
        if not is_static_shape(new_shape):
            return None
        if -1 in new_shape:
            return None
        if not g.has_shape(reshape_node.input[0]):
            return None
        shape = g.get_shape(reshape_node.input[0])
        mapped = self._align_shape(shape, new_shape)
        if mapped is None:
            return None

        if len(p2) <= len(p1):
            # move the reshape after the next transpose
            if len(mapped) != len(p2):
                return None

            # mapping is done, build new permutation
            new_perm = []
            for p in p2:
                new_perm.extend(mapped[p][0])

            new_reshape = [0 for s in p2]
            for i, p in enumerate(p2):
                new_reshape[i] = new_shape[p]

            return new_perm, new_reshape, True

        # move the reshape before the previous transpose
        if len(mapped) != len(p1):
            return None

        # mapping is done, build new permutation and shape
        rev_p1 = [0 for _ in p1]
        for i, p in enumerate(p1):
            rev_p1[p] = i
        indices = []
        for p in rev_p1:
            indices.extend(mapped[p][1])
        new_reshape = [new_shape[i] for i in indices]
        rev_indices = [0 for _ in indices]
        for i, p in enumerate(indices):
            rev_indices[p] = i
        new_perm = rev_indices

        return new_perm, new_reshape, False

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        t1_node: NodeProto,
        reshape_node: NodeProto,
        t2_node: NodeProto,
    ) -> List[NodeProto]:
        new_perm, new_shape, after = self._new_shape_perm(g, t1_node, reshape_node, t2_node)
        new_name = g.unique_name(f"{self.__class__.__name__}_{t1_node.output[0]}")
        new_shape_name = g.make_initializer(
            "",
            np.array(new_shape, dtype=np.int64),
            source="TransposeReshapeTransposePattern.apply.new_shape_name",
        )
        if after:
            return [
                t1_node,
                g.make_node(
                    "Transpose",
                    [t1_node.output[0]],
                    [new_name],
                    perm=new_perm,
                    name=f"{self.__class__.__name__}--C--{t2_node.name}",
                    doc_string=t2_node.doc_string,
                ),
                g.make_node(
                    "Reshape",
                    [new_name, new_shape_name],
                    t2_node.output,
                    name=f"{self.__class__.__name__}--D--{reshape_node.name}",
                    doc_string=reshape_node.doc_string,
                ),
            ]

        return [
            g.make_node(
                "Reshape",
                [t1_node.input[0], new_shape_name],
                [new_name],
                name=f"{self.__class__.__name__}--A--{reshape_node.name}",
                doc_string=reshape_node.doc_string,
            ),
            g.make_node(
                "Transpose",
                [new_name],
                [t2_node.input[0]],
                perm=new_perm,
                name=f"{self.__class__.__name__}--B--{t1_node.name}",
                doc_string=t1_node.doc_string,
            ),
            t2_node,
        ]


class TransposeEqualReshapePattern(PatternOptimization):
    """
    Replaces a Transpose by a Reshape when switched dimensions are
    all equal to 1 but one.
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Transpose" or node.domain != "":
            return self.none()
        if not g.has_shape(node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        perms = list(enumerate(g.get_attribute(node, "perm").ints))
        first = None
        for i, p in perms:
            if i != p:
                break
            first = i
        last = None
        for i, p in reversed(perms):
            if i != p:
                break
            last = i
        begin = first + 1 if first is not None else 0
        end = last if last is not None else len(perms)
        shape = g.get_shape(node.input[0])
        not_one = 0
        for i in range(begin, end):
            if shape[i] != 1:
                not_one += 1
        if not_one > 1:
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, [node], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        transpose_node: NodeProto,
    ) -> List[NodeProto]:
        perms = list(enumerate(g.get_attribute(transpose_node, "perm").ints))
        shape = g.get_shape(transpose_node.input[0])
        new_shape = []
        for i, p in perms:
            if i == p:
                new_shape.append(0)
            elif shape[p] == 1:
                new_shape.append(1)
            else:
                new_shape.append(-1)
        return [
            g.make_node(
                "Reshape",
                [
                    transpose_node.input[0],
                    g.make_initializer(
                        "",
                        np.array(new_shape, dtype=np.int64),
                        source="TransposeEqualReshapePattern.apply.new_shape",
                    ),
                ],
                transpose_node.output,
                name=f"{self.__class__.__name__}--B--{transpose_node.name}",
            )
        ]
