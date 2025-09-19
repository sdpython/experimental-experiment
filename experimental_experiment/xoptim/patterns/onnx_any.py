import inspect
from typing import List, Optional
from onnx import NodeProto
from ...helpers import make_idn
from ..patterns_api import MatchResult, PatternOptimization


class SameChildrenPattern(PatternOptimization):
    """Checks there is no duplicated node doing the same than another one beside."""

    @classmethod
    def _cmp(cls, n1: NodeProto, n2: NodeProto) -> bool:
        "Compares two nodes and say if they are the same."
        assert make_idn(n1) != make_idn(
            n2
        ), f"Two nodes are the same not identical copies {n1}"
        if n1.input != n2.input:
            return False
        if len(n1.output) != len(n2.output):
            return False
        if len(n1.attribute) != len(n2.attribute):
            return False
        for att1, att2 in zip(n1.attribute, n2.attribute):
            if att1.name != att2.name:
                return False
            if att1.type != att2.type:
                return False
            if att1.SerializeToString() != att2.SerializeToString():
                return False
        return True

    def __init__(self, verbose: int = 0, priority: int = 0):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        next_nodes = g.next_nodes(node.output[0])
        if len(next_nodes) <= 1:
            return self.none()

        if len(next_nodes) == 2:
            n1, n2 = next_nodes
            if n1.op_type != n2.op_type or n1.op_type == "Identity":
                return self.none()
            if not self._cmp(n1, n2):
                return self.none(node, inspect.currentframe().f_lineno)
            nodes = [n1, n2]
        else:
            cp = {}
            for n in next_nodes:
                if n.op_type == "Identity":
                    continue
                if n.op_type in cp:
                    cp[n.op_type].append(n)
                else:
                    cp[n.op_type] = [n]
            nodes = []
            for v in cp.values():
                if len(v) <= 1:
                    continue
                if len(v) == 2:
                    n1, n2 = v
                    if not self._cmp(n1, n2):
                        continue
                    nodes.extend([n1, n2])
                    continue
                enough = False
                for i in range(len(v) - 1):
                    for j in range(i + 1, len(v)):
                        if self._cmp(v[i], v[j]):
                            nodes.extend([v[i], v[j]])
                            enough = True
                            break
                    if enough:
                        break
            if len(nodes) == 0:
                return self.none(node, inspect.currentframe().f_lineno)

        for i in range(0, len(nodes), 2):
            n1, n2 = nodes[i : i + 2]
            assert len(n1.output) > 0, "A node should not have no output in this pattern."
            assert (
                not g.has_type(n1.output[0])
                or not g.has_type(n2.output[0])
                or g.get_type(n1.output[0]) == g.get_type(n2.output[0])
            ), (
                f"Nodes n1 and n2 have different output type for outputs "
                f"{n1.output[0]!r}, {n2.output[0]}, and types "
                f"{g.get_type(n1.output[0])} != {g.get_type(n2.output[0])})"
            )
        assert "Identity" not in set(n.op_type for n in nodes), (
            f"Identity nodes should be covered by this pattern "
            f"{set(n.op_type for n in nodes)}, type={node.op_type!r}, "
            f"name={node.name!r}."
        )
        return MatchResult(self, nodes, self.apply)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        *nodes: NodeProto,
    ) -> List[NodeProto]:
        """
        The function receives pairs of nodes. We replace every odd node
        by an identity node.
        """
        assert (
            len(nodes) % 2 == 0
        ), f"Expecting an even number of nodes but len(nodes) == {len(nodes)}"
        new_nodes = []
        for i in range(0, len(nodes), 2):
            new_nodes.append(nodes[i])

            for o1, o2 in zip(nodes[i].output, nodes[i + 1].output):
                new_nodes.append(
                    g.make_node(
                        "Identity",
                        [o1],
                        [o2],
                        name=f"{self.__class__.__name__}--{nodes[i+1].name}",
                        doc_string=nodes[i + 1].doc_string,
                    )
                )
        return new_nodes


class IdentityPattern(PatternOptimization):
    """
    Replaces operator such as
    Div(X, 1), Mul(X, 1), Add(X, 0), Sub(X, 0), Transpose(X, [0, 1, 2, ...])
    by identity nodes.
    """

    @classmethod
    def _any_value_to_scalar(cls, cst):
        try:
            return float(cst)
        except TypeError:
            return complex(cst)

    @classmethod
    def _has_unique_value(cls, cst):
        return cst.min() == cst.max()

    def __init__(self, verbose: int = 0, priority: int = 0):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if (
            node.op_type not in {"Add", "Mul", "Div", "Sub", "Transpose", "Slice", "And", "Or"}
            or node.domain != ""
        ):
            return self.none()

        if node.op_type == "Slice":
            if len(node.input) == 5:
                steps = node.input[4]
                if (
                    not g.is_constant(steps)
                    or g.get_computed_constant(steps) is None
                    or set(g.get_computed_constant(steps)) != {1}
                ):
                    return self.none(node, inspect.currentframe().f_lineno)
            starts, ends, _axes = node.input[1:4]
            if (
                not g.is_constant(starts)
                or g.get_computed_constant(starts) is None
                or set(g.get_computed_constant(starts)) != {0}
            ):
                return self.none(node, inspect.currentframe().f_lineno)
            if (
                not g.is_constant(ends)
                or g.get_computed_constant(ends) is None
                or set(g.get_computed_constant(ends))
                != {9223372036854775807}  # this a value used by torch
            ):
                return self.none(node, inspect.currentframe().f_lineno)
            return MatchResult(self, [node], self.apply, insert_at=node)

        if node.op_type == "Transpose":
            perm = list(g.get_attribute(node, "perm").ints)
            expected = list(range(len(perm)))
            if perm != expected:
                return self.none(node, inspect.currentframe().f_lineno)
            return MatchResult(self, [node], self.apply, insert_at=node)

        assert len(node.input) == 2, (
            f"Unexpected number of inputs {len(node.input)} "
            f"for node type {node.op_type!r} and name {node.name!r}, "
            f"node.input={node.input}"
        )
        if g.is_constant(node.input[1]):
            if g.has_rank(node.input[1]) and g.get_rank(node.input[1]) > 1:
                # No need to go further.
                return self.none(node, inspect.currentframe().f_lineno)
            shape = g.get_constant_shape(node.input[1], exc=False)
            if shape is None:
                return self.none(node, inspect.currentframe().f_lineno)
            if shape in (tuple(), (1,)):
                # simple case
                if not g.is_constant_scalar(node.input[1]):
                    return self.none(node, inspect.currentframe().f_lineno)
                cst = g.get_constant_scalar(node.input[1])
                val = self._any_value_to_scalar(cst)
                if val == 0:
                    if node.op_type in {"Add", "Sub", "Or"}:
                        return MatchResult(self, [node], self.apply, insert_at=node)
                elif val is False:
                    if node.op_type in {"Or"}:
                        return MatchResult(self, [node], self.apply, insert_at=node)
                elif val == 1:
                    if node.op_type in {"Mul", "Div", "And"}:
                        return MatchResult(self, [node], self.apply, insert_at=node)
                elif val is True:
                    if node.op_type in {"And"}:
                        return MatchResult(self, [node], self.apply, insert_at=node)
            elif len(shape) == 1 and node.op_type in {"Add", "Mul", "Sub", "Div", "And", "Or"}:
                # less simple case, the tensor is multiplied on its last dimension.
                cst = g.get_computed_constant(node.input[1])
                if cst is None:
                    return self.none(node, inspect.currentframe().f_lineno)
                if not self._has_unique_value(cst):
                    return self.none(node, inspect.currentframe().f_lineno)
                unique = cst[0]
                if not g.has_shape(node.input[0]):
                    return self.none(node, inspect.currentframe().f_lineno)
                shape = g.get_shape(node.input[0])
                if shape[-1] != cst.shape[0]:
                    return self.none(node, inspect.currentframe().f_lineno)
                if node.op_type in {"Add", "Sub", "And"} and unique != 0:
                    return self.none(node, inspect.currentframe().f_lineno)
                if node.op_type in {"And"} and unique is not True:
                    return self.none(node, inspect.currentframe().f_lineno)
                if node.op_type in {"Mul", "Div", "Or"} and unique != 1:
                    return self.none(node, inspect.currentframe().f_lineno)
                if node.op_type in {"Or"} and unique is not False:
                    return self.none(node, inspect.currentframe().f_lineno)
                return MatchResult(self, [node], self.apply, insert_at=node)
        elif g.is_constant(node.input[0]):
            if g.has_rank(node.input[0]) and g.get_rank(node.input[0]) > 1:
                # No need to go further.
                return self.none(node, inspect.currentframe().f_lineno)
            shape = g.get_constant_shape(node.input[0], exc=False)
            if shape is None:
                return self.none(node, inspect.currentframe().f_lineno)
            if shape in (tuple(), (1,)):
                # simple case
                if not g.is_constant_scalar(node.input[0]):
                    return self.none(node, inspect.currentframe().f_lineno)
                cst = g.get_constant_scalar(node.input[0])
                val = self._any_value_to_scalar(cst)
                if val == 0:
                    if node.op_type in {"Add", "Or"}:
                        return MatchResult(self, [node], self.apply, insert_at=node)
                elif val is False:
                    if node.op_type in {"Or"}:
                        return MatchResult(self, [node], self.apply, insert_at=node)
                elif val == 1:
                    if node.op_type in {"Mul", "And"}:
                        return MatchResult(self, [node], self.apply, insert_at=node)
                elif val is True:
                    if node.op_type in {"And"}:
                        return MatchResult(self, [node], self.apply, insert_at=node)

        return self.none(node, inspect.currentframe().f_lineno)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node: NodeProto,
    ) -> List[NodeProto]:
        name = node.input[
            (
                1
                if g.is_constant(node.input[0])
                and g.has_shape(node.input[0])
                and g.get_shape(node.input[0]) in {(), (1,)}
                else 0
            )
        ]
        return [
            g.make_node(
                "Identity",
                [name],
                [node.output[0]],
                name=f"{self.__class__.__name__}--{node.name}",
            )
        ]
