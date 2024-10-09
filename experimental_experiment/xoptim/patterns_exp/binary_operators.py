import collections
import inspect
from typing import List, Optional
from onnx import NodeProto, TensorProto
from ..patterns_api import MatchResult, PatternOptimization


class _common:
    def __init__(self, broadcast: bool):
        self.broadcast = broadcast

    def __repr__(self) -> str:
        if self.broadcast:
            return f"{self.__class__.__name__}(broadcast=True)"
        return f"{self.__class__.__name__}()"

    @classmethod
    def _same_shape(
        cls,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        name1: str,
        name2: str,
        broadcast: bool = False,
    ) -> bool:
        if not g.has_shape(name1) or not g.has_shape(name2):
            return False
        if broadcast:
            sh1 = g.get_shape(name1)
            sh2 = g.get_shape(name2)
            if len(sh1) != len(sh2):
                rk = max(len(sh1), len(sh2))
                sh1 = (1,) * (rk - len(sh1)) + sh1
                sh2 = (1,) * (rk - len(sh2)) + sh2
            allow_one1 = True
            allow_one2 = True
            for a, b in zip(sh1, sh2):
                if a == b:
                    if a != 1:
                        allow_one1 = False
                    if b != 1:
                        allow_one2 = False
                    continue
                if a == 1 and allow_one1:
                    allow_one2 = False
                    continue
                if b == 1 and allow_one2:
                    allow_one1 = False
                    continue
                return False
            return True
        return g.get_shape(name1) == g.get_shape(name2)


class AddAddMulMulPattern(PatternOptimization, _common):
    """
    Replaces Add + Add by AddAdd or Mul + Mul by MulMul
    if they operate on the same shape.

    :param broadcast: allow broadcast on the first dimensions.
    """

    def __init__(self, verbose: int = 0, priority: int = 3, broadcast: bool = False):
        PatternOptimization.__init__(self, verbose, priority)
        _common.__init__(self, broadcast)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if not g.has_processor("CUDA"):
            return self.none()
        if node.op_type not in {"Add", "Mul"} or node.domain != "":
            return self.none()
        if g.get_type(node.input[0]) in {TensorProto.INT64, TensorProto.INT32}:
            return self.none()
        if not self._same_shape(g, *node.input, broadcast=self.broadcast):
            return self.none(node, inspect.currentframe().f_lineno)

        node_left = g.node_before(node.input[0])
        if (
            node_left is not None
            and not g.is_used_more_than_once(node.input[0])
            and node_left.op_type == node.op_type
            and self._same_shape(g, *node_left.input, broadcast=self.broadcast)
        ):
            return MatchResult(self, [node_left, None, node], self.apply, insert_at=node)

        node_right = g.node_before(node.input[1])
        if (
            node_right is not None
            and not g.is_used_more_than_once(node.input[1])
            and node_right.op_type == node.op_type
            and self._same_shape(g, *node_right.input, broadcast=self.broadcast)
        ):
            return MatchResult(self, [None, node_right, node], self.apply, insert_at=node)

        return self.none(node, inspect.currentframe().f_lineno)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node_left: NodeProto,
        node_right: NodeProto,
        node: NodeProto,
    ) -> List[NodeProto]:
        assert (
            node_left is not None or node_right is not None
        ), "node_left and node_right cannot be both None"
        if node_left is None:
            # node_right
            new_node = g.make_node(
                node.op_type * 2,
                [node.input[0], *node_right.input],
                node.output,
                domain="onnx_extended.ortops.optim.cuda",
                name=f"{self.__class__.__name__}--{node.name}",
            )
        else:
            new_node = g.make_node(
                node.op_type * 2,
                [
                    *node_left.input,
                    node.input[1],
                ],
                node.output,
                domain="onnx_extended.ortops.optim.cuda",
                name=f"{self.__class__.__name__}--{node.name}",
            )
        return [new_node]


class AddAddMulMulBroadcastPattern(AddAddMulMulPattern):
    """
    Replaces Add + Add by AddAdd or Mul + Mul by MulMul
    if they operate on the same shape.

    :param broadcast: allow broadcast on the first dimensions.
    """

    def __init__(self, verbose: int = 0, priority: int = 4, broadcast: bool = True):
        AddAddMulMulPattern.__init__(self, verbose, priority, broadcast)


class AddMulPattern(PatternOptimization, _common):
    """
    Replaces Add + Mul by AddMul or Mul + Add by MulAdd
    if they operate on the same shape.

    :param broadcast: allow broadcast on the first dimensions.
    """

    def __init__(self, verbose: int = 0, priority: int = 3, broadcast: bool = False):
        PatternOptimization.__init__(self, verbose, priority)
        _common.__init__(self, broadcast)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if not g.has_processor("CUDA"):
            return self.none()

        if node.op_type not in {"Add", "Mul"} or node.domain != "":
            return self.none()
        if g.get_type(node.input[0]) in {TensorProto.INT64, TensorProto.INT32}:
            return self.none()
        if not self._same_shape(g, *node.input, broadcast=self.broadcast):
            return self.none(node, inspect.currentframe().f_lineno)

        node_left = g.node_before(node.input[0])
        if (
            node_left is not None
            and not g.is_used_more_than_once(node.input[0])
            and node_left.op_type in {"Add", "Mul"}
            and node_left.op_type != node.op_type
            and self._same_shape(g, *node_left.input, broadcast=self.broadcast)
        ):
            return MatchResult(self, [node_left, None, node], self.apply, insert_at=node)

        node_right = g.node_before(node.input[1])
        if (
            node_right is not None
            and not g.is_used_more_than_once(node.input[1])
            and node_right.op_type in {"Add", "Mul"}
            and node_right.op_type != node.op_type
            and self._same_shape(g, *node_right.input, broadcast=self.broadcast)
        ):
            return MatchResult(self, [None, node_right, node], self.apply, insert_at=node)

        return self.none(node, inspect.currentframe().f_lineno)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node_left: NodeProto,
        node_right: NodeProto,
        node: NodeProto,
    ) -> List[NodeProto]:
        assert (
            node_left is not None or node_right is not None
        ), "node_left and node_right cannot be both None"
        if node_left is None:
            # node_right
            new_node = g.make_node(
                f"{node_right.op_type}{node.op_type}",
                [*node_right.input, node.input[0]],
                node.output,
                domain="onnx_extended.ortops.optim.cuda",
                name=f"{self.__class__.__name__}--{node.name}",
            )
        else:
            new_node = g.make_node(
                f"{node_left.op_type}{node.op_type}",
                [*node_left.input, node.input[1]],
                node.output,
                domain="onnx_extended.ortops.optim.cuda",
                name=f"{self.__class__.__name__}--{node.name}",
            )
        return [new_node]


class AddMulBroadcastPattern(AddMulPattern):
    """
    Replaces Add + Mul by AddMul or Mul + Add by MulAdd
    if they operate on the same shape.

    :param broadcast: allow broadcast on the first dimensions.
    """

    def __init__(self, verbose: int = 0, priority: int = 4, broadcast: bool = True):
        AddMulPattern.__init__(self, verbose, priority, broadcast)


class MulSigmoidPattern(PatternOptimization):
    """
    Replaces Mul + Sigmoid by MulSigmoid
    if they operate on the same input.
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if not g.has_processor("CUDA"):
            return self.none()
        if node.op_type not in {"Sigmoid"} or node.domain != "":
            return self.none()
        if g.get_type(node.input[0]) in {TensorProto.INT64, TensorProto.INT32}:
            return self.none()
        if g.is_used_more_than_once(node.output[0]):
            return self.none(node, inspect.currentframe().f_lineno)

        next_node = g.next_node(node.output[0])
        if next_node.op_type != "Mul" or node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        index = list(next_node.input).index(node.output[0])
        other_index = 1 - index
        if next_node.input[other_index] != node.input[0]:
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [node, next_node], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node_sigmoid: NodeProto,
        node_mul: NodeProto,
    ) -> List[NodeProto]:
        new_node = g.make_node(
            "MulSigmoid",
            node_sigmoid.input,
            node_mul.output,
            domain="onnx_extended.ortops.optim.cuda",
            name=f"{self.__class__.__name__}--{node_sigmoid.name}",
        )
        return [new_node]


class NegXplus1Pattern(PatternOptimization):
    """
    Replaces 1 - X by NegXplus1
    if they operate on the same input.
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if not g.has_processor("CUDA"):
            return self.none()
        if node.op_type not in {"Sub"} or node.domain != "":
            return self.none()
        if g.get_type(node.input[0]) in {TensorProto.INT64, TensorProto.INT32}:
            return self.none()

        if not g.is_constant(node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant_scalar(node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        cst = g.get_constant_scalar(node.input[0])
        if cst != 1:
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [node], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node: NodeProto,
    ) -> List[NodeProto]:
        new_node = g.make_node(
            "NegXplus1",
            node.input[1:],
            node.output,
            domain="onnx_extended.ortops.optim.cuda",
            name=f"{self.__class__.__name__}--{node.name}",
        )
        return [new_node]


class SubMulPattern(PatternOptimization, _common):
    """
    Replaces Sub + Mul by AddMul or Mul + Add by MulAdd
    if they operate on the same shape.

    :param broadcast: allow broadcast on the first dimensions.
    """

    def __init__(self, verbose: int = 0, priority: int = 3, broadcast: bool = False):
        PatternOptimization.__init__(self, verbose, priority)
        _common.__init__(self, broadcast)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if not g.has_processor("CUDA"):
            return self.none()
        if node.op_type not in {"Sub", "Mul"} or node.domain != "":
            return self.none()
        if g.get_type(node.input[0]) in {TensorProto.INT64, TensorProto.INT32}:
            return self.none()
        if not self._same_shape(g, *node.input, broadcast=self.broadcast):
            return self.none(node, inspect.currentframe().f_lineno)

        node_left = g.node_before(node.input[0])
        if (
            node_left is not None
            and not g.is_used_more_than_once(node.input[0])
            and node_left.op_type in {"Sub", "Mul"}
            and node_left.op_type != node.op_type
            and self._same_shape(g, *node_left.input, broadcast=self.broadcast)
        ):
            return MatchResult(self, [node_left, None, node], self.apply, insert_at=node)

        node_right = g.node_before(node.input[1])
        if (
            node_right is not None
            and not g.is_used_more_than_once(node.input[1])
            and node_right.op_type in {"Sub", "Mul"}
            and node_right.op_type != node.op_type
            and self._same_shape(g, *node_right.input, broadcast=self.broadcast)
        ):
            return MatchResult(self, [None, node_right, node], self.apply, insert_at=node)

        return self.none(node, inspect.currentframe().f_lineno)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node_left: NodeProto,
        node_right: NodeProto,
        node: NodeProto,
    ) -> List[NodeProto]:
        assert (
            node_left is not None or node_right is not None
        ), "node_left and node_right cannot be both None"
        if node_left is None:
            # node_right
            kwargs = {}
            if node.op_type == "Sub":
                kwargs["negative"] = 1
            new_node = g.make_node(
                f"{node_right.op_type}{node.op_type}",
                [*node_right.input, node.input[0]],
                node.output,
                domain="onnx_extended.ortops.optim.cuda",
                name=f"{self.__class__.__name__}--{node.name}",
                **kwargs,
            )
        else:
            kwargs = {}
            new_node = g.make_node(
                f"{node_left.op_type}{node.op_type}",
                [*node_left.input, node.input[1]],
                node.output,
                domain="onnx_extended.ortops.optim.cuda",
                name=f"{self.__class__.__name__}--{node.name}",
                **kwargs,
            )
        return [new_node]


class SubMulBroadcastPattern(SubMulPattern):
    """
    Replaces Add + Mul by AddMul or Mul + Add by MulAdd
    if they operate on the same shape.

    :param broadcast: allow broadcast on the first dimensions.
    """

    def __init__(self, verbose: int = 0, priority: int = 4, broadcast: bool = True):
        SubMulPattern.__init__(self, verbose, priority, broadcast)


class AddMulSharedInputPattern(PatternOptimization, _common):
    """
    Replaces Add(A, B) and Add(A, C) by AddSharedInput(A, B, C)
    if they operate on the same shape. Does the same for
    operator Mul.

    :param broadcast: allow broadcast on the first dimensions.
    """

    def __init__(self, verbose: int = 0, priority: int = 3, broadcast: bool = False):
        PatternOptimization.__init__(self, verbose, priority)
        _common.__init__(self, broadcast)

    def can_fuse(cls, g: "GraphBuilder", nodes: List[NodeProto]) -> bool:  # noqa: F821
        """
        Checks that one node if not using the output of another.
        """
        assert len(nodes) == 2, f"Not implemented for {len(nodes)} nodes."
        p1 = g.get_position(nodes[0])
        p2 = g.get_position(nodes[1])
        if p1 < p2:
            output = nodes[1].output[0]
            dont = nodes[0].output[0]
        else:
            output = nodes[0].output[0]
            dont = nodes[1].output
        predecessors = {output}
        stack = [output]
        while stack:
            get = stack.pop()
            node = g.node_before(get)
            if node is None:
                continue
            for i in node.input:
                if i == dont:
                    return False
                if i in predecessors:
                    continue
                predecessors.add(i)
                stack.append(i)
        return True

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if not g.has_processor("CUDA"):
            return self.none()
        if node.op_type not in {"Add", "Mul"} or node.domain != "":
            return self.none()
        if g.get_type(node.input[0]) in {TensorProto.INT64, TensorProto.INT32}:
            return self.none()
        if not self._same_shape(g, *node.input, broadcast=self.broadcast):
            return self.none(node, inspect.currentframe().f_lineno)

        cons_left = [n for n in g.next_nodes(node.input[0]) if n.op_type == node.op_type]
        if len(cons_left) == 2:
            ok = True
            for n in cons_left:
                if not self._same_shape(g, *n.input, broadcast=self.broadcast):
                    ok = False
                    break
            if ok and self.can_fuse(g, cons_left):
                return MatchResult(self, cons_left, self.apply)

        cons_right = [n for n in g.next_nodes(node.input[1]) if n.op_type == node.op_type]
        if len(cons_right) == 2:
            ok = True
            for n in cons_right:
                if not self._same_shape(g, *n.input, broadcast=self.broadcast):
                    ok = False
                    break
            if ok and self.can_fuse(g, cons_right):
                return MatchResult(self, cons_right, self.apply)

        return self.none(node, inspect.currentframe().f_lineno)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        *nodes: NodeProto,
    ) -> List[NodeProto]:
        assert len(nodes) == 2, f"not implemented for {len(nodes)} nodes"
        names = collections.Counter([*nodes[0].input, *nodes[1].input])
        assert len(names) == 3, f"Expects three distinct inputs not {names}"
        common_name = None
        other_names = []
        for k, v in names.items():
            if v == 2:
                assert not common_name, f"Expects three distinct inputs not {names}"
                common_name = k
                continue
            other_names.append(k)
        assert len(other_names) == 2, f"Expects three distinct inputs not {names}"

        new_node = g.make_node(
            f"{nodes[0].op_type}SharedInput",
            [common_name, *other_names],
            [nodes[0].output[0], nodes[1].output[0]],
            domain="onnx_extended.ortops.optim.cuda",
            name=f"{self.__class__.__name__}--{nodes[0].name}",
        )

        return [new_node]


class AddMulSharedInputBroadcastPattern(AddMulSharedInputPattern):
    """
    Replaces Add(A, B) and Add(A, C) by AddSharedInput(A, B, C)
    if they operate on the same shape. Does the same for
    operator Mul.

    :param broadcast: allow broadcast on the first dimensions.
    """

    def __init__(self, verbose: int = 0, priority: int = 4, broadcast: bool = True):
        AddMulSharedInputPattern.__init__(self, verbose, priority, broadcast)


class AddMulTransposePattern(PatternOptimization):
    """
    Replaces (AddMul|MulAdd) + Transpose by (AddMul|MulAdd)(., transposeMiddle=1)
    if it is possible.
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if (
            node.op_type not in {"AddMul", "MulAdd"}
            or node.domain != "onnx_extended.ortops.optim.cuda"
        ):
            return self.none()
        if g.is_used_more_than_once(node.output[0]):
            return self.none(node, inspect.currentframe().f_lineno)

        next_node = g.next_nodes(node.output[0])[0]
        if next_node.op_type != "Transpose" or next_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        perm = g.get_attribute(next_node, "perm").ints
        if len(perm) != 4:
            return self.none(node, inspect.currentframe().f_lineno)

        if list(perm) != [0, 2, 1, 3]:
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [node, next_node], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node: NodeProto,
        transpose_node: NodeProto,
    ) -> List[NodeProto]:
        att = g.get_attribute(node, "transposeMiddle", exc=False)
        if att is None or att.i == 0:
            return [
                g.make_node(
                    node.op_type,
                    node.input,
                    transpose_node.output,
                    transposeMiddle=1,
                    domain=node.domain,
                    name=f"{self.__class__.__name__}--{node.name}",
                )
            ]
        # Otherwise, the next transpose nullifies the one already set in node.
        return [
            g.make_node(
                node.op_type,
                node.input,
                transpose_node.output,
                domain=node.domain,
                name=f"{self.__class__.__name__}--{node.name}",
            )
        ]
