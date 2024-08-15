import inspect
from enum import IntEnum
from typing import List, Optional
import numpy as np
from onnx import NodeProto
from ...xbuilder._shape_helper import DYNAMIC_SHAPE
from ..patterns_api import MatchResult, PatternOptimization


class MulMulMulScalarPattern(PatternOptimization):
    """
    Replaces the sequence {Div | Mul} and {Div | Mul} + {Div | Mul} with {Div | Mul} Mul.
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type not in {"Div", "Mul"} or node.domain != "":
            return self.none()
        if g.is_used_more_than_once(node.input[0]) or g.is_used_more_than_once(
            node.input[1]
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        node_left = g.node_before(node.input[0])
        if (
            node_left is None
            or node_left.op_type not in {"Div", "Mul"}
            or node.domain != ""
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        node_right = g.node_before(node.input[1])
        if (
            node_right is None
            or node_right.op_type not in {"Div", "Mul"}
            or node.domain != ""
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        # checking for the constant (right)
        if not g.is_constant(node_left.input[1]) or not g.is_constant(
            node_right.input[1]
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        cst_left = g.get_computed_constant(node_left.input[1])
        cst_right = g.get_computed_constant(node_right.input[1])
        if cst_left.shape not in {tuple(), (1,)} or cst_right.shape not in {
            tuple(),
            (1,),
        }:
            return self.none(node, inspect.currentframe().f_lineno)

        nodes = [node, node_left, node_right]

        return MatchResult(self, nodes, self.apply)

    def apply(
        self,
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


class SwitchOrderBinaryPattern(PatternOptimization):
    """
    If it makes sense, switches the order of two multiplications
    or two addtions if the broadcasting reduces one operator to
    a an insignificant number.
    """

    class BroadcastType(IntEnum):
        """
        Kind of broadcast.
        """

        FALSE = 0
        TRUE = 1
        MAYBE = 2
        BOTH = 3

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type not in {"Add", "Mul"} or node.domain != "":
            return self.none()

        if not g.has_shape(node.input[0]) or not g.has_shape(node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)

        op_type = node.op_type

        left = g.node_before(node.input[0])
        right = g.node_before(node.input[1])
        left_type = getattr(left, "op_type", None)
        right_type = getattr(right, "op_type", None)
        if op_type not in {left_type, right_type}:
            return self.none()

        if left_type is None:
            choose = 1
        elif right_type is None:
            choose = 0
        else:
            # Both left and right do the same operator.
            if (
                left.op_type != op_type
                or not g.has_shape(left.input[0])
                or not g.has_shape(left.input[1])
            ):
                if right.op_type != op_type:
                    return self.none(node, inspect.currentframe().f_lineno)
                choose = 1
            elif (
                right.op_type != op_type
                or not g.has_shape(right.input[0])
                or not g.has_shape(right.input[1])
            ):
                if left.op_type != op_type:
                    return self.none(node, inspect.currentframe().f_lineno)
                choose = 0
            elif right.op_type != op_type:
                if left.op_type != op_type:
                    return self.none(node, inspect.currentframe().f_lineno)
                choose = 0
            elif left.op_type != op_type:
                choose = 1
            else:
                # all have shapes and the right type
                choose = 3

        other_node = left if choose == 0 else right
        assert (
            other_node.op_type == node.op_type
        ), f"Type mismatch {node.op_type} != {other_node.op_type}"
        if not g.has_shape(other_node.input[0]) or not g.has_shape(other_node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)

        shape_left = g.get_shape(node.input[0])
        shape_right = g.get_shape(node.input[1])
        before_left = g.get_shape(other_node.input[0])
        before_right = g.get_shape(other_node.input[1])

        if (
            self.switch_order(
                shape_left, shape_right, before_left, before_right, choose
            )
            == 0
        ):
            if choose < 3:
                return self.none(node, inspect.currentframe().f_lineno)
            choose = 1
            other_node = right
            before_left = g.get_shape(other_node.input[0])
            before_right = g.get_shape(other_node.input[1])
            if (
                self.switch_order(
                    shape_left, shape_right, before_left, before_right, choose
                )
                == 0
            ):
                return self.none(node, inspect.currentframe().f_lineno)

        assert choose in (0, 1), f"Unexpected value for choose={choose}"
        assert (
            other_node.op_type == node.op_type
        ), f"Type mismatch {node.op_type} != {other_node.op_type}"
        if g.is_used_more_than_once(other_node.output[0]):
            return self.none(node, inspect.currentframe().f_lineno)

        nodes = [node, left if choose == 0 else None, right if choose == 1 else None]

        return MatchResult(self, nodes, self.apply, insert_at=node)

    def _align_shape(self, sh: DYNAMIC_SHAPE, rk: int) -> DYNAMIC_SHAPE:
        """
        Aligns shapes to the same size.
        """
        if len(sh) == rk:
            return sh
        return (1,) * (rk - len(sh)) + sh

    def switch_order(
        self,
        shape_left: DYNAMIC_SHAPE,
        shape_right: DYNAMIC_SHAPE,
        shape_before_left: DYNAMIC_SHAPE,
        shape_before_right: DYNAMIC_SHAPE,
        side: int,
    ) -> int:
        """
        Tells if the order should be switched.
        `side==0` indicates if `shape_left` comes from
        `Op(shape_before_left, shape_before_mul)`.

        if ``side == 0``:

        * Case 0: (B + C) + A: ``Op(Op(shape_before_left, shape_before_right), shape_right)``
        * Case 1: (B + A) + C: ``Op(Op(shape_before_left, shape_right), shape_before_right)``
        * Case 2: (A + C) + B: ``Op(Op(shape_right, shape_before_left), shape_before_left)``

        The function returns the case.
        """
        if side == 1:
            return self.switch_order(
                shape_right, shape_left, shape_before_left, shape_before_right, 0
            )

        # option

        r_left = len(shape_left)
        r_right = len(shape_right)
        r_b_left = len(shape_before_left)
        r_b_right = len(shape_before_right)

        rk = max(max(r_left, r_right), max(r_b_left, r_b_right))
        assert max(r_left, r_right) == rk, (
            f"Inconsistencies with shapes (side={side}) shape_left={shape_left}, "
            f"shape_right={shape_right}, shape_before_left={shape_before_left}, "
            f"shape_before_right={shape_before_right}"
        )
        cases = [
            max(r_b_left, r_b_right),
            max(r_right, r_b_left),
            max(r_right, r_b_right),
        ]

        if cases[0] < min(cases[1], cases[2]):
            return 0
        if cases[1] < min(cases[0], cases[2]):
            return 1
        if cases[2] < min(cases[0], cases[1]):
            return 2

        # Ranks cannot be used to determine if switch is recommended.
        rk = max(cases)
        # shape_left = self._align_shape(shape_left, rk)
        shape_right = self._align_shape(shape_right, rk)
        shape_before_left = self._align_shape(shape_before_left, rk)
        shape_before_right = self._align_shape(shape_before_right, rk)

        for b, c, a in zip(shape_before_left, shape_before_right, shape_right):
            if b == c == a:
                continue

            if isinstance(a, int) and isinstance(b, int) and isinstance(c, int):
                cases = [max(b, c), max(b, a), max(a, c)]

                if cases[0] < min(cases[1], cases[2]):
                    return 0
                if cases[1] < min(cases[0], cases[2]):
                    return 1
                if cases[2] < min(cases[0], cases[1]):
                    return 2

            # Dynamic shapes is not implemented yet but it should
            # take place here.

        # No change.
        return 0

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node: NodeProto,
        node_left: NodeProto,
        node_right: NodeProto,
    ) -> List[NodeProto]:
        side = 1 if node_left is None else 0
        other_node = node_right if node_left is None else node_left
        assert (
            other_node.op_type == node.op_type
        ), f"Type mismatch {node.op_type} != {other_node.op_type}"

        shape_left = g.get_shape(node.input[0])
        shape_right = g.get_shape(node.input[1])
        before_left = g.get_shape(other_node.input[0])
        before_right = g.get_shape(other_node.input[1])

        case = self.switch_order(
            shape_left, shape_right, before_left, before_right, side
        )
        assert case in (1, 2), (
            f"case={case}, the matching should not have happened "
            f"(side={side}) shape_left={shape_left}, "
            f"shape_right={shape_right}, before_left={before_left}, "
            f"before_right={before_right}"
        )

        # For side == 0
        # Case 0: (B + C) + A
        # Case 1: (B + A) + C
        # Case 2: (A + C) + B

        op_type = node.op_type
        final = node.output[0]
        if side == 0:
            B, C, A = other_node.input[0], other_node.input[1], node.input[1]
            if case == 1:
                op1 = g.make_node(
                    op_type, [B, A], name=f"{self.__class__.__name__}--{node.name}"
                )
                op2 = g.make_node(
                    op_type,
                    [op1.output[0], C],
                    [final],
                    doc_string=node.doc_string,
                    name=f"{self.__class__.__name__}--{node.name}",
                )
                return [op1, op2]

            # case 2
            op1 = g.make_node(
                op_type, [C, A], name=f"{self.__class__.__name__}--{node.name}"
            )
            op2 = g.make_node(
                op_type,
                [op1.output[0], B],
                [final],
                doc_string=node.doc_string,
                name=f"{self.__class__.__name__}--{node.name}",
            )
            return [op1, op2]

        # side 1
        B, C, A = other_node.input[0], other_node.input[1], node.input[0]
        if case == 1:
            op1 = g.make_node(
                op_type, [B, A], name=f"{self.__class__.__name__}--{node.name}"
            )
            op2 = g.make_node(
                op_type,
                [op1.output[0], C],
                [final],
                doc_string=node.doc_string,
                name=f"{self.__class__.__name__}--{node.name}",
            )
            return [op1, op2]

        # case 2
        op1 = g.make_node(
            op_type, [C, A], name=f"{self.__class__.__name__}--{node.name}"
        )
        op2 = g.make_node(
            op_type,
            [op1.output[0], B],
            [final],
            doc_string=node.doc_string,
            name=f"{self.__class__.__name__}--{node.name}",
        )
        return [op1, op2]
