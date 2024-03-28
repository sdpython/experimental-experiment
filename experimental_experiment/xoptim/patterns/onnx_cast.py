import inspect
from typing import List, Optional
from onnx import NodeProto, TensorProto
from ..patterns_api import MatchResult, PatternOptimization


class CastPattern(PatternOptimization):
    """
    Checks that a Cast is really needed.
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Cast" or node.domain != "":
            return self.none()

        if not g.has_type(node.input[0]):
            itype = g.try_infer_type(node.input[0])
            if itype == 0:
                return self.none(node, inspect.currentframe().f_lineno)
        else:
            itype = g.get_type(node.input[0])

        att = g.get_attribute(node, "to")

        if att.i != itype:
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [node], self.apply, insert_at=node)

    @classmethod
    def apply(cls, g: "GraphBuilder", node: NodeProto) -> List[NodeProto]:  # noqa: F821
        new_node = g.make_node(
            "Identity",
            node.input,
            node.output,
            name=f"{cls.__name__}--{node.name}",
            doc_string=node.doc_string,
        )
        return [new_node]


class CastCastBinaryPattern(PatternOptimization):
    """
    Moves two cast operators beyond a binary operator
    The cast must cast from a float type to another float type.
    """

    _dtypes_allowed = {
        TensorProto.FLOAT16,
        TensorProto.BFLOAT16,
        TensorProto.FLOAT,
        TensorProto.DOUBLE,
    }

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type not in {"Add", "Div", "Mul", "Sub"} or node.domain != "":
            return self.none()

        if g.is_used_more_than_once(node.input[0]) or g.is_used_more_than_once(
            node.input[1]
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        dtype_left, dtype_right = g.get_type(node.input[0]), g.get_type(node.input[1])
        if (
            dtype_left not in self._dtypes_allowed
            or dtype_right not in self._dtypes_allowed
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        left, right = g.node_before(node.input[0]), g.node_before(node.input[1])
        if left is None or left.op_type != "Cast" or left.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if right is None or right.op_type != "Cast" or right.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        dtype_left, dtype_right = g.get_type(left.input[0]), g.get_type(right.input[0])
        if (
            dtype_left not in self._dtypes_allowed
            or dtype_right not in self._dtypes_allowed
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [left, right, node], self.apply, insert_at=node)

    @classmethod
    def apply(
        cls,
        g: "GraphBuilder",  # noqa: F821
        left: NodeProto,
        right: NodeProto,
        node: NodeProto,
    ) -> List[NodeProto]:

        to = g.get_attribute(left, "to")

        new_node = g.make_node(
            node.op_type,
            [left.input[0], right.input[0]],
            name=f"{cls.__name__}--{node.name}",
        )
        cast_node = g.make_node(
            "Cast",
            new_node.output,
            node.output,
            to=to.i,
            name=f"{cls.__name__}--{node.name}",
            doc_string=node.doc_string,
        )
        return [new_node, cast_node]
