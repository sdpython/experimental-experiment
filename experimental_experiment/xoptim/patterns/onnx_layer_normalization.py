import inspect
from typing import List, Optional
from onnx import NodeProto, TensorProto
from ..patterns_api import MatchResult, PatternOptimization


class LayerNormalizationPattern(PatternOptimization):
    """
    Checks that a Cast is really needed.
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "ReduceMean" or node.domain != "":
            return self.none()

        # before

        pow = g.node_before(node.input[0])
        if pow.op_type != "Pow" or len(g.next_nodes(pow.output[0])) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        if (
            not g.is_constant_scalar(pow.input[1])
            or g.get_constant_scalar(pow.input[1]) != 2
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        sub = g.node_before(pow.input[0])
        if sub.op_type != "Sub" or len(g.next_nodes(sub.output[0])) != 2:
            return self.none(node, inspect.currentframe().f_lineno)
        red = g.node_before(sub.input[1])
        if red.op_type != "ReduceMean" or len(g.next_nodes(red.output[0])) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        if sub.input[0] != red.input[0]:
            return self.none(node, inspect.currentframe().f_lineno)
        kp = g.get_attribute(red, "keepdims", exc=False)
        if kp is None or kp.i != 1:
            return self.none(node, inspect.currentframe().f_lineno)

        # after
        add = g.next_nodes(node.output[0])
        if len(add) != 1 or add[0].op_type != "Add":
            return self.none(node, inspect.currentframe().f_lineno)
        add = add[0]
        if not g.is_constant_scalar(add.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        sqrt = g.next_nodes(add.output[0])
        if len(sqrt) != 1 or sqrt[0].op_type != "Sqrt":
            return self.none(node, inspect.currentframe().f_lineno)
        sqrt = sqrt[0]
        div = g.next_nodes(sqrt.output[0])
        if len(div) != 1 or div[0].op_type != "Div":
            return self.none(node, inspect.currentframe().f_lineno)
        div = div[0]
        if div.input[0] != sub.output[0]:
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(
            self, [red, sub, pow, node, add, sqrt, div], self.apply, insert_at=node
        )

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        red: NodeProto,
        sub: NodeProto,
        pow: NodeProto,
        node: NodeProto,
        add: NodeProto,
        sqrt: NodeProto,
        div: NodeProto,
    ) -> List[NodeProto]:
        eps = g.get_constant_scalar(add.input[1])
        new_node = g.make_node(
            "LayerNormalization",
            [red.input[0]],
            [div.output[0], div.input[1]],
            epsilon=float(eps),
            name=f"{self.__class__.__name__}--{node.name}",
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

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        left: NodeProto,
        right: NodeProto,
        node: NodeProto,
    ) -> List[NodeProto]:

        to = g.get_attribute(left, "to")

        new_node = g.make_node(
            node.op_type,
            [left.input[0], right.input[0]],
            name=f"{self.__class__.__name__}--{node.name}",
        )
        cast_node = g.make_node(
            "Cast",
            new_node.output,
            node.output,
            to=to.i,
            name=f"{self.__class__.__name__}--{node.name}",
            doc_string=node.doc_string,
        )
        return [new_node, cast_node]


class CastOpCastPattern(PatternOptimization):
    """
    Removes two cast surrounding another operator.
    """

    _dtypes_allowed = {
        TensorProto.FLOAT16,
        TensorProto.BFLOAT16,
        TensorProto.FLOAT,
        TensorProto.DOUBLE,
    }

    _unary_ops = {"MulSigmoid", "Neg", "Sigmoid", "Softmax"}
    _binary_ops = {"Add", "Sub", "Mul", "Div"}
    _other_ops = {"SoftmaxGrad"}

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if (
            (node.op_type not in self._binary_ops or node.domain != "")
            and node.op_type not in self._other_ops
            and node.op_type not in self._unary_ops
        ):
            return self.none()
        if "ComputationCastOpCastPattern--" in node.name:
            return self.none(node, inspect.currentframe().f_lineno)

        if g.is_used_more_than_once(node.input[0]) or (
            len(node.input) > 1 and g.is_used_more_than_once(node.input[1])
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        next_nodes = g.next_nodes(node.output[0])
        if len(next_nodes) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        cast_out_node = next_nodes[0]
        if cast_out_node.op_type != "Cast" or cast_out_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        cast_in_left = g.node_before(node.input[0])
        cast_in_right = g.node_before(node.input[1]) if len(node.input) > 1 else None
        if "Cast" not in (
            "" if cast_in_left is None else cast_in_left.op_type,
            "" if cast_in_right is None else cast_in_right.op_type,
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        if cast_out_node is None and (
            cast_in_left is None
            or cast_in_left.op_type != "Cast"
            or cast_in_right is None
            or cast_in_right.op_type != "Cast"
        ):
            # Then we only allow this if the computation type is lower precision.
            compute_type = g.get_type(node.output[0])
            before_type = g.get_type((cast_in_left or cast_in_right).input[0])
            if not (
                compute_type == TensorProto.FLOAT
                and before_type in (TensorProto.FLOAT16, TensorProto.BFLOAT16)
            ):
                return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(
            self,
            [
                (
                    cast_in_left
                    if cast_in_left is not None and cast_in_left.op_type == "Cast"
                    else None
                ),
                (
                    cast_in_right
                    if cast_in_right is not None and cast_in_right.op_type == "Cast"
                    else None
                ),
                node,
                cast_out_node,
            ],
            self.apply,
            insert_at=node,
        )

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        cast_in_left: NodeProto,
        cast_in_right: NodeProto,
        node: NodeProto,
        cast_out_node: NodeProto,
    ) -> List[NodeProto]:
        # to = g.get_attribute(cast_in_left or cast_in_right, "to").i
        to_out = g.get_attribute(cast_out_node, "to").i
        left_input = None
        right_input = None
        new_nodes = []

        if cast_in_left is None:
            left_input = g.unique_name(f"{self.__class__.__name__}--{node.output[0]}")
            new_nodes.append(
                g.make_node(
                    "Cast",
                    [node.input[0]],
                    [left_input],
                    to=to_out,
                    name=f"{self.__class__.__name__}--CastL",
                )
            )
        else:
            left_input = cast_in_left.input[0]

        if cast_in_right is None and len(node.input) > 1:
            right_input = g.unique_name(f"{self.__class__.__name__}--{node.output[0]}")
            new_nodes.append(
                g.make_node(
                    "Cast",
                    [node.input[1]],
                    [right_input],
                    to=to_out,
                    name=f"{self.__class__.__name__}--CastR",
                )
            )
        else:
            right_input = None if cast_in_right is None else cast_in_right.input[0]

        new_node = g.make_node(
            node.op_type,
            [left_input] if right_input is None else [left_input, right_input],
            cast_out_node.output,
            domain=node.domain,
            name=f"{self.__class__.__name__}--{node.name}",
        )
        if node.attribute:
            new_node.attribute.extend(node.attribute)
        new_nodes.append(new_node)

        if g.is_used_more_than_once(node.output[0]):
            final_cast = g.make_node(
                "Cast",
                [new_node.output[0]],
                [node.output[0]],
                to=g.get_type(node.output[0]),
                name=f"{self.__class__.__name__}--{node.name}",
            )
            new_nodes.append(final_cast)

        return new_nodes


class ComputationCastOpCastPattern(PatternOptimization):
    """
    Changes the computation type to make it faster if one of the inputs
    was just casted before.
    """

    _dtypes_allowed = {
        TensorProto.FLOAT16,
        TensorProto.BFLOAT16,
        TensorProto.FLOAT,
        TensorProto.DOUBLE,
    }

    _binary_ops = {"Add", "Sub", "Mul", "Div"}

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type not in self._binary_ops or node.domain != "":
            return self.none()

        node_left = g.node_before(node.input[0])
        node_right = g.node_before(node.input[1])
        type_left = "" if node_left is None else node_left.op_type
        type_right = "" if node_right is None else node_right.op_type
        if not ((type_left == "Cast") ^ (type_right == "Cast")):
            # only one cast allowed
            return self.none(node, inspect.currentframe().f_lineno)

        if type_left:
            node_right = None
        else:
            node_left = None
        node_before = node_left or node_right
        output_type = g.get_type(node.output[0])
        before_type = g.get_type(node_before.input[0])
        if not (
            output_type == TensorProto.FLOAT
            and before_type in (TensorProto.FLOAT16, TensorProto.BFLOAT16)
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        if g.is_used_more_than_once(node_before.output[0]):
            return self.none(node, inspect.currentframe().f_lineno)

        next_nodes = g.next_nodes(node.output[0])
        op_types = [n.op_type for n in next_nodes]
        if "Cast" in op_types:
            return self.none(node, inspect.currentframe().f_lineno)

        # At this stage, we know the computation type is float and one input
        # has a lower type precision. Let's change it.
        return MatchResult(
            self, [node_left, node_right, node], self.apply, insert_at=node
        )

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node_left: Optional[NodeProto],
        node_right: Optional[NodeProto],
        node: NodeProto,
    ) -> List[NodeProto]:

        to_type = g.get_type(node.output[0])

        inputs = []
        if node_left is None:
            before_type = g.get_type(node_right.input[0])
            name = g.unique_name(f"{self.__class__.__name__}--{node.input[0]}")
            cast_node = g.make_node(
                "Cast",
                [node.input[0]],
                [name],
                to=before_type,
                name=f"{self.__class__.__name__}--{node.name}",
            )
            inputs = [name, node_right.input[0]]
        else:
            before_type = g.get_type(node_left.input[0])
            name = g.unique_name(f"{self.__class__.__name__}--{node.input[1]}")
            cast_node = g.make_node(
                "Cast",
                [node.input[1]],
                [name],
                to=before_type,
                name=f"{self.__class__.__name__}--{node.name}",
            )
            inputs = [node_left.input[0], name]

        name = g.unique_name(f"{self.__class__.__name__}--{node.output[0]}")
        new_node = g.make_node(
            node.op_type,
            inputs,
            [name],
            domain=node.domain,
            name=f"{self.__class__.__name__}--{node.name}",
        )
        final_cast = g.make_node(
            "Cast",
            [name],
            [node.output[0]],
            to=to_type,
            name=f"{self.__class__.__name__}--{node.name}",
        )
        return [cast_node, new_node, final_cast]