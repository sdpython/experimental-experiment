import inspect
from typing import List, Optional
from onnx import NodeProto
from ...xbuilder._onnx_helper import element_wise_op_types
from .patterns_api import MatchResult, PatternOptimization


class ReduceReshapePattern(PatternOptimization):
    """
    Replaces the sequence Reduce* Reshape if reshape is only
    introduces to deal with a dimension kept because keepdim=1.
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if not node.op_type.startswith("Reduce") or node.domain != "":
            return self.none()

        if g.is_used_more_than_once(node.output[0]):
            return self.none(node, inspect.currentframe().f_lineno)

        att = g.get_attribute(node, "keepdims", exc=False)
        keepdims = 1 if att is None else att.i
        if keepdims == 0:
            # not keeping the dimension so Reshape means to restore them.
            return self.none(node, inspect.currentframe().f_lineno)

        if len(node.input) == 2:
            if not g.is_constant(node.input[1]):
                return self.none(node, inspect.currentframe().f_lineno)
            axes = tuple(g.get_computed_constant(node.input[1]))
        else:
            att = g.get_attribute(node, "axes", exc=False)
            axes = (
                tuple(range(g.get_rank(node.input[0])))
                if att is None
                else tuple(att.ints)
            )

        next_nodes = g.next_nodes(node.output[0])
        if len(next_nodes) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        next_node = next_nodes[0]
        if next_node.op_type != "Reshape" or node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if next_node.input[0] != node.output[0]:
            return self.none(node, inspect.currentframe().f_lineno)

        if g.get_rank(node.input[0]) != g.get_rank(next_node.output[0]) + len(axes):
            return self.none(node, inspect.currentframe().f_lineno)

        if g.get_rank(next_node.output[0]) > 1:
            if not g.has_shape(node.input[0]):
                return self.none(node, inspect.currentframe().f_lineno)
            set_axes = set(axes)
            shape = g.get_shape(node.input[0])
            reduced_shape = [s for i, s in enumerate(shape) if i not in set_axes]
            reshaped_shape = g.get_shape(next_node.output[0])
            if reduced_shape != reshaped_shape:
                print(reduced_shape, reshaped_shape, axes)
                return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [node, next_node], self.apply, insert_at=node)

    @classmethod
    def apply(
        cls, g: "GraphBuilder", node: NodeProto, next_node: NodeProto  # noqa: F821
    ) -> List[NodeProto]:
        axes = g.get_attribute(node, "axes", exc=False)
        if axes is None:
            new_node = g.make_node(
                node.op_type,
                node.input,
                next_node.output,
                keepdims=0,
                name=f"{cls.__class__.__name__}--{node.name}",
                doc_string=node.doc_string,
            )
            return [new_node]

        # older opset
        new_node = g.make_node(
            node.op_type,
            node.input,
            next_node.output,
            keepdims=0,
            axes=list(axes.ints),
            name=f"{cls.__class__.__name__}--{node.name}",
            doc_string=node.doc_string,
        )
        return [new_node]


class ReshapeReshapePattern(PatternOptimization):
    """
    Replaces the sequence Reshape, Reshape by Reshape.
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Reshape" or node.domain != "":
            return self.none()

        if g.is_used_more_than_once(node.output[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        next_nodes = g.next_nodes(node.output[0])
        if len(next_nodes) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        next_node = next_nodes[0]
        if next_node.op_type != "Reshape" or node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if next_node.input[0] != node.output[0]:
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [node, next_node], self.apply, insert_at=next_node)

    @classmethod
    def apply(
        cls, g: "GraphBuilder", node: NodeProto, next_node: NodeProto  # noqa: F821
    ) -> List[NodeProto]:
        new_node = g.make_node(
            "Reshape",
            [node.input[0], next_node.input[1]],
            next_node.output,
            name=f"{cls.__class__.__name__}--{node.name}",
            doc_string=next_node.doc_string,
        )
        return [new_node]


class Reshape2Of3Pattern(PatternOptimization):
    """
    Replaces the reshapes around element-wise operators.
    It can be 3 or 2 out of 3.
    """

    _op_types = element_wise_op_types()

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type not in self._op_types or node.domain != "":
            return self.none()

        if (
            not g.has_shape(node.output[0])
            or not g.has_shape(node.input[0])
            or not g.has_shape(node.input[1])
        ):
            # Shapes are missing. They should be populated as much as possible.
            return self.none(node, inspect.currentframe().f_lineno)

        shape_out = g.get_shape(node.output[0])
        shape_in = g.get_shape(node.input[0]), g.get_shape(node.input[1])
        if not (shape_out == shape_in[0] == shape_in[1]):
            # Broadcasting is involved.
            return self.none(node, inspect.currentframe().f_lineno)

        next_nodes = g.next_nodes(node.output[0])
        if len(next_nodes) > 1 or (
            len(next_nodes) == 0 and not g.is_output(node.output[0])
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        next_node = None if len(next_nodes) == 0 else next_nodes[0]
        type_out = None if next_node is None else next_node.op_type

        node_left = g.node_before(node.input[0])
        node_right = g.node_before(node.input[1])
        type_left = None if node_left is None else node_left.op_type
        type_right = None if node_right is None else node_right.op_type

        types = [type_left, type_right, type_out, node.op_type]
        n_reshape = len([_ for _ in types if _ == "Reshape"])
        if n_reshape < 2:
            return self.none(node, inspect.currentframe().f_lineno)

        if node_left is not None and node_left.op_type != "Reshape":
            node_left = None
        if node_right is not None and node_right.op_type != "Reshape":
            node_right = None
        if next_node is not None and next_node.op_type != "Reshape":
            next_node = None

        shapes = [
            (None if node_left is None else g.get_shape(node_left.input[0])),
            (None if node_right is None else g.get_shape(node_right.input[0])),
            (None if next_node is None else g.get_shape(next_node.output[0])),
        ]

        if len(set(_ for _ in shapes if _ is not None)) != 1:
            # Not the same shapes.
            return self.none(node, inspect.currentframe().f_lineno)

        nodes = [node_left, node_right, next_node, node]

        return MatchResult(self, nodes, self.apply)

    @classmethod
    def apply(
        cls,
        g: "GraphBuilder",  # noqa: F821
        node_left: NodeProto,
        node_right: NodeProto,
        next_node: NodeProto,
        node: NodeProto,
    ) -> List[NodeProto]:

        compute_shape_name = (
            node_left.input[1] if node_right is None else node_right.input[1]
        )
        final_shape_name = (
            compute_shape_name if next_node is None else next_node.input[1]
        )

        res = []

        # node left
        if node_left is None:
            left_name = g.unique_name(f"{cls.__class__.__name__}L_{node.input[0]}")
            res.append(
                g.make_node("Reshape", [node.input[0], final_shape_name], [left_name])
            )
        elif g.is_used_more_than_once(node_left.output[0]):
            res.append(node_left)
            left_name = node_left.input[0]
        else:
            left_name = node_left.input[0]

        # node right
        if node_right is None:
            right_name = g.unique_name(f"{cls.__class__.__name__}R_{node.input[1]}")
            res.append(
                g.make_node("Reshape", [node.input[1], final_shape_name], [right_name])
            )
        elif g.is_used_more_than_once(node_right.output[0]):
            res.append(node_right)
            right_name = node_right.input[0]
        else:
            right_name = node_right.input[0]

        # node and next node
        if next_node is None:
            # Reshape is needed.
            new_name = g.unique_name(f"{cls.__class__.__name__}L_{node.output[0]}")
            res.extend(
                [
                    g.make_node(
                        node.op_type,
                        [left_name, right_name],
                        [new_name],
                    ),
                    g.make_node(
                        "Reshape",
                        [new_name, final_shape_name],
                        [node.output[0]],
                    ),
                ]
            )
        else:
            main_node = g.make_node(
                node.op_type,
                [left_name, right_name],
                [next_node.output[0]],
            )
            res.append(main_node)

            if g.is_used_more_than_once(node.output[0]):
                res.append(
                    g.make_node(
                        "Reshape",
                        [main_node.output[0], compute_shape_name],
                        [node.output[0]],
                    )
                )

        return res
