from typing import List, Optional
from onnx import NodeProto
from ...xbuilder._onnx_helper import element_wise_op_types
from .patterns_api import MatchResult, PatternOptimization


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
            return None

        if g.is_used_more_than_once(node.output[0]):
            return None
        next_nodes = g.next_nodes(node.output[0])
        if len(next_nodes) != 1:
            return None
        next_node = next_nodes[0]
        if next_node.op_type != "Reshape" or node.domain != "":
            return None
        if next_node.input[0] != node.output[0]:
            return None

        def apply(
            g: "GraphBuilder", node: NodeProto, next_node: NodeProto  # noqa: F821
        ) -> List[NodeProto]:
            new_node = g.make_node(
                "Reshape",
                [node.input[0], next_node.input[1]],
                next_node.output,
                name=f"{self.__class__.__name__}--{node.name}",
                doc_string=next_node.doc_string,
            )
            return [new_node]

        return MatchResult(self, [node, next_node], apply, insert_at=next_node)


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
            return None

        shape_out = g.get_shape(node.output[0])
        shape_in = g.get_shape(node.input[0]), g.get_shape(node.input[1])
        if not (shape_out == shape_in[0] == shape_in[1]):
            # Broadcasting is involved.
            return None

        next_nodes = g.next_nodes(node.output[0])
        if len(next_nodes) > 1 or (
            len(next_nodes) == 0 and not g.is_output(node.output[0])
        ):
            return None
        next_node = None if len(next_nodes) == 0 else next_nodes[0]
        type_out = None if next_node is None else next_node.op_type

        node_left = g.node_before(node.input[0])
        node_right = g.node_before(node.input[1])
        type_left = None if node_left is None else node_left.op_type
        type_right = None if node_right is None else node_right.op_type

        types = [type_left, type_right, type_out, node.op_type]
        n_reshape = len([_ for _ in types if _ == "Reshape"])
        if n_reshape < 2:
            return None

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
            return None

        nodes = [node_left, node_right, next_node, node]

        def apply(
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
            if node_left is not None and g.is_used_more_than_once(node_left.output[0]):
                res.append(node_left)
            if node_right is not None and g.is_used_more_than_once(
                node_right.output[0]
            ):
                res.append(node_right)

            if node_left is None:
                left_name = g.unique_name(f"{self.__class__.__name__}L_{node.input[0]}")
                res.append(
                    g.make_node(
                        "Reshape", [node.input[0], final_shape_name], [left_name]
                    )
                )
            else:
                left_name = node_left.input[0]

            if node_right is None:
                right_name = g.unique_name(
                    f"{self.__class__.__name__}R_{node.input[1]}"
                )
                res.append(
                    g.make_node(
                        "Reshape", [node.input[1], final_shape_name], [right_name]
                    )
                )
            else:
                right_name = node_right.input[0]

            if next_node is None:
                # Reshape is needed.
                new_name = g.unique_name(f"{self.__class__.__name__}L_{node.output[0]}")
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

        return MatchResult(self, nodes, apply)
