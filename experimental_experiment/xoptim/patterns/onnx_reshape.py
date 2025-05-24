import inspect
from typing import List, Optional
import numpy as np
from onnx import NodeProto
from ...xbuilder._onnx_helper import element_wise_binary_op_types
from ...xbuilder._shape_helper import all_int
from ..patterns_api import MatchResult, PatternOptimization


class ReshapePattern(PatternOptimization):
    """
    Checks that a Reshape is really needed.
    """

    def __init__(self, verbose: int = 0, priority: int = 0):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Reshape" or node.domain != "":
            return self.none()
        if not g.has_shape(node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        shape = g.get_shape(node.input[0])
        if not all_int(shape):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant(node.input[1]):
            # It may be a symbolic shape.
            return self.none(node, inspect.currentframe().f_lineno)
        value = g.get_computed_constant(node.input[1])
        if value is None:
            return self.none(node, inspect.currentframe().f_lineno)
        new_shape = tuple(int(i) for i in value)
        if shape != new_shape:
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, [node], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node: NodeProto,
    ) -> List[NodeProto]:
        new_node = g.make_node(
            "Identity",
            node.input,
            node.output,
            name=f"{self.__class__.__name__}--{node.name}",
            doc_string=node.doc_string,
        )
        return [new_node]


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
            if not g.has_rank(node.input[0]):
                return self.none(node, inspect.currentframe().f_lineno)
            att = g.get_attribute(node, "axes", exc=False)
            axes = tuple(range(g.get_rank(node.input[0]))) if att is None else tuple(att.ints)

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
                return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [node, next_node], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node: NodeProto,
        next_node: NodeProto,
    ) -> List[NodeProto]:
        axes = g.get_attribute(node, "axes", exc=False)
        if axes is None:
            new_node = g.make_node(
                node.op_type,
                node.input,
                next_node.output,
                keepdims=0,
                name=f"{self.__class__.__name__}--{node.name}",
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
            name=f"{self.__class__.__name__}--{node.name}",
            doc_string=node.doc_string,
        )
        return [new_node]


class ReshapeReshapePattern(PatternOptimization):
    """
    Replaces the sequence Reshape, Reshape by Reshape.
    """

    def __init__(self, verbose: int = 0, priority: int = 0):
        super().__init__(verbose, priority)

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
        if g.is_constant(node.input[1]):
            cst = g.get_computed_constant(node.input[1])
            if -1 in cst.tolist():
                # Then we only allow it the shape is static.
                if not g.is_constant(next_node.input[1]):
                    return self.none(node, inspect.currentframe().f_lineno)
                cst = g.get_computed_constant(next_node.input[1])
                if cst.min() <= 0:
                    return self.none(node, inspect.currentframe().f_lineno)
        if (
            not g.has_rank(node.input[0])
            or not g.has_rank(next_node.output[0])
            or not g.has_rank(node.output[0])
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        sh1 = g.builder.value_as_shape(node.input[1])
        sh2 = g.builder.value_as_shape(next_node.input[1])
        if sh1 is None or sh2 is None:
            return self.none(node, inspect.currentframe().f_lineno)
        if -1 in sh1 or -1 in sh2:
            return self.none(node, inspect.currentframe().f_lineno)

        # If g.get_rank(node.input[0]) != g.get_rank(next_node.output[0]),
        # the bet is, when the shape is not a constant, then using 0 is not really
        # useful. Since 0 is only valid for ONNX, 0 should not be found
        # in a non constant shape used to reshape.
        # If it is a constant that should be ok too.
        return MatchResult(self, [node, next_node], self.apply, insert_at=next_node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node: NodeProto,
        next_node: NodeProto,
    ) -> List[NodeProto]:
        second_input = next_node.input[1]
        pre_nodes = []
        if (
            g.get_rank(node.input[0]) != g.get_rank(next_node.output[0])
            and g.get_rank(node.output[0]) == g.get_rank(next_node.output[0])
            and g.is_constant(next_node.input[1])
        ):
            cst = tuple(g.get_computed_constant(next_node.input[1]))
            if 0 in cst:
                if g.is_constant(node.input[1]):
                    shape0 = tuple(g.get_computed_constant(node.input[1]))
                    assert len(shape0) == len(cst), (
                        f"This should be true due to the first test but cst={cst}, "
                        f"shape0={shape0}"
                    )
                    new_shape = [(s if s != 0 else s0) for s, s0 in zip(cst, shape0)]
                    assert (
                        len(new_shape) >= len([s for s in new_shape if s != 0]) - 1
                    ), f"new_shape={new_shape} has two -1. This is not possible."
                    second_input = g.make_initializer(
                        "",
                        np.array(new_shape, dtype=np.int64),
                        source="ReshapeReshapePattern.new_shape.1",
                    )
                else:
                    # This code has one loop hole. It could produce shapes with two -1.
                    # Let's extract the missing information.
                    names = []
                    for axis, dim in enumerate(cst):
                        if dim == 0:
                            d_name = g.unique_name(f"{next_node.input[0]}--dim{axis}")
                            d_init = g.make_initializer(
                                "",
                                np.array([axis], dtype=np.int64),
                                source=f"ReshapeReshapePattern.axis.{axis}.1",
                            )
                            pre_nodes.append(
                                g.make_node(
                                    "Gather",
                                    [node.input[1], d_init],
                                    [d_name],
                                    axis=0,
                                    name=f"{next_node.name}--axis{axis}",
                                )
                            )
                            names.append(d_name)
                        else:
                            d_init = g.make_initializer(
                                "",
                                np.array([dim], dtype=np.int64),
                                source=f"ReshapeReshapePattern.axis.{axis}.2",
                            )
                            names.append(d_init)
                    second_input = g.unique_name(f"{next_node.input[0]}--concat")
                    pre_nodes.append(
                        g.make_node(
                            "Concat",
                            names,
                            [second_input],
                            axis=0,
                            name=f"{next_node.name}--concat",
                        )
                    )

        new_node = g.make_node(
            "Reshape",
            [node.input[0], second_input],
            next_node.output,
            name=f"{self.__class__.__name__}--{node.name}",
            doc_string=next_node.doc_string,
        )
        return [*pre_nodes, new_node]


class Reshape2Of3Pattern(PatternOptimization):
    """
    Replaces the reshapes around element-wise operators.
    It can be 3 or 2 out of 3.
    """

    _op_types = element_wise_binary_op_types()

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
        if len(next_nodes) > 1 or (len(next_nodes) == 0 and not g.is_output(node.output[0])):
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
            (
                None
                if (node_left is None or not g.has_shape(node_left.input[0]))
                else g.get_shape(node_left.input[0])
            ),
            (
                None
                if (node_right is None or not g.has_shape(node_right.input[0]))
                else g.get_shape(node_right.input[0])
            ),
            (
                None
                if (next_node is None or not g.has_shape(next_node.output[0]))
                else g.get_shape(next_node.output[0])
            ),
        ]

        if len(set(_ for _ in shapes if _ is not None)) != 1:
            # Not the same shapes.
            return self.none(node, inspect.currentframe().f_lineno)

        nodes = [node_left, node_right, next_node, node]

        return MatchResult(self, nodes, self.apply)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node_left: NodeProto,
        node_right: NodeProto,
        next_node: NodeProto,
        node: NodeProto,
    ) -> List[NodeProto]:
        compute_shape_name = node_left.input[1] if node_right is None else node_right.input[1]
        final_shape_name = compute_shape_name if next_node is None else next_node.input[1]

        res = []

        # node left
        if node_left is None:
            left_name = g.unique_name(f"{self.__class__.__name__}L_{node.input[0]}")
            res.append(
                g.make_node(
                    "Reshape",
                    [node.input[0], final_shape_name],
                    [left_name],
                    name=f"{self.__class__.__name__}--{node.name}",
                )
            )
        elif g.is_used_more_than_once(node_left.output[0]):
            res.append(node_left)
            left_name = node_left.input[0]
        else:
            left_name = node_left.input[0]

        # node right
        if node_right is None:
            right_name = g.unique_name(f"{self.__class__.__name__}R_{node.input[1]}")
            res.append(
                g.make_node(
                    "Reshape",
                    [node.input[1], final_shape_name],
                    [right_name],
                    name=f"{self.__class__.__name__}--{node.name}",
                )
            )
        elif g.is_used_more_than_once(node_right.output[0]):
            res.append(node_right)
            right_name = node_right.input[0]
        else:
            right_name = node_right.input[0]

        # node and next node
        if next_node is None:
            # Reshape is needed.
            new_name = g.unique_name(f"{self.__class__.__name__}L_{node.output[0]}")
            res.extend(
                [
                    g.make_node(
                        node.op_type,
                        [left_name, right_name],
                        [new_name],
                        name=f"{self.__class__.__name__}--{node.name}",
                    ),
                    g.make_node(
                        "Reshape",
                        [new_name, final_shape_name],
                        [node.output[0]],
                        name=f"{self.__class__.__name__}--{node.name}",
                    ),
                ]
            )
        else:
            main_node = g.make_node(
                node.op_type,
                [left_name, right_name],
                [next_node.output[0]],
                name=f"{self.__class__.__name__}--{node.name}",
            )
            res.append(main_node)

            if g.is_used_more_than_once(node.output[0]):
                res.append(
                    g.make_node(
                        "Reshape",
                        [main_node.output[0], compute_shape_name],
                        [node.output[0]],
                        name=f"{self.__class__.__name__}--{node.name}",
                    )
                )

        return res


class ReshapeReshapeBinaryPattern(PatternOptimization):
    """
    Moves two reshape operators beyond a binary operator
    if it is possible.
    """

    _op_types = element_wise_binary_op_types()

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type not in self._op_types or node.domain != "":
            return self.none()

        if g.is_used_more_than_once(node.input[0]) or g.is_used_more_than_once(node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)

        left, right = g.node_before(node.input[0]), g.node_before(node.input[1])
        if left is None or left.op_type != "Reshape" or left.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if right is None or right.op_type != "Reshape" or right.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant(left.input[1]) or not g.is_constant(right.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)

        if not g.has_shape(node.output[0]):
            return self.none(node, inspect.currentframe().f_lineno)

        cst_left = g.get_computed_constant(left.input[1]).tolist()
        cst_right = g.get_computed_constant(right.input[1]).tolist()
        if cst_left != cst_right:
            return self.none(node, inspect.currentframe().f_lineno)

        shape1 = g.get_shape(left.input[0]) if g.has_shape(left.input[0]) else None
        shape2 = g.get_shape(right.input[0]) if g.has_shape(right.input[0]) else None
        if shape1 is None or shape2 is None or shape1 != shape2:
            return self.none(node, inspect.currentframe().f_lineno)

        # If there is not broadcast involved then it is ok.
        # At this stage, we know shapes are equal before the reshaped operators
        # and the same reshape is applied. So checking the output shape
        # is not necesssary.
        return MatchResult(self, [left, right, node], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        left: NodeProto,
        right: NodeProto,
        node: NodeProto,
    ) -> List[NodeProto]:
        new_node = g.make_node(
            node.op_type,
            [left.input[0], right.input[0]],
            name=f"{self.__class__.__name__}--{node.name}",
        )
        reshape_node = g.make_node(
            "Reshape",
            [new_node.output[0], left.input[1]],
            node.output,
            name=f"{self.__class__.__name__}--{node.name}",
            doc_string=node.doc_string,
        )
        return [new_node, reshape_node]
