import inspect
from typing import List, Optional
import numpy as np
from onnx import NodeProto, TensorProto
from ..patterns_api import MatchResult, PatternOptimization


class SqueezeUnsqueezePattern(PatternOptimization):
    """Replaces the sequence Squeeze, Unsqueeze by Identity."""

    def __init__(self, verbose: int = 0, priority: int = 0):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Unsqueeze" or node.domain != "":
            return self.none()
        if len(node.input) < 2:
            return self.none(node, inspect.currentframe().f_lineno)
        if g.is_used_more_than_once(node.input[0]) and (
            not g.has_type(node.input[0])
            or g.get_type(node.input[0]) != TensorProto.INT64
            or not g.has_shape(node.input[0])
            or g.get_shape(node.input[0]) not in (tuple(), (1,))
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        node_before = g.node_before(node.input[0])
        if node_before is None or node_before.op_type != "Squeeze" or node_before.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        axes1 = (
            None
            if len(node_before.input) == 1
            else g.get_computed_constant(node_before.input[1])
        )
        axes2 = g.get_computed_constant(node.input[1])
        if axes1 is None:
            if (
                axes2 is None
                or tuple(map(int, axes2)) != (0,)
                or not g.has_shape(node_before.input[0])
                or g.get_shape(node_before.input[0]) != (1,)
            ):
                return self.none(node, inspect.currentframe().f_lineno)
        elif axes2 is None:
            return self.none(node, inspect.currentframe().f_lineno)
        else:
            if tuple(map(int, axes1)) != tuple(map(int, axes2)):
                return self.none(node, inspect.currentframe().f_lineno)
            if len(axes1) > 1 and tuple(map(int, axes1)) != tuple(
                range(min(axes1), max(axes1) + 1)
            ):
                return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(
            self,
            [node_before, node],
            self.apply,
            insert_at=node_before if g.is_used_more_than_once(node.input[0]) else node,
        )

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node_squ: NodeProto,
        node_uns: NodeProto,
    ) -> List[NodeProto]:
        new_node = g.make_node(
            "Identity",
            [node_squ.input[0]],
            [node_uns.output[0]],
            name=f"{self.__class__.__name__}--{node_uns.name}",
            doc_string=node_uns.doc_string,
        )
        return (
            [node_squ, new_node] if g.is_used_more_than_once(node_uns.input[0]) else [new_node]
        )


class UnsqueezeUnsqueezePattern(PatternOptimization):
    """Replaces the sequence Unsqueeze, Unsqueeze by Unsqueeze."""

    def __init__(self, verbose: int = 0, priority: int = 0):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Unsqueeze" or node.domain != "":
            return self.none()
        if g.is_used_more_than_once(node.output[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        next_nodes = g.next_nodes(node.output[0])
        if len(next_nodes) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        next_node = next_nodes[0]
        if next_node.op_type != "Unsqueeze" or node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if next_node.input[0] != node.output[0]:
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [node, next_node], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node: NodeProto,
        next_node: NodeProto,
    ) -> List[NodeProto]:
        axis1 = g.get_constant_or_attribute(node, "axis", 1)
        axis2 = g.get_constant_or_attribute(next_node, "axis", 1)
        ma = max(axis1.max(), axis2.max()) + 1
        all_axes = list(range(ma))
        axis1 = axis1.reshape((-1,))
        axis2 = axis2.reshape((-1,))
        for a in axis1[::-1]:
            all_axes.insert(a, -2)
        for a in axis2[::-1]:
            all_axes.insert(a, -2)
        new_axes = [i for i, a in enumerate(all_axes) if a == -2]
        new_axis = g.make_initializer(
            "",
            np.array(new_axes, dtype=np.int64),
            source="UnsqueezeUnsqueezePattern.apply.new_axis",
        )
        new_node = g.make_node(
            "Unsqueeze",
            [node.input[0], new_axis],
            next_node.output,
            name=f"{self.__class__.__name__}--{node.name}",
            doc_string=next_node.doc_string,
        )
        return [new_node]


class SqueezeAddPattern(PatternOptimization):
    """Replaces the sequence Add(Squeeze, Squeeze) by Squeeze(Add)."""

    def __init__(self, verbose: int = 0, priority: int = 0):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Add" or node.domain != "" or g.builder.main_opset < 13:
            return self.none()
        node_before = [g.node_before(node.input[0]), g.node_before(node.input[1])]
        if (
            not node_before[0]
            or not node_before[1]
            or node_before[0].op_type != "Squeeze"
            or node_before[1].op_type != "Squeeze"
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        if len(node_before[0].input) == 2:
            s1 = g.builder.value_as_shape(node_before[0].input[1])
        else:
            if not g.has_shape(node_before[0].input[0]) or g.get_shape(
                node_before[0].input[0]
            ) != (1,):
                return self.none(node, inspect.currentframe().f_lineno)
            s1 = (0,)

        if len(node_before[1].input) == 2:
            s2 = g.builder.value_as_shape(node_before[1].input[1])
        else:
            if not g.has_shape(node_before[1].input[0]) or g.get_shape(
                node_before[1].input[0]
            ) != (1,):
                return self.none(node, inspect.currentframe().f_lineno)
            s2 = (0,)

        if s1 is None or s2 is None or s1 != s2:
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, [*node_before, node], self.apply)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        squeeze1: NodeProto,
        squeeze2: NodeProto,
        add: NodeProto,
    ) -> List[NodeProto]:
        new_name = g.unique_name(f"{self.__class__.__name__}_{add.output[0]}")
        new_nodes = [
            g.make_node(
                "Add",
                [squeeze1.input[0], squeeze2.input[0]],
                [new_name],
                name=f"{self.__class__.__name__}--{add.name}",
                doc_string=add.doc_string,
            ),
            g.make_node(
                "Squeeze",
                [new_name, *squeeze1.input[1:]],
                add.output,
                name=f"{self.__class__.__name__}--{squeeze1.name}",
                doc_string=squeeze1.doc_string,
            ),
        ]
        if g.is_used_more_than_once(add.input[1]):
            new_nodes = [squeeze2, *new_nodes]
        if g.is_used_more_than_once(add.input[0]):
            new_nodes = [squeeze1, *new_nodes]
        return new_nodes
