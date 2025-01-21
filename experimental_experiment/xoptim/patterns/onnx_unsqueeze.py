import inspect
from typing import List, Optional
import numpy as np
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization


class SqueezeUnsqueezePattern(PatternOptimization):
    """Replaces the sequence Squeeze, Unsqueeze by the Identity."""

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
        if g.is_used_more_than_once(node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        node_before = g.node_before(node.input[0])
        if node_before.op_type != "Squeeze" or node_before.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        axes1 = g.get_computed_constant(node_before.input[1])
        axes2 = g.get_computed_constant(node.input[1])
        if axes1 is None or axes2 is None:
            return self.none(node, inspect.currentframe().f_lineno)
        if tuple(map(int, axes1)) != tuple(map(int, axes2)):
            return self.none(node, inspect.currentframe().f_lineno)
        if len(axes1) > 1 and tuple(map(int, axes1)) != tuple(
            range(min(axes1), max(axes1) + 1)
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, [node_before, node], self.apply, insert_at=node)

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
        return [new_node]


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
        new_axis = g.make_initializer(
            "", np.hstack([axis1, axis2]), source="UnsqueezeUnsqueezePattern.apply.new_axis"
        )
        new_node = g.make_node(
            "Unsqueeze",
            [node.input[0], new_axis],
            next_node.output,
            name=f"{self.__class__.__name__}--{node.name}",
            doc_string=next_node.doc_string,
        )
        return [new_node]
