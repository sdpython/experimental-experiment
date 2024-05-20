import inspect
from typing import List, Optional
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization


class UnsqueezeEqualPattern(PatternOptimization):
    """
    Replaces the sequence R -> Equal -> Unsqueeze, R -> Unsqueeze,
    into R -> Unsqueeze -> Equal.
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Equal" or node.domain != "":
            return self.none()
        if not g.is_constant_scalar(node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)

        after = g.next_nodes(node.output[0])
        if len(after) != 1:
            return self.none(node, inspect.currentframe().f_lineno)

        next_path = g.next_nodes(node.input[0])
        if len(next_path) != 2:
            return self.none(node, inspect.currentframe().f_lineno)

        if next_path[0].op_type == node.op_type and next_path[1].op_type == "Unsqueeze":
            if next_path[1].input[1] != after[0].input[1]:
                return self.none(node, inspect.currentframe().f_lineno)
            return MatchResult(self, [next_path[1], node, after[0]], self.apply)
        if next_path[1].op_type == node.op_type and next_path[0].op_type == "Unsqueeze":
            if next_path[0].input[1] != after[0].input[1]:
                return self.none(node, inspect.currentframe().f_lineno)
            return MatchResult(self, [next_path[0], node, after[0]], self.apply)
        return self.none(node, inspect.currentframe().f_lineno)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node_unsqueeze: NodeProto,
        node_equal: NodeProto,
        node_equal_unsqueeze: NodeProto,
    ) -> List[NodeProto]:

        return [
            node_unsqueeze,
            g.make_node(
                node_equal.op_type,
                [node_unsqueeze.output[0], node_equal.input[1]],
                [node_equal_unsqueeze.output[0]],
                domain=node_equal.domain,
                name=f"{self.__class__.__name__}--{node_equal.name}",
            ),
        ]
