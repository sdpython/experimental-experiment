import inspect
from typing import List, Optional
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization


class ClipClipPattern(PatternOptimization):
    """
    Merges consecutive clips if one is defining min and the other max.
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Clip" or node.domain != "":
            return self.none()
        before = g.node_before(node.input[0])
        if (
            before is None
            or g.is_used_more_than_once(node.input[0])
            or before.op_type != "Clip"
            or before.domain != ""
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        min1 = before.input[1] if len(before.input) > 1 else ""
        min2 = node.input[1] if len(node.input) > 1 else ""
        if (min1 and min2) or (not min1 and not min2):
            return self.none(node, inspect.currentframe().f_lineno)
        max1 = before.input[2] if len(before.input) > 2 else ""
        max2 = node.input[2] if len(node.input) > 2 else ""
        if (max1 and max2) or (not max1 and not max2):
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [before, node], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        before: NodeProto,
        node: NodeProto,
    ) -> List[NodeProto]:
        # merges clips
        min1 = before.input[1] if len(before.input) > 1 else ""
        min2 = node.input[1] if len(node.input) > 1 else ""
        max1 = before.input[2] if len(before.input) > 2 else ""
        max2 = node.input[2] if len(node.input) > 2 else ""

        return [
            g.make_node(
                "Clip",
                [before.input[0], min1 or min2, max1 or max2],
                node.output,
                name=f"{self.__class__.__name__}--{node.name}",
            )
        ]
