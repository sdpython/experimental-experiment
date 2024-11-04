import inspect
from typing import List, Optional
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization


class DropoutPattern(PatternOptimization):
    """
    Checks that a Cast is really needed.
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Dropout" or node.domain != "":
            return None

        for o in node.output[1:]:
            if o and g.is_used(o):
                return self.none(node, inspect.currentframe().f_lineno)

        if not (
            len(node.input) >= 3
            and node.input[2] != ""
            and g.is_constant_scalar(node.input[2])
            and not g.get_constant_scalar(node.input[2])
        ):
            return MatchResult(self, [node], self.apply, insert_at=node)

        if (
            len(node.input) >= 2
            and node.input[1] != ""
            and g.is_constant_scalar(node.input[2])
            and g.get_constant_scalar(node.input[2]) != 0
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [node], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        dropout_node: NodeProto,
    ) -> List[NodeProto]:
        return [
            g.make_node(
                "Identity",
                dropout_node.input[:1],
                dropout_node.output[:1],
                name=f"{self.__class__.__name__}--{dropout_node.name}",
                doc_string=dropout_node.doc_string,
            )
        ]
