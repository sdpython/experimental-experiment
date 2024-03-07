import inspect
from typing import List, Optional
from onnx import NodeProto
from .patterns_api import MatchResult, PatternOptimization


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
