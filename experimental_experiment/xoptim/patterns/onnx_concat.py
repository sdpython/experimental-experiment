import inspect
from typing import List, Optional
import numpy as np
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization


class ConcatGatherPattern(PatternOptimization):
    """
    Checks if Gather(Concat) can be replaced by Identity.
    """

    def __init__(self, verbose: int = 0, priority: int = 0):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Gather" or node.domain != "":
            return self.none()
        if not g.is_constant(node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        cst = g.get_computed_constant(node.input[1])
        if cst is None or cst.dtype != np.int64 or cst.shape != (1,):
            return self.none(node, inspect.currentframe().f_lineno)
        before = g.node_before(node.input[0])
        if before.op_type != "Concat":
            return self.none(node, inspect.currentframe().f_lineno)
        if any(not g.has_shape(i) for i in before.input):
            return self.none(node, inspect.currentframe().f_lineno)
        if any(g.get_shape(i) != (1,) for i in before.input):
            return self.none(node, inspect.currentframe().f_lineno)
        assert cst[0] < len(before.input), (
            f"Concat concatenates many dimensions into one but "
            f"cst={cst} and before.input={before.input}"
        )
        return MatchResult(self, [before, node], self.apply)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        concat_node: NodeProto,
        gather_node: NodeProto,
    ) -> List[NodeProto]:
        index = g.get_constant_scalar(gather_node.input[1])
        new_node = g.make_node(
            "Identity",
            [concat_node.input[index]],
            gather_node.output,
            name=f"{self.__class__.__name__}--{gather_node.name}",
            doc_string=gather_node.doc_string,
        )
        return (
            [concat_node, new_node]
            if g.is_used_more_than_once(concat_node.output[0])
            else [new_node]
        )
