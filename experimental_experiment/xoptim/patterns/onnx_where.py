import inspect
from typing import List, Optional
import numpy as np
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization


class WhereAddPattern(PatternOptimization):
    """
    Replaces the sequence Add(X, Where(bool_mask, 0, -inf)) -> Where(bool_mask, X, -inf).
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Where" or node.domain != "":
            return self.none()
        if not g.is_constant_scalar(node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant_scalar(node.input[2]):
            return self.none(node, inspect.currentframe().f_lineno)
        cst1 = g.get_constant_scalar(node.input[1])
        cst2 = g.get_constant_scalar(node.input[2])
        if cst1 is None or cst2 is None:
            return self.none(node, inspect.currentframe().f_lineno)
        if cst1 != 0:
            return self.none(node, inspect.currentframe().f_lineno)
        if not np.isinf(cst2):
            return self.none(node, inspect.currentframe().f_lineno)

        add_nodes = g.next_nodes(node.output[0])
        if len(add_nodes) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        if add_nodes[0].op_type != "Add" or add_nodes[0].domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, [node, add_nodes[0]], self.apply, insert_at=add_nodes[0])

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        where_node: NodeProto,
        add_node: NodeProto,
    ) -> List[NodeProto]:
        where_input1 = add_node.input[1 if add_node.input[0] == where_node.output[0] else 0]
        return [
            g.make_node(
                "Where",
                [where_node.input[0], where_input1, where_node.input[2]],
                [add_node.output[0]],
                name=f"{self.__class__.__name__}--{where_node.name}",
                doc_string=where_node.doc_string,
            )
        ]
