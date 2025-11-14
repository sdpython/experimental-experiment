import inspect
from typing import List, Optional
from onnx import NodeProto, TensorProto
from ..patterns_api import MatchResult, PatternOptimization


class MissingRangePattern(PatternOptimization):
    """Replaces Range by Cast Range Cast because of some missing kernels."""

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Range" or node.domain != "":
            return self.none()
        if not g.has_type(node.input[0]) or g.get_type(node.input[0]) in {
            TensorProto.INT32,
            TensorProto.INT64,
            TensorProto.FLOAT,
        }:
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, [node], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node: NodeProto,
    ) -> List[NodeProto]:
        to = g.get_type(node.input[0])
        other_to = (
            TensorProto.FLOAT
            if to in {TensorProto.FLOAT16, TensorProto.BFLOAT16}
            else TensorProto.INT32
        )
        nodes = []
        new_inputs = []
        for i in node.input:
            n = g.unique_name(f"{self.__class__.__name__}--{i}")
            nodes.append(
                g.make_node(
                    "Cast", [i], [n], to=other_to, name=f"{self.__class__.__name__}--Cast"
                )
            )
            new_inputs.append(n)
        n = g.unique_name(f"{self.__class__.__name__}--{i}")
        nodes.append(
            g.make_node(
                node.op_type,
                new_inputs,
                [n],
                domain=node.domain,
                name=f"{self.__class__.__name__}",
            )
        )
        nodes.append(
            g.make_node(
                "Cast", [n], [node.output[0]], to=to, name=f"{self.__class__.__name__}--Cast"
            )
        )
        return nodes


class MissingCosSinPattern(PatternOptimization):
    """Replaces Cos/Sin by Cast Cos/Sin Cast because of some missing kernels."""

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type not in {"Cos", "Sin", "Log"} or node.domain != "":
            return self.none()
        if not g.has_type(node.input[0]) or g.get_type(node.input[0]) in {
            TensorProto.DOUBLE,
            TensorProto.FLOAT16,
            TensorProto.FLOAT,
        }:
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, [node], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node: NodeProto,
    ) -> List[NodeProto]:
        to = g.get_type(node.input[0])
        other_to = TensorProto.FLOAT
        n1 = g.unique_name(f"{self.__class__.__name__}--{node.input[0]}")
        n2 = g.unique_name(f"{self.__class__.__name__}--{node.output[0]}")
        return [
            g.make_node(
                "Cast",
                [node.input[0]],
                [n1],
                to=other_to,
                name=f"{self.__class__.__name__}--Cast",
            ),
            g.make_node(
                node.op_type,
                [n1],
                [n2],
                domain=node.domain,
                name=f"{self.__class__.__name__}",
            ),
            g.make_node(
                "Cast",
                [n2],
                [node.output[0]],
                to=to,
                name=f"{self.__class__.__name__}--Cast",
            ),
        ]
