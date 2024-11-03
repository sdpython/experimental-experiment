import inspect
from typing import List, Optional
from onnx import NodeProto, TensorProto
from ..patterns_api import EasyPatternOptimization


class GeluPattern(EasyPatternOptimization):
    """
    Detects the decomposed version of Gelu with Tanh

    .. math::

        y = \\frac{x}{2}
        \\left(1 + \\tanh\\left(\\sqrt{\\frac{2}{\\pi}} (x + 0.044715 * x^3)\\right)\\right)
    """

    def __init__(
        self, verbose: int = 0, priority: int = 0, min_opset: int = 20, domain: str = ""
    ):
        super().__init__(verbose, priority, min_opset=min_opset)
        self.domain = domain

    def match_pattern(self, g: "GraphBuilder", x, c3, c04, cpi, one, c2):  # noqa: F821
        x3 = g.op.Pow(x, c3)  # 3
        cx3 = g.op.Mul(x3, c04)  # 0.044715
        add = g.op.Add(x, cx3)
        addm = g.op.Mul(add, cpi)  # 0.7978515625 = 2/pi
        tanh = g.op.Tanh(addm)
        tanh1 = g.op.Add(tanh, one)  # 1
        x2 = g.op.Mul(x, c2)  # 0.5
        return g.op.Mul(x2, tanh1)

    def apply_pattern(
        self,
        g: "GraphBuilder",  # noqa: F821
        x,
        c3,
        c04,
        cpi,
        one,
        c2,
    ):
        return g.op.Gelu(x, approximate="tanh", domain=self.domain)

    def validate_mapping(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        deleted_nodes: List[NodeProto],
        pattern_nodes: Optional[List[NodeProto]] = None,
    ) -> bool:
        assert len(deleted_nodes) == 8, f"Unexpected pattern length {len(deleted_nodes)}"
        assert deleted_nodes[0].op_type == "Pow", f"-- {deleted_nodes[0]}"
        c3 = deleted_nodes[0].input[1]
        assert deleted_nodes[1].op_type == "Mul", f"-- {deleted_nodes[1]}"
        cx3 = deleted_nodes[1].input[1]
        assert deleted_nodes[3].op_type == "Mul", f"-- {deleted_nodes[3]}"
        cpi = deleted_nodes[3].input[1]
        assert deleted_nodes[5].op_type == "Add", f"-- {deleted_nodes[5]}"
        one = deleted_nodes[5].input[1]
        assert deleted_nodes[6].op_type == "Mul", f"-- {deleted_nodes[5]}"
        c2 = deleted_nodes[6].input[1]

        node = deleted_nodes[0]

        if not g.is_constant_scalar(c3) or g.get_constant_scalar(c3) != 3:
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant_scalar(cx3) or g.get_constant_scalar(cx3) not in (
            0.044715,
            0.044708251953125,
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant_scalar(cpi) or g.get_constant_scalar(cpi) != 0.7978515625:
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant_scalar(one) or g.get_constant_scalar(one) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant_scalar(c2) or g.get_constant_scalar(c2) != 0.5:
            return self.none(node, inspect.currentframe().f_lineno)
        return True


class LeakyReluPattern(EasyPatternOptimization):
    """
    Detects the decomposed version of LeakyRelu.
    """

    def __init__(self, verbose: int = 0, priority: int = 0, min_opset: int = 6):
        super().__init__(verbose, priority, min_opset=min_opset)

    def match_pattern(self, g: "GraphBuilder", x, zero, slope):  # noqa: F821
        return g.op.Where(g.op.Greater(x, zero), x, g.op.Mul(x, slope))

    def apply_pattern(
        self,
        g: "GraphBuilder",  # noqa: F821
        x,
        zero,
        slope,
    ):
        # g is not the GraphBuilder for the main graph.
        return g.op.LeakyRelu(x, alpha=self.get_validate_param("slope"))

    def validate_mapping(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        deleted_nodes: List[NodeProto],
        pattern_nodes: Optional[List[NodeProto]] = None,
    ) -> bool:
        assert len(deleted_nodes) == 3, f"Unexpected pattern length {len(deleted_nodes)}"
        assert deleted_nodes[2].op_type == "Where", f"-- {deleted_nodes[0]}"
        greater, mul = (
            (deleted_nodes[0], deleted_nodes[1])
            if deleted_nodes[0].op_type == "Greater"
            else (deleted_nodes[1], deleted_nodes[0])
        )
        zero = greater.input[1]
        slope = mul.input[1]

        if not g.is_constant_scalar(zero) or g.get_constant_scalar(zero) != 0:
            return self.none(greater, inspect.currentframe().f_lineno)
        if not g.is_constant_scalar(slope):
            return self.none(mul, inspect.currentframe().f_lineno)
        self.add_validate_param("slope", g.get_constant_scalar(slope))
        return True


class SoftmaxCrossEntropyLossCastPattern(EasyPatternOptimization):
    """
    Detects one decomposed version of SoftmaxCrossEntropyLoss.
    """

    def __init__(
        self, verbose: int = 0, priority: int = 0, min_opset: int = 14, domain: str = ""
    ):
        super().__init__(verbose, priority, min_opset=min_opset)
        self.domain = domain

    def match_pattern(
        self,
        g: "GraphBuilder",  # noqa: F821
        X,
        indices,
        axis,
        zerof,
        zeroi,
        b,
    ):
        neq1 = g.op.Not(g.op.Equal(indices, b))
        wh1 = g.op.Where(neq1, indices, zeroi)
        uns = g.op.Unsqueeze(wh1, axis)
        ge = g.op.GatherElements(g.op.LogSoftmax(X, axis=1), uns, axis=1)
        wh2 = g.op.Where(neq1, g.op.Neg(g.op.Squeeze(ge, axis)), zerof)
        denominator = g.op.Cast(
            g.op.ReduceSum(
                g.op.Cast(neq1, to=TensorProto.FLOAT),
                keepdims=0,
            ),
            to=TensorProto.FLOAT16,
        )
        numerator = g.op.Cast(
            g.op.ReduceSum(
                g.op.Cast(wh2, to=TensorProto.FLOAT),
                keepdims=0,
            ),
            to=TensorProto.FLOAT16,
        )
        return g.op.Div(numerator, denominator)

    @classmethod
    def apply_pattern(
        cls,
        g: "GraphBuilder",  # noqa: F821
        X,
        indices,
        axis,
        zerof,
        zeroi,
        b,
    ):
        return g.op.SoftmaxCrossEntropyLoss(X, indices, ignore_index=-100, reduction="mean")

    def validate_mapping(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        deleted_nodes: List[NodeProto],
        pattern_nodes: Optional[List[NodeProto]] = None,
    ) -> bool:
        assert len(deleted_nodes) == 16, f"Unexpected pattern length {len(deleted_nodes)}"
        node = deleted_nodes[-1]

        for n in deleted_nodes:
            if n.op_type in {"Squeeze", "Unsqueeze"}:
                c = n.input[1]
                if not g.is_constant_scalar(c) or g.get_constant_scalar(c) != 1:
                    return self.none(node, inspect.currentframe().f_lineno)
                continue
            if n.op_type in {"Equal"}:
                c = n.input[1]
                if not g.is_constant_scalar(c) or g.get_constant_scalar(c) != -100:
                    return self.none(node, inspect.currentframe().f_lineno)
                continue
        return True
