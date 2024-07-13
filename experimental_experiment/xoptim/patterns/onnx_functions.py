import inspect
from typing import List, Optional
from onnx import NodeProto
from ..patterns_api import EasyPatternOptimization


class GeluPattern(EasyPatternOptimization):
    """
    Detects the decomposed version of Gelu with Tanh

    .. math::

        y = \\frac{x}{2} \\left(1 + \\tanh\\left(\\sqrt{\\frac{2}{\\pi}} (x + 0.044715 * x^3)\\rigth)\\rigth)
    """

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
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        x,
        c3,
        c04,
        cpi,
        one,
        c2,
    ):
        return g.op.Gelu(x, approximate="tanh")

    def validate_mapping(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        deleted_nodes: List[NodeProto],
        pattern_nodes: Optional[List[NodeProto]] = None,
    ) -> bool:
        assert (
            len(deleted_nodes) == 8
        ), f"Unexpected pattern length {len(deleted_nodes)}"
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
