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

    Model with nodes to be fused:

    .. gdot::
        :script: DOT-SECTION
        :process:

        from experimental_experiment.doc import to_dot
        import numpy as np
        import ml_dtypes
        import onnx
        import onnx.helper as oh
        import onnx.numpy_helper as onh
        from onnx_array_api.translate_api.make_helper import make_node_extended

        opset_imports = [
            oh.make_opsetid("", 20),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info(
                "linear_5", onnx.TensorProto.FLOAT16, shape=(4, 512, 16384)
            )
        )
        nodes.append(
            make_node_extended(
                "Constant",
                [],
                ["init10_s1_5"],
                value=onh.from_array(np.array([3.0], dtype=np.float16), name="value"),
            )
        )
        nodes.append(
            make_node_extended(
                "Constant",
                [],
                ["init10_s_8"],
                value=onh.from_array(
                    np.array(0.044708251953125, dtype=np.float16), name="value"
                ),
            )
        )
        nodes.append(
            make_node_extended(
                "Constant",
                [],
                ["init10_s_9"],
                value=onh.from_array(np.array(0.7978515625, dtype=np.float16), name="value"),
            )
        )
        nodes.append(
            make_node_extended(
                "Constant",
                [],
                ["init10_s_10"],
                value=onh.from_array(np.array(1.0, dtype=np.float16), name="value"),
            )
        )
        nodes.append(
            make_node_extended(
                "Constant",
                [],
                ["init10_s_7"],
                value=onh.from_array(np.array(0.5, dtype=np.float16), name="value"),
            )
        )
        nodes.append(make_node_extended("Pow", ["linear_5", "init10_s1_5"], ["pow_1"]))
        nodes.append(make_node_extended("Mul", ["pow_1", "init10_s_8"], ["_onx_mul05"]))
        nodes.append(make_node_extended("Add", ["linear_5", "_onx_mul05"], ["add_4"]))
        nodes.append(make_node_extended("Mul", ["add_4", "init10_s_9"], ["_onx_mul06"]))
        nodes.append(make_node_extended("Tanh", ["_onx_mul06"], ["tanh"]))
        nodes.append(make_node_extended("Add", ["tanh", "init10_s_10"], ["add_5"]))
        nodes.append(make_node_extended("Mul", ["linear_5", "init10_s_7"], ["_onx_mul04"]))
        nodes.append(make_node_extended("Mul", ["_onx_mul04", "add_5"], ["mul_4"]))
        outputs.append(
            oh.make_tensor_value_info("mul_4", onnx.TensorProto.FLOAT16, shape=(4, 512, 16384))
        )
        graph = oh.make_graph(
            nodes,
            "pattern",
            inputs,
            outputs,
            initializers,
            sparse_initializer=sparse_initializers,
        )
        model = oh.make_model(graph, functions=functions, opset_imports=opset_imports)

        print("DOT-SECTION", to_dot(model))

    Outcome of the fusion:

    .. gdot::
        :script: DOT-SECTION
        :process:

        from experimental_experiment.doc import to_dot
        import numpy as np
        import ml_dtypes
        import onnx
        import onnx.helper as oh
        import onnx.numpy_helper as onh
        from onnx_array_api.translate_api.make_helper import make_node_extended

        opset_imports = [
            oh.make_opsetid("", 20),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info(
                "linear_5", onnx.TensorProto.FLOAT16, shape=(4, 512, 16384)
            )
        )
        nodes.append(make_node_extended("Gelu", ["linear_5"], ["mul_4"], approximate="tanh"))
        outputs.append(
            oh.make_tensor_value_info("mul_4", onnx.TensorProto.FLOAT16, shape=(4, 512, 16384))
        )
        graph = oh.make_graph(
            nodes,
            "pattern",
            inputs,
            outputs,
            initializers,
            sparse_initializer=sparse_initializers,
        )
        model = oh.make_model(graph, functions=functions, opset_imports=opset_imports)

        print("DOT-SECTION", to_dot(model))
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

    Model with nodes to be fused:

    .. gdot::
        :script: DOT-SECTION
        :process:

        from experimental_experiment.doc import to_dot
        import numpy as np
        import ml_dtypes
        import onnx
        import onnx.helper as oh
        import onnx.numpy_helper as onh
        from onnx_array_api.translate_api.make_helper import make_node_extended

        opset_imports = [
            oh.make_opsetid("", 18),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(oh.make_tensor_value_info("X1", onnx.TensorProto.FLOAT, shape=(3, 3)))
        nodes.append(
            make_node_extended(
                "Constant",
                [],
                ["zero"],
                value=onh.from_array(np.array([0.0], dtype=np.float32), name="value"),
            )
        )
        nodes.append(
            make_node_extended(
                "Constant",
                [],
                ["slope2"],
                value=onh.from_array(
                    np.array([-0.33000001311302185], dtype=np.float32), name="value"
                ),
            )
        )
        nodes.append(make_node_extended("Greater", ["X1", "zero"], ["xpos2"]))
        nodes.append(make_node_extended("Mul", ["X1", "slope2"], ["xmul2"]))
        nodes.append(make_node_extended("Where", ["xpos2", "X1", "xmul2"], ["Y"]))
        outputs.append(oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape=(3, 3)))
        graph = oh.make_graph(
            nodes,
            "pattern",
            inputs,
            outputs,
            initializers,
            sparse_initializer=sparse_initializers,
        )
        model = oh.make_model(graph, functions=functions, opset_imports=opset_imports)

        print("DOT-SECTION", to_dot(model))

    Outcome of the fusion:

    .. gdot::
        :script: DOT-SECTION
        :process:

        from experimental_experiment.doc import to_dot
        import numpy as np
        import ml_dtypes
        import onnx
        import onnx.helper as oh
        import onnx.numpy_helper as onh
        from onnx_array_api.translate_api.make_helper import make_node_extended

        opset_imports = [
            oh.make_opsetid("", 18),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(oh.make_tensor_value_info("X1", onnx.TensorProto.FLOAT, shape=(3, 3)))
        nodes.append(make_node_extended("LeakyRelu", ["X1"], ["Y"], alpha=-0.33000001311302185))
        outputs.append(oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape=(3, 3)))
        graph = oh.make_graph(
            nodes,
            "pattern",
            inputs,
            outputs,
            initializers,
            sparse_initializer=sparse_initializers,
        )
        model = oh.make_model(graph, functions=functions, opset_imports=opset_imports)

        print("DOT-SECTION", to_dot(model))
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
    """Detects one decomposed version of SoftmaxCrossEntropyLoss."""

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
