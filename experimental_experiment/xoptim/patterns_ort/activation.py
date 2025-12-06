import inspect
from typing import List, Optional
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization, EasyPatternOptimization
from ..patterns.onnx_functions import GeluPattern


class BiasGeluPattern(PatternOptimization):
    """
    Replaces by ``y = BiasGelu(x, B)``::

        t = x + B
        y = t ( Erf(1 / t) + 1)

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

        opset_imports = [
            oh.make_opsetid("", 18),
            oh.make_opsetid("com.microsoft", 1),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(oh.make_tensor_value_info("B", onnx.TensorProto.FLOAT, shape=(8,)))
        inputs.append(
            oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape=(2, 2, 4, 8))
        )
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["B"],
                value=onh.from_array(
                    np.array(
                        [
                            0.10000000149011612,
                            0.20000000298023224,
                            0.30000001192092896,
                            0.4000000059604645,
                            0.5,
                            0.6000000238418579,
                            -0.4000000059604645,
                            -0.10000000149011612,
                        ],
                        dtype=np.float32,
                    ),
                    name="value",
                ),
            )
        )
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["sq2"],
                value=onh.from_array(np.array([1.4140625], dtype=np.float32), name="value"),
            )
        )
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["one"],
                value=onh.from_array(np.array([1.0], dtype=np.float32), name="value"),
            )
        )
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["half"],
                value=onh.from_array(np.array([0.5], dtype=np.float32), name="value"),
            )
        )
        nodes.append(oh.make_node("Add", ["X", "B"], ["xb"]))
        nodes.append(oh.make_node("Div", ["xb", "sq2"], ["xbinv"]))
        nodes.append(oh.make_node("Erf", ["xbinv"], ["xerf"]))
        nodes.append(oh.make_node("Add", ["xerf", "one"], ["xerf1"]))
        nodes.append(oh.make_node("Mul", ["xb", "xerf1"], ["y2"]))
        nodes.append(oh.make_node("Mul", ["y2", "half"], ["Y"]))
        outputs.append(
            oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape=(2, 2, 4, 8))
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

        opset_imports = [
            oh.make_opsetid("", 18),
            oh.make_opsetid("com.microsoft", 1),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(oh.make_tensor_value_info("B", onnx.TensorProto.FLOAT, shape=(8,)))
        inputs.append(
            oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape=(2, 2, 4, 8))
        )
        nodes.append(oh.make_node("BiasGelu", ["X", "B"], ["Y"], domain="com.microsoft"))
        outputs.append(
            oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape=(2, 2, 4, 8))
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

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Erf" or node.domain != "":
            return self.none()
        if g.is_used_more_than_once(node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)

        div = g.node_before(node.input[0])
        if (
            not g.is_constant_scalar(div.input[1])
            or g.get_constant_scalar(div.input[1]) != 1.4140625
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        add = g.node_before(div.input[0])
        if add.op_type != "Add" or add.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant(add.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        add1_nexts = g.next_nodes(add.output[0])
        if len(add1_nexts) != 2:
            return self.none(node, inspect.currentframe().f_lineno)

        add_next = g.next_nodes(node.output[0])
        if len(add_next) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        add_1 = add_next[0]
        if add_1.op_type != "Add" or add_1.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant_scalar(add_1.input[1]) or g.get_constant_scalar(add_1.input[1]) != 1:
            return self.none(node, inspect.currentframe().f_lineno)

        muls = g.next_nodes(add_1.output[0])
        if len(muls) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        mul = muls[0]
        if mul.op_type != "Mul" or mul.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if set(mul.input) != {add.output[0], add_1.output[0]}:
            return self.none(node, inspect.currentframe().f_lineno)

        halfs = g.next_nodes(mul.output[0])
        if len(halfs) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        half = halfs[0]
        if half.op_type != "Mul" or half.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        index = 1 if half.input[0] == mul.output[0] else 0
        if (
            not g.is_constant_scalar(half.input[index])
            or g.get_constant_scalar(half.input[index]) != 0.5
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [add, div, node, add_1, mul, half], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        add_node: NodeProto,
        div_node: NodeProto,
        erf_node: NodeProto,
        add_1_node: NodeProto,
        mul_node: NodeProto,
        half_node: NodeProto,
    ) -> List[NodeProto]:
        return [
            g.make_node(
                "BiasGelu",
                add_node.input,
                half_node.output,
                domain="com.microsoft",
                doc_string=erf_node.doc_string,
                name=f"{self.__class__.__name__}--{erf_node.name}",
            )
        ]


class GeluOrtPattern(GeluPattern):
    """
    Detects the decomposed version of Gelu with Tanh

    .. math::

        y = \\frac{x}{2} \\left(1 + \\tanh\\left(\\sqrt{\\frac{2}{\\pi}}
        (x + 0.044715 * x^3)\\right)\\right)

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

        opset_imports = [
            oh.make_opsetid("", 18),
            oh.make_opsetid("com.microsoft", 1),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info(
                "linear_73", onnx.TensorProto.FLOAT16, shape=(4, 512, 128)
            )
        )
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["init10_s1_63"],
                value=onh.from_array(np.array([3.0], dtype=np.float16), name="value"),
            )
        )
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["init10_s_89"],
                value=onh.from_array(
                    np.array(0.044708251953125, dtype=np.float16), name="value"
                ),
            )
        )
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["init10_s_90"],
                value=onh.from_array(np.array(0.7978515625, dtype=np.float16), name="value"),
            )
        )
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["init10_s_91"],
                value=onh.from_array(np.array(1.0, dtype=np.float16), name="value"),
            )
        )
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["init10_s_88"],
                value=onh.from_array(np.array(0.5, dtype=np.float16), name="value"),
            )
        )
        nodes.append(oh.make_node("Pow", ["linear_73", "init10_s1_63"], ["pow_13"]))
        nodes.append(oh.make_node("Mul", ["pow_13", "init10_s_89"], ["_onx_mul064"]))
        nodes.append(oh.make_node("Add", ["linear_73", "_onx_mul064"], ["add_62"]))
        nodes.append(oh.make_node("Mul", ["add_62", "init10_s_90"], ["_onx_mul065"]))
        nodes.append(oh.make_node("Tanh", ["_onx_mul065"], ["tanh_12"]))
        nodes.append(oh.make_node("Add", ["tanh_12", "init10_s_91"], ["add_63"]))
        nodes.append(oh.make_node("Mul", ["linear_73", "init10_s_88"], ["_onx_mul063"]))
        nodes.append(oh.make_node("Mul", ["_onx_mul063", "add_63"], ["mul_52"]))
        outputs.append(
            oh.make_tensor_value_info("mul_52", onnx.TensorProto.FLOAT16, shape=(4, 512, 128))
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

        opset_imports = [
            oh.make_opsetid("", 18),
            oh.make_opsetid("com.microsoft", 1),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info(
                "linear_73", onnx.TensorProto.FLOAT16, shape=(4, 512, 128)
            )
        )
        nodes.append(
            oh.make_node(
                "Gelu", ["linear_73"], ["mul_52"], domain="com.microsoft", approximate="tanh"
            )
        )
        outputs.append(
            oh.make_tensor_value_info("mul_52", onnx.TensorProto.FLOAT16, shape=(4, 512, 128))
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
        self,
        verbose: int = 0,
        priority: int = 0,
        min_opset: int = 1,
        domain: str = "com.microsoft",
    ):
        super().__init__(verbose, priority, min_opset=min_opset)
        self.domain = domain


class GeluErfPattern(EasyPatternOptimization):
    """
    Detects the decomposed version of Gelu with Erf.

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

        opset_imports = [
            oh.make_opsetid("", 18),
            oh.make_opsetid("com.microsoft", 1),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape=(2, 2, 4, 8))
        )
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["sq2"],
                value=onh.from_array(np.array([1.4140625], dtype=np.float32), name="value"),
            )
        )
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["one"],
                value=onh.from_array(np.array([1.0], dtype=np.float32), name="value"),
            )
        )
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["half"],
                value=onh.from_array(np.array([0.5], dtype=np.float32), name="value"),
            )
        )
        nodes.append(oh.make_node("Div", ["X", "sq2"], ["xd"]))
        nodes.append(oh.make_node("Erf", ["xd"], ["exd"]))
        nodes.append(oh.make_node("Add", ["exd", "one"], ["aexd"]))
        nodes.append(oh.make_node("Mul", ["X", "aexd"], ["y2"]))
        nodes.append(oh.make_node("Mul", ["half", "y2"], ["Y"]))
        outputs.append(
            oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape=(2, 2, 4, 8))
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

        opset_imports = [
            oh.make_opsetid("", 18),
            oh.make_opsetid("com.microsoft", 1),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape=(2, 2, 4, 8))
        )
        nodes.append(oh.make_node("Gelu", ["X"], ["Y"], domain="com.microsoft"))
        outputs.append(
            oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape=(2, 2, 4, 8))
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

    def __init__(self, verbose: int = 0, priority: int = 0, min_opset: int = 1):
        super().__init__(verbose, priority, min_opset=min_opset)

    def match_pattern(self, g: "GraphBuilder", x, cst2, one, c05):  # noqa: F821
        xd = g.op.Div(x, cst2)  # 1.4140625
        exd = g.op.Erf(xd)
        aexd = g.op.Add(exd, one)  # 1
        mul = g.op.Mul(x, aexd)
        return g.op.Mul(c05, mul)  # 0.5

    def apply_pattern(self, g: "GraphBuilder", x, cst2, one, c05):  # noqa: F821
        return g.anyop.Gelu(x, domain="com.microsoft")

    def validate_mapping(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        deleted_nodes: List[NodeProto],
        pattern_nodes: Optional[List[NodeProto]] = None,
    ) -> bool:
        assert len(deleted_nodes) == 5, f"Unexpected pattern length {len(deleted_nodes)}"
        assert deleted_nodes[0].op_type == "Div", f"-- {deleted_nodes[0]}"
        cst2 = deleted_nodes[0].input[1]
        assert deleted_nodes[2].op_type == "Add", f"-- {deleted_nodes[2]}"
        one = deleted_nodes[2].input[1]
        assert deleted_nodes[4].op_type == "Mul", f"-- {deleted_nodes[4]}"
        c05 = deleted_nodes[4].input[0]

        node = deleted_nodes[1]
        if not g.is_constant_scalar(cst2) or g.get_constant_scalar(cst2) != 1.4140625:
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant_scalar(one) or g.get_constant_scalar(one) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant_scalar(c05) or g.get_constant_scalar(c05) != 0.5:
            return self.none(node, inspect.currentframe().f_lineno)
        return True


class FastGeluPattern(PatternOptimization):
    """
    Replaces Gelu by FastGelu.

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

        opset_imports = [
            oh.make_opsetid("", 20),
            oh.make_opsetid("com.microsoft", 1),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info(
                "linear_65", onnx.TensorProto.FLOAT16, shape=(4, 512, 16384)
            )
        )
        nodes.append(oh.make_node("Gelu", ["linear_65"], ["mul_44"], approximate="tanh"))
        outputs.append(
            oh.make_tensor_value_info("mul_44", onnx.TensorProto.FLOAT16, shape=(4, 512, 16384))
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

        opset_imports = [
            oh.make_opsetid("", 20),
            oh.make_opsetid("com.microsoft", 1),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info(
                "linear_65", onnx.TensorProto.FLOAT16, shape=(4, 512, 16384)
            )
        )
        nodes.append(
            oh.make_node("FastGelu", ["linear_65"], ["mul_44"], domain="com.microsoft")
        )
        outputs.append(
            oh.make_tensor_value_info("mul_44", onnx.TensorProto.FLOAT16, shape=(4, 512, 16384))
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

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Gelu" or node.domain not in ("", "com.microsoft"):
            return self.none()
        return MatchResult(self, [node], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        gelu_node: NodeProto,
    ) -> List[NodeProto]:
        return [
            g.make_node(
                "FastGelu",
                gelu_node.input,
                gelu_node.output,
                domain="com.microsoft",
                doc_string=gelu_node.doc_string,
                name=f"{self.__class__.__name__}--{gelu_node.name}",
            )
        ]


class BiasSoftmaxPattern(PatternOptimization):
    """
    Replaces Softmax(Add(x,y), axis=-1) by BiasSoftmax(x,y,axis=-1)

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

        opset_imports = [
            oh.make_opsetid("", 18),
            oh.make_opsetid("com.microsoft", 1),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape=(16, 8, 4, 8))
        )
        inputs.append(
            oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape=(16, 1, 4, 8))
        )
        nodes.append(oh.make_node("Add", ["X", "Y"], ["xy"]))
        nodes.append(oh.make_node("Softmax", ["xy"], ["Z"], axis=-1))
        outputs.append(
            oh.make_tensor_value_info("Z", onnx.TensorProto.FLOAT, shape=(16, 8, 4, 8))
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

        opset_imports = [
            oh.make_opsetid("", 18),
            oh.make_opsetid("com.microsoft", 1),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape=(16, 8, 4, 8))
        )
        inputs.append(
            oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape=(16, 1, 4, 8))
        )
        nodes.append(
            oh.make_node(
                "BiasSoftmax",
                ["X", "Y"],
                ["Z"],
                domain="com.microsoft",
                axis=-1,
                is_inner_broadcast=0,
            )
        )
        outputs.append(
            oh.make_tensor_value_info("Z", onnx.TensorProto.FLOAT, shape=(16, 8, 4, 8))
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

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if not g.has_processor("CUDA"):
            return self.none()
        if node.op_type != "Softmax" or node.domain != "":
            return self.none()
        if g.is_used_more_than_once(node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        atts = g.get_attributes_with_default(node, axis=-1)
        if atts["axis"] != -1:
            return self.none(node, inspect.currentframe().f_lineno)
        before = g.node_before(node.input[0])
        if before is None or before.op_type != "Add":
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, [before, node], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        add_node: NodeProto,
        softmax_node: NodeProto,
    ) -> List[NodeProto]:
        return [
            g.make_node(
                "BiasSoftmax",
                add_node.input,
                softmax_node.output,
                axis=-1,
                is_inner_broadcast=0,
                domain="com.microsoft",
                doc_string=softmax_node.doc_string,
                name=f"{self.__class__.__name__}--{softmax_node.name}",
            )
        ]


class QuickGeluPattern(PatternOptimization):
    """
    Replaces Mul(x, Sigmoid(x)) by QuickGelu(x, alpha=1)

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

        opset_imports = [
            oh.make_opsetid("", 18),
            oh.make_opsetid("com.microsoft", 1),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape=(1, 8, 6, 6))
        )
        nodes.append(oh.make_node("Sigmoid", ["X"], ["S"]))
        nodes.append(oh.make_node("Mul", ["X", "S"], ["Y"]))
        outputs.append(
            oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape=(1, 8, 6, 6))
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

        opset_imports = [
            oh.make_opsetid("", 18),
            oh.make_opsetid("com.microsoft", 1),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape=(1, 8, 6, 6))
        )
        nodes.append(
            oh.make_node("QuickGelu", ["X"], ["Y"], domain="com.microsoft", alpha=1.0)
        )
        outputs.append(
            oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape=(1, 8, 6, 6))
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

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Sigmoid" or node.domain != "":
            return self.none()
        if g.is_used_more_than_once(node.output[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        after = g.next_nodes(node.output[0])
        if not after or after[0].op_type != "Mul":
            return self.none(node, inspect.currentframe().f_lineno)
        mul_node = after[0]
        if node.input[0] not in mul_node.input:
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, [node, mul_node], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        sigmoid: NodeProto,
        mul_node: NodeProto,
    ) -> List[NodeProto]:
        return [
            g.make_node(
                "QuickGelu",
                sigmoid.input,
                mul_node.output,
                alpha=1.0,
                domain="com.microsoft",
                doc_string=sigmoid.doc_string,
                name=f"{self.__class__.__name__}--{sigmoid.name}",
            )
        ]
