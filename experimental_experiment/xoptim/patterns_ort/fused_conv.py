import inspect
from typing import List, Optional
from onnx import NodeProto, TensorProto
from ..patterns_api import MatchResult, PatternOptimization


class FusedConvPattern(PatternOptimization):
    """
    Replaces the Conv + Relu into FusedConv.

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
            oh.make_tensor_value_info("W", onnx.TensorProto.FLOAT, shape=(8, 8, 3, 3))
        )
        inputs.append(
            oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape=(1, 8, 6, 6))
        )
        inputs.append(oh.make_tensor_value_info("B", onnx.TensorProto.FLOAT, shape=(8,)))
        nodes.append(
            oh.make_node(
                "Conv",
                ["X", "W", "B"],
                ["c"],
                dilations=[1, 1],
                group=1,
                pads=[1, 1, 1, 1],
                strides=[1, 1],
            )
        )
        nodes.append(oh.make_node("Relu", ["c"], ["Y"]))
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
            oh.make_tensor_value_info("W", onnx.TensorProto.FLOAT, shape=(8, 8, 3, 3))
        )
        inputs.append(
            oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape=(1, 8, 6, 6))
        )
        inputs.append(oh.make_tensor_value_info("B", onnx.TensorProto.FLOAT, shape=(8,)))
        nodes.append(
            oh.make_node(
                "FusedConv",
                ["X", "W", "B"],
                ["Y"],
                domain="com.microsoft",
                activation="Relu",
                dilations=[1, 1],
                group=1,
                pads=[1, 1, 1, 1],
                strides=[1, 1],
            )
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

    def __init__(self, verbose: int = 0, priority: int = 2):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Conv" or node.domain != "":
            return self.none()

        next_nodes = g.next_nodes(node.output[0])
        if len(next_nodes) != 1:
            return self.none(node, inspect.currentframe().f_lineno)

        op_type = next_nodes[0].op_type
        if op_type != "Relu":
            return self.none(node, inspect.currentframe().f_lineno)

        # FusedConv only exists for float32.
        dtypes = [(g.get_type(i) if g.has_type(i) else None) for i in node.input]
        if TensorProto.FLOAT not in dtypes:
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [node, next_nodes[0]], self.apply, insert_at=next_nodes[0])

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node: NodeProto,
        node_act: NodeProto,
    ) -> List[NodeProto]:
        fc = g.make_node(
            "FusedConv",
            node.input,
            node_act.output,
            domain="com.microsoft",
            activation=node_act.op_type,
            name=f"{self.__class__.__name__}--{node.name}",
        )
        fc.attribute.extend(node.attribute)
        return [fc]
