import inspect
from typing import List, Optional
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization


class ConvBiasNullPattern(PatternOptimization):
    """
    Checks that a Conv has a null bias.

    Model with nodes to be fused:

    .. gdot::
        :script: DOT-SECTION
        :process:

        from onnx_array_api.plotting.dot_plot import to_dot
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
        inputs.append(
            oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape=(512, 3, 64, 64))
        )
        inputs.append(
            oh.make_tensor_value_info("W", onnx.TensorProto.FLOAT, shape=(64, 3, 4, 4))
        )
        nodes.append(
            make_node_extended(
                "Conv",
                ["X", "W", "B2"],
                ["Y"],
                dilations=[1, 1],
                group=1,
                kernel_shape=[4, 4],
                pads=[1, 1, 1, 1],
                strides=[2, 2],
            )
        )
        outputs.append(
            oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape=(512, 64, 32, 32))
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

        print(to_dot(model))

    Outcome of the fusion:

    .. gdot::
        :script: DOT-SECTION
        :process:

        from onnx_array_api.plotting.dot_plot import to_dot
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
        inputs.append(
            oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape=(512, 3, 64, 64))
        )
        inputs.append(
            oh.make_tensor_value_info("W", onnx.TensorProto.FLOAT, shape=(64, 3, 4, 4))
        )
        nodes.append(
            make_node_extended(
                "Conv",
                ["X", "W"],
                ["Y"],
                dilations=[1, 1],
                group=1,
                kernel_shape=[4, 4],
                pads=[1, 1, 1, 1],
                strides=[2, 2],
            )
        )
        outputs.append(
            oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape=(512, 64, 32, 32))
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

        print(to_dot(model))
    """

    def __init__(self, verbose: int = 0, priority: int = 0):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Conv" or node.domain != "":
            return self.none()
        if len(node.input) < 3:
            return self.none(node, inspect.currentframe().f_lineno)

        if not g.is_constant(node.input[2]):
            return self.none(node, inspect.currentframe().f_lineno)

        cst = g.get_computed_constant(node.input[2])
        if cst is None or cst.min() != 0 or cst.max() != 0:
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, [node], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node: NodeProto,
    ) -> List[NodeProto]:
        new_node = g.make_node(
            "Conv",
            node.input[:2],
            node.output,
            name=f"{self.__class__.__name__}--{node.name}",
            doc_string=node.doc_string,
        )
        new_node.attribute.extend(node.attribute)
        return [new_node]
