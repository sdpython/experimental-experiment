import inspect
from typing import List, Optional
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization


class ShapeBasedShapeShapeAddPattern(PatternOptimization):
    """
    Tries to find another way to get a dimension obtained with the addition of two.

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
            oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape=(2, 3, "d", 1))
        )
        nodes.append(make_node_extended("Expand", ["X", "d"], ["Y"]))
        outputs.append(
            oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape=(2, 3, "d", 2))
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
            oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape=(2, 3, "d", 1))
        )
        nodes.append(
            make_node_extended(
                "Constant",
                [],
                ["init7_s4_1_1_1_2"],
                value=onh.from_array(np.array([1, 1, 1, 2], dtype=np.int64), name="value"),
            )
        )
        nodes.append(make_node_extended("Expand", ["X", "init7_s4_1_1_1_2"], ["Y"]))
        outputs.append(
            oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape=(2, 3, "d", 2))
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
        if node.op_type != "Add" or node.domain != "":
            return self.none()
        shape1 = g.node_before(node.input[0])
        if shape1 is None or shape1.op_type != "Shape" or shape1.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        shape2 = g.node_before(node.input[1])
        if shape2 is None or shape2.op_type != "Shape" or shape2.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        # ishape1 = g.get_shape_renamed(shape1.input[0])
        # ishape2 = g.get_shape_renamed(shape2.input[0])
        # value1 = g.builder.value_as_shape(node.input[0])
        # value2 = g.builder.value_as_shape(node.input[1])
        # input_shapes = [g.get_shape_renamed(i) for i in g.builder.input_names]
        # g.builder._known_value_shape
        # g.builder.constraints_)
        # g.builder.replacements_dimensions_
        return self.none(node, inspect.currentframe().f_lineno)
        # return MatchResult(self, [shape1, shape2, node], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        shape1_node: NodeProto,
        shape2_node: NodeProto,
        add_node: NodeProto,
    ) -> List[NodeProto]:
        raise NotImplementedError(f"{self.___class__.__name__} is not implemented yet.")
