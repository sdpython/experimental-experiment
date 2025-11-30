import inspect
from typing import List, Optional
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization


class ReduceSumNormalizePattern(PatternOptimization):
    """
    Nodes equivalent to a reduction.

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
            oh.make_opsetid("", 26),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape=("a", "b")))
        inputs.append(
            oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT16, shape=("a", "b"))
        )
        inputs.append(oh.make_tensor_value_info("axis", onnx.TensorProto.INT64, shape=[]))
        nodes.append(
            make_node_extended(
                "Constant",
                [],
                ["axis"],
                value=onh.from_array(np.array(-1, dtype=np.int64), name="value"),
            )
        )
        nodes.append(make_node_extended("Cast", ["X"], ["xc"], to=1))
        nodes.append(make_node_extended("ReduceSum", ["xc", "axis"], ["red"], keepdims=1))
        nodes.append(make_node_extended("Mul", ["red", "Y"], ["mul"]))
        nodes.append(make_node_extended("Sub", ["xc", "mul"], ["subc"]))
        nodes.append(make_node_extended("Cast", ["subc"], ["Z"], to=10))
        outputs.append(
            oh.make_tensor_value_info("Z", onnx.TensorProto.FLOAT16, shape=("a", "b"))
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
            oh.make_opsetid("", 26),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape=("a", "b")))
        inputs.append(
            oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT16, shape=("a", "b"))
        )
        inputs.append(oh.make_tensor_value_info("axis", onnx.TensorProto.INT64, shape=[]))
        nodes.append(
            make_node_extended(
                "ReduceSum", ["X", "axis"], ["ReduceSumNormalizePattern_red"], keepdims=1
            )
        )
        nodes.append(make_node_extended("Cast", ["Y"], ["ReduceSumNormalizePattern_Y"], to=10))
        nodes.append(
            make_node_extended(
                "Mul",
                ["ReduceSumNormalizePattern_red", "ReduceSumNormalizePattern_Y"],
                ["ReduceSumNormalizePattern_mul"],
            )
        )
        nodes.append(make_node_extended("Sub", ["X", "ReduceSumNormalizePattern_mul"], ["Z"]))
        outputs.append(
            oh.make_tensor_value_info("Z", onnx.TensorProto.FLOAT16, shape=("a", "b"))
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

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "ReduceSum" or node.domain != "":
            return self.none()

        cast_node = g.node_before(node.input[0])
        if cast_node is None or cast_node.op_type != "Cast":
            return self.none(node, inspect.currentframe().f_lineno)

        mul_node = g.next_nodes(node.output[0])
        if len(mul_node) != 1 or mul_node[0].op_type != "Mul":
            return self.none(node, inspect.currentframe().f_lineno)

        sub_node = g.next_nodes(mul_node[0].output[0])
        if len(sub_node) != 1 or sub_node[0].op_type != "Sub":
            return self.none(node, inspect.currentframe().f_lineno)

        cast2_node = g.next_nodes(sub_node[0].output[0])
        if len(cast2_node) != 1 or cast2_node[0].op_type != "Cast":
            return self.none(node, inspect.currentframe().f_lineno)

        if not (set(sub_node[0].input) & set(node.input)):
            return self.none(node, inspect.currentframe().f_lineno)

        if g.get_type(cast_node.input[0]) != g.get_type(cast2_node[0].output[0]):
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(
            self, [cast_node, node, mul_node[0], sub_node[0], cast2_node[0]], self.apply
        )

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        cast_node: NodeProto,
        node: NodeProto,
        mul_node: NodeProto,
        sub_node: NodeProto,
        cast2_node: NodeProto,
    ) -> List[NodeProto]:
        new_name = g.unique_name(f"{self.__class__.__name__}_{node.output[0]}")
        new_red = g.make_node(
            node.op_type,
            [cast_node.input[0], node.input[1]],
            [new_name],
            name=f"{self.__class__.__name__}--{node.name}",
        )
        new_red.attribute.extend(node.attribute)
        other_name = [n for n in mul_node.input if n != node.output[0]]
        assert len(other_name) == 1, f"Unexpected name {other_name!r}"
        new_name2 = g.unique_name(f"{self.__class__.__name__}_{other_name[0]}")
        new_cast = g.make_node(
            "Cast",
            other_name,
            [new_name2],
            to=g.get_attribute(cast2_node, "to").i,
            name=f"{self.__class__.__name__}--{cast_node.name}",
        )

        new_m = g.unique_name(f"{self.__class__.__name__}_{mul_node.output[0]}")
        new_mul = g.make_node(
            mul_node.op_type,
            [new_name, new_name2],
            [new_m],
            name=f"{self.__class__.__name__}--{mul_node.name}",
        )

        if mul_node.output[0] == sub_node.input[0]:
            inputs = [new_m, new_red.input[0]]
        else:
            inputs = [new_red.input[0], new_m]
        new_sub = g.make_node(
            sub_node.op_type,
            inputs,
            cast2_node.output,
            name=f"{self.__class__.__name__}--{sub_node.name}",
        )

        return [new_red, new_cast, new_mul, new_sub]
