import inspect
from typing import List, Optional
import numpy as np
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization


class ReduceSumNormalizePattern(PatternOptimization):
    """
    Nodes equivalent to a reduction.

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
            oh.make_node(
                "Constant",
                [],
                ["axis"],
                value=onh.from_array(np.array(-1, dtype=np.int64), name="value"),
            )
        )
        nodes.append(oh.make_node("Cast", ["X"], ["xc"], to=1))
        nodes.append(oh.make_node("ReduceSum", ["xc", "axis"], ["red"], keepdims=1))
        nodes.append(oh.make_node("Mul", ["red", "Y"], ["mul"]))
        nodes.append(oh.make_node("Sub", ["xc", "mul"], ["subc"]))
        nodes.append(oh.make_node("Cast", ["subc"], ["Z"], to=10))
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
            oh.make_node(
                "ReduceSum", ["X", "axis"], ["ReduceSumNormalizePattern_red"], keepdims=1
            )
        )
        nodes.append(oh.make_node("Cast", ["Y"], ["ReduceSumNormalizePattern_Y"], to=10))
        nodes.append(
            oh.make_node(
                "Mul",
                ["ReduceSumNormalizePattern_red", "ReduceSumNormalizePattern_Y"],
                ["ReduceSumNormalizePattern_mul"],
            )
        )
        nodes.append(oh.make_node("Sub", ["X", "ReduceSumNormalizePattern_mul"], ["Z"]))
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

        print("DOT-SECTION", to_dot(model))
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


class ReduceArgTopKPattern(PatternOptimization):
    """
    Fuses ReduceMin(X, axis), ArgMin(X, axis) into TopK(, k=1).

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
            oh.make_opsetid("", 22),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape=("a", "b")))
        inputs.append(oh.make_tensor_value_info("one", onnx.TensorProto.INT64, shape=(1,)))
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["one"],
                value=onh.from_array(np.array([1], dtype=np.int64), name="value"),
            )
        )
        nodes.append(oh.make_node("ReduceMin", ["X", "one"], ["Y1"], keepdims=0))
        nodes.append(oh.make_node("ArgMin", ["X"], ["Y2"], axis=1, keepdims=0))
        outputs.append(oh.make_tensor_value_info("Y2", onnx.TensorProto.INT64, shape=("a",)))
        outputs.append(oh.make_tensor_value_info("Y1", onnx.TensorProto.FLOAT, shape=("a",)))
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
            oh.make_opsetid("", 22),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape=("a", "b")))
        inputs.append(oh.make_tensor_value_info("one", onnx.TensorProto.INT64, shape=(1,)))
        nodes.append(
            oh.make_node(
                "TopK",
                ["X", "one"],
                ["ReduceArgTopKPattern_Y1", "ReduceArgTopKPattern_Y2"],
                axis=1,
                largest=0,
            )
        )
        nodes.append(oh.make_node("Squeeze", ["ReduceArgTopKPattern_Y1", "one"], ["Y1"]))
        nodes.append(oh.make_node("Squeeze", ["ReduceArgTopKPattern_Y2", "one"], ["Y2"]))
        outputs.append(oh.make_tensor_value_info("Y2", onnx.TensorProto.INT64, shape=("a",)))
        outputs.append(oh.make_tensor_value_info("Y1", onnx.TensorProto.FLOAT, shape=("a",)))
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
        if g.main_opset < 18:
            return self.none()
        if node.op_type not in ("ArgMin", "ArgMax") or node.domain != "":
            return self.none()

        next_nodes = g.next_nodes(node.input[0])
        if len(next_nodes) < 2:
            return self.none(node, inspect.currentframe().f_lineno)
        look_for = f"Reduce{node.op_type[3:]}"
        reduce = [n for n in next_nodes if n.op_type == look_for]
        if len(reduce) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        reduce_node = reduce[0]

        if not g.is_constant(reduce_node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        cst = g.get_computed_constant(reduce_node.input[1])
        if not cst:
            return self.none(node, inspect.currentframe().f_lineno)
        axes = tuple(cst)
        if len(axes) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        axis = g.get_attribute_with_default(node, "axis", 0)
        if axis != cst[0]:
            return self.none(node, inspect.currentframe().f_lineno)

        if g.get_attribute_with_default(node, "keepdims", 1) != g.get_attribute_with_default(
            reduce_node, "keepdims", 1
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        if g.get_attribute_with_default(reduce_node, "noop_with_empty_axes", 0):
            return self.none(node, inspect.currentframe().f_lineno)
        if g.get_attribute_with_default(reduce_node, "select_last_index", 0) == 1:
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [reduce_node, node], self.apply)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        reduce_node: NodeProto,
        arg_node: NodeProto,
    ) -> List[NodeProto]:
        one = g.make_initializer(
            "",
            np.array([1], dtype=np.int64),
            source=f"{self.__class__.__name__}.K",
        )
        keepdims = g.get_attribute_with_default(arg_node, "keepdims", 1)
        axis = g.get_attribute_with_default(arg_node, "axis", 0)

        topk_names = (
            [reduce_node.output[0], arg_node.output[0]]
            if keepdims
            else [
                g.unique_name(f"{self.__class__.__name__}_{reduce_node.output[0]}"),
                g.unique_name(f"{self.__class__.__name__}_{arg_node.output[0]}"),
            ]
        )
        nodes = [
            g.make_node(
                "TopK",
                [reduce_node.input[0], one],
                topk_names,
                axis=axis,
                largest=1 if "Max" in arg_node.op_type else 0,
                name=f"{self.__class__.__name__}--{arg_node.name}",
            )
        ]
        if not keepdims:
            axis = g.make_initializer(
                "",
                np.array([axis], dtype=np.int64),
                source=f"{self.__class__.__name__}.K",
            )
            nodes.extend(
                [
                    g.make_node(
                        "Squeeze",
                        [topk_names[0], axis],
                        [reduce_node.output[0]],
                        name=f"{self.__class__.__name__}--{reduce_node.name}",
                    ),
                    g.make_node(
                        "Squeeze",
                        [topk_names[1], axis],
                        [arg_node.output[0]],
                        name=f"{self.__class__.__name__}--{arg_node.name}",
                    ),
                ]
            )
        return nodes
