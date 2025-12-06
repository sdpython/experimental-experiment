import inspect
from typing import List, Optional
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization


class Sub1MulPattern(PatternOptimization):
    """
    Replaces the sequence `(1 - X) x Y`  by `Y - X x Y` to avoid the creation
    of a constant in the graph. `x` means element wise multiplication.

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
        inputs.append(oh.make_tensor_value_info("input3", onnx.TensorProto.FLOAT, shape=(1,)))
        nodes.append(
            make_node_extended(
                "Constant",
                [],
                ["init1_s1_"],
                value=onh.from_array(np.array([1.0], dtype=np.float32), name="value"),
            )
        )
        nodes.append(make_node_extended("Mul", ["input3", "_onx_sub0"], ["_onx_mul0"]))
        nodes.append(make_node_extended("Sub", ["init1_s1_", "input3"], ["_onx_sub0"]))
        outputs.append(
            oh.make_tensor_value_info("_onx_mul0", onnx.TensorProto.FLOAT, shape=(1,))
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
            oh.make_opsetid("", 18),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(oh.make_tensor_value_info("input3", onnx.TensorProto.FLOAT, shape=(1,)))
        nodes.append(
            make_node_extended("Mul", ["input3", "input3"], ["Sub1MulPattern--_onx_mul0"])
        )
        nodes.append(
            make_node_extended("Sub", ["input3", "Sub1MulPattern--_onx_mul0"], ["_onx_mul0"])
        )
        outputs.append(
            oh.make_tensor_value_info("_onx_mul0", onnx.TensorProto.FLOAT, shape=(1,))
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
        if node.op_type != "Mul" or node.domain != "":
            return self.none()

        node_left = g.node_before(node.input[0])
        node_right = g.node_before(node.input[1])
        op_left = None if node_left is None else node_left.op_type
        op_right = None if node_right is None else node_right.op_type

        if op_left != "Sub" and op_right != "Sub":
            return self.none(node, inspect.currentframe().f_lineno)

        if (op_left == "Sub" and g.is_used_more_than_once(node.input[0])) or (
            op_right == "Sub" and g.is_used_more_than_once(node.input[1])
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        cst_left, cst_right = None, None

        if op_left == "Sub" and g.is_constant(node_left.input[0]):
            cst_min, cst_max = g.get_computed_constant(node_left.input[0], ["min", "max"])
            if cst_min is None or cst_max is None:
                return self.none(node, inspect.currentframe().f_lineno)

            if cst_min == cst_max == 1:
                cst_left = cst_min

        if op_right == "Sub" and g.is_constant(node_right.input[0]):
            cst_min, cst_max = g.get_computed_constant(node_right.input[0], ["min", "max"])
            if cst_min is None or cst_max is None:
                return self.none(node, inspect.currentframe().f_lineno)
            if cst_min == cst_max == 1:
                cst_right = cst_min

        if cst_left is None and cst_right is None:
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [node, node_left, node_right], self.apply)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node: NodeProto,
        node_left: NodeProto,
        node_right: NodeProto,
    ) -> List[NodeProto]:
        cst_left = None
        if (
            node_left is not None
            and node_left.op_type == "Sub"
            and g.is_constant(node_left.input[0])
        ):
            cst = g.get_computed_constant(node_left.input[0])
            if cst.min() == cst.max() == 1:
                cst_left = cst

        if cst_left is not None:
            # rewrite `(1 - X) x Y` into `Y - X x Y`
            mul_node = g.make_node(
                "Mul",
                [node_left.input[1], node.input[1]],
                [g.unique_name(f"{self.__class__.__name__}--{node.output[0]}")],
                name=f"{self.__class__.__name__}--{node.name}",
            )
            sub_node = g.make_node(
                "Sub",
                [node.input[1], mul_node.output[0]],
                node.output,
                name=f"{self.__class__.__name__}--{node.name}",
                doc_string=node.doc_string,
            )
            keep_node = node_right
        else:
            # rewrite `Y x (1 - X)` into `Y - (Y - X)`
            mul_node = g.make_node(
                "Mul",
                [node.input[0], node_right.input[1]],
                [g.unique_name(f"{self.__class__.__name__}--{node.output[0]}")],
                name=f"{self.__class__.__name__}--{node.name}",
            )
            sub_node = g.make_node(
                "Sub",
                [node.input[0], mul_node.output[0]],
                node.output,
                name=f"{self.__class__.__name__}--{node.name}",
                doc_string=node.doc_string,
            )
            keep_node = node_left

        if keep_node is None:
            return [mul_node, sub_node]
        return [keep_node, mul_node, sub_node]
