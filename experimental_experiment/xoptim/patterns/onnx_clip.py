import inspect
from typing import List, Optional
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization


class ClipClipPattern(PatternOptimization):
    """
    Merges consecutive clips if one is defining min and the other max.

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
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(oh.make_tensor_value_info("zero", onnx.TensorProto.FLOAT, shape=(1,)))
        inputs.append(oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape=("a", "b")))
        inputs.append(oh.make_tensor_value_info("one", onnx.TensorProto.FLOAT, shape=(1,)))
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["zero"],
                value=onh.from_array(np.array([0.0], dtype=np.float32), name="value"),
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
        nodes.append(oh.make_node("Clip", ["X", "zero"], ["x1"]))
        nodes.append(oh.make_node("Clip", ["x1", "", "one"], ["Y"]))
        outputs.append(oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape=("c", "d")))
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
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(oh.make_tensor_value_info("zero", onnx.TensorProto.FLOAT, shape=(1,)))
        inputs.append(oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape=("a", "b")))
        inputs.append(oh.make_tensor_value_info("one", onnx.TensorProto.FLOAT, shape=(1,)))
        nodes.append(oh.make_node("Clip", ["X", "zero", "one"], ["Y"]))
        outputs.append(oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape=("c", "d")))
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
        if node.op_type != "Clip" or node.domain != "":
            return self.none()
        before = g.node_before(node.input[0])
        if (
            before is None
            or g.is_used_more_than_once(node.input[0])
            or before.op_type != "Clip"
            or before.domain != ""
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        min1 = before.input[1] if len(before.input) > 1 else ""
        min2 = node.input[1] if len(node.input) > 1 else ""
        if (min1 and min2) or (not min1 and not min2):
            return self.none(node, inspect.currentframe().f_lineno)
        max1 = before.input[2] if len(before.input) > 2 else ""
        max2 = node.input[2] if len(node.input) > 2 else ""
        if (max1 and max2) or (not max1 and not max2):
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [before, node], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        before: NodeProto,
        node: NodeProto,
    ) -> List[NodeProto]:
        # merges clips
        min1 = before.input[1] if len(before.input) > 1 else ""
        min2 = node.input[1] if len(node.input) > 1 else ""
        max1 = before.input[2] if len(before.input) > 2 else ""
        max2 = node.input[2] if len(node.input) > 2 else ""

        return [
            g.make_node(
                "Clip",
                [before.input[0], min1 or min2, max1 or max2],
                node.output,
                name=f"{self.__class__.__name__}--{node.name}",
            )
        ]
