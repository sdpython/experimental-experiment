import inspect
from typing import List, Optional
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization


class DropoutPattern(PatternOptimization):
    """
    Checks that a Cast is really needed.

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
        inputs.append(
            oh.make_tensor_value_info(
                "_onx_add02", onnx.TensorProto.FLOAT16, shape=(4, 512, 128)
            )
        )
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["init10_s_3"],
                value=onh.from_array(np.array(0.0, dtype=np.float16), name="value"),
            )
        )
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["init9_s_"],
                value=onh.from_array(np.array(False, dtype=np.bool_), name="value"),
            )
        )
        nodes.append(
            oh.make_node(
                "Dropout", ["_onx_add02", "init10_s_3", "init9_s_"], ["dropout", ""]
            )
        )
        outputs.append(
            oh.make_tensor_value_info("dropout", onnx.TensorProto.FLOAT16, shape=(4, 512, 128))
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
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info(
                "_onx_add02", onnx.TensorProto.FLOAT16, shape=(4, 512, 128)
            )
        )
        nodes.append(oh.make_node("Identity", ["_onx_add02"], ["dropout"]))
        outputs.append(
            oh.make_tensor_value_info("dropout", onnx.TensorProto.FLOAT16, shape=(4, 512, 128))
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
        if node.op_type != "Dropout" or node.domain != "":
            return None

        for o in node.output[1:]:
            if o and g.is_used(o):
                return self.none(node, inspect.currentframe().f_lineno)

        if not (
            len(node.input) >= 3
            and node.input[2] != ""
            and g.is_constant_scalar(node.input[2])
            and not g.get_constant_scalar(node.input[2])
        ):
            return MatchResult(self, [node], self.apply, insert_at=node)

        if (
            len(node.input) >= 2
            and node.input[1] != ""
            and g.is_constant_scalar(node.input[2])
            and g.get_constant_scalar(node.input[2]) != 0
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [node], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        dropout_node: NodeProto,
    ) -> List[NodeProto]:
        return [
            g.make_node(
                "Identity",
                dropout_node.input[:1],
                dropout_node.output[:1],
                name=f"{self.__class__.__name__}--{dropout_node.name}",
                doc_string=dropout_node.doc_string,
            )
        ]
