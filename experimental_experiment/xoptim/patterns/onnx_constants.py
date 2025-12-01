from typing import List, Optional
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization


class ConstantToInitializerPattern(PatternOptimization):
    """
    Replaces a node Constant by an initializer and a node Identity.

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
        nodes.append(
            make_node_extended(
                "Constant",
                [],
                ["cst"],
                value=onh.from_array(np.array([1.0, 2.0], dtype=np.float32), name="value"),
            )
        )
        outputs.append(oh.make_tensor_value_info("cst", onnx.TensorProto.FLOAT, shape=(2,)))
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
        nodes.append(
            make_node_extended(
                "Constant",
                [],
                ["cst_cst2init"],
                value=onh.from_array(np.array([1.0, 2.0], dtype=np.float32), name="value"),
            )
        )
        nodes.append(make_node_extended("Identity", ["cst_cst2init"], ["cst"]))
        outputs.append(oh.make_tensor_value_info("cst", onnx.TensorProto.FLOAT, shape=(2,)))
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
        if node.op_type != "Constant" or node.domain != "":
            return self.none()
        return MatchResult(self, [node], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node: NodeProto,
    ) -> List[NodeProto]:
        cst = g.get_computed_constant(node.output[0])
        assert (
            cst is not None
        ), f"Node {g.pretty_node(cst)} is a constant, it must be possible to evaluate it."
        init = g.make_initializer(f"{node.output[0]}_cst2init", cst)
        return [
            g.make_node(
                "Identity", [init], node.output, name=f"{self.__class__.__name__}--{node.name}"
            )
        ]
