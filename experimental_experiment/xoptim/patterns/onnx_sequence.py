import inspect
from typing import List, Optional
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization


class SequenceConstructAtPattern(PatternOptimization):
    """
    Replaces the sequence ``SequenceConstruct(x1, x2, ...)`` followed
    by ``SequenceAt(seq, 0)``, ``SequenceAt(seq, 1)``, ...

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
        inputs.append(oh.make_tensor_value_info("X1", onnx.TensorProto.FLOAT, shape=("a", "b")))
        inputs.append(oh.make_tensor_value_info("X2", onnx.TensorProto.FLOAT, shape=("c", "d")))
        nodes.append(
            make_node_extended(
                "Constant",
                [],
                ["i0"],
                value=onh.from_array(np.array(0, dtype=np.int64), name="value"),
            )
        )
        nodes.append(
            make_node_extended(
                "Constant",
                [],
                ["i1"],
                value=onh.from_array(np.array(1, dtype=np.int64), name="value"),
            )
        )
        nodes.append(make_node_extended("SequenceConstruct", ["X1", "X2"], ["seq"]))
        nodes.append(make_node_extended("SequenceAt", ["seq", "i0"], ["Y1"]))
        nodes.append(make_node_extended("SequenceAt", ["seq", "i1"], ["Y2"]))
        outputs.append(
            oh.make_tensor_value_info("Y1", onnx.TensorProto.FLOAT, shape=("a", "b"))
        )
        outputs.append(
            oh.make_tensor_value_info("Y2", onnx.TensorProto.FLOAT, shape=("c", "d"))
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
        inputs.append(oh.make_tensor_value_info("X1", onnx.TensorProto.FLOAT, shape=("a", "b")))
        inputs.append(oh.make_tensor_value_info("X2", onnx.TensorProto.FLOAT, shape=("c", "d")))
        nodes.append(make_node_extended("Identity", ["X1"], ["Y1"]))
        nodes.append(make_node_extended("Identity", ["X2"], ["Y2"]))
        outputs.append(
            oh.make_tensor_value_info("Y1", onnx.TensorProto.FLOAT, shape=("a", "b"))
        )
        outputs.append(
            oh.make_tensor_value_info("Y2", onnx.TensorProto.FLOAT, shape=("c", "d"))
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
        if node.op_type != "SequenceConstruct" or node.domain != "":
            return self.none()

        next_nodes = g.next_nodes(node.output[0])
        if len(next_nodes) != len(node.input):
            return self.none(node, inspect.currentframe().f_lineno)
        if any(n.op_type != "SequenceAt" for n in next_nodes):
            return self.none(node, inspect.currentframe().f_lineno)

        ats = [n.input[1] for n in next_nodes]
        if any(not g.is_constant_scalar(a) for a in ats):
            return self.none(node, inspect.currentframe().f_lineno)

        cst = [g.get_constant_scalar(a) for a in ats]
        if set(cst) != set(range(len(ats))):
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [node, *next_nodes], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node_seq: NodeProto,
        *node_ats: NodeProto,
    ) -> List[NodeProto]:
        assert len(node_seq.input) == len(
            node_ats
        ), f"Matching failed because len({node_seq.input}) != {len(node_ats)}"

        new_nodes = []
        for n in node_ats:
            i = g.get_constant_scalar(n.input[1])
            new_nodes.append(
                g.make_node(
                    "Identity",
                    [node_seq.input[i]],
                    n.output,
                    name=f"{self.__class__.__name__}--{node_seq.name}",
                )
            )
        return new_nodes
