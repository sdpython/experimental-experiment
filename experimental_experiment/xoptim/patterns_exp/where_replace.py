import inspect
from typing import List, Optional
from onnx import NodeProto, TensorProto
from ..patterns_api import MatchResult, PatternOptimization


class ReplaceZeroPattern(PatternOptimization):
    """
    Replaces Where(bool(X), value, X) into ReplaceZero(X, by=by).

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
            oh.make_opsetid("onnx_extended.ortops.optim.cuda", 1),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info(
                "X", onnx.TensorProto.FLOAT, shape=("UNKNOWNDIM", "UNKNOWNDIM1")
            )
        )
        nodes.append(
            make_node_extended(
                "Constant",
                [],
                ["cst"],
                value=onh.from_array(
                    np.array([5.670000076293945], dtype=np.float32), name="value"
                ),
            )
        )
        nodes.append(make_node_extended("Cast", ["X"], ["xb"], to=9))
        nodes.append(make_node_extended("Where", ["xb", "cst", "X"], ["Y"]))
        outputs.append(
            oh.make_tensor_value_info(
                "Y", onnx.TensorProto.FLOAT, shape=("UNKNOWNDIM2", "UNKNOWNDIM3")
            )
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
            oh.make_opsetid("onnx_extended.ortops.optim.cuda", 1),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info(
                "X", onnx.TensorProto.FLOAT, shape=("UNKNOWNDIM", "UNKNOWNDIM1")
            )
        )
        nodes.append(
            make_node_extended(
                "ReplaceZero",
                ["X"],
                ["Y"],
                domain="onnx_extended.ortops.optim.cuda",
                by=5.670000076293945,
                equal=0,
            )
        )
        outputs.append(
            oh.make_tensor_value_info(
                "Y", onnx.TensorProto.FLOAT, shape=("UNKNOWNDIM2", "UNKNOWNDIM3")
            )
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
        if not g.has_processor("CUDA"):
            return self.none()
        if node.op_type != "Where" or node.domain != "":
            return self.none()

        if not g.is_constant(node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant_scalar(node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)

        if g.is_used_more_than_once(node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        cast_node = g.node_before(node.input[0])
        if cast_node is None or cast_node.op_type != "Cast" or node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        to = g.get_attribute(cast_node, "to").i
        if to != TensorProto.BOOL:
            return self.none(node, inspect.currentframe().f_lineno)

        if node.input[2] != cast_node.input[0]:
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [cast_node, node], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        cast_node: NodeProto,
        where_node: NodeProto,
    ) -> List[NodeProto]:
        cst = g.get_constant_scalar(where_node.input[1])
        new_node = g.make_node(
            "ReplaceZero",
            cast_node.input,
            where_node.output,
            by=cst,
            equal=False,
            name=f"{self.__class__.__name__}--{where_node.name}",
            domain="onnx_extended.ortops.optim.cuda",
        )
        return [new_node]
