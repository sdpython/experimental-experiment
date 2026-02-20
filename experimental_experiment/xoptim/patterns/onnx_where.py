import inspect
from typing import List, Optional
import numpy as np
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization


class NotWherePattern(PatternOptimization):
    """
    Replaces the sequence Where(Not(cond), X, Y) -> Where(cond, Y, X).

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
        inputs.append(oh.make_tensor_value_info("A", onnx.TensorProto.INT64, shape=("a", "b")))
        inputs.append(oh.make_tensor_value_info("X", onnx.TensorProto.BOOL, shape=("a", "b")))
        inputs.append(oh.make_tensor_value_info("B", onnx.TensorProto.INT64, shape=("a", "b")))
        nodes.append(oh.make_node("Not", ["X"], ["nx"]))
        nodes.append(oh.make_node("Where", ["nx", "A", "B"], ["Y"]))
        outputs.append(oh.make_tensor_value_info("Y", onnx.TensorProto.INT64, shape=("a", "b")))
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
        inputs.append(oh.make_tensor_value_info("A", onnx.TensorProto.INT64, shape=("a", "b")))
        inputs.append(oh.make_tensor_value_info("X", onnx.TensorProto.BOOL, shape=("a", "b")))
        inputs.append(oh.make_tensor_value_info("B", onnx.TensorProto.INT64, shape=("a", "b")))
        nodes.append(oh.make_node("Where", ["X", "B", "A"], ["Y"]))
        outputs.append(oh.make_tensor_value_info("Y", onnx.TensorProto.INT64, shape=("a", "b")))
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
        if node.op_type != "Not" or node.domain != "":
            return self.none()
        wheres = g.next_nodes(node.output[0])
        if len(wheres) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        where = wheres[0]
        if where.op_type != "Where" or where.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, [node, where], self.apply, insert_at=where)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        not_node: NodeProto,
        where_node: NodeProto,
    ) -> List[NodeProto]:
        return [
            g.make_node(
                "Where",
                [not_node.input[0], where_node.input[2], where_node.input[1]],
                [where_node.output[0]],
                name=f"{self.__class__.__name__}--{where_node.name}",
                doc_string=where_node.doc_string,
            )
        ]


class WhereAddPattern(PatternOptimization):
    """
    Replaces the sequence Add(X, Where(bool_mask, 0, -inf)) -> Where(bool_mask, X, -inf).

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
        inputs.append(oh.make_tensor_value_info("inf", onnx.TensorProto.FLOAT, shape=(1,)))
        inputs.append(oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape=("a", "b")))
        inputs.append(
            oh.make_tensor_value_info("mask", onnx.TensorProto.BOOL, shape=("a", "b"))
        )
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
                ["inf"],
                value=onh.from_array(np.array([-np.inf], dtype=np.float32), name="value"),
            )
        )
        nodes.append(oh.make_node("Where", ["mask", "zero", "inf"], ["fmask"]))
        nodes.append(oh.make_node("Add", ["fmask", "X"], ["Y"]))
        outputs.append(oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape=("a", "b")))
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
        inputs.append(oh.make_tensor_value_info("inf", onnx.TensorProto.FLOAT, shape=(1,)))
        inputs.append(oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape=("a", "b")))
        inputs.append(
            oh.make_tensor_value_info("mask", onnx.TensorProto.BOOL, shape=("a", "b"))
        )
        nodes.append(oh.make_node("Where", ["mask", "X", "inf"], ["Y"]))
        outputs.append(oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape=("a", "b")))
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
        if node.op_type != "Where" or node.domain != "":
            return self.none()
        if not g.is_constant_scalar(node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant_scalar(node.input[2]):
            return self.none(node, inspect.currentframe().f_lineno)
        cst1 = g.get_constant_scalar(node.input[1])
        cst2 = g.get_constant_scalar(node.input[2])
        if cst1 is None or cst2 is None:
            return self.none(node, inspect.currentframe().f_lineno)
        if cst1 != 0:
            return self.none(node, inspect.currentframe().f_lineno)
        if not np.isinf(cst2):
            return self.none(node, inspect.currentframe().f_lineno)

        add_nodes = g.next_nodes(node.output[0])
        if len(add_nodes) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        if add_nodes[0].op_type != "Add" or add_nodes[0].domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, [node, add_nodes[0]], self.apply, insert_at=add_nodes[0])

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        where_node: NodeProto,
        add_node: NodeProto,
    ) -> List[NodeProto]:
        where_input1 = add_node.input[1 if add_node.input[0] == where_node.output[0] else 0]
        return [
            g.make_node(
                "Where",
                [where_node.input[0], where_input1, where_node.input[2]],
                [add_node.output[0]],
                name=f"{self.__class__.__name__}--{where_node.name}",
                doc_string=where_node.doc_string,
            )
        ]
