import inspect
from typing import List, Optional
import numpy as np
import onnx.numpy_helper as onh
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization


class TriMatrixPattern(PatternOptimization):
    """
    Replaces a sequence of nodes creating a triangular matrix
    with operator TriMatrix(...).

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
            oh.make_opsetid("onnx_extended.ortops.optim.cuda", 1),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(oh.make_tensor_value_info("shape", onnx.TensorProto.INT64, shape=(2,)))
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["zero"],
                value=onh.from_array(np.array(0, dtype=np.int64), name="value"),
            )
        )
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["dim"],
                value=onh.from_array(np.array(1024, dtype=np.int64), name="value"),
            )
        )
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["onei"],
                value=onh.from_array(np.array(1, dtype=np.int64), name="value"),
            )
        )
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["one"],
                value=onh.from_array(np.array([1], dtype=np.int64), name="value"),
            )
        )
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["shape1"],
                value=onh.from_array(np.array([1024, 1], dtype=np.int64), name="value"),
            )
        )
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["zerof"],
                value=onh.from_array(np.array([0.0], dtype=np.float32), name="value"),
            )
        )
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["shape"],
                value=onh.from_array(np.array([1024, 1024], dtype=np.int64), name="value"),
            )
        )
        nodes.append(oh.make_node("Range", ["zero", "dim", "onei"], ["ar"]))
        nodes.append(oh.make_node("Add", ["ar", "one"], ["ad"]))
        nodes.append(oh.make_node("Reshape", ["ad", "shape1"], ["re"]))
        nodes.append(oh.make_node("Less", ["ar", "re"], ["le"]))
        nodes.append(oh.make_node("Where", ["le", "zerof", "cst"], ["Y"]))
        nodes.append(
            oh.make_node(
                "ConstantOfShape",
                ["shape"],
                ["cst"],
                value=onh.from_array(
                    np.array([-3.4028234663852886e38], dtype=np.float32), name="value"
                ),
            )
        )
        outputs.append(
            oh.make_tensor_value_info(
                "Y", onnx.TensorProto.FLOAT, shape=("UNKNOWNDIM", "UNKNOWNDIM1")
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
        inputs.append(oh.make_tensor_value_info("shape", onnx.TensorProto.INT64, shape=(2,)))
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["TriMatrixPattern--"],
                value=onh.from_array(
                    np.array([0.0, 0.0, -3.4028234663852886e38], dtype=np.float32), name="value"
                ),
            )
        )
        nodes.append(
            oh.make_node(
                "TriMatrix",
                ["shape", "TriMatrixPattern--"],
                ["Y"],
                domain="onnx_extended.ortops.optim.cuda",
            )
        )
        outputs.append(
            oh.make_tensor_value_info(
                "Y", onnx.TensorProto.FLOAT, shape=("UNKNOWNDIM", "UNKNOWNDIM1")
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
        if node.op_type != "Range" or node.domain != "":
            return self.none()

        if (
            len(node.input) != 3
            or not g.is_constant_scalar(node.input[0])
            or not g.is_constant_scalar(node.input[1])
            or not g.is_constant_scalar(node.input[2])
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        start, limit, delta = [g.get_constant_scalar(i) for i in node.input]
        if start != 0 or delta != 1:
            return self.none(node, inspect.currentframe().f_lineno)

        next_nodes = g.next_nodes(node.output[0])
        if len(next_nodes) != 2:
            return self.none(node, inspect.currentframe().f_lineno)

        types = {n.op_type for n in next_nodes}
        if types != {"Add", "Less"}:
            return self.none(node, inspect.currentframe().f_lineno)

        if next_nodes[0].op_type == "Add":
            add_node, less_node = next_nodes
        else:
            less_node, add_node = next_nodes

        if (
            not g.is_constant_scalar(add_node.input[1])
            or g.get_constant_scalar(add_node.input[1]) != 1
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        resh_node = g.next_nodes(add_node.output[0])
        if len(resh_node) != 1 or resh_node[0].op_type != "Reshape":
            return self.none(node, inspect.currentframe().f_lineno)

        reshape_node = resh_node[0]

        shape = g.get_computed_constant(reshape_node.input[1])
        if shape.tolist() != [limit, 1]:
            return self.none(node, inspect.currentframe().f_lineno)

        if less_node.input != [node.output[0], reshape_node.output[0]]:
            return self.none(node, inspect.currentframe().f_lineno)

        where_node = g.next_nodes(less_node.output[0])
        if len(where_node) != 1 or where_node[0].op_type != "Where":
            return self.none(node, inspect.currentframe().f_lineno)

        where_node = where_node[0]
        if not g.is_constant_scalar(where_node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)

        cst_node = g.node_before(where_node.input[2])
        if cst_node.op_type != "ConstantOfShape":
            return self.none(node, inspect.currentframe().f_lineno)

        shape = g.get_computed_constant(cst_node.input[0])
        if shape.tolist() != [limit, limit]:
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(
            self,
            [node, add_node, reshape_node, less_node, where_node, cst_node],
            self.apply,
            insert_at=where_node,
        )

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        range_node: NodeProto,
        add_node: NodeProto,
        reshape_node: NodeProto,
        less_node: NodeProto,
        where_node: NodeProto,
        cst_node: NodeProto,
    ) -> List[NodeProto]:
        cst_upper = onh.to_array(g.get_attribute(cst_node, "value").t)
        dtype = cst_upper.dtype
        cst_lower = np.array([g.get_constant_scalar(where_node.input[1])], dtype=dtype)
        cst_diag = cst_lower
        csts_array = np.hstack([cst_lower, cst_diag, cst_upper]).astype(dtype)
        assert csts_array.shape == (3,), f"Wrong constant array: {csts_array}"

        cst_name = g.make_initializer(
            f"{self.__class__.__name__}--{where_node.name}",
            csts_array,
            source="TriMatrixPattern.apply.cst",
        )
        new_node = g.make_node(
            "TriMatrix",
            [cst_node.input[0], cst_name],
            where_node.output,
            name=f"{self.__class__.__name__}--{where_node.name}",
            domain="onnx_extended.ortops.optim.cuda",
        )
        return [new_node]
