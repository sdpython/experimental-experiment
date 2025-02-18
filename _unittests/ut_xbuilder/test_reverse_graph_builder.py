import textwrap
import unittest
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
import numpy as np
from experimental_experiment.ext_test_case import ExtTestCase, requires_onnx_array_api
from experimental_experiment.xbuilder.reverse_graph_builder import to_graph_builder_code

TFLOAT = onnx.TensorProto.FLOAT
TINT64 = onnx.TensorProto.INT64


class TestReverseGraphBuilder(ExtTestCase):
    @requires_onnx_array_api("0.3.1")
    def test_constant_of_shape(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "ConstantOfShape",
                        ["shape"],
                        ["cst"],
                        value=onh.from_array(np.array([0], dtype=np.float32)),
                    ),
                    oh.make_node(
                        "ScatterND",
                        ["cst", "indices", "updates"],
                        ["Z"],
                        reduction="add",
                    ),
                ],
                "create_graph",
                [
                    oh.make_tensor_value_info("shape", TINT64, [None]),
                    oh.make_tensor_value_info("indices", TINT64, [None, None]),
                    oh.make_tensor_value_info("updates", TFLOAT, [None, None, None]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, [None, None, None])],
            ),
            opset_imports=[
                oh.make_opsetid("", 18),
            ],
            ir_version=9,
        )
        code = to_graph_builder_code(model)

        expected = (
            textwrap.dedent(
                """
        import numpy as np
        from onnx import TensorProto
        from onnx.numpy_helper import from_array
        from experimental_experiment.xbuilder import GraphBuilder



        def create_graph(
            op: "GraphBuilder",
            shape: "INT64[]",
            indices: "INT64[, ]",
            updates: "FLOAT[, , ]",
        ):
            cst = __LONG__
            Z = op.ScatterND(cst, indices, updates, reduction='add')
            op.Identity(Z, outputs=["Z"])
            return Z


        def make_my_model() -> "ModelProto":
            g = GraphBuilder({'': 18}, ir_version=9)
            g.make_tensor_input("shape", TensorProto.INT64, ('',))
            g.make_tensor_input("indices", TensorProto.INT64, ('', ''))
            g.make_tensor_input("updates", TensorProto.FLOAT, ('', '', ''))
            create_graph(g.op, "shape", "indices", "updates")
            g.make_tensor_output("Z", TensorProto.FLOAT, ('', '', ''))
            model = g.to_onnx()
            return model


        model = make_my_model()
        """
            )
            .strip("\n")
            .replace(
                "__LONG__",
                "op.ConstantOfShape(shape, value=from_array"
                "(np.array([0.0], dtype=np.float32), name='value'))",
            )
        )
        self.maxDiff = None
        self.assertEqual(expected, code.strip("\n"))

    @requires_onnx_array_api("0.3.1")
    def test_squeeze(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "ConstantOfShape",
                        ["shape"],
                        ["cst"],
                        value=onh.from_array(np.array([0], dtype=np.float32)),
                    ),
                    oh.make_node("Squeeze", ["cst", "axes"], ["Z"]),
                ],
                "create_graph",
                [
                    oh.make_tensor_value_info("shape", TINT64, [None]),
                    oh.make_tensor_value_info("axes", TINT64, [None]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, [None, None, None])],
            ),
            opset_imports=[
                oh.make_opsetid("", 18),
            ],
            ir_version=9,
        )
        code = to_graph_builder_code(model)

        expected = (
            textwrap.dedent(
                """
        import numpy as np
        from onnx import TensorProto
        from onnx.numpy_helper import from_array
        from experimental_experiment.xbuilder import GraphBuilder



        def create_graph(
            op: "GraphBuilder",
            shape: "INT64[]",
            axes: "INT64[]",
        ):
            cst = __LONG__
            Z = op.SqueezeAnyOpset(cst, axes)
            op.Identity(Z, outputs=["Z"])
            return Z


        def make_my_model() -> "ModelProto":
            g = GraphBuilder({'': 18}, ir_version=9)
            g.make_tensor_input("shape", TensorProto.INT64, ('',))
            g.make_tensor_input("axes", TensorProto.INT64, ('',))
            create_graph(g.op, "shape", "axes")
            g.make_tensor_output("Z", TensorProto.FLOAT, ('', '', ''))
            model = g.to_onnx()
            return model


        model = make_my_model()
        """
            )
            .strip("\n")
            .replace(
                "__LONG__",
                "op.ConstantOfShape(shape, value=from_array"
                "(np.array([0.0], dtype=np.float32), name='value'))",
            )
        )
        self.maxDiff = None
        self.assertEqual(expected, code.strip("\n"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
