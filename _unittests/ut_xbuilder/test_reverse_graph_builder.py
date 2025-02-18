import textwrap
import unittest
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
import numpy as np
from experimental_experiment.ext_test_case import ExtTestCase, requires_onnx_array_api
from experimental_experiment.xbuilder.reverse_graph_builder import to_graph_builder_code
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.xbuilder import GraphBuilder, FunctionOptions

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

    @requires_onnx_array_api("0.3.1")
    def test_local_function(self):
        new_domain = "custom"

        linear_regression = oh.make_function(
            new_domain,
            "LinearRegression",
            ["x", "a", "b"],
            ["y"],
            [
                oh.make_node("MatMul", ["x", "a"], ["xa"]),
                oh.make_node("Add", ["xa", "b"], ["y"]),
            ],
            [oh.make_opsetid("", 14)],
            [],
        )

        graph = oh.make_graph(
            [
                oh.make_node("LinearRegression", ["X", "A", "B"], ["Y1"], domain=new_domain),
                oh.make_node("Abs", ["Y1"], ["Y"]),
            ],
            "example",
            [
                oh.make_tensor_value_info("X", TFLOAT, [None, None]),
                oh.make_tensor_value_info("A", TFLOAT, [None, None]),
                oh.make_tensor_value_info("B", TFLOAT, [None, None]),
            ],
            [oh.make_tensor_value_info("Y", TFLOAT, None)],
        )

        onnx_model = oh.make_model(
            graph,
            opset_imports=[oh.make_opsetid("", 14), oh.make_opsetid(new_domain, 1)],
            functions=[linear_regression],
        )
        code = to_graph_builder_code(onnx_model)

        expected = (
            textwrap.dedent(
                """
            import numpy as np
            from onnx import TensorProto
            from onnx.numpy_helper import from_array
            from experimental_experiment.xbuilder import GraphBuilder



            def example(
                op: "GraphBuilder",
                X: "FLOAT[, ]",
                A: "FLOAT[, ]",
                B: "FLOAT[, ]",
            ):
                Y1 = op.LinearRegression(X, A, B, domain='custom')
                Y = op.Abs(Y1)
                op.Identity(Y, outputs=["Y"])
                return Y


            def make_custom_LinearRegression(g: "GraphBuilder"):
                gr = GraphBuilder({'': 14}, as_function=True)
                x = gr.make_tensor_input('x')
                a = gr.make_tensor_input('a')
                b = gr.make_tensor_input('b')
                op = gr.op
                xa = op.MatMul(x, a)
                y = op.Add(xa, b)
                gr.make_tensor_output(y)
                opts = FunctionOptions(name='LinearRegression', domain='custom')
                g.make_local_function(gr, opts, optimize=False)
                return gr


            def make_my_model() -> "ModelProto":
                g = GraphBuilder({'': 14, 'custom': 1}, ir_version=11)
                g.make_tensor_input("X", TensorProto.FLOAT, ('', ''))
                g.make_tensor_input("A", TensorProto.FLOAT, ('', ''))
                g.make_tensor_input("B", TensorProto.FLOAT, ('', ''))
                example(g.op, "X", "A", "B")
                g.make_tensor_output("Y", TensorProto.FLOAT, ()__SUFFIX__)
                make_custom_LinearRegression(g)
                model = g.to_onnx()
                return model


            model = make_my_model()
        """
            )
            .strip("\n")
            .replace("__SUFFIX__", ", is_dimension=False, indexed=False")
        )
        self.maxDiff = None
        self.assertEqual(expected, code.strip("\n"))

    def test_check_local_function(self):

        def example(
            op: GraphBuilder,
            X: "FLOAT[:,C]",  # noqa: F821
            A: "FLOAT[C]",  # noqa: F821
            B: "FLOAT[C]",  # noqa: F821
        ):
            Y1 = op.LinearRegression(X, A, B, domain="custom")
            Y = op.Abs(Y1)
            op.Identity(Y, outputs=["Y"])
            return Y

        def make_custom_LinearRegression(g):
            gr = GraphBuilder({"": 14}, as_function=True)
            x = gr.make_tensor_input("x")
            a = gr.make_tensor_input("a")
            b = gr.make_tensor_input("b")
            op = gr.op
            xa = op.MatMul(x, a)
            y = op.Add(xa, b)
            gr.make_tensor_output(y)
            opts = FunctionOptions(name="LinearRegression", domain="custom")
            g.make_local_function(gr, opts, optimize=False)
            return gr

        def make_my_model() -> onnx.ModelProto:
            g = GraphBuilder({"": 14, "custom": 1}, ir_version=11)
            g.make_tensor_input("X", TFLOAT, (None, None))
            g.make_tensor_input("A", TFLOAT, (None, None))
            g.make_tensor_input("B", TFLOAT, (None, None))
            example(g.op, "X", "A", "B")
            g.make_tensor_output("Y", TFLOAT, (None, None), is_dimension=False, indexed=False)
            make_custom_LinearRegression(g)
            model = g.to_onnx()
            return model

        model = make_my_model()
        ref = ExtendedReferenceEvaluator(model)
        ref.run(
            None,
            {
                "X": np.random.randn(3, 3),
                "A": np.random.randn(3, 3),
                "B": np.random.randn(1, 3),
            },
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
