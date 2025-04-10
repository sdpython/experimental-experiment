import inspect
import textwrap
import unittest
from typing import List, Optional
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
import numpy as np
from experimental_experiment.ext_test_case import ExtTestCase, requires_onnx_array_api
from experimental_experiment.xbuilder.reverse_graph_builder import (
    to_graph_builder_code,
    to_graph_pattern_matching,
)
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.xbuilder import GraphBuilder, FunctionOptions, OptimizationOptions
from experimental_experiment.xbuilder._internal.onnx_export import export2numpy
from experimental_experiment.xoptim.patterns_api import MatchResult, PatternOptimization

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
        from experimental_experiment.xbuilder import GraphBuilder, FunctionOptions



        def create_graph(
            op: "GraphBuilder",
            shape: "INT64[]",
            axes: "INT64[]",
        ):
            cst = __LONG__
            Z = op.SqueezeAnyOpset(cst, axes, outputs=['Z'])
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
            from experimental_experiment.xbuilder import GraphBuilder, FunctionOptions



            def example(
                op: "GraphBuilder",
                X: "FLOAT[, ]",
                A: "FLOAT[, ]",
                B: "FLOAT[, ]",
            ):
                Y1 = op.LinearRegression(X, A, B, domain='custom', outputs=['Y1'])
                Y = op.Abs(Y1, outputs=['Y'])
                op.Identity(Y, outputs=["Y"])
                return Y


            def make_custom_LinearRegression(g: "GraphBuilder"):
                gr = GraphBuilder({'': 14}, as_function=True)
                x = gr.make_tensor_input('x')
                a = gr.make_tensor_input('a')
                b = gr.make_tensor_input('b')
                op = gr.op
                xa = op.MatMul(x, a, outputs=['xa'])
                y = op.Add(xa, b, outputs=['y'])
                gr.make_tensor_output(y)
                opts = FunctionOptions(
                    name='LinearRegression',
                    domain='custom',
                    move_initializer_to_constant=True,
                )
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

    def test_to_graph_pattern_matching(self):
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
        code = to_graph_pattern_matching(model)
        self.assertIn("return self.none()", code)

    def test_to_graph_pattern_matching_6(self):

        class DummyPatternMatching(PatternOptimization):
            def match(
                self,
                g: "GraphBuilderPatternOptimization",  # noqa: F821
                node: onnx.NodeProto,
                matched: List[MatchResult],
            ) -> Optional[MatchResult]:
                node_6_Reshape = node
                if node_6_Reshape.op_type != "Reshape" or node_6_Reshape.domain != "":
                    return self.none()
                xm = node_6_Reshape.input[0]
                # shape3 = node_6_Reshape.input[1]

                # shape3 has no predecessor.

                if g.is_used_more_than_once(xm):
                    return self.none(node, inspect.currentframe().f_lineno)
                node_5_MatMul = g.node_before(xm)
                if (
                    node_5_MatMul is None
                    or node_5_MatMul.op_type != "MatMul"
                    or node_5_MatMul.domain != ""
                ):
                    return self.none(node, inspect.currentframe().f_lineno)
                xm1 = node_5_MatMul.input[0]
                xm2 = node_5_MatMul.input[1]

                if g.is_used_more_than_once(xm2):
                    return self.none(node, inspect.currentframe().f_lineno)
                node_4_Cast = g.node_before(xm2)
                if (
                    node_4_Cast is None
                    or node_4_Cast.op_type != "Cast"
                    or node_4_Cast.domain != ""
                ):
                    return self.none(node, inspect.currentframe().f_lineno)
                xm2c = node_4_Cast.input[0]

                if g.is_used_more_than_once(xm2c):
                    return self.none(node, inspect.currentframe().f_lineno)
                node_3_Reshape = g.node_before(xm2c)
                if (
                    node_3_Reshape is None
                    or node_3_Reshape.op_type != "Reshape"
                    or node_3_Reshape.domain != ""
                ):
                    return self.none(node, inspect.currentframe().f_lineno)
                # Y = node_3_Reshape.input[0]
                # shape2 = node_3_Reshape.input[1]

                # shape2 has no predecessor.

                # Y has no predecessor.

                if g.is_used_more_than_once(xm1):
                    return self.none(node, inspect.currentframe().f_lineno)
                node_2_Reshape = g.node_before(xm1)
                if (
                    node_2_Reshape is None
                    or node_2_Reshape.op_type != "Reshape"
                    or node_2_Reshape.domain != ""
                ):
                    return self.none(node, inspect.currentframe().f_lineno)
                xu2 = node_2_Reshape.input[0]
                # shape1 = node_2_Reshape.input[1]

                # shape1 has no predecessor.

                if g.is_used_more_than_once(xu2):
                    return self.none(node, inspect.currentframe().f_lineno)
                node_1_Unsqueeze = g.node_before(xu2)
                if (
                    node_1_Unsqueeze is None
                    or node_1_Unsqueeze.op_type != "Unsqueeze"
                    or node_1_Unsqueeze.domain != ""
                ):
                    return self.none(node, inspect.currentframe().f_lineno)
                xu1 = node_1_Unsqueeze.input[0]
                # un = node_1_Unsqueeze.input[1]

                # un has no predecessor.

                if g.is_used_more_than_once(xu1):
                    return self.none(node, inspect.currentframe().f_lineno)
                node_0_Unsqueeze = g.node_before(xu1)
                if (
                    node_0_Unsqueeze is None
                    or node_0_Unsqueeze.op_type != "Unsqueeze"
                    or node_0_Unsqueeze.domain != ""
                ):
                    return self.none(node, inspect.currentframe().f_lineno)
                # X = node_0_Unsqueeze.input[0]
                # zero = node_0_Unsqueeze.input[1]

                # zero has no predecessor.

                # X has no predecessor.

                # list of nodes
                nodes = [
                    node_6_Reshape,
                    node_5_MatMul,
                    node_4_Cast,
                    node_3_Reshape,
                    node_2_Reshape,
                    node_1_Unsqueeze,
                    node_0_Unsqueeze,
                ]
                return MatchResult(self, nodes, self.apply)

            def apply(
                self,
                g: "GraphBuilder",  # noqa: F821
                *nodes: onnx.NodeProto,
            ) -> List[onnx.NodeProto]:
                new_node = g.make_node(
                    "Dummy",
                    nodes[-1].input[0],
                    nodes[0].output,
                    name=f"{self.__class__.__name__}--{nodes[0].name}",
                    doc_string=nodes[0].doc_string,
                    domain="dummy",
                )
                return [new_node]

        _mkv_ = oh.make_tensor_value_info
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Unsqueeze", ["X", "zero"], ["xu1"]),
                    oh.make_node("Unsqueeze", ["xu1", "un"], ["xu2"]),
                    oh.make_node("Reshape", ["xu2", "shape1"], ["xm1"]),
                    oh.make_node("Reshape", ["Y", "shape2"], ["xm2c"]),
                    oh.make_node("Cast", ["xm2c"], ["xm2"], to=1),
                    oh.make_node("MatMul", ["xm1", "xm2"], ["xm"]),
                    oh.make_node("Reshape", ["xm", "shape3"], ["Z"]),
                ],
                "dummy",
                [
                    _mkv_("X", TFLOAT, ["D32", "D128"]),
                    _mkv_("Y", TFLOAT, ["batch", "channel", "D128", "D64"]),
                ],
                [_mkv_("Z", TFLOAT, ["batch", "channel", "D32", "64"])],
                [
                    onh.from_array(np.array([0], dtype=np.int64), name="zero"),
                    onh.from_array(np.array([1], dtype=np.int64), name="un"),
                    onh.from_array(np.array([1, 32, 128], dtype=np.int64), name="shape1"),
                    onh.from_array(np.array([15, 128, 64], dtype=np.int64), name="shape2"),
                    onh.from_array(np.array([3, 5, 32, 64], dtype=np.int64), name="shape3"),
                ],
            )
        )
        code = to_graph_pattern_matching(model)
        self.assertIn("nodes = [", code)
        # print(code)
        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(
                patterns=[DummyPatternMatching(verbose=10)], verbose=10
            ),
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(len(opt_onx.graph.node), 1)

    def test_export2numpy(self):
        _mkv_ = oh.make_tensor_value_info
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Unsqueeze", ["X", "zero"], ["xu1"]),
                    oh.make_node("Unsqueeze", ["xu1", "un"], ["xu2"]),
                    oh.make_node("Reshape", ["xu2", "shape1"], ["xm1"]),
                    oh.make_node("Reshape", ["Y", "shape2"], ["xm2c"]),
                    oh.make_node("Cast", ["xm2c"], ["xm2"], to=1),
                    oh.make_node("MatMul", ["xm1", "xm2"], ["xm"]),
                    oh.make_node("Reshape", ["xm", "shape3"], ["Z"]),
                ],
                "dummy",
                [
                    _mkv_("X", TFLOAT, ["D32", "D128"]),
                    _mkv_("Y", TFLOAT, ["batch", "channel", "D128", "D64"]),
                ],
                [_mkv_("Z", TFLOAT, ["batch", "channel", "D32", "64"])],
                [
                    onh.from_array(np.array([0], dtype=np.int64), name="zero"),
                    onh.from_array(np.array([1], dtype=np.int64), name="un"),
                    onh.from_array(np.array([1, 32, 128], dtype=np.int64), name="shape1"),
                    onh.from_array(np.array([15, 128, 64], dtype=np.int64), name="shape2"),
                    onh.from_array(np.array([3, 5, 32, 64], dtype=np.int64), name="shape3"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
        )

        code = export2numpy(model)
        self.assertIn("xm = xm1 @ xm2", code)


if __name__ == "__main__":
    unittest.main(verbosity=2)
