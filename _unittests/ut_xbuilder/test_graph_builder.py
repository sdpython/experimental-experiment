import unittest
import onnx.helper as oh
import numpy as np
from onnx import FunctionProto, TensorProto
from experimental_experiment.ext_test_case import ExtTestCase, hide_stdout, ignore_warnings
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.xbuilder import GraphBuilder, OnnxType


class TestTools(ExtTestCase):

    @ignore_warnings(DeprecationWarning)
    @hide_stdout
    def test_inline_1_function(self):
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
                oh.make_tensor_value_info("X", TensorProto.FLOAT, [None, None]),
                oh.make_tensor_value_info("A", TensorProto.FLOAT, [None, None]),
                oh.make_tensor_value_info("B", TensorProto.FLOAT, [None, None]),
            ],
            [oh.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
        )

        onnx_model = oh.make_model(
            graph,
            opset_imports=[oh.make_opsetid("", 14), oh.make_opsetid(new_domain, 1)],
            functions=[linear_regression],
        )
        ref = ExtendedReferenceEvaluator(onnx_model)
        feeds = dict(
            X=np.arange(9).reshape((3, 3)).astype(np.float32),
            A=np.arange(9).reshape((3, 3)).astype(np.float32),
            B=np.arange(9).reshape((3, 3)).astype(np.float32),
        )
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(onnx_model)
        self.assertEqual(len(gr.functions), 1)
        onx = gr.to_onnx()
        self.assertEqual(len(onx.functions), 1)

        self.assertRaise(
            lambda: gr.to_onnx(as_function=True, function_name="lr"), AssertionError
        )
        gr.inline_functions(verbose=1)
        function_proto = gr.to_onnx(
            as_function=True,
            function_name="lr",
            function_domain="custom_domain",
        )
        self.assertNotEmpty(function_proto)

        onx = gr.to_onnx()
        self.assertEqual(len(gr.functions), 0)
        self.assertEqual(len(onx.functions), 0)
        ref2 = ExtendedReferenceEvaluator(onx)
        got = ref2.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_inline_2_functions(self):
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

        linear_add = oh.make_function(
            new_domain,
            "LinearAdd",
            ["x", "a"],
            ["y"],
            [
                oh.make_node("Add", ["x", "a"], ["y"]),
            ],
            [oh.make_opsetid("", 14)],
            [],
        )

        graph = oh.make_graph(
            [
                oh.make_node("LinearRegression", ["X", "A", "B"], ["Y1"], domain=new_domain),
                oh.make_node("LinearAdd", ["Y1", "B"], ["Y2"], domain=new_domain),
                oh.make_node("Abs", ["Y2"], ["Y"]),
            ],
            "example",
            [
                oh.make_tensor_value_info("X", TensorProto.FLOAT, [None, None]),
                oh.make_tensor_value_info("A", TensorProto.FLOAT, [None, None]),
                oh.make_tensor_value_info("B", TensorProto.FLOAT, [None, None]),
            ],
            [oh.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
        )

        onnx_model = oh.make_model(
            graph,
            opset_imports=[oh.make_opsetid("", 14), oh.make_opsetid(new_domain, 1)],
            functions=[linear_regression, linear_add],
        )
        ref = ExtendedReferenceEvaluator(onnx_model)
        feeds = dict(
            X=np.arange(9).reshape((3, 3)).astype(np.float32),
            A=np.arange(9).reshape((3, 3)).astype(np.float32),
            B=np.arange(9).reshape((3, 3)).astype(np.float32),
        )
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(onnx_model)
        self.assertEqual(len(gr.functions), 2)
        onx = gr.to_onnx()
        self.assertEqual(len(onx.functions), 2)

        self.assertRaise(
            lambda: gr.to_onnx(as_function=True, function_name="lr"),
            AssertionError,
        )
        gr.inline_functions()
        function_proto = gr.to_onnx(
            as_function=True,
            function_name="lr",
            function_domain="custom_domain",
        )
        self.assertNotEmpty(function_proto)

        onx = gr.to_onnx()
        self.assertEqual(len(gr.functions), 0)
        self.assertEqual(len(onx.functions), 0)
        ref2 = ExtendedReferenceEvaluator(onx)
        got = ref2.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_inline_2_functions_recursive(self):
        new_domain = "custom"

        linear_add = oh.make_function(
            new_domain,
            "LinearAdd",
            ["x", "a"],
            ["y"],
            [
                oh.make_node("Add", ["x", "a"], ["y"]),
            ],
            [oh.make_opsetid("", 14)],
            [],
        )

        linear_regression = oh.make_function(
            new_domain,
            "LinearRegression",
            ["x", "a", "b"],
            ["y"],
            [
                oh.make_node("MatMul", ["x", "a"], ["xa"]),
                oh.make_node("LinearAdd", ["xa", "b"], ["y"], domain=new_domain),
            ],
            [oh.make_opsetid("", 14), oh.make_opsetid(new_domain, 1)],
            [],
        )

        graph = oh.make_graph(
            [
                oh.make_node("LinearRegression", ["X", "A", "B"], ["Y2"], domain=new_domain),
                oh.make_node("Abs", ["Y2"], ["Y"]),
            ],
            "example",
            [
                oh.make_tensor_value_info("X", TensorProto.FLOAT, [None, None]),
                oh.make_tensor_value_info("A", TensorProto.FLOAT, [None, None]),
                oh.make_tensor_value_info("B", TensorProto.FLOAT, [None, None]),
            ],
            [oh.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
        )

        onnx_model = oh.make_model(
            graph,
            opset_imports=[oh.make_opsetid("", 14), oh.make_opsetid(new_domain, 1)],
            functions=[linear_add, linear_regression],
        )
        ref = ExtendedReferenceEvaluator(onnx_model)
        feeds = dict(
            X=np.arange(9).reshape((3, 3)).astype(np.float32),
            A=np.arange(9).reshape((3, 3)).astype(np.float32),
            B=np.arange(9).reshape((3, 3)).astype(np.float32),
        )
        expected = ref.run(None, feeds)[0]

        gr = GraphBuilder(onnx_model)
        self.assertEqual(len(gr.functions), 2)
        onx = gr.to_onnx()
        self.assertEqual(len(onx.functions), 2)

        self.assertRaise(
            lambda: gr.to_onnx(as_function=True, function_name="lr"), AssertionError
        )
        gr.inline_functions()
        function_proto = gr.to_onnx(
            as_function=True,
            function_name="lr",
            function_domain="custom_domain",
        )
        self.assertNotEmpty(function_proto)

        onx = gr.to_onnx()
        self.assertEqual(len(gr.functions), 0)
        self.assertEqual(len(onx.functions), 0)
        ref2 = ExtendedReferenceEvaluator(onx)
        got = ref2.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    @ignore_warnings(DeprecationWarning)
    def test_as_function_constant(self):
        g = GraphBuilder(18, ir_version=9, as_function=True)
        g.make_tensor_input("X", None, None, False)
        np_weights = np.random.randn(4, 3).astype(np.float32)
        np_bias = np.random.randn(1, 3).astype(np.float32)
        init = g.make_initializer("weights", np_weights)
        bias = g.make_initializer("bias", np_bias)
        g.op.Add(g.op.MatMul("X", init, name="linear"), bias, name="linear", outputs=["Y"])
        g.make_tensor_output("Y", is_dimension=False, indexed=False)
        g.move_initializers_to_constant()
        fct = g.to_onnx(
            as_function=True,
            function_name="linear",
            function_domain="mine",
        )
        feeds = dict(X=np.random.randn(2, 4).astype(np.float32))
        expected = feeds["X"] @ np_weights + np_bias
        ref = ExtendedReferenceEvaluator(fct)
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0])

    @ignore_warnings(DeprecationWarning)
    def test_as_function_second(self):
        gf = GraphBuilder(18, ir_version=9, as_function=True)
        gf.make_tensor_input("X", None, None, False)
        np_weights = np.arange(12).reshape((4, 3)).astype(np.float32) / 10
        np_bias = np.arange(3).reshape((1, 3)).astype(np.float32) + 10
        np_bias2 = np.arange(3).reshape((1, 3)).astype(np.float32) + 1000

        init = gf.make_initializer("weights", np_weights)
        bias = gf.make_initializer("bias", np_bias)
        gf.op.Add(gf.op.MatMul("X", init, name="linear"), bias, name="linear", outputs=["Y"])
        gf.make_tensor_output("Y", is_dimension=False, indexed=False)
        self.assertEqualArray(gf.initializers_dict["weights"], np_weights)

        g = GraphBuilder(18, ir_version=9, as_function=True)
        g.make_tensor_input("X", None, None, False)
        new_inits, _ = g.make_local_function(
            "Regression", gf, domain="custom", move_initializer_to_constant=False
        )
        self.assertEqual(new_inits, ["Regression_weights", "Regression_bias"])
        self.assertEqualArray(g.initializers_dict["Regression_weights"], np_weights)

        bias2 = g.make_initializer("bias2", np_bias2)
        g.op.Add(
            g.anyop.Regression("X", *new_inits, name="linear", domain="custom"),
            bias2,
            outputs=["Y"],
        )
        g.make_tensor_output("Y", is_dimension=False, indexed=False)
        nodes = [(node.domain, node.op_type, node.input, node.output) for node in g.nodes]
        self.assertEqual(
            nodes,
            [
                (
                    "custom",
                    "Regression",
                    ["X", "Regression_weights", "Regression_bias"],
                    ["_onx_regression0"],
                ),
                ("", "Add", ["_onx_regression0", "bias2"], ["Y"]),
            ],
        )

        self.assertRaise(
            lambda: g.to_onnx(
                as_function=True, function_name="linear", function_domain="mine"
            ),
            AssertionError,
        )
        self.assertEqual(len(g.functions), 1)

        # finally, the conversion to onnx
        text = g.print_text()
        self.assertIn("['_onx_regression0', 'bias2']", text)
        fct = g.to_onnx(
            as_function=OnnxType.FUNCTION_AND_INITIALIZERS,
            function_name="linear",
            function_domain="mine",
        )

        self.assertIsInstance(fct, dict)
        self.assertEqual(
            set(fct), {"proto", "functions", "initializers_name", "initializers_dict"}
        )
        self.assertIsInstance(fct["proto"], FunctionProto)
        self.assertIsInstance(fct["functions"], list)
        self.assertTrue(all(isinstance(p, FunctionProto) for p in fct["functions"]))
        self.assertIsInstance(fct["initializers_name"], list)
        self.assertEqual(
            fct["initializers_name"], ["Regression_weights", "bias2", "Regression_bias"]
        )
        self.assertIsInstance(fct["initializers_dict"], dict)
        self.assertTrue(
            all(isinstance(p, np.ndarray) for p in fct["initializers_dict"].values())
        )
        self.assertEqual(len(fct["initializers_name"]), len(fct["initializers_dict"]))
        proto = fct["proto"]
        self.assertEqual(proto.output, ["Y"])
        self.assertEqual(proto.input, ["X", "Regression_weights", "bias2", "Regression_bias"])
        self.assertEqual(proto.domain, "mine")
        self.assertEqual(proto.name, "linear")
        f1 = fct["functions"][0]
        self.assertEqual(f1.domain, "custom")
        self.assertEqual(f1.name, "Regression")
        self.assertEqual(f1.output, ["Y"])
        self.assertEqual(f1.input, ["X", "weights", "bias"])

        feeds = dict(X=np.random.randn(2, 4).astype(np.float32))
        feeds.update(fct["initializers_dict"])
        self.assertEqualArray(np_weights, feeds["Regression_weights"])
        self.assertEqualArray(np_bias, feeds["Regression_bias"])
        self.assertEqualArray(np_bias2, feeds["bias2"])
        self.assertEqual(set(feeds), {"X", "Regression_weights", "bias2", "Regression_bias"})
        expected = feeds["X"] @ np_weights + np_bias + np_bias2
        ref = ExtendedReferenceEvaluator(fct["proto"], functions=fct["functions"])
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0])

    @ignore_warnings(DeprecationWarning)
    def test_as_function_nested_unique(self):
        np_weights = np.arange(12).reshape((4, 3)).astype(np.float32) / 10
        np_bias = np.arange(3).reshape((1, 3)).astype(np.float32) + 10
        np_bias2 = np.arange(3).reshape((1, 3)).astype(np.float32) + 100
        np_bias3 = np.arange(3).reshape((1, 3)).astype(np.float32) + 1000

        # first function
        gf = GraphBuilder(18, ir_version=9, as_function=True)
        gf.make_tensor_input("X", None, None, False)
        init = gf.make_initializer("weights", np_weights)
        bias = gf.make_initializer("bias", np_bias)
        gf.op.Add(gf.op.MatMul("X", init, name="linear"), bias, name="linear", outputs=["Y"])
        gf.make_tensor_output("Y", is_dimension=False, indexed=False)
        self.assertEqualArray(gf.initializers_dict["weights"], np_weights)

        # second function calling the first one
        g2 = GraphBuilder(18, ir_version=9, as_function=True)
        g2.make_tensor_input("X", None, None, False)
        new_inits, _ = g2.make_local_function(
            "Regression", gf, domain="custom", move_initializer_to_constant=False
        )

        bias2 = g2.make_initializer("bias2", np_bias2)
        g2.op.Add(
            g2.anyop.Regression("X", *new_inits, name="addc", domain="custom"),
            bias2,
            outputs=["Y"],
        )
        g2.make_tensor_output("Y", is_dimension=False, indexed=False)

        # a last step
        # second function calling the first one
        g = GraphBuilder(18, ir_version=9, as_function=True)
        g.make_tensor_input("X", None, None, False)
        new_inits, _ = g.make_local_function(
            "RegressionBias", g2, domain="custom", move_initializer_to_constant=False
        )
        self.assertEqual(len(g.functions), 2)

        bias3 = g.make_initializer("bias3", np_bias3)
        g.op.Add(
            g.anyop.RegressionBias("X", *new_inits, name="addd", domain="custom"),
            bias3,
            outputs=["Y"],
        )
        g.make_tensor_output("Y", is_dimension=False, indexed=False)

        # finally, the conversion to onnx
        self.assertIn("FUNC RegressionBias[custom]", g.print_text())

        fct = g.to_onnx(
            as_function=OnnxType.FUNCTION_AND_INITIALIZERS,
            function_name="linear",
            function_domain="mine",
        )

        self.assertIsInstance(fct, dict)
        self.assertEqual(
            set(fct), {"proto", "functions", "initializers_name", "initializers_dict"}
        )
        self.assertIsInstance(fct["proto"], FunctionProto)
        self.assertIsInstance(fct["functions"], list)
        self.assertTrue(all(isinstance(p, FunctionProto) for p in fct["functions"]))
        self.assertIsInstance(fct["initializers_name"], list)
        self.assertEqual(
            fct["initializers_name"],
            [
                "RegressionBias_Regression_weights",
                "bias3",
                "RegressionBias_bias2",
                "RegressionBias_Regression_bias",
            ],
        )
        self.assertIsInstance(fct["initializers_dict"], dict)
        self.assertTrue(
            all(isinstance(p, np.ndarray) for p in fct["initializers_dict"].values())
        )
        self.assertEqual(len(fct["initializers_name"]), len(fct["initializers_dict"]))
        proto = fct["proto"]
        self.assertEqual(proto.output, ["Y"])
        self.assertEqual(
            proto.input,
            [
                "X",
                "RegressionBias_Regression_weights",
                "bias3",
                "RegressionBias_bias2",
                "RegressionBias_Regression_bias",
            ],
        )
        self.assertEqual(proto.domain, "mine")
        self.assertEqual(proto.name, "linear")
        self.assertEqual(2, len(fct["functions"]))
        f1 = fct["functions"][0]
        self.assertEqual(f1.domain, "custom")
        self.assertEqual(f1.name, "Regression")
        self.assertEqual(f1.output, ["Y"])
        self.assertEqual(f1.input, ["X", "weights", "bias"])
        f2 = fct["functions"][1]
        self.assertEqual(f2.domain, "custom")
        self.assertEqual(f2.name, "RegressionBias")
        self.assertEqual(f2.output, ["Y"])
        self.assertEqual(f2.input, ["X", "Regression_weights", "bias2", "Regression_bias"])

        feeds = dict(X=np.random.randn(2, 4).astype(np.float32))
        feeds.update(fct["initializers_dict"])
        self.assertEqualArray(np_weights, feeds["RegressionBias_Regression_weights"])
        self.assertEqualArray(np_bias, feeds["RegressionBias_Regression_bias"])
        self.assertEqualArray(np_bias2, feeds["RegressionBias_bias2"])
        self.assertEqualArray(np_bias3, feeds["bias3"])
        self.assertEqual(
            set(feeds),
            {
                "X",
                "RegressionBias_Regression_weights",
                "RegressionBias_Regression_bias",
                "bias3",
                "RegressionBias_bias2",
            },
        )
        expected = feeds["X"] @ np_weights + np_bias + np_bias2 + np_bias3

        # Evaluation of a function
        self.assertIn("opset: '': 18", g.print_text())
        ref = ExtendedReferenceEvaluator(fct["proto"], functions=fct["functions"])
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0])

        # Same with a model
        proto = g.to_onnx()
        self.assertEqual(len(proto.functions), 2)
        ref = ExtendedReferenceEvaluator(proto)
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0])

    @ignore_warnings(DeprecationWarning)
    def test_as_function_second_twice(self):
        np_weights = np.arange(12).reshape((4, 3)).astype(np.float32) / 10
        np_bias = np.arange(3).reshape((1, 3)).astype(np.float32) + 10

        # function 1
        gf = GraphBuilder(18, ir_version=9, as_function=True)
        gf.make_tensor_input("X", None, None, False)
        init = gf.make_initializer("weights", np_weights)
        bias = gf.make_initializer("bias", np_bias)
        gf.op.Add(gf.op.MatMul("X", init, name="linear"), bias, name="linear", outputs=["Y"])
        gf.make_tensor_output("Y", is_dimension=False, indexed=False)
        self.assertEqualArray(gf.initializers_dict["weights"], np_weights)

        # main graph
        g = GraphBuilder(18, ir_version=9, as_function=True)
        g.make_tensor_input("X", None, None, False)
        new_inits, _ = g.make_local_function(
            "Regression", gf, domain="custom", move_initializer_to_constant=False
        )
        self.assertEqual(len(g.functions), 1)
        self.assertEqual(new_inits, ["Regression_weights", "Regression_bias"])
        self.assertEqualArray(g.initializers_dict["Regression_weights"], np_weights)

        # function 3: the same name but different
        gf = GraphBuilder(18, ir_version=9, as_function=True)
        gf.make_tensor_input("X", None, None, False)

        init = gf.make_initializer("weights", np_weights)
        bias = gf.make_initializer("bias", np_bias)
        gf.op.Sub(gf.op.MatMul("X", init, name="linear"), bias, name="linear", outputs=["Y"])
        gf.make_tensor_output("Y", is_dimension=False, indexed=False)
        self.assertEqualArray(gf.initializers_dict["weights"], np_weights)

        self.assertEqual(len(g.functions), 1)
        new_inits_2, (domain_name, function_name) = g.make_local_function(
            "Regression",
            gf,
            domain="custom",
            move_initializer_to_constant=False,
            rename_allowed=True,
        )
        self.assertEqual(len(g.functions), 2)
        self.assertEqual(new_inits, ["Regression_weights", "Regression_bias"])
        self.assertEqualArray(g.initializers_dict["Regression_weights"], np_weights)

        # two functions
        g.op.Add(
            g.anyop.Regression("X", *new_inits, name="linear", domain="custom"),
            g.make_node(function_name, ["X", *new_inits_2], name="linear", domain=domain_name),
            outputs=["Y"],
        )
        g.make_tensor_output("Y", is_dimension=False, indexed=False)
        self.assertEqual(len(g.functions), 2)

        # finally, the conversion to onnx
        fct = g.to_onnx(
            as_function=OnnxType.FUNCTION_AND_INITIALIZERS,
            function_name="linear",
            function_domain="mine",
        )

        self.assertIsInstance(fct, dict)
        self.assertEqual(
            set(fct), {"proto", "functions", "initializers_name", "initializers_dict"}
        )
        self.assertIsInstance(fct["proto"], FunctionProto)
        self.assertIsInstance(fct["functions"], list)
        self.assertTrue(all(isinstance(p, FunctionProto) for p in fct["functions"]))
        self.assertIsInstance(fct["initializers_name"], list)
        self.assertEqual(
            fct["initializers_name"],
            [
                "Regression_weights2",
                "Regression_weights",
                "Regression_bias2",
                "Regression_bias",
            ],
        )
        self.assertIsInstance(fct["initializers_dict"], dict)
        self.assertTrue(
            all(isinstance(p, np.ndarray) for p in fct["initializers_dict"].values())
        )
        self.assertEqual(len(fct["initializers_name"]), len(fct["initializers_dict"]))
        proto = fct["proto"]
        self.assertEqual(proto.output, ["Y"])
        self.assertEqual(
            proto.input,
            [
                "X",
                "Regression_weights2",
                "Regression_weights",
                "Regression_bias2",
                "Regression_bias",
            ],
        )
        self.assertEqual(proto.domain, "mine")
        self.assertEqual(proto.name, "linear")
        f1 = fct["functions"][0]
        self.assertEqual(f1.domain, "custom")
        self.assertEqual(f1.name, "Regression")
        self.assertEqual(f1.output, ["Y"])
        self.assertEqual(f1.input, ["X", "weights", "bias"])
        f2 = fct["functions"][1]
        self.assertEqual(f2.domain, "custom")
        self.assertEqual(f2.name, "Regression_2")
        self.assertEqual(f2.output, ["Y"])
        self.assertEqual(f2.input, ["X", "weights", "bias"])

        feeds = dict(X=np.random.randn(2, 4).astype(np.float32))
        feeds.update(fct["initializers_dict"])
        expected = feeds["X"] @ np_weights + np_bias + feeds["X"] @ np_weights - np_bias
        ref = ExtendedReferenceEvaluator(fct["proto"], functions=fct["functions"])
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0], atol=1e-5)

    @ignore_warnings(DeprecationWarning)
    def test_as_function_nested_twice(self):

        def _make_function():
            np_weights = np.arange(12).reshape((4, 3)).astype(np.float32) / 10
            np_bias = np.arange(3).reshape((1, 3)).astype(np.float32) + 10
            np_bias2 = np.arange(3).reshape((1, 3)).astype(np.float32) + 100

            # first function
            gf = GraphBuilder(18, ir_version=9, as_function=True)
            gf.make_tensor_input("X", None, None, False)
            init = gf.make_initializer("weights", np_weights)
            bias = gf.make_initializer("bias", np_bias)
            gf.op.Add(
                gf.op.MatMul("X", init, name="linear"), bias, name="linear", outputs=["Y"]
            )
            gf.make_tensor_output("Y", is_dimension=False, indexed=False)
            self.assertEqualArray(gf.initializers_dict["weights"], np_weights)

            # second function calling the first one
            g2 = GraphBuilder(18, ir_version=9, as_function=True)
            g2.make_tensor_input("X", None, None, False)
            new_inits, _ = g2.make_local_function(
                "Regression", gf, domain="custom", move_initializer_to_constant=False
            )

            bias2 = g2.make_initializer("bias2", np_bias2)
            g2.op.Add(
                g2.anyop.Regression("X", *new_inits, name="addc", domain="custom"),
                bias2,
                outputs=["Y"],
            )
            g2.make_tensor_output("Y", is_dimension=False, indexed=False)
            return g2

        np_weights = np.arange(12).reshape((4, 3)).astype(np.float32) / 10
        np_bias = np.arange(3).reshape((1, 3)).astype(np.float32) + 10
        np_bias2 = np.arange(3).reshape((1, 3)).astype(np.float32) + 100

        # a last step
        # second function calling the first one
        g = GraphBuilder(18, ir_version=9, as_function=True)
        g.make_tensor_input("X", None, None, False)

        # let's add the first function
        g1 = _make_function()
        new_inits_1, _ = g.make_local_function(
            "RegressionBias", g1, domain="custom", move_initializer_to_constant=False
        )
        self.assertEqual(len(g.functions), 2)
        # let's add the second function
        g2 = _make_function()
        new_inits_2, (domain_name, function_name) = g.make_local_function(
            "RegressionBias",
            g2,
            domain="custom",
            move_initializer_to_constant=False,
            rename_allowed=True,
        )
        self.assertEqual(len(g.functions), 4)

        g.op.Add(
            g.anyop.RegressionBias("X", *new_inits_1, name="reg2", domain="custom"),
            g.make_node(function_name, ["X", *new_inits_2], name="reg2", domain=domain_name),
            outputs=["Y"],
        )
        g.make_tensor_output("Y", is_dimension=False, indexed=False)

        # finally, the conversion to onnx
        self.assertIn("FUNC RegressionBias[custom]", g.print_text())

        fct = g.to_onnx(
            as_function=OnnxType.FUNCTION_AND_INITIALIZERS,
            function_name="linear",
            function_domain="mine",
        )

        self.assertIsInstance(fct, dict)
        self.assertEqual(
            set(fct), {"proto", "functions", "initializers_name", "initializers_dict"}
        )
        self.assertIsInstance(fct["proto"], FunctionProto)
        self.assertIsInstance(fct["functions"], list)
        self.assertTrue(all(isinstance(p, FunctionProto) for p in fct["functions"]))
        self.assertIsInstance(fct["initializers_name"], list)
        self.assertEqual(
            fct["initializers_name"],
            [
                "RegressionBias_Regression_weights2",
                "RegressionBias_Regression_weights",
                "RegressionBias_bias22",
                "RegressionBias_bias2",
                "RegressionBias_Regression_bias2",
                "RegressionBias_Regression_bias",
            ],
        )
        self.assertIsInstance(fct["initializers_dict"], dict)
        self.assertTrue(
            all(isinstance(p, np.ndarray) for p in fct["initializers_dict"].values())
        )
        self.assertEqual(len(fct["initializers_name"]), len(fct["initializers_dict"]))
        proto = fct["proto"]
        self.assertEqual(proto.output, ["Y"])
        self.assertEqual(
            proto.input,
            [
                "X",
                "RegressionBias_Regression_weights2",
                "RegressionBias_Regression_weights",
                "RegressionBias_bias22",
                "RegressionBias_bias2",
                "RegressionBias_Regression_bias2",
                "RegressionBias_Regression_bias",
            ],
        )
        self.assertEqual(proto.domain, "mine")
        self.assertEqual(proto.name, "linear")
        self.assertEqual(4, len(fct["functions"]))
        f1 = fct["functions"][0]
        self.assertEqual(f1.domain, "custom")
        self.assertEqual(f1.name, "Regression")
        self.assertEqual(f1.output, ["Y"])
        self.assertEqual(f1.input, ["X", "weights", "bias"])
        f2 = fct["functions"][1]
        self.assertEqual(f2.domain, "custom")
        self.assertEqual(f2.name, "RegressionBias")
        self.assertEqual(f2.output, ["Y"])
        self.assertEqual(f2.input, ["X", "Regression_weights", "bias2", "Regression_bias"])

        feeds = dict(X=np.random.randn(2, 4).astype(np.float32))
        feeds.update(fct["initializers_dict"])
        self.assertEqualArray(np_weights, feeds["RegressionBias_Regression_weights"])
        self.assertEqualArray(np_bias, feeds["RegressionBias_Regression_bias"])
        self.assertEqualArray(np_bias2, feeds["RegressionBias_bias2"])
        expected = (feeds["X"] @ np_weights + np_bias + np_bias2) * 2

        # Evaluation of a function
        self.assertIn("opset: '': 18", g.print_text())
        ref = ExtendedReferenceEvaluator(fct["proto"], functions=fct["functions"])
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0])

        # Same with a model
        proto = g.to_onnx()
        self.assertEqual(len(proto.functions), 4)
        ref = ExtendedReferenceEvaluator(proto)
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
