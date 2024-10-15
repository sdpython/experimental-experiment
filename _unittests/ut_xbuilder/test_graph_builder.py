import unittest
import onnx.helper as oh
import numpy as np
from onnx import TensorProto
from experimental_experiment.ext_test_case import ExtTestCase, hide_stdout, ignore_warnings
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.xbuilder import GraphBuilder


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
