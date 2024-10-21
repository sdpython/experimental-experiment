import unittest
import numpy as np
import onnx
import onnx.helper as oh
import onnx.checker as oc
from onnx import TensorProto
from experimental_experiment.ext_test_case import ExtTestCase
from experimental_experiment.xbuilder.graph_builder import GraphBuilder
from experimental_experiment.reference import ExtendedReferenceEvaluator


class TestGraphSimplification(ExtTestCase):
    def call_optimizer(self, onx):
        gr = GraphBuilder(onx, infer_shapes=True)
        assert hasattr(gr, "_debug_stop")
        gr.remove_unused()
        return gr.to_onnx()

    def test_remove_unused_nodes(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 8, opset_import: [ "": 18]>
            agraph (float[N] x) => (float[N] z) {
                two = Constant <value_float=2.0> ()
                four = Add(two, two)
                z = Mul(x, x)
            }"""
        )
        onx = self.call_optimizer(model)
        self.assertEqual(len(onx.graph.node), 1)
        self.assertEqual(onx.graph.node[0].op_type, "Mul")

    def test_initializers(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 8, opset_import: [ "": 18]>
            agraph (float[N] x) => (float[N] z)
            <float two = {2.0}> {
                four = Add(two, two)
                z = Mul(x, x)
            }"""
        )
        self.assertEqual(len(model.graph.initializer), 1)
        onx = self.call_optimizer(model)
        self.assertEqual(len(onx.graph.node), 1)
        self.assertEqual(onx.graph.node[0].op_type, "Mul")
        self.assertEqual(len(onx.graph.initializer), 0)

    def test_keep_unused_outputs(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 8, opset_import: [ "": 18]>
            agraph (float[N] x) => (float[M] z) {
                w1, w2, w3 = Split (x)
                z = Mul(w3, w3)
            }"""
        )
        onx = self.call_optimizer(model)
        self.assertEqual(len(onx.graph.node), 2)
        self.assertEqual(onx.graph.node[0].op_type, "Split")

    def test_remove_identity(self):
        opset_imports = [oh.make_opsetid("", 12)]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(oh.make_tensor_value_info("input", TensorProto.FLOAT, shape=(2, 3)))
        nodes.append(oh.make_node("Softmax", ["input"], ["output_0"], axis=0))
        nodes.append(oh.make_node("Identity", ["output_0"], ["output_1"]))
        outputs.append(oh.make_tensor_value_info("output_0", TensorProto.FLOAT, shape=(2, 3)))
        outputs.append(oh.make_tensor_value_info("output_1", TensorProto.FLOAT, shape=(2, 3)))
        graph = oh.make_graph(
            nodes,
            "experiment",
            inputs,
            outputs,
            initializers,
            sparse_initializer=sparse_initializers,
        )
        model = oh.make_model(graph, functions=functions, opset_imports=opset_imports)
        self.assertEqual(len(model.graph.node), 2)
        self.assertEqual(model.graph.node[0].op_type, "Softmax")
        self.assertEqual(model.graph.node[1].op_type, "Identity")
        self.assertEqual(len(model.graph.output), 2)
        onx = self.call_optimizer(model)
        self.assertEqual(len(onx.graph.node), 2)
        self.assertEqual(onx.graph.node[0].op_type, "Softmax")
        self.assertEqual(onx.graph.node[1].op_type, "Identity")
        self.assertEqual(len(model.graph.output), 2)

    def test_builder(self):
        gr = GraphBuilder(18, ir_version=9)
        gr.make_tensor_input("X", TensorProto.FLOAT, ("a", "b"), is_dimension=False)
        weight = gr.make_initializer("", np.array([[0.4, 0.5, 0.6]], dtype=np.float32).T)
        bias = gr.make_initializer("", np.array([[0.4, 0.5, 0.6]], dtype=np.float32))
        mm = gr.make_node("MatMul", ["X", weight], name="ut")
        out = gr.make_node("Add", [mm, bias], ["Y"], name="ut")
        gr.make_tensor_output(
            out, TensorProto.FLOAT, ("a",), indexed=False, is_dimension=False
        )
        onx = gr.to_onnx()

        ref = ExtendedReferenceEvaluator(onx)
        x = np.random.rand(10, 3).astype(np.float32)
        y = ref.run(None, {"X": x})[0]
        self.assertEqual(y.dtype, np.float32)

    def test_builder_api2(self):
        gr = GraphBuilder(18, ir_version=9)
        gr.make_tensor_input("X", TensorProto.FLOAT, ("a", "b"), is_dimension=False)
        mm = gr.op.MatMul("X", np.array([[0.4, 0.5, 0.6]], dtype=np.float32).T)
        out = gr.op.Add(mm, np.array([0.4, 0.5, 0.6], dtype=np.float32), outputs=["Y"])
        gr.make_tensor_output(
            out, TensorProto.FLOAT, ("a",), indexed=False, is_dimension=False
        )
        onx = gr.to_onnx()

        ref = ExtendedReferenceEvaluator(onx)
        x = np.random.rand(10, 3).astype(np.float32)
        y = ref.run(None, {"X": x})[0]
        self.assertEqual(y.dtype, np.float32)

    def test_remove_identity_two_paths1(self):
        opset_imports = [oh.make_opsetid("", 12)]
        nodes = [
            oh.make_node("Add", ["X", "Y"], ["add"]),
            oh.make_node("Identity", ["add"], ["add1"]),
            oh.make_node("Identity", ["add1"], ["add2"]),
            oh.make_node("Sub", ["add2", "X"], ["output1"]),
            oh.make_node("Identity", ["add"], ["output0"]),
        ]

        graph = oh.make_graph(
            nodes,
            "experiment",
            [
                oh.make_tensor_value_info("X", TensorProto.FLOAT, [4, 5]),
                oh.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 5]),
            ],
            [
                oh.make_tensor_value_info("output0", TensorProto.FLOAT, [4, 5]),
                oh.make_tensor_value_info("output1", TensorProto.FLOAT, [4, 5]),
            ],
        )
        model = oh.make_model(graph, opset_imports=opset_imports)
        onx = self.call_optimizer(model)
        self.assertEqual(["Add", "Sub"], [n.op_type for n in onx.graph.node])
        oc.check_model(onx)

    def test_remove_identity_two_paths2(self):
        opset_imports = [oh.make_opsetid("", 12)]
        nodes = [
            oh.make_node("Add", ["X", "Y"], ["add"]),
            oh.make_node("Identity", ["add"], ["add1"]),
            oh.make_node("Identity", ["add1"], ["add2"]),
            oh.make_node("Identity", ["add"], ["output0"]),
            oh.make_node("Sub", ["add2", "X"], ["output1"]),
        ]

        graph = oh.make_graph(
            nodes,
            "experiment",
            [
                oh.make_tensor_value_info("X", TensorProto.FLOAT, [4, 5]),
                oh.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 5]),
            ],
            [
                oh.make_tensor_value_info("output0", TensorProto.FLOAT, [4, 5]),
                oh.make_tensor_value_info("output1", TensorProto.FLOAT, [4, 5]),
            ],
        )
        model = oh.make_model(graph, opset_imports=opset_imports)
        onx = self.call_optimizer(model)
        self.assertEqual(["Add", "Sub"], [n.op_type for n in onx.graph.node])
        oc.check_model(onx)

    def test_remove_identity_two_paths3(self):
        opset_imports = [oh.make_opsetid("", 12)]
        nodes = [
            oh.make_node("Add", ["X", "Y"], ["add"]),
            oh.make_node("Identity", ["add"], ["add1"]),
            oh.make_node("Identity", ["add1"], ["output1"]),
            oh.make_node("Identity", ["add"], ["output0"]),
        ]

        graph = oh.make_graph(
            nodes,
            "experiment",
            [
                oh.make_tensor_value_info("X", TensorProto.FLOAT, [4, 5]),
                oh.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 5]),
            ],
            [
                oh.make_tensor_value_info("output0", TensorProto.FLOAT, [4, 5]),
                oh.make_tensor_value_info("output1", TensorProto.FLOAT, [4, 5]),
            ],
        )
        model = oh.make_model(graph, opset_imports=opset_imports)
        onx = self.call_optimizer(model)
        self.assertEqual(["Add", "Identity"], [n.op_type for n in onx.graph.node])
        oc.check_model(onx)


if __name__ == "__main__":
    unittest.main(verbosity=2)
