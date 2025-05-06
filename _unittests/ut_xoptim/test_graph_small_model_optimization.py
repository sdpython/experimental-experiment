import unittest
import numpy as np
import onnx.helper as oh
import onnx.numpy_helper as onh
from onnx import TensorProto, ModelProto
from onnx.checker import check_model
from experimental_experiment.ext_test_case import ExtTestCase
from experimental_experiment.xbuilder.graph_builder import (
    GraphBuilder,
    OptimizationOptions,
)
from experimental_experiment.torch_interpreter import to_onnx

TFLOAT = TensorProto.FLOAT


class TestGraphSmallModelOptimization(ExtTestCase):
    def _check_with_ort(self, proto: ModelProto):
        from onnxruntime import InferenceSession, get_available_providers

        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")
        InferenceSession(proto.SerializeToString(), providers=providers)

    def test_remove_unused_nodes_np(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Sqrt", ["init"], ["init_sqrt"]),
                    oh.make_node("Add", ["init_sqrt", "X"], ["Y"]),
                ],
                "dummy",
                [oh.make_tensor_value_info("X", TFLOAT, ["a", "b"])],
                [oh.make_tensor_value_info("Y", TFLOAT, ["a", "b"])],
                [onh.from_array(np.array([2], dtype=np.float32), name="init")],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        check_model(model)

        gr = GraphBuilder(model, infer_shapes_options=True)
        onx = gr.to_onnx(optimize=False)
        before = [n.op_type for n in onx.graph.node if n.op_type == "Sqrt"]
        self.assertEqual(len(before), 1)

        gr = GraphBuilder(
            model,
            optimization_options=OptimizationOptions(constant_folding=True, patterns=None),
            infer_shapes_options=True,
            verbose=0,
        )
        onx = gr.to_onnx(optimize=True)
        after = [n.op_type for n in onx.graph.node if n.op_type == "Sqrt"]
        self.assertEqual(len(after), 0)
        self._check_with_ort(onx)

    def test_remove_unused_nodes_par(self):
        import torch

        class Dummy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.par = torch.nn.Parameter(torch.Tensor([2.0]))

            def forward(self, x):
                return x + torch.sqrt(self.par)

        model = Dummy()
        x = torch.rand(5, 3)
        onx = to_onnx(
            model,
            (x,),
            options=OptimizationOptions(constant_folding=True, patterns=None),
        )
        after = [n.op_type for n in onx.graph.node if n.op_type == "Sqrt"]
        self.assertEqual(len(after), 0)
        self._check_with_ort(onx)

    def test_remove_unused_nodes_cst(self):
        import torch

        class Dummy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.cst = torch.Tensor([2.0])

            def forward(self, x):
                return x + torch.sqrt(self.cst)

        model = Dummy()
        x = torch.rand(5, 3)
        onx = to_onnx(
            model,
            (x,),
            options=OptimizationOptions(constant_folding=True, patterns=None),
            verbose=0,
        )
        after = [n.op_type for n in onx.graph.node if n.op_type == "Sqrt"]
        self.assertEqual(len(after), 0)
        self._check_with_ort(onx)


if __name__ == "__main__":
    unittest.main(verbosity=2)
