import os
import unittest
import onnx
from onnx import ModelProto
from experimental_experiment.ext_test_case import ExtTestCase, skipif_ci_windows
from experimental_experiment.xbuilder.graph_builder import (
    GraphBuilder,
    OptimizationOptions,
)


class TestGraphFullModelPatternOptimization(ExtTestCase):
    def _check_with_ort(self, proto: ModelProto):
        from onnxruntime import InferenceSession, get_available_providers

        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")
        InferenceSession(proto.SerializeToString(), providers=providers)

    def _get_model(self, name: str) -> onnx.ModelProto:
        p = os.path.join(os.path.dirname(__file__), "..", "ut_xbuilder", "data", name)
        if not os.path.exists(p):
            p = os.path.join(os.path.dirname(__file__), "data", name)
        self.assertExists(p)
        return onnx.load(p)

    def test_transpose_matmul(self):
        origin = self._get_model("llama_forward.onnx")
        before = [n.op_type for n in origin.graph.node if n.op_type == "Gemm"]
        gr = GraphBuilder(
            origin,
            optimization_options=OptimizationOptions(patterns=["TransposeMatMul"]),
            infer_shapes=True,
        )
        onx = gr.to_onnx(optimize=True)
        after = [n.op_type for n in onx.graph.node if n.op_type == "Gemm"]
        self.assertEqual(len(before), 0)
        self.assertEqual(len(after), 7)

    @skipif_ci_windows("too much memory allocated")
    def test_fused_matmul(self):
        origin = self._get_model("opt-llama-custom-backward.onnx")
        before = [n.op_type for n in origin.graph.node if n.op_type == "FusedMatMul"]
        gr = GraphBuilder(
            origin,
            optimization_options=OptimizationOptions(patterns=["FusedMatMul"], verbose=0),
            infer_shapes=True,
        )
        onx = gr.to_onnx(optimize=True)
        after = [n.op_type for n in onx.graph.node if n.op_type == "FusedMatMul"]
        self.assertEqual(len(before), 2)
        self.assertEqual(len(after), 4)

    @skipif_ci_windows("Too mcuh memory allocated.")
    def test_scatter_constant_of_shape(self):
        origin = self._get_model("opt-llama-custom-backward.onnx")
        before = [n.op_type for n in origin.graph.node if n.op_type == "ScatterNDOfShape"]
        gr = GraphBuilder(
            origin,
            optimization_options=OptimizationOptions(
                patterns=["ConstantOfShapeScatterND"],
                verbose=0,
                processor="CUDA",
            ),
            infer_shapes=True,
        )
        onx = gr.to_onnx(optimize=True)
        after = [n.op_type for n in onx.graph.node if n.op_type == "ScatterNDOfShape"]
        self.assertEqual(len(before), 0)
        self.assertEqual(len(after), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
