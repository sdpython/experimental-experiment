import os
import unittest
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
from typing import Optional
from experimental_experiment.ext_test_case import ExtTestCase
from experimental_experiment.xbuilder.graph_builder import (
    GraphBuilder,
    OptimizationOptions,
)
from experimental_experiment.xoptim import get_pattern_list
from experimental_experiment.reference import ExtendedReferenceEvaluator

TFLOAT = onnx.TensorProto.FLOAT
_mkv_ = oh.make_tensor_value_info


class TestGraphPatternOptimizationInvestigation(ExtTestCase):
    def _range(self, *shape, bias: Optional[float] = None):
        n = np.prod(shape)
        x = np.arange(n).astype(np.float32) / n
        if bias:
            x = x + bias
        return x.reshape(tuple(shape)).astype(np.float32)

    def _get_model(self, name: str) -> onnx.ModelProto:
        p = os.path.join(os.path.dirname(__file__), "..", "ut_xbuilder", "data", name)
        if not os.path.exists(p):
            p = os.path.join(os.path.dirname(__file__), "data", name)
        self.assertExists(p)
        return onnx.load(p)

    def test_get_pattern_list(self):
        res = get_pattern_list("investigation")
        names = set(r.__class__.__name__ for r in res)
        self.assertIn("BinaryInvestigation", names)

    def test_binary_ops(self):
        model = self._get_model("noopt-llama-custom__1.onnx")

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(
                patterns=["BinaryInvestigation"], verbose=1
            ),
        )
        opt_onx, out, err = self.capture(lambda: gr.to_onnx(optimize=True))
        self.assertIn("[BinaryInvestigation] Mul(2x2x1024x512, 1x1x1024x512)", out)
        self.assertGreater(len(model.graph.node), len(opt_onx.graph.node))

    def test_dump_applied_patterns(self):
        model = self._get_model("noopt-llama-custom__1.onnx")

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(
                patterns=["RotaryConcatPart"],
                dump_applied_patterns="test_dump_applied_patterns",
                verbose=10,
            ),
        )
        opt_onx, out, err = self.capture(lambda: gr.to_onnx(optimize=True))
        self.assertIn("save", out)
        assert os.path.exists(
            "test_dump_applied_patterns"
        ), "folder 'test_dump_applied_patterns' not found"
        assert os.listdir("test_dump_applied_patterns"), (
            f"No file found in 'test_dump_applied_patterns': "
            f"{os.listdir('test_dump_applied_patterns')}"
        )

    # @hide_stdout()
    def test_packed_matmul(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Shape", ["ids_weight"], ["shape"], start=0, end=2),
                    oh.make_node("Concat", ["shape", "init328"], ["new_shape"], axis=0),
                    oh.make_node("MatMul", ["ids_weight", "A"], ["A1"]),
                    oh.make_node("MatMul", ["ids_weight", "B"], ["B1"]),
                    oh.make_node("MatMul", ["ids_weight", "C"], ["C1"]),
                    oh.make_node("Reshape", ["A1", "new_shape"], ["Areshaped"]),
                    oh.make_node("Reshape", ["B1", "new_shape"], ["Breshaped"]),
                    oh.make_node("Reshape", ["C1", "new_shape"], ["Creshaped"]),
                    oh.make_node("Transpose", ["Areshaped"], ["At"], perm=[0, 2, 1, 3]),
                    oh.make_node("Transpose", ["Breshaped"], ["Bt"], perm=[0, 2, 1, 3]),
                    oh.make_node("Transpose", ["Creshaped"], ["Ct"], perm=[0, 2, 1, 3]),
                ],
                "dummy",
                [_mkv_("ids_weight", TFLOAT, ["batch", "seq", 256])],
                [
                    _mkv_("At", TFLOAT, ["batch", 32, "seq", 8]),
                    _mkv_("Bt", TFLOAT, ["batch", 32, "seq", 8]),
                    _mkv_("Ct", TFLOAT, ["batch", 32, "seq", 8]),
                ],
                [
                    onh.from_array(np.array([32, 8], dtype=np.int64), name="init328"),
                    onh.from_array(np.random.randn(256, 256).astype(np.float32), name="A"),
                    onh.from_array(np.random.randn(256, 256).astype(np.float32), name="B"),
                    onh.from_array(np.random.randn(256, 256).astype(np.float32), name="C"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )

        gr = GraphBuilder(
            model,
            infer_shapes_options=True,
            optimization_options=OptimizationOptions(
                patterns=["FunctionPackedMatMul"],
                verbose=0,
                constant_folding=True,
            ),
            verbose=0,
        )
        opt_onx = gr.to_onnx(optimize=True)
        self.assertEqual(
            ["Shape", "Concat", "PackedMatMulReshapeTranspose3_0_2_1_3"],
            [_.op_type for _ in opt_onx.graph.node],
        )

        feeds = {"ids_weight": self._range(2, 3, 256)}
        ref = ExtendedReferenceEvaluator(model)
        expected = ref.run(None, feeds)

        opt_ref = ExtendedReferenceEvaluator(opt_onx, verbose=10)
        got = opt_ref.run(None, feeds)
        self.assertEqual(len(expected), len(got))
        for a, b in zip(expected, got):
            self.assertEqualArray(a, b, atol=1e-2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
