import unittest
from experimental_experiment.torch_interpreter import to_onnx
from experimental_experiment.ext_test_case import ExtTestCase
from experimental_experiment.reference import ExtendedReferenceEvaluator


class TestOnnxExportTransformers(ExtTestCase):
    def test_aten_transformers_grouped_mm_no_offset(self):
        import torch
        import transformers.integrations.moe as tmoe

        if not hasattr(tmoe, "_grouped_mm_fallback"):
            self.skipTest(
                "transformers.integrations.moe._grouped_mm_fallback "
                "introduced in transformers>=5.3"
            )

        class Model(torch.nn.Module):
            def forward(self, a, b, offs):
                res = torch.ops.transformers.grouped_mm_fallback(a, b.transpose(-2, -1), offs)
                return res.to(torch.float32)

        M, N, K = 6, 2, 16
        a = torch.ones(M, K, device="cpu", dtype=torch.float32)
        b = torch.ones(3, N, K, device="cpu", dtype=torch.float32)
        offs = torch.tensor([1, 4, 6], dtype=torch.int32)
        model = Model()
        expected = model(a, b, offs)
        torch.export.export(model, (a, b, offs))
        onx = to_onnx(model, (a, b, offs), dynamic_shapes=({0: "G"}, {}, {}))
        self.dump_onnx("test_aten_transformers_grouped_mm_no_offset.onnx", onx)
        ref = ExtendedReferenceEvaluator(onx, verbose=0)
        got = ref.run(None, dict(a=a.numpy(), b=b.numpy(), offs=offs.numpy()))
        self.assertEqualArray(expected, got[0])
        self.assert_conversion_with_ort_on_cpu(onx, expected, (a, b, offs), atol=1e-3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
