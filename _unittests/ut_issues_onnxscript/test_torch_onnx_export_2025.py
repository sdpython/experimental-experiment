import unittest
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    requires_torch,
    ignore_warnings,
    hide_stdout,
    requires_onnxscript,
)


class TestTorchOnnxExport2025(ExtTestCase):

    @ignore_warnings((UserWarning, DeprecationWarning, FutureWarning))
    @requires_torch("2.5")
    @hide_stdout()
    @requires_onnxscript("0.5")
    def test_fft_c2r(self):
        import torch
        from onnxruntime import InferenceSession

        class Model(torch.nn.Module):
            def forward(self, real, img, bias):
                cpl = torch.ops.aten.complex.default(real, img)
                fft = torch.ops.aten._fft_c2r.default(cpl, [2, 3], 1, 80)
                return fft + bias

        # static
        model = Model()
        x, y, bias = (
            torch.randn(1, 192, 80, 41, dtype=torch.float32),
            torch.randn(1, 192, 80, 41, dtype=torch.float32),
            torch.randn(1, 192, 80, 80, dtype=torch.float32),
        )
        expected = model(x, y, bias)

        ep = torch.export.export(model, (x, y, bias))
        assert ep
        onx = torch.onnx.export(model, (x, y, bias), dynamo=True).model_proto
        self.dump_onnx("test_fft_c2r.onnx", onx)

        feeds = {
            "real": x.detach().cpu().numpy(),
            "img": y.detach().cpu().numpy(),
            "bias": bias.detach().cpu().numpy(),
        }
        ref = InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0], atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
