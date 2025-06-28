import unittest
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    requires_torch,
    ignore_warnings,
    hide_stdout,
    requires_onnxscript,
)
from experimental_experiment.reference import ExtendedReferenceEvaluator


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

    def test_of_scaled_dot_product_attention_23(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, query, key, value):
                return torch.nn.functional.scaled_dot_product_attention(query, key, value)

        query = torch.rand(32, 8, 128, 64, dtype=torch.float16)
        key = torch.rand(32, 8, 128, 64, dtype=torch.float16)
        value = torch.rand(32, 8, 128, 64, dtype=torch.float16)
        inputs = (query, key, value)
        model = Model()
        expected = model(*inputs)
        ds1 = {0: "batch", 2: "cache_length", 3: "last_dim"}
        ds = (ds1, ds1, ds1)
        opset = 23
        onx = torch.onnx.export(
            model, inputs, dynamic_shapes=ds, opset_version=opset, dynamo=True
        ).model_proto
        self.dump_onnx(f"test_of_scaled_dot_product_attention_{opset}.onnx", onx)
        self.assertEqual(["Attention"], [n.op_type for n in onx.graph.node])
        self.assertEqual(
            ("", opset), (onx.opset_import[0].domain, onx.opset_import[0].version)
        )

        feeds = dict(
            zip(["query", "key", "value"], [x.detach().cpu().numpy() for x in inputs])
        )
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-2)

        import onnxruntime

        sess = onnxruntime.InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
