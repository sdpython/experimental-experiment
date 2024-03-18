import unittest
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
)
from experimental_experiment.torch_test_helper import export_to_onnx, check_model_ort


def has_cuda():
    import onnxruntime

    available_providers = [
        provider for provider in onnxruntime.get_available_providers()
    ]
    return "CUDAExecutionProvider" in available_providers


class TestEdTransformer(ExtTestCase):

    @ignore_warnings(UserWarning)
    def test_transformer_export_new(self):
        import torch

        transformer_model = torch.nn.Transformer(
            nhead=2, num_encoder_layers=2, d_model=16, dim_feedforward=32, dropout=0.1
        )
        src = torch.rand((10, 32, 16))
        tgt = torch.rand((20, 32, 16))

        expected = transformer_model(src, tgt)

        ret = export_to_onnx(
            transformer_model,
            src,
            tgt,
            rename_inputs=False,
            optimize=True,
            # prefix="test_phi_export",
        )

        onx = ret["proto"]
        names = [i.name for i in onx.graph.input]
        xp = [x.numpy() for x in [src, tgt]]
        feeds = dict(zip(names, xp))
        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, feeds)
        self.assertEqualArray(expected[0].detach().numpy(), results[0], atol=1e-5)
        if has_cuda():
            sess = check_model_ort(onx, providers="cuda")
            results = sess.run(None, feeds)
            self.assertEqualArray(expected[0].detach().numpy(), results[0], atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
