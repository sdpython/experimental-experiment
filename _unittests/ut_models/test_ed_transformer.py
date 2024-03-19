import unittest
import numpy as np
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
    skipif_ci_windows,
)
from experimental_experiment.torch_test_helper import export_to_onnx, check_model_ort


def has_cuda():
    import onnxruntime

    available_providers = [
        provider for provider in onnxruntime.get_available_providers()
    ]
    return "CUDAExecutionProvider" in available_providers


class TestEdTransformer(ExtTestCase):

    @skipif_ci_windows("dynamo exporter not on windows")
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
            rename_inputs=True,
            optimize=True,
            prefix="test_transformer_export",
            # failing due to TypeError: scaled_dot_product_attention(): argument 'is_causal' (position 6) must be bool, not Tensor
            torch_script=False,
        )

        onx = ret["proto"]
        names = [i.name for i in onx.graph.input]
        xp = [x.numpy() for x in [src, tgt]]
        feeds = dict(zip(names, xp))
        ref = ExtendedReferenceEvaluator(onx, verbose=1)
        results = ref.run(None, feeds)
        # DropOut makes it difficult to compare to.
        self.assertEqualArray(expected.detach().numpy(), results[0], atol=5)
        diff1 = np.abs(expected.detach().numpy() - results[0]).sum()
        self.assertGreater(diff1, 1000)
        if has_cuda():
            sess = check_model_ort(onx, providers="cuda")
            results = sess.run(None, feeds)
            self.assertEqualArray(expected.detach().numpy(), results[0], atol=5)
            diff2 = np.abs(expected.detach().numpy() - results[0]).sum()
            self.assertGreater(diff2, 1000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
