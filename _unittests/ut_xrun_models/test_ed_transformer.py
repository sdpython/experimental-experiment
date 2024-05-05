import unittest
import numpy as np
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
    skipif_ci_windows,
    has_cuda,
)
from experimental_experiment.torch_test_helper import export_to_onnx, check_model_ort


class TestEdTransformer(ExtTestCase):

    def _get_model(self):
        if hasattr(self, "model"):
            return self.model, self.expected, self.src, self.tgt

        import torch

        transformer_model = torch.nn.Transformer(
            nhead=2, num_encoder_layers=2, d_model=16, dim_feedforward=32, dropout=0.1
        )
        src = torch.rand((10, 32, 16))
        tgt = torch.rand((20, 32, 16))

        expected = transformer_model(src, tgt)
        self.model, self.expected, self.src, self.tgt = (
            transformer_model,
            expected,
            src,
            tgt,
        )
        return transformer_model, expected, src, tgt

    @skipif_ci_windows("dynamo exporter not on windows")
    @ignore_warnings(UserWarning)
    def test_transformer_export_new_all(self):
        transformer_model, expected, src, tgt = self._get_model()

        ret = export_to_onnx(
            transformer_model,
            src,
            tgt,
            rename_inputs=True,
            optimize=True,
            prefix="test_transformer_export_all",
            # failing due to TypeError: scaled_dot_product_attention():
            # argument 'is_causal' (position 6) must be bool, not Tensor
            torch_script=False,
            verbose=0,
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

    @skipif_ci_windows("dynamo exporter not on windows")
    @ignore_warnings(UserWarning)
    def test_transformer_export_new_none(self):
        transformer_model, expected, src, tgt = self._get_model()
        ret = export_to_onnx(
            transformer_model,
            src,
            tgt,
            rename_inputs=True,
            optimize=False,
            prefix="test_transformer_export_none",
            # failing due to TypeError: scaled_dot_product_attention():
            # argument 'is_causal' (position 6) must be bool, not Tensor
            torch_script=False,
            verbose=0,
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

    @skipif_ci_windows("dynamo exporter not on windows")
    @ignore_warnings(UserWarning)
    def test_transformer_export_new_2of3_only(self):
        transformer_model, expected, src, tgt = self._get_model()
        ret = export_to_onnx(
            transformer_model,
            src,
            tgt,
            rename_inputs=True,
            optimize="MatMulReshape2Of3",
            prefix="test_transformer_export_2of3_only",
            # failing due to TypeError: scaled_dot_product_attention():
            # argument 'is_causal' (position 6) must be bool, not Tensor
            torch_script=False,
            verbose=0,
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

    @skipif_ci_windows("dynamo exporter not on windows")
    @ignore_warnings(UserWarning)
    def test_transformer_export_new_2of3_expand(self):
        transformer_model, expected, src, tgt = self._get_model()
        ret = export_to_onnx(
            transformer_model,
            src,
            tgt,
            rename_inputs=True,
            optimize="MatMulReshape2Of3,Expand,ReshapeReshape",
            prefix="test_transformer_export_2of3_expand",
            # failing due to TypeError: scaled_dot_product_attention():
            # argument 'is_causal' (position 6) must be bool, not Tensor
            torch_script=False,
            verbose=0,
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

    @skipif_ci_windows("dynamo exporter not on windows")
    @ignore_warnings(UserWarning)
    def test_transformer_export_new_default(self):
        transformer_model, expected, src, tgt = self._get_model()
        ret = export_to_onnx(
            transformer_model,
            src,
            tgt,
            rename_inputs=True,
            optimize="default",
            prefix="test_transformer_export_default",
            # failing due to TypeError: scaled_dot_product_attention():
            # argument 'is_causal' (position 6) must be bool, not Tensor
            torch_script=False,
            verbose=0,
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
