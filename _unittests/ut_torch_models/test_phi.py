import onnxruntime  # noqa: F401
import unittest
from experimental_experiment.ext_test_case import ExtTestCase
from experimental_experiment.reference import ExtendedReferenceEvaluator


class TestPhi(ExtTestCase):

    def test_get_phi_model_export(self):
        import torch
        from experimental_experiment.torch_models.phi_helper import (
            get_phi_model,
        )

        model, model_inputs = get_phi_model()
        expected = model(*model_inputs[0])
        filename = self.get_dump_file("test_get_phi_model_export.onnx")
        torch.onnx.export(
            model, model_inputs[0], filename, input_names=["input0", "input1"]
        )
        ref = ExtendedReferenceEvaluator(filename)
        feeds = dict(
            zip(["input0", "input1"], [t.detach().numpy() for t in model_inputs[0]])
        )
        got = ref.run(None, feeds)
        self.assertEqualArray(expected[0], got[0], atol=1e-5)

    def test_get_phi_model_mask_eager(self):
        from experimental_experiment.torch_models.phi_helper import (
            get_phi_model,
        )

        model, model_inputs = get_phi_model(
            _attn_implementation="eager", with_mask=True
        )
        self.assertEqual(len(model_inputs[0]), 2)
        expected = model(*model_inputs[0])
        self.assertNotEmpty(expected)

    def test_get_phi_model_nomask_eager(self):
        from experimental_experiment.torch_models.phi_helper import (
            get_phi_model,
        )

        model, model_inputs = get_phi_model(
            _attn_implementation="eager", with_mask=False
        )
        self.assertEqual(len(model_inputs[0]), 1)
        expected = model(*model_inputs[0])
        self.assertNotEmpty(expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
