import unittest
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    requires_onnxruntime_training,
    ignore_warnings,
)
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.torch_models.phi3_helper import has_phi3


class TestPhi3(ExtTestCase):
    @unittest.skipIf(not has_phi3(), reason="transformers not recent enough")
    @ignore_warnings("TracerWarning")
    def test_get_phi3_model_export(self):
        import torch
        from experimental_experiment.torch_models.phi3_helper import (
            get_phi3_model,
        )

        model, model_inputs = get_phi3_model()
        expected = model(*model_inputs[0])
        filename = self.get_dump_file("test_get_phi3_model_export.onnx")
        torch.onnx.export(model, model_inputs[0], filename, input_names=["input0", "input1"])
        ref = ExtendedReferenceEvaluator(filename)
        feeds = dict(zip(["input0", "input1"], [t.detach().numpy() for t in model_inputs[0]]))
        got = ref.run(None, feeds)
        self.assertEqualArray(expected[0], got[0], atol=1e-5)

    @unittest.skipIf(not has_phi3(), reason="transformers not recent enough")
    def test_get_phi3_model_mask_eager(self):
        from experimental_experiment.torch_models.phi3_helper import (
            get_phi3_model,
        )

        model, model_inputs = get_phi3_model(_attn_implementation="eager", with_mask=True)
        self.assertEqual(len(model_inputs[0]), 2)
        expected = model(*model_inputs[0])
        self.assertNotEmpty(expected)

    @unittest.skipIf(not has_phi3(), reason="transformers not recent enough")
    def test_get_phi3_model_mask_eager_backward(self):
        from experimental_experiment.torch_models.phi3_helper import (
            get_phi3_model,
        )

        model, model_inputs = get_phi3_model(_attn_implementation="eager", with_mask=True)
        self.assertEqual(len(model_inputs[0]), 2)
        expected = model(*model_inputs[0])
        self.assertNotEmpty(expected)
        back = expected[0].sum().backward()
        self.assertEmpty(back)

    @unittest.skipIf(not has_phi3(), reason="transformers not recent enough")
    @requires_onnxruntime_training()
    def test_get_phi3_model_mask_eager_ortmodule(self):
        import onnxruntime
        from onnxruntime.training.ortmodule import ORTModule
        from experimental_experiment.torch_models.phi3_helper import (
            get_phi3_model,
        )

        model, model_inputs = get_phi3_model(_attn_implementation="eager", with_mask=True)
        self.assertEqual(len(model_inputs[0]), 2)
        try:
            omodel = ORTModule(model)
        except onnxruntime.training.ortmodule._fallback_exceptions.ORTModuleInitException:
            raise unittest.SkipTest("ORTModule extensions are not installed.")  # noqa: B904
        expected = omodel(*model_inputs[0])
        self.assertNotEmpty(expected)

    @unittest.skipIf(not has_phi3(), reason="transformers not recent enough")
    def test_get_phi3_model_nomask_eager(self):
        from experimental_experiment.torch_models.phi3_helper import (
            get_phi3_model,
        )

        model, model_inputs = get_phi3_model(_attn_implementation="eager", with_mask=False)
        self.assertEqual(len(model_inputs[0]), 1)
        expected = model(*model_inputs[0])
        self.assertNotEmpty(expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
