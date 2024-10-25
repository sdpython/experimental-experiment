import unittest
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    requires_onnxruntime_training,
    requires_cuda,
    ignore_warnings,
)
from experimental_experiment.reference import ExtendedReferenceEvaluator


class TestPhi(ExtTestCase):
    @ignore_warnings("TracerWarning")
    def test_get_phi_model_export(self):
        import torch
        from experimental_experiment.torch_models.phi_helper import (
            get_phi_model,
        )

        model, model_inputs = get_phi_model()
        expected = model(*model_inputs[0])
        filename = self.get_dump_file("test_get_phi_model_export.onnx")
        torch.onnx.export(model, model_inputs[0], filename, input_names=["input0", "input1"])
        ref = ExtendedReferenceEvaluator(filename)
        feeds = dict(zip(["input0", "input1"], [t.detach().numpy() for t in model_inputs[0]]))
        got = ref.run(None, feeds)
        self.assertEqualArray(expected[0], got[0], atol=1e-5)

    def test_get_phi_model_mask_eager(self):
        from experimental_experiment.torch_models.phi_helper import (
            get_phi_model,
        )

        model, model_inputs = get_phi_model(_attn_implementation="eager", with_mask=True)
        self.assertEqual(len(model_inputs[0]), 2)
        expected = model(*model_inputs[0])
        self.assertNotEmpty(expected)

    def test_get_phi_model_mask_eager_backward(self):
        from experimental_experiment.torch_models.phi_helper import (
            get_phi_model,
        )

        model, model_inputs = get_phi_model(_attn_implementation="eager", with_mask=True)
        self.assertEqual(len(model_inputs[0]), 2)
        expected = model(*model_inputs[0])
        self.assertNotEmpty(expected)
        back = expected[0].sum().backward()
        self.assertEmpty(back)

    @requires_onnxruntime_training()
    def test_get_phi_model_mask_eager_ortmodule(self):
        import onnxruntime
        from onnxruntime.training.ortmodule import ORTModule
        from experimental_experiment.torch_models.phi_helper import (
            get_phi_model,
        )

        model, model_inputs = get_phi_model(_attn_implementation="eager", with_mask=True)
        self.assertEqual(len(model_inputs[0]), 2)
        try:
            omodel = ORTModule(model)
        except onnxruntime.training.ortmodule._fallback_exceptions.ORTModuleInitException:
            raise unittest.SkipTest("ORTModule extensions are not installed.")  # noqa: B904
        expected = omodel(*model_inputs[0])
        self.assertNotEmpty(expected)

    @requires_onnxruntime_training()
    def test_get_phi_model_mask_eager_ortmodule_backward(self):
        import onnxruntime
        from onnxruntime.training.ortmodule import ORTModule, DebugOptions
        from experimental_experiment.torch_models.phi_helper import (
            get_phi_model,
        )

        opts = DebugOptions(
            save_onnx=True,
            onnx_prefix="test_get_phi_model_mask_eager_ortmodule_backward",
        )

        model, model_inputs = get_phi_model(_attn_implementation="eager", with_mask=True)
        self.assertEqual(len(model_inputs[0]), 2)
        try:
            omodel = ORTModule(model, opts)
        except onnxruntime.training.ortmodule._fallback_exceptions.ORTModuleInitException:
            raise unittest.SkipTest("ORTModule extensions are not installed.")  # noqa: B904
        expected = omodel(*model_inputs[0])
        self.assertNotEmpty(expected)
        back = expected[0].sum().backward()
        self.assertEmpty(back)

    @requires_onnxruntime_training()
    @requires_cuda()
    def test_get_phi_model_mask_eager_ortmodule_backward_cuda(self):
        from onnxruntime.training.ortmodule import ORTModule, DebugOptions
        from experimental_experiment.torch_models.phi_helper import (
            get_phi_model,
        )

        opts = DebugOptions(
            save_onnx=True,
            onnx_prefix="test_get_phi_model_mask_eager_ortmodule_backward",
        )

        model, model_inputs = get_phi_model(_attn_implementation="eager", with_mask=True)
        model = model.to("cuda")
        model_inputs = [[t.to("cuda") for t in ts] for ts in model_inputs]
        self.assertEqual(len(model_inputs[0]), 2)
        omodel = ORTModule(model, opts)
        expected = omodel(*model_inputs[0])
        self.assertNotEmpty(expected)
        back = expected[0].sum().backward()
        self.assertEmpty(back)

    def test_get_phi_model_nomask_eager(self):
        from experimental_experiment.torch_models.phi_helper import (
            get_phi_model,
        )

        model, model_inputs = get_phi_model(_attn_implementation="eager", with_mask=False)
        self.assertEqual(len(model_inputs[0]), 1)
        expected = model(*model_inputs[0])
        self.assertNotEmpty(expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
