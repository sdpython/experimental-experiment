import unittest
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
    skipif_ci_windows,
    hide_stdout,
    requires_cuda,
)
from experimental_experiment.torch_models.phi3_helper import has_phi3
from experimental_experiment.torch_models.dummy_inputs import generate_dummy_inputs
from experimental_experiment.torch_models.dummy_inputs.llm_dummy_inputs import (
    restore_dummy_inputs_for_phi_35_vision_instruct,
)
from experimental_experiment.mini_onnx_builder import create_input_tensors_from_onnx_model
from experimental_experiment.helpers import string_type


class TestLlmModelInputs(ExtTestCase):

    @unittest.skipIf(not has_phi3(), reason="transformers not recent enough")
    @skipif_ci_windows("not supported")
    @ignore_warnings("TracerWarning")
    @ignore_warnings(UserWarning)
    @requires_cuda()
    @hide_stdout()
    def test_generate_dummy_inputs(self):
        from experimental_experiment.torch_models.llm_model_helper import (
            get_phi_35_vision_instruct,
        )

        device = "cuda"

        filenames = generate_dummy_inputs(
            "microsoft/Phi-3.5-vision-instruct",
            num_hidden_layers=1,
            device=device,
            n_iterations=2,
            with_images=False,
            prefix="test_",
            verbose=1,
        )
        for f in filenames:
            self.assertExists(f)

            model, _ = get_phi_35_vision_instruct(num_hidden_layers=1)
            model = model.to(device)
            args, kwargs = create_input_tensors_from_onnx_model(
                f, device=device, engine="onnxruntime"
            )
            text = string_type((args, kwargs), with_shape=True, with_min_max=True)
            self.assertIn("input_ids", text)
            model(*args, **kwargs)

    @unittest.skipIf(not has_phi3(), reason="transformers not recent enough")
    @skipif_ci_windows("not supported")
    @ignore_warnings("TracerWarning")
    @ignore_warnings(UserWarning)
    @requires_cuda()
    @hide_stdout()
    def test_generate_dummy_inputs_with_images(self):
        from experimental_experiment.torch_models.llm_model_helper import (
            get_phi_35_vision_instruct,
        )

        device = "cuda"

        filenames = generate_dummy_inputs(
            "microsoft/Phi-3.5-vision-instruct",
            num_hidden_layers=1,
            device=device,
            n_iterations=2,
            prefix="test_",
            verbose=1,
            with_images=True,
        )
        for f in filenames:
            self.assertExists(f)

            model, _ = get_phi_35_vision_instruct(num_hidden_layers=1)
            model = model.to(device)
            args, kwargs = create_input_tensors_from_onnx_model(
                f, device=device, engine="onnxruntime"
            )
            model(*args, **kwargs)

    def test_restore_dummy_inputs(self):
        dummies = restore_dummy_inputs_for_phi_35_vision_instruct(
            num_hidden_layers=1, with_images=True
        )
        self.assertIsInstance(dummies, tuple)
        self.assertEqual(len(dummies), 2)
        self.assertIsInstance(dummies[0], tuple)
        self.assertIsInstance(dummies[1], dict)
        for k, v in dummies[1].items():
            if k == "past_key_values":
                self.assertIn("DynamicCache", str(type(v)))

    def test_restore_dummy_inputs_with_images(self):
        dummies = restore_dummy_inputs_for_phi_35_vision_instruct(
            num_hidden_layers=1, with_images=True
        )
        self.assertIsInstance(dummies, tuple)
        self.assertEqual(len(dummies), 2)
        self.assertIsInstance(dummies[0], tuple)
        self.assertIsInstance(dummies[1], dict)
        for k, v in dummies[1].items():
            if k == "past_key_values":
                self.assertIn("DynamicCache", str(type(v)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
