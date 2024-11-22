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


class TestLlmModelInputs(ExtTestCase):

    @unittest.skipIf(not has_phi3(), reason="transformers not recent enough")
    @skipif_ci_windows("not supported")
    @ignore_warnings("TracerWarning")
    @ignore_warnings(UserWarning)
    @requires_cuda()
    @hide_stdout()
    def test_generate_dummy_inputs(self):

        filenames = generate_dummy_inputs(
            "microsoft/Phi-3.5-vision-instruct",
            num_hidden_layers=1,
            device="cuda",
            n_iterations=2,
            prefix="test_",
            verbose=1,
        )
        for f in filenames:
            self.assertExists(f)

    def test_restore_dummy_inputs(self):
        dummies = restore_dummy_inputs_for_phi_35_vision_instruct(num_hidden_layers=1)
        self.assertIsInstance(dummies, tuple)
        self.assertEqual(len(dummies), 2)
        self.assertIsInstance(dummies[0], tuple)
        self.assertIsInstance(dummies[1], dict)
        for k, v in dummies[1].items():
            if k == "past_key_values":
                self.assertIn("DynamicCache", str(type(v)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
