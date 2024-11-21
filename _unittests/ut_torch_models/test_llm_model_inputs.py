import unittest
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
    skipif_ci_windows,
    hide_stdout,
)
from experimental_experiment.torch_models.phi3_helper import has_phi3


class TestLlmModelInputs(ExtTestCase):

    @unittest.skipIf(not has_phi3(), reason="transformers not recent enough")
    @skipif_ci_windows("not supported")
    @ignore_warnings("TracerWarning")
    @ignore_warnings(UserWarning)
    @hide_stdout()
    def test_generate_dummy_inputs(self):
        from experimental_experiment.torch_models.dummy_inputs import generate_dummy_inputs

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
