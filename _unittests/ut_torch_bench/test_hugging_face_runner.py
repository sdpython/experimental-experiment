import unittest
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    skipif_ci_windows,
    # ignore_warnings,
    requires_torch,
    # requires_onnxruntime_training,
)
from experimental_experiment.torch_bench.bash_bench_huggingface import (
    create_huggingface_runner,
)


class TestHuggingFaceRunner(ExtTestCase):

    @requires_torch("2.3")
    def test_create_runner_cpu(self):
        runner = create_huggingface_runner(device="cpu")
        names = runner.get_model_name_list()
        self.assertIn("BartForCausalLM", names)

    @requires_torch("2.3")
    def test_create_runner_cuda(self):
        runner = create_huggingface_runner(device="cuda")
        names = runner.get_model_name_list()
        self.assertIn("BartForCausalLM", names)

    @requires_torch("2.3")
    @skipif_ci_windows("not useful")
    def test_load_model(self):
        runner = create_huggingface_runner(
            device="cpu", include_model_names={"MobileBertForMaskedLM"}
        )
        res = list(runner.iter_models())
        self.assertEqual(len(res), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
