import unittest
from experimental_experiment.torch_bench._bash_bench_model_runner import ModelRunner
from experimental_experiment.torch_bench._bash_bench_set_huggingface import (
    HuggingfaceRunner,
)
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    hide_stdout,
    # ignore_warnings,
    requires_torch,
    skipif_ci_windows,
)


class TestBashBenchRunner(ExtTestCase):
    @requires_torch("2.3")
    def test_create_runner_cpu(self):
        runner = HuggingfaceRunner(device="cpu")
        names = runner.get_model_name_list()
        self.assertIn("BartForCausalLM", names)

    @requires_torch("2.3")
    def test_create_runner_cuda(self):
        runner = HuggingfaceRunner(device="cuda")
        names = runner.get_model_name_list()
        self.assertIn("BartForCausalLM", names)

    @skipif_ci_windows("not useful")
    @requires_torch("2.3")
    @hide_stdout()
    def test_load_model(self):
        runner = HuggingfaceRunner(
            device="cpu", include_model_names={"MobileBertForMaskedLM"}, verbose=3
        )
        res = list(runner.enumerate_load_models())
        self.assertIsInstance(res, list)
        self.assertEqual(len(res), 1)
        self.assertIsInstance(res[0], ModelRunner)

    @skipif_ci_windows("not useful")
    @requires_torch("2.3")
    @hide_stdout()
    def test_run_model(self):
        runner = HuggingfaceRunner(device="cpu", include_model_names={"101Dummy"}, verbose=2)
        data = list(runner.enumerate_run_models())
        self.assertEqual(len(data), 1)

    @skipif_ci_windows("not useful")
    @requires_torch("2.5")
    @hide_stdout()
    def test_test_model_32(self):
        runner = HuggingfaceRunner(device="cpu", include_model_names={"101Dummy"}, verbose=2)
        data = list(runner.enumerate_test_models(process=False, exporter="custom"))
        # print(data)
        self.assertEqual(1, len(data))

    @skipif_ci_windows("not useful")
    @requires_torch("2.5")
    @hide_stdout()
    def test_test_model_16(self):
        runner = HuggingfaceRunner(
            device="cpu", include_model_names={"101Dummy16"}, verbose=2
        )
        data = list(runner.enumerate_test_models(process=False, exporter="custom"))
        print(data)
        self.assertEqual(len(data), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
