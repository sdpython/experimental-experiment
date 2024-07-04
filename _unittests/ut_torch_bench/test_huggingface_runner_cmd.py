import contextlib
import io
import logging
import unittest
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    skipif_ci_windows,
    ignore_warnings,
    requires_torch,
    requires_onnxruntime_training,
)


class TestHuggingFaceRunnerCmd(ExtTestCase):
    @classmethod
    def setUpClass(cls):
        logger = logging.getLogger("onnxscript.optimizer.constant_folding")
        logger.setLevel(logging.ERROR)
        ExtTestCase.setUpClass()

    def _huggingface_export_bench_cpu(self, exporter, models):
        from experimental_experiment.torch_bench.bash_bench_huggingface import main

        args = [
            "--model",
            models,
            "--device",
            "cpu",
            "--exporter",
            exporter,
            "--verbose",
            "0",
        ]
        st = io.StringIO()
        with contextlib.redirect_stdout(st):
            main(args=args)
        out = st.getvalue()
        if "," in models:
            self.assertIn("Prints", out)
        else:
            self.assertIn(":model_name,", out)

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_onnxruntime_training()
    @requires_torch("2.4")
    def test_huggingface_export_bench_custom_cpu(self):
        self._huggingface_export_bench_cpu("custom", "dummy")

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    def test_huggingface_export_bench_export_cpu(self):
        self._huggingface_export_bench_cpu("export", "dummy")

    @ignore_warnings((DeprecationWarning, UserWarning))
    def test_huggingface_export_bench_eager_cpu(self):
        self._huggingface_export_bench_cpu("eager", "dummy")

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_onnxruntime_training()
    @requires_torch("2.4")
    def test_huggingface_export_bench_custom_cpu2(self):
        self._huggingface_export_bench_cpu("custom", "dummy,dummy16")

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_onnxruntime_training()
    @requires_torch("2.4")
    def test_huggingface_export_bench_custom_cpu_last(self):
        self._huggingface_export_bench_cpu("custom", "-1")

    @ignore_warnings((DeprecationWarning, UserWarning))
    def test_huggingface_export_bench_script_cpu(self):
        self._huggingface_export_bench_cpu("script", "dummy")

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.5")
    def test_huggingface_export_bench_dynamo_cpu(self):
        self._huggingface_export_bench_cpu("dynamo", "dummy")

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    def test_huggingface_export_bench_dynamo2_cpu(self):
        self._huggingface_export_bench_cpu("dynamo2", "dummy")


if __name__ == "__main__":
    unittest.main(verbosity=2)
