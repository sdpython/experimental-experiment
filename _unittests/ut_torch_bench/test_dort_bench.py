import contextlib
import io
import logging
import unittest
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    skipif_ci_windows,
    ignore_warnings,
    requires_torch,
)


class TestDortBench(ExtTestCase):
    @classmethod
    def setUpClass(cls):
        logger = logging.getLogger("onnxscript.optimizer.constant_folding")
        logger.setLevel(logging.ERROR)
        ExtTestCase.setUpClass()

    def _dort_bench_small_llama_cpu(self, backend, config="small"):
        from experimental_experiment.torch_bench.dort_bench import main

        args = [
            "--model",
            "llama",
            "--config",
            config,
            "--device",
            "cpu",
            "--backend",
            backend,
            "--verbose",
            "1",
        ]
        st = io.StringIO()
        with contextlib.redirect_stdout(st):
            main(args=args)
        out = st.getvalue()
        self.assertIn(":time,", out)
        self.assertIn(":warmup_time,", out)
        self.assertIn(":patterns,", out)
        self.assertIn(":implementation,", out)
        self.assertIn(":order,", out)

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    def test_dort_bench_small_llama_cpu_custom(self):
        self._dort_bench_small_llama_cpu("custom")

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    def test_dort_bench_small_llama_cpu_ort_plus(self):
        self._dort_bench_small_llama_cpu("ort+", config="medium")

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    def test_dort_bench_small_llama_cpu_debug(self):
        self._dort_bench_small_llama_cpu("debug")


if __name__ == "__main__":
    unittest.main(verbosity=2)
