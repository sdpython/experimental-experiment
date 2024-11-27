import contextlib
import io
import logging
import unittest
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    skipif_ci_apple,
    skipif_ci_windows,
    ignore_warnings,
    requires_torch,
    requires_onnxruntime_training,
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
            "--memory_spy",
            "0",
            # "--disable_pattern", "default"
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
    @skipif_ci_apple(
        "AttributeError: 'onnxruntime.capi.onnxruntime_pybind11_state.OrtVal'"
        "object has no attribute 'push_back_batch'"
    )
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_onnxruntime_training()
    def test_dort_bench_small_llama_cpu_custom(self):
        self._dort_bench_small_llama_cpu("custom")

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    @unittest.skip("broken")
    def test_dort_bench_small_llama_cpu_ort_plus(self):
        self._dort_bench_small_llama_cpu("ort+", config="medium")

    @skipif_ci_windows("exporter does not work on Windows")
    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.7", "convert_element_type_default")
    def test_dort_bench_small_llama_cpu_debug(self):
        self._dort_bench_small_llama_cpu("debug")


if __name__ == "__main__":
    unittest.main(verbosity=2)
