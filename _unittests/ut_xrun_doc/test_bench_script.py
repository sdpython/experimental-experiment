import unittest
from experimental_experiment.ext_test_case import ExtTestCase
from experimental_experiment.bench_run import (
    _extract_metrics,
    _cmd_line,
    run_benchmark,
    get_machine,
)


class TestBenchScript(ExtTestCase):
    def test_reg(self):
        text = ":m,6;"
        m = _extract_metrics(text)
        self.assertEqual(m, {"m": "6"})

    def test_cmd(self):
        cmd = _cmd_line("l", m=6)
        self.assertEqual(cmd[1:], ["-m", "l", "--m", "6"])

    def test_machine(self):
        ma = get_machine()
        self.assertIn("machine", ma)
        self.assertIn("processor", ma)
        self.assertIn("cpu", ma)
        self.assertIn("has_cuda", ma)

    def test_run_script(self):
        script_name = "experimental_experiment._bench_test"
        configs = [dict(m=6)]
        res = run_benchmark(script_name, configs)
        self.assertEqual(len(res), 1)
        expected = {"metric1": "0.5", "metric2": "5", "metric3": "dummy", "m": 6}
        got = res[0]
        for k, v in expected.items():
            self.assertIn(k, got)
            self.assertEqual(v, got[k])


if __name__ == "__main__":
    unittest.main(verbosity=2)
