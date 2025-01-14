import unittest
from experimental_experiment.helpers import max_diff
from experimental_experiment.bench_run import (
    BenchmarkError,
    _cmd_line,
    _extract_metrics,
    get_machine,
    make_configs,
    run_benchmark,
)
from experimental_experiment.ext_test_case import ExtTestCase, hide_stdout


class TestBenchRun(ExtTestCase):
    def test_reg(self):
        text = ":m,6;"
        m = _extract_metrics(text)
        self.assertEqual(m, {"m": 6})

    def test_cmd(self):
        cmd = _cmd_line("l", m=6)
        self.assertEqual(cmd[1:], ["-m", "l", "--m", "6"])

    def test_machine(self):
        ma = get_machine()
        self.assertIn("machine", ma)
        self.assertIn("processor", ma)
        self.assertIn("cpu", ma)
        self.assertIn("has_cuda", ma)
        self.assertIn("processor_name", ma)

    def test_run_script(self):
        script_name = "experimental_experiment._bench_test"
        configs = [dict(m=6)]
        try:
            res = run_benchmark(script_name, configs)
        except BenchmarkError as e:
            raise unittest.SkipTest(f"Probably no metric collected due to {e}")  # noqa: B904
        self.assertEqual(len(res), 1)
        expected = {"metric1": 0.5, "metric2": 5, "metric3": "dummy", "m": 6}
        got = res[0]
        for k, v in expected.items():
            self.assertIn(k, got)
            self.assertEqual(v, got[k])

    def test_make_configs(self):
        kwargs = {"single": "1", "multi2": "1,2", "multi3": "A,B,C"}
        confs = make_configs(kwargs)
        self.assertEqual(
            confs,
            [
                {"single": "1", "multi2": "1", "multi3": "A"},
                {"single": "1", "multi2": "1", "multi3": "B"},
                {"single": "1", "multi2": "1", "multi3": "C"},
                {"single": "1", "multi2": "2", "multi3": "A"},
                {"single": "1", "multi2": "2", "multi3": "B"},
                {"single": "1", "multi2": "2", "multi3": "C"},
            ],
        )

    def test_make_configs_last(self):
        kwargs = {"single": "1", "multi2": "1,2", "multi3": "5, 6"}
        confs = make_configs(kwargs, last=["multi2"])
        self.assertEqual(
            confs,
            [
                {"single": "1", "multi3": "5", "multi2": "1"},
                {"single": "1", "multi3": "5", "multi2": "2"},
                {"single": "1", "multi3": " 6", "multi2": "1"},
                {"single": "1", "multi3": " 6", "multi2": "2"},
            ],
        )

    def test_make_configs_filter(self):
        def filter_out(kwargs):
            if kwargs["multi2"] == "1" and kwargs["multi3"] == "A":
                return True
            return False

        kwargs = {"single": "1", "multi2": "1,2", "multi3": "A,B,C"}
        confs = make_configs(kwargs, filter_function=filter_out)
        self.assertEqual(confs, [{"single": "1", "multi2": "1", "multi3": "A"}])

    def test_make_configs_drop(self):
        kwargs = {"single": "1", "multi2": "1,2", "multi3": "A,B,C"}
        confs = make_configs(kwargs, drop=["multi3"])
        self.assertEqual(
            confs,
            [{"single": "1", "multi2": "1"}, {"single": "1", "multi2": "2"}],
        )

    def test_make_configs_replace(self):
        kwargs = {"single": "1", "multi2": "1,2", "multi3": "A,B,C"}
        confs = make_configs(kwargs, replace={"multi3": "ZZZ"})
        self.assertEqual(
            confs,
            [
                {"single": "1", "multi2": "1", "multi3": "ZZZ"},
                {"single": "1", "multi2": "2", "multi3": "ZZZ"},
            ],
        )

    def test_max_diff(self):
        import torch

        self.assertEqual(
            max_diff(torch.Tensor([1, 2]), torch.Tensor([1, 2])),
            {"abs": 0.0, "rel": 0.0, "sum": 0.0, "n": 2.0, "dnan": 0.0},
        )
        self.assertEqual(
            max_diff(
                (torch.Tensor([1, 2]),),
                (torch.Tensor([1, 2])),
            ),
            {"abs": 0.0, "rel": 0.0, "sum": 0.0, "n": 2.0, "dnan": 0.0},
        )
        self.assertEqual(
            max_diff(
                (torch.Tensor([1, 2]), (torch.Tensor([1, 2]),)),
                (torch.Tensor([1, 2]), (torch.Tensor([1, 2]),)),
            ),
            {"abs": 0.0, "rel": 0.0, "sum": 0.0, "n": 4.0, "dnan": 0.0},
        )
        self.assertEqual(
            max_diff(
                {"a": torch.Tensor([1, 2])},
                {"a": torch.Tensor([1, 2])},
            ),
            {"abs": 0.0, "rel": 0.0, "sum": 0.0, "n": 2.0, "dnan": 0.0},
        )
        self.assertEqual(
            max_diff(
                {"a": torch.Tensor([1, 2])},
                [torch.Tensor([1, 2])],
            ),
            {"abs": 0.0, "rel": 0.0, "sum": 0.0, "n": 2.0, "dnan": 0.0},
        )
        self.assertEqual(
            max_diff(
                {"a": torch.Tensor([1, float("nan")])},
                [torch.Tensor([1, 2])],
            ),
            {
                "abs": 9999999998.0,
                "dnan": 1.0,
                "n": 2.0,
                "rel": 0.9999999997999001,
                "sum": 9999999998.0,
            },
        )

    @hide_stdout()
    def test_max_diff_dynamic_cache(self):
        import torch
        import transformers

        t1 = torch.tensor([0, 1], dtype=torch.float32)
        cache = transformers.cache_utils.DynamicCache(1)
        cache.update(torch.ones((2, 2)), (torch.ones((2, 2)) * 2), 0)
        md = max_diff(
            (t1, cache),
            (t1, cache.key_cache[0], cache.value_cache[0]),
            flatten=True,
            verbose=10,
        )
        self.assertEqual(md, {"abs": 0.0, "rel": 0.0, "sum": 0.0, "n": 10.0, "dnan": 0})


if __name__ == "__main__":
    unittest.main(verbosity=2)
