import unittest
from experimental_experiment.ext_test_case import ExtTestCase, requires_torch
from experimental_experiment.torch_interpreter.eval import discover, evaluation


class TestEval(ExtTestCase):
    @requires_torch("2.6", "scan")
    def test_discover(self):
        res = discover()
        self.assertNotEmpty(res)
        for mod in res.values():
            m = mod()
            if isinstance(m._inputs, tuple):
                m(*m._inputs)
            else:
                m(*m._inputs[0])

    @requires_torch("2.6", "ONNXProgram.optimize missing")
    def test_eval(self):
        d = list(discover().items())[0]  # noqa: RUF015
        ev = evaluation(
            quiet=False,
            cases={d[0]: d[1]},
            exporters=(
                "export-strict",
                "export-nostrict",
                "export-tracing",
                "custom-strict",
                "custom-nostrict",
                "custom-strict-dec",
                "custom-nostrict-dec",
                "dynamo",
                "dynamo-ir",
            ),
        )
        self.assertIsInstance(ev, list)
        self.assertIsInstance(ev[0], dict)

    @requires_torch("2.6", "ONNXProgram.optimize missing")
    def test_run_exporter(self):
        evaluation(
            cases="SignatureListFixedLength",
            exporters="custom-strict",
            quiet=False,
            dynamic=False,
        )

    @requires_torch("2.6", "ONNXProgram.optimize missing")
    def test_run_exporter_regex(self):
        evaluation(cases=".*Aten.*", exporters="custom-strict", quiet=False, dynamic=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
