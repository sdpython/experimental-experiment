import unittest
from typing import Any, List
import onnx
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    skipif_ci_windows,
)
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.torch_interpreter import to_onnx, ExportOptions


class TestOnnxExportAtenAsFunction(ExtTestCase):

    def _call_exporter(
        self,
        test_name: str,
        exporter: str,
        model: "torch.nn.Module",  # noqa: F821
        inputs: List[Any],
        decomposition: bool = False,
        verbose: int = 0,
        optimize: bool = False,
        strict: bool = False,
    ) -> str:
        filename = f"{test_name}_{exporter}_{'dec' if decomposition else ''}.onnx"
        export_options = ExportOptions(
            decomposition_table="all" if decomposition else None,
            strict=strict,
            aten_as_function=True,
        )
        to_onnx(
            model,
            inputs,
            filename=filename,
            export_options=export_options,
            verbose=verbose,
            optimize=optimize,
        )
        return filename

    @skipif_ci_windows("not working on windows")
    def test_aten_roll_relu(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.relu(torch.roll(x, -1, -1))

        model = Model()
        x = (torch.arange(4 * 3) + 10).reshape((1, -1, 4)).to(torch.float32)
        expected = model(x)
        model_path = self._call_exporter("test_aten_roll_relu", "custom", model, (x,))
        onx = onnx.load(model_path)
        op_types = [n.op_type for n in onx.graph.node]
        self.assertEqual(op_types, ["aten_roll", "aten_relu"])
        op_domains = [n.domain for n in onx.graph.node]
        self.assertEqual(op_domains, ["aten", "aten"])

        sess = ExtendedReferenceEvaluator(model_path)
        feeds = dict(zip(sess.input_names, [x.numpy()]))
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected.to(int), got.astype(int))


if __name__ == "__main__":
    unittest.main(verbosity=2)
