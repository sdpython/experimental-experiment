import unittest
from typing import Any, List
import numpy as np
import onnx
from experimental_experiment.ext_test_case import ExtTestCase
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
        dynamic_shapes=None,
        inline=True,
    ) -> str:
        filename = f"{test_name}_{exporter}_{'dec' if decomposition else ''}.onnx"
        export_options = ExportOptions(
            decomposition_table="all" if decomposition else None,
            strict=strict,
            aten_as_function=True,
            backed_size_oblivious=False,
        )
        to_onnx(
            model,
            inputs,
            filename=filename,
            export_options=export_options,
            verbose=verbose,
            optimize=optimize,
            dynamic_shapes=dynamic_shapes,
            inline=inline,
        )
        return filename

    def test_aten_roll_relu_static(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.relu(torch.roll(x, -1, -1))

        model = Model()
        x = (torch.arange(4 * 3) + 10).reshape((1, -1, 4)).to(torch.float32)
        expected = model(x)
        model_path = self._call_exporter(
            "test_aten_roll_relu_static", "custom", model, (x,), inline=False
        )
        onx = onnx.load(model_path)
        op_types = [n.op_type for n in onx.graph.node]
        self.assertEqual(op_types, ["aten_roll_default", "aten_relu_default", "Identity"])
        op_domains = [n.domain for n in onx.graph.node]
        self.assertEqual(op_domains, ["aten", "aten", ""])

        sess = ExtendedReferenceEvaluator(model_path)
        feeds = dict(zip(sess.input_names, [x.numpy()]))
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected.to(torch.int64), got.astype(np.int64))

    def test_aten_roll_relu_dynamic(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.relu(torch.roll(x, -1, -1))

        model = Model()
        x = (torch.arange(8 * 3) + 10).reshape((2, -1, 4)).to(torch.float32)
        expected = model(x)

        model_path = self._call_exporter(
            "test_aten_roll_relu_dynamic",
            "custom",
            model,
            (x,),
            dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=1024)}},
            inline=False,
        )
        onx = onnx.load(model_path)
        op_types = [n.op_type for n in onx.graph.node]
        self.assertEqual(op_types, ["aten_roll_default", "aten_relu_default", "Identity"])
        op_domains = [n.domain for n in onx.graph.node]
        self.assertEqual(op_domains, ["aten", "aten", ""])

        sess = ExtendedReferenceEvaluator(model_path)
        feeds = dict(zip(sess.input_names, [x.numpy()]))
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected.to(torch.int64), got.astype(np.int64))


if __name__ == "__main__":
    unittest.main(verbosity=2)
