import unittest
from typing import Any, List
import numpy as np
from onnx.checker import check_model
from experimental_experiment.ext_test_case import ExtTestCase, skipif_ci_windows
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.torch_interpreter import to_onnx, ExportOptions


class TestOnnxExportAten(ExtTestCase):

    def _call_exporter(
        self,
        test_name: str,
        exporter: str,
        model: "torch.nn.Module",  # noqa: F821
        inputs: List[Any],
        decomposition: bool = False,
        verbose: int = 0,
        optimize: bool = False,
    ) -> str:
        import torch

        filename = f"{test_name}_{exporter}_{'dec' if decomposition else ''}.onnx"
        if exporter == "script":
            torch.onnx.export(model, inputs, filename, opset_version=18)
        elif exporter == "dynamo":
            torch.onnx.export(model, inputs, filename, dynamo=True)
        else:
            export_options = (
                ExportOptions(decomposition_table="all") if decomposition else None
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
    def test_aten_roll_neg(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.roll(x, -1, -1)

        model = Model()
        x = (torch.arange(4 * 8192) + 10).reshape((1, -1, 4)).to(torch.float32)
        expected = model(x)
        model_path = self._call_exporter("test_aten_roll_neg", "custom", model, (x,))
        check_model(model_path)

        sess = ExtendedReferenceEvaluator(model_path)
        feeds = dict(zip(sess.input_names, [x.numpy()]))
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected.to(int), got.astype(int))

    @skipif_ci_windows("not working on windows")
    def test_aten_roll_pos(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.roll(x, 1, -1)

        model = Model()
        x = (torch.arange(4 * 3) + 10).reshape((1, -1, 4)).to(torch.float32)
        expected = model(x)
        model_path = self._call_exporter("test_aten_roll_pos", "custom", model, (x,))
        check_model(model_path)

        sess = ExtendedReferenceEvaluator(model_path)
        feeds = dict(zip(sess.input_names, [x.numpy()]))
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected.to(int), got.astype(int))

    @skipif_ci_windows("not working on windows")
    def test_aten_index_put_3d_nd_case_1(self):
        import torch

        class Model(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.params = torch.zeros((1, 3, 4), dtype=torch.float32)

            def forward(self, index, update):
                copy = self.params.clone()
                copy[..., index] = update
                return copy

        model = Model()
        update = (torch.arange(4 * 8192) + 10).reshape((-1, 4)).to(torch.float32)
        index = torch.from_numpy(np.array([0, 3, 2, 1])).to(torch.int64)
        expected = model(index, update)
        model_path = self._call_exporter(
            "test_aten_index_put_3d_nd_case_1", "custom", model, (index, update)
        )
        check_model(model_path)

        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        sess = onnxruntime.InferenceSession(
            model_path, sess_options=sess_options, providers=[("CPUExecutionProvider")]
        )
        feeds = dict(zip([i.name for i in sess.get_inputs()], [index.numpy(), update.numpy()]))
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    @skipif_ci_windows("not working on windows")
    def test_aten_index_put_3d_nd_case_2(self):
        import torch

        class Model(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.params = torch.zeros((1, 8192, 6), dtype=torch.float32)

            def forward(self, index, update):
                copy = self.params.clone()
                copy[..., index] = update
                return copy

        model = Model()
        update = (torch.arange(4 * 8192) + 10).reshape((-1, 4)).to(torch.float32)
        index = torch.from_numpy(np.array([0, 3, 2, 5])).to(torch.int64)
        expected = model(index, update)
        model_path = self._call_exporter(
            "test_aten_index_put_3d_nd_case_2", "custom", model, (index, update)
        )
        check_model(model_path)

        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        sess = onnxruntime.InferenceSession(
            model_path, sess_options=sess_options, providers=[("CPUExecutionProvider")]
        )
        feeds = dict(zip([i.name for i in sess.get_inputs()], [index.numpy(), update.numpy()]))
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got)


if __name__ == "__main__":
    unittest.main(verbosity=2)
