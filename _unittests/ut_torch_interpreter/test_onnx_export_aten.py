import unittest
from typing import Any, List
import numpy as np
import onnx.helper as oh
from onnx.checker import check_model
from experimental_experiment.ext_test_case import ExtTestCase, skipif_ci_windows
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.torch_interpreter import to_onnx, ExportOptions
from experimental_experiment.xbuilder._dtype_helper import torch_dtype_to_onnx_dtype


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
        strict: bool = False,
    ) -> str:
        import torch

        filename = f"{test_name}_{exporter}_{'dec' if decomposition else ''}.onnx"
        if exporter == "script":
            torch.onnx.export(model, inputs, filename, opset_version=18)
        elif exporter == "dynamo":
            torch.onnx.export(model, inputs, filename, dynamo=True)
        else:
            export_options = ExportOptions(
                decomposition_table="all" if decomposition else None, strict=strict
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
        x = (torch.arange(4 * 3) + 10).reshape((1, -1, 4)).to(torch.float32)
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
                self.params = torch.zeros((1, 8192, 4), dtype=torch.float32)

            def forward(self, index, update):
                copy = self.params.clone()
                copy[..., index] = update
                return copy

        model = Model()
        update = (torch.arange(4 * 8192) + 10).reshape((-1, 4)).to(torch.float32)
        index = torch.from_numpy(np.array([0, 3, 2, 1])).to(torch.int64)
        expected = model(index, update)
        model_path = self._call_exporter(
            "test_aten_index_put_3d_nd_case_1", "custom", model, (index, update), strict=True
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
            "test_aten_index_put_3d_nd_case_2", "custom", model, (index, update), strict=True
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
    def test_aten_interpolate_bilinear(self):
        import torch

        class Model(torch.nn.Module):

            def __init__(self):
                super().__init__()

            def forward(self, x):
                y = torch.nn.functional.interpolate(
                    x,
                    scale_factor=2.0,
                    mode="bilinear",
                    recompute_scale_factor=False,
                )
                return y

        model = Model()
        x = torch.randn(1, 2, 3, 4, requires_grad=False)
        expected = model(x)
        model_path = self._call_exporter(
            "test_aten_interpolate_bilinear", "custom", model, (x,)
        )
        check_model(model_path)

        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        sess = onnxruntime.InferenceSession(
            model_path, sess_options=sess_options, providers=[("CPUExecutionProvider")]
        )
        feeds = dict(zip([i.name for i in sess.get_inputs()], [x.detach().numpy()]))
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-5)

    @skipif_ci_windows("not working on windows")
    def test_aten_nonzero_1(self):
        import torch

        class Model(torch.nn.Module):

            def __init__(self):
                super().__init__()

            def forward(self, x):
                y = torch.nonzero(x)
                return y

        model = Model()
        x = torch.randn(3, 4, requires_grad=False)
        expected = model(x)
        model_path = self._call_exporter("test_aten_nonzeros", "custom", model, (x,))
        check_model(model_path)

        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        sess = onnxruntime.InferenceSession(
            model_path, sess_options=sess_options, providers=[("CPUExecutionProvider")]
        )
        feeds = dict(zip([i.name for i in sess.get_inputs()], [x.detach().numpy()]))
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-5)

    @skipif_ci_windows("not working on windows")
    def test_aten_nonzero_tuple(self):
        import torch

        class Model(torch.nn.Module):

            def __init__(self):
                super().__init__()

            def forward(self, x):
                y = torch.nonzero(x, as_tuple=True)
                return y

        model = Model()
        x = torch.randn(3, 4, requires_grad=False)
        expected = model(x)
        model_path = self._call_exporter("test_aten_nonzeros", "custom", model, (x,))
        check_model(model_path)

        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        sess = onnxruntime.InferenceSession(
            model_path, sess_options=sess_options, providers=[("CPUExecutionProvider")]
        )
        feeds = dict(zip([i.name for i in sess.get_inputs()], [x.detach().numpy()]))
        got = sess.run(None, feeds)
        self.assertEqual(len(expected), 2)
        self.assertEqual(len(got), 2)
        for a, b in zip(expected, got):
            self.assertEqualArray(a, b, atol=1e-5)

    def test_as_strided_numpy(self):
        import torch

        def np_as_strided_1(x, shape, strides):
            assert len(shape) == len(x.shape) == 4
            y = np.empty(
                shape, dtype=oh.tensor_dtype_to_np_dtype(torch_dtype_to_onnx_dtype(x.dtype))
            )
            cs = (shape[1] * shape[2] * shape[3], shape[2] * shape[3], shape[3], 1)
            x_flat = x.flatten()
            y_flat = y.flatten()

            for i in range(0, shape[0]):
                for j in range(0, shape[1]):
                    for k in range(0, shape[2]):
                        for l in range(0, shape[3]):  # noqa: E741
                            pos_y = i * cs[0] + j * cs[1] + k * cs[2] + l * cs[3]
                            pos_x = (
                                i * strides[0]
                                + j * strides[1]
                                + k * strides[2]
                                + l * strides[3]
                            )
                            y_flat[pos_y] = x_flat[pos_x]
            return y_flat.reshape(shape)

        def np_as_strided_2(x, shape, strides):
            assert len(shape) == len(x.shape) == 4
            n_elems = np.prod(shape)
            indices = np.zeros(np.prod(shape), dtype=np.int64)
            shape_c = (shape[3] * shape[2] * shape[1], shape[2] * shape[3], shape[3], 1)
            for dim, dimc, stride in zip(shape, shape_c, strides):
                i = ((np.arange(n_elems) // dimc) % dim) * stride
                indices += i
            return x.flatten()[indices].reshape(shape)

        shape_x = (2, 2, 8, 8)
        shape = (2, 2, 8, 4)
        strides = (128, 8, 16, 1)
        x = torch.arange(np.prod(shape_x)).reshape(shape_x).to(int)
        expected = torch.as_strided(x, shape, strides)
        y = np_as_strided_1(x, shape, strides)
        self.assertEqualArray(expected, y)
        y = np_as_strided_2(x, shape, strides)
        self.assertEqualArray(expected, y)

    @skipif_ci_windows("not working on windows")
    def test_aten_as_strided(self):
        import torch

        class Model(torch.nn.Module):

            def __init__(self):
                super().__init__()

            def forward(self, x):
                y = torch.as_strided(x, (2, 2, 8, 4), (128, 8, 16, 1))
                return y

        model = Model()
        x = torch.randn((2, 2, 8, 8), requires_grad=False)
        expected = model(x)
        model_path = self._call_exporter("test_aten_as_strided", "custom", model, (x,))
        check_model(model_path)

        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        sess = onnxruntime.InferenceSession(
            model_path, sess_options=sess_options, providers=[("CPUExecutionProvider")]
        )
        feeds = dict(zip([i.name for i in sess.get_inputs()], [x.detach().numpy()]))
        got = sess.run(None, feeds)
        self.assertEqualArray(expected, got[0], atol=1e-5)

    @skipif_ci_windows("not working on windows")
    def test_aten_batch_norm(self):
        import torch

        class Model(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Buffer(torch.tensor([5, 7]).to(torch.float32)) * 2
                self.bias = torch.nn.Buffer(torch.tensor([4, 5]).to(torch.float32)) * 0.1
                self.running_mean = (
                    torch.nn.Buffer(torch.tensor([1, 4]).to(torch.float32)) * 0.5
                )
                self.running_var = (
                    torch.nn.Buffer(torch.tensor([1, 4]).to(torch.float32)) * 0.6
                )

            def forward(self, x):
                return torch.batch_norm(
                    x,
                    self.weight,
                    self.bias,
                    running_mean=self.running_mean,
                    running_var=self.running_var,
                    training=False,
                    momentum=0.5,
                    eps=0.6,
                    cudnn_enabled=False,
                )

        model = Model()
        x = torch.randn((2, 2, 8, 8), requires_grad=False)
        expected = model(x)
        model_path = self._call_exporter("test_aten_batch_norm", "custom", model, (x,))
        check_model(model_path)

        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        sess = onnxruntime.InferenceSession(
            model_path, sess_options=sess_options, providers=[("CPUExecutionProvider")]
        )
        feeds = dict(zip([i.name for i in sess.get_inputs()], [x.detach().numpy()]))
        got = sess.run(None, feeds)
        self.assertEqualArray(expected, got[0], atol=1e-5)

    @skipif_ci_windows("not working on windows")
    def test_aten_index_tensor_2_5(self):
        import torch

        class Model(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.i1 = torch.nn.Buffer(
                    torch.tensor([[[5, 7]]]).to(torch.int64).reshape((2, 1, 1))
                )
                self.i2 = torch.nn.Buffer(
                    torch.tensor([[2, 3, 4]]).to(torch.int64).reshape((3, 1))
                )
                self.i3 = torch.nn.Buffer(torch.tensor([7, 8, 9, 10]).to(torch.int64))

            def forward(self, x):
                return x[:, :, self.i1, self.i2, self.i3]

        model = Model()
        x = torch.arange(2 * 2 * 8 * 8 * 16).reshape((2, 2, 8, 8, 16)).to(torch.float32)
        expected = model(x)
        model_path = self._call_exporter("test_aten_index_tensor_2_5", "custom", model, (x,))
        check_model(model_path)

        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        sess = onnxruntime.InferenceSession(
            model_path, sess_options=sess_options, providers=[("CPUExecutionProvider")]
        )
        feeds = dict(zip([i.name for i in sess.get_inputs()], [x.detach().numpy()]))
        got = sess.run(None, feeds)
        self.assertEqualArray(expected, got[0], atol=1e-5)

    @skipif_ci_windows("not working on windows")
    def test_aten_index_tensor_2_4(self):
        import torch

        class Model(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.i2 = torch.nn.Buffer(
                    torch.tensor([[2, 3, 4]]).to(torch.int64).reshape((3, 1))
                )
                self.i3 = torch.nn.Buffer(torch.tensor([7, 8, 9, 10]).to(torch.int64))

            def forward(self, x):
                return x[:, :, self.i2, self.i3]

        model = Model()
        x = torch.arange(2 * 2 * 8 * 16).reshape((2, 2, 8, 16)).to(torch.float32)
        expected = model(x)
        model_path = self._call_exporter("test_aten_index_tensor_2_4", "custom", model, (x,))
        check_model(model_path)

        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        sess = onnxruntime.InferenceSession(
            model_path, sess_options=sess_options, providers=[("CPUExecutionProvider")]
        )
        feeds = dict(zip([i.name for i in sess.get_inputs()], [x.detach().numpy()]))
        got = sess.run(None, feeds)
        self.assertEqualArray(expected, got[0], atol=1e-5)

    @skipif_ci_windows("not working on windows")
    def test_aten_index_tensor_2_4_1_1(self):
        import torch

        class Model(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.i2 = torch.nn.Buffer(torch.randint(0, 14, (4, 14, 1, 1)).to(torch.int64))
                self.i3 = torch.nn.Buffer(torch.randint(0, 12, (4, 14)).to(torch.int64))

            def forward(self, x):
                return x[:, :, self.i2, self.i3]

        model = Model()
        x = torch.arange(128 * 24 * 56 * 56).reshape((128, 24, 56, 56)).to(torch.float32)
        expected = model(x)
        model_path = self._call_exporter(
            "test_aten_index_tensor_2_4_1_1", "custom", model, (x,)
        )
        check_model(model_path)

        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        sess = onnxruntime.InferenceSession(
            model_path, sess_options=sess_options, providers=[("CPUExecutionProvider")]
        )
        feeds = dict(zip([i.name for i in sess.get_inputs()], [x.detach().numpy()]))
        got = sess.run(None, feeds)
        self.assertEqualArray(expected.reshape((-1,)), got[0].reshape((-1,)), atol=1e-5)
        self.assertEqualArray(expected, got[0], atol=1e-5)

    @skipif_ci_windows("not working on windows")
    def test_aten_index_tensor_1_3(self):
        import torch

        class Model(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.i2 = torch.nn.Buffer(
                    torch.tensor([[2, 3, 4]]).to(torch.int64).reshape((3, 1))
                )
                self.i3 = torch.nn.Buffer(torch.tensor([7, 8, 9, 10]).to(torch.int64))

            def forward(self, x):
                return x[:, self.i2, self.i3]

        model = Model()
        x = torch.arange(2 * 8 * 16).reshape((2, 8, 16)).to(torch.float32)
        expected = model(x)
        model_path = self._call_exporter("test_aten_index_tensor_1_3", "custom", model, (x,))
        check_model(model_path)

        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        sess = onnxruntime.InferenceSession(
            model_path, sess_options=sess_options, providers=[("CPUExecutionProvider")]
        )
        feeds = dict(zip([i.name for i in sess.get_inputs()], [x.detach().numpy()]))
        got = sess.run(None, feeds)
        self.assertEqualArray(expected, got[0], atol=1e-5)

    @skipif_ci_windows("not working on windows")
    def test_aten_fmod(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x):
                return x.fmod(42)

        model = Model()
        x = torch.arange(2 * 8 * 16).reshape((2, 8, 16)).to(torch.float32)
        expected = model(x)
        model_path = self._call_exporter("test_aten_fmod", "custom", model, (x,))
        check_model(model_path)

        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        sess = onnxruntime.InferenceSession(
            model_path, sess_options=sess_options, providers=[("CPUExecutionProvider")]
        )
        feeds = dict(zip([i.name for i in sess.get_inputs()], [x.detach().numpy()]))
        got = sess.run(None, feeds)
        self.assertEqualArray(expected, got[0], atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
