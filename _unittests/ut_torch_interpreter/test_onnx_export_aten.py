import os
import unittest
from typing import Any, List, Optional
import numpy as np
import onnx.helper as oh
from onnx.checker import check_model
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    skipif_ci_windows,
    requires_cuda,
    requires_torch,
)
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.torch_interpreter import to_onnx, ExportOptions
from experimental_experiment.xbuilder import OptimizationOptions
from experimental_experiment.helpers import torch_dtype_to_onnx_dtype


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
        patterns: Optional[str] = None,
        dynamic_shapes: Optional[Any] = None,
        processor: str = "CPU",
    ) -> str:
        import torch

        if not os.path.exists("dump_test"):
            os.mkdir("dump_test")
        filename = os.path.join(
            "dump_test", f"{test_name}_{exporter}_{'dec' if decomposition else ''}.onnx"
        )
        if exporter == "script":
            torch.onnx.export(model, inputs, filename, opset_version=18)
        elif exporter == "dynamo":
            torch.onnx.export(
                model, inputs, filename, dynamo=True, dynamic_shapes=dynamic_shapes
            )
        else:
            export_options = ExportOptions(
                decomposition_table="all" if decomposition else None, strict=strict
            )
            opt_options = (
                OptimizationOptions(patterns=patterns, processor=processor)
                if patterns or processor != "CPU"
                else None
            )
            to_onnx(
                model,
                inputs,
                filename=filename,
                export_options=export_options,
                verbose=verbose,
                optimize=optimize,
                options=opt_options,
                dynamic_shapes=dynamic_shapes,
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
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
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
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
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
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
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
        x = (torch.randn(3, 4, requires_grad=False) < 0.4).to(torch.float32)
        expected = model(x)
        model_path = self._call_exporter("test_aten_nonzeros_1", "custom", model, (x,))
        check_model(model_path)

        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        sess = onnxruntime.InferenceSession(
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        feeds = dict(zip([i.name for i in sess.get_inputs()], [x.detach().numpy()]))
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-5)

    @skipif_ci_windows("not working on windows")
    def test_aten_nonzero_1_d3(self):
        import torch

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                y = torch.nonzero(x)
                return y

        model = Model()
        x = (torch.randn(3, 4, 5, requires_grad=False) < 0.4).to(torch.float32)
        expected = model(x)
        model_path = self._call_exporter("test_aten_nonzeros_1_d3", "custom", model, (x,))
        check_model(model_path)

        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        sess = onnxruntime.InferenceSession(
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        feeds = dict(zip([i.name for i in sess.get_inputs()], [x.detach().numpy()]))
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-5)

    @skipif_ci_windows("not working on windows")
    def test_aten_nonzero_1_d1(self):
        import torch

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                y = torch.nonzero(x)
                return y

        model = Model()
        x = (torch.randn(20, requires_grad=False) < 0.4).to(torch.float32)
        expected = model(x)
        model_path = self._call_exporter("test_aten_nonzeros_1_d1", "custom", model, (x,))
        check_model(model_path)

        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        sess = onnxruntime.InferenceSession(
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
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
        x = (torch.randn(3, 4, requires_grad=False) < 0.4).to(torch.float32)
        expected = model(x)
        model_path = self._call_exporter("test_aten_nonzeros_tuple", "custom", model, (x,))
        check_model(model_path)

        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        sess = onnxruntime.InferenceSession(
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        feeds = dict(zip([i.name for i in sess.get_inputs()], [x.detach().numpy()]))
        got = sess.run(None, feeds)
        self.assertEqual(len(expected), 2)
        self.assertEqual(len(got), 2)
        for a, b in zip(expected, got):
            self.assertEqualArray(a, b, atol=1e-5)

    @skipif_ci_windows("not working on windows")
    def test_aten_nonzero_tuple_d3(self):
        import torch

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                y = torch.nonzero(x, as_tuple=True)
                return y

        model = Model()
        x = (torch.randn(3, 4, 5, requires_grad=False) < 0.4).to(torch.float32)
        expected = model(x)
        model_path = self._call_exporter("test_aten_nonzeros_tuple_d3", "custom", model, (x,))
        check_model(model_path)

        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        sess = onnxruntime.InferenceSession(
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        feeds = dict(zip([i.name for i in sess.get_inputs()], [x.detach().numpy()]))
        got = sess.run(None, feeds)
        self.assertEqual(len(expected), 3)
        self.assertEqual(len(got), 3)
        for a, b in zip(expected, got):
            self.assertEqualArray(a, b, atol=1e-5)

    @skipif_ci_windows("not working on windows")
    def test_aten_nonzero_tuple_d1(self):
        import torch

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                y = torch.nonzero(x, as_tuple=True)
                return y

        model = Model()
        x = (torch.randn(34, requires_grad=False) < 0.4).to(torch.float32)
        expected = model(x)
        model_path = self._call_exporter("test_aten_nonzeros_tuple_d1", "custom", model, (x,))
        check_model(model_path)

        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        sess = onnxruntime.InferenceSession(
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        feeds = dict(zip([i.name for i in sess.get_inputs()], [x.detach().numpy()]))
        got = sess.run(None, feeds)
        self.assertEqual(len(expected), 1)
        self.assertEqual(len(got), 1)
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
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        feeds = dict(zip([i.name for i in sess.get_inputs()], [x.detach().numpy()]))
        got = sess.run(None, feeds)
        self.assertEqualArray(expected, got[0], atol=1e-5)

    @skipif_ci_windows("not working on windows")
    def test_aten_batch_norm_notraining(self):
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
        model_path = self._call_exporter(
            "test_aten_batch_norm_notraining", "custom", model, (x,)
        )
        check_model(model_path)

        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        sess = onnxruntime.InferenceSession(
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        feeds = dict(zip([i.name for i in sess.get_inputs()], [x.detach().numpy()]))
        got = sess.run(None, feeds)
        self.assertEqualArray(expected, got[0], atol=1e-5)

    @skipif_ci_windows("not working on windows")
    @requires_torch("2.6", "avoid _native_batch_norm_legit_functional")
    def test_aten_batch_norm_training(self):
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
                    training=True,
                    momentum=0.5,
                    eps=0.6,
                    cudnn_enabled=False,
                )

        model = Model()
        x = torch.randn((2, 2, 8, 8), requires_grad=False)
        expected = model(x)
        model_path = self._call_exporter(
            "test_aten_batch_norm_training", "custom", model, (x,)
        )
        check_model(model_path)

        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        sess = onnxruntime.InferenceSession(
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        feeds = dict(zip([i.name for i in sess.get_inputs()], [x.detach().numpy()]))
        got = sess.run(None, feeds)
        self.assertEqualArray(expected, got[0], atol=1e-5)

    @skipif_ci_windows("not working on windows")
    @requires_torch("2.6", "avoid _native_batch_norm_legit_functional")
    def test_aten_batch_norm_training16(self):
        import torch

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Buffer(
                    torch.abs(torch.randn((16,), dtype=torch.float16)) + 1
                )
                self.bias = torch.nn.Buffer(torch.randn((16,), dtype=torch.float16))
                self.running_mean = torch.nn.Buffer(torch.randn((16,), dtype=torch.float16))
                self.running_var = torch.nn.Buffer(
                    torch.abs(torch.randn((16,), dtype=torch.float16)) + 1
                )

            def forward(self, x):
                return torch.batch_norm(
                    x,
                    self.weight,
                    self.bias,
                    running_mean=self.running_mean,
                    running_var=self.running_var,
                    training=True,
                    momentum=0,
                    eps=0.006,
                    cudnn_enabled=False,
                )

        model = Model()
        x = torch.randn((1, 16, 27), requires_grad=False).to(torch.float16)
        expected = model(x)
        model_path = self._call_exporter(
            "test_aten_batch_norm_training16", "custom", model, (x,)
        )
        check_model(model_path)

        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        sess = onnxruntime.InferenceSession(
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        feeds = dict(zip([i.name for i in sess.get_inputs()], [x.detach().numpy()]))
        got = sess.run(None, feeds)
        self.assertEqualArray(expected, got[0], atol=1e-2)

    @skipif_ci_windows("not working on windows")
    @requires_cuda()
    def test_aten_batch_norm_training_cuda(self):
        import torch

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Buffer(
                    torch.abs(torch.randn((16,), dtype=torch.float32)) + 1
                )
                self.bias = torch.nn.Buffer(torch.randn((16,), dtype=torch.float32))
                self.running_mean = torch.nn.Buffer(torch.randn((16,), dtype=torch.float32))
                self.running_var = torch.nn.Buffer(
                    torch.abs(torch.randn((16,), dtype=torch.float32)) + 1
                )

            def forward(self, x):
                return torch.batch_norm(
                    x,
                    self.weight,
                    self.bias,
                    running_mean=self.running_mean,
                    running_var=self.running_var,
                    training=True,
                    momentum=1,
                    eps=0.006,
                    cudnn_enabled=False,
                )

        model = Model().to("cuda")
        x = torch.randn((1, 16, 27), requires_grad=False).to(torch.float32).to("cuda")
        expected = model(x)

        # check on CPU

        model_path = self._call_exporter(
            "test_aten_batch_norm_training_cudano",
            "custom",
            model,
            (x,),
            optimize=True,
            patterns="default-BatchNormalizationTraining",
        )
        check_model(model_path)
        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        sess = onnxruntime.InferenceSession(
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        feeds = dict(zip([i.name for i in sess.get_inputs()], [x.detach().cpu().numpy()]))
        got = sess.run(None, feeds)
        ref = ExtendedReferenceEvaluator(model_path, verbose=0)
        gotr = ref.run(None, feeds)
        self.assertEqualArray(got[0], gotr[0], atol=1e-4)
        self.assertEqualArray(expected, got[0], atol=1e-4)

        # check on CPU decomposed

        model_path = self._call_exporter(
            "test_aten_batch_norm_training_cuda_dec",
            "custom",
            model,
            (x,),
            optimize=True,
            patterns="default",
        )
        check_model(model_path)
        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        sess = onnxruntime.InferenceSession(
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        feeds = dict(zip([i.name for i in sess.get_inputs()], [x.detach().cpu().numpy()]))
        got = sess.run(None, feeds)
        ref = ExtendedReferenceEvaluator(model_path, verbose=0)
        gotr = ref.run(None, feeds)
        self.assertEqualArray(got[0], gotr[0], atol=1e-4)
        self.assertEqualArray(expected, got[0], atol=1e-4)

        # check on CUDA

        model_path = self._call_exporter(
            "test_aten_batch_norm_training_cuda",
            "custom",
            model,
            (x,),
            optimize=True,
            patterns="onnxruntime",
            processor="CUDA",
        )
        check_model(model_path)

        sess_options = onnxruntime.SessionOptions()
        sess = onnxruntime.InferenceSession(
            model_path, sess_options=sess_options, providers=["CUDAExecutionProvider"]
        )
        feeds = dict(zip([i.name for i in sess.get_inputs()], [x.detach().cpu().numpy()]))
        got = sess.run(None, feeds)
        ref = ExtendedReferenceEvaluator(model_path, verbose=0)
        gotr = ref.run(None, feeds)
        self.assertEqualArray(got[0], gotr[0], atol=1e-4)
        self.assertEqualArray(expected, got[0], atol=1e-4)

    @skipif_ci_windows("not working on windows")
    def test_aten_batch_norm_training16_none(self):
        import torch

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Buffer(
                    torch.abs(torch.randn((16,), dtype=torch.float16)) + 1
                )
                self.bias = torch.nn.Buffer(torch.randn((16,), dtype=torch.float16))

            def forward(self, x):
                return torch.batch_norm(
                    x,
                    self.weight,
                    self.bias,
                    running_mean=None,
                    running_var=None,
                    training=True,
                    momentum=0,
                    eps=1e-5,
                    cudnn_enabled=False,
                )

        model = Model()
        x = torch.randn((1, 16, 27), requires_grad=False).to(torch.float16)
        expected = model(x)
        model_path = self._call_exporter(
            "test_aten_batch_norm_training16_none", "custom", model, (x,)
        )
        check_model(model_path)

        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        sess = onnxruntime.InferenceSession(
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        feeds = dict(zip([i.name for i in sess.get_inputs()], [x.detach().numpy()]))
        got = sess.run(None, feeds)
        self.assertEqualArray(expected, got[0], atol=1e-2)

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
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
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
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
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
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
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
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
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
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        feeds = dict(zip([i.name for i in sess.get_inputs()], [x.detach().numpy()]))
        got = sess.run(None, feeds)
        self.assertEqualArray(expected, got[0], atol=1e-5)

    @skipif_ci_windows("not working on windows")
    def test_aten_conv(self):
        import torch

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv1d(16, 32, 1)

            def forward(self, x):
                return self.conv(x) + torch.tensor([1], dtype=x.dtype)

        model = Model()
        xs = (torch.randn((2, 16, 24)),)
        expected = model(*xs)
        model_path = self._call_exporter(
            "test_aten_conv",
            "custom",
            model,
            xs,
            dynamic_shapes={"x": {0: torch.export.Dim("batch")}},
        )
        sess = ExtendedReferenceEvaluator(model_path, verbose=0)
        feeds = dict(zip(sess.input_names, [x.numpy() for x in xs]))
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-5)

        # checking with onnxruntime as well
        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        sess = onnxruntime.InferenceSession(
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-5)

    @skipif_ci_windows("not working on windows")
    def test_aten_clone_index_Tensor(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x, sumx):
                K_33 = x.clone()
                K_33[2:-2, 2:-2, :-1] = sumx[None, :, None]
                K_33[2:-2, 2:-2, -1] = 0.0
                return K_33

        model = Model()
        xs = (
            (torch.arange(7 * 9 * 11) + 10).reshape((7, 9, 11)).to(torch.float32),
            torch.arange(5).to(torch.float32),
        )
        expected = model(*xs)
        model_path = self._call_exporter("test_aten_clone_index_Tensor", "custom", model, xs)
        sess = ExtendedReferenceEvaluator(model_path, verbose=0)
        feeds = dict(zip(sess.input_names, [x.numpy() for x in xs]))
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

        # checking with onnxruntime as well
        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        sess = onnxruntime.InferenceSession(
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    @skipif_ci_windows("not working on windows")
    def test_aten_clone_index_Tensor_0_2(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x, ind1, ind2):
                return x[ind1, ind2]

        model = Model()
        xs = (
            (torch.arange(2) + 10).reshape((1, 2)).to(torch.float32),
            torch.zeros((1, 1), dtype=torch.int64),
            torch.zeros((1, 2), dtype=torch.int64),
        )
        xsf = (
            torch.zeros((0, 2)),
            torch.zeros((0, 1), dtype=torch.int64),
            torch.zeros((0, 2), dtype=torch.int64),
        )
        expected = model(*xsf)
        model_path = self._call_exporter(
            "test_aten_clone_index_Tensor_0_2", "custom", model, xs
        )
        sess = ExtendedReferenceEvaluator(model_path, verbose=0)
        feeds = dict(zip(sess.input_names, [x.numpy() for x in xsf]))
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    @skipif_ci_windows("not working on windows")
    def test_aten_index_put_mask_bool_fixed_size(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x, values):
                x = x.clone()
                x[x < 2] = values
                return x

        model = Model()
        xs = (torch.arange(6).to(torch.float32), torch.tensor([-5, -6], dtype=torch.float32))
        expected = model(*xs)
        model_path = self._call_exporter(
            "test_aten_index_put_mask_bool_fixed_size", "custom", model, xs
        )
        sess = ExtendedReferenceEvaluator(model_path, verbose=0)
        feeds = dict(zip(sess.input_names, [x.numpy() for x in xs]))
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

        # checking with onnxruntime as well
        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        sess = onnxruntime.InferenceSession(
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    @skipif_ci_windows("not working on windows")
    def test_aten_index_put_mask_bool_fixed_broadcast_2d(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x, values):
                x = x.clone()
                mask = torch.tensor([True, False, True, True, False]).to(torch.bool)
                x[mask] = values
                return x

        model = Model()
        xs = (
            torch.arange(25).reshape((5, 5)).to(torch.float32),
            torch.tensor([700, 800, 900, 1000, 1100], dtype=torch.float32),
        )
        expected = model(*xs)
        model_path = self._call_exporter(
            "test_aten_index_put_mask_bool_fixed_broadcast_2d", "custom", model, xs
        )
        sess = ExtendedReferenceEvaluator(model_path, verbose=0)
        feeds = dict(zip(sess.input_names, [x.numpy() for x in xs]))
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

        # checking with onnxruntime as well
        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        sess = onnxruntime.InferenceSession(
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    @skipif_ci_windows("not working on windows")
    def test_aten_index_put_mask_bool_fixed_broadcast_3d(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x, values):
                x = x.clone()
                mask = torch.tensor([True, False]).to(torch.bool)
                x[mask] = values
                return x
                # return torch.ops.aten.index_put(x, (mask,), values)

        model = Model()
        xs = (
            torch.arange(2 * 3 * 5).reshape((2, 3, 5)).to(torch.float32),
            torch.tensor([700, 800, 900, 1000, 1100], dtype=torch.float32),
        )
        expected = model(*xs)
        model_path = self._call_exporter(
            "test_aten_index_put_mask_bool_fixed_broadcast_3d", "custom", model, xs
        )
        sess = ExtendedReferenceEvaluator(model_path, verbose=0)
        feeds = dict(zip(sess.input_names, [x.numpy() for x in xs]))
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

        # checking with onnxruntime as well
        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        sess = onnxruntime.InferenceSession(
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    @skipif_ci_windows("not working on windows")
    def test_aten_index_put_mask_bool_fixed_broadcast_3d_2(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x, values):
                x = x.clone()
                mask = torch.tensor([[True, False, False], [True, True, False]]).to(torch.bool)
                x[mask] = values
                return x
                # return torch.ops.aten.index_put(x, (mask,), values)

        model = Model()
        xs = (
            torch.arange(2 * 3 * 5).reshape((2, 3, 5)).to(torch.float32),
            torch.tensor([700, 800, 900, 1000, 1100], dtype=torch.float32),
        )
        expected = model(*xs)
        model_path = self._call_exporter(
            "test_aten_index_put_mask_bool_fixed_broadcast_3d", "custom", model, xs
        )
        sess = ExtendedReferenceEvaluator(model_path, verbose=0)
        feeds = dict(zip(sess.input_names, [x.numpy() for x in xs]))
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

        # checking with onnxruntime as well
        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        sess = onnxruntime.InferenceSession(
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_aten_take(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x, ind):
                return x.take(ind)

        model = Model()
        xs = (
            torch.arange(2 * 3).reshape((2, 3)).to(torch.float32),
            torch.tensor([0, 2, 5], dtype=torch.int64),
        )
        expected = model(*xs)
        model_path = self._call_exporter("test_aten_take", "custom", model, xs)
        sess = ExtendedReferenceEvaluator(model_path, verbose=0)
        feeds = dict(zip(sess.input_names, [x.numpy() for x in xs]))
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

        # checking with onnxruntime as well
        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        sess = onnxruntime.InferenceSession(
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_aten_index_tensor_2_1_2(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x, ind1, ind2):
                return x[ind1, ind2]

        model = Model()
        xs = (
            torch.arange(4 * 5).reshape((4, -1)).to(torch.float32),
            torch.tensor([1, 2], dtype=torch.int64).reshape((-1, 1)),
            torch.tensor([[0, 1], [2, 3]], dtype=torch.int64),
        )
        expected = model(*xs)
        model_path = self._call_exporter("test_aten_index_tensor_2_1_2", "custom", model, xs)
        sess = ExtendedReferenceEvaluator(model_path, verbose=10)
        feeds = dict(zip(sess.input_names, [x.numpy() for x in xs]))
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

        # checking with onnxruntime as well
        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        sess = onnxruntime.InferenceSession(
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_aten_index_put_inplace_column(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x, ind, vals):
                x = x.clone()
                col = x[:, 0]
                col[ind] = vals
                return x

        model = Model()
        xs = (
            torch.arange(4 * 5).reshape((4, -1)).to(torch.float32),
            torch.tensor([1, 2], dtype=torch.int64),
            torch.tensor([100, 101], dtype=torch.float32),
        )
        expected = model(*xs)

        DYNAMIC = torch.export.Dim.DYNAMIC
        ds = ({0: DYNAMIC, 1: DYNAMIC}, {0: DYNAMIC}, {0: DYNAMIC})
        ep1 = torch.export.export(model, xs, dynamic_shapes=ds, strict=False)
        s1 = str(ep1)
        ep1 = ep1.run_decompositions()
        s2 = str(ep1)
        self.assertNotEqual(s1, s2)

        model_path = self._call_exporter(
            "test_aten_index_put_inplace_column",
            "custom",
            model,
            xs,
            dynamic_shapes=ds,
            decomposition=True,
        )
        sess = ExtendedReferenceEvaluator(model_path, verbose=10)
        feeds = dict(zip(sess.input_names, [x.numpy() for x in xs]))
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

        # checking with onnxruntime as well
        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        sess = onnxruntime.InferenceSession(
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got)


if __name__ == "__main__":
    unittest.main(verbosity=2)
