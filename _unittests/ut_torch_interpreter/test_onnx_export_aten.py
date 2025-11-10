import itertools
import os
import unittest
from typing import Any, List, Optional
import numpy as np
import onnx
import onnx.helper as oh
from onnx.checker import check_model
from onnx_diagnostic.torch_export_patches import torch_export_patches
from onnx_diagnostic.torch_export_patches.patch_inputs import use_dyn_not_str
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    skipif_ci_windows,
    requires_cuda,
    requires_onnx_diagnostic,
    requires_torch,
    ignore_warnings,
    hide_stdout,
)
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.torch_interpreter import to_onnx, ExportOptions
from experimental_experiment.xbuilder import OptimizationOptions
from experimental_experiment.helpers import torch_dtype_to_onnx_dtype, string_type
from experimental_experiment.torch_test_helper import to_numpy


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
            torch.onnx.export(model, inputs, filename, dynamo=True, dynamic_shapes=dynamic_shapes)
        else:
            export_options = ExportOptions(
                decomposition_table="all" if decomposition else None,
                strict=strict,
                backed_size_oblivious=False,
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

    @skipif_ci_windows("broken")
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

    @skipif_ci_windows("broken")
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
        model_path = self._call_exporter("test_aten_interpolate_bilinear", "custom", model, (x,))
        check_model(model_path)

        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        sess = onnxruntime.InferenceSession(
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        feeds = dict(zip([i.name for i in sess.get_inputs()], [x.detach().numpy()]))
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-5)

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
                                i * strides[0] + j * strides[1] + k * strides[2] + l * strides[3]
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
                self.running_mean = torch.nn.Buffer(torch.tensor([1, 4]).to(torch.float32)) * 0.5
                self.running_var = torch.nn.Buffer(torch.tensor([1, 4]).to(torch.float32)) * 0.6

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
        model_path = self._call_exporter("test_aten_batch_norm_notraining", "custom", model, (x,))
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
                self.running_mean = torch.nn.Buffer(torch.tensor([1, 4]).to(torch.float32)) * 0.5
                self.running_var = torch.nn.Buffer(torch.tensor([1, 4]).to(torch.float32)) * 0.6

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
        model_path = self._call_exporter("test_aten_batch_norm_training", "custom", model, (x,))
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
        model_path = self._call_exporter("test_aten_batch_norm_training16", "custom", model, (x,))
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
        model_path = self._call_exporter("test_aten_index_tensor_2_4_1_1", "custom", model, (x,))
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
        model_path = self._call_exporter("test_aten_clone_index_Tensor_0_2", "custom", model, xs)
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
        sess = ExtendedReferenceEvaluator(model_path)
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

    def test_aten_index_put_inplace_column_0(self):
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

    def test_aten_index_put_inplace_column_1(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x, ind, vals):
                x = x.clone()
                col = x[1, :]
                col[ind] = vals
                return x

        model = Model()
        xs = (
            torch.arange(4 * 4).reshape((4, -1)).to(torch.float32),
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

    def test_aten_batch_norm_momentum(self):
        import torch

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = torch.nn.BatchNorm2d(128, affine=False, momentum=0.3)

            def forward(self, x):
                return self.bn(x)

        model = Model()
        xs = (torch.randn(128, 128, 1, 1),)
        expected = model(*xs)
        model_path = self._call_exporter("test_aten_batch_node", "custom", model, xs)
        sess = ExtendedReferenceEvaluator(model_path, verbose=0)
        feeds = dict(zip(sess.input_names, [x.numpy() for x in xs]))
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-4)

        # checking with onnxruntime as well
        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        sess = onnxruntime.InferenceSession(
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-4)

    @ignore_warnings(UserWarning)
    def test_aten_scatter_reduce_include_self(self):
        import torch

        skip_ort = {
            ("sum", False, "float16"),
            ("sum", True, "float16"),
            ("prod", False, "float16"),
            ("prod", True, "float16"),
        }

        for red, include, stype in itertools.product(
            ["prod", "sum", "amin", "amax"],
            [False, True],
            ["bfloat16", "float32", "float16", "int32", "int64", "float64"],
        ):
            with self.subTest(reduce=red, include=include, type=stype):
                dtype = getattr(torch, stype)

                class Model(torch.nn.Module):
                    def __init__(self, include, red):
                        super().__init__()
                        self.include = include
                        self.red = red

                    def forward(self, x, indices, updates):
                        x = x.clone()
                        return x.scatter_reduce(
                            0, indices, updates, self.red, include_self=self.include
                        )

                model = Model(include, red)
                xs = (
                    torch.tensor([[-2, 0, 2], [2, -2, 0]], dtype=dtype),
                    torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.int64),
                    torch.tensor([[-1, -1, -1], [-1, -1, -1]], dtype=dtype),
                )
                expected = model(*xs)
                model_path = self._call_exporter(
                    f"test_aten_scatter_{red}_{'include' if include else 'exclude'}_{stype}",
                    "custom",
                    model,
                    xs,
                )
                sess = ExtendedReferenceEvaluator(model_path, verbose=0)
                feeds = dict(zip(sess.input_names, [to_numpy(x) for x in xs]))
                got = sess.run(None, feeds)[0]
                self.assertEqualArray(expected, got, atol=1e-6)

                # checking with onnxruntime as well
                if stype == "bfloat16":
                    # not supported
                    continue
                key = red, include, stype
                if key in skip_ort:
                    self.todo(
                        self.test_aten_scatter_reduce_include_self,
                        f"case {key} not supported by onnxruntime",
                    )
                    continue
                import onnxruntime

                sess_options = onnxruntime.SessionOptions()
                sess = onnxruntime.InferenceSession(
                    model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
                )
                got = sess.run(None, feeds)[0]
                self.assertEqualArray(expected, got, atol=1e-5)

    @ignore_warnings(UserWarning)
    def test_aten_isin(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, elements, test_elements):
                return torch.isin(elements, test_elements)

        model = Model()

        for rk in (1, 2, 3):
            with self.subTest(rank=rk):
                xs = (
                    torch.arange(2 * 3 * 5)
                    .to(torch.int64)
                    .reshape((2 * 3 * 5,) if rk == 1 else ((2 * 3, 5) if rk == 2 else (2, 3, 5))),
                    torch.arange(2 * 3 * 5)[::2].to(torch.int64),
                )
                expected = model(*xs)
                model_path = self._call_exporter("test_aten_isin", "custom", model, xs)
                sess = ExtendedReferenceEvaluator(model_path)
                feeds = dict(zip(sess.input_names, [x.numpy() for x in xs]))
                got = sess.run(None, feeds)[0]
                self.assertEqualArray(expected, got)

                # checking with onnxruntime as well
                import onnxruntime

                sess = onnxruntime.InferenceSession(
                    model_path, providers=["CPUExecutionProvider"]
                )
                got = sess.run(None, feeds)[0]
                self.assertEqualArray(expected, got)

    @ignore_warnings(UserWarning)
    def test_aten_multinomial(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, elements):
                return torch.multinomial(elements, 5)

        model = Model()

        for rk in (2, 1):
            with self.subTest(rank=rk):
                xs = (
                    (torch.arange(2 * 3 * 5) / (2 * 3 * 5))
                    .to(torch.float32)
                    .reshape((2 * 3 * 5,) if rk == 1 else ((2 * 3, 5) if rk == 2 else (2, 3, 5))),
                )
                expected = model(*xs)
                model_path = self._call_exporter(
                    f"test_aten_multinomial{rk}", "custom", model, xs
                )
                onx = onnx.load(model_path)
                feeds = dict(zip([i.name for i in onx.graph.input], [x.numpy() for x in xs]))

                import onnxruntime

                sess = onnxruntime.InferenceSession(
                    model_path, providers=["CPUExecutionProvider"]
                )
                got = sess.run(None, feeds)[0]
                self.assertEqual(expected.shape, got.shape)
                self.assertEqual(expected.numpy().dtype, got.dtype)

    @ignore_warnings(UserWarning)
    def test_aten_masked_scatter(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x, mask, updates):
                return x.masked_scatter(mask, updates)

        model = Model()

        for rk in (1, 2):
            with self.subTest(rank=rk):
                xs = (
                    torch.tensor([0, 1, 0] if rk == 1 else [[0, 1, 0]], dtype=torch.float32),
                    torch.tensor([[0, 1, 0], [1, 1, 0]], dtype=torch.bool),
                    torch.tensor([[-10, -11, -12], [-13, -14, -15]], dtype=torch.float32),
                )
                self.assertIn("x", string_type(xs, with_shape=True))
                expected = model(*xs)
                model_path = self._call_exporter(
                    f"test_aten_masked_scatter{rk}", "custom", model, xs
                )
                onx = onnx.load(model_path)
                feeds = dict(zip([i.name for i in onx.graph.input], [x.numpy() for x in xs]))

                import onnxruntime

                sess = onnxruntime.InferenceSession(
                    model_path, providers=["CPUExecutionProvider"]
                )
                got = sess.run(None, feeds)[0]
                self.assertEqual(expected.shape, got.shape)
                self.assertEqual(expected.numpy().dtype, got.dtype)

    @ignore_warnings(UserWarning)
    def test_aten_masked_scatter_samedim(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x, mask, updates):
                return x.masked_scatter(mask, updates)

        model = Model()

        xs = (
            torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=torch.float32),
            torch.tensor([[0, 0, 0, 1, 1], [1, 1, 0, 1, 1]], dtype=torch.bool),
            torch.tensor([[11, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=torch.float32),
        )
        expected = model(*[x.clone() for x in xs])
        DYN = torch.export.Dim.DYNAMIC
        onx = to_onnx(
            model,
            xs,
            dynamic_shapes=({0: DYN, 1: DYN}, {0: DYN, 1: DYN}, {0: DYN, 1: DYN}),
            export_options=ExportOptions(aten_as_function=set()),
        )
        self.dump_onnx("test_aten_masked_scatter_samedim.onnx", onx)

        import onnxruntime

        sess = onnxruntime.InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        feeds = dict(zip([i.name for i in sess.get_inputs()], [t.numpy() for t in xs]))
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    @ignore_warnings(UserWarning)
    def test_aten_masked_scatter_notsamedim(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x, mask, updates):
                return x.masked_scatter(mask, updates)

        model = Model()

        xs = (
            torch.tensor([0, 0, 0, 0, 0], dtype=torch.float32),
            torch.tensor([[0, 0, 0, 1, 1], [1, 1, 0, 1, 1]], dtype=torch.bool),
            torch.tensor([[11, 1, 2, 3, 4, 12], [5, 6, 7, 8, 9, 13]], dtype=torch.float32),
        )
        expected = model(*[x.clone() for x in xs])
        DYN = torch.export.Dim.DYNAMIC
        onx = to_onnx(
            model,
            xs,
            dynamic_shapes=({0: DYN}, {0: DYN, 1: DYN}, {0: DYN, 1: DYN}),
            export_options=ExportOptions(aten_as_function=set()),
        )
        self.dump_onnx("test_aten_masked_scatter_notsamedim.onnx", onx)

        import onnxruntime

        sess = onnxruntime.InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        feeds = dict(zip([i.name for i in sess.get_inputs()], [t.numpy() for t in xs]))
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    @ignore_warnings(UserWarning)
    @requires_torch("2.8")
    def test_symbolic(self):
        import torch

        class CustomSymOp(torch.nn.Module):
            def forward(self, x: torch.Tensor, mode: torch.Tensor) -> torch.Tensor:
                val = torch.onnx.ops.symbolic(
                    "custom_domain::CustomSymOp",
                    (x, mode),
                    dtype=x.dtype,
                    shape=(*x.shape[:-1], x.shape[-1] * 3),
                    version=1,
                )
                return val

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sub = CustomSymOp()

            def forward(self, x, mod):
                return self.sub(x, mod)

        model = Model()
        inputs = (torch.rand((3, 3)), torch.tensor([1], dtype=torch.int64))
        model.eval()
        onx = to_onnx(model, inputs, export_options=ExportOptions(strict=False))
        names = [n.op_type for n in onx.graph.node]
        self.assertEqual(names, ["CustomSymOp"])
        domains = [d.domain for d in onx.opset_import]
        self.assertEqual(domains, ["", "custom_domain"])

    @ignore_warnings(UserWarning)
    def test_aten_index_tensor_rk2_rk4_rk4(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x, ind1, ind2):
                return x[ind1, ind2]

        model = Model()
        xs = (
            torch.rand((2, 33)),
            torch.randint(0, 1, (2, 1, 2, 33)),
            torch.randint(0, 32, (2, 1, 2, 33)),
        )
        expected = model(*xs)
        model_path = self._call_exporter(
            "test_aten_index_tensor_rk2_rk4_rk4", "custom", model, xs
        )
        sess = ExtendedReferenceEvaluator(model_path, verbose=0)
        feeds = dict(zip(sess.input_names, [to_numpy(x) for x in xs]))
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-6)

        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        sess = onnxruntime.InferenceSession(
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-5)

    @ignore_warnings(UserWarning)
    def test_aten_meshgrid(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x, y, z):
                return torch.meshgrid(x, y, z)

        x = torch.arange(3).to(torch.int64)
        y = torch.arange(4).to(torch.int64)
        z = torch.arange(5).to(torch.int64)

        model = Model()
        xs = (x, y, z)
        expected = tuple(t.numpy() for t in model(*xs))
        model_path = self._call_exporter("test_aten_meshgrid", "custom", model, xs)
        sess = ExtendedReferenceEvaluator(model_path, verbose=0)
        feeds = dict(zip(sess.input_names, [to_numpy(x) for x in xs]))
        got = sess.run(None, feeds)
        self.assertEqualAny(expected, tuple(got), atol=1e-6)

        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        sess = onnxruntime.InferenceSession(
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, feeds)
        self.assertEqualAny(expected, tuple(got), atol=1e-5)

    @ignore_warnings(UserWarning)
    def test_aten_pow_tensor_scalar(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.pow(x, 0.466)

        x = torch.tensor(4.5, dtype=torch.float32)
        model = Model()
        xs = (x,)
        expected = model(*xs).numpy()
        model_path = self._call_exporter("test_aten_pow_tensor_scalar", "custom", model, xs)
        sess = ExtendedReferenceEvaluator(model_path, verbose=0)
        feeds = dict(zip(sess.input_names, [to_numpy(x) for x in xs]))
        got = sess.run(None, feeds)
        self.assertEqualAny(expected, got[0], atol=1e-6)

        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        sess = onnxruntime.InferenceSession(
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, feeds)
        self.assertEqualAny(expected, got[0], atol=1e-5)

    @ignore_warnings(UserWarning)
    def test_aten_where_only_condition(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.where(x.to(bool))

        x = torch.tensor([0, 1, 3, 0], dtype=torch.float32)
        model = Model()
        xs = (x,)
        expected = [t.numpy() for t in model(*xs)]
        model_path = self._call_exporter("test_aten_where_only_condition", "custom", model, xs)
        sess = ExtendedReferenceEvaluator(model_path, verbose=0)
        feeds = dict(zip(sess.input_names, [to_numpy(x) for x in xs]))
        got = sess.run(None, feeds)
        self.assertEqualAny(expected, got, atol=1e-6)

        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        sess = onnxruntime.InferenceSession(
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, feeds)
        self.assertEqualAny(expected, got, atol=1e-5)

    @ignore_warnings(UserWarning)
    def test_aten_index_put_no_dimension_2_2_0(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x, ind1, ind2):
                cst = torch.tensor(5, dtype=x.dtype)
                x = x.clone()
                x[ind1, ind2] = cst
                return x

        x = torch.zeros((4, 4), dtype=torch.float32)
        ind1 = torch.tensor([1, 2], dtype=torch.int64)
        ind2 = torch.tensor([1, 3], dtype=torch.int64)
        model = Model()
        xs = (x, ind1, ind2)
        expected = model(*xs)
        model_path = self._call_exporter("test_aten_index_put_no_dimension", "custom", model, xs)
        sess = ExtendedReferenceEvaluator(model_path, verbose=0)
        feeds = dict(zip(sess.input_names, [to_numpy(x) for x in xs]))
        got = sess.run(None, feeds)[0]
        self.assertEqualAny(expected.numpy(), got, atol=1e-6)

        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        sess = onnxruntime.InferenceSession(
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, feeds)[0]
        self.assertEqualAny(expected.numpy(), got, atol=1e-5)

    @ignore_warnings(UserWarning)
    def test_aten_index_put_no_dimension_3_2_2(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x, ind1, ind2, values):
                x = x.clone()
                x[ind1, ind2] = values
                return x

        x = torch.zeros((4, 4, 4), dtype=torch.float32)
        ind1 = torch.tensor([1, 2], dtype=torch.int64)
        ind2 = torch.tensor([1, 3], dtype=torch.int64)
        values = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.float32)
        model = Model()
        xs = (x, ind1, ind2, values)
        expected = model(*xs)
        model_path = self._call_exporter("test_aten_index_put_no_dimension", "custom", model, xs)
        sess = ExtendedReferenceEvaluator(model_path, verbose=0)
        feeds = dict(zip(sess.input_names, [to_numpy(x) for x in xs]))
        got = sess.run(None, feeds)[0]
        self.assertEqualAny(expected.numpy(), got, atol=1e-6)

        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        sess = onnxruntime.InferenceSession(
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, feeds)[0]
        self.assertEqualAny(expected.numpy(), got, atol=1e-5)

    def test_getitem_index_put1(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x, value):
                x = x.clone()
                x[:, :, :, : value.shape[-1]] = value
                return x

        inputs = (torch.randn(2, 2, 3, 4), torch.randn(2, 2, 3, 3))
        model = Model()
        expected = model(*inputs)

        # static
        onx = to_onnx(model, inputs, dynamic_shapes=({3: "M"}, {3: "N"}), verbose=0)
        self.dump_onnx("test_getitem_index_put1.onnx", onx)
        feeds = dict(zip(["x", "value"], [x.detach().cpu().numpy() for x in inputs]))
        ref = ExtendedReferenceEvaluator(onx, verbose=0)
        got = ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-2)

        import onnxruntime

        sess = onnxruntime.InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-2)

    def test_getitem_index_put2(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x, value):
                x = x.clone()
                x[:, :, :, 1 : value.shape[-1] + 1] = value
                return x

        inputs = (torch.randn(2, 2, 3, 4), torch.randn(2, 2, 3, 3))
        model = Model()
        expected = model(*inputs)

        onx = to_onnx(model, inputs, dynamic_shapes=({3: "M"}, {3: "N"}), verbose=0)
        self.dump_onnx("test_getitem_index_put2.onnx", onx)
        feeds = dict(zip(["x", "value"], [x.detach().cpu().numpy() for x in inputs]))
        ref = ExtendedReferenceEvaluator(onx, verbose=0)
        got = ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-2)

        import onnxruntime

        sess = onnxruntime.InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-2)

    def test_getitem_index_put3(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x, value):
                x = x.clone()
                x[:, :, 1 : value.shape[-2] + 1, 1 : value.shape[-1] + 1] = value
                return x

        inputs = (torch.randn(2, 2, 4, 4), torch.randn(2, 2, 3, 3))
        model = Model()
        expected = model(*inputs)

        onx = to_onnx(
            model,
            inputs,
            dynamic_shapes=({2: "M1", 3: "M1"}, {2: "N1", 3: "N2"}),
            verbose=0,
        )
        self.dump_onnx("test_getitem_index_put3.onnx", onx)
        feeds = dict(zip(["x", "value"], [x.detach().cpu().numpy() for x in inputs]))
        ref = ExtendedReferenceEvaluator(onx, verbose=0)
        got = ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-2)

        import onnxruntime

        sess = onnxruntime.InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-2)

    @requires_onnx_diagnostic("0.7.16")
    def test_index_Tensor_21_2(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x, ind1, ind2):
                torch._check(x.shape[1] == ind2.shape[0])
                return x[ind1, ind2]

        inputs = (
            torch.randn(2, 1024),
            torch.tensor([[0, 1]], dtype=torch.int64).T,
            torch.arange(1024, dtype=torch.int64),
        )
        model = Model()
        expected = model(*inputs)

        with torch_export_patches(patch_torch=2):
            onx = to_onnx(
                model,
                inputs,
                dynamic_shapes=({0: "A", 1: "B"}, {1: "D"}, {0: "E"}),
                verbose=0,
                export_options=ExportOptions(aten_as_function=set()),
            )
        self.dump_onnx("test_index_Tensor_21_2.onnx", onx)
        feeds = dict(zip(["x", "ind1", "ind2"], [x.detach().cpu().numpy() for x in inputs]))
        ref = ExtendedReferenceEvaluator(onx, verbose=0)
        got = ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-2)

        import onnxruntime

        sess = onnxruntime.InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-2)

    @requires_onnx_diagnostic("0.7.16")
    @hide_stdout()
    def test_index_Tensor_21_2_oblivious(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x, ind1, ind2):
                return x[ind1, ind2]

        inputs = (
            torch.randn(2, 1024),
            torch.tensor([[0, 1]], dtype=torch.int64).T,
            torch.arange(1024, dtype=torch.int64),
        )
        model = Model()
        expected = model(*inputs)

        dynshapes = ({0: "A", 1: "B"}, {0: "C", 1: "D"}, {0: "E"})
        with torch_export_patches(patch_torch=2, verbose=10), torch.fx.experimental._config.patch(
            backed_size_oblivious=True
        ):
            ep = torch.export.export(model, inputs, dynamic_shapes=use_dyn_not_str(dynshapes))
            onx = to_onnx(ep, dynamic_shapes=dynshapes, verbose=0)
        self.dump_onnx("test_index_Tensor_21_2_oblivious.onnx", onx)
        feeds = dict(zip(["x", "ind1", "ind2"], [x.detach().cpu().numpy() for x in inputs]))
        ref = ExtendedReferenceEvaluator(onx, verbose=0)
        got = ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-2)

        import onnxruntime

        sess = onnxruntime.InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-2)

    def test_index_put_dynamic(self):
        import torch

        for dimension in [3, 4, 2]:
            with self.subTest(dimension=dimension):

                class Model(torch.nn.Module):

                    def __init__(self, dimension):
                        super().__init__()
                        self.params = torch.zeros(
                            (4, 5)
                            if dimension == 2
                            else ((2, 4, 5) if dimension == 3 else (1, 1, 4, 5))
                        )
                        self.dimension = dimension

                    def forward(self, update, index1, index2):
                        copy = self.params.clone()
                        if self.dimension == 2:
                            copy[index1, index2] = update
                        elif self.dimension == 3:
                            copy[:, index1, index2] = update
                        else:
                            copy[:, :, index1, index2] = update
                        return copy

                update = (torch.arange(2) + 10).reshape((2,)).to(torch.float32)
                index1 = torch.tensor([1, 2], dtype=torch.int64)
                index2 = torch.tensor([3, 4], dtype=torch.int64)
                feeds = dict(zip(["update", "index1", "index2"], (update, index1, index2)))
                expected = Model(dimension)(**feeds)
                onx = to_onnx(
                    Model(dimension),
                    kwargs=feeds,
                    dynamic_shapes={
                        "update": {0: "dn"},
                        "index1": {0: "dn"},
                        "index2": {0: "dn"},
                    },
                    verbose=0,
                    options=OptimizationOptions(patterns=None, verbose=0),
                )
                self.dump_onnx(f"test_index_put_dynamic_{dimension}.onnx", onx)
                feeds = {k: v.detach().cpu().numpy() for k, v in feeds.items()}
                ref = ExtendedReferenceEvaluator(onx, verbose=0)
                got = ref.run(None, feeds)[0]
                self.assertEqualArray(expected, got, atol=1e-2)

                import onnxruntime

                sess = onnxruntime.InferenceSession(
                    onx.SerializeToString(), providers=["CPUExecutionProvider"]
                )
                got = sess.run(None, feeds)[0]
                self.assertEqualArray(expected, got, atol=1e-2)

    def test_cast_cast_float(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x):
                xb = x.to(torch.bfloat16)
                return (xb + xb).to(torch.float32)

        inputs = (torch.randn(2, 10),)
        model = Model()
        expected = model(*inputs)
        self.assertEqual(expected.dtype, torch.float32)

        onx = to_onnx(
            model,
            inputs,
            dynamic_shapes=({0: "A", 1: "B"},),
            verbose=0,
            options=OptimizationOptions(patterns="default", verbose=0),
        )
        self.dump_onnx("test_cast_cast_float.onnx", onx)
        op_types = [n.op_type for n in onx.graph.node]
        self.assertEqual(["Add", "Cast", "Cast"], op_types)

    def test_cast_cast_int(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x):
                xb = x.to(torch.int32)
                return (xb + xb).to(torch.float32)

        inputs = (torch.randn(2, 10),)
        model = Model()
        expected = model(*inputs)
        self.assertEqual(expected.dtype, torch.float32)

        onx = to_onnx(
            model,
            inputs,
            dynamic_shapes=({0: "A", 1: "B"},),
            verbose=0,
            options=OptimizationOptions(patterns="default", verbose=0),
        )
        self.dump_onnx("test_cast_cast_int.onnx", onx)
        op_types = [n.op_type for n in onx.graph.node]
        self.assertEqual(["Cast", "Add", "Cast"], op_types)

    def test_convolution_valid(self):
        import torch

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    in_channels=1,
                    out_channels=1,
                    kernel_size=3,
                    stride=1,
                    padding=0,
                    bias=False,
                )

            def forward(self, x):
                return self.conv(x)

        inputs = (
            torch.tensor(
                [
                    [
                        [
                            [1.0, 2.0, 3.0, 4.0, 5.0],
                            [6.0, 7.0, 8.0, 9.0, 10.0],
                            [11.0, 12.0, 13.0, 14.0, 15.0],
                            [16.0, 17.0, 18.0, 19.0, 20.0],
                            [21.0, 22.0, 23.0, 24.0, 25.0],
                        ]
                    ]
                ]
            ),
        )
        model = Model()
        expected = model(*inputs)
        self.assertEqual(expected.dtype, torch.float32)

        onx = to_onnx(
            model,
            inputs,
            verbose=0,
            options=OptimizationOptions(patterns="default", verbose=0),
        )
        self.dump_onnx("test_convolution_valid.onnx", onx)
        op_types = [n.op_type for n in onx.graph.node]
        self.assertEqual(["Conv"], op_types)

        feeds = dict(zip(["x"], [x.detach().cpu().numpy() for x in inputs]))
        ref = ExtendedReferenceEvaluator(onx, verbose=0)
        got = ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-2)

        import onnxruntime

        sess = onnxruntime.InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-2)

    def test_repeat_interleave_strategy(self):
        import torch

        rep = 2
        t = torch.arange((2 * 3 * 2), dtype=torch.float32).reshape((2, 3, -1))
        r = torch.repeat_interleave(t, rep, dim=1)
        r2 = (
            t.unsqueeze(2)
            .expand((*t.shape[:2], rep, *t.shape[2:]))
            .reshape((*t.shape[:1], t.shape[1] * rep, *t.shape[2:]))
        )
        self.assertEqualArray(r, r2)

    def test_repeat_interleave_int_1(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.repeat_interleave(x, 3, dim=1)

        inputs = (torch.randn(2, 3),)
        model = Model()
        expected = model(*inputs)
        self.assertEqual(expected.dtype, torch.float32)

        onx = to_onnx(
            model,
            inputs,
            dynamic_shapes=({0: "A", 1: "B"},),
            verbose=0,
            options=OptimizationOptions(patterns="default", verbose=0),
        )
        self.dump_onnx("test_repeat_interleave_int_1.onnx", onx)
        feeds = dict(zip(["x"], [x.detach().cpu().numpy() for x in inputs]))
        ref = ExtendedReferenceEvaluator(onx, verbose=0)
        got = ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-2)

    def test_repeat_interleave_int_2(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.repeat_interleave(x, 3, dim=1)

        inputs = (torch.randn(2, 3, 4),)
        model = Model()
        expected = model(*inputs)
        self.assertEqual(expected.dtype, torch.float32)

        onx = to_onnx(
            model,
            inputs,
            dynamic_shapes=({0: "A", 1: "B"},),
            verbose=0,
            options=OptimizationOptions(patterns="default", verbose=0),
        )
        self.dump_onnx("test_repeat_interleave_int_2.onnx", onx)
        feeds = dict(zip(["x"], [x.detach().cpu().numpy() for x in inputs]))
        ref = ExtendedReferenceEvaluator(onx, verbose=0)
        got = ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-2)

    def test_repeat_interleave_tensor(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x, ind):
                return torch.repeat_interleave(x, ind, dim=0)

        inputs = (
            torch.arange(6, dtype=torch.float32).reshape((2, 3)),
            torch.tensor([1, 2], dtype=torch.int64),
        )
        model = Model()
        expected = model(*inputs)
        self.assertEqual(expected.dtype, torch.float32)

        onx = to_onnx(
            model,
            inputs,
            dynamic_shapes=({0: "A", 1: "B"}, {0: "C"}),
            verbose=0,
            options=OptimizationOptions(patterns="default", verbose=0),
        )
        self.dump_onnx("test_repeat_interleave_tensor.onnx", onx)
        feeds = dict(zip(["x", "ind"], [x.detach().cpu().numpy() for x in inputs]))
        ref = ExtendedReferenceEvaluator(onx, verbose=0)
        got = ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-2)

    def test_repeat_interleave_tensor_3d(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x, ind):
                return torch.repeat_interleave(x, ind, dim=0)

        inputs = (
            torch.arange(12, dtype=torch.float32).reshape((2, 3, 2)),
            torch.tensor([1, 2], dtype=torch.int64),
        )
        model = Model()
        expected = model(*inputs)
        self.assertEqual(expected.dtype, torch.float32)

        onx = to_onnx(
            model,
            inputs,
            dynamic_shapes=({0: "A", 1: "B"}, {0: "C"}),
            verbose=0,
            options=OptimizationOptions(patterns="default", verbose=0),
        )
        self.dump_onnx("test_repeat_interleave_tensor_3d.onnx", onx)
        feeds = dict(zip(["x", "ind"], [x.detach().cpu().numpy() for x in inputs]))
        ref = ExtendedReferenceEvaluator(onx, verbose=0)
        got = ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-2)

    def test_repeat_interleave_tensor_none(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x, ind):
                return torch.repeat_interleave(x, ind)

        inputs = (
            torch.arange(4, dtype=torch.float32).reshape((2, 2)),
            torch.tensor([1, 2, 3, 2], dtype=torch.int64),
        )
        model = Model()
        expected = model(*inputs)
        self.assertEqual(expected.dtype, torch.float32)

        onx = to_onnx(
            model,
            inputs,
            dynamic_shapes=({0: "A", 1: "B"}, {0: "C"}),
            verbose=0,
            options=OptimizationOptions(patterns="default", verbose=0),
        )
        self.dump_onnx("test_repeat_interleave_tensor_none.onnx", onx)
        feeds = dict(zip(["x", "ind"], [x.detach().cpu().numpy() for x in inputs]))
        ref = ExtendedReferenceEvaluator(onx, verbose=0)
        got = ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-2)

    def test_repeat_interleave_tensor_none_decompose(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x, ind):
                return torch.repeat_interleave(x, ind)

        inputs = (
            torch.arange(4, dtype=torch.float32).reshape((2, 2)),
            torch.tensor([1, 2, 3, 2], dtype=torch.int64),
        )
        model = Model()
        expected = model(*inputs)
        self.assertEqual(expected.dtype, torch.float32)

        onx = to_onnx(
            model,
            inputs,
            dynamic_shapes=({0: "A", 1: "B"}, {0: "C"}),
            verbose=0,
            options=OptimizationOptions(patterns="default", verbose=0),
            export_options=ExportOptions(decomposition_table="all"),
        )
        self.dump_onnx("test_repeat_interleave_tensor_none_decompose.onnx", onx)
        feeds = dict(zip(["x", "ind"], [x.detach().cpu().numpy() for x in inputs]))
        ref = ExtendedReferenceEvaluator(onx, verbose=0)
        got = ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-2)

    def test_repeat_interleave_symbolic_tensor(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x, y):
                return torch.repeat_interleave(x, y.shape[1], dim=1) * torch.repeat_interleave(
                    y, x.shape[1], dim=1
                )

        inputs = (
            torch.arange(4, dtype=torch.float32).reshape((2, 2)),
            torch.arange(6, dtype=torch.float32).reshape((2, 3)),
        )
        model = Model()
        expected = model(*inputs)
        self.assertEqual(expected.dtype, torch.float32)

        onx = to_onnx(
            model,
            inputs,
            dynamic_shapes=({0: "A", 1: "B"}, {0: "C"}),
            verbose=0,
            options=OptimizationOptions(patterns="default", verbose=0),
            export_options=ExportOptions(decomposition_table="all"),
        )
        self.dump_onnx("test_repeat_interleave_symbolic_tensor.onnx", onx)
        feeds = dict(zip(["x", "y"], [x.detach().cpu().numpy() for x in inputs]))
        ref = ExtendedReferenceEvaluator(onx, verbose=0)
        got = ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-2)

    def test_attention_scale_dot_product_attention(self):
        import torch

        class ScaledDotProductAttention(torch.nn.Module):
            def forward(self, query, key, value, attn_mask):
                return torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, attn_mask=attn_mask
                )

        model = ScaledDotProductAttention()
        attn_mask = torch.ones(2, 4, 8, 8).bool()
        attn_mask[0, 0, 0, :] = False
        query = torch.randn(2, 4, 8, 16)
        key = torch.randn(2, 4, 8, 16)
        value = torch.randn(2, 4, 8, 16)
        inputs = query, key, value, attn_mask
        expected = model(*inputs)
        self.assertEqual(expected.dtype, torch.float32)

        onx = to_onnx(
            model,
            inputs,
            verbose=0,
            options=OptimizationOptions(patterns="default", verbose=0),
        )
        self.dump_onnx("test_attention_scale_dot_product_attention.onnx", onx)
        feeds = dict(
            zip([i.name for i in onx.graph.input], [x.detach().cpu().numpy() for x in inputs])
        )
        ref = ExtendedReferenceEvaluator(onx, verbose=0)
        got = ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-2)

    def test_index_copy(self):
        import torch

        class Model(torch.nn.Module):
            def __init__(self, dim: int, alpha: float):
                super().__init__()
                self.dim = dim
                self.alpha = alpha

            def forward(self, x, index, source):
                return x.index_copy(self.dim, index, source)

        model = Model(0, 1)
        x = torch.ones(5, 3, dtype=torch.float32)
        t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
        index = torch.tensor([0, 4, 2])
        inputs = (x, index, t)
        expected = model(*inputs)
        self.assertEqual(expected.dtype, torch.float32)

        onx = to_onnx(
            model,
            inputs,
            verbose=0,
            options=OptimizationOptions(patterns="default", verbose=0),
        )
        self.dump_onnx("test_index_copy.onnx", onx)
        feeds = dict(
            zip([i.name for i in onx.graph.input], [x.detach().cpu().numpy() for x in inputs])
        )
        ref = ExtendedReferenceEvaluator(onx, verbose=0)
        got = ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-2)

    def test_index_add_d0(self):
        import torch

        class Model(torch.nn.Module):
            def __init__(self, dim: int, alpha: float):
                super().__init__()
                self.dim = dim
                self.alpha = alpha

            def forward(self, x, index, source):
                return x.index_add(self.dim, index, source, alpha=self.alpha)

        for alpha in (1, 1.34):
            model = Model(0, alpha)
            x = torch.ones(5, 3, dtype=torch.float32)
            t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
            index = torch.tensor([0, 4, 2])
            inputs = (x, index, t)
            expected = model(*inputs)
            self.assertEqual(expected.dtype, torch.float32)

            onx = to_onnx(
                model,
                inputs,
                verbose=0,
                options=OptimizationOptions(patterns="default", verbose=0),
            )
            self.dump_onnx(f"test_index_add_{alpha}.onnx", onx)
            feeds = dict(
                zip(
                    [i.name for i in onx.graph.input],
                    [x.detach().cpu().numpy() for x in inputs],
                )
            )
            ref = ExtendedReferenceEvaluator(onx, verbose=0)
            got = ref.run(None, feeds)[0]
            self.assertEqualArray(expected, got, atol=1e-2)

    def test_index_add_d1(self):
        import torch

        class Model(torch.nn.Module):
            def __init__(self, dim: int, alpha: float):
                super().__init__()
                self.dim = dim
                self.alpha = alpha

            def forward(self, x, index, source):
                return x.index_add(self.dim, index, source, alpha=self.alpha)

        model = Model(1, 1.34)
        x = torch.ones(3, 5, dtype=torch.float32)
        t = torch.arange(12).to(torch.float).reshape((3, -1))
        index = torch.tensor([0, 4, 2, 4])
        inputs = (x, index, t)
        expected = model(*inputs)
        self.assertEqual(expected.dtype, torch.float32)

        onx = to_onnx(
            model,
            inputs,
            verbose=0,
            options=OptimizationOptions(patterns="default", verbose=0),
        )
        self.dump_onnx("test_index_add_d1.onnx", onx)
        feeds = dict(
            zip([i.name for i in onx.graph.input], [x.detach().cpu().numpy() for x in inputs])
        )
        import onnxruntime

        ref = onnxruntime.InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        # ref = ExtendedReferenceEvaluator(onx, verbose=10)
        got = ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-2)

    def test_index_one_hot_d1(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.one_hot(x, 3)

        model = Model()
        inputs = (torch.arange(0, 5, dtype=torch.int64) % 3,)
        expected = model(*inputs)
        self.assertEqual(expected.dtype, torch.int64)

        onx = to_onnx(
            model,
            inputs,
            verbose=0,
            options=OptimizationOptions(patterns="default", verbose=0),
        )
        self.dump_onnx("test_onx_hot_d1.onnx", onx)
        feeds = dict(
            zip([i.name for i in onx.graph.input], [x.detach().cpu().numpy() for x in inputs])
        )
        ref = ExtendedReferenceEvaluator(onx, verbose=0)
        got = ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-2)

    def test_index_one_hot_d2(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.one_hot(x, 3)

        model = Model()
        inputs = (torch.arange(0, 10, dtype=torch.int64).reshape((5, -1)) % 3,)
        expected = model(*inputs)
        self.assertEqual(expected.dtype, torch.int64)

        onx = to_onnx(
            model,
            inputs,
            verbose=0,
            options=OptimizationOptions(patterns="default", verbose=0),
        )
        self.dump_onnx("test_onx_hot_d2.onnx", onx)
        feeds = dict(
            zip([i.name for i in onx.graph.input], [x.detach().cpu().numpy() for x in inputs])
        )
        ref = ExtendedReferenceEvaluator(onx, verbose=0)
        got = ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-2)

    def test_expand(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x):
                return x.expand(-1, 2)

        model = Model()
        inputs = (torch.arange(0, 5, dtype=torch.int64).reshape((-1, 1)) % 3,)
        expected = model(*inputs)
        self.assertEqual(expected.dtype, torch.int64)

        onx = to_onnx(
            model,
            inputs,
            verbose=0,
            options=OptimizationOptions(patterns="default", verbose=0),
        )
        self.dump_onnx("test_expand.onnx", onx)
        feeds = dict(
            zip([i.name for i in onx.graph.input], [x.detach().cpu().numpy() for x in inputs])
        )
        ref = ExtendedReferenceEvaluator(onx, verbose=0)
        got = ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-2)

        import onnxruntime

        ref = onnxruntime.InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-2)

    def test_diff(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.diff(x, n=2, dim=-1)

        model = Model()
        inputs = (torch.arange(18, dtype=torch.int64).reshape((3, 6)) ** 2,)
        expected = model(*inputs)
        self.assertEqual(expected.dtype, torch.int64)

        onx = to_onnx(
            model,
            inputs,
            verbose=0,
            options=OptimizationOptions(patterns="default", verbose=0),
        )
        self.dump_onnx("test_diff.onnx", onx)
        feeds = dict(
            zip([i.name for i in onx.graph.input], [x.detach().cpu().numpy() for x in inputs])
        )
        ref = ExtendedReferenceEvaluator(onx, verbose=0)
        got = ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-2)

        import onnxruntime

        ref = onnxruntime.InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-2)

    def test_diff_prepend(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x, prepend):
                return torch.diff(x, n=2, dim=-1, prepend=prepend)

        model = Model()
        inputs = (
            torch.arange(18, dtype=torch.int64).reshape((3, 6)) ** 2,
            torch.arange(6, dtype=torch.int64).reshape((3, 2)) ** 2,
        )
        expected = model(*inputs)
        self.assertEqual(expected.dtype, torch.int64)

        onx = to_onnx(
            model,
            inputs,
            verbose=0,
            options=OptimizationOptions(patterns="default", verbose=0),
        )
        self.dump_onnx("test_diff_prepend.onnx", onx)
        feeds = dict(
            zip([i.name for i in onx.graph.input], [x.detach().cpu().numpy() for x in inputs])
        )
        ref = ExtendedReferenceEvaluator(onx, verbose=0)
        got = ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-2)

        import onnxruntime

        ref = onnxruntime.InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-2)

    def test_diff_dynamic(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.diff(x, n=2, dim=-1)

        model = Model()
        inputs = (torch.arange(18, dtype=torch.int64).reshape((3, 6)) ** 2,)
        expected = model(*inputs)
        self.assertEqual(expected.dtype, torch.int64)

        onx = to_onnx(
            model,
            inputs,
            verbose=0,
            dynamic_shapes=({1: torch.export.Dim.DYNAMIC},),
            options=OptimizationOptions(patterns="default", verbose=0),
        )
        self.dump_onnx("test_diff_dynamic.onnx", onx)
        feeds = dict(
            zip([i.name for i in onx.graph.input], [x.detach().cpu().numpy() for x in inputs])
        )
        ref = ExtendedReferenceEvaluator(onx, verbose=0)
        got = ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-2)

        import onnxruntime

        ref = onnxruntime.InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-2)

    def test_aten_index_put_mixed_dimensions(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x, index, update):
                copy = x.clone()
                copy[:, index] = update
                return copy

        model = Model()
        x = (torch.arange(12) + 10).reshape((1, 3, 4)).to(torch.float32)
        index = torch.tensor([0, 1], dtype=torch.int64)
        update = (torch.arange(8) + 10).reshape((1, 2, 4)).to(torch.float32)
        expected = model(x, index, update)
        model_path = self._call_exporter(
            "test_aten_index_put_mixed_dimensions",
            "custom",
            model,
            (x, index, update),
            strict=False,
            dynamic_shapes=({1: "b", 2: "c"}, {0: "d"}, {1: "f", 2: "g"}),
        )
        check_model(model_path)

        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        sess = onnxruntime.InferenceSession(
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        feeds = dict(
            zip([i.name for i in sess.get_inputs()], [x.numpy(), index.numpy(), update.numpy()])
        )
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_aten_index_copy_0(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x, indices, update):
                return x.index_copy(0, indices, update)

        model = Model()
        inputs = (
            torch.zeros((5, 3), dtype=torch.float32),
            torch.tensor([0, 4, 2, 1], dtype=torch.int64),
            torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=torch.float32),
        )
        expected = model(*inputs)
        DYN = torch.export.Dim.DYNAMIC
        model_path = self._call_exporter(
            "test_aten_index_copy_0",
            "custom",
            model,
            inputs,
            strict=False,
            dynamic_shapes=({0: DYN, 1: DYN}, {0: DYN}, {0: DYN, 1: DYN}),
        )
        check_model(model_path)

        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        sess = onnxruntime.InferenceSession(
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        feeds = dict(zip([i.name for i in sess.get_inputs()], [i.numpy() for i in inputs]))
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_aten_index_copy_1(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x, indices, update):
                return x.index_copy(1, indices, update)

        model = Model()
        inputs = (
            torch.zeros((5, 7), dtype=torch.float32),
            torch.tensor([0, 4, 2, 1], dtype=torch.int64),
            torch.arange(20, dtype=torch.float32).reshape((5, 4)),
        )
        expected = model(*inputs)
        DYN = torch.export.Dim.DYNAMIC
        model_path = self._call_exporter(
            "test_aten_index_copy_1",
            "custom",
            model,
            inputs,
            strict=False,
            dynamic_shapes=({0: DYN, 1: DYN}, {0: DYN}, {0: DYN, 1: DYN}),
        )
        check_model(model_path)

        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        sess = onnxruntime.InferenceSession(
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        feeds = dict(zip([i.name for i in sess.get_inputs()], [i.numpy() for i in inputs]))
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_aten_index_copy_1_3d(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x, indices, update):
                return x.index_copy(1, indices, update)

        model = Model()
        inputs = (
            torch.zeros((5, 7, 3), dtype=torch.float32),
            torch.tensor([0, 4, 2, 1], dtype=torch.int64),
            torch.arange(60, dtype=torch.float32).reshape((5, 4, 3)),
        )
        expected = model(*inputs)
        DYN = torch.export.Dim.DYNAMIC
        model_path = self._call_exporter(
            "test_aten_index_copy_1_3d",
            "custom",
            model,
            inputs,
            strict=False,
            dynamic_shapes=({0: DYN, 1: DYN}, {0: DYN}, {0: DYN, 1: DYN}),
        )
        check_model(model_path)

        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        sess = onnxruntime.InferenceSession(
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        feeds = dict(zip([i.name for i in sess.get_inputs()], [i.numpy() for i in inputs]))
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    @skipif_ci_windows("broken")
    def test_aten_index_put_3d_none(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x, index, update):
                x = x.clone()
                return torch.ops.aten.index_put(x, [None, index, None], update)

        class Model2(torch.nn.Module):
            def forward(self, x, index, update):
                x = x.clone()
                return torch.ops.aten.index_put(
                    x.transpose(1, 0), [index, None, None], update.transpose(1, 0)
                ).transpose(1, 0)

        model = Model()
        shape = (2, 3, 2)
        N = int(np.prod(shape))
        x = torch.arange(N, dtype=torch.float32).reshape(shape)
        update = (torch.arange(N, dtype=torch.float32).reshape(shape) + 1) * 100
        index = ((torch.arange(shape[-2])).to(torch.int64) + 1) % shape[-2]
        expected = model(x, index, update)
        expected2 = Model2()(x, index, update)
        self.assertEqualArray(expected, expected2)
        DYN = torch.export.Dim.DYNAMIC
        ds_ = [
            ({0: DYN, 1: DYN, 2: DYN}, {0: DYN}, {0: DYN, 1: DYN, 2: DYN}),
            ({0: DYN, 1: DYN}, {0: DYN}, {0: DYN, 1: DYN}),
        ]
        for ii, ds in enumerate(ds_):
            onx = to_onnx(
                model,
                (x, index, update),
                dynamic_shapes=ds,
                export_options=ExportOptions(aten_as_function=set()),
            )
            self.dump_onnx(f"test_aten_index_put_3d_none_{ii}.onnx", onx)

            import onnxruntime

            sess_options = onnxruntime.SessionOptions()
            sess = onnxruntime.InferenceSession(
                onx.SerializeToString(),
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )
            feeds = dict(
                zip(
                    [i.name for i in sess.get_inputs()],
                    [x.numpy(), index.numpy(), update.numpy()],
                )
            )
            got = sess.run(None, feeds)[0]
            self.assertEqualArray(expected, got)

    @skipif_ci_windows("broken")
    def test_aten_index_put_4d_none(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x, index, update):
                x = x.clone()
                return torch.ops.aten.index_put(x, [None, None, index, None], update)

        model = Model()
        shape = (2, 1, 3, 2)
        N = int(np.prod(shape))
        x = torch.arange(N, dtype=torch.float32).reshape(shape)
        update = (torch.arange(N, dtype=torch.float32).reshape(shape) + 1) * 100
        index = torch.arange(shape[-2], dtype=torch.int64)
        expected = model(x, index, update)
        DYN = torch.export.Dim.DYNAMIC
        ds = ({0: DYN, 2: DYN}, {0: DYN}, {0: DYN, 2: DYN})
        onx = to_onnx(
            model,
            (x, index, update),
            dynamic_shapes=ds,
            export_options=ExportOptions(aten_as_function=set()),
        )
        self.dump_onnx("test_aten_index_put_4d_none.onnx", onx)

        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        sess = onnxruntime.InferenceSession(
            onx.SerializeToString(), sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        feeds = dict(
            zip([i.name for i in sess.get_inputs()], [x.numpy(), index.numpy(), update.numpy()])
        )
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    @skipif_ci_windows("broken")
    def test_aten_index_put_55_2_25(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x, index, update):
                return torch.ops.aten.index_put(x, [index], update)

        model = Model()
        x = torch.zeros((5, 5), dtype=torch.float32)
        index = torch.tensor([2, 1], dtype=torch.int64)
        update = (torch.arange(10) + 10).reshape((2, -1)).to(torch.float32)
        expected = model(x, index, update)
        DYN = torch.export.Dim.DYNAMIC
        onx = to_onnx(
            model,
            (x, index, update),
            dynamic_shapes=({0: DYN, 1: DYN}, {0: DYN}, {0: DYN, 1: DYN}),
            export_options=ExportOptions(aten_as_function=set()),
        )
        self.dump_onnx("test_aten_index_put_55_2_25.onnx", onx)

        import onnxruntime

        sess = onnxruntime.InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        feeds = dict(
            zip([i.name for i in sess.get_inputs()], [x.numpy(), index.numpy(), update.numpy()])
        )
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    @skipif_ci_windows("broken")
    def test_aten_index_put_55_12_25(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x, index, update):
                return torch.ops.aten.index_put(x, [index], update)

        model = Model()
        x = torch.zeros((6, 5), dtype=torch.float32)
        index = torch.tensor([[2, 1]], dtype=torch.int64)
        update = (torch.arange(10) + 10).reshape((2, -1)).to(torch.float32)
        expected = model(x, index, update)
        DYN = torch.export.Dim.DYNAMIC
        onx = to_onnx(
            model,
            (x, index, update),
            dynamic_shapes=({0: DYN, 1: DYN}, {1: DYN}, {0: DYN, 1: DYN}),
            export_options=ExportOptions(aten_as_function=set()),
        )
        self.dump_onnx("test_aten_index_put_55_12_25.onnx", onx)

        import onnxruntime

        sess = onnxruntime.InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        feeds = dict(
            zip([i.name for i in sess.get_inputs()], [x.numpy(), index.numpy(), update.numpy()])
        )
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    def test_aten_inplace_add(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x):
                x[:2, :3] = 1
                return x + 2

        model = Model()
        x = torch.zeros((6, 5), dtype=torch.float32)
        expected = model(x)
        DYN = torch.export.Dim.DYNAMIC
        onx = to_onnx(model, (x,), dynamic_shapes=({0: DYN, 1: DYN},))
        self.dump_onnx("test_aten_inplace_add.onnx", onx)
        self.assert_conversion_with_ort_on_cpu(onx, (expected,), (x,))

    def test_aten_flip(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.flip(x, [-1]) + 1

        model = Model()
        x = torch.zeros((6, 5), dtype=torch.float32)
        expected = model(x)
        self.assertEqual(x.shape, expected.shape)
        DYN = torch.export.Dim.DYNAMIC
        onx = to_onnx(model, (x,), dynamic_shapes=({0: DYN, 1: DYN},))
        self.dump_onnx("test_aten_flip.onnx", onx)
        self.assert_conversion_with_ort_on_cpu(onx, (expected,), (x,))

    @unittest.skip("unbind not ready yet")
    def test_aten_unbind(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x):
                u = torch.unbind(x, dim=1)
                return torch.cat(u, dim=0)

        model = Model()
        x = torch.zeros((6, 5), dtype=torch.float32)
        expected = model(x)
        DYN = torch.export.Dim.DYNAMIC
        onx = to_onnx(model, (x,), dynamic_shapes=({0: DYN, 1: DYN},))
        self.dump_onnx("test_aten_unbind.onnx", onx)
        self.assert_conversion_with_ort_on_cpu(onx, (expected,), (x,))

    def test_aten_unique_consecutive(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.unique_consecutive(x)

        model = Model()
        x = torch.tensor([0, 1, 2, 2, 3, 3, 0, 0], dtype=torch.int64)
        expected = model(x)
        onx = to_onnx(
            model,
            (x,),
            dynamic_shapes=({0: "length"},),
            export_options=ExportOptions(
                save_ep=self.get_dump_file("test_aten_unique_consecutive.ep")
            ),
        )
        self.dump_onnx("test_aten_unique_consecutive.onnx", onx)
        self.assert_conversion_with_ort_on_cpu(onx, (expected,), (x,))

    def test_aten_unique_consecutive_int32(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.unique_consecutive(x)

        model = Model()
        x = torch.tensor([0, 1, 2, 2, 3, 3, 0, 0], dtype=torch.int32)
        expected = model(x)
        onx = to_onnx(
            model,
            (x,),
            dynamic_shapes=({0: "length"},),
            export_options=ExportOptions(
                save_ep=self.get_dump_file("test_aten_unique_consecutive_32.ep")
            ),
        )
        self.dump_onnx("test_aten_unique_consecutive_32.onnx", onx)
        self.assert_conversion_with_ort_on_cpu(onx, (expected,), (x,))

    def test_aten_unique_consecutive_return(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.unique_consecutive(x, return_inverse=True, return_counts=True)

        model = Model()
        x = torch.tensor([0, 1, 2, 2, 3, 3, 3, 0, 0], dtype=torch.int64)
        expected = model(x)
        onx = to_onnx(
            model,
            (x,),
            dynamic_shapes=({0: "length"},),
            export_options=ExportOptions(
                save_ep=self.get_dump_file("test_aten_unique_consecutive_return.ep")
            ),
        )
        self.dump_onnx("test_aten_unique_consecutive_return.onnx", onx)
        self.assert_conversion_with_ort_on_cpu(onx, expected, (x,))

    def test_aten_unique_consecutive_return_32(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.unique_consecutive(x, return_inverse=True, return_counts=True)

        model = Model()
        x = torch.tensor([0, 1, 2, 2, 3, 3, 3, 0, 0], dtype=torch.int32)
        # type in fx graph differs from one we can see here
        expected = tuple(t.to(torch.int32) for t in model(x))
        onx = to_onnx(
            model,
            (x,),
            dynamic_shapes=({0: "length"},),
            export_options=ExportOptions(
                save_ep=self.get_dump_file("test_aten_unique_consecutive_return_32.ep")
            ),
        )
        self.dump_onnx("test_aten_unique_consecutive_return_32.onnx", onx)
        self.assert_conversion_with_ort_on_cpu(onx, expected, (x,))


if __name__ == "__main__":
    unittest.main(verbosity=2)
