import itertools
import unittest
import numpy as np
import onnx
from onnx.checker import check_model
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    requires_cuda,
    hide_stdout,
    ignore_warnings,
    requires_torch,
    requires_onnxscript,
    requires_onnxruntime_training,
    skipif_ci_windows,
    has_torch,
)
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.torch_interpreter import to_onnx, ExportOptions


class TestIssuesPytorch2024(ExtTestCase):

    @ignore_warnings((UserWarning, FutureWarning))
    @requires_onnxruntime_training(ortmodule=True)
    def test_dort(self):
        import torch

        def _make_aot_ort(dynamic: bool = False) -> tuple:
            from torch.onnx import (
                _OrtBackend as OrtBackend,
                _OrtBackendOptions as OrtBackendOptions,
                ExportOptions,
            )

            export_options = ExportOptions(dynamic_shapes=dynamic)
            options = OrtBackendOptions(export_options=export_options)
            ort_backend = OrtBackend(options=options)
            return ort_backend

        class Linear(torch.nn.Module):
            def __init__(self):
                super(Linear, self).__init__()
                self.linear = torch.nn.Linear(128, 10)
                self.activation = torch.nn.ReLU()

            def forward(self, *inputs):
                input = self.linear(inputs[0])
                input = self.activation(input)
                return input

        model = Linear()
        model.train()
        loss_fn = torch.nn.MSELoss()

        input = torch.randn((64, 128), requires_grad=True)
        labels = torch.randn((64, 10), requires_grad=True)

        compiled_model = torch.compile(model, backend=_make_aot_ort())
        output = compiled_model(*input)
        loss = loss_fn(output, labels)
        loss.backward()

    @ignore_warnings((UserWarning, DeprecationWarning))
    @requires_onnxruntime_training()
    def test_cort(self):
        import torch
        from torch._dynamo.backends.common import aot_autograd
        from experimental_experiment.torch_dynamo import onnx_custom_backend

        backend_onnx = lambda *args, **kwargs: onnx_custom_backend(  # noqa: E731
            *args, target_opset=18, verbose=0, **kwargs
        )

        class Linear(torch.nn.Module):
            def __init__(self):
                super(Linear, self).__init__()
                self.linear = torch.nn.Linear(128, 10)
                self.activation = torch.nn.ReLU()

            def forward(self, *inputs):
                input = self.linear(inputs[0])
                input = self.activation(input)
                return input

        model = Linear()
        model.train()
        loss_fn = torch.nn.MSELoss()

        x = torch.randn((64, 128), requires_grad=True)
        labels = torch.randn((64, 10), requires_grad=True)

        aot_compiler = aot_autograd(fw_compiler=backend_onnx)
        compiled_model = torch.compile(model, backend=aot_compiler)
        output = compiled_model(x)
        loss = loss_fn(output, labels)
        loss.backward()

    @ignore_warnings((DeprecationWarning, UserWarning))
    @hide_stdout()
    @requires_onnxscript("0.2")
    @requires_torch("2.6")
    def test_export_set_dynamo(self):
        import torch
        import onnxruntime as rt
        from torch import nn

        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x):
                mask = torch.ones((1, 3, 3), dtype=bool)
                x[mask] = 0
                return x

        model = Model()
        input_tensor = torch.randn((1, 3, 3))
        expected = model(input_tensor)
        onnx_program = torch.onnx.export(model, input_tensor, dynamo=True)
        session = rt.InferenceSession(
            onnx_program.model_proto.SerializeToString(),
            providers=["CPUExecutionProvider"],
        )
        feeds = {onnx_program.model_proto.graph.input[0].name: input_tensor.cpu().numpy()}
        try:
            results = session.run(None, feeds)
        except ValueError as e:
            if "are missing from input feed" in str(e):
                raise unittest.SkipTest(f"bug in dynamo exporter: {e}")
        self.assertEqualArray(expected, results[0])

    @ignore_warnings((DeprecationWarning, UserWarning))
    @requires_torch("2.4")
    def test_export_set_custom(self):
        import torch
        import onnxruntime as rt
        from torch import nn
        from experimental_experiment.torch_interpreter import to_onnx

        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x):
                mask = torch.ones((1, 3, 3), dtype=bool)
                x[mask] = 2
                return x

        model = Model()
        input_tensor = torch.randn((1, 3, 3))
        expected = model(input_tensor)
        onx = to_onnx(model, (input_tensor,), verbose=0, optimize=False)
        session = rt.InferenceSession(
            onx.SerializeToString(),
            providers=["CPUExecutionProvider"],
        )
        results = session.run(None, {"x": input_tensor.cpu().numpy()})
        self.assertEqualArray(expected, results[0])

    def _updated_parameter(self, exporter, d3, decomposition=False, dynamic=False):
        # https://github.com/pytorch/pytorch/issues/135233

        import torch

        if d3:

            class UpdateModel(torch.nn.Module):

                def __init__(self):
                    super().__init__()
                    self.params = torch.zeros((2, 1, 10))

                def forward(self, update: torch.Tensor, kv_index: torch.LongTensor):
                    indices = torch.arange(update.shape[0])
                    middle = torch.zeros((1,), dtype=torch.long)
                    copy = self.params.clone()
                    copy[kv_index, middle, indices] = update.transpose(1, 0)
                    return copy

            model = UpdateModel()

        else:

            class UpdateModel(torch.nn.Module):

                def __init__(self):
                    super().__init__()
                    self.params = torch.zeros((2, 10))

                def forward(self, update: torch.Tensor, kv_index: torch.LongTensor):
                    indices = torch.arange(update.shape[0])
                    copy = self.params.clone()
                    copy[kv_index, indices] = update.transpose(1, 0)
                    return copy

            model = UpdateModel()

        n = 6
        update = torch.arange(n).reshape((n, 1)).to(torch.float32)
        kv_index = torch.tensor([0])
        model(update, kv_index)

        model_path = (
            f"test_updated_cache_{exporter}_d{3 if d3 else 2}"
            f"D{1 if decomposition else 0}-{'dyn' if dynamic else 'sta'}.onnx"
        )

        if exporter == "script":
            torch.onnx.export(
                model,
                (update, kv_index),
                model_path,
                input_names=["update", "kv_index"],
                output_names=["updated"],
                dynamic_axes={"update": {0: "n"}} if dynamic else None,
                opset_version=13,
                verbose=False,
            )
        elif exporter == "dynamo":
            torch.onnx.export(
                model,
                (update, kv_index),
                model_path,
                input_names=["update", "kv_index"],
                output_names=["updated"],
                dynamic_shapes=(
                    {"update": {0: torch.export.Dim("n")}, "kv_index": {}} if dynamic else None
                ),
                verbose=False,
                dynamo=True,
            )
        else:
            export_options = ExportOptions(
                decomposition_table="all" if decomposition else None, strict=True
            )
            to_onnx(
                model,
                (update, kv_index),
                filename=model_path,
                export_options=export_options,
                dynamic_shapes=(
                    {"update": {0: torch.export.Dim("n")}, "kv_index": {}} if dynamic else None
                ),
                verbose=0,
                optimize=True,
            )

        check_model(model_path)

        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        session = onnxruntime.InferenceSession(
            model_path, sess_options=sess_options, providers=[("CPUExecutionProvider")]
        )

        def gen_numpy_inputs(n: int, idx: int):
            return {
                "update": np.arange(n).reshape((n, 1)).astype(np.float32),
                "kv_index": np.array([idx], dtype=np.int64),
            }

        input_n = gen_numpy_inputs(n, 0)
        expected = model(
            torch.Tensor(input_n["update"]), torch.Tensor(input_n["kv_index"]).to(int)
        )
        # _ = ExtendedReferenceEvaluator(model_path, verbose=10).run(None, input_n)
        # self.assertEqualArray(expected, _[0])
        e1 = session.run(None, input_n)[0]
        self.assertEqualArray(expected, e1)

        input_2 = gen_numpy_inputs(2, 0)
        expected = model(
            torch.Tensor(input_2["update"]), torch.Tensor(input_2["kv_index"]).to(int)
        )
        if dynamic:
            # ExtendedReferenceEvaluator(model_path, verbose=10).run(None, input_2)
            e2 = session.run(None, input_2)[0]
            self.assertEqualArray(expected, e2)

    @unittest.skip("index_put fails with torch_script")
    def test_index_put_update_parameter_script_2d(self):
        self._updated_parameter("script", False)

    @unittest.skip("index_put fails with torch_script")
    def test_index_put_update_parameter_script_3d(self):
        self._updated_parameter("script", True)

    @requires_onnxscript("0.2")
    @hide_stdout()
    def test_index_put_update_parameter_dynamo_2d_static(self):
        self._updated_parameter("dynamo", False, dynamic=False)

    @requires_onnxscript("0.2")
    @hide_stdout()
    def test_index_put_update_parameter_dynamo_2d_dynamic(self):
        self._updated_parameter("dynamo", False, dynamic=True)

    @requires_onnxscript("0.2")
    @hide_stdout()
    def test_index_put_update_parameter_dynamo_3d(self):
        self._updated_parameter("dynamo", True)

    def test_index_put_update_parameter_custom_2d_static(self):
        self._updated_parameter("custom", False, dynamic=False)

    def test_index_put_update_parameter_custom_2d_dynamic(self):
        self._updated_parameter("custom", False, dynamic=True)

    @ignore_warnings(UserWarning)
    def test_index_put_update_parameter_custom_2d_nodec(self):
        self._updated_parameter("custom", False, decomposition=False)

    @ignore_warnings(UserWarning)
    def test_index_put_update_parameter_custom_2d_dec(self):
        self._updated_parameter("custom", False, decomposition=True)

    @skipif_ci_windows("not working on Windows")
    def test_index_put_update_parameter_custom_3d_static(self):
        self._updated_parameter("custom", True, dynamic=False)

    def test_index_put_update_parameter_custom_3d_dynamic(self):
        self._updated_parameter("custom", True, dynamic=True)

    def _scaled_dot_product_attention(self, exporter):
        # https://github.com/pytorch/pytorch/issues/135615

        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class ScaledDotProductAttentionModel(nn.Module):
            def __init__(self, d_model, scale):
                super(ScaledDotProductAttentionModel, self).__init__()
                self.scale = scale  # scaling factor for attention scores
                self.d_model = d_model  # dimensionality of input embeddings
                self.query_linear = nn.Linear(d_model, d_model)
                self.key_linear = nn.Linear(d_model, d_model)
                self.value_linear = nn.Linear(d_model, d_model)

            def forward(self, query_states, key_states, value_states):
                # Project the input states
                query = self.query_linear(query_states)
                key = self.key_linear(key_states)
                value = self.value_linear(value_states)

                # Perform scaled dot product attention
                attn_output = F.scaled_dot_product_attention(
                    query, key, value, scale=self.scale
                )
                return attn_output

        d_model = 64
        scale = 1.0 / (d_model**0.5)
        model = ScaledDotProductAttentionModel(d_model, scale)

        batch_size = 2
        seq_length_q = 10  # length of query
        seq_length_kv = 15  # length of key and value
        embedding_dim = d_model

        query_states = torch.randn(batch_size, seq_length_q, embedding_dim)
        key_states = torch.randn(batch_size, seq_length_kv, embedding_dim)
        value_states = torch.randn(batch_size, seq_length_kv, embedding_dim)
        expected_output = model(query_states.clone(), key_states.clone(), value_states.clone())

        onnx_file_path = f"test_scaled_dot_product_attention_{exporter}.onnx"

        if exporter == "script":
            torch.onnx.export(
                model,
                (query_states, key_states, value_states),
                onnx_file_path,
                input_names=[
                    "query_states",
                    "key_states",
                    "value_states",
                ],
                output_names=["attn_output"],
                opset_version=13,
            )
        elif exporter == "dynamo":
            torch.onnx.export(
                model,
                (query_states, key_states, value_states),
                onnx_file_path,
                input_names=[
                    "query_states",
                    "key_states",
                    "value_states",
                ],
                output_names=["attn_output"],
                dynamo=True,
            )
        else:
            to_onnx(model, (query_states, key_states, value_states), filename=onnx_file_path)

        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        # sess_options.graph_optimization_level, sess_options.optimized_model_filepath = (
        #    onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL,
        #    f"test_scaled_dot_product_attention_{exporter}.opt.onnx",
        # )
        session = onnxruntime.InferenceSession(
            onnx_file_path,
            sess_options=sess_options,
            providers=[("CPUExecutionProvider")],
        )
        inputs_names = [i.name for i in session.get_inputs()]
        feeds = dict(
            zip(inputs_names, (query_states.numpy(), key_states.numpy(), value_states.numpy()))
        )
        # got = ExtendedReferenceEvaluator(onnx_file_path, verbose=10).run(None, feeds)
        # self.assertEqualArray(expected_output, got[0], atol=1e-5)
        output = session.run(None, feeds)
        self.assertEqualArray(expected_output, output[0], atol=1e-5)

    @unittest.skip("not implemented")
    def test_scaled_dot_product_attention_script(self):
        self._scaled_dot_product_attention("script")

    @hide_stdout()
    def test_scaled_dot_product_attention_dynamo(self):
        self._scaled_dot_product_attention("dynamo")

    def test_scaled_dot_product_attention_custom(self):
        self._scaled_dot_product_attention("custom")

    def _in_projection_packed(self, exporter):
        # https://github.com/pytorch/pytorch/issues/135615

        import torch
        import torch.nn.functional as F

        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.w = torch.nn.Parameter(torch.randn(231, 77))
                self.b = torch.nn.Parameter(torch.randn(231))

            def forward(self, x):
                q, k, v = x, x, x
                q, k, v = F._in_projection_packed(q, k, v, self.w, self.b)
                return q + k + v

        model = SimpleModel()

        example_input = torch.randint(0, 11, (1, 77), dtype=torch.float32)
        model(example_input)

        onnx_file_path = f"test__in_projection_packed_{exporter}.onnx"

        if exporter == "script":
            torch.onnx.export(
                model,
                (example_input,),
                onnx_file_path,
                input_names=["input"],
                output_names=["output"],
                opset_version=18,
            )
        elif exporter == "dynamo":
            torch.onnx.export(
                model,
                (example_input,),
                onnx_file_path,
                input_names=["input"],
                output_names=["output"],
                dynamo=True,
            )
        else:
            to_onnx(model, (example_input,), filename=onnx_file_path, verbose=0)

        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        session = onnxruntime.InferenceSession(
            onnx_file_path,
            sess_options=sess_options,
            providers=[("CPUExecutionProvider")],
        )
        inputs_names = [i.name for i in session.get_inputs()]
        output = session.run(None, dict(zip(inputs_names, (example_input.numpy(),))))
        expected_output = model(example_input)
        self.assertEqualArray(expected_output, output[0], atol=1e-3)

    def test_in_projection_packed_script(self):
        self._in_projection_packed("script")

    @hide_stdout()
    def test_in_projection_packed_dynamo(self):
        self._in_projection_packed("dynamo")

    def test_in_projection_packed_custom(self):
        self._in_projection_packed("custom")

    def _flash_attn(self, exporter):
        # https://github.com/pytorch/pytorch/issues/135645

        import torch

        try:
            from flash_attn.flash_attn_interface import flash_attn_func
        except ImportError as e:
            raise unittest.SkipTest(f"flash_attn not installed {e}")

        class FlashAttention(torch.nn.Module):
            def __init__(self, softmax_scale=None):
                super().__init__()
                self.softmax_scale = softmax_scale

            def forward(self, qkv):
                q = qkv[:, :, 0, ...]  # torch.Size([9, 1025, 16, 64])
                k = qkv[:, :, 1, ...]
                v = qkv[:, :, 2, ...]

                output = flash_attn_func(q, k, v, softmax_scale=self.softmax_scale)
                return output

        example_input = torch.ones((9, 1025, 3, 16, 64)).to(torch.float16).cuda()
        softmax_scale = example_input.shape[-1] ** (-0.5)
        model = FlashAttention(softmax_scale).cuda().eval()
        model(example_input)
        onnx_file_path = f"test_flash_attn_{exporter}.onnx"

        # with torch.no_grad():
        if exporter == "script":
            torch.onnx.export(
                model,
                (example_input,),
                onnx_file_path,
                input_names=["input0"],
                output_names=["qkv_out"],
                opset_version=11,
            )
        elif exporter == "dynamo":
            torch.onnx.export(
                model,
                (example_input,),
                onnx_file_path,
                input_names=["input0"],
                output_names=["qkv_out"],
                dynamo=True,
            )
        else:
            to_onnx(model, (example_input,), filename=onnx_file_path)

        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        session = onnxruntime.InferenceSession(
            onnx_file_path,
            sess_options=sess_options,
            providers=[("CPUExecutionProvider")],
        )
        inputs_names = [i.name for i in session.get_inputs()]
        output = session.run(None, dict(zip(inputs_names, (example_input.numpy(),))))
        expected_output = model(example_input)
        self.assertEqual(expected_output.shape, output[0].shape)
        self.assertEqualArray(expected_output, output[0], atol=1e-4)

    @requires_cuda()
    def test__flash_attn_script(self):
        self._flash_attn("script")

    @requires_cuda()
    @hide_stdout()
    def test__flash_attn_dynamo(self):
        self._flash_attn("dynamo")

    @requires_cuda()
    def test__flash_attn_custom(self):
        self._flash_attn("custom")

    def _complex_weights(self, exporter):
        # https://github.com/onnx/onnx/issues/6388

        import torch

        class CLinear(torch.nn.Module):
            def __init__(self, in_features, out_features, bias=True):
                super().__init__()
                self.weight = torch.nn.Parameter(
                    torch.zeros(out_features, in_features, dtype=torch.complex64)
                )
                self.bias = torch.nn.Parameter(
                    torch.zeros(1, out_features, dtype=torch.complex64), requires_grad=bias
                )

                torch.nn.init.xavier_uniform_(self.weight)
                if bias:
                    torch.nn.init.xavier_uniform_(self.bias)

            def forward(self, inp):
                return torch.matmul(inp, self.weight.T) + self.bias

        example_input = torch.ones((2, 10)).to(torch.complex64)
        model = CLinear(10, 5).eval()
        model(example_input)
        onnx_file_path = f"test_complex_weights_{exporter}.onnx"

        # with torch.no_grad():
        if exporter == "script":
            torch.onnx.export(
                model,
                (example_input,),
                onnx_file_path,
                input_names=["input0"],
                output_names=["out"],
                opset_version=18,
            )
        elif exporter == "dynamo":
            torch.onnx.export(
                model,
                (example_input,),
                onnx_file_path,
                input_names=["input0"],
                output_names=["out"],
                dynamo=True,
            )
        else:
            onx, builder = to_onnx(
                model, (example_input,), filename=onnx_file_path, return_builder=True
            )
            self.assertNotEmpty(builder._parameter_renaming)
            self.assertEqual(len(onx.graph.initializer), 2)

        expected_output = model(example_input)

        ref = ExtendedReferenceEvaluator(onnx_file_path)
        inputs_names = [i.name for i in ref.proto_.graph.input]
        feeds = dict(zip(inputs_names, (example_input.numpy(),)))
        output = ref.run(None, feeds)
        self.assertEqual(expected_output.shape, output[0].shape)
        self.assertEqualArray(expected_output, output[0], atol=1e-4)

    @unittest.skip("torch_script not supported on complex numbers")
    def test_complex_weights_script(self):
        self._complex_weights("script")

    @requires_onnxscript("0.3")
    @hide_stdout()
    def test_complex_weights_dynamo(self):
        self._complex_weights("dynamo")

    def test_complex_weights_custom(self):
        self._complex_weights("custom")

    def _slice_4d(self, exporter):
        # https://github.com/onnx/onnx/issues/6420

        import torch

        class DummySlice(torch.nn.Module):
            def forward(self, x):
                x1 = x[:, :, 0:1, :]
                x2 = x[:, :, 1:2, :]
                return x1 + x2

        example_input = torch.ones((3, 4, 2, 5)).to(torch.float16)
        model = DummySlice().eval()
        model(example_input)
        onnx_file_path = f"test_slice_4d_{exporter}.onnx"

        # with torch.no_grad():
        if exporter == "script":
            torch.onnx.export(
                model,
                (example_input,),
                onnx_file_path,
                input_names=["input0"],
                output_names=["output"],
                opset_version=13,
                dynamic_axes={"input0": {0: "batch_size"}},
                do_constant_folding=True,
                export_params=True,
            )
        elif exporter == "dynamo":
            batch = torch.export.Dim("batch", min=1, max=1024)
            torch.onnx.export(
                model,
                (example_input,),
                onnx_file_path,
                input_names=["input"],
                output_names=["output"],
                dynamo=True,
                dynamic_shapes={"input": {0: batch}},
                fallback=False,
            )
        else:
            batch = torch.export.Dim("batch", min=1, max=1024)
            to_onnx(
                model,
                (example_input,),
                filename=onnx_file_path,
                dynamic_shapes={"x": {0: batch}},
            )

        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        session = onnxruntime.InferenceSession(
            onnx_file_path,
            sess_options=sess_options,
            providers=[("CPUExecutionProvider")],
        )
        inputs_names = [i.name for i in session.get_inputs()]
        output = session.run(None, dict(zip(inputs_names, (example_input.numpy(),))))
        expected_output = model(example_input)
        self.assertEqual(expected_output.shape, output[0].shape)
        self.assertEqualArray(expected_output, output[0], atol=1e-4)

    def test_dyn_slice_4d_script(self):
        self._slice_4d("script")

    @hide_stdout()
    def test_dyn_slice_4d_dynamo(self):
        self._slice_4d("dynamo")

    def test_dyn_slice_4d_custom(self):
        self._slice_4d("custom")

    def _index_put_ellipsis(self, exporter):
        # https://github.com/pytorch/pytorch/issues/131349

        import torch
        import onnxruntime

        # 3D

        class UpdateModel(torch.nn.Module):
            def forward(
                self, x: torch.Tensor, update: torch.Tensor, kv_index: torch.LongTensor
            ):
                x = x.clone()
                x[..., kv_index] = update
                return x

        model = UpdateModel()
        example_inputs = (
            torch.ones((4, 4, 10)).to(torch.float32),
            (torch.arange(2) + 10).to(torch.float32).reshape((1, 1, 2)),
            torch.Tensor([1, 2]).to(torch.int32),
        )
        expected = model(*example_inputs)
        onnx_file_path = f"test_index_put_ellipsis_{exporter}_3d.onnx"

        # with torch.no_grad():
        if exporter == "dynamo":
            torch.onnx.export(
                model,
                example_inputs,
                onnx_file_path,
                output_names=["out"],
                dynamo=True,
                fallback=False,
            )
        else:
            to_onnx(model, example_inputs, filename=onnx_file_path)

        onx = onnx.load(onnx_file_path)
        # ref = ExtendedReferenceEvaluator(onnx_file_path)
        ref = onnxruntime.InferenceSession(onnx_file_path, providers=["CPUExecutionProvider"])
        inputs_names = [i.name for i in onx.graph.input]
        feeds = dict(zip(inputs_names, tuple(i.numpy() for i in example_inputs)))
        output = ref.run(None, feeds)
        self.assertEqualArray(expected, output[0], atol=1e-4)

        # 2D

        model = UpdateModel()
        example_inputs = (
            torch.ones((2, 10)).to(torch.float32),
            torch.arange(2).to(torch.float32).reshape((2, 1)),
            torch.Tensor([1]).to(torch.int32),
        )
        expected = model(*example_inputs)
        onnx_file_path = f"test_index_put_ellipsis_{exporter}_2d.onnx"

        # with torch.no_grad():
        if exporter == "dynamo":
            torch.onnx.export(
                model,
                example_inputs,
                onnx_file_path,
                output_names=["out"],
                dynamo=True,
                fallback=False,
            )
        else:
            to_onnx(model, example_inputs, filename=onnx_file_path, optimize=False)

        onx = onnx.load(onnx_file_path)
        # ref = ExtendedReferenceEvaluator(onnx_file_path)
        ref = onnxruntime.InferenceSession(onnx_file_path, providers=["CPUExecutionProvider"])
        inputs_names = [i.name for i in onx.graph.input]
        feeds = dict(zip(inputs_names, tuple(i.numpy() for i in example_inputs)))
        output = ref.run(None, feeds)
        self.assertEqualArray(expected, output[0], atol=1e-4)

    def test_index_put_ellipsis(self):
        self._index_put_ellipsis("custom")

    def _index_put_no_none(self, exporter, d3, decomposition=False, dynamic=False):
        # https://github.com/pytorch/pytorch/issues/135233

        import torch

        if d3:

            class UpdateModel(torch.nn.Module):

                def __init__(self):
                    super().__init__()
                    self.params = torch.zeros((4, 4, 10))

                def forward(self, update, index1, index2):
                    copy = self.params.clone()
                    copy[
                        index1, torch.from_numpy(np.array([1, 2], dtype=np.int64)), index2
                    ] = update
                    return copy

            model = UpdateModel()

        else:

            class UpdateModel(torch.nn.Module):

                def __init__(self):
                    super().__init__()
                    self.params = torch.zeros((4, 10))

                def forward(self, update, index1, index2):
                    copy = self.params.clone()
                    copy[index1, index2] = update
                    return copy

            model = UpdateModel()

        update = (torch.arange(2) + 10).reshape((2,)).to(torch.float32)
        index1 = torch.from_numpy(np.array([1, 2])).to(torch.int64)
        index2 = torch.from_numpy(np.array([7, 8])).to(torch.int64)
        model(update, index1, index2)

        model_path = (
            f"test_index_put_no_node_{exporter}_d{3 if d3 else 2}"
            f"D{1 if decomposition else 0}-{'dyn' if dynamic else 'sta'}.onnx"
        )

        if exporter == "script":
            torch.onnx.export(
                model,
                (update, index1, index2),
                model_path,
                input_names=["update", "index1", "index2"],
                output_names=["updated"],
                dynamic_axes={"update": {0: "n"}} if dynamic else None,
                opset_version=13,
                verbose=False,
            )
        elif exporter == "dynamo":
            dn = torch.export.Dim("n")
            torch.onnx.export(
                model,
                (update, index1, index2),
                model_path,
                input_names=["update", "index1", "index2"],
                output_names=["updated"],
                dynamic_shapes=(
                    {"update": {0: dn}, "index1": {0: dn}, "index2": {0: dn}}
                    if dynamic
                    else None
                ),
                verbose=False,
                dynamo=True,
            )
        else:
            if decomposition:
                if not has_torch("2.7"):
                    # see issue https://github.com/pytorch/pytorch/issues/141336
                    raise unittest.SkipTest("run_decompositions fails on this example")

            export_options = ExportOptions(
                decomposition_table="all" if decomposition else None, strict=True
            )
            dn = torch.export.Dim("n")
            to_onnx(
                model,
                (update, index1, index2),
                filename=model_path,
                export_options=export_options,
                dynamic_shapes=(
                    {"update": {0: dn}, "index1": {0: dn}, "index2": {0: dn}}
                    if dynamic
                    else None
                ),
                verbose=0,
                optimize=True,
            )

        check_model(model_path)

        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        session = onnxruntime.InferenceSession(
            model_path, sess_options=sess_options, providers=[("CPUExecutionProvider")]
        )

        input_n = dict(
            zip(
                ["update", "index1", "index2"],
                [update.numpy(), index1.numpy(), index2.numpy()],
            )
        )
        expected = model(update, index1, index2)
        # _ = ExtendedReferenceEvaluator(model_path, verbose=10).run(None, input_n)
        # self.assertEqualArray(expected, _[0])
        e1 = session.run(None, input_n)[0]
        self.assertEqualArray(expected, e1)

    @ignore_warnings(UserWarning)
    def test_index_put_no_none(self):
        for exporter, d3, decomposition, dynamic in itertools.product(
            ["custom", "onnx_dynamo", "torch_script"],
            [True, False],
            [False, True],
            [False, True],
        ):
            with self.subTest(
                exporter=exporter, d3=d3, decomposition=decomposition, dynamic=dynamic
            ):
                self._index_put_no_none(
                    exporter=exporter, d3=d3, decomposition=decomposition, dynamic=dynamic
                )

    def test_sequence_ops_embedding_bag(self):
        # https://github.com/pytorch/pytorch/issues/138485
        import torch

        model = torch.nn.EmbeddingBag(num_embeddings=49157, embedding_dim=32, mode="sum")
        a = torch.tensor([[39906]]).long()
        example_args = (a,)
        model_eval = model.eval()
        onx = to_onnx(model_eval, example_args, verbose=0)
        with open("test_sequence_ops_embedding_bag_custom.onnx", "wb") as f:
            f.write(onx.SerializeToString())

        expected = model(*example_args)

        from onnxruntime import InferenceSession

        sess = InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
        got = sess.run(None, {onx.graph.input[0].name: example_args[0].numpy()})
        self.assertEqualArray(expected, got[0])

        """
        torch.onnx.export(
            model_eval,
            example_args,
            "test_sequence_ops_embedding_bag_torchscript.onnx",
            dynamo=False,
        )
        """


if __name__ == "__main__":
    unittest.main(verbosity=2)
