import unittest
import numpy as np
from onnx.checker import check_model
from experimental_experiment.ext_test_case import ExtTestCase, requires_cuda, hide_stdout
from experimental_experiment.torch_interpreter import to_onnx


class TestIssuesOnnxExporter(ExtTestCase):

    def _updated_parameter(self, exporter):
        # https://github.com/pytorch/pytorch/issues/135233

        import torch

        class UpdateModel(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.params = torch.zeros((2, 1, 10))

            def forward(self, update: torch.Tensor, index: torch.LongTensor):
                indices = torch.arange(update.shape[0])
                middle = torch.zeros((1,), dtype=torch.long)
                copy = self.params.clone()
                copy[index, middle, indices] = update.transpose(1, 0)
                return copy

        model = UpdateModel()

        n = 6
        update = torch.ones((n, 1))
        kv_index = torch.tensor([0])
        model(update, kv_index)

        model_path = f"test_updated_cache_{exporter}.onnx"

        if exporter == "script":
            torch.onnx.export(
                model,
                (update, kv_index),
                model_path,
                input_names=["update", "kv_index"],
                output_names=["updated"],
                dynamic_axes={"update": {0: "n"}},
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
                dynamic_axes={"update": {0: "n"}},
                verbose=False,
                dynamo=True,
            )
        else:
            to_onnx(model, (update, kv_index), filename=model_path)

        check_model(model_path)

        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        session = onnxruntime.InferenceSession(
            model_path, sess_options=sess_options, providers=[("CPUExecutionProvider")]
        )

        def gen_numpy_inputs(n: int, idx: int):
            return {
                "update": 5 * np.ones((n, 1), dtype=np.float32),
                "index": np.array([idx], dtype=np.int64),
            }

        input_n = gen_numpy_inputs(n, 0)
        e1 = session.run(None, input_n)[0]
        self.assertEqual(e1[0].shape, model.params.shape)

        input_2 = gen_numpy_inputs(2, 0)
        e2 = session.run(None, input_2)[0]
        self.assertEqual(e2[0].shape, model.params.shape)

    def test_update_parameter_script(self):
        self._updated_parameter("script")

    def test_update_parameter_dynamo(self):
        self._updated_parameter("dynamo")

    def test_update_parameter_custom(self):
        self._updated_parameter("custom")

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

        output = model(query_states, key_states, value_states)

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
        session = onnxruntime.InferenceSession(
            onnx_file_path,
            sess_options=sess_options,
            providers=[("CPUExecutionProvider")],
        )
        inputs_names = [i.name for i in session.get_inputs()]
        output = session.run(
            None,
            dict(
                zip(
                    inputs_names,
                    (query_states.numpy(), key_states.numpy(), value_states.numpy()),
                )
            ),
        )
        expected_output = model(query_states, key_states, value_states)
        self.assertEqual(expected_output[0].shape, output[0].shape)
        self.assertEqualArray(expected_output[0], output[0])

    def test_scaled_dot_product_attention_script(self):
        self._scaled_dot_product_attention("script")

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

    def test_in_projection_packed_script(self):
        self._in_projection_packed("script")

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

    def test_complex_weights_script(self):
        self._complex_weights("script")

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
            torch.onnx.export(
                model,
                (example_input,),
                onnx_file_path,
                input_names=["input"],
                output_names=["output"],
                dynamo=True,
                dynamic_axes={"input": {0: "batch_size"}},
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

    def test_dyn_slice_4d_dynamo(self):
        self._slice_4d("dynamo")

    def test_dyn_slice_4d_custom(self):
        self._slice_4d("custom")


if __name__ == "__main__":
    unittest.main(verbosity=2)
