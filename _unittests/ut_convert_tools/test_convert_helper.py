import onnxruntime  # noqa: F401
import unittest
from onnx import ModelProto
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    skipif_ci_windows,
    requires_cuda,
    ignore_warnings,
)
from experimental_experiment.convert.convert_helper import (
    optimize_model_proto,
    inline_model_proto,
    ort_optimize,
)
from experimental_experiment.torch_interpreter import to_onnx

input_dims = ((2, 1024),)


class TestConvertHelper(ExtTestCase):
    @ignore_warnings(UserWarning)
    def test_optimize_llama(self):
        import torch
        from experimental_experiment.torch_models.llama_helper import (
            get_llama_attention,
        )

        model, example_args_collection = get_llama_attention(input_dims=input_dims)
        model(*example_args_collection[0])
        model = torch.onnx.dynamo_export(model, *example_args_collection[0])
        model_proto = model.model_proto
        model_proto = optimize_model_proto(model_proto)
        self.assertIsInstance(model_proto, ModelProto)

    @skipif_ci_windows("dynamo not working on windows")
    @ignore_warnings(UserWarning)
    def test_inline_llama(self):
        import torch
        from experimental_experiment.torch_models.llama_helper import (
            get_llama_attention,
        )

        model, example_args_collection = get_llama_attention(input_dims=input_dims)
        model(*example_args_collection[0])
        model = torch.onnx.dynamo_export(model, *example_args_collection[0])
        model_proto = model.model_proto
        model_proto = inline_model_proto(model_proto)
        self.assertIsInstance(model_proto, ModelProto)

    @skipif_ci_windows("dynamo not working on windows")
    @ignore_warnings(UserWarning)
    @unittest.skipIf(True, reason="unstable")
    def test_ort_optimize_dynamo_cpu(self):
        import torch
        from experimental_experiment.torch_models.llama_helper import (
            get_llama_attention,
        )

        model, example_args_collection = get_llama_attention(input_dims=input_dims)
        model(*example_args_collection[0])
        model = torch.onnx.dynamo_export(model, *example_args_collection[0])
        model_proto = model.model_proto
        ort_optimize(
            model_proto, providers="cpu", output="test_ort_optimize_dynamo_cpu.onnx"
        )

    @requires_cuda()
    @ignore_warnings(UserWarning)
    @unittest.skipIf(True, reason="unstable")
    def test_ort_optimize_dynamo_cuda(self):
        import torch
        from experimental_experiment.torch_models.llama_helper import (
            get_llama_attention,
        )

        model, example_args_collection = get_llama_attention(input_dims=input_dims)
        model(*example_args_collection[0])
        model = torch.onnx.dynamo_export(model, *example_args_collection[0])
        model_proto = model.model_proto
        ort_optimize(
            model_proto, providers="cuda", output="test_ort_optimize_dynamo_cuda.onnx"
        )

    @skipif_ci_windows("dynamo not working on windows")
    @ignore_warnings(UserWarning)
    @unittest.skipIf(True, reason="unstable")
    def test_ort_optimize_cpu(self):
        from experimental_experiment.torch_models.llama_helper import (
            get_llama_attention,
        )

        model, example_args_collection = get_llama_attention(input_dims=input_dims)
        model(*example_args_collection[0])
        model = to_onnx(model, example_args_collection[0])
        model_proto = model.model_proto
        ort_optimize(model_proto, providers="cpu", output="test_ort_optimize_cpu.onnx")

    @requires_cuda()
    @ignore_warnings(UserWarning)
    @unittest.skipIf(True, reason="unstable")
    def test_ort_optimize_cuda(self):
        from experimental_experiment.torch_models.llama_helper import (
            get_llama_attention,
        )

        model, example_args_collection = get_llama_attention(input_dims=input_dims)
        model(*example_args_collection[0])
        model = to_onnx(model, example_args_collection[0])
        model_proto = model.model_proto
        ort_optimize(
            model_proto, providers="cuda", output="test_ort_optimize_cuda.onnx"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
