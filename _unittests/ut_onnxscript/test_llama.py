import onnxruntime  # noqa: F401
import copy
import unittest
from typing import Optional
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
    skipif_ci_windows,
    requires_torch,
    requires_cuda,
    requires_onnxscript,
    hide_stdout,
)
from experimental_experiment.torch_models.dump_helper import assert_all_close
from experimental_experiment.torch_models.training_helper import make_aot_ort


class TestLlama(ExtTestCase):
    def _assert_model_numerically(
        self,
        model,
        dynamo_backend,
        example_args_collection_cpu,
        fullgraph: bool = True,
        test_backward: bool = False,
        dynamic: bool = False,
        atol: float = 1e-4,
        rtol: float = 1e-4,
        onnx_export: Optional[str] = None,
        device: str = "cpu",
    ):
        import torch

        assert onnx_export, "No export name was given"
        torch._dynamo.reset()

        model = model.to(device)
        example_args_collection = [
            tuple(t.to(device) for t in examples) for examples in example_args_collection_cpu
        ]

        compiled_model = torch.compile(
            copy.deepcopy(model),
            backend=dynamo_backend,
            dynamic=dynamic,
            fullgraph=fullgraph,
        )

        for example_args in example_args_collection:
            baseline_result = model(*example_args)
            result = compiled_model(*example_args)
            assert_all_close(baseline_result, result, atol=atol, rtol=rtol)
            if test_backward:
                baseline_result[0].sum().backward()
                result[0].sum().backward()
                base_grads = tuple(_.grad for _ in model.parameters())
                grads = tuple(_.grad for _ in compiled_model.parameters())
                assert_all_close(base_grads, grads, atol=atol, rtol=rtol)

    def common_test_model(
        self,
        model,
        example_args_collection,
        test_backward: bool,
        dynamic: bool,
        fullgraph: bool = True,
        onnx_export=None,
        expected_graph_break=0,
        device="cpu",
    ):
        import torch

        torch._dynamo.reset()
        local_aot_ort, local_ort = make_aot_ort(dynamic=dynamic)

        self._assert_model_numerically(
            model,
            local_aot_ort,
            example_args_collection,
            test_backward=test_backward,
            fullgraph=fullgraph,
            onnx_export=onnx_export,
            device=device,
        )

    @ignore_warnings((UserWarning, DeprecationWarning))
    @skipif_ci_windows("torch.compile not supported on Windows")
    @requires_torch("2.4")
    @hide_stdout()
    def test_ort_mlp(self):
        import torch

        class MLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(2, 4, bias=True)
                self.fc2 = torch.nn.Linear(4, 2, bias=True)

            def forward(self, tensor_x: torch.Tensor):
                tensor_x = self.fc1(tensor_x)
                tensor_x = torch.sigmoid(tensor_x)
                tensor_x = self.fc2(tensor_x)
                tensor_x = torch.sigmoid(tensor_x)
                return tensor_x

        # with static shape (dynamic=False), the conversion to onnx is done
        # every time the batch size changes
        batch_sizes = [3, 3, 3, 3, 3]

        example_args_collection = tuple(
            (torch.randn(batch, 2, dtype=torch.float32),) for batch in batch_sizes
        )

        self.common_test_model(
            MLP(), example_args_collection, False, False, onnx_export="test_ort_mlp"
        )

    @ignore_warnings((UserWarning, DeprecationWarning))
    @skipif_ci_windows("torch.compile not supported on Windows")
    @requires_torch("2.4")
    @requires_onnxscript("0.3")  # type promotion
    @hide_stdout()
    def test_ort_mlp_backward(self):
        import torch

        class MLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(2, 4, bias=True)
                self.fc2 = torch.nn.Linear(4, 2, bias=True)

            def forward(self, tensor_x: torch.Tensor):
                tensor_x = self.fc1(tensor_x)
                tensor_x = torch.sigmoid(tensor_x)
                tensor_x = self.fc2(tensor_x)
                tensor_x = torch.sigmoid(tensor_x)
                return tensor_x

        # with static shape (dynamic=False), the conversion to onnx is done
        # every time the batch size changes
        batch_sizes = [3, 3, 3, 3, 3]

        example_args_collection = tuple(
            (torch.randn(batch, 2, dtype=torch.float32),) for batch in batch_sizes
        )

        self.common_test_model(
            MLP(),
            example_args_collection,
            test_backward=True,
            dynamic=False,
            onnx_export="test_ort_mlp_backward",
        )

    @classmethod
    def get_input_dims(cls, dynamic: bool):
        if dynamic:
            input_dims = ((2, 8), (4, 7), (9, 15))
        else:
            input_dims = ((9, 15), (9, 15), (9, 15))
        return input_dims

    @ignore_warnings((UserWarning, DeprecationWarning))
    @skipif_ci_windows("torch.compile not supported on Windows")
    @requires_torch("2.4")
    @hide_stdout()
    def test_ort_llama_model(self):
        from experimental_experiment.torch_models.llama_helper import (
            get_llama_model,
        )

        input_dims = self.get_input_dims(False)
        model, example_args_collection = get_llama_model(input_dims=input_dims)
        self.common_test_model(
            model,
            example_args_collection,
            test_backward=False,
            dynamic=False,
            fullgraph=True,
            onnx_export="test_ort_llama_model",
            expected_graph_break=0,
        )

    @ignore_warnings((UserWarning, DeprecationWarning))
    @skipif_ci_windows("torch.compile not supported on Windows")
    @requires_torch("2.4")
    @requires_cuda()
    @hide_stdout()
    def test_ort_llama_model_cuda(self):
        from experimental_experiment.torch_models.llama_helper import (
            get_llama_model,
        )

        input_dims = self.get_input_dims(False)
        model, example_args_collection = get_llama_model(input_dims=input_dims)
        self.common_test_model(
            model,
            example_args_collection,
            test_backward=False,
            dynamic=False,
            fullgraph=True,
            onnx_export="test_ort_llama_model",
            expected_graph_break=0,
            device="cuda",
        )

    @ignore_warnings((UserWarning, DeprecationWarning))
    @skipif_ci_windows("torch.compile not supported on Windows")
    @requires_torch("2.4")
    @hide_stdout()
    def test_ort_llama_model_backward(self):
        from experimental_experiment.torch_models.llama_helper import (
            get_llama_model,
        )

        input_dims = self.get_input_dims(False)
        model, example_args_collection = get_llama_model(input_dims=input_dims)
        self.common_test_model(
            model,
            example_args_collection,
            test_backward=True,
            dynamic=False,
            fullgraph=True,
            onnx_export="test_ort_llama_model_backward",
            expected_graph_break=0,
        )

    @ignore_warnings((UserWarning, DeprecationWarning))
    @skipif_ci_windows("torch.compile not supported on Windows")
    @requires_torch("2.4")
    @requires_cuda()
    @hide_stdout()
    def test_ort_llama_model_backward_cuda(self):
        from experimental_experiment.torch_models.llama_helper import (
            get_llama_model,
        )

        input_dims = self.get_input_dims(False)
        model, example_args_collection = get_llama_model(input_dims=input_dims)
        self.common_test_model(
            model,
            example_args_collection,
            test_backward=True,
            dynamic=False,
            fullgraph=True,
            onnx_export="test_ort_llama_model_backward",
            expected_graph_break=0,
            device="cuda",
        )

    def test_get_llama_model_mask_sdpa(self):
        from experimental_experiment.torch_models.llama_helper import (
            get_llama_model,
        )

        model, model_inputs = get_llama_model(_attn_implementation="sdpa", with_mask=True)
        self.assertEqual(len(model_inputs[0]), 2)
        expected = model(*model_inputs[0])
        self.assertNotEmpty(expected)

    def test_get_llama_model_nomask_sdpa(self):
        from experimental_experiment.torch_models.llama_helper import (
            get_llama_model,
        )

        model, model_inputs = get_llama_model(_attn_implementation="sdpa", with_mask=False)
        self.assertEqual(len(model_inputs[0]), 1)
        expected = model(*model_inputs[0])
        self.assertNotEmpty(expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
