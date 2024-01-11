import copy
import unittest
from typing import Tuple
from experimental_experiment.ext_test_case import ExtTestCase, ignore_warnings


def make_aot_ort(dynamic: bool = False):
    from torch.onnx import (
        _OrtBackend as OrtBackend,
        _OrtBackendOptions as OrtBackendOptions,
        ExportOptions,
    )

    ort_backend = OrtBackend(
        options=OrtBackendOptions(
            export_options=ExportOptions(
                dynamic_shapes=dynamic,
            )
        )
    )
    return ort_backend, ort_backend


class TestLlama(ExtTestCase):
    def _test_model_numerically(
        self,
        model,
        dynamo_backend,
        example_args_collection,
        fullgraph: bool = True,
        test_backward: bool = False,
        dynamic: bool = False,
        atol: float = 1e-5,
        rtol: float = 1e-6,
    ):
        """
        Run original and compiled model and compare the results.
        Args:
            model: The model to test.
            dynamo_backend: The dynamo backend to use. Here we use string `onnxrt` or
              the first returned value of `make_aot_ort(dynamic=True)`.
            example_args_collection: A tuple of example arguments to test. E.g.,
                (
                  (torch.randn(2), torch.randn(2)),
                  (torch.randn(4), torch.randn(4)),
                )
              if you want to test
                model(torch.randn(2), torch.randn(2)) and
                model(torch.randn(4), torch.randn(4))
              .
        """
        import torch

        compiled_model = torch.compile(
            copy.deepcopy(model),
            backend=dynamo_backend,
            dynamic=dynamic,
            fullgraph=fullgraph,
        )

        for example_args in example_args_collection:
            baseline_result = model(*example_args)
            result = compiled_model(*example_args)
            if isinstance(baseline_result, torch.Tensor):
                torch.testing.assert_close(baseline_result, result)
                torch.testing.assert_close(
                    baseline_result, result, atol=atol, rtol=rtol
                )
                if test_backward:
                    baseline_result.sum().backward()
                    result.sum().backward()
                    for baseline_param, param in zip(
                        model.parameters(), compiled_model.parameters()
                    ):
                        torch.testing.assert_close(
                            baseline_param.grad, param.grad, atol=atol, rtol=rtol
                        )
            else:
                assert (
                    test_backward is False
                ), "Calculating backward with multiple outputs is not supported yet."
                for baseline_elem, result_elem in zip(baseline_result, result):
                    torch.testing.assert_close(baseline_elem, result_elem)
                    torch.testing.assert_close(
                        baseline_elem, result_elem, atol=atol, rtol=rtol
                    )

    def _assert_counting_information(
        self,
        ort_backend: "OrtBackend",  # noqa: F821
        expected_execution_count: int,
        number_of_cached_graph_modules: int,
        number_of_exported_onnx_models_for_all_graph_modules: Tuple[int, ...],
    ):
        self.assertEqual(expected_execution_count, ort_backend.execution_count)
        self.assertEqual(
            len(ort_backend._all_ort_execution_info.execution_info_per_graph_module),
            number_of_cached_graph_modules,
        )
        self.assertEqual(
            len(ort_backend._all_ort_execution_info.execution_info_per_graph_module),
            len(number_of_exported_onnx_models_for_all_graph_modules),
        )
        for (
            onnx_info,
            expected_number_of_onnx_models,
        ) in zip(
            ort_backend._all_ort_execution_info.execution_info_per_graph_module.values(),
            number_of_exported_onnx_models_for_all_graph_modules,
        ):
            self.assertEqual(len(onnx_info), expected_number_of_onnx_models)

    def common_test_model(
        self,
        model,
        example_args_collection,
        test_backward: bool,
        dynamic: bool,
    ):
        local_aot_ort, local_ort = make_aot_ort(dynamic=dynamic)

        self._test_model_numerically(
            model,
            local_aot_ort,
            example_args_collection,
            fullgraph=True,
            test_backward=test_backward,
        )

        number_of_captured_graphs = 2 if test_backward else 1
        execution_count = len(example_args_collection) * number_of_captured_graphs
        self._assert_counting_information(
            local_ort,
            expected_execution_count=execution_count,
            number_of_cached_graph_modules=number_of_captured_graphs,
            number_of_exported_onnx_models_for_all_graph_modules=(1,)
            * number_of_captured_graphs,
        )

    @ignore_warnings((UserWarning, DeprecationWarning))
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

        self.common_test_model(MLP(), example_args_collection, False, False)

    @ignore_warnings((UserWarning, DeprecationWarning))
    def test_ort_llama_decoder(self):
        from experimental_experiment.torch_helper.llama_helper import get_llama_decoder

        dynamic = False
        if dynamic:
            input_dims = ((2, 8), (4, 7), (9, 15))
        else:
            input_dims = ((9, 15), (9, 15), (9, 15))
        model, example_args_collection = get_llama_decoder(input_dims=input_dims)
        self.common_test_model(model, example_args_collection, False, False)

    @ignore_warnings((UserWarning, DeprecationWarning))
    def test_ort_llama_attention(self):
        from experimental_experiment.torch_helper.llama_helper import (
            get_llama_attention,
        )

        dynamic = False
        if dynamic:
            input_dims = ((2, 8), (4, 7), (9, 15))
        else:
            input_dims = ((9, 15), (9, 15), (9, 15))
        model, example_args_collection = get_llama_attention(input_dims=input_dims)
        self.common_test_model(model, example_args_collection, False, False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
