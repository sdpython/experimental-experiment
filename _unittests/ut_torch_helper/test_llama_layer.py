import copy
import unittest
import packaging.version as pv
from typing import Optional, Tuple
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
    skipif_ci_windows,
)


def torch_min(v: str) -> bool:
    import torch

    return pv.Version(torch.__version__) < pv.Version(v)


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
    def _assert_model_numerically(
        self,
        model,
        dynamo_backend,
        example_args_collection,
        fullgraph: bool = True,
        test_backward: bool = False,
        dynamic: bool = False,
        atol: float = 1e-4,
        rtol: float = 1e-4,
        onnx_export: Optional[str] = None,
    ):
        import torch

        assert onnx_export, "No export name was given"
        compiled_model = torch.compile(
            copy.deepcopy(model),
            backend=dynamo_backend,
            dynamic=dynamic,
            fullgraph=fullgraph,
        )

        one_example = None
        for example_args in example_args_collection:
            baseline_result = model(*example_args)
            one_example = example_args

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
                if hasattr(baseline_result, "to_tuple"):
                    baseline_result = baseline_result.to_tuple()
                    result = result.to_tuple()
                assert len(baseline_result) == len(
                    result
                ), f"Mismatch number of outputs {len(baseline_result)} != {len(result)}"
                for baseline_elem, result_elem in zip(baseline_result, result):
                    torch.testing.assert_close(baseline_elem, result_elem)
                    torch.testing.assert_close(
                        baseline_elem, result_elem, atol=atol, rtol=rtol
                    )
                if test_backward:

                    def _do_sum(x):
                        if isinstance(x, torch.Tensor):
                            return x.sum()
                        if isinstance(x, tuple):
                            s = _do_sum(x[0])
                            for i in range(1, len(x)):
                                s = s + _do_sum(x[i])
                            return s
                        raise TypeError(f"unexpected type {type(x)}")

                    baseline_sum = _do_sum(baseline_result)
                    result_sum = _do_sum(result)
                    baseline_sum.backward()
                    result_sum.backward()
                    for baseline_param, param in zip(
                        model.parameters(), compiled_model.parameters()
                    ):
                        torch.testing.assert_close(
                            baseline_param.grad, param.grad, atol=atol, rtol=rtol
                        )

        # export to onnx
        try:
            torch.onnx.export(
                copy.deepcopy(model), *one_example, f"{onnx_export}_script.onnx"
            )
        except Exception as e:
            print("torch.onnx.export failed:", e)
        try:
            torch.onnx.dynamo_export(copy.deepcopy(model), *one_example).save(
                f"{onnx_export}_dynamo.onnx"
            )
        except Exception as e:
            print("torch.onnx.dynamo_export failed:", e)

    def _assert_counting_information(
        self,
        ort_backend: "OrtBackend",  # noqa: F821
        expected_execution_count: int,
        number_of_cached_graph_modules: int,
        number_of_exported_onnx_models_for_all_graph_modules: Tuple[int, ...],
        expected_graph_break=0,
    ):
        self.assertEqual(
            expected_execution_count * (expected_graph_break + 1),
            ort_backend.execution_count,
        )
        self.assertEqual(
            len(ort_backend._all_ort_execution_info.execution_info_per_graph_module),
            number_of_cached_graph_modules * (expected_graph_break + 1),
        )
        self.assertEqual(
            len(ort_backend._all_ort_execution_info.execution_info_per_graph_module),
            len(number_of_exported_onnx_models_for_all_graph_modules)
            * (expected_graph_break + 1),
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
        fullgraph: bool = True,
        onnx_export=None,
        expected_graph_break=0,
        assert_counting=True,
    ):
        local_aot_ort, local_ort = make_aot_ort(dynamic=dynamic)

        self._assert_model_numerically(
            model,
            local_aot_ort,
            example_args_collection,
            test_backward=test_backward,
            fullgraph=fullgraph,
            onnx_export=onnx_export,
        )

        number_of_captured_graphs = 2 if test_backward else 1
        execution_count = len(example_args_collection) * number_of_captured_graphs
        if assert_counting:
            self._assert_counting_information(
                local_ort,
                expected_execution_count=execution_count,
                number_of_cached_graph_modules=number_of_captured_graphs,
                number_of_exported_onnx_models_for_all_graph_modules=(1,)
                * number_of_captured_graphs,
                expected_graph_break=expected_graph_break,
            )

    @ignore_warnings((UserWarning, DeprecationWarning))
    @skipif_ci_windows("torch.compile not supported on Windows")
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
    @unittest.skipIf(torch_min("2.2"), reason="missing kernel")
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
            True,
            False,
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
    def test_ort_llama_decoder_forward(self):
        from experimental_experiment.torch_helper.llama_helper import get_llama_decoder

        input_dims = self.get_input_dims(False)
        model, example_args_collection = get_llama_decoder(input_dims=input_dims)
        self.common_test_model(
            model,
            example_args_collection,
            False,
            False,
            onnx_export="test_ort_llama_decoder",
        )

    @ignore_warnings((UserWarning, DeprecationWarning))
    @skipif_ci_windows("torch.compile not supported on Windows")
    def test_ort_llama_decoder_backward(self):
        from experimental_experiment.torch_helper.llama_helper import get_llama_decoder

        input_dims = self.get_input_dims(False)
        model, example_args_collection = get_llama_decoder(input_dims=input_dims)
        self.common_test_model(
            model,
            example_args_collection,
            True,
            False,
            onnx_export="test_ort_llama_decoder_backward",
        )

    @ignore_warnings((UserWarning, DeprecationWarning))
    @skipif_ci_windows("torch.compile not supported on Windows")
    def test_ort_llama_attention(self):
        from experimental_experiment.torch_helper.llama_helper import (
            get_llama_attention,
        )

        input_dims = self.get_input_dims(False)
        model, example_args_collection = get_llama_attention(input_dims=input_dims)
        self.common_test_model(
            model,
            example_args_collection,
            False,
            False,
            onnx_export="test_ort_llama_attention",
        )

    @ignore_warnings((UserWarning, DeprecationWarning))
    @skipif_ci_windows("torch.compile not supported on Windows")
    def test_ort_llama_attention_backward(self):
        from experimental_experiment.torch_helper.llama_helper import (
            get_llama_attention,
        )

        input_dims = self.get_input_dims(False)
        model, example_args_collection = get_llama_attention(input_dims=input_dims)
        self.common_test_model(
            model,
            example_args_collection,
            True,
            False,
            onnx_export="test_ort_llama_attention_backward",
        )

    @ignore_warnings((UserWarning, DeprecationWarning))
    @skipif_ci_windows("torch.compile not supported on Windows")
    @unittest.skipIf(torch_min("2.2"), reason="missing kernel")
    def test_ort_llama_model_nofullgraph(self):
        from experimental_experiment.torch_helper.llama_helper import (
            get_llama_model,
        )

        input_dims = self.get_input_dims(False)
        model, example_args_collection = get_llama_model(input_dims=input_dims)
        self.common_test_model(
            model,
            example_args_collection,
            False,
            False,
            fullgraph=False,
            onnx_export="test_ort_llama_model_nofullgraph",
            expected_graph_break=4,
        )

    @ignore_warnings((UserWarning, DeprecationWarning))
    @skipif_ci_windows("torch.compile not supported on Windows")
    @unittest.skipIf(torch_min("2.2"), reason="missing kernel")
    def test_ort_llama_model_backward_nofullgraph(self):
        from experimental_experiment.torch_helper.llama_helper import (
            get_llama_model,
        )

        input_dims = self.get_input_dims(False)
        model, example_args_collection = get_llama_model(input_dims=input_dims)
        self.common_test_model(
            model,
            example_args_collection,
            True,
            False,
            fullgraph=False,
            onnx_export="test_ort_llama_model_backward_nofullgraph",
            expected_graph_break=7,
            assert_counting=False,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
