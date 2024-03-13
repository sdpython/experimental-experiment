import onnxruntime  # noqa: F401
import copy
import unittest
import packaging.version as pv
from typing import Optional, Tuple
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
    skipif_ci_windows,
    requires_torch,
)
from experimental_experiment.torch_helper.dump_helper import assert_all_close
from experimental_experiment.torch_helper.training_helper import make_aot_ort


def has_cuda():
    import torch

    return torch.cuda.is_available()


def torch_min(v: str) -> bool:
    import torch

    return pv.Version(torch.__version__) < pv.Version(v)


class TestMistral(ExtTestCase):
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
            tuple(t.to(device) for t in examples)
            for examples in example_args_collection_cpu
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

    def _assert_counting_information(
        self,
        ort_backend: "OrtBackend",  # noqa: F821
        expected_execution_count: int,
        number_of_cached_graph_modules: int,
        number_of_exported_onnx_models_for_all_graph_modules: Tuple[int, ...],
        expected_graph_break=0,
        example_args_collection=None,
    ):
        self.assertEqual(
            expected_execution_count * (expected_graph_break + 1),
            ort_backend.execution_count,
            msg=f"expected_execution_count={expected_execution_count}, "
            f"expected_graph_break={expected_graph_break}, "
            f"ort_backend.execution_count={ort_backend.execution_count}, "
            f"number_of_cached_graph_modules={number_of_cached_graph_modules}",
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

        number_of_captured_graphs = 2 if test_backward else 1
        if assert_counting:
            self._assert_counting_information(
                local_ort,
                expected_execution_count=len(example_args_collection)
                * number_of_captured_graphs,
                number_of_cached_graph_modules=number_of_captured_graphs,
                number_of_exported_onnx_models_for_all_graph_modules=(1,)
                * number_of_captured_graphs,
                expected_graph_break=expected_graph_break,
                example_args_collection=example_args_collection,
            )

    @classmethod
    def get_input_dims(cls, dynamic: bool):
        if dynamic:
            input_dims = ((13, 7), (14, 7), (15, 8))
        else:
            input_dims = ((13, 7), (13, 7), (13, 7))
        return input_dims

    @ignore_warnings((UserWarning, DeprecationWarning))
    @skipif_ci_windows("torch.compile not supported on Windows")
    @requires_torch("2.3", "missing kernel")
    @unittest.skipIf(
        True, reason=" NOT_IMPLEMENTED : Could not find an implementation for Trilu(14"
    )
    def test_ort_mistral_model(self):
        from experimental_experiment.torch_helper.mistral_helper import (
            get_mistral_model,
        )

        input_dims = self.get_input_dims(False)
        model, example_args_collection = get_mistral_model(input_dims=input_dims)
        self.common_test_model(
            model,
            example_args_collection,
            test_backward=False,
            dynamic=False,
            fullgraph=True,
            onnx_export="test_ort_mistral_model",
            expected_graph_break=0,
        )

    @ignore_warnings((UserWarning, DeprecationWarning))
    @skipif_ci_windows("torch.compile not supported on Windows")
    @unittest.skipIf(torch_min("2.2"), reason="missing kernel")
    @unittest.skipIf(not has_cuda(), reason="cuda not available")
    @unittest.skipIf(
        True, reason=" NOT_IMPLEMENTED : Could not find an implementation for Trilu(14"
    )
    def test_ort_mistral_model_cuda(self):
        from experimental_experiment.torch_helper.mistral_helper import (
            get_mistral_model,
        )

        input_dims = self.get_input_dims(False)
        model, example_args_collection = get_mistral_model(input_dims=input_dims)
        self.common_test_model(
            model,
            example_args_collection,
            test_backward=False,
            dynamic=False,
            fullgraph=True,
            onnx_export="test_ort_mistral_model",
            expected_graph_break=0,
            device="cuda",
        )

    @ignore_warnings((UserWarning, DeprecationWarning))
    @skipif_ci_windows("torch.compile not supported on Windows")
    @requires_torch("2.3", "missing kernel")
    @unittest.skipIf(
        True, reason=" NOT_IMPLEMENTED : Could not find an implementation for Trilu(14"
    )
    def test_ort_mistral_model_backward(self):
        from experimental_experiment.torch_helper.mistral_helper import (
            get_mistral_model,
        )

        input_dims = self.get_input_dims(False)
        model, example_args_collection = get_mistral_model(input_dims=input_dims)
        self.common_test_model(
            model,
            example_args_collection,
            test_backward=True,
            dynamic=False,
            fullgraph=True,
            onnx_export="test_ort_mistral_model_backward",
            expected_graph_break=0,
            assert_counting=True,
        )

    @ignore_warnings((UserWarning, DeprecationWarning))
    @skipif_ci_windows("torch.compile not supported on Windows")
    @unittest.skipIf(torch_min("2.2"), reason="missing kernel")
    @unittest.skipIf(not has_cuda(), reason="cuda not available")
    def test_ort_mistral_model_backward_cuda(self):
        from experimental_experiment.torch_helper.mistral_helper import (
            get_mistral_model,
        )

        input_dims = self.get_input_dims(False)
        model, example_args_collection = get_mistral_model(input_dims=input_dims)
        self.common_test_model(
            model,
            example_args_collection,
            test_backward=True,
            dynamic=False,
            fullgraph=True,
            onnx_export="test_ort_mistral_model_backward",
            expected_graph_break=0,
            assert_counting=True,
            device="cuda",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
