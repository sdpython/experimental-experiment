import copy
import unittest
import packaging.version as pv
from typing import Any, List, Optional, Tuple
from onnx.reference import ReferenceEvaluator
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
    skipif_ci_windows,
)
from experimental_experiment.torch_exp.onnx_export import to_onnx


def torch_min(v: str) -> bool:
    import torch

    return pv.Version(torch.__version__) < pv.Version(v)


class TestDynamoLlama(ExtTestCase):
    @ignore_warnings((UserWarning, DeprecationWarning))
    def test_aaaa(self):
        import torch
        from transformers import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaAttention

        config = LlamaConfig(
            num_hidden_layers=1,
            vocab_size=1024,
            hidden_size=16,
            intermediate_size=16,
            max_position_embeddings=256,
            num_attention_heads=2,
            hidden_dropout_prob=0.0,
            attention_dropout_prob=0.0,
        )
        model = LlamaAttention(config, layer_idx=0)
        torch.save(model, "temp.llama.torch.pkl")

    def _assert_model_numerically(
        self,
        model,
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

        class backend:
            def __init__(self):
                self.execution_count = 0

        _b = backend()

        def onnx_compiler(
            graph_module: torch.fx.GraphModule, args: List[torch.Tensor], _b=_b
        ):
            input_names = (
                ["input"] if len(args) == 1 else [f"input{i}" for i in range(len(args))]
            )

            onx = to_onnx(
                graph_module,
                tuple(args),
                input_names=input_names,
                remove_unused=True,
                constant_folding=True,
                verbose=4,
            )
            try:
                sess = ReferenceEvaluator(onx, verbose=10)
            except Exception as e:
                from onnx_array_api.plotting.text_plot import onnx_simple_text_plot

                raise AssertionError(
                    f"Unable to run onnx graph ({str(e)})\n{onnx_simple_text_plot(onx)}"
                ) from e
            names = [i.name for i in onx.graph.input]

            def run(*inputs, sess=sess, names=names, _b=_b):
                # not efficient
                xnp = [x.detach().numpy() for x in inputs]
                feeds = dict(zip(names, xnp))
                res = tuple(torch.Tensor(y) for y in sess.run(None, feeds))
                _b.execution_count += 1
                return res

            return run

        compiled_model = torch.compile(
            copy.deepcopy(model),
            backend=onnx_compiler,
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

        return _b

    def _assert_counting_information(
        self,
        ort_backend: Any,
        expected_execution_count: int,
        number_of_cached_graph_modules: int,
        number_of_exported_onnx_models_for_all_graph_modules: Tuple[int, ...],
        expected_graph_break=0,
    ):
        self.assertEqual(
            expected_execution_count * (expected_graph_break + 1),
            ort_backend.execution_count,
        )

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
        local_ort = self._assert_model_numerically(
            model,
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
    def test_ort_amlp(self):
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
    def test_ort_amlp_backward(self):
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
    def test_ort_llama_decoder(self):
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
            expected_graph_break=7,
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
