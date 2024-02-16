import copy
import unittest
import packaging.version as pv
from typing import Optional
import onnxruntime  # noqa: F401
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
    skipif_ci_windows,
)
from experimental_experiment.torch_helper.dump_helper import dump_onnx, assert_all_close
from experimental_experiment.torch_dynamo import (
    onnx_debug_backend,
    get_decomposition_table,
)


def torch_min(v: str) -> bool:
    import torch

    return pv.Version(torch.__version__) < pv.Version(v)


def implements(name: str) -> bool:
    import experimental_experiment.torch_exp._aten_functions as atf

    return hasattr(atf, name)


def has_cuda():
    import torch

    return torch.cuda.is_available()


class TestDynamoLlama(ExtTestCase):
    @ignore_warnings((UserWarning, DeprecationWarning))
    def test_aaaa(self):
        from transformers import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaAttention

        config = LlamaConfig(
            num_hidden_layers=1,
            vocab_size=1024,
            hidden_size=16,
            intermediate_size=16,
            max_position_embeddings=256,
            num_attention_heads=2,
        )
        LlamaAttention(config, layer_idx=0)

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
        impl="ort",
        verbose: int = 0,
        decompositions=False,
        mixed=False,
    ):
        import torch

        assert onnx_export, "No export name was given"

        class backend:
            def __init__(self):
                self.execution_count = 0

        def _flatten(a):
            if isinstance(a, tuple):
                r = []
                for i in a:
                    r.extend(_flatten(i))
                return tuple(_ for _ in r if _ is not None)
            return (a,) if a is not None else tuple()

        storage = {}
        backend_debug = lambda *args, **kwargs: onnx_debug_backend(  # noqa: E731
            *args,
            # dump_prefix=os.path.join(folder, "llama_debug"),
            backend=impl,
            target_opset=18,
            storage=storage,
            verbose=verbose,
            **kwargs,
        )

        if test_backward:
            from torch._dynamo.backends.common import aot_autograd

            if decompositions:
                aot_compiler = aot_autograd(
                    fw_compiler=backend_debug,
                    decompositions=torch._decomp.decomposition_table,
                )
            else:
                aot_compiler = aot_autograd(
                    fw_compiler=backend_debug, decompositions=get_decomposition_table()
                )

            compiled_model = torch.compile(
                copy.deepcopy(model),
                backend=aot_compiler,
                dynamic=dynamic,
                fullgraph=fullgraph,
            )
        else:
            assert fullgraph
            compiled_model = torch.compile(
                copy.deepcopy(model),
                backend=backend_debug,
                dynamic=dynamic,
                fullgraph=fullgraph,
            )

        for example_args in example_args_collection:
            if mixed:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    baseline_result = model(*example_args)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    result = compiled_model(*example_args)
            else:
                baseline_result = model(*example_args)
                result = compiled_model(*example_args)
            assert_all_close(baseline_result, result, atol=atol, rtol=rtol)
            if test_backward is True:
                if mixed:
                    if isinstance(baseline_result, tuple):
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            baseline_result[0].sum().backward()
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            result[0].sum().backward()
                    else:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            baseline_result.sum().backward()
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            result.sum().backward()
                else:
                    if isinstance(baseline_result, tuple):
                        baseline_result[0].sum().backward()
                        result[0].sum().backward()
                    else:
                        baseline_result.sum().backward()
                        result.sum().backward()
                base_grads = tuple(_.grad for _ in model.parameters())
                grads = tuple(_.grad for _ in compiled_model.parameters())
                assert_all_close(base_grads, grads, atol=atol, rtol=rtol)

        return storage

    def common_test_model(
        self,
        model,
        example_args_collection,
        test_backward: bool,
        dynamic: bool,
        fullgraph: bool = True,
        onnx_export=None,
        impl="ort",
        verbose: int = 0,
        decompositions: bool = False,
        atol: float = 1e-4,
        rtol: float = 1e-4,
        mixed=False,
    ):
        storage = self._assert_model_numerically(
            model,
            example_args_collection,
            test_backward=test_backward,
            fullgraph=fullgraph,
            onnx_export=onnx_export,
            impl=impl,
            verbose=verbose,
            decompositions=decompositions,
            atol=atol,
            rtol=rtol,
            mixed=mixed,
        )
        self.assertIsInstance(storage, dict)

    @ignore_warnings((UserWarning, DeprecationWarning))
    @skipif_ci_windows("torch.compile not supported on Windows")
    def test_mlp_forward(self):
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
    def test_mlp_backward_ort(self):
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
            onnx_export="test_ort_mlp_backward_ort",
        )

    @ignore_warnings((UserWarning, DeprecationWarning))
    @skipif_ci_windows("torch.compile not supported on Windows")
    @unittest.skipIf(torch_min("2.2"), reason="missing kernel")
    def test_mlp_backward_ref(self):
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
            onnx_export="test_ort_mlp_backward_ref",
            impl="ref",
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
    def test_llama_decoder_forward(self):
        from experimental_experiment.torch_helper.llama_helper import get_llama_decoder

        input_dims = self.get_input_dims(False)
        model, example_args_collection = get_llama_decoder(input_dims=input_dims)
        self.common_test_model(
            model,
            example_args_collection,
            test_backward=False,
            dynamic=False,
            onnx_export="test_llama_decoder_forward",
        )

    @ignore_warnings((UserWarning, DeprecationWarning))
    @skipif_ci_windows("torch.compile not supported on Windows")
    def test_llama_decoder_backward(self):
        from experimental_experiment.torch_helper.llama_helper import get_llama_decoder

        input_dims = self.get_input_dims(False)
        model, example_args_collection = get_llama_decoder(input_dims=input_dims)
        self.common_test_model(
            model,
            example_args_collection,
            test_backward=True,
            dynamic=False,
            onnx_export="test_llama_decoder_backward",
        )

    @ignore_warnings((UserWarning, DeprecationWarning))
    @skipif_ci_windows("torch.compile not supported on Windows")
    def test_llama_attention_forward(self):
        from experimental_experiment.torch_helper.llama_helper import (
            get_llama_attention,
        )

        input_dims = self.get_input_dims(False)
        model, example_args_collection = get_llama_attention(input_dims=input_dims)
        self.common_test_model(
            model,
            example_args_collection,
            test_backward=False,
            dynamic=False,
            onnx_export="test_llama_attention_forward",
            impl="ref",
        )

    @ignore_warnings((UserWarning, DeprecationWarning))
    @skipif_ci_windows("torch.compile not supported on Windows")
    def test_llama_attention_backward(self):
        from experimental_experiment.torch_helper.llama_helper import (
            get_llama_attention,
        )

        input_dims = self.get_input_dims(False)
        model, example_args_collection = get_llama_attention(input_dims=input_dims)
        self.common_test_model(
            model,
            example_args_collection,
            test_backward=True,
            dynamic=False,
            onnx_export="test_llama_attention_backward",
            impl="ref",
        )

    @ignore_warnings((UserWarning, DeprecationWarning))
    @skipif_ci_windows("torch.compile not supported on Windows")
    @unittest.skipIf(torch_min("2.2"), reason="missing kernel")
    @unittest.skip("aten_embedding receives the inputs in the other way")
    def test_llama_model_forward(self):
        import torch
        from experimental_experiment.torch_helper.llama_helper import get_llama_model

        input_dims = self.get_input_dims(False)
        model, example_args_collection = get_llama_model(input_dims=input_dims)

        # onnxrt backend
        compiled_model = torch.compile(
            copy.deepcopy(model),
            backend="onnxrt",
            dynamic=False,
            fullgraph=True,
        )
        folder = "temp_llama_model_forward"
        with dump_onnx("llama_onnxrt", folder=folder, clean=True):
            compiled_model(*example_args_collection[0])

        self.common_test_model(
            model,
            example_args_collection,
            test_backward=False,
            dynamic=False,
            fullgraph=True,
            onnx_export="test_llama_model_forward",
            expected_graph_break=7,
            impl="ort",
        )

    @ignore_warnings((UserWarning, DeprecationWarning))
    @skipif_ci_windows("torch.compile not supported on Windows")
    @unittest.skipIf(torch_min("2.2"), reason="missing kernel")
    def test_llama_model_backward_forward(self):
        import torch
        from experimental_experiment.torch_helper.llama_helper import get_llama_model

        input_dims = self.get_input_dims(False)
        model, example_args_collection = get_llama_model(input_dims=input_dims)

        # onnxrt backend
        compiled_model = torch.compile(
            copy.deepcopy(model),
            backend="onnxrt",
            dynamic=False,
            fullgraph=True,
        )
        folder = "temp_llama_model_backward_forward"
        with dump_onnx("llama_onnxrt", folder=folder, clean=True):
            compiled_model(*example_args_collection[0])

        self.common_test_model(
            model,
            example_args_collection,
            test_backward=1,
            dynamic=False,
            fullgraph=True,
            onnx_export="test_llama_model_backward_forward",
            impl="ort",
        )

    @ignore_warnings((UserWarning, DeprecationWarning))
    @skipif_ci_windows("torch.compile not supported on Windows")
    @unittest.skipIf(torch_min("2.2"), reason="missing kernel")
    def test_llama_model_backward_undec(self):
        from experimental_experiment.torch_helper.llama_helper import get_llama_model

        input_dims = self.get_input_dims(False)
        model, example_args_collection = get_llama_model(input_dims=input_dims)
        self.common_test_model(
            model,
            example_args_collection,
            test_backward=True,
            dynamic=False,
            fullgraph=True,
            onnx_export="test_llama_model_backward",
        )

    @ignore_warnings((UserWarning, DeprecationWarning))
    @skipif_ci_windows("torch.compile not supported on Windows")
    @unittest.skipIf(torch_min("2.2"), reason="missing kernel")
    @unittest.skipIf(
        not implements("prims_collapsed_view"), reason="not yet implemented"
    )
    def test_llama_model_backward_decomposition(self):
        from experimental_experiment.torch_helper.llama_helper import get_llama_model

        input_dims = self.get_input_dims(False)
        model, example_args_collection = get_llama_model(input_dims=input_dims)
        self.common_test_model(
            model,
            example_args_collection,
            test_backward=True,
            dynamic=False,
            fullgraph=True,
            onnx_export="test_llama_model_backward_decomposition",
            decompositions=True,
        )

    @ignore_warnings((UserWarning, DeprecationWarning))
    @skipif_ci_windows("torch.compile not supported on Windows")
    @unittest.skipIf(torch_min("2.2"), reason="missing kernel")
    @unittest.skipIf(
        not implements("prims_collapsed_view"), reason="not yet implemented"
    )
    def test_llama_model_backward_forward_decomposition(self):
        from experimental_experiment.torch_helper.llama_helper import get_llama_model

        input_dims = self.get_input_dims(False)
        model, example_args_collection = get_llama_model(input_dims=input_dims)
        self.common_test_model(
            model,
            example_args_collection,
            test_backward=1,
            dynamic=False,
            fullgraph=True,
            onnx_export="test_llama_model_backward_decomposition",
            decompositions=True,
        )

    @ignore_warnings((UserWarning, DeprecationWarning))
    @skipif_ci_windows("torch.compile not supported on Windows")
    @unittest.skipIf(torch_min("2.2"), reason="missing kernel")
    def test_llama_model_backward_ref(self):
        from experimental_experiment.torch_helper.llama_helper import get_llama_model

        model, example_args_collection = get_llama_model(
            input_dims=[(2, 1024)] * 2,
            hidden_size=16,
            num_hidden_layers=1,
            vocab_size=1024,
            intermediate_size=16,
            max_position_embeddings=1024,
            num_attention_heads=2,
            _attn_implementation="eager",
        )
        self.common_test_model(
            model,
            example_args_collection,
            test_backward=True,
            dynamic=False,
            fullgraph=True,
            onnx_export="test_llama_model_backward",
            impl="ref",
            verbose=0,
            atol=3e-2,
        )

    @ignore_warnings((UserWarning, DeprecationWarning))
    @skipif_ci_windows("torch.compile not supported on Windows")
    @unittest.skipIf(torch_min("2.2"), reason="missing kernel")
    @unittest.skipIf(not has_cuda(), "cuda is needed for autocast")
    def test_llama_model_backward_forward_mixed(self):
        import torch
        from experimental_experiment.torch_helper.llama_helper import get_llama_model

        input_dims = self.get_input_dims(False)
        model, example_args_collection = get_llama_model(input_dims=input_dims)

        # onnxrt backend
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            compiled_model = torch.compile(
                copy.deepcopy(model),
                backend="onnxrt",
                dynamic=False,
                fullgraph=True,
            )

        folder = "test_llama_model_backward_forward_mixed"
        with dump_onnx("llama_onnxrt", folder=folder, clean=True):
            compiled_model(*example_args_collection[0])

        self.common_test_model(
            model,
            example_args_collection,
            test_backward=1,
            dynamic=False,
            fullgraph=True,
            onnx_export="test_llama_model_backward_forward_mixed",
            impl="ort",
            mixed=True,
        )

    @ignore_warnings((UserWarning, DeprecationWarning))
    @skipif_ci_windows("torch.compile not supported on Windows")
    @unittest.skipIf(torch_min("2.2"), reason="missing kernel")
    @unittest.skipIf(not has_cuda(), "cuda is needed for autocast")
    def test_llama_model_backward_mixed(self):
        import torch
        from experimental_experiment.torch_helper.llama_helper import get_llama_model

        input_dims = self.get_input_dims(False)
        model, example_args_collection = get_llama_model(input_dims=input_dims)

        # onnxrt backend
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            compiled_model = torch.compile(
                copy.deepcopy(model),
                backend="onnxrt",
                dynamic=False,
                fullgraph=True,
            )

        folder = "test_llama_model_backward_forward_mixed"
        with dump_onnx("llama_onnxrt", folder=folder, clean=True):
            compiled_model(*example_args_collection[0])

        self.common_test_model(
            model,
            example_args_collection,
            test_backward=True,
            dynamic=False,
            fullgraph=True,
            onnx_export="test_llama_model_backward_mixed",
            impl="ort",
            mixed=True,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
