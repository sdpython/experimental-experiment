import copy
import unittest
from typing import Optional
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
    skipif_ci_windows,
    requires_torch,
)
from experimental_experiment.torch_models.dump_helper import assert_all_close
from experimental_experiment.torch_dynamo import (
    onnx_debug_backend,
    get_decomposition_table,
    filter_decomposition_table,
)


class TestDynamoLlama(ExtTestCase):

    @classmethod
    def setUp(cls):
        import torch

        if hasattr(torch._dynamo.variables.misc, "LoggingLoggerVariable"):
            cls._old_value = torch._dynamo.variables.misc.LoggingLoggerVariable.call_method
            torch._dynamo.variables.misc.LoggingLoggerVariable.call_method = lambda *_, **__: None

    @classmethod
    def tearDown(cls):
        import torch

        if hasattr(torch._dynamo.variables.misc, "LoggingLoggerVariable"):
            torch._dynamo.variables.misc.LoggingLoggerVariable.call_method = cls._old_value

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
        raise_list=None,
        enable_pattern="default",
    ):
        import torch

        assert onnx_export, "No export name was given"

        storage = {}
        backend_debug = lambda *args, **kwargs: onnx_debug_backend(  # noqa: E731
            *args,
            # dump_prefix=os.path.join(folder, "llama_debug"),
            backend=impl,
            target_opset=18,
            storage=storage,
            verbose=verbose,
            dump_prefix=onnx_export,
            raise_list=raise_list,
            enable_pattern=enable_pattern,
            optimize=bool(enable_pattern),
            **kwargs,
        )

        if test_backward:
            from torch._dynamo.backends.common import aot_autograd

            if decompositions:
                aot_compiler = aot_autograd(
                    fw_compiler=backend_debug,
                    decompositions=(
                        filter_decomposition_table() if decompositions is True else decompositions
                    ),
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
        raise_list=None,
        enable_pattern="default",
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
            raise_list=raise_list,
            enable_pattern=enable_pattern,
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
            MLP(),
            example_args_collection,
            test_backward=False,
            dynamic=False,
            onnx_export="test_ort_mlp",
        )

    @ignore_warnings((UserWarning, DeprecationWarning))
    @skipif_ci_windows("torch.compile not supported on Windows")
    def test_tensorrt(self):
        import torch

        try:
            import tensorrt  # noqa: F401
            import torch_tensorrt
        except ImportError as e:
            raise unittest.SkipTest(f"Cannot import tensorrt due to {e}.")

        class MLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(2, 1024, bias=True)
                self.fc2 = torch.nn.Linear(1024, 2048, bias=True)
                self.fc3 = torch.nn.Linear(2048, 2, bias=True)

            def forward(self, tensor_x: torch.Tensor):
                tensor_x = self.fc1(tensor_x)
                tensor_x = torch.sigmoid(tensor_x)
                tensor_x = self.fc2(tensor_x)
                tensor_x = torch.sigmoid(tensor_x)
                tensor_x = self.fc3(tensor_x)
                tensor_x = torch.sigmoid(tensor_x)
                return tensor_x

        # with static shape (dynamic=False), the conversion to onnx is done
        # every time the batch size changes
        batch_sizes = [3, 3, 3, 3, 3]

        example_args_collection = [
            (torch.randn(batch, 2, dtype=torch.float32).cuda(),) for batch in batch_sizes
        ]

        model = MLP().eval().cuda()
        inputs = example_args_collection[0]
        exp_program = torch.export.export(model, tuple(inputs))
        trt_gm = torch_tensorrt.dynamo.compile(exp_program, inputs)
        trt_gm(*inputs)

    @ignore_warnings((UserWarning, DeprecationWarning))
    @skipif_ci_windows("torch.compile not supported on Windows")
    @requires_torch("2.2", "missing kernel")
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
    @requires_torch("2.2", "missing kernel")
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
    @requires_torch("2.2", "missing kernel")
    @unittest.skip("requires silu_backward")
    def test_llama_model_backward_undec(self):
        input_dims = self.get_input_dims(False)
        model, example_args_collection = self.get_llama_model(input_dims=input_dims)
        self.common_test_model(
            model,
            example_args_collection,
            test_backward=True,
            dynamic=False,
            fullgraph=True,
            onnx_export="test_llama_model_backward",
            atol=5e-4,
        )

    @ignore_warnings((UserWarning, DeprecationWarning))
    @skipif_ci_windows("torch.compile not supported on Windows")
    @requires_torch("2.2", "missing kernel")
    @unittest.skip("requires silu_backward")
    def test_llama_model_backward_ref(self):
        model, example_args_collection = self.get_llama_model(
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
            atol=5e-2,
        )

    @ignore_warnings((UserWarning, DeprecationWarning))
    @skipif_ci_windows("torch.compile not supported on Windows")
    def test_tensorrt_llama(self):
        try:
            import tensorrt  # noqa: F401
            import torch_tensorrt
        except ImportError as e:
            raise unittest.SkipTest(f"Cannot import tensorrt due to {e}.")
        import torch

        input_dims = self.get_input_dims(False)
        model, example_args_collection = self.get_llama_model(input_dims=input_dims)
        model = model.cuda()

        inputs = tuple(_.cuda() for _ in example_args_collection[0])
        exp_program = torch.export.export(model, inputs)
        trt_gm = torch_tensorrt.dynamo.compile(exp_program, inputs)
        trt_gm(*inputs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
