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
    skipif_transformers,
)
from experimental_experiment.torch_models.dump_helper import assert_all_close
from experimental_experiment.torch_dynamo import (
    onnx_debug_backend,
    onnx_custom_backend,
    get_decomposition_table,
)
from experimental_experiment.torch_models.training_helper import make_aot_ort


def implements(name: str) -> bool:
    import experimental_experiment.torch_interpreter._aten_functions as atf

    return hasattr(atf, name)


class TestDynamoLlamaDynamic(ExtTestCase):
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

    @classmethod
    def get_input_dims(cls, dynamic: bool):
        if dynamic:
            input_dims = ((2, 8), (4, 7), (9, 15))
        else:
            input_dims = ((9, 15), (9, 15), (9, 15))
        return input_dims

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
        dynamic: bool = True,
        atol: float = 1e-4,
        rtol: float = 1e-4,
        onnx_export: Optional[str] = None,
        impl="ort",
        verbose: int = 0,
        decompositions=False,
        mixed=False,
        raise_list=None,
        dump_prefix=None,
        disable_pattern=None,
    ):
        import torch

        assert onnx_export, "No export name was given"

        storage = {}

        if impl == "onnxrt":
            _local_aot_ort, local_ort = make_aot_ort(dynamic=dynamic, rewrite=True)
            compiled_model = torch.compile(copy.deepcopy(model), backend=local_ort)
        else:
            if impl == "fast":
                backend_debug = lambda *args, **kwargs: onnx_custom_backend(  # noqa: E731
                    *args,
                    backend="ort",
                    target_opset=18,
                    storage=storage,
                    verbose=verbose,
                    dump_prefix=dump_prefix,
                    disable_pattern=disable_pattern,
                    **kwargs,
                )
            else:
                backend_debug = lambda *args, **kwargs: onnx_debug_backend(  # noqa: E731
                    *args,
                    # dump_prefix=os.path.join(folder, "llama_debug"),
                    backend=impl,
                    target_opset=18,
                    storage=storage,
                    verbose=verbose,
                    raise_list=raise_list,
                    dump_prefix=dump_prefix,
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
                        fw_compiler=backend_debug,
                        decompositions=get_decomposition_table(),
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
        dump_prefix=None,
        disable_pattern=None,
    ):
        storage = self._assert_model_numerically(
            model,
            example_args_collection,
            test_backward=test_backward,
            dynamic=dynamic,
            fullgraph=fullgraph,
            onnx_export=onnx_export,
            impl=impl,
            verbose=verbose,
            decompositions=decompositions,
            atol=atol,
            rtol=rtol,
            mixed=mixed,
            raise_list=raise_list,
            dump_prefix=dump_prefix,
            disable_pattern=disable_pattern,
        )
        self.assertIsInstance(storage, dict)
        return storage

    @ignore_warnings((UserWarning, DeprecationWarning))
    @skipif_ci_windows("torch.compile not supported on Windows")
    @requires_torch("2.3", "missing kernel")
    def __test_llama_model_b_forward_dynamic(self):
        from experimental_experiment.torch_models.llama_helper import get_llama_model

        input_dims = self.get_input_dims(True)
        model, example_args_collection = get_llama_model(input_dims=input_dims)

        self.common_test_model(
            model,
            example_args_collection,
            test_backward=1,
            dynamic=True,
            fullgraph=True,
            onnx_export="tt_test_llama_model_backward_forward_dynamic",
            impl="ort",
        )

    @ignore_warnings((UserWarning, DeprecationWarning))
    @skipif_ci_windows("torch.compile not supported on Windows")
    @requires_torch("2.2", "missing kernel")
    @requires_cuda()
    def test_llama_model_backward_mixed_dynamic_debug(self):
        from experimental_experiment.torch_models.llama_helper import get_llama_model

        input_dims = self.get_input_dims(True)
        model, example_args_collection = get_llama_model(input_dims=input_dims)

        self.common_test_model(
            model,
            example_args_collection,
            test_backward=True,
            dynamic=True,
            fullgraph=True,
            onnx_export="tt_test_llama_model_backward_mixed_dynamic",
            impl="ref",
            mixed=True,
            # verbose=10,
            dump_prefix="tt_temp_llama_model_backward_mixed_dynamic",
        )

    @ignore_warnings((UserWarning, DeprecationWarning))
    @skipif_ci_windows("torch.compile not supported on Windows")
    @requires_torch("2.2", "missing kernel")
    @requires_cuda()
    def test_llama_model_backward_mixed_dynamic_fast_backend(self):
        from experimental_experiment.torch_models.llama_helper import get_llama_model

        input_dims = self.get_input_dims(True)
        model, example_args_collection = get_llama_model(input_dims=input_dims)

        self.common_test_model(
            model,
            example_args_collection,
            test_backward=True,
            dynamic=True,
            fullgraph=True,
            onnx_export="tt_test_llama_model_backward_mixed_dynamic_fastbackend",
            impl="fast",
            mixed=True,
            verbose=0,
            dump_prefix="tt_temp_llama_model_backward_mixed_dynamic_fastbackend",
        )

    @ignore_warnings((UserWarning, DeprecationWarning))
    @skipif_ci_windows("torch.compile not supported on Windows")
    @requires_torch("2.2", "missing kernel")
    @requires_cuda()
    def __test_llama_model_backward_dynamic_fast_backend(self):
        from experimental_experiment.torch_models.llama_helper import get_llama_model

        input_dims = self.get_input_dims(True)
        model, example_args_collection = get_llama_model(input_dims=input_dims)

        self.common_test_model(
            model,
            example_args_collection,
            test_backward=True,
            dynamic=True,
            fullgraph=True,
            onnx_export="tt_test_llama_model_backward_dynamic_fastbackend",
            impl="fast",
            mixed=False,
            verbose=0,
            dump_prefix="tt_temp_llama_model_backward_dynamic_fastbackend",
        )

    @ignore_warnings((UserWarning, DeprecationWarning))
    @skipif_ci_windows("torch.compile not supported on Windows")
    @requires_torch("2.2", "missing kernel")
    @requires_cuda()
    def test_llama_model_backward_mixed_dynamic_fast_backend_1024(self):
        from experimental_experiment.torch_models.llama_helper import get_llama_model

        input_dims = [(2, 1014)]
        model, example_args_collection = get_llama_model(
            input_dims=input_dims,
            num_hidden_layers=2,
            hidden_size=1024,
            vocab_size=1024,
            intermediate_size=1024,
            max_position_embeddings=1024,
            num_attention_heads=2,
        )

        self.common_test_model(
            model,
            example_args_collection,
            test_backward=True,
            dynamic=True,
            fullgraph=True,
            onnx_export="tt_test_llama_model_backward_mixed_dynamic_fastbackend_1024",
            impl="fast",
            mixed=False,
            verbose=0,
            dump_prefix="tt_test_llama_model_backward_mixed_dynamic_fastbackend_1024",
            disable_pattern="default",
            atol=(1e-2, 0.8),
        )

    @ignore_warnings((UserWarning, DeprecationWarning))
    @skipif_ci_windows("torch.compile not supported on Windows")
    @requires_torch("2.5", "missing kernel")
    @requires_cuda()
    @skipif_transformers(
        "4.38.2",
        "INVALID_ARGUMENT : Failed to load model with error:, "
        "Graph output (aten_mean_dim_267_dim_2) does not exist in the graph.",
    )
    @requires_onnxscript(
        "0.3",
        "something is off, it works when run independently "
        "from the other tests, it fails otherwise",
    )
    def test_llama_model_backward_mixed_dynamic_onnxrt_1024(self):
        from experimental_experiment.torch_models.llama_helper import get_llama_model

        input_dims = [(2, 1024)]
        model, example_args_collection = get_llama_model(
            input_dims=input_dims,
            num_hidden_layers=2,
            hidden_size=1024,
            vocab_size=1024,
            intermediate_size=1024,
            max_position_embeddings=1024,
            num_attention_heads=2,
        )

        self.common_test_model(
            model,
            example_args_collection,
            test_backward=True,
            dynamic=True,
            fullgraph=True,
            onnx_export="tt_test_llama_model_backward_mixed_dynamic_onnxrt_1024",
            impl="onnxrt",
            mixed=False,
            verbose=0,
            dump_prefix="tt_test_llama_model_backward_mixed_dynamic_onnxrt_1024",
            atol=(1e-2, 0.8),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
