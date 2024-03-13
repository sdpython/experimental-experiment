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
from experimental_experiment.torch_helper.dump_helper import assert_all_close
from experimental_experiment.torch_dynamo import (
    onnx_debug_backend,
    get_decomposition_table,
    filter_decomposition_table,
)


def torch_min(v: str) -> bool:
    import torch

    return pv.Version(torch.__version__) < pv.Version(v)


def has_cuda():
    import torch

    return torch.cuda.is_available()


class TestDynamoLlamaSdpa3(ExtTestCase):

    @classmethod
    def get_input_dims(cls, dynamic: bool):
        if dynamic:
            input_dims = ((2, 8), (4, 7), (9, 15))
        else:
            input_dims = ((9, 15), (9, 15), (9, 15))
        return input_dims

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
            **kwargs,
        )

        if test_backward:
            from torch._dynamo.backends.common import aot_autograd

            if decompositions:
                aot_compiler = aot_autograd(
                    fw_compiler=backend_debug,
                    decompositions=(
                        filter_decomposition_table()
                        if decompositions is True
                        else decompositions
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
        )
        self.assertIsInstance(storage, dict)

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
            onnx_export="test_llama_model_backward_sdpa",
            impl="ref",
            verbose=0,
            atol=3e-2,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
