import copy
import unittest
from typing import Optional
from experimental_experiment.ext_test_case import ExtTestCase
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
            local_aot_ort, local_ort = make_aot_ort(dynamic=dynamic, rewrite=True)
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
