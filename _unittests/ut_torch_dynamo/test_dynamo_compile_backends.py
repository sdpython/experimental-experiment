import copy
import inspect
import itertools
import unittest
import sys
import torch
from torch._dynamo.backends.common import aot_autograd
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
    requires_torch,
    requires_cuda,
    requires_onnxruntime_training,
)
from experimental_experiment.torch_models.dump_helper import assert_all_close
from experimental_experiment.torch_dynamo import onnx_debug_backend, onnx_custom_backend


class FuncModule(torch.nn.Module):
    def __init__(self, f, params=None):
        if params is None:
            params = ()
        super().__init__()
        self.f = f
        self.ppp = torch.nn.Parameter(torch.Tensor([1]))
        self.params = torch.nn.ParameterList(list(params))

    def forward(self, *args):
        f_args = list(itertools.chain(args, self.params))
        f_args[0] = f_args[0] + self.ppp
        res = self.f(*f_args)
        return res


class FuncModule0(torch.nn.Module):
    def __init__(self, f, params=None):
        if params is None:
            params = ()
        super().__init__()
        self.f = f
        self.ppp = torch.nn.Parameter(torch.Tensor([1]))
        self.params = torch.nn.ParameterList(list(params))

    def forward(self, *args):
        args = tuple(args[0] + self.ppp, *args[1:])
        res = self.f(*args)
        return res


class FuncModuleModule(torch.nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f
        self.mod = f
        self.ppp = torch.nn.Parameter(torch.Tensor([1]))

    def forward(self, *args):
        x = args[0] + self.ppp
        res = self.mod(x, *args[1:])
        return res


class TestDynamoCompileBackend(ExtTestCase):
    def setUp(self):
        import torch

        torch._dynamo.reset()

    def _assertONNX(
        self,
        backend: str,
        f,
        args,
        onnx_export: str = "",
        params=None,
        fullgraph: bool = True,
        atol=1e-6,
        rtol=1e-6,
        opset_version=None,
        test_backward=True,
        #
        input_names=None,
        dynamic_axes=False,
        verbose=0,
        input_index=0,
    ):
        if sys.platform == "win32":
            raise unittest.SkipTest("Windows not supported yet.")
        assert isinstance(onnx_export, str), f"Export onnx is wrong for f={f}"
        torch._dynamo.reset()
        if isinstance(args, torch.Tensor):
            args = [args]
        if params is None:
            params = ()
        if isinstance(f, torch.nn.Module):
            model = FuncModuleModule(f)
        elif input_index is None:
            model = FuncModule(f, params)
        else:
            assert input_index == 0, f"Not implemented for input_index={input_index}"
            model = FuncModule0(f, params)
        model.eval()
        storage = {}

        if backend == "debug":
            args = tuple(a.to(torch.device("cpu")) for a in args)
            backend_onnx = lambda *args, **kwargs: onnx_debug_backend(  # noqa: E731
                *args,
                target_opset=opset_version,
                storage=storage,
                backend="ort",
                verbose=verbose,
                **kwargs,
            )
        elif backend == "fast":
            max_device = max(i.get_device() for i in args)
            if max_device >= 0:
                model = model.to(torch.device("cuda"))

            backend_onnx = lambda *args, **kwargs: onnx_custom_backend(  # noqa: E731
                *args,
                target_opset=opset_version,
                storage=storage,
                verbose=verbose,
                **kwargs,
            )
        else:
            raise AssertionError(f"unexpected value {backend!r}")

        if test_backward:
            if verbose:
                print("-- test_backward")
            aot_compiler = aot_autograd(fw_compiler=backend_onnx)

            compiled_model = torch.compile(
                copy.deepcopy(model),
                backend=aot_compiler,
                dynamic=dynamic_axes,
                fullgraph=fullgraph,
            )

            if verbose > 1:
                print("-- torch model", len(args))
            baseline_result = model(*args)
            if verbose > 1:
                print("-- done")
                print("-- compiled_model", len(args))
            result = compiled_model(*args)
            if verbose > 1:
                print("-- done", len(args))

            if isinstance(baseline_result, torch.Tensor):
                assert_all_close(baseline_result, result, atol=atol, rtol=rtol)
                baseline_result.sum().backward()
                result.sum().backward()

                base_grads = tuple(_.grad for _ in model.parameters())
                grads = tuple(_.grad for _ in compiled_model.parameters())
                self.assertEqual(len(base_grads), len(grads))
                assert_all_close(base_grads, grads, atol=atol, rtol=rtol)
                assert len(grads) > 0, "No gradient was checked"
            else:
                raise AssertionError(f"Unexpected type {type(baseline_result)}.")
        else:
            # forward only
            if verbose:
                print("-- forward")
            compiled_model = torch.compile(
                copy.deepcopy(model),
                backend=backend_onnx,
                dynamic=dynamic_axes,
                fullgraph=fullgraph,
            )
            if verbose > 1:
                print("-- torch model", len(args))
            baseline_result = model(*args)
            if verbose > 1:
                print("-- done")
                print("-- compiled_model", len(args))
            result = compiled_model(*args)
            if verbose > 1:
                print("-- done")
            assert_all_close(baseline_result, result, atol=atol, rtol=rtol)

    def assertONNX(
        self,
        f,
        args,
        onnx_export: str = "",
        params=None,
        fullgraph: bool = True,
        atol=1e-6,
        rtol=1e-6,
        opset_version=None,
        test_backward=True,
        #
        input_names=None,
        dynamic_axes=False,
        verbose=0,
        backends=None,
    ):
        if backends is None:
            backends = ["fast", "debug"]
        for backend in backends:
            if verbose:
                print(f"---- backend={backend!r}")
            with self.subTest(backend=backend):
                onnx_export_ = "" if onnx_export else f"{onnx_export}_{backend}"
                self._assertONNX(
                    backend,
                    f,
                    args,
                    onnx_export=onnx_export_,
                    params=params,
                    fullgraph=fullgraph,
                    atol=atol,
                    rtol=rtol,
                    opset_version=opset_version,
                    test_backward=test_backward,
                    input_names=input_names,
                    dynamic_axes=dynamic_axes,
                    verbose=verbose,
                )

    @requires_torch("2.2.1", "onnxrt not fully implemented")
    @ignore_warnings((UserWarning, RuntimeWarning, DeprecationWarning))
    @requires_onnxruntime_training()
    def test_aaaa_forward_cpu(self):
        x = torch.rand(3, 4, requires_grad=True)

        self.assertONNX(
            lambda x: x.cos(),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
            verbose=0,
            test_backward=False,
        )

    @requires_torch("2.2.1", "onnxrt not fully implemented")
    @ignore_warnings((UserWarning, RuntimeWarning, DeprecationWarning))
    @requires_onnxruntime_training()
    def test_aaaa_backward_cpu(self):
        x = torch.rand(3, 4, requires_grad=True)

        self.assertONNX(
            lambda x: x.cos(),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
            verbose=0,
            test_backward=True,
        )

    @requires_torch("2.2.1", "onnxrt not fully implemented")
    @requires_cuda()
    @ignore_warnings((UserWarning, RuntimeWarning, DeprecationWarning))
    def test_aaaa_backward_cuda(self):
        x = torch.rand(3, 4, requires_grad=True).to(torch.device("cuda"))

        self.assertONNX(
            lambda x: x.cos(),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
            verbose=0,
            test_backward=True,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
