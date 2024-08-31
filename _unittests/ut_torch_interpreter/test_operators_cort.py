"""
to fail on error, use::

    clear&&EXPDORAISE=1 \\
    python _unittests/ut_torch_interpreter/test_operators_cort.py -f -k relu

or::

    clear&&EXPDORAISE=1 \\
    python _unittests/ut_torch_interpreter/test_operators_cort.py -f
"""

import copy
import inspect
import itertools
import operator
import os
import unittest
import sys
import warnings
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
from torch.autograd import Function
from torch.nn import functional, Module, Parameter
from torch._dynamo.backends.common import aot_autograd
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
    requires_torch,
)
from experimental_experiment.torch_interpreter import FunctionNotFoundError
from experimental_experiment.torch_models.dump_helper import assert_all_close
from experimental_experiment.torch_dynamo import (
    onnx_debug_backend,
    get_decomposition_table,
)

BATCH_SIZE = 2
RNN_BATCH_SIZE = 7
RNN_SEQUENCE_LENGTH = 11
RNN_INPUT_SIZE = 5
RNN_HIDDEN_SIZE = 3
DYNAMIC_SHAPE_SUPPORTED = False
DICT_SUPPORTED = False
OP_BOOL_SUPPORTED = False


class FuncModule(Module):
    def __init__(self, f, params=None, dtype=torch.float32):
        if params is None:
            params = ()
        super().__init__()
        self.f = f
        rg = dtype == torch.float32
        val = torch.ones((1,), requires_grad=rg, dtype=dtype)
        self.ppp = Parameter(val, requires_grad=rg)
        val2 = torch.ones((1,), requires_grad=rg, dtype=dtype)
        self.ppp2 = Parameter(val2, requires_grad=rg)
        self.params = nn.ParameterList(list(params))

    def forward(self, *args):
        f_args = list(itertools.chain(args, self.params))
        f_args[0] = f_args[0] * self.ppp
        res = self.f(*f_args) * self.ppp2
        return res


class FuncModuleSimple(Module):
    def __init__(self, f, params=None, dtype=torch.float32):
        if params is None:
            params = ()
        super().__init__()
        self.f = f
        rg = dtype == torch.float32
        val = torch.ones((1,), requires_grad=rg, dtype=dtype)
        self.ppp = Parameter(val, requires_grad=rg)
        self.params = nn.ParameterList(list(params))

    def forward(self, *args):
        f_args = list(itertools.chain(args, self.params))
        f_args[0] = f_args[0] * self.ppp
        res = self.f(*f_args)
        return res


class FuncModule0(Module):
    def __init__(self, f):
        super().__init__()
        self.f = f
        self.ppp = Parameter(torch.Tensor([1]).to(torch.float32))
        self.ppp2 = Parameter(torch.Tensor([2]).to(torch.float32))

    def forward(self, *args):
        if isinstance(args[0], tuple):
            args = ((args[0][0] * self.ppp, *args[0][1:]),)
            res = self.f(*args) * self.ppp2
            return res
        else:
            args = (args[0] * self.ppp, *args[1:])
            res = self.f(*args) * self.ppp2
            return res


class FuncModule1(Module):
    def __init__(self, f):
        super().__init__()
        self.f = f
        self.ppp = Parameter(torch.Tensor([1]).to(torch.float32))

    def forward(self, *args):
        args = (args[0], args[1] * self.ppp, *args[2:])
        res = self.f(*args)
        return res


class FuncModuleModule(Module):
    def __init__(self, f):
        super().__init__()
        self.f = f
        self.mod = f
        self.ppp = Parameter(torch.Tensor([1]))

    def forward(self, *args):
        x = args[0] * self.ppp
        res = self.mod(x, *args[1:])
        return res


class TestOperatorsCort(ExtTestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()

    def assertONNX(
        self,
        f,
        args,
        onnx_export: str,
        params=None,
        fullgraph: bool = True,
        atol=1e-6,
        rtol=1e-6,
        opset_version=None,
        test_backward=True,
        impl="ort",
        #
        input_names=None,
        dynamic_axes=False,
        keep_initializers_as_inputs=None,
        training=None,
        input_index: Optional[int] = None,
        square_loss=False,
        use_decomposition=False,
        verbose=0,
        raise_list=None,
        save_onnx=False,
        optimize=True,
        intermediate=False,
    ):
        if sys.platform == "win32":
            raise unittest.SkipTest("Windows not supported yet.")
        assert isinstance(onnx_export, str), f"Export onnx is wrong for f={f}"
        if isinstance(args, torch.Tensor):
            args = [args]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            if input_index == "all":
                assert params is None
                if verbose:
                    print("[assertONNX] +None")
                model = f
            elif isinstance(f, nn.Module):
                if verbose:
                    print("[assertONNX] +FuncModuleModule")
                model = FuncModuleModule(f)
            elif input_index == "simple":
                if params is None:
                    params = ()
                if verbose:
                    print("[assertONNX] +FuncModuleSimple")
                model = FuncModuleSimple(f, params, dtype=args[0].dtype)
            elif input_index is None:
                if params is None:
                    params = ()
                if verbose:
                    print("[assertONNX] +FuncModule")
                model = FuncModule(f, params, dtype=args[0].dtype)
            elif input_index == 0:
                assert params is None, f"not implemented with params={params}"
                if verbose:
                    print("[assertONNX] +FuncModule0")
                model = FuncModule0(f)
            elif input_index == 1:
                assert params is None, f"not implemented with params={params}"
                if verbose:
                    print("[assertONNX] +FuncModule1")
                model = FuncModule1(f)
            else:
                assert input_index in (
                    0,
                    1,
                ), f"Not implemented for input_index={input_index}"
            model.eval()
            storage = {}

            if verbose >= 10:
                for i, arg in enumerate(args):
                    print(f"[assertONNX] i={i}, arg={arg.dtype}:{arg.shape}")

            backend_debug = lambda *args, **kwargs: onnx_debug_backend(  # noqa: E731
                *args,
                target_opset=opset_version,
                storage=storage,
                backend=impl,
                raise_list=raise_list,
                optimize=optimize,
                verbose=(verbose, 10) if intermediate else verbose,
                **kwargs,
            )

            if intermediate:
                from experimental_experiment.torch_dynamo import dynger_backend

                backend_dynger = lambda *args, **kwargs: dynger_backend(  # noqa: E731
                    *args, optimize=optimize, verbose=10, **kwargs
                )
                aot_compiler_dynger = (
                    aot_autograd(
                        fw_compiler=backend_dynger,
                        decompositions=(
                            get_decomposition_table()
                            if use_decomposition is True
                            else use_decomposition
                        ),
                    )
                    if use_decomposition
                    else aot_autograd(fw_compiler=backend_dynger)
                )
                compiled_model_dynger = torch.compile(
                    copy.deepcopy(model),
                    backend=aot_compiler_dynger,
                    dynamic=dynamic_axes not in (None, False),
                    fullgraph=fullgraph,
                )
                results_dynger = compiled_model_dynger(*args)
                if test_backward:
                    results_dynger.sum().backward()

            if test_backward:
                # forward/backward
                aot_compiler = (
                    aot_autograd(
                        fw_compiler=backend_debug,
                        decompositions=(
                            get_decomposition_table()
                            if use_decomposition is True
                            else use_decomposition
                        ),
                    )
                    if use_decomposition
                    else aot_autograd(fw_compiler=backend_debug)
                )

                compiled_model = torch.compile(
                    copy.deepcopy(model),
                    backend=aot_compiler,
                    dynamic=dynamic_axes not in (None, False),
                    fullgraph=fullgraph,
                )

                baseline_result = model(*args)
                try:
                    result = compiled_model(*args)
                except torch._dynamo.exc.BackendCompilerFailed as e:
                    if not os.environ.get(
                        "EXPDORAISE", False
                    ) and "FunctionNotFoundError" in str(e):
                        raise unittest.SkipTest(f"MISSING FOR FORWARD {e}")
                    raise

                if isinstance(baseline_result, torch.Tensor):
                    assert_all_close(
                        baseline_result,
                        result,
                        atol=atol,
                        rtol=rtol,
                        msg="FORWARD-BACKWARD",
                    )

                    if square_loss:
                        (baseline_result.sum() ** 2).backward()
                        try:
                            (result.sum() ** 2).backward()
                        except FunctionNotFoundError as e:
                            if not os.environ.get("EXPDORAISE", False):
                                raise unittest.SkipTest(f"MISSING FOR BACKWARD {e}")
                            raise
                    else:
                        baseline_result.sum().backward()
                        try:
                            result.sum().backward()
                        except FunctionNotFoundError as e:
                            if not os.environ.get("EXPDORAISE", False):
                                raise unittest.SkipTest(f"MISSING FOR BACKWARD {e}")
                            assert (
                                len(storage["instance"]) == 1
                            ), f"Unexpected number of instance {len(storage['instance'])}"
                            instance = storage["instance"][0]
                            forward_onnx = instance["onnx"]
                            folder = "dump_test_operators_forward"
                            if not os.path.exists(folder):
                                os.mkdir(folder)
                            grad_name = os.path.join(
                                folder, f"{onnx_export}_forward.onnx"
                            )
                            with open(grad_name, "wb") as f:
                                f.write(forward_onnx.SerializeToString())
                            # gradient from onnxruntime
                            from experimental_experiment.gradient.grad_helper import (
                                DerivativeOptions,
                                onnx_derivative,
                            )

                            grad = onnx_derivative(
                                forward_onnx,
                                options=DerivativeOptions.KeepYieldOp,
                                verbose=1,
                            )
                            with open(
                                os.path.join(
                                    folder, f"{onnx_export}_ort_yield_grad.onnx"
                                ),
                                "wb",
                            ) as f:
                                f.write(grad.SerializeToString())

                            grad = onnx_derivative(
                                forward_onnx,
                                options=DerivativeOptions.Zero,
                                verbose=1,
                            )
                            with open(
                                os.path.join(folder, f"{onnx_export}_ort_grad.onnx"),
                                "wb",
                            ) as f:
                                f.write(grad.SerializeToString())
                            raise

                    if save_onnx:
                        assert storage["instance"]
                        for i, inst in enumerate(storage["instance"]):
                            with open(f"{onnx_export}_{i}.onnx", "wb") as f:
                                f.write(inst["onnx"].SerializeToString())

                    base_grads = tuple(_.grad for _ in model.parameters())
                    grads = tuple(_.grad for _ in compiled_model.parameters())
                    self.assertEqual(len(base_grads), len(grads))
                    assert_all_close(
                        base_grads, grads, atol=atol, rtol=rtol, msg="BACKWARD"
                    )
                    assert len(grads) > 0, "No gradient was checked"
                else:
                    if save_onnx:
                        assert storage["instance"]
                        for i, inst in enumerate(storage["instance"]):
                            with open(f"{onnx_export}_{i}.onnx", "wb") as f:
                                f.write(inst["onnx"].SerializeToString())

                    # tuple
                    assert_all_close(
                        baseline_result,
                        result,
                        atol=atol,
                        rtol=rtol,
                        msg="FORWARD-BACKWARD",
                    )

                    if square_loss:
                        (baseline_result[0].sum() ** 2).backward()
                        try:
                            (result[0].sum() ** 2).backward()
                        except FunctionNotFoundError as e:
                            if not os.environ.get("EXPDORAISE", False):
                                raise unittest.SkipTest(f"MISSING FOR BACKWARD {e}")
                            raise
                    else:
                        baseline_result[0].sum().backward()
                        try:
                            result[0].sum().backward()
                        except FunctionNotFoundError as e:
                            if not os.environ.get("EXPDORAISE", False):
                                raise unittest.SkipTest(f"MISSING FOR BACKWARD {e}")
                            raise

                    base_grads = tuple(_.grad for _ in model.parameters())
                    grads = tuple(_.grad for _ in compiled_model.parameters())
                    self.assertEqual(len(base_grads), len(grads))
                    assert_all_close(
                        base_grads, grads, atol=atol, rtol=rtol, msg="BACKWARD"
                    )
                    assert len(grads) > 0, "No gradient was checked"

            else:
                assert (
                    not use_decomposition
                ), "not implemented for use_decomposition=True"
                # forward only
                compiled_model = torch.compile(
                    copy.deepcopy(model),
                    backend=backend_debug,
                    dynamic=dynamic_axes not in (None, False),
                    fullgraph=fullgraph,
                )
                baseline_result = model(*args)
                result = compiled_model(*args)
                if save_onnx:
                    assert storage["instance"]
                    for i, inst in enumerate(storage["instance"]):
                        with open(f"{onnx_export}_{i}.onnx", "wb") as f:
                            f.write(inst["onnx"].SerializeToString())
                assert_all_close(
                    baseline_result, result, atol=atol, rtol=rtol, msg="FORWARD"
                )

    @ignore_warnings(UserWarning)
    def test_aaa(self):
        x = torch.rand(3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: x.acos(),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
            verbose=0,
        )

    def test_basic_static(self):
        x = torch.tensor([0.4], requires_grad=True)
        y = torch.tensor([0.7], requires_grad=True)
        self.assertONNX(
            lambda x, y: -torch.sigmoid(torch.tanh(x * (x + y))),
            (x, y),
            onnx_export=inspect.currentframe().f_code.co_name,
            dynamic_axes=(False, True),
            impl="ref",
            save_onnx=False,
        )

    def test_basic_dynamic(self):
        x = torch.tensor([0.4], requires_grad=True)
        y = torch.tensor([0.7], requires_grad=True)
        self.assertONNX(
            lambda x, y: -torch.sigmoid(torch.tanh(x * (x + y))),
            (x, y),
            onnx_export=inspect.currentframe().f_code.co_name,
            dynamic_axes={"x": {0: "batch"}},
        )

    def test_view(self):
        x = torch.tensor([0.0], requires_grad=True)
        self.assertONNX(
            lambda x: x.view(1, 1), x, onnx_export=inspect.currentframe().f_code.co_name
        )

    def test_view_dynamic(self):
        x = torch.tensor([0.0], requires_grad=True)
        self.assertONNX(
            lambda x: x.view(1, 1),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
            dynamic_axes={"x": {0: "batch"}},
        )

    def test_index_i(self):
        x = torch.tensor([[0.0]], requires_grad=True)
        self.assertONNX(
            lambda x: x[0], x, onnx_export=inspect.currentframe().f_code.co_name
        )

    def test_index_tensor(self):
        x = torch.arange(12, requires_grad=True, dtype=torch.float32).reshape((-1, 4))
        y = x[[0, 2]]
        assert y.shape == (2, 4), f"{y.shape}"
        self.assertONNX(
            lambda x: x[[0, 2]],
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
            test_backward=False,
        )

    def test_index_tensor_f(self):
        x = torch.arange(12, requires_grad=True, dtype=torch.float32).reshape((-1, 4))
        y = torch.index_select(x.clone(), 0, torch.tensor([0, 2]))
        assert y.shape == (2, 4)
        self.assertONNX(
            lambda x: torch.index_select(x.clone(), 0, torch.tensor([0, 2])),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
            test_backward=False,
        )

    def test_index_select_ort(self):
        x = torch.arange(12, requires_grad=True, dtype=torch.float32).reshape((-1, 4))
        self.assertONNX(
            lambda x: torch.index_select(x.clone(), 1, torch.tensor([0, 2])),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
            test_backward=False,
            impl="ort",
        )

    def test_index_select_ref(self):
        x = torch.arange(12, requires_grad=True, dtype=torch.float32).reshape((-1, 4))
        self.assertONNX(
            lambda x: torch.index_select(x.clone(), 1, torch.tensor([0, 2])),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
            test_backward=False,
            impl="ref",
        )

    def test_type_as(self):
        x = torch.tensor([0.0], requires_grad=True)
        self.assertONNX(
            lambda x: x.type_as(x), x, onnx_export=inspect.currentframe().f_code.co_name
        )

    @ignore_warnings(UserWarning)
    def test_addconstant(self):
        x = torch.randn(2, 3, requires_grad=True).double()
        self.assertONNX(
            lambda x: x + 1, x, onnx_export=inspect.currentframe().f_code.co_name
        )

    @ignore_warnings(UserWarning)
    def test_add_broadcast(self):
        x = torch.randn(2, 3, requires_grad=True).double()
        y = torch.randn(3, requires_grad=True).double()
        r = x + y
        print(r.type, r.shape)
        self.assertONNX(
            operator.add, (x, y), onnx_export=inspect.currentframe().f_code.co_name
        )

    @ignore_warnings(UserWarning)
    def test_add_left_broadcast(self):
        x = torch.randn(3, requires_grad=True).double()
        y = torch.randn(2, 3, requires_grad=True).double()
        self.assertONNX(
            operator.add, (x, y), onnx_export=inspect.currentframe().f_code.co_name
        )

    @ignore_warnings(UserWarning)
    def test_add_size1_broadcast(self):
        x = torch.randn(2, 3, requires_grad=True).double()
        y = torch.randn(2, 1, requires_grad=True).double()
        self.assertONNX(
            operator.add, (x, y), onnx_export=inspect.currentframe().f_code.co_name
        )

    @ignore_warnings(UserWarning)
    def test_add_size1_right_broadcast(self):
        x = torch.randn(2, 3, requires_grad=True).double()
        y = torch.randn(3, requires_grad=True).double()
        self.assertONNX(
            operator.add, (x, y), onnx_export=inspect.currentframe().f_code.co_name
        )

    @ignore_warnings(UserWarning)
    def test_add_size1_singleton_broadcast(self):
        x = torch.randn(2, 3, requires_grad=True).double()
        y = torch.randn(1, 3, requires_grad=True).double()
        self.assertONNX(
            operator.add, (x, y), onnx_export=inspect.currentframe().f_code.co_name
        )

    def test_rsub(self):
        x = torch.randn(2, 3, requires_grad=True).double()
        self.assertONNX(
            lambda x: 1 - x, (x,), onnx_export=inspect.currentframe().f_code.co_name
        )

    @unittest.skipIf(
        not OP_BOOL_SUPPORTED, reason="multiplication of boolean not supported"
    )
    def test_mul_bool(self):
        x = torch.tensor([True, False, True, False])
        y = torch.tensor([True, True, False, False])
        self.assertONNX(
            lambda x, y: torch.mul(x, y),
            (x, y),
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    @unittest.skipIf(
        not OP_BOOL_SUPPORTED, reason="multiplication of boolean not supported"
    )
    def test_mul_fp_bool(self):
        x = torch.tensor([9.4, 1.7, 3.6])
        y = torch.tensor([True, True, False])
        self.assertONNX(
            lambda x, y: torch.mul(x, y),
            (x, y),
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_transpose(self):
        x = torch.tensor([[0.0, 1.0], [2.0, 3.0]], requires_grad=True)
        self.assertONNX(
            lambda x: x.transpose(0, 1).transpose(1, 0),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_chunk(self):
        x = torch.tensor([0.0, 1.0, 2.0], requires_grad=True)
        self.assertONNX(
            lambda x: x.chunk(2),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
            input_index="simple",
        )

    def test_split(self):
        x = torch.tensor(
            [[0.0, 1.0, 1.0, 0.0, 2.0, 2.0], [2.0, 3.0, 3.0, 2.0, 1.0, 1.0]]
        )
        self.assertONNX(
            lambda x: torch.split(x, 2, 1),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
            input_index="simple",
        )

    def test_split_with_sizes(self):
        x = torch.tensor(
            [[0.0, 1.0, 1.0, 0.0, 2.0, 2.0], [2.0, 3.0, 3.0, 2.0, 1.0, 1.0]]
        )
        self.assertONNX(
            lambda x: torch.split(x, [2, 1, 3], 1),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
            input_index="simple",
        )

    def test_concat2(self):
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        self.assertONNX(
            lambda inputs: torch.cat(inputs, 1),
            ((x, y),),
            onnx_export=inspect.currentframe().f_code.co_name,
            test_backward=False,
            input_index=0,
        )

    def test_stack(self):
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        self.assertONNX(
            lambda inputs: torch.stack(inputs, dim=1),
            ((x, y),),
            onnx_export=inspect.currentframe().f_code.co_name,
            test_backward=False,
            input_index=0,
        )

    def test_mm(self):
        m1 = torch.randn(2, 3, requires_grad=True)
        m2 = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(
            torch.mm, (m1, m2), onnx_export=inspect.currentframe().f_code.co_name
        )

    @ignore_warnings(UserWarning)
    def test_addmm(self):
        m1 = torch.randn(2, 3, requires_grad=True)
        m2 = torch.randn(3, 4, requires_grad=True)
        m3 = torch.randn(4, requires_grad=True)
        self.assertONNX(
            lambda x, y, z: torch.addmm(torch.addmm(z, x, y), x, y),
            (m1, m2, m3),
            onnx_export=inspect.currentframe().f_code.co_name,
            atol=1e-4,
        )

    def test_permute2(self):
        x = torch.tensor([[[[[[0.0]]]]]], requires_grad=True)
        self.assertONNX(
            lambda x: x.permute(0, 1, 4, 2, 5, 3),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_pad_op(self):
        x = torch.tensor(
            [[[[0.0, 1.0, 1.0, 1.0], [2.0, 3.0, 7.0, 7.0]]]], requires_grad=True
        )
        self.assertONNX(
            nn.ReflectionPad2d((2, 3, 0, 1)),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_pad_1(self):
        x = torch.tensor(
            [[0.0, 1.0, 1.0, 1.0], [2.0, 3.0, 7.0, 7.0]], requires_grad=True
        )
        self.assertONNX(
            lambda x: nn.functional.pad(x, (1, 2, 3, 4)),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_pad_2(self):
        x = torch.tensor(
            [[0.0, 1.0, 1.0, 1.0], [2.0, 3.0, 7.0, 7.0]], requires_grad=True
        )
        self.assertONNX(
            lambda x: nn.functional.pad(
                x,
                (
                    1,
                    2,
                ),
            ),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_pad_3(self):
        x = torch.tensor(
            [[[0.0, 1.0, 1.0, 1.0], [2.0, 3.0, 7.0, 7.0]]], requires_grad=True
        )
        self.assertONNX(
            lambda x: nn.functional.pad(x, (1, 2, 3, 4, 5, 6)),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_params(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True))
        self.assertONNX(
            lambda x, y: -torch.sigmoid(torch.tanh(x * (x + y))),
            x,
            params=(y,),
            keep_initializers_as_inputs=True,
            onnx_export=inspect.currentframe().f_code.co_name,
            atol=1e-4,
        )

    def test_params_onnx_irv4(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True))
        self.assertONNX(
            lambda x, y: -torch.sigmoid(torch.tanh(x * (x + y))),
            x,
            params=(y,),
            keep_initializers_as_inputs=False,
            onnx_export=inspect.currentframe().f_code.co_name,
            atol=1e-4,
        )

    def test_batchnorm(self):
        x = torch.ones(2, 2, 2, 2, requires_grad=True)
        self.assertONNX(
            nn.BatchNorm2d(2),
            x,
            keep_initializers_as_inputs=True,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_batchnorm_onnx_irv4(self):
        x = torch.ones(2, 2, 2, 2, requires_grad=True)
        self.assertONNX(
            nn.BatchNorm2d(2), x, onnx_export=inspect.currentframe().f_code.co_name
        )

    def test_batchnorm_1d(self):
        x = torch.ones(2, 2, requires_grad=True)
        self.assertONNX(
            nn.BatchNorm1d(2),
            x,
            keep_initializers_as_inputs=True,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_batchnorm_training(self):
        x = torch.ones(2, 2, 2, 2, requires_grad=True)
        self.assertONNX(
            nn.BatchNorm2d(2),
            x,
            training=torch.onnx.TrainingMode.TRAINING,
            keep_initializers_as_inputs=True,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_conv(self):
        x = torch.ones(20, 16, 50, 40, requires_grad=True)
        self.assertONNX(
            nn.Conv2d(16, 13, 3, bias=False),
            x,
            keep_initializers_as_inputs=True,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_conv_onnx_irv4(self):
        x = torch.ones(20, 16, 50, 40, requires_grad=True)
        self.assertONNX(
            nn.Conv2d(16, 13, 3, bias=False),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_conv_onnx_irv4_opset8(self):
        # This test point checks that for opset 8 (or lower), even if
        # keep_initializers_as_inputs is set to False, it is ignored,
        # and initializers are listed as ONNX graph input, in accordance
        # with ONNX IR v3 semantics (which apply to opset version <= 8).
        x = torch.ones(1, 2, 5, 7, requires_grad=True)
        conv_node = nn.Conv2d(2, 4, 3, bias=False)
        conv_node.weight.data.fill_(1.0)
        self.assertONNX(
            conv_node,
            x,
            opset_version=8,
            keep_initializers_as_inputs=False,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_convtranspose(self):
        x = torch.ones(2, 3, 4, 5, requires_grad=True)
        self.assertONNX(
            nn.ConvTranspose2d(
                3, 3, 3, stride=3, bias=False, padding=1, output_padding=2
            ),
            x,
            keep_initializers_as_inputs=True,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_maxpool(self):
        x = torch.randn(20, 16, 50)
        self.assertONNX(
            nn.MaxPool1d(3, stride=2),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_maxpool_dilations_10(self):
        x = torch.randn(20, 16, 50)
        self.assertONNX(
            nn.MaxPool1d(2, stride=1, dilation=2),
            x,
            opset_version=10,
            onnx_export=inspect.currentframe().f_code.co_name,
            impl="ref",
            verbose=10,
            optimize=False,
        )

    def test_maxpool_dilations_18(self):
        x = torch.randn(20, 16, 50)
        self.assertONNX(
            nn.MaxPool1d(2, stride=1, dilation=2),
            x,
            opset_version=18,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_avg_pool2d(self):
        x = torch.randn(20, 16, 50, 32)
        self.assertONNX(
            nn.AvgPool2d(5, stride=2),
            x,
            impl="ref",
            onnx_export=inspect.currentframe().f_code.co_name,
            verbose=0,
            save_onnx=True,
            optimize=False,
            # intermediate=True,
            rtol=1e-4,
        )

    def test_adaptative_avg_pool2d_global(self):
        x = torch.randn(20, 16, 50, 32)
        self.assertONNX(
            nn.AdaptiveAvgPool2d([1, 1]),
            x,
            impl="ref",
            onnx_export=inspect.currentframe().f_code.co_name,
            verbose=0,
            save_onnx=True,
            optimize=False,
            # intermediate=True,
            rtol=1e-4,
        )

    def test_maxpool_indices(self):
        x = torch.randn(20, 16, 50)
        self.assertONNX(
            nn.MaxPool1d(3, stride=2, return_indices=True),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
            verbose=0,
        )

    @ignore_warnings((UserWarning, DeprecationWarning))
    @requires_torch(
        "2.3.0",
        "torch._dynamo.exc.InternalTorchDynamoError: type object "
        "'FunctionMeta' has no attribute 'forward'",
    )
    def test_at_op(self):
        x = torch.randn(3, 4)

        class MyFun(Function):
            @staticmethod
            def symbolic(g, x):
                return g.at("add", x, x)

            @staticmethod
            def forward(ctx, x):
                return x + x

        class MyModule(Module):
            def forward(self, x):
                return MyFun.apply(x)

        with torch.no_grad():
            self.assertONNX(
                MyModule(),
                x,
                onnx_export=inspect.currentframe().f_code.co_name,
                test_backward=False,
            )

    def test_clip(self):
        x = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: torch.clamp(x, min=-0.5, max=0.5),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_clip_min(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: x.clamp(min=-0.1),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_clip_max(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: x.clamp(max=0.1),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_hardtanh(self):
        x = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: torch.nn.Hardtanh(-0.5, 0.5)(x),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_full(self):
        x = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: torch.full(x.shape, 2.0),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
            test_backward=False,
        )

    def test_full_like(self):
        x = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: torch.full_like(x, 2),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
            test_backward=False,
        )

    def test_max(self):
        x = torch.randn(3, 4, requires_grad=True)
        y = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(
            lambda x, y: torch.max(x, y),
            (x, y),
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_min(self):
        x = torch.randn(3, 4, requires_grad=True)
        y = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(
            lambda x, y: torch.min(x, y),
            (x, y),
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_mean(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: torch.mean(x),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_reduced_mean(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: torch.mean(x, dim=2),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_reduced_mean_keepdim(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: torch.mean(x, dim=(2, 3), keepdim=True),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_mean_dtype(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: torch.mean(x, dtype=torch.double),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_reduced_mean_dtype(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: torch.mean(x, dim=0, dtype=torch.double),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_sum(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: torch.sum(x), x, onnx_export=inspect.currentframe().f_code.co_name
        )

    def test_sum_dtype(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: torch.sum(x, dtype=torch.double),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_reduced_sum_dtype(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: torch.sum(x, dim=0, dtype=torch.double),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_reduced_sum(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: torch.sum(x, dim=(1, 2)),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_reduced_sum_keepdim(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: torch.sum(x, dim=2, keepdim=True),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_prod(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: torch.prod(x),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
            atol=2e-4,
        )

    def test_reduced_prod(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: torch.prod(x, dim=2),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_reduced_prod_keepdim(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: torch.prod(x, dim=2, keepdim=True),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_prod_dtype(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: torch.prod(x, dtype=torch.double),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_reduced_prod_dtype(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: torch.prod(x, dim=0, dtype=torch.double),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_sqrt(self):
        x = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: torch.sqrt(x),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_rsqrt(self):
        x = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: torch.rsqrt(x),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_equal(self):
        x = (
            torch.randn(
                1,
                2,
                3,
                1,
                requires_grad=False,
            )
            .to(torch.int32)
            .to(torch.float32)
        )
        y = (
            torch.randn(1, 2, 1, 1, requires_grad=False)
            .to(torch.int32)
            .to(torch.float32)
        )
        self.assertONNX(
            lambda x, y: x == y,
            (x, y),
            onnx_export=inspect.currentframe().f_code.co_name,
            impl="ref",
            test_backward=False,
            input_index="simple",
        )

    def test_lt(self):
        x = torch.randn(1, 2, 3, 1, requires_grad=True)
        y = torch.randn(1, 4, requires_grad=True)
        self.assertONNX(
            operator.lt,
            (x, y),
            onnx_export=inspect.currentframe().f_code.co_name,
            impl="ref",
            test_backward=False,
            input_index="simple",
        )

    def test_gt(self):
        x = torch.randn(1, 2, 3, 1, requires_grad=False).to(torch.int64)
        y = torch.randn(1, 4, requires_grad=False).to(torch.int64)
        self.assertONNX(
            operator.gt,
            (x, y),
            onnx_export=inspect.currentframe().f_code.co_name,
            test_backward=False,
            input_index="simple",
        )

    def test_le(self):
        x = torch.randn(3, 4, requires_grad=False).int()
        y = torch.randn(3, 4, requires_grad=False).int()
        self.assertONNX(
            operator.le,
            (x, y),
            onnx_export=inspect.currentframe().f_code.co_name,
            test_backward=False,
            input_index="simple",
        )

    def test_op_ge_int(self):
        x = torch.randn(3, 4, requires_grad=False).to(torch.int64)
        y = torch.randn(3, 4, requires_grad=False).to(torch.int64)
        self.assertONNX(
            operator.ge,
            (x, y),
            onnx_export=inspect.currentframe().f_code.co_name,
            test_backward=False,
            input_index="simple",
        )

    def test_op_gef(self):
        x = torch.randn(3, 4, requires_grad=False, dtype=torch.float32)
        y = torch.randn(3, 4, requires_grad=False, dtype=torch.float32)
        self.assertONNX(
            operator.ge,
            (x, y),
            onnx_export=inspect.currentframe().f_code.co_name,
            test_backward=False,
            input_index="simple",
        )

    def test_exp(self):
        x = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: x.exp(), x, onnx_export=inspect.currentframe().f_code.co_name
        )

    def test_sin(self):
        x = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: x.sin(), x, onnx_export=inspect.currentframe().f_code.co_name
        )

    def test_cos(self):
        x = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: x.cos(), x, onnx_export=inspect.currentframe().f_code.co_name
        )

    def test_tan(self):
        x = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: x.tan(), x, onnx_export=inspect.currentframe().f_code.co_name
        )

    def test_tanh(self):
        x = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: x.tanh(),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
            impl="ref",
            square_loss=True,
            atol=1e-5,
        )

    @ignore_warnings(UserWarning)
    def test_asin(self):
        x = torch.rand(3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: x.asin(), x, onnx_export=inspect.currentframe().f_code.co_name
        )

    @ignore_warnings(UserWarning)
    def test_acos(self):
        x = torch.rand(3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: x.acos(), x, onnx_export=inspect.currentframe().f_code.co_name
        )

    @unittest.skipIf(
        True,
        reason="https://github.com/pytorch/pytorch/issues/104505#issuecomment-1919745791",
    )
    def test_slice_ort_view(self):
        x = torch.arange(20, requires_grad=True, dtype=torch.float32).reshape((-1, 4))
        self.assertONNX(
            lambda x: x[:, 1:2],
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_slice_ort_clone(self):
        x = torch.arange(20, requires_grad=True, dtype=torch.float32).reshape((-1, 4))
        self.assertONNX(
            lambda x: x.clone()[:, 1:2],
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    @unittest.skipIf(
        True,
        reason="https://github.com/pytorch/pytorch/issues/104505#issuecomment-1919745791",
    )
    def test_slice_ref_view(self):
        x = torch.arange(20, requires_grad=True, dtype=torch.float32).reshape((-1, 4))
        self.assertONNX(
            lambda x: x[:, 1:2],
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
            impl="ref",
        )

    def test_slice_ref_clone(self):
        x = torch.arange(20, requires_grad=True, dtype=torch.float32).reshape((-1, 4))
        self.assertONNX(
            lambda x: x.clone()[:, 1:2],
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
            impl="ref",
        )

    def test_slice_dynamic1(self):
        x = torch.rand(3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: x[x.size(0) :, x.size(1) - 3],
            x,
            opset_version=10,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_slice_dynamic2(self):
        x = torch.rand(7, 9, requires_grad=True)
        self.assertONNX(
            lambda x: x[1:, x.size(1) - 3],
            x,
            opset_version=10,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_slice_dynamic3(self):
        x = torch.rand(7, 9, requires_grad=True)
        self.assertONNX(
            lambda x: x[1:4, 2 : x.size(1) - 3],
            x,
            opset_version=10,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_slice_scatter_1_forward(self):
        a = torch.zeros(8, 8)
        b = torch.ones(2, 8)
        self.assertONNX(
            lambda a, b: torch.slice_scatter(a, b, start=6),
            (a, b),
            opset_version=18,
            onnx_export=inspect.currentframe().f_code.co_name,
            impl="ref",
            test_backward=False,
            raise_list=None,  # {"_onx_scatterelements0"},
        )

    def test_slice_scatter_1_backward(self):
        a = torch.zeros(8, 8)
        b = torch.ones(2, 8)
        self.assertONNX(
            lambda a, b: torch.slice_scatter(a, b, start=6),
            (a, b),
            opset_version=18,
            onnx_export=inspect.currentframe().f_code.co_name,
            impl="ref",
            test_backward=True,
        )

    def test_slice_scatter_2_forward(self):
        a = torch.zeros(8, 8)
        b = torch.ones(8, 2)
        self.assertONNX(
            lambda a, b: torch.slice_scatter(a, b, dim=1, start=2, end=6, step=2),
            (a, b),
            opset_version=18,
            onnx_export=inspect.currentframe().f_code.co_name,
            impl="ref",
            test_backward=False,
        )

    def test_slice_scatter_2_backward(self):
        a = torch.zeros(8, 8)
        b = torch.ones(8, 2)
        self.assertONNX(
            lambda a, b: torch.slice_scatter(a, b, dim=1, start=2, end=6, step=2),
            (a, b),
            opset_version=18,
            onnx_export=inspect.currentframe().f_code.co_name,
            impl="ref",
            test_backward=True,
        )

    def test_sign(self):
        x = torch.rand(3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: x.sign(), x, onnx_export=inspect.currentframe().f_code.co_name
        )

    def test_narrow(self):
        x = torch.randn(3, 3, requires_grad=True)
        self.assertONNX(
            lambda x: torch.narrow(x, 0, 0, 2),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    @ignore_warnings(UserWarning)
    def test_atan(self):
        x = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: x.atan(), x, onnx_export=inspect.currentframe().f_code.co_name
        )

    def test_view_flatten(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: x.view(x.size()[0], x.numel() // x.size()[0]),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_flatten(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: torch.flatten(x),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_flatten2D(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: torch.flatten(x, 1),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_isnan(self):
        x = torch.tensor([1, float("nan"), 2])
        self.assertONNX(
            lambda x: torch.isnan(x),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_argmax(self):
        x = torch.randn(4, 4, requires_grad=True)
        self.assertONNX(
            lambda x: torch.argmax(x, dim=1),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
            test_backward=False,
            input_index="simple",
        )

    def test_logsoftmax(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(
            nn.LogSoftmax(dim=3),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
            atol=1e-4,
        )

    def test_pow(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        y = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(
            lambda x, y: x.pow(y),
            (x, y),
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_elu(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(
            torch.nn.functional.elu,
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_selu(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(
            torch.nn.functional.selu,
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_repeat(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: x.repeat(1, 2, 3, 4),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
            atol=1e-4,
        )

    def test_repeat_dim_overflow(self):
        x = torch.randn(1, 2, requires_grad=True)
        self.assertONNX(
            lambda x: x.repeat(1, 2, 3, 4),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_norm_p1(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: x.norm(p=1, dim=2),
            (x),
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_norm_p2(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: x.norm(p=2, dim=2),
            (x),
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_upsample_nearest_scale(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: nn.functional.interpolate(
                x,
                scale_factor=2.0,
                mode="nearest",
                recompute_scale_factor=False,
            ),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_upsample_nearest_scale_default_scale_factor(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: nn.functional.interpolate(x, scale_factor=2.0, mode="nearest"),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_upsample_nearest_size(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: nn.functional.interpolate(x, size=16, mode="nearest"),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_unsqueeze(self):
        x = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: x.unsqueeze(len(x.shape)),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_batchnorm_noaffine(self):
        x = torch.randn(128, 128, 1, 1, requires_grad=True)
        self.assertONNX(
            nn.BatchNorm2d(128, affine=False, momentum=0.3),
            x,
            keep_initializers_as_inputs=True,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_embedding_bags(self):
        emb_bag = nn.EmbeddingBag(10, 8)
        input = torch.tensor([1, 2, 3, 4]).long()
        offset = torch.tensor([0]).long()
        self.assertONNX(
            emb_bag,
            (input, offset),
            keep_initializers_as_inputs=True,
            input_index="all",
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_implicit_expand(self):
        x = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: x + 1, x, onnx_export=inspect.currentframe().f_code.co_name
        )

    def test_reduce_sum_negative_indices(self):
        x = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: x.sum(-1), x, onnx_export=inspect.currentframe().f_code.co_name
        )

    def test_randn(self):
        x = torch.randn(1, 2, 3, 4)
        self.assertONNX(
            lambda x: torch.randn(1, 2, 3, 4) + x,
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_rand(self):
        x = torch.rand(1, 2, 3, 4)
        self.assertONNX(
            lambda x: torch.rand(1, 2, 3, 4) + x,
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_relu(self):
        x = torch.randn(1, 2, 3, 4)
        self.assertONNX(
            torch.nn.ReLU(), x, onnx_export=inspect.currentframe().f_code.co_name
        )

    @requires_torch(
        "2.3.0",
        "rrelu_with_noise() missing 2 required positional arguments: 'lower' and 'upper'",
    )
    def test_rrelu(self):
        x = torch.randn(1, 2, 3, 4)
        self.assertONNX(
            torch.nn.RReLU(),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
            use_decomposition=True,
        )

    def test_prelu(self):
        x = torch.randn(1, 2, 3, 4)
        self.assertONNX(
            torch.nn.PReLU(2),
            x,
            keep_initializers_as_inputs=True,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_log_sigmoid(self):
        x = torch.randn(1, 2, 3, 4)
        self.assertONNX(
            torch.nn.LogSigmoid(), x, onnx_export=inspect.currentframe().f_code.co_name
        )

    def test_sigmoid(self):
        with self.subTest(dim=2, impl="ref"):
            x = torch.arange(12, dtype=torch.float32, requires_grad=True).reshape(
                (3, 4)
            )
            self.assertONNX(
                lambda x: torch.sigmoid(x),
                x,
                onnx_export=inspect.currentframe().f_code.co_name,
                impl="ref",
                square_loss=True,
            )

        with self.subTest(dim=2):
            x = torch.arange(12, dtype=torch.float32, requires_grad=True).reshape(
                (3, 4)
            )
            self.assertONNX(
                lambda x: torch.sigmoid(x),
                x,
                onnx_export=inspect.currentframe().f_code.co_name,
                square_loss=True,
                atol=1e-4,
            )

        with self.subTest(dim=4):
            x = torch.randn(1, 2, 3, 4, requires_grad=True)
            self.assertONNX(
                lambda x: torch.sigmoid(x),
                x,
                onnx_export=inspect.currentframe().f_code.co_name,
            )

        with self.subTest(dim=1):
            x = torch.randn(4, requires_grad=True)
            self.assertONNX(
                lambda x: torch.sigmoid(x),
                x,
                onnx_export=inspect.currentframe().f_code.co_name,
            )

    def test_linear(self):
        x = torch.randn(3, 4)
        self.assertONNX(
            torch.nn.Linear(4, 5, bias=True),
            x,
            keep_initializers_as_inputs=True,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_zeros_like(self):
        x = torch.randn(5, 8, requires_grad=True)
        self.assertONNX(
            lambda x: torch.zeros_like(x),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_ones_like(self):
        x = torch.randn(6, 10, requires_grad=False)
        self.assertONNX(
            lambda x: torch.ones_like(x),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
            test_backward=False,
        )

    @unittest.skipIf(
        True,
        reason="https://github.com/pytorch/pytorch/issues/104505#issuecomment-1919745791",
    )
    def test_expand(self):
        x = torch.randn(6, 1, requires_grad=True)
        self.assertONNX(
            lambda x: x.expand(4, 6, 2),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_ne(self):
        x = torch.randn(1, 2, 3, 1, requires_grad=False).int().to(torch.float32)
        y = torch.randn(1, 4, requires_grad=False).int().to(torch.float32)
        self.assertONNX(
            lambda x, y: torch.ne(x, y),
            (x, y),
            onnx_export=inspect.currentframe().f_code.co_name,
            test_backward=False,
            input_index="simple",
        )

    def test_reducemax(self):
        x = torch.randn(1, 2, 3, 4)
        self.assertONNX(
            lambda x: torch.max(x), x, onnx_export=inspect.currentframe().f_code.co_name
        )

    def test_reducemin(self):
        x = torch.randn(1, 2, 3, 4)
        self.assertONNX(
            lambda x: torch.min(x), x, onnx_export=inspect.currentframe().f_code.co_name
        )

    def test_erf(self):
        x = torch.randn(1, 2, 3, 4)
        self.assertONNX(
            lambda x: x.erf(), x, onnx_export=inspect.currentframe().f_code.co_name
        )

    def test_dropout(self):
        x = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: torch.max(functional.dropout(x, training=False)),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_dropout_default(self):
        x = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: torch.max(functional.dropout(x)),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_dropout_training(self):
        x = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: torch.max(functional.dropout(x)),
            x,
            training=torch.onnx.TrainingMode.TRAINING,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_dropout_opset12(self):
        x = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: torch.max(functional.dropout(x, training=False)),
            x,
            opset_version=12,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_dropout_training_opset12(self):
        x = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: torch.max(functional.dropout(x)),
            x,
            opset_version=12,
            training=torch.onnx.TrainingMode.TRAINING,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    @unittest.skipIf(not DYNAMIC_SHAPE_SUPPORTED, reason="dynamic shape")
    def test_nonzero(self):
        x = torch.tensor(
            [[[2.0, 2.0], [1.0, 0.0]], [[0.0, 0.0], [1.0, 1.0]]], requires_grad=True
        )
        self.assertONNX(
            lambda x: torch.nonzero(x),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_gather(self):
        data = torch.randn(3, 4, 3, requires_grad=True)
        index = torch.tensor([2, 0]).view(1, 2, 1).expand(3, 2, 3)
        self.assertONNX(
            lambda data, index: data.gather(1, index),
            (data, index),
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_gather_opset11(self):
        data = torch.randn(3, 4, 3, requires_grad=True)
        index = torch.tensor([2, 0]).view(1, 2, 1).expand(3, 2, 3)
        self.assertONNX(
            lambda data, index: data.gather(1, index),
            (data, index),
            opset_version=11,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_scatter_add(self):
        data = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        indices = torch.tensor([[1, 0], [0, 1], [0, 1]], dtype=torch.int64)
        values = torch.tensor([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
        self.assertONNX(
            lambda data, index: data.scatter_add(1, indices, values),
            (data, (indices, values)),
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_scatter_add_opset11(self):
        data = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        indices = torch.tensor([[1, 0], [0, 1], [0, 1]], dtype=torch.int64)
        values = torch.tensor([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
        self.assertONNX(
            lambda data, index: data.scatter_add(1, indices, values),
            (data, (indices, values)),
            opset_version=11,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_scatter_add_opset16(self):
        data = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        indices = torch.tensor([[0, 0], [1, 1], [0, 1]], dtype=torch.int64)
        values = torch.tensor([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
        self.assertONNX(
            lambda data, index: data.scatter_add(1, indices, values),
            (data, (indices, values)),
            opset_version=16,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_master_opset(self):
        x = torch.randn(2, 3).float()
        y = torch.randn(2, 3).float()
        self.assertONNX(
            operator.add,
            (x, y),
            opset_version=10,
            onnx_export=inspect.currentframe().f_code.co_name,
            test_backward=False,
        )

    def test_std(self):
        x = torch.randn(2, 3, 4).float()
        self.assertONNX(
            lambda x: torch.std(x, dim=(0, 1), unbiased=True, keepdim=True),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_cumsum(self):
        x = torch.randn(2, 3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: torch.cumsum(x, dim=1),
            x,
            opset_version=11,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    @unittest.skipIf(not DICT_SUPPORTED, reason="only tensor are supported")
    def test_dict(self):
        class MyModel(torch.nn.Module):
            def forward(self, x_in, *args, **kwargs):
                x_out = {}
                x_out["test_key_out"] = torch.add(
                    x_in[next(x_in.keys())],
                    next(x_in.keys()),
                )
                return x_out

        x = {torch.tensor(1.0): torch.randn(1, 2, 3)}
        self.assertONNX(
            MyModel(), (x, {}), onnx_export=inspect.currentframe().f_code.co_name
        )

    @unittest.skipIf(not DICT_SUPPORTED, reason="only tensor are supported")
    def test_dict_str(self):
        class MyModel(torch.nn.Module):
            def forward(self, x_in, *args, **kwargs):
                x_out = {}
                x_out["test_key_out"] = torch.add(x_in["test_key_in"], 2.0)
                return x_out

        x = {"test_key_in": torch.randn(1, 2, 3)}
        self.assertONNX(
            MyModel(), (x, {}), onnx_export=inspect.currentframe().f_code.co_name
        )

    @unittest.skipIf(not DYNAMIC_SHAPE_SUPPORTED, reason="dynamic shape")
    def test_arange_dynamic(self):
        class TestModel(torch.nn.Module):
            def forward(self, input):
                return torch.arange(input.shape[0], input.shape[0] + 5, 0.5)

        input = torch.randn(5, 3, 2)
        self.assertONNX(
            TestModel(),
            input,
            opset_version=11,
            onnx_export=inspect.currentframe().f_code.co_name,
            test_backward=False,
        )

    @unittest.skip("+1 in ModuleModule is failing this test")
    def test_bitshift(self):
        class BitshiftModel(torch.nn.Module):
            def forward(self, input):
                return input >> 1, input >> 2

        input = torch.arange(24, dtype=torch.uint8).reshape(3, 4, 2)
        self.assertONNX(
            BitshiftModel(),
            input,
            opset_version=11,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_layer_norm_aten(self):
        model = torch.nn.LayerNorm([10, 10])
        x = torch.randn(20, 5, 10, 10)
        self.assertONNX(
            model,
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
            atol=3e-4,
        )

    def test_pixel_shuffle(self):
        x = torch.randn(2, 8, 3, 4).float()
        self.assertONNX(
            lambda x: torch.pixel_shuffle(x, upscale_factor=2),
            x,
            opset_version=11,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_frobenius_norm(self):
        x = torch.randn(2, 3, 4).float()
        self.assertONNX(
            lambda x: torch.norm(x, p="fro", dim=(0, 1), keepdim=True),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_unfold(self):
        x = torch.randn(2, 3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: x.unfold(dimension=2, size=2, step=2),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_remainder_float(self):
        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 1, 4)
        self.assertONNX(
            lambda x, y: torch.remainder(x, y),
            (x, y),
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_remainder_int(self):
        x = (torch.randn(2, 3, 4).abs() + 1).to(torch.int64)
        y = (torch.randn(2, 1, 4).abs() + 2).to(torch.int64)
        self.assertONNX(
            lambda x, y: torch.remainder(x, y),
            (x, y),
            onnx_export=inspect.currentframe().f_code.co_name,
            test_backward=False,
        )

    def test_fmod_10(self):
        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 1, 4)
        self.assertONNX(
            lambda x, y: torch.fmod(x, y),
            (x, y),
            opset_version=10,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_gelu_none_18(self):
        x = torch.randn(2, 3, 4, 5, requires_grad=True)
        self.assertONNX(
            lambda x: torch.nn.functional.gelu(x, approximate="none"),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
            opset_version=18,
        )

    def test_gelu_tanh_20(self):
        x = torch.randn(2, 3, 4, 5, requires_grad=True)
        self.assertONNX(
            lambda x: torch.nn.functional.gelu(x, approximate="tanh"),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
            opset_version=20,
        )

    def test_gelu_tanh_18(self):
        x = torch.randn(2, 3, 4, 5, requires_grad=True)
        self.assertONNX(
            lambda x: torch.nn.functional.gelu(x, approximate="tanh"),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
            opset_version=18,
        )

    @unittest.skipIf(not DYNAMIC_SHAPE_SUPPORTED, reason="dynamic shape")
    def test_unique(self):
        x = torch.randint(3, (2, 3, 4, 5)).float()
        self.assertONNX(
            lambda x: torch.unique(
                x, dim=0, sorted=True, return_inverse=False, return_counts=True
            ),
            x,
            opset_version=11,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    @unittest.skipIf(
        True,
        reason="https://github.com/pytorch/pytorch/issues/104505#issuecomment-1919745791",
    )
    def test_meshgrid(self):
        x = torch.ones(3, requires_grad=True)
        y = torch.zeros(4, requires_grad=True)
        z = torch.ones(5, requires_grad=True)
        self.assertONNX(
            lambda x, y, z: torch.meshgrid(x, y, z),
            (x, y, z),
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    @unittest.skipIf(
        True,
        reason="https://github.com/pytorch/pytorch/issues/104505#issuecomment-1919745791",
    )
    def test_meshgrid_indexing(self):
        x = torch.ones(3, requires_grad=True)
        y = torch.zeros(4, requires_grad=True)
        z = torch.ones(5, requires_grad=True)
        self.assertONNX(
            lambda x, y, z: torch.meshgrid(x, y, z, indexing="xy"),
            (x, y, z),
            opset_version=9,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    @unittest.skipIf(not DYNAMIC_SHAPE_SUPPORTED, reason="dynamic shape")
    def test_topk(self):
        x = torch.arange(1.0, 6.0, requires_grad=True)
        k = torch.tensor(3)
        self.assertONNX(
            lambda x, k: torch.topk(x, k),
            (x, k),
            opset_version=10,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    @unittest.skipIf(not DYNAMIC_SHAPE_SUPPORTED, reason="dynamic shape")
    def test_topk_smallest_unsorted(self):
        x = torch.arange(1.0, 6.0, requires_grad=True)
        k = torch.tensor(3)
        self.assertONNX(
            lambda x, k: torch.topk(x, k, largest=False, sorted=False),
            (x, k),
            opset_version=11,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_baddbmm(self):
        x = torch.randn(10, 3, 5)
        b1 = torch.randn(10, 3, 4)
        b2 = torch.randn(10, 4, 5)
        self.assertONNX(
            lambda x, b1, b2: torch.baddbmm(x, b1, b2),
            (x, b1, b2),
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_round(self):
        x = torch.tensor([0.9920, -1.0362, -1.5000, 2.5000], requires_grad=True)
        self.assertONNX(
            lambda x: torch.round(x),
            x,
            opset_version=11,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_dim(self):
        x = torch.ones((2, 2), requires_grad=True, dtype=torch.float32)
        self.assertONNX(
            lambda x: torch.scalar_tensor(x.dim()),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_det(self):
        x = torch.randn(2, 3, 5, 5, device=torch.device("cpu"))
        self.assertONNX(
            lambda x: torch.det(x),
            x,
            opset_version=11,
            onnx_export=inspect.currentframe().f_code.co_name,
        )
        self.assertONNX(
            lambda x: torch.linalg.det(x),
            x,
            opset_version=11,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_softmaxcrossentropy(self):
        x = torch.randn(3, 5)
        y = torch.empty(3, dtype=torch.long).random_(5)
        self.assertONNX(
            torch.nn.CrossEntropyLoss(),
            (x, y),
            opset_version=12,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_softmaxcrossentropy_ignore_index(self):
        x = torch.randn(3, 5)
        y = torch.empty(3, dtype=torch.long).random_(5)
        self.assertONNX(
            torch.nn.CrossEntropyLoss(ignore_index=1),
            (x, y),
            opset_version=12,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_softmaxcrossentropy_weights(self):
        x = torch.randn(3, 5)
        y = torch.empty(3, dtype=torch.long).random_(5)
        self.assertONNX(
            torch.nn.CrossEntropyLoss(weight=torch.randn(5)),
            (x, y),
            opset_version=12,
            onnx_export=inspect.currentframe().f_code.co_name,
            atol=1e-3,
        )

    def test_softmaxcrossentropy_3d(self):
        x = torch.randn(3, 5, 2)
        y = torch.empty(3, 2, dtype=torch.long).random_(5)
        self.assertONNX(
            torch.nn.CrossEntropyLoss(),
            (x, y),
            opset_version=12,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_softmaxcrossentropy_3d_none(self):
        x = torch.randn(3, 5, 2)
        y = torch.empty(3, 2, dtype=torch.long).random_(5)
        self.assertONNX(
            torch.nn.CrossEntropyLoss(reduction="none"),
            (x, y),
            opset_version=12,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_softmaxcrossentropy_4d(self):
        x = torch.randn(3, 5, 2, 1)
        y = torch.empty(3, 2, 1, dtype=torch.long).random_(5)
        self.assertONNX(
            torch.nn.CrossEntropyLoss(),
            (x, y),
            opset_version=12,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    @unittest.skipIf(
        True, reason="TorchDynamo purposely graph breaks on RNN, GRU, LSTMs"
    )
    def test_lstm_none_sequence_lens(self):
        """Test symbolic shape inference for LSTM when the input sequence_lens = None."""
        input = torch.randn(RNN_SEQUENCE_LENGTH, BATCH_SIZE, RNN_INPUT_SIZE)
        h0 = torch.randn(1, BATCH_SIZE, RNN_HIDDEN_SIZE)
        c0 = torch.randn(1, BATCH_SIZE, RNN_HIDDEN_SIZE)

        class LSTMModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.rnn = torch.nn.LSTM(
                    RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, 1, bidirectional=False
                )

            def forward(self, x, h0, c0):
                a, b = self.rnn(x, (h0, c0))
                return torch.ones(b[0].shape)

        self.assertONNX(
            LSTMModel(),
            (input, h0, c0),
            input_names=["x", "y"],
            dynamic_axes={"x": {0: "batch"}},
            opset_version=12,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    @unittest.skipIf(not DYNAMIC_SHAPE_SUPPORTED, reason="dynamic axes not supported")
    def test_dynamic_axes_add(self):
        m1 = torch.randn(2, 3, requires_grad=True)
        m2 = torch.randn(2, 1, requires_grad=True)
        self.assertONNX(
            lambda x, y: torch.add(x, y),
            (m1, m2),
            input_names=["input_1", "input_2"],
            dynamic_axes={"input_1": {1: "dim_1"}, "input_2": {1: "dim_2"}},
            opset_version=12,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    @unittest.skipIf(not DYNAMIC_SHAPE_SUPPORTED, reason="dynamic axes not supported")
    def test_dynamic_axes_add_inputs_same_symbolic_shape(self):
        m1 = torch.randn(2, 3, requires_grad=True)
        self.assertONNX(
            lambda x: torch.add(x, x),
            (m1,),
            input_names=["input_1"],
            dynamic_axes={"input_1": {1: "dim_1"}},
            opset_version=12,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    @unittest.skipIf(not DYNAMIC_SHAPE_SUPPORTED, reason="dynamic axes not supported")
    def test_dynamic_axes_matmul_ort(self):
        m1 = torch.randn(2, 2, 4, requires_grad=True)
        m2 = torch.randn(2, 4, 3, requires_grad=True)
        self.assertONNX(
            lambda x, y: torch.matmul(x, y),
            (m1, m2),
            input_names=["input_1", "input_2"],
            dynamic_axes={"input_1": {1: "dim_0"}, "input_2": {2: "dim_1"}},
            opset_version=12,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    @unittest.skipIf(not DYNAMIC_SHAPE_SUPPORTED, reason="dynamic axes not supported")
    def test_dynamic_axes_matmul_ref(self):
        m1 = torch.randn(2, 2, 4, requires_grad=True)
        m2 = torch.randn(2, 4, 3, requires_grad=True)
        self.assertONNX(
            lambda x, y: torch.matmul(x, y),
            (m1, m2),
            input_names=["input_1", "input_2"],
            dynamic_axes={"input_1": {1: "dim_0"}, "input_2": {2: "dim_1"}},
            opset_version=12,
            onnx_export=inspect.currentframe().f_code.co_name,
            impl="ref",
        )

    @unittest.skipIf(not DYNAMIC_SHAPE_SUPPORTED, reason="dynamic axes not supported")
    def test_dynamic_axes_reduce_mean_12(self):
        m1 = torch.arange(24, dtype=torch.float32, requires_grad=True).reshape(
            (2, 3, 4)
        )
        self.assertONNX(
            lambda x: torch.mean(x, dim=1),
            (m1,),
            input_names=["input"],
            dynamic_axes={"input": {1: "dim_1", 2: "dim_2"}},
            opset_version=12,
            onnx_export=inspect.currentframe().f_code.co_name,
            impl="ref",
        )

    @unittest.skipIf(not DYNAMIC_SHAPE_SUPPORTED, reason="dynamic axes not supported")
    def test_dynamic_axes_reduce_mean_18(self):
        m1 = torch.arange(24, dtype=torch.float32, requires_grad=True).reshape(
            (2, 3, 4)
        )
        self.assertONNX(
            lambda x: torch.mean(x, dim=1),
            (m1,),
            input_names=["input"],
            dynamic_axes={"input": {1: "dim_1", 2: "dim_2"}},
            opset_version=18,
            onnx_export=inspect.currentframe().f_code.co_name,
            impl="ref",
        )

    @unittest.skipIf(not DYNAMIC_SHAPE_SUPPORTED, reason="dynamic axes not supported")
    def test_dynamic_axes_unchange_softmax_ort(self):
        m1 = torch.arange(6, requires_grad=True, dtype=torch.float32).reshape((-1, 3))
        self.assertONNX(
            lambda x: torch.softmax(x, dim=0),
            (m1,),
            input_names=["input"],
            dynamic_axes={"input": {1: "dim_1"}},
            opset_version=13,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    @unittest.skipIf(not DYNAMIC_SHAPE_SUPPORTED, reason="dynamic axes not supported")
    def test_dynamic_axes_unchange_softmax_ref(self):
        m1 = torch.arange(6, requires_grad=True, dtype=torch.float32).reshape((-1, 3))
        self.assertONNX(
            lambda x: torch.softmax(x, dim=0),
            (m1,),
            input_names=["input"],
            dynamic_axes={"input": {1: "dim_1"}},
            opset_version=14,
            onnx_export=inspect.currentframe().f_code.co_name,
            impl="ref",
        )

    @unittest.skipIf(not DYNAMIC_SHAPE_SUPPORTED, reason="dynamic shape")
    def test_shape_value_map(self):
        # Without shapeValueMap, the onnx graph looks like:
        # graph(%0 : Float(*, 1, 128, 1, strides=[128, 128, 1, 1],
        #       requires_grad=0, device=cpu)):
        #   %2 : Long(4, strides=[1], device=cpu) = onnx::Shape(%0)
        #   %4 : Long(device=cpu) = onnx::Constant[value={0}]()
        #   %5 : Long(device=cpu) = onnx::Gather[axis=0](%2, %4)
        #   %6 : Long(device=cpu) = onnx::Constant[value={1}]()
        #   %7 : Long(device=cpu) = onnx::Constant[value={2}]()
        #   %8 : Long(device=cpu) = onnx::Constant[value={-1}]()
        #   %9 : int[] = prim::ListConstruct(%5, %6, %7, %8)
        #   %10 : Float(*, *, *, *, strides=[128, 128, 64, 1],
        #       requires_grad=0, device=cpu) = onnx::Reshape(%0, %9)
        #   ...
        # With shapeValueMap, it becomes:
        #   ...
        #   %10 : Float(*, 1, 2, 64, strides=[128, 128, 64, 1],
        #       requires_grad=0, device=cpu) = onnx::Reshape(%0, %9)
        #   ...
        class RSoftMax(torch.nn.Module):
            def __init__(self, radix, cardinality):
                super().__init__()
                self.radix = radix
                self.cardinality = cardinality

            def forward(self, x, onnx_export=inspect.currentframe().f_code.co_name):
                batch = x.size(0)
                x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
                x = F.softmax(x, dim=1)
                x = x.reshape(batch, -1)
                return x

        radix = 2
        cardinality = 1
        shape = (10, 1, 128, 1)
        x = torch.arange(np.prod(shape), dtype=torch.float32).reshape(shape)
        self.assertONNX(
            RSoftMax(radix, cardinality),
            (x,),
            input_names=["x"],
            dynamic_axes={"x": {0: "dim_0"}},
            onnx_export=inspect.currentframe().f_code.co_name,
            test_backward=False,
        )

    @unittest.skipIf(True, reason="bug with as_strided")
    def test_as_strided_0(self):
        x = torch.arange(12, requires_grad=True, dtype=torch.float32).reshape((-1, 3))
        self.assertONNX(
            lambda x: torch.as_strided(x, (3, 3), (1, 2)),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    @unittest.skipIf(True, reason="bug with as_strided")
    def test_as_strided_1(self):
        import torch

        new_table = {}
        for k, v in torch._decomp.decomposition_table.items():
            if k.name() in {
                "aten::slice_backward",
                "aten::select_backward.out",
                "aten::slice.Tensor",
            }:
                new_table[k] = v

        shape = (9, 2, 15, 4)
        x = torch.arange(
            np.prod(shape), requires_grad=True, dtype=torch.float32
        ).reshape(shape)
        self.assertONNX(
            lambda x: x[:, :, 4:, :],
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
            use_decomposition=new_table,
        )

    def test_embedding_simple(self):
        ix = torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]], dtype=torch.int64)
        embedding_matrix = torch.arange(
            30, dtype=torch.float32, requires_grad=True
        ).reshape((-1, 3))
        torch.embedding(embedding_matrix, ix)
        self.assertONNX(
            lambda ix, mat: torch.embedding(mat, ix),
            (ix, embedding_matrix),
            onnx_export=inspect.currentframe().f_code.co_name,
            input_index=1,
            impl="ref",
            verbose=0,
            use_decomposition=True,
        )

    def test_unbind(self):
        x = torch.tensor(
            [[0.0, 1.0, 1.0, 1.0], [2.0, 3.0, 7.0, 7.0]], requires_grad=True
        )
        self.assertONNX(
            lambda x: torch.unbind(x, dim=0),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
            input_index="simple",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
