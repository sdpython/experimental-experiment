import copy
import inspect
import itertools
import operator
import os
import unittest
import sys
from typing import List, Optional, Union
import packaging.version as pv
import numpy as np
from onnx import ModelProto
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
from torch.autograd import Function
from torch.nn import functional, Module
from torch.onnx.symbolic_helper import (
    _get_tensor_dim_size,
    _get_tensor_sizes,
    parse_args,
)
from torch._dynamo.backends.common import aot_autograd
from experimental_experiment.ext_test_case import ExtTestCase, ignore_warnings
from experimental_experiment.torch_exp.onnx_export import to_onnx
from experimental_experiment.torch_exp._exceptions import FunctionNotFoundError

BATCH_SIZE = 2
RNN_BATCH_SIZE = 7
RNN_SEQUENCE_LENGTH = 11
RNN_INPUT_SIZE = 5
RNN_HIDDEN_SIZE = 3
DYNAMIC_SHAPE_SUPPORTED = False
DICT_SUPPORTED = False
OP_BOOL_SUPPORTED = False


class FuncModule(Module):
    def __init__(self, f, params=None):
        if params is None:
            params = ()
        super().__init__()
        self.f = f
        self.params = nn.ParameterList(list(params))

    def forward(self, *args):
        f_args = list(itertools.chain(args, self.params))
        res = self.f(*f_args)
        return res


def get_session(
    onx: ModelProto, impl: str = "ref", exc: bool = True
) -> Union["ReferenceEvaluator", "InferenceSession"]:  # noqa: F821
    if exc:
        try:
            return get_session(onx, impl, exc=False)
        except Exception as e:
            from onnx_array_api.plotting.text_plot import onnx_simple_text_plot

            raise AssertionError(
                f"Unable to build session ({str(e)})\n{onnx_simple_text_plot(onx)}"
            ) from e

    if impl == "ref":
        from onnx.reference import ReferenceEvaluator

        return ReferenceEvaluator(onx, verbose=10)
    else:
        import onnxruntime

        return onnxruntime.InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )


def onnx_compiler(
    graph_module: torch.fx.GraphModule,
    args: List[torch.Tensor],
    onnx_export: str = "?",
    counter: Optional[List[int]] = None,
    opset_version: Optional[int] = None,
    impl: str = "ort",
):
    assert isinstance(counter, list), f"unexpected type {type(counter)} for counter"
    input_names = (
        ["input"] if len(args) == 1 else [f"input{i}" for i in range(len(args))]
    )

    onx = to_onnx(
        graph_module,
        tuple(args),
        input_names=input_names,
        remove_unused=True,
        constant_folding=False,
        verbose=0 if impl == "ort" else 4,
        target_opset=opset_version,
    )

    if not os.path.exists("temp_dump"):
        os.mkdir("temp_dump")

    counter[0] += 1
    if onnx_export != "?":
        name = os.path.join("temp_dump", f"{onnx_export}_{counter[0]}.onnx")
        with open(name, "wb") as f:
            f.write(onx.SerializeToString())
        with open(name + ".txt", "w") as f:
            f.write(str(graph_module.graph))
            f.write("\n")

    sess = get_session(onx, impl, exc=True)

    names = [i.name for i in onx.graph.input]

    _dtype = {
        np.dtype("float32"): torch.float32,
        np.dtype("float64"): torch.float64,
        np.dtype("int32"): torch.int32,
        np.dtype("int64"): torch.int64,
        np.float32: torch.float32,
        np.float64: torch.float64,
        np.int32: torch.int32,
        np.int64: torch.int64,
    }

    def run(*inputs, sess=sess, names=names):
        # not efficient
        xnp = [x.detach().numpy() for x in inputs]
        feeds = dict(zip(names, xnp))
        results = sess.run(None, feeds)
        res = tuple(torch.Tensor(y).to(_dtype[y.dtype]) for y in results)
        if len(res) == 1:
            return res[0]
        return res

    return run


class TestOperators(ExtTestCase):
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
        operator_export_type=None,
        impl="ort",
        #
        input_names=None,
        dynamic_axes=None,
        keep_initializers_as_inputs=None,
        training=None,
    ):
        if sys.platform == "win32":
            raise unittest.SkipTest("Windows not supported yet.")
        assert isinstance(onnx_export, str), f"Export onnx is wrong for f={f}"
        if isinstance(args, torch.Tensor):
            args = [args]
        if params is None:
            params = ()
        if isinstance(f, nn.Module):
            model = f
        else:
            model = FuncModule(f, params)
        model.eval()

        counter = [0]

        if test_backward:
            # forward/backward
            aot_compiler = aot_autograd(
                fw_compiler=lambda *args: onnx_compiler(
                    *args,
                    onnx_export=onnx_export,
                    counter=counter,
                    opset_version=opset_version,
                    impl=impl,
                )
            )

            compiled_model = torch.compile(
                copy.deepcopy(model),
                backend=aot_compiler,
                dynamic=False,
                fullgraph=fullgraph,
            )

            baseline_result = model(*args)
            try:
                result = compiled_model(*args)
            except torch._dynamo.exc.BackendCompilerFailed as e:
                if "FunctionNotFoundError" in str(e):
                    raise unittest.SkipTest(f"MISSING FOR FORWARD {e}")
                raise

            if isinstance(baseline_result, torch.Tensor):
                self.assertEqualArray(
                    baseline_result.detach().numpy(),
                    result.detach().numpy(),
                    atol=atol,
                    rtol=rtol,
                )
                try:
                    torch.testing.assert_close(
                        baseline_result, result, atol=atol, rtol=rtol
                    )
                except AssertionError as e:
                    if "nan" not in str(e):
                        raise

                baseline_result.sum().backward()
                try:
                    result.sum().backward()
                except FunctionNotFoundError as e:
                    raise unittest.SkipTest(f"MISSING FOR BACKWARD {e}")

                for baseline_param, param in zip(
                    model.parameters(), compiled_model.parameters()
                ):
                    self.assertEqualArray(
                        baseline_param.grad.detach().numpy(),
                        param.grad.detach().numpy(),
                        atol=atol,
                        rtol=rtol,
                    )
                    torch.testing.assert_close(
                        baseline_param.grad,
                        param.grad,
                        atol=atol,
                        rtol=rtol,
                    )
            else:
                raise AssertionError(f"Unexpected type {type(baseline_result)}.")
        else:
            # forward only
            compiled_model = torch.compile(
                copy.deepcopy(model),
                backend=lambda *args: onnx_compiler(
                    *args,
                    onnx_export=onnx_export,
                    counter=counter,
                    opset_version=opset_version,
                    impl=impl,
                ),
                dynamic=False,
                fullgraph=fullgraph,
            )

            baseline_result = model(*args)
            result = compiled_model(*args)

            if isinstance(baseline_result, torch.Tensor):
                self.assertEqualArray(
                    baseline_result.detach().numpy(),
                    result.detach().numpy(),
                    atol=atol,
                    rtol=rtol,
                )
                torch.testing.assert_close(
                    baseline_result, result, atol=atol, rtol=rtol
                )

    @ignore_warnings(UserWarning)
    def test_aaa(self):
        x = torch.rand(3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: x.acos(), x, onnx_export=inspect.currentframe().f_code.co_name
        )

    def test_basic(self):
        x = torch.tensor([0.4], requires_grad=True)
        y = torch.tensor([0.7], requires_grad=True)
        self.assertONNX(
            lambda x, y: -torch.sigmoid(torch.tanh(x * (x + y))),
            (x, y),
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_view(self):
        x = torch.tensor([0.0], requires_grad=True)
        self.assertONNX(
            lambda x: x.view(1, 1), x, onnx_export=inspect.currentframe().f_code.co_name
        )

    def test_index(self):
        x = torch.tensor([[0.0]], requires_grad=True)
        self.assertONNX(
            lambda x: x[0], x, onnx_export=inspect.currentframe().f_code.co_name
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
            lambda x: x.chunk(2), x, onnx_export=inspect.currentframe().f_code.co_name
        )

    def test_split(self):
        x = torch.tensor(
            [[0.0, 1.0, 1.0, 0.0, 2.0, 2.0], [2.0, 3.0, 3.0, 2.0, 1.0, 1.0]]
        )
        self.assertONNX(
            lambda x: torch.split(x, 2, 1),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_split_with_sizes(self):
        x = torch.tensor(
            [[0.0, 1.0, 1.0, 0.0, 2.0, 2.0], [2.0, 3.0, 3.0, 2.0, 1.0, 1.0]]
        )
        self.assertONNX(
            lambda x: torch.split(x, [2, 1, 3], 1),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_concat2(self):
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        self.assertONNX(
            lambda inputs: torch.cat(inputs, 1),
            ((x, y),),
            onnx_export=inspect.currentframe().f_code.co_name,
            test_backward=False,
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
        )

    def test_permute2(self):
        x = torch.tensor([[[[[[0.0]]]]]], requires_grad=True)
        self.assertONNX(
            lambda x: x.permute(0, 1, 4, 2, 5, 3),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_pad(self):
        x = torch.tensor(
            [[[[0.0, 1.0, 1.0, 1.0], [2.0, 3.0, 7.0, 7.0]]]], requires_grad=True
        )
        self.assertONNX(
            nn.ReflectionPad2d((2, 3, 0, 1)),
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

    def test_maxpool_dilations(self):
        x = torch.randn(20, 16, 50)
        self.assertONNX(
            nn.MaxPool1d(2, stride=1, dilation=2),
            x,
            opset_version=10,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_avg_pool2d(self):
        x = torch.randn(20, 16, 50, 32)
        self.assertONNX(
            nn.AvgPool2d(3, stride=2),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_maxpool_indices(self):
        x = torch.randn(20, 16, 50)
        self.assertONNX(
            nn.MaxPool1d(3, stride=2, return_indices=True),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    @ignore_warnings((UserWarning, DeprecationWarning))
    def test_at_op(self):
        x = torch.randn(3, 4)

        class MyFun(Function):
            @staticmethod
            def symbolic(g, x, onnx_export=inspect.currentframe().f_code.co_name):
                return g.at(
                    "add", x, x, onnx_export=inspect.currentframe().f_code.co_name
                )

            @staticmethod
            def forward(ctx, x, onnx_export=inspect.currentframe().f_code.co_name):
                return x + x

        class MyModule(Module):
            def forward(self, x, onnx_export=inspect.currentframe().f_code.co_name):
                return MyFun.apply(x)

        self.assertONNX(
            MyModule(),
            x,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
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
        x = torch.randn(1, 2, 3, 1, requires_grad=False).int()
        y = torch.randn(1, 4, requires_grad=False).int()
        self.assertONNX(
            operator.eq, (x, y), onnx_export=inspect.currentframe().f_code.co_name
        )

    def test_lt(self):
        x = torch.randn(1, 2, 3, 1, requires_grad=False).int()
        y = torch.randn(1, 4, requires_grad=False).int()
        self.assertONNX(
            operator.lt, (x, y), onnx_export=inspect.currentframe().f_code.co_name
        )

    def test_gt(self):
        x = torch.randn(1, 2, 3, 1, requires_grad=False).int()
        y = torch.randn(1, 4, requires_grad=False).int()
        self.assertONNX(
            operator.gt, (x, y), onnx_export=inspect.currentframe().f_code.co_name
        )

    def test_le(self):
        x = torch.randn(3, 4, requires_grad=False).int()
        y = torch.randn(3, 4, requires_grad=False).int()
        self.assertONNX(
            operator.le, (x, y), onnx_export=inspect.currentframe().f_code.co_name
        )

    def test_ge(self):
        x = torch.randn(3, 4, requires_grad=False).int()
        y = torch.randn(3, 4, requires_grad=False).int()
        self.assertONNX(
            operator.ge, (x, y), onnx_export=inspect.currentframe().f_code.co_name
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

    def test_slice_dynamic(self):
        x = torch.rand(3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: x[x.size(0) :, x.size(1) - 3],
            x,
            opset_version=10,
            onnx_export=inspect.currentframe().f_code.co_name,
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
        )

    def test_logsoftmax(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(
            nn.LogSoftmax(dim=3), x, onnx_export=inspect.currentframe().f_code.co_name
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
        self.assertONNX(nn.ELU(), x, onnx_export=inspect.currentframe().f_code.co_name)

    def test_selu(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(nn.SELU(), x, onnx_export=inspect.currentframe().f_code.co_name)

    def test_repeat(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: x.repeat(1, 2, 3, 4),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
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
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
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

    @unittest.skipIf(
        pv.Version(torch.__version__) < pv.Version("2.3.0"),
        reason="rrelu_with_noise() missing 2 required positional arguments: 'lower' and 'upper'",
    )
    def test_rrelu(self):
        x = torch.randn(1, 2, 3, 4)
        self.assertONNX(
            torch.nn.RReLU(), x, onnx_export=inspect.currentframe().f_code.co_name
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

    def test_linear(self):
        x = torch.randn(3, 4)
        self.assertONNX(
            torch.nn.Linear(4, 5, bias=True),
            x,
            keep_initializers_as_inputs=True,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_empty_like(self):
        x = torch.randn(5, 8, requires_grad=True)
        self.assertONNX(
            lambda x: torch.empty_like(x),
            x,
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
        x = torch.randn(6, 10, requires_grad=True)
        self.assertONNX(
            lambda x: torch.ones_like(x),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_expand(self):
        x = torch.randn(6, 1, requires_grad=True)
        self.assertONNX(
            lambda x: x.expand(4, 6, 2),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_ne(self):
        x = torch.randn(1, 2, 3, 1, requires_grad=False).int()
        y = torch.randn(1, 4, requires_grad=False).int()
        self.assertONNX(
            lambda x, y: torch.ne(x, y),
            (x, y),
            onnx_export=inspect.currentframe().f_code.co_name,
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
            lambda x: torch.max(
                functional.dropout(
                    x,
                )
            ),
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
                    x_in[list(x_in.keys())[0]], list(x_in.keys())[0]  # noqa: RUF015
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
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
            onnx_export=inspect.currentframe().f_code.co_name,
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

    def test_remainder(self):
        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 1, 4)
        self.assertONNX(
            lambda x, y: torch.remainder(x, y),
            (x, y),
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_fmod(self):
        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 1, 4)
        self.assertONNX(
            lambda x, y: torch.fmod(x, y),
            (x, y),
            opset_version=10,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_gelu(self):
        x = torch.randn(2, 3, 4, 5, requires_grad=True)
        self.assertONNX(
            lambda x: torch.nn.functional.gelu(x),
            x,
            onnx_export=inspect.currentframe().f_code.co_name,
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

    def test_meshgrid(self):
        x = torch.ones(3, requires_grad=True)
        y = torch.zeros(4, requires_grad=True)
        z = torch.ones(5, requires_grad=True)
        self.assertONNX(
            lambda x, y, z: torch.meshgrid(x, y, z),
            (x, y, z),
            onnx_export=inspect.currentframe().f_code.co_name,
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
        x = torch.ones((2, 2), requires_grad=True)
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

    def test_dynamic_axes_matmul(self):
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

    def test_dynamic_axes_reduce_mean(self):
        m1 = torch.randn(2, 3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: torch.mean(x, dim=1),
            (m1),
            input_names=["input"],
            dynamic_axes={"input": {1: "dim_1", 2: "dim_2"}},
            opset_version=12,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_dynamic_axes_unchange(self):
        """Test ProcessUnchangeNode in symbolic shape inference."""
        m1 = torch.randn(2, 3, requires_grad=True)
        self.assertONNX(
            lambda x: torch.softmax(x, dim=0),
            (m1,),
            input_names=["input"],
            dynamic_axes={"input": {1: "dim_1"}},
            opset_version=12,
            onnx_export=inspect.currentframe().f_code.co_name,
        )

    def test_aten_embedding_1(self):
        _onnx_opset_version = 12

        @parse_args("v", "v", "i", "b", "b")
        def embedding(g, weight, indices, padding_idx, scale_grad_by_freq, sparse):
            custom_attributes_json = (
                "{"
                f'"padding_idx":{str(padding_idx)},'
                f'"scale_grad_by_freq":{str(scale_grad_by_freq).lower()},'
                f'"sparse":{str(sparse).lower()}'
                "}"
            )
            output = g.at(
                "embedding",
                weight,
                indices,
                custom_attributes_json_s=custom_attributes_json,
            )
            return output

        torch.onnx.register_custom_op_symbolic(
            "::embedding", embedding, _onnx_opset_version
        )

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Embedding(4, 8)

            def forward(self, x, y):
                res = self.emb(x)
                res = res + y
                return torch.ones(res.shape[0])

        model = Model()
        x = torch.ones(32, dtype=torch.long)
        y = torch.randn(1, 8)
        self.assertONNX(
            model,
            (x, y),
            opset_version=_onnx_opset_version,
            onnx_export=inspect.currentframe().f_code.co_name,
            test_backward=False,
        )

        torch.onnx.unregister_custom_op_symbolic("::embedding", _onnx_opset_version)

    # This is test_aten_embedding_1 with shape inference on custom symbolic aten::embedding.
    def test_aten_embedding_2(self):
        _onnx_opset_version = 12

        @parse_args("v", "v", "i", "b", "b")
        def embedding(g, weight, indices, padding_idx, scale_grad_by_freq, sparse):
            custom_attributes_json = (
                "{"
                f'"padding_idx":{str(padding_idx)},'
                f'"scale_grad_by_freq":{str(scale_grad_by_freq).lower()},'
                f'"sparse":{str(sparse).lower()}'
                "}"
            )
            output = g.at(
                "embedding",
                weight,
                indices,
                custom_attributes_json_s=custom_attributes_json,
            )

            # do shape inference and set it via setType
            indices_shape = _get_tensor_sizes(indices)
            if indices_shape is not None and hasattr(weight.type(), "with_sizes"):
                output_type = weight.type().with_sizes(
                    indices_shape + [_get_tensor_dim_size(weight, 1)]
                )
                output.setType(output_type)
            return output

        torch.onnx.register_custom_op_symbolic(
            "::embedding", embedding, _onnx_opset_version
        )

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Embedding(4, 8)

            def forward(self, x, y):
                res = self.emb(x)
                res = res + y
                return torch.ones(res.shape[0])

        model = Model()
        x = torch.ones(32, dtype=torch.long)
        y = torch.randn(1, 8)
        self.assertONNX(
            model,
            (x, y),
            opset_version=_onnx_opset_version,
            input_names=["input_1", "input_2"],
            dynamic_axes={"input_1": {0: "dim_0"}, "input_2": {0: "dim_1", 1: "dim_2"}},
            keep_initializers_as_inputs=False,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
            onnx_export=inspect.currentframe().f_code.co_name,
            test_backward=False,
        )

        torch.onnx.unregister_custom_op_symbolic("::embedding", _onnx_opset_version)

    # Without shapeValueMap, the onnx graph looks like:
    # graph(%0 : Float(*, 1, 128, 1, strides=[128, 128, 1, 1], requires_grad=0, device=cpu)):
    #   %2 : Long(4, strides=[1], device=cpu) = onnx::Shape(%0)
    #   %4 : Long(device=cpu) = onnx::Constant[value={0}]()
    #   %5 : Long(device=cpu) = onnx::Gather[axis=0](%2, %4)
    #   %6 : Long(device=cpu) = onnx::Constant[value={1}]()
    #   %7 : Long(device=cpu) = onnx::Constant[value={2}]()
    #   %8 : Long(device=cpu) = onnx::Constant[value={-1}]()
    #   %9 : int[] = prim::ListConstruct(%5, %6, %7, %8)
    #   %10 : Float(*, *, *, *, strides=[128, 128, 64, 1], requires_grad=0, device=cpu) = onnx::Reshape(%0, %9)
    #   ...
    # With shapeValueMap, it becomes:
    #   ...
    #   %10 : Float(*, 1, 2, 64, strides=[128, 128, 64, 1], requires_grad=0, device=cpu) = onnx::Reshape(%0, %9)
    #   ...
    def test_shape_value_map(self):
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
        x = torch.randn(10, 1, 128, 1)
        self.assertONNX(
            RSoftMax(radix, cardinality),
            (x,),
            input_names=["x"],
            dynamic_axes={"x": {0: "dim_0"}},
            onnx_export=inspect.currentframe().f_code.co_name,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
