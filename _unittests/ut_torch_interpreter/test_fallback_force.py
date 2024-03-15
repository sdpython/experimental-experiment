import unittest
import numpy as np
from onnx.reference.op_run import OpRun
from experimental_experiment.ext_test_case import ExtTestCase, skipif_ci_windows
from experimental_experiment.reference import ExtendedReferenceEvaluator


class celu_default(OpRun):
    op_domain = "aten.lib"

    def _run(self, x, alpha=1.0):
        y = np.maximum(x, 0) + np.minimum(alpha * (np.exp(x / alpha) - 1), 0)
        return (y.astype(x.dtype),)


def _f_scaled_dot_product_flash_attention_for_cpu_default(
    query,
    key,
    value,
    dropout_p=0.0,
    is_causal=False,
    return_debug_mask=False,
):
    from torch.nn import functional as F

    return F.scaled_dot_product_attention(
        query, key, value, dropout_p=dropout_p, is_causal=is_causal
    )


class _scaled_dot_product_flash_attention_for_cpu_default(OpRun):
    op_domain = "aten.lib"

    def _run(
        self,
        query,
        key,
        value,
        dropout_p=0.0,
        is_causal=False,
        return_debug_mask=False,
    ):
        import torch

        tquery = torch.Tensor(query)
        tkey = torch.Tensor(key)
        tvalue = torch.Tensor(value)
        res = _f_scaled_dot_product_flash_attention_for_cpu_default(
            tquery, tkey, tvalue, dropout_p=0.0, is_causal=True
        )
        if isinstance(res, torch.Tensor):
            return (res.numpy(),)
        return tuple(r.numpy() for r in res)


class TestFallbackForce(ExtTestCase):

    @skipif_ci_windows("dynamo not supported on Windows")
    def test_fallback_force(self):
        import torch
        from experimental_experiment.torch_interpreter import to_onnx
        from experimental_experiment.torch_interpreter.dispatcher import (
            ForceDispatcher,
        )

        class Neuron(torch.nn.Module):
            def __init__(self, n_dims: int, n_targets: int):
                super(Neuron, self).__init__()
                self.linear = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                return torch.celu(self.linear(x))

        x = torch.rand(5, 3)
        model = Neuron(3, 1)

        onx = to_onnx(model, (x,), input_names=["x"], dispatcher=ForceDispatcher())
        self.assertEqual([n.op_type for n in onx.graph.node], ["Gemm", "celu_default"])
        self.assertEqual([n.domain for n in onx.graph.node], ["", "aten.lib"])
        ext = ExtendedReferenceEvaluator(onx, new_ops=[celu_default])
        got = ext.run(None, {"x": x.numpy()})[0]
        self.assertEqual(got.shape, (5, 1))

    @skipif_ci_windows("dynamo not supported on Windows")
    def test_fallback_force_llama_sdpa(self):
        from experimental_experiment.torch_helper.llama_helper import get_llama_model
        from experimental_experiment.torch_interpreter import to_onnx
        from experimental_experiment.torch_interpreter.dispatcher import (
            ForceDispatcher,
        )

        model, example_args_collection = get_llama_model(
            input_dims=[(9, 15)], _attn_implementation="sdpa", with_mask=False
        )
        expected = model(*example_args_collection[0])
        onx = to_onnx(
            model,
            example_args_collection[0],
            input_names=["input0"],
            dispatcher=ForceDispatcher(
                {
                    "_scaled_dot_product_flash_attention_for_cpu_default": _f_scaled_dot_product_flash_attention_for_cpu_default
                }
            ),
        )
        dot = [n for n in onx.graph.node if "scaled" in n.op_type]
        self.assertEqual(len(dot), 1)
        dot = dot[0]
        self.assertEqual(
            dot.op_type, "_scaled_dot_product_flash_attention_for_cpu_default"
        )
        self.assertEqual(len(dot.attribute), 2)
        att = dot.attribute[0]
        self.assertEqual(att.name, "dropout_p")
        self.assertEqual(att.f, 0)

        ext = ExtendedReferenceEvaluator(
            onx, new_ops=[_scaled_dot_product_flash_attention_for_cpu_default]
        )
        names = [i.name for i in onx.graph.input]
        got = ext.run(
            None, dict(zip(names, [i.numpy() for i in example_args_collection[0]]))
        )
        self.assertEqualArray(expected[0], got[0], atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
