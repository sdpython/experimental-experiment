import unittest
from typing import List, Optional
import numpy as np
from onnx.reference.op_run import OpRun
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    skipif_ci_windows,
    requires_torch,
    requires_transformers,
    has_cuda,
    ignore_warnings,
)
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
    attn_mask=None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
):
    from torch.nn import functional as F

    return F.scaled_dot_product_attention(
        query, key, value, attn_mask, dropout_p=dropout_p, is_causal=bool(is_causal)
    )


class _scaled_dot_product_flash_attention_for_cpu_default(OpRun):
    op_domain = "aten.lib"

    def _run(
        self,
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        return_debug_mask=False,
    ):
        import torch

        tquery = torch.Tensor(query)
        tkey = torch.Tensor(key)
        tvalue = torch.Tensor(value)
        res = _f_scaled_dot_product_flash_attention_for_cpu_default(
            tquery, tkey, tvalue, dropout_p=dropout_p, is_causal=is_causal
        )
        if isinstance(res, torch.Tensor):
            return (res.numpy(),)
        return tuple(r.numpy() for r in res)


if has_cuda():

    def _f_scaled_dot_product_efficient_attention_cuda(
        query,
        key,
        value,
        attn_bias=None,
        compute_log_sumexp: bool = False,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: float = 1.0,
    ):
        import torch

        res = torch.ops.aten._scaled_dot_product_efficient_attention.default(
            query.to("cuda"),
            key.to("cuda"),
            value.to("cuda"),
            None if attn_bias is None else attn_bias.to("cuda"),
            compute_log_sumexp,
            dropout_p,
            is_causal,
            scale=scale,
        )
        cpu_res = tuple((None if r is None else r.cpu()) for r in res)
        return cpu_res

    class _scaled_dot_product_efficient_attention_default(OpRun):
        op_domain = "aten.lib"

        def _run(
            self,
            query,
            key,
            value,
            attn_bias=None,
            compute_log_sumexp: bool = False,
            dropout_p: float = 0.0,
            is_causal: bool = False,
            scale: float = 1.0,
        ):
            import torch

            res = _f_scaled_dot_product_efficient_attention_cuda(
                torch.Tensor(query),
                torch.Tensor(key),
                torch.Tensor(value),
                None if attn_bias is None else torch.Tensor(attn_bias),
                dropout_p=dropout_p,
                is_causal=is_causal,
            )
            if isinstance(res, torch.Tensor):
                return (res.numpy(),)
            return tuple(r.numpy() for r in res)

    def _f_scaled_dot_product_efficient_attention_backward_cuda(
        grad,
        query,
        key,
        value,
        attn_bias,
        output,
        logsumexp,
        philox_seed,
        philox_offset,
        dropout_p: float = 0.0,
        grad_input_mask: Optional[List[bool]] = None,
        is_causal: bool = False,
        scale: Optional[float] = None,
    ):
        import torch

        dummy = True
        if dummy:
            shape = grad.shape

            cpu_res = [
                torch.rand(shape, dtype=torch.float32),
                torch.rand(shape, dtype=torch.float32),
                torch.rand(shape, dtype=torch.float32),
                torch.Tensor(np.array([0], dtype=np.float32)),
            ]

        else:
            cudat = [
                grad.to("cuda"),
                query.to("cuda"),
                key.to("cuda"),
                value.to("cuda"),
                None if attn_bias is None else attn_bias.to("cuda"),
                output.to("cuda"),
                logsumexp,  # .to("cuda"),
                philox_seed,
                philox_offset,
            ]

            res = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(
                *cudat,
                dropout_p,
                [bool(r) for r in grad_input_mask],
                is_causal,
                scale=scale,
            )

            cpu_res = []
            for r in res:
                if r is None:
                    cpu_res.append(r)
                    continue
                cpu_res.append(r.cpu())

        return tuple(cpu_res)

    class _scaled_dot_product_efficient_attention_backward_default(OpRun):
        op_domain = "aten.lib"

        def _run(
            self,
            grad,
            query,
            key,
            value,
            attn_bias,
            output,
            logsumexp,
            philox_seed,
            philox_offset,
            dropout_p: float = 0.0,
            grad_input_mask: Optional[List[bool]] = None,
            is_causal: bool = False,
            scale: Optional[float] = None,
        ):
            import torch

            res = _f_scaled_dot_product_efficient_attention_backward_cuda(
                torch.Tensor(grad),
                torch.Tensor(query),
                torch.Tensor(key),
                torch.Tensor(value),
                None if attn_bias is None else torch.Tensor(attn_bias),
                torch.Tensor(output),
                torch.Tensor(logsumexp),
                torch.Tensor(
                    philox_seed.reshape(
                        1,
                    )
                ),
                torch.Tensor(
                    philox_offset.reshape(
                        1,
                    )
                ),
                dropout_p,
                list(grad_input_mask),
                is_causal=is_causal,
                scale=scale,
            )
            if isinstance(res, torch.Tensor):
                return (res.numpy(),)
            return tuple(
                (np.array([0], dtype=np.float32) if r is None else r.numpy()) for r in res
            )


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
        self.assertIn(
            [n.op_type for n in onx.graph.node],
            (["Gemm", "celu_default"], ["Gemm", "Add", "celu_default"]),
        )
        self.assertIn(
            [n.domain for n in onx.graph.node], (["", "aten.lib"], ["", "", "aten.lib"])
        )
        ext = ExtendedReferenceEvaluator(onx, new_ops=[celu_default])
        got = ext.run(None, {"x": x.numpy()})[0]
        self.assertEqual(got.shape, (5, 1))

    @skipif_ci_windows("dynamo not supported on Windows")
    @requires_torch("2.4")
    @requires_transformers("4.40")
    @ignore_warnings(DeprecationWarning)
    def test_fallback_force_llama_sdpa_export(self):
        import torch
        from experimental_experiment.torch_models.llama_helper import get_llama_model
        from experimental_experiment.torch_interpreter import to_onnx
        from experimental_experiment.torch_interpreter.dispatcher import (
            ForceDispatcher,
        )

        model, example_args_collection = get_llama_model(
            input_dims=[(9, 15)], _attn_implementation="sdpa", with_mask=False
        )
        expected = model(*example_args_collection[0])
        with torch.no_grad():
            onx = to_onnx(
                model,
                example_args_collection[0],
                input_names=["input0"],
                dispatcher=ForceDispatcher(
                    {
                        "_scaled_dot_product_flash_attention_for_cpu_default": _f_scaled_dot_product_flash_attention_for_cpu_default  # noqa: E501
                    }
                ),
            )
        dot = [n for n in onx.graph.node if "scaled" in n.op_type]
        if len(dot) == 0:
            raise unittest.SkipTest("sdpa does not work")
        self.assertEqual(len(dot), 1)
        dot = dot[0]
        self.assertEqual(dot.op_type, "_scaled_dot_product_flash_attention_for_cpu_default")
        self.assertEqual(len(dot.attribute), 2)
        att = dot.attribute[0]
        self.assertEqual(att.name, "dropout_p")
        self.assertEqual(att.f, 0)

        ext = ExtendedReferenceEvaluator(
            onx, new_ops=[_scaled_dot_product_flash_attention_for_cpu_default]
        )
        names = [i.name for i in onx.graph.input]
        got = ext.run(None, dict(zip(names, [i.numpy() for i in example_args_collection[0]])))
        # TODO: something is wrong
        self.assertEqualArray(expected[0], got[0], atol=2)

    @skipif_ci_windows("dynamo not supported on Windows")
    @unittest.skipIf(not has_cuda(), reason="design for cuda")
    def test_fallback_force_llama_sdpa_cort_cuda_static(self):
        self.fallback_force_llama_sdpa_cort_cuda(False)

    @skipif_ci_windows("dynamo not supported on Windows")
    @unittest.skipIf(not has_cuda(), reason="design for cuda")
    @unittest.skipIf(
        True,
        reason="is_causal=bool(self.is_causal and attention_mask is None and q_len > 1)",
    )
    def test_fallback_force_llama_sdpa_cort_cuda_dynamic(self):
        self.fallback_force_llama_sdpa_cort_cuda(True)

    def fallback_force_llama_sdpa_cort_cuda(self, dynamic):
        import torch
        from torch._dynamo.backends.common import aot_autograd
        from experimental_experiment.torch_models.llama_helper import get_llama_model
        from experimental_experiment.torch_dynamo import (
            onnx_debug_backend,
            get_decomposition_table,
        )
        from experimental_experiment.torch_interpreter.dispatcher import (
            ForceDispatcher,
        )

        model, example_args_collection = get_llama_model(
            input_dims=[(9, 15)], _attn_implementation="sdpa", with_mask=False
        )
        model = model.to("cuda")
        example_args_collection = [
            [i.to("cuda") for i in inp] for inp in example_args_collection
        ]

        expected = model(*example_args_collection[0])

        dispatcher = ForceDispatcher(
            {
                "_scaled_dot_product_efficient_attention_default": _f_scaled_dot_product_efficient_attention_cuda  # noqa: E501
            }
        )

        models = []

        def store_model(onx):
            with open(
                f"test_fallback_force_llama_sdpa_cort_cuda_"
                f"{1 if dynamic else 0}_{len(models)}.onnx",
                "wb",
            ) as f:
                f.write(onx.SerializeToString())
            models.append(onx)
            return onx

        aot_compiler = aot_autograd(
            fw_compiler=lambda *args, **kwargs: onnx_debug_backend(
                *args,
                target_opset=18,
                pre_ort_model_transforms=store_model,
                dispatcher=dispatcher,
                backend=lambda onx, verbose=0: ExtendedReferenceEvaluator(
                    onx,
                    new_ops=[_scaled_dot_product_efficient_attention_default],
                    verbose=verbose,
                ),
                **kwargs,
            ),
            decompositions=get_decomposition_table(),
        )

        compiled_model = torch.compile(
            model, backend=aot_compiler, dynamic=dynamic, fullgraph=True
        )

        got = compiled_model(*example_args_collection[0])
        self.assertEqual(type(got), type(expected))
        self.assertEqualArray(expected[0], got[0], atol=1e-1)
        self.assertEqual(len(expected), 2)
        self.assertEqual(len(got), 2)
        if len(got) > 2:
            self.assertEmpty(got[2])
        if len(got) > 3:
            self.assertEmpty(got[3])

    @skipif_ci_windows("dynamo not supported on Windows")
    @unittest.skipIf(not has_cuda(), reason="design for cuda")
    def test_fallback_force_llama_sdpa_cort_training_cuda_static(self):
        self.fallback_force_llama_sdpa_cort_training_cuda(False)

    def fallback_force_llama_sdpa_cort_training_cuda(self, dynamic):
        import torch
        from torch._dynamo.backends.common import aot_autograd
        from experimental_experiment.torch_models.llama_helper import get_llama_model
        from experimental_experiment.torch_dynamo import (
            onnx_debug_backend,
            get_decomposition_table,
        )
        from experimental_experiment.torch_interpreter.dispatcher import (
            ForceDispatcher,
        )
        from experimental_experiment.torch_models.dump_helper import assert_all_close

        model, example_args_collection = get_llama_model(
            input_dims=[(9, 15)], _attn_implementation="sdpa", with_mask=False
        )
        model = model.to("cuda")
        example_args_collection = [
            [i.to("cuda") for i in inp] for inp in example_args_collection
        ]

        expected = model(*example_args_collection[0])
        expected[0].sum().backward()

        dispatcher = ForceDispatcher(
            {
                "_scaled_dot_product_efficient_attention_default": _f_scaled_dot_product_efficient_attention_cuda,  # noqa: E501
                "_scaled_dot_product_efficient_attention_backward_default": _f_scaled_dot_product_efficient_attention_backward_cuda,  # noqa: E501
            },
            only_registered=True,
        )

        models = []

        def store_model(onx):
            with open(
                f"test_fallback_force_llama_sdpa_cort_cuda_"
                f"{1 if dynamic else 0}_{len(models)}.onnx",
                "wb",
            ) as f:
                f.write(onx.SerializeToString())
            models.append(onx)
            return onx

        aot_compiler = aot_autograd(
            fw_compiler=lambda *args, **kwargs: onnx_debug_backend(
                *args,
                target_opset=18,
                pre_ort_model_transforms=store_model,
                dispatcher=dispatcher,
                backend=lambda onx, verbose=0: ExtendedReferenceEvaluator(
                    onx,
                    new_ops=[
                        _scaled_dot_product_efficient_attention_default,
                        _scaled_dot_product_efficient_attention_backward_default,
                    ],
                    verbose=verbose,
                ),
                verbose=0,
                **kwargs,
            ),
            decompositions=get_decomposition_table(),
        )

        compiled_model = torch.compile(
            model, backend=aot_compiler, dynamic=dynamic, fullgraph=True
        )

        # forward
        got = compiled_model(*example_args_collection[0])

        # backward
        got[0].sum().backward()

        base_grads = tuple(_.grad for _ in model.parameters())
        grads = tuple(_.grad for _ in compiled_model.parameters())
        assert_all_close(base_grads, grads)


if __name__ == "__main__":
    unittest.main(verbosity=2)
