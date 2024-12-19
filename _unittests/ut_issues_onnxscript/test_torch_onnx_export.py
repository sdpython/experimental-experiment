import unittest
from typing import Optional
from onnx.reference import ReferenceEvaluator
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    skipif_ci_windows,
    requires_torch,
    ignore_warnings,
    hide_stdout,
)


class TestTorchOnnxExport(ExtTestCase):

    @skipif_ci_windows("not supported yet on Windows")
    @ignore_warnings((UserWarning, DeprecationWarning))
    @requires_torch("2.5")
    @hide_stdout()
    def test_oxs_linear_regression_dynamic_derived_batch(self):
        import torch

        class TorchLinearRegression(torch.nn.Module):
            def __init__(self, n_dims: int, n_targets: int):
                super(TorchLinearRegression, self).__init__()
                self.linear = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                return self.linear(x)

        # static
        model = TorchLinearRegression(3, 1)
        x = torch.randn(11, 3, dtype=torch.float32)
        expected = model(x)
        self.assertEqual(expected.shape, (x.shape[0], 1))

        onx = torch.onnx.export(model, (x,), dynamo=True).model_proto
        shape = tuple(
            d.dim_param if d.dim_param else d.dim_value
            for d in onx.graph.input[0].type.tensor_type.shape.dim
        )
        self.assertEqual(shape, (11, 3))
        shape = tuple(
            d.dim_param if d.dim_param else d.dim_value
            for d in onx.graph.output[0].type.tensor_type.shape.dim
        )
        self.assertEqual(shape, (11, 1))

        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"x": x.detach().cpu().numpy()})
        self.assertEqualArray(expected, got[0], atol=1e-5)

        # dynamic
        model = TorchLinearRegression(3, 1)
        x = torch.randn(10, 3, dtype=torch.float32)
        expected = model(x)
        self.assertEqual(expected.shape, (x.shape[0], 1))

        dynamic_shapes = {"x": {0: torch.export.Dim("batch") * 2}}
        onx = torch.onnx.export(
            model,
            (x,),
            dynamic_shapes=dynamic_shapes,
            dynamo=True,
        ).model_proto

        shape = tuple(
            d.dim_param if d.dim_param else d.dim_value
            for d in onx.graph.input[0].type.tensor_type.shape.dim
        )
        self.assertEqual(("2*s1", 3), shape)
        shape = tuple(
            d.dim_param if d.dim_param else d.dim_value
            for d in onx.graph.output[0].type.tensor_type.shape.dim
        )
        self.assertEqual(("2*s1", 1), shape)

        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"x": x.detach().cpu().numpy()})
        self.assertEqualArray(expected, got[0], atol=1e-5)

    def test_torch_masked_fill(self):
        import torch

        def _make_causal_mask(
            input_ids_shape: torch.Size,
            dtype: torch.dtype,
            device: torch.device = "cpu",
            past_key_values_length: int = 0,
            sliding_window: Optional[int] = None,
        ):
            """
            Make causal mask used for bi-directional self-attention.
            """
            bsz, tgt_len = input_ids_shape
            mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
            mask_cond = torch.arange(mask.size(-1), device=device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)

            mask = mask.to(dtype)

            if past_key_values_length > 0:
                mask = torch.cat(
                    [
                        torch.zeros(
                            tgt_len, past_key_values_length, dtype=dtype, device=device
                        ),
                        mask,
                    ],
                    dim=-1,
                )

            # add lower triangular sliding window mask if necessary
            if sliding_window is not None:
                diagonal = past_key_values_length - sliding_window - 1

                context_mask = torch.tril(
                    torch.ones_like(mask, dtype=torch.bool), diagonal=diagonal
                )
                mask.masked_fill_(context_mask, torch.finfo(dtype).min)

            return mask[None, None, :, :].expand(
                bsz, 1, tgt_len, tgt_len + past_key_values_length
            )

        class FailingModule(torch.nn.Module):

            def forward(self, input_ids):

                return _make_causal_mask(
                    input_ids.shape, torch.float32, past_key_values_length=32
                )

        input_ids = torch.randint(31730, (1, 30))
        model = FailingModule()
        # expected = model(input_ids)
        ep = torch.export.export(model, (input_ids,))
        self.assertIn("target=torch.ops.aten.expand.default", str(ep.graph))

    def test_torch_upsample(self):
        import torch
        from experimental_experiment.torch_interpreter import to_onnx

        # https://github.com/pytorch/pytorch/issues/142866
        torch.use_deterministic_algorithms(True)
        upsample = torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        x = torch.randn(1, 3, 64, 64).to("cuda:0")
        type(x.size()[0])
        upsample(x)
        onx = to_onnx(upsample, (x,))
        self.assertNotEmpty(onx)
        # torch.onnx.export(
        #    upsample,
        #    (x,),
        #    "test_torch_upsample.onnx",
        #    input_names=["x"],
        #    output_names=["y"],
        #    opset_version=18,
        #    fallback=False,
        #    report=True,
        #    dump_exported_program=True,
        #    dynamo=True,
        # )


if __name__ == "__main__":
    unittest.main(verbosity=2)
