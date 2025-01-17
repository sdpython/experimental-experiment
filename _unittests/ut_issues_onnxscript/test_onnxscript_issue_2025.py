import unittest
import onnx
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
    has_onnxscript,
    requires_torch,
)
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.torch_interpreter import to_onnx, ExportOptions


class TestOnnxScriptIssue2025(ExtTestCase):
    @requires_torch("2.6")
    @ignore_warnings((FutureWarning, DeprecationWarning))
    def test_adaptive_enc_mask(self):
        import torch

        def adaptive_enc_mask(x_len, chunk_start_idx, left_window=0, right_window=0):
            # first idx of each chunk, such as [0,18,36,48].
            chunk_start_idx = torch.Tensor(chunk_start_idx).long()
            # append 0 to the beginning, so it becomes [0, 0, 18, 36, 48]
            start_pad = torch.nn.functional.pad(chunk_start_idx, (1, 0))
            # append x_len to the end, so it becomes [0,18,36,48, x_len]
            end_pad = torch.nn.functional.pad(chunk_start_idx, (0, 1), value=x_len)
            # seq_range size: [x_len, 1]
            seq_range = torch.arange(0, x_len).unsqueeze(-1)
            # idx size: [x_len]
            idx = ((seq_range < end_pad) & (seq_range >= start_pad)).nonzero()[:, 1]
            # boundary size: [x_len]
            # boundary = end_pad[idx]
            # seq_range_expand size [x_len, x_len]
            seq_range_expand = torch.arange(0, x_len).unsqueeze(0).expand(x_len, -1)
            idx_left = idx - left_window
            idx_left[idx_left < 0] = 0
            boundary_left = start_pad[idx_left]
            mask_left = seq_range_expand >= boundary_left.unsqueeze(-1)
            idx_right = idx + right_window
            idx_right[idx_right > len(chunk_start_idx)] = len(chunk_start_idx)
            boundary_right = end_pad[idx_right]
            mask_right = seq_range_expand < boundary_right.unsqueeze(-1)
            return mask_left & mask_right

        class MyModule(torch.nn.Module):
            def forward(self, X):
                x_len = 10  # 368
                chunk_start_idx = [4]
                left_window = 18
                result = adaptive_enc_mask(x_len, chunk_start_idx, left_window, right_window=0)
                return X + torch.unsqueeze(result, -1)

        torch_model = MyModule()
        torch_model.eval()
        inputs = (torch.randn(1, 1, 368),)
        expected = torch_model(*inputs)

        onx = to_onnx(torch_model, inputs)
        input_name = onx.graph.input[0].name
        onnx.save(onx, "test_adaptive_enc_mask_custom.onnx")
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, {input_name: inputs[0].numpy()})
        self.assertEqualArray(expected, got[0])

        onx = to_onnx(
            torch_model,
            inputs,
            export_options=ExportOptions(strict=False, decomposition_table="default"),
        )
        onnx.save(onx, "test_adaptive_enc_mask_custom_dec.onnx")
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, {input_name: inputs[0].numpy()})
        self.assertEqualArray(expected, got[0])

        if not has_onnxscript("0.2"):
            # This still fails.
            return

        program = torch.onnx.export(torch_model, inputs, dynamo=True)
        program.save(r"test_adaptive_enc_mask_not_optimized.onnx")
        program.optimize()
        program.save(r"test_adaptive_enc_mask.onnx")
        ref = ExtendedReferenceEvaluator(program.model_proto)
        got = ref.run(None, {"x": inputs[0].numpy()})
        self.assertEqualArray(expected, got[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
