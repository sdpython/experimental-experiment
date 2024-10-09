# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.torch_interpreter import to_onnx


class TestTorchIssues2024(unittest.TestCase):

    def test_issue_126972(self):
        # https://github.com/pytorch/pytorch/issues/126972
        import torch

        class BSSModel(torch.nn.Module):
            def __init__(self, n_fft: int = 2048):
                super().__init__()
                self.win_a = torch.hamming_window(n_fft, dtype=torch.float)[:, None]
                self.win_s = self.win_a / (
                    self.win_a**2 + torch.roll(self.win_a, n_fft // 2) ** 2
                )

            def forward(
                self,
                frame: torch.Tensor,
                prev_in: torch.Tensor,
                prev_out: torch.Tensor,
                W: torch.Tensor,
            ):
                buffer = torch.cat((prev_in, frame))
                x = torch.fft.rfft(buffer * self.win_a, axis=0)
                y = (W @ x.unsqueeze(-1)).squeeze(-1)
                curr_out, next_out = torch.chunk(torch.fft.irfft(y, axis=0) * self.win_s, 2)
                return curr_out + prev_out, frame, next_out

        frame_size = 1024
        channels = 4

        model = BSSModel()
        frame = torch.rand(*(frame_size, channels), dtype=torch.float)
        W = torch.rand(*(frame_size + 1, channels, channels), dtype=torch.cfloat)
        expected = model(frame, frame, frame, W)

        onx = to_onnx(model, (frame, frame, frame, W), verbose=0)
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(
            None, dict(zip([_.name for _ in onx.graph.input], (frame, frame, frame, W)))
        )
        self.assertEqualArray(expected.detach().cpu().numpy(), got[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
