import unittest
import torch
from experimental_experiment.ext_test_case import ExtTestCase
from experimental_experiment.torch_interpreter.tracing_vector import TracingTensor, TracingContext


class TestTracingVeector(ExtTestCase):
    def test_tracing_vector(self):
        class Level1(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.linear1 = torch.nn.Linear(n_dims, n_targets)

            def local_method(self, x):
                return torch.abs(x)

            def forward(self, x):
                z1 = self.linear1(x)
                ones = torch.ones(z1.shape, dtype=z1.dtype, device=z1.device)
                ones[0, 0] = 0
                return torch.sigmoid(self.local_method(z1) + ones)

        class Level2(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.sublevela = Level1(n_dims, n_targets)
                self.linear2 = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                y1 = self.linear2(x)
                y2 = self.sublevela(x)
                ones = torch.ones(y2.shape, dtype=y2.dtype, device=y2.device)
                ones[0, 0] = 0
                return torch.sigmoid(y1 + ones) + y2

        model = Level2()
        inputs = (torch.randn(2, 5),)
        expected = model(*inputs)
        inputs = (TracingTensor(inputs[0]),)
        context = TracingContext()
        with context:
            got = model(*inputs)
            self.assertIsInstance(got, TracingTensor)
        # This comparison cannot happen in with context
        self.assertEqualArray(expected, got)
        self.assertEqual(context.level, 0)
        self.assertEqual(80, len(context.stack))


if __name__ == "__main__":
    unittest.main(verbosity=2)
