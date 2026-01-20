import unittest
import torch
from experimental_experiment.ext_test_case import ExtTestCase, hide_stdout
from experimental_experiment.torch_interpreter.tracing_vector import TracingTensor, TracingContext


class TestTracingVeector(ExtTestCase):
    @hide_stdout()
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
                x = x.squeeze(-1)
                y1 = self.linear2(x)
                y2 = self.sublevela(x)
                ones = torch.ones(y2.shape, dtype=y2.dtype, device=y2.device)
                ones[0, 0] = 0
                return torch.sigmoid(y1 + ones) + y2

        model = Level2()
        inputs = (torch.randn(2, 5, 1),)
        expected = model(*inputs)

        all_ops = []

        class DispatchLog(torch.utils._python_dispatch.TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args, kwargs=None):
                all_ops.append(func)
                return func(*args, **(kwargs or {}))

        with DispatchLog():
            model(*inputs)

        self.assertEqual(len(all_ops), 23)

        traced_inputs = (TracingTensor(inputs[0]),)
        context = TracingContext(model, verbose=1, debug_counts={"input": 1})
        with context:
            got = model(*traced_inputs)
            self.assertIsInstance(got, TracingTensor)
        # This comparison cannot happen in with context
        self.assertEqual(expected.tolist(), got.tolist())
        self.assertEqual(context.level, 0)
        self.assertEqual(156, len(context.stack))

        ep = torch.export.export(model, inputs, dynamic_shapes=({0: torch.export.Dim.DYNAMIC},))
        print("--------")
        print(ep.graph)
        print("--------")
        print(context.graph)
        print("--------")


if __name__ == "__main__":
    unittest.main(verbosity=2)
