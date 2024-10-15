import unittest
from experimental_experiment.ext_test_case import ExtTestCase, requires_torch


class TestExportDynamicShapes(ExtTestCase):
    @classmethod
    def get_class_model(cls):
        import torch

        class Neuron(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.linear = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                return torch.sigmoid(self.linear(x))

        return Neuron

    @classmethod
    def get_class_model_2(cls, wrapped=False):
        import torch

        class Neuron(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.linear = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x, y):
                return torch.sigmoid(self.linear(x + y))

        if not wrapped:
            return Neuron

        class Wrapped(Neuron):
            def forward(self, *args):
                return super().forward(*args)

        return Wrapped

    def test_dynamic_batch_2(self):
        import torch

        Neuron = self.get_class_model()
        nn = Neuron()
        args = (torch.randn(2, 5),)
        batch = torch.export.Dim("batch", min=1, max=1024)
        compiled = torch.export.export(nn, args, dynamic_shapes=({0: batch},))
        expected = nn(*args)
        mod = compiled.module()
        got = mod(*args)
        self.assertEqualArray(expected=expected, value=got)

    @requires_torch("2.7")
    def test_dynamic_batch_1b(self):
        import torch

        Neuron = self.get_class_model()
        nn = Neuron()
        args = (torch.randn(1, 5),)
        batch = torch.export.Dim("batch", min=1, max=1024)
        compiled = torch.export.export(nn, args, dynamic_shapes=({0: batch},))
        expected = nn(*args)
        mod = compiled.module()
        got = mod(*args)
        self.assertEqualArray(expected=expected, value=got)

    def test_dynamic_batch_2_inputs_tuple(self):
        import torch

        Neuron2 = self.get_class_model_2()
        nn = Neuron2()
        args = (torch.randn(2, 5), torch.randn(2, 5))
        batch = torch.export.Dim("batch", min=1, max=1024)

        compiled = torch.export.export(nn, args, dynamic_shapes=({0: batch}, {0: batch}))
        expected = nn(*args)
        mod = compiled.module()
        got = mod(*args)
        self.assertEqualArray(expected=expected, value=got)

    def test_dynamic_batch_2_inputs_dict(self):
        import torch

        Neuron2 = self.get_class_model_2()
        nn = Neuron2()
        args = (torch.randn(2, 5), torch.randn(2, 5))
        batch = torch.export.Dim("batch", min=1, max=1024)

        compiled = torch.export.export(
            nn, args, dynamic_shapes={"x": {0: batch}, "y": {0: batch}}
        )
        expected = nn(*args)
        mod = compiled.module()
        got = mod(*args)
        self.assertEqualArray(expected=expected, value=got)

    def test_dynamic_batch_2_inputs_wrapped_tuple(self):
        import torch

        Wrapped = self.get_class_model_2(wrapped=True)
        nn = Wrapped()
        args = (torch.randn(2, 5), torch.randn(2, 5))
        batch = torch.export.Dim("batch", min=1, max=1024)

        compiled = torch.export.export(nn, args, dynamic_shapes=(({0: batch}, {0: batch}),))
        expected = nn(*args)
        mod = compiled.module()
        got = mod(*args)
        self.assertEqualArray(expected=expected, value=got)

    def test_dynamic_batch_2_inputs_wrapped_dict(self):
        import torch

        Wrapped = self.get_class_model_2(wrapped=True)
        nn = Wrapped()
        args = (torch.randn(2, 5), torch.randn(2, 5))
        batch = torch.export.Dim("batch", min=1, max=1024)

        compiled = torch.export.export(
            nn, args, dynamic_shapes=({"args": ({0: batch}, {0: batch})})
        )
        expected = nn(*args)
        mod = compiled.module()
        got = mod(*args)
        self.assertEqualArray(expected=expected, value=got)


if __name__ == "__main__":
    unittest.main(verbosity=2)
