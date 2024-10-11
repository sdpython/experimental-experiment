import unittest
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
    requires_onnxruntime_training,
)


class TestIssuePytorch_124525(ExtTestCase):
    @ignore_warnings(UserWarning)
    def test_dort(self):
        import torch

        def _make_aot_ort(dynamic: bool = False) -> tuple:
            from torch.onnx import (
                _OrtBackend as OrtBackend,
                _OrtBackendOptions as OrtBackendOptions,
                ExportOptions,
            )

            export_options = ExportOptions(dynamic_shapes=dynamic)
            options = OrtBackendOptions(export_options=export_options)
            ort_backend = OrtBackend(options=options)
            return ort_backend

        class Linear(torch.nn.Module):
            def __init__(self):
                super(Linear, self).__init__()
                self.linear = torch.nn.Linear(128, 10)
                self.activation = torch.nn.ReLU()

            def forward(self, *inputs):
                input = self.linear(inputs[0])
                input = self.activation(input)
                return input

        model = Linear()
        model.train()
        loss_fn = torch.nn.MSELoss()

        input = torch.randn((64, 128), requires_grad=True)
        labels = torch.randn((64, 10), requires_grad=True)

        compiled_model = torch.compile(model, backend=_make_aot_ort())
        output = compiled_model(*input)
        loss = loss_fn(output, labels)
        loss.backward()

    @ignore_warnings((UserWarning, DeprecationWarning))
    @requires_onnxruntime_training()
    def test_cort(self):
        import torch
        from torch._dynamo.backends.common import aot_autograd
        from experimental_experiment.torch_dynamo import onnx_custom_backend

        backend_onnx = lambda *args, **kwargs: onnx_custom_backend(  # noqa: E731
            *args, target_opset=18, verbose=0, **kwargs
        )

        class Linear(torch.nn.Module):
            def __init__(self):
                super(Linear, self).__init__()
                self.linear = torch.nn.Linear(128, 10)
                self.activation = torch.nn.ReLU()

            def forward(self, *inputs):
                input = self.linear(inputs[0])
                input = self.activation(input)
                return input

        model = Linear()
        model.train()
        loss_fn = torch.nn.MSELoss()

        x = torch.randn((64, 128), requires_grad=True)
        labels = torch.randn((64, 10), requires_grad=True)

        aot_compiler = aot_autograd(fw_compiler=backend_onnx)
        compiled_model = torch.compile(model, backend=aot_compiler)
        output = compiled_model(x)
        loss = loss_fn(output, labels)
        loss.backward()


if __name__ == "__main__":
    unittest.main(verbosity=2)
