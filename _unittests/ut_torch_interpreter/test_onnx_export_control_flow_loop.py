import unittest
from experimental_experiment.ext_test_case import ExtTestCase, requires_onnx_diagnostic
from experimental_experiment.torch_interpreter import to_onnx


class TestOnnxExportControlLoop(ExtTestCase):
    @requires_onnx_diagnostic("0.8.6")
    def test_simple_loop_for_1(self):
        import torch
        from onnx_diagnostic.export.cf_simple_loop_for import simple_loop_for

        class Model(torch.nn.Module):
            def forward(self, n_iter, x):
                def body(i, x):
                    return (x[: i.item() + 1].unsqueeze(1),)

                return simple_loop_for(n_iter, body, (x,))

        model = Model()
        n_iter = torch.tensor(4, dtype=torch.int64)
        x = torch.arange(10, dtype=torch.float32)
        expected = model(n_iter, x)
        onx = to_onnx(model, (n_iter, x), dynamic_shapes=({}, {0: "dimdyn"}))
        self.dump_onnx("test_simple_loop_for_1.onnx", onx)
        ref = self.check_ort(onx)
        feeds = dict(n_iter=n_iter, x=x.numpy())
        got = ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)


if __name__ == "__main__":
    unittest.main(verbosity=2)
