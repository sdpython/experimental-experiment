import unittest
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    requires_onnx_diagnostic,
    requires_torch,
)
from experimental_experiment.torch_interpreter import to_onnx
from onnx_diagnostic.export.api import to_onnx as to_onnx_diagnostic
from onnx_diagnostic.export.control_flow_onnx import loop_for_onnx


class TestOnnxExportControlLoop(ExtTestCase):
    @requires_torch("2.9.99")
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
        feeds = dict(n_iter=n_iter.numpy(), x=x.numpy())
        got = ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    @requires_torch("2.9.99")
    @requires_onnx_diagnostic("0.8.6")
    def test_loop_two_custom_concatenation_dims(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, n_iter, x):
                def body(i, x):
                    return x[: i.item() + 1].unsqueeze(1), x[: i.item() + 1].unsqueeze(0) + 1

                res = loop_for_onnx(n_iter, body, (x,), concatenation_dims=[0, 1])
                return res[0] + res[1].T

        model = Model()
        n_iter = torch.tensor(4, dtype=torch.int64)
        x = torch.arange(10, dtype=torch.float32)
        expected = torch.tensor([1, 1, 3, 1, 3, 5, 1, 3, 5, 7], dtype=x.dtype).unsqueeze(1)
        got = model(n_iter, x)
        self.assertEqualArray(expected, got)

        ep = torch.export.export(
            model, (n_iter, x), dynamic_shapes=({}, ({0: torch.export.Dim.DYNAMIC}))
        )
        self.assertIn(
            "torch.ops.onnx_higher_ops.loop_for_onnx_TestOnnxExportControlLoop_test_loop_two_custom_concatenation_dims_L_Model_forward_L_body_u1x1_1xu2_0x1",
            str(ep),
        )

        onx = to_onnx_diagnostic(
            model,
            (n_iter, x),
            dynamic_shapes=({}, ({0: torch.export.Dim.DYNAMIC})),
            exporter="custom",
            use_control_flow_dispatcher=True,
        ).model_proto
        self.dump_onnx("test_loop_two_custom_concatenation_dims.onnx", onx)
        ref = self.check_ort(onx)
        feeds = dict(n_iter=n_iter.numpy(), x=x.numpy())
        got = ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)


if __name__ == "__main__":
    unittest.main(verbosity=2)
