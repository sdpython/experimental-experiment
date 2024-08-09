import sys
import unittest
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
    requires_onnxscript,
    requires_torch,
    requires_transformers,
)
from experimental_experiment.torch_models.llama_helper import get_llama_model


class TestDynamoExportDynamicShapes(ExtTestCase):
    @unittest.skipIf(sys.platform == "win32", reason="not supported yet on Windows")
    @requires_onnxscript("0.2", "issue in rewriter")
    @requires_torch("2.6", "bug")
    @requires_transformers("4.41.0", "dynamic shapes issue")
    @ignore_warnings(DeprecationWarning)
    # @unittest.skipIf(True, reason="torch._dynamo.export does not work")
    def test_export_llama_model_dynamic_shapes(self):
        import torch
        import onnxruntime
        from torch.onnx._internal import exporter

        with torch.no_grad():
            input_dims = [(2, 1024), (3, 1024)]
            model, input_tensors = get_llama_model(input_dims, with_mask=False)
            try:
                exported_program = torch.export.export(
                    model,
                    input_tensors[0],
                    dynamic_shapes={
                        "input_ids": {0: torch.export.Dim("batch", min=2, max=8192)}
                    },
                )
            except torch._export.verifier.SpecViolationError:
                exported_program = torch.export._trace._export(
                    model,
                    input_tensors[0],
                    dynamic_shapes={
                        "input_ids": {0: torch.export.Dim("batch", min=2, max=8192)}
                    },
                    pre_dispatch=False,
                )

            export_options = torch.onnx.ExportOptions(dynamic_shapes=True)
            export_options = exporter.ResolvedExportOptions(
                export_options, model=exported_program
            )
            params = dict(exported_program.named_parameters())
            # model.layers.0.self_attn.q_proj.weight -->
            # p_model_layers_0_self_attn_q_proj_weight (in onnx model)
            onnx_program = torch.onnx.dynamo_export(
                exported_program,
                *input_tensors,
                export_options=export_options,
                **params,
            )
            onx = onnx_program.model_proto

            with open("debug.onnx", "wb") as f:
                f.write(onx.SerializeToString())

            for i in range(0, len(input_tensors)):
                expected = model(*input_tensors[i])
                sess = onnxruntime.InferenceSession(
                    onx.SerializeToString(), providers=["CPUExecutionProvider"]
                )
                feeds = {}
                for n, t in zip(sess.get_inputs(), input_tensors[i]):
                    feeds[n.name] = t.detach().cpu().numpy()
                results = sess.run(None, feeds)
                self.assertEqualArray(
                    expected[0].detach().numpy(),
                    results[0],
                    atol=1e-5,
                    msg=f"input {i} failed",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
