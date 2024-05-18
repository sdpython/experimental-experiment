import sys
import unittest
from onnx.reference import ReferenceEvaluator
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
    requires_cuda,
    requires_torch,
    requires_transformers,
)
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.xbuilder import OptimizationOptions
from experimental_experiment.torch_interpreter import to_onnx
from experimental_experiment.torch_models.llama_helper import get_llama_model


class TestOnnxExportDynamicShapes(ExtTestCase):

    @unittest.skipIf(sys.platform == "win32", reason="not supported yet on Windows")
    @ignore_warnings((UserWarning, DeprecationWarning))
    def test_linear_regression_dynamic_batch(self):
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

        onx = to_onnx(
            model,
            (x,),
            input_names=["x"],
            options=OptimizationOptions(patterns=None),
        )
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
        x = torch.randn(11, 3, dtype=torch.float32)
        expected = model(x)
        self.assertEqual(expected.shape, (x.shape[0], 1))

        dynamic_shapes = {"x": {0: torch.export.Dim("batch")}}
        onx = to_onnx(
            model,
            (x,),
            input_names=["x"],
            options=OptimizationOptions(patterns=None),
            dynamic_shapes=dynamic_shapes,
            verbose=0,
        )

        shape = tuple(
            d.dim_param if d.dim_param else d.dim_value
            for d in onx.graph.input[0].type.tensor_type.shape.dim
        )
        self.assertEqual(("batch", 3), shape)
        shape = tuple(
            d.dim_param if d.dim_param else d.dim_value
            for d in onx.graph.output[0].type.tensor_type.shape.dim
        )
        self.assertEqual(("batch", 1), shape)

        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"x": x.detach().cpu().numpy()})
        self.assertEqualArray(expected, got[0], atol=1e-5)

    @unittest.skipIf(sys.platform == "win32", reason="not supported yet on Windows")
    @requires_torch("2.3", "bug")
    @requires_transformers("4.41.0", "dynamic shapes issue")
    @ignore_warnings(DeprecationWarning)
    def test_export_llama_model_dynamic_shapes_cpu(self):
        import torch
        import onnxruntime

        with torch.no_grad():
            input_dims = [(2, 1024), (3, 1024)]
            model, input_tensors = get_llama_model(input_dims, with_mask=False)
            onx = to_onnx(
                model,
                input_tensors[0],
                dynamic_shapes={"input_ids": {0: torch.export.Dim("batch", min=2)}},
            )

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

    @unittest.skipIf(sys.platform == "win32", reason="not supported yet on Windows")
    @requires_cuda()
    @requires_torch("2.3", "bug")
    @requires_transformers("4.41.0", "dynamic shapes issue")
    @ignore_warnings(DeprecationWarning)
    def test_export_llama_model_dynamic_shapes_cuda(self):
        import torch
        import onnxruntime

        with torch.no_grad():
            input_dims = [(2, 1024), (3, 1024)]
            model, input_tensors = get_llama_model(input_dims, with_mask=False)
            model = model.to("cuda")
            input_tensors = [tuple(t.to("cuda") for t in p) for p in input_tensors]
            onx = to_onnx(
                model,
                input_tensors[0],
                dynamic_shapes={"input_ids": {0: torch.export.Dim("batch", min=2)}},
            )

            for i in range(0, len(input_tensors)):
                expected = model(*input_tensors[i])
                sess = onnxruntime.InferenceSession(
                    onx.SerializeToString(),
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                    processor="CUDA",
                )
                feeds = {}
                for n, t in zip(sess.get_inputs(), input_tensors[i]):
                    feeds[n.name] = t.detach().cpu().numpy()
                results = sess.run(None, feeds)
                self.assertEqualArray(
                    expected[0].detach().cpu().numpy(),
                    results[0],
                    atol=1e-5,
                    msg=f"input {i} failed",
                )

    @unittest.skipIf(sys.platform == "win32", reason="not supported yet on Windows")
    @requires_cuda()
    @requires_torch("2.3", "bug")
    @requires_transformers("4.41.0", "dynamic shapes issue")
    @ignore_warnings(DeprecationWarning)
    def test_export_llama_model_dynamic_shapes_fused_cuda(self):
        import torch
        import onnxruntime
        from experimental_experiment.convert.ort_helper import append_custom_libraries

        try:
            from onnx_extended.ortops.optim.cuda import get_ort_ext_libs
        except ImportError:
            get_ort_ext_libs = None

        with torch.no_grad():
            input_dims = [(2, 1024), (3, 1024)]
            model, input_tensors = get_llama_model(input_dims, with_mask=False)
            model = model.to("cuda")
            input_tensors = [tuple(t.to("cuda") for t in p) for p in input_tensors]
            onx = to_onnx(
                model,
                input_tensors[0],
                dynamic_shapes={"input_ids": {0: torch.export.Dim("batch", min=2)}},
                options=OptimizationOptions(
                    patterns=(
                        "default+onnxruntime+experimental"
                        if get_ort_ext_libs is not None
                        else "default+onnxruntime"
                    ),
                    verbose=1,
                    processor="CUDA",
                ),
            )

        with open("test_llama_export_dynamic_fused.onnx", "wb") as f:
            f.write(onx.SerializeToString())
        opts = onnxruntime.SessionOptions()
        append_custom_libraries(onx, opts)

        for i in range(0, len(input_tensors)):
            expected = model(*input_tensors[i])
            sess = onnxruntime.InferenceSession(
                onx.SerializeToString(),
                opts,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            feeds = {}
            for n, t in zip(sess.get_inputs(), input_tensors[i]):
                feeds[n.name] = t.detach().cpu().numpy()
            ref = ExtendedReferenceEvaluator(onx)
            results = ref.run(None, feeds)
            self.assertEqualArray(
                expected[0].detach().cpu().numpy(),
                results[0],
                atol=1e-5,
                msg=f"input {i} failed with ExtendedReferenceEvaluator",
            )
            results = sess.run(None, feeds)
            self.assertEqualArray(
                expected[0].detach().cpu().numpy(),
                results[0],
                atol=1e-5,
                msg=f"input {i} failed with InferenceSession",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
