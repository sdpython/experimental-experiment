import unittest
from onnx.reference import ReferenceEvaluator
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
    requires_cuda,
    requires_onnxruntime,
    requires_torch,
    requires_transformers,
    skipif_ci_windows,
)
from experimental_experiment.reference import ExtendedReferenceEvaluator, OrtEval
from experimental_experiment.xbuilder import OptimizationOptions
from experimental_experiment.torch_interpreter import to_onnx
from experimental_experiment.torch_models.llama_helper import get_llama_model


class TestOnnxExportDynamicShapes(ExtTestCase):
    @skipif_ci_windows("not supported yet on Windows")
    @ignore_warnings((UserWarning, DeprecationWarning))
    @requires_torch("2.5")
    def test_linear_regression_dynamic_batch_first(self):
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
        onx, _builder = to_onnx(
            model,
            (x,),
            input_names=["x"],
            options=OptimizationOptions(patterns=None),
            dynamic_shapes=dynamic_shapes,
            verbose=0,
            return_builder=True,
        )
        s = _builder.pretty_text()
        self.assertIn("batch x 1", s)
        self.assertIn("batch x 3", s)

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

    @skipif_ci_windows("not supported yet on Windows")
    @ignore_warnings((UserWarning, DeprecationWarning))
    @requires_torch("2.5")
    def test_linear_regression_dynamic_derived_batch(self):
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

        onx, builder = to_onnx(
            model,
            (x,),
            input_names=["x"],
            options=OptimizationOptions(patterns=None),
            return_builder=True,
        )
        self.assertIn("|T1: 11 x 1", builder.pretty_text())
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
        x = torch.randn(10, 3, dtype=torch.float32)
        expected = model(x)
        self.assertEqual(expected.shape, (x.shape[0], 1))

        dynamic_shapes = {"x": {0: torch.export.Dim("batch") * 2}}
        onx, _builder = to_onnx(
            model,
            (x,),
            input_names=["x"],
            options=OptimizationOptions(patterns=None),
            dynamic_shapes=dynamic_shapes,
            verbose=0,
            return_builder=True,
        )
        self.assertIn("|T1: 2*batch x 1", _builder.pretty_text())
        shape = tuple(
            d.dim_param if d.dim_param else d.dim_value
            for d in onx.graph.input[0].type.tensor_type.shape.dim
        )
        self.assertEqual(("2*batch", 3), shape)
        shape = tuple(
            d.dim_param if d.dim_param else d.dim_value
            for d in onx.graph.output[0].type.tensor_type.shape.dim
        )
        self.assertEqual(("2*batch", 1), shape)

        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {"x": x.detach().cpu().numpy()})
        self.assertEqualArray(expected, got[0], atol=1e-5)

    @skipif_ci_windows("not supported yet on Windows")
    @ignore_warnings((UserWarning, DeprecationWarning))
    @requires_torch("2.5")
    def test_linear_regression_dynamic_batch_as_tuple(self):
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

        dynamic_shapes = ({0: torch.export.Dim("batch")},)
        onx, _builder = to_onnx(
            model,
            (x,),
            input_names=["x"],
            options=OptimizationOptions(patterns=None),
            dynamic_shapes=dynamic_shapes,
            verbose=0,
            return_builder=True,
        )
        self.assertIn("dynals: s0 -> 'batch'", _builder.pretty_text())

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

    @skipif_ci_windows("not supported yet on Windows")
    @ignore_warnings((UserWarning, DeprecationWarning))
    @requires_torch("2.5")
    def test_linear_regression_dynamic_batch_only_dynamic(self):
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

        dynamic_shapes = {"x": {0: torch.export.Dim("batch")}}
        onx, builder = to_onnx(
            model,
            (x,),
            input_names=["x"],
            options=OptimizationOptions(patterns=None),
            dynamic_shapes=dynamic_shapes,
            verbose=0,
            return_builder=True,
        )
        self.assertNotIn("constraints", builder.get_debug_msg())

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

    @skipif_ci_windows("not supported yet on Windows")
    @requires_torch("2.4", "bug")
    @requires_transformers("4.41.0", "dynamic shapes issue")
    @ignore_warnings(DeprecationWarning)
    def test_export_llama_model_dynamic_shapes_x1_cpu(self):
        import torch
        import onnxruntime

        with torch.no_grad():
            input_dims = [(2, 1024), (3, 1024)]
            model, input_tensors = get_llama_model(input_dims, with_mask=False)
            onx = to_onnx(
                model,
                input_tensors[0],
                dynamic_shapes={"input_ids": {0: torch.export.Dim("batch", min=2, max=8192)}},
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

    @skipif_ci_windows("not supported yet on Windows")
    @requires_torch("2.4", "bug")
    @requires_transformers("4.41.0", "dynamic shapes issue")
    @requires_onnxruntime("1.18")
    @ignore_warnings(DeprecationWarning)
    def test_export_llama_model_dynamic_shapes_x2_cpu_1(self):
        import torch
        import onnxruntime

        with torch.no_grad():
            input_dims = [(2, 1024), (3, 1024)]
            model, input_tensors = get_llama_model(input_dims, with_mask=False)
            onx = to_onnx(
                model,
                input_tensors[0],
                dynamic_shapes={
                    "input_ids": {
                        0: torch.export.Dim("batch", min=2, max=1024),
                        1: torch.export.Dim("length", min=1, max=2048),
                    }
                },
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

    @skipif_ci_windows("not supported yet on Windows")
    @requires_torch("2.5", "bug")
    @requires_transformers("4.41.0", "dynamic shapes issue")
    @requires_onnxruntime("1.18")
    @ignore_warnings(DeprecationWarning)
    def test_export_llama_model_dynamic_shapes_x2_cpu_tuple(self):
        import torch
        import onnxruntime

        with torch.no_grad():
            input_dims = [(2, 1024), (3, 1024)]
            model, input_tensors = get_llama_model(input_dims, with_mask=False)
            onx, builder = to_onnx(
                model,
                input_tensors[0],
                dynamic_shapes=(
                    {
                        0: torch.export.Dim("batch", min=2, max=1024),
                        1: torch.export.Dim("length", min=1, max=2048),
                    },
                ),
                optimize=False,
                verbose=0,
                return_builder=True,
            )
            self.assertIn("+ 1)", builder.pretty_text())
            if __name__ == "__main__":
                with open(
                    "test_export_llama_model_dynamic_shapes_x2_cpu_tuple.onnx", "wb"
                ) as f:
                    f.write(onx.SerializeToString())

            for i in onx.graph.input:
                shape = i.type.tensor_type.shape
                value = tuple(d.dim_param or d.dim_value for d in shape.dim)
                self.assertEqual(value, ("batch", "length"))
            for i in onx.graph.output:
                shape = i.type.tensor_type.shape
                value = tuple(d.dim_param or d.dim_value for d in shape.dim)
                if value != ("batch", "length", 16):
                    raise AssertionError(f"value={value!r}\n{builder.get_debug_msg()}")
                self.assertEqual(("batch", "length", 16), value)

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

    @skipif_ci_windows("not supported yet on Windows")
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
                dynamic_shapes={"input_ids": {0: torch.export.Dim("batch", min=2, max=8192)}},
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
                    atol=1e-3,
                    msg=f"input {i} failed",
                )

    def _investigate(
        self,
        expected,
        feeds,
        onx,
        opts,
        providers,
        verbose: int = 0,
        atol: float = 1e-4,
    ):
        ref = ExtendedReferenceEvaluator(onx, verbose=verbose)
        results = ref.run(None, feeds)
        self.assertEqualArray(expected[0].detach().cpu().numpy(), results[0], atol=1e-5)
        expected_ref = ref.run(None, feeds, intermediate=True)

        ort_eval = OrtEval(onx, options=opts, providers=providers)
        got_ort = ort_eval.run(None, feeds, intermediate=True)
        for k, v in expected_ref.items():
            if k == "":
                continue
            g = got_ort[k]
            self.assertEqualArray(
                v,
                g,
                atol=atol,
                msg=f"outut {k!r} is different between "
                f"ExtendedReferenceEvaluator and OrtEval",
            )

    @skipif_ci_windows("not supported yet on Windows")
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
                dynamic_shapes={"input_ids": {0: torch.export.Dim("batch", min=2, max=8192)}},
                options=OptimizationOptions(
                    patterns=(
                        "default+onnxruntime+experimental"
                        if get_ort_ext_libs is not None
                        else "default+onnxruntime"
                    ),
                    verbose=0,
                    processor="CUDA",
                ),
            )

        if __name__ == "__main__":
            with open("test_llama_export_dynamic_fused_batch.onnx", "wb") as f:
                f.write(onx.SerializeToString())
        opts = onnxruntime.SessionOptions()
        append_custom_libraries(onx, opts)
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        # self.print_model(onx)

        for i in range(0, len(input_tensors)):
            expected = model(*input_tensors[i])
            sess = onnxruntime.InferenceSession(
                onx.SerializeToString(), opts, providers=providers
            )
            feeds = {}
            for n, t in zip(sess.get_inputs(), input_tensors[i]):
                feeds[n.name] = t.detach().cpu().numpy()
            if __name__ == "__main__":
                self._investigate(expected, feeds, onx, opts, providers, atol=1e-2)
            results = sess.run(None, feeds)
            self.assertEqualArray(
                expected[0].detach().cpu().numpy(),
                results[0],
                atol=1e-3,
                msg=f"input {i} failed with InferenceSession",
            )

    @skipif_ci_windows("not supported yet on Windows")
    @requires_cuda()
    @requires_torch("2.3", "bug")
    @requires_transformers("4.41.0", "dynamic shapes issue")
    @ignore_warnings(DeprecationWarning)
    def test_export_llama_model_dynamic_shapes_x2_fused_cuda(self):
        import torch
        import onnxruntime
        from experimental_experiment.convert.ort_helper import append_custom_libraries

        try:
            from onnx_extended.ortops.optim.cuda import get_ort_ext_libs
        except ImportError:
            get_ort_ext_libs = None

        with torch.no_grad():
            input_dims = [(2, 1024), (3, 1025)]
            model, input_tensors = get_llama_model(input_dims, with_mask=False)
            model = model.to("cuda")
            input_tensors = [tuple(t.to("cuda") for t in p) for p in input_tensors]
            onx = to_onnx(
                model,
                input_tensors[0],
                dynamic_shapes={
                    "input_ids": {
                        0: torch.export.Dim("batch", min=2, max=1024),
                        1: torch.export.Dim("length", min=1, max=2048),
                    }
                },
                options=OptimizationOptions(
                    patterns=(
                        "default+onnxruntime+experimental"
                        if get_ort_ext_libs is not None
                        else "default+onnxruntime"
                    ),
                    verbose=0,
                    processor="CUDA",
                ),
            )

        if __name__ == "__main__":
            with open("test_llama_export_dynamic_fused_batch_length.onnx", "wb") as f:
                f.write(onx.SerializeToString())
        opts = onnxruntime.SessionOptions()
        append_custom_libraries(onx, opts)
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        for i in range(len(input_tensors)):
            expected = model(*input_tensors[i])
            sess = onnxruntime.InferenceSession(
                onx.SerializeToString(), opts, providers=providers
            )
            feeds = {}
            for n, t in zip(sess.get_inputs(), input_tensors[i]):
                feeds[n.name] = t.detach().cpu().numpy()
            # if __name__ == "__main__":
            #    self._investigate(expected, feeds, onx, opts, providers, verbose=0)
            results = sess.run(None, feeds)
            self.assertEqualArray(
                expected[0].detach().cpu().numpy(),
                results[0],
                atol=1e-3,
                msg=f"input {i} failed with InferenceSession",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
