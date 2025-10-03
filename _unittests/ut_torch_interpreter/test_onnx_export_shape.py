import unittest
from typing import Any, List, Optional
import onnx
from experimental_experiment.ext_test_case import ExtTestCase, requires_torch
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.torch_interpreter import to_onnx, ExportOptions
from experimental_experiment.xbuilder import OptimizationOptions


class TestOnnxExportShape(ExtTestCase):
    def _call_exporter(
        self,
        test_name: str,
        exporter: str,
        model: "torch.nn.Module",  # noqa: F821
        inputs: List[Any],
        decomposition: bool = False,
        verbose: int = 0,
        optimize: bool = False,
        strict: bool = False,
        patterns: Optional[str] = None,
        dynamic_shapes: Optional[Any] = None,
        output_dynamic_shapes: Optional[Any] = None,
        processor: str = "CPU",
        output_names: Optional[List[str]] = None,
        constant_folding: bool = False,
        oblivious: bool = False,
        patch: bool = False,
    ) -> str:
        import torch

        filename = f"{test_name}_{exporter}_{'dec' if decomposition else ''}.onnx"
        if exporter == "script":
            torch.onnx.export(model, inputs, filename, opset_version=18)
        elif exporter == "dynamo":
            torch.onnx.export(model, inputs, filename, dynamo=True)
        else:
            export_options = ExportOptions(
                decomposition_table="all" if decomposition else None,
                strict=strict,
                oblivious=oblivious,
            )
            opt_options = (
                OptimizationOptions(
                    patterns=patterns,
                    processor=processor,
                    constant_folding=constant_folding,
                )
                if patterns or processor != "CPU"
                else None
            )
            if patch:
                from onnx_diagnostic.torch_export_patches import torch_export_patches

                with torch_export_patches():
                    to_onnx(
                        model,
                        inputs,
                        filename=filename,
                        export_options=export_options,
                        verbose=verbose,
                        optimize=optimize,
                        options=opt_options,
                        dynamic_shapes=dynamic_shapes,
                        output_names=output_names,
                        output_dynamic_shapes=output_dynamic_shapes,
                    )
            else:
                to_onnx(
                    model,
                    inputs,
                    filename=filename,
                    export_options=export_options,
                    verbose=verbose,
                    optimize=optimize,
                    options=opt_options,
                    dynamic_shapes=dynamic_shapes,
                    output_names=output_names,
                    output_dynamic_shapes=output_dynamic_shapes,
                )
        return filename

    @requires_torch("2.6", "torch.export.Dim.AUTO")
    def test_shape_AUTO(self):
        import torch

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv1d(16, 32, 1)

            def forward(self, x):
                return self.conv(x) + torch.tensor([1], dtype=x.dtype)

        model = Model()
        xs = (torch.randn((2, 16, 24)),)
        expected = model(*xs)
        model_path = self._call_exporter(
            "test_shape_DYN",
            "custom",
            model,
            xs,
            dynamic_shapes={
                "x": {
                    0: torch.export.Dim.AUTO,
                    1: torch.export.Dim.AUTO,
                    2: torch.export.Dim.AUTO,
                }
            },
        )
        onx = onnx.load(model_path)
        shape_x = [d.dim_param for d in onx.graph.input[0].type.tensor_type.shape.dim]
        self.assertEqual(shape_x, ["batch", "channel", "D0"])
        sess = ExtendedReferenceEvaluator(model_path, verbose=0)
        feeds = dict(zip(sess.input_names, [x.numpy() for x in xs]))
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-5)

        # checking with onnxruntime as well
        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        sess = onnxruntime.InferenceSession(
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-5)

    @requires_torch("2.6", "torch.export.Dim.AUTO")
    def test_shape_reshape(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x):
                return x.reshape((-1, 1024)).reshape((-1, 2, 1024))

        model = Model()
        xs = (torch.randn((8, 2048)),)
        expected = model(*xs)
        model_path = self._call_exporter(
            "test_shape_reshape",
            "custom",
            model,
            xs,
            dynamic_shapes={"x": {0: torch.export.Dim.AUTO}},
        )
        onx = onnx.load(model_path)
        shape_x = [d.dim_param for d in onx.graph.input[0].type.tensor_type.shape.dim]
        self.assertEqual(shape_x, ["batch", ""])
        for obs in onx.graph.value_info:
            shape = tuple((d.dim_param or d.dim_value) for d in obs.type.tensor_type.shape.dim)
            self.assertIn(shape, (("batch*2048//1024", 1024), ("batch", 2, 1024)))
        sess = ExtendedReferenceEvaluator(model_path, verbose=0)
        feeds = dict(zip(sess.input_names, [x.numpy() for x in xs]))
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-5)

        # checking with onnxruntime as well
        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        sess = onnxruntime.InferenceSession(
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-5)

    @requires_torch("2.6", "torch.export.Dim.AUTO")
    def test_output_name(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x):
                return x.reshape((-1, 1024)).reshape((-1, 2, 1024))

        model = Model()
        xs = (torch.randn((8, 2048)),)
        expected = model(*xs)
        model_path = self._call_exporter(
            "test_shape_reshape",
            "custom",
            model,
            xs,
            dynamic_shapes={"x": {0: torch.export.Dim.AUTO}},
            output_names=["Y"],
        )
        onx = onnx.load(model_path)
        self.assertEqual(len(onx.graph.output[0].name), 1)
        self.assertEqual(onx.graph.output[0].name, "Y")
        sess = ExtendedReferenceEvaluator(model_path, verbose=0)
        feeds = dict(zip(sess.input_names, [x.numpy() for x in xs]))
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-5)

        # checking with onnxruntime as well
        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        sess = onnxruntime.InferenceSession(
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-5)

    @requires_torch("2.6", "torch.export.Dim.AUTO")
    def test_shape_named_dynamic_shapes(self):
        import torch

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv1d(16, 32, 1)

            def forward(self, x):
                return self.conv(x) + torch.tensor([1], dtype=x.dtype)

        model = Model()
        xs = (torch.randn((2, 16, 24)),)
        expected = model(*xs)
        model_path = self._call_exporter(
            "test_shape_named_dynamic_shapes",
            "custom",
            model,
            xs,
            dynamic_shapes={"x": {0: "num_audios", 2: "num_last"}},
            patch=True,
            oblivious=True,
        )
        onx = onnx.load(model_path)
        shape_x = [
            d.dim_param or d.dim_value for d in onx.graph.input[0].type.tensor_type.shape.dim
        ]
        self.assertEqual(shape_x, ["num_audios", 16, "num_last"])
        sess = ExtendedReferenceEvaluator(model_path, verbose=0)
        feeds = dict(zip(sess.input_names, [x.numpy() for x in xs]))
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-5)

        # checking with onnxruntime as well
        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        sess = onnxruntime.InferenceSession(
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-5)

    @requires_torch("2.6", "torch.export.Dim.AUTO")
    def test_oblivious_dynamic_shapes(self):
        import torch

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv1d(16, 32, 1)

            def forward(self, x):
                return self.conv(x) + torch.tensor([1], dtype=x.dtype)

        model = Model()
        xs = (torch.randn((1, 16, 24)),)
        expected = model(*xs)
        model_path = self._call_exporter(
            "test_shape_named_dynamic_shapes",
            "custom",
            model,
            xs,
            dynamic_shapes={"x": {0: "num_audios", 2: "num_last"}},
            oblivious=True,
            patch=True,
        )
        onx = onnx.load(model_path)
        shape_x = [
            d.dim_param or d.dim_value for d in onx.graph.input[0].type.tensor_type.shape.dim
        ]
        self.assertEqual(shape_x, ["num_audios", 16, "num_last"])
        sess = ExtendedReferenceEvaluator(model_path, verbose=0)
        feeds = dict(zip(sess.input_names, [x.numpy() for x in xs]))
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-5)

        # checking with onnxruntime as well
        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        sess = onnxruntime.InferenceSession(
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-5)

    @requires_torch("2.6", "torch.export.Dim.AUTO")
    def test_reshape_folding(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x):
                return x + torch.tensor([6, 7, 7, 8]).reshape((-1, 4)).to(torch.int64).to(
                    torch.float32
                )

        model = Model()
        xs = (torch.randn((2, 4)),)
        expected = model(*xs)
        model_path = self._call_exporter(
            "test_reshape_folding",
            "custom",
            model,
            xs,
            dynamic_shapes={"x": {0: "dx"}},
            optimize=True,
            patterns="default",
            constant_folding=True,
            verbose=0,
            patch=True,
        )
        onx = onnx.load(model_path)
        self.assertEqual(["Add"], [n.op_type for n in onx.graph.node])
        shape_x = [
            d.dim_param or d.dim_value for d in onx.graph.input[0].type.tensor_type.shape.dim
        ]
        self.assertEqual(shape_x, ["dx", 4])
        sess = ExtendedReferenceEvaluator(model_path, verbose=0)
        feeds = dict(zip(sess.input_names, [x.numpy() for x in xs]))
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-5)

    @requires_torch("2.6", "torch.export.Dim.AUTO")
    def test_rename_output_dynamic_dimension(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.nonzero(x)

        model = Model()
        xs = (torch.randn((2, 4)),)
        expected = model(*xs)
        model_path = self._call_exporter(
            "test_rename_output_dynamic_dimension",
            "custom",
            model,
            xs,
            dynamic_shapes={"x": {0: "dx", 1: "dy"}},
            output_dynamic_shapes={"Y": {0: "numf"}},
            output_names=["Y"],
            optimize=True,
            patterns="default",
            constant_folding=True,
            verbose=0,
        )
        onx = onnx.load(model_path)
        self.assertEqual(["NonZero", "Transpose"], [n.op_type for n in onx.graph.node])
        self.assertEqual(onx.graph.output[0].name, "Y")
        shape_x = [d.dim_param for d in onx.graph.input[0].type.tensor_type.shape.dim]
        self.assertEqual(shape_x, ["dx", "dy"])
        shape_y = [
            d.dim_param or d.dim_value for d in onx.graph.output[0].type.tensor_type.shape.dim
        ]
        self.assertEqual(shape_y, ["numf", 2])
        sess = ExtendedReferenceEvaluator(model_path, verbose=0)
        feeds = dict(zip(sess.input_names, [x.numpy() for x in xs]))
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-5)

    def test_dynamic_01_dimension(self):
        import torch
        from experimental_experiment.export_helpers import torch_export

        class Model(torch.nn.Module):
            def forward(self, x):
                return x @ torch.arange(x.shape[1], dtype=torch.float32).reshape((-1, 1))

        model = Model()
        x = torch.arange(6, dtype=torch.float32).reshape((-1, 3))
        expected = model(x)
        DYN = torch.export.Dim.DYNAMIC
        ds = ({0: DYN, 1: DYN},)
        ep = torch.export.export(model, (x,), dynamic_shapes=ds)
        y = ep.module()(x)
        self.assertEqualArray(expected, y)

        # 0
        with torch.fx.experimental._config.patch(backed_size_oblivious=True):
            ep = torch.export.export(
                model, (torch.empty((0, 3), dtype=torch.float32),), dynamic_shapes=ds
            )
        y = ep.module()(x)
        self.assertEqualArray(expected, y)

        # 1
        with torch.fx.experimental._config.patch(backed_size_oblivious=True):
            ep = torch.export.export(
                model, (torch.zeros((1, 3), dtype=torch.float32),), dynamic_shapes=ds
            )
        y = ep.module()(x)
        self.assertEqualArray(expected, y)

        # w 1
        ep = torch_export(
            model,
            (torch.empty((1, 3), dtype=torch.float32),),
            dynamic_shapes=ds,
            backed_size_oblivious="auto",
        )
        y = ep.module()(x)
        self.assertEqualArray(expected, y)

        # w 0
        ep = torch_export(
            model,
            (torch.empty((0, 3), dtype=torch.float32),),
            dynamic_shapes=ds,
            backed_size_oblivious="auto",
        )
        y = ep.module()(x)
        self.assertEqualArray(expected, y)


if __name__ == "__main__":
    unittest.main(verbosity=2)
