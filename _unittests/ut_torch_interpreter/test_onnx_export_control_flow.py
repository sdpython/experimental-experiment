import unittest
import warnings
from collections import Counter
import onnx
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    requires_torch,
    skipif_ci_windows,
    hide_stdout,
)
from experimental_experiment.helpers import string_type
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.torch_interpreter import to_onnx, ExportOptions


class TestOnnxExportControlFlow(ExtTestCase):

    @classmethod
    def get_custom_model(cls):
        import torch

        class Bad1Fixed(torch.nn.Module):
            def forward(self, x):
                def true_fn(x):
                    return torch.sin(x)

                def false_fn(x):
                    return torch.cos(x)

                return torch.cond(x.sum() > 0, true_fn, false_fn, [x])

        return Bad1Fixed, torch.rand(5, 3)

    @skipif_ci_windows("not yet supported on Windows")
    @requires_torch("2.9")
    def test_controlflow_dynamo(self):
        import torch

        cls, x = self.get_custom_model()
        model = cls()
        filename = "test_controlflow_dynamo.onnx"
        torch.onnx.export(
            model,
            (x,),
            filename,
            input_names=["x"],
            opset_version=18,
            dynamo=True,
            fallback=False,
        )
        with open(filename, "rb") as f:
            onx = onnx.load(f)
        ref = ExtendedReferenceEvaluator(onx)

        for _x in (x, -x):
            expected = model(_x)
            got = ref.run(None, {"x": _x.detach().numpy()})[0]
            self.assertEqualArray(expected, got, atol=1e-6)

    @skipif_ci_windows("not yet supported on Windows")
    @requires_torch("2.4")
    def test_controlflow_custom_if_1(self):
        import onnxruntime

        cls, x = self.get_custom_model()
        model = cls()
        onx = to_onnx(model, (x,), inline=False)
        co = Counter([n.op_type for n in onx.graph.node])
        self.assertEqual(co, {"ReduceSum": 1, "Greater": 1, "If": 1})
        self.assertEqual(len(onx.functions), 2)
        ref = ExtendedReferenceEvaluator(onx)
        sess = onnxruntime.InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )

        for _x in (x, -x):
            expected = model(_x)
            got = ref.run(None, {"x": _x.detach().numpy()})[0]
            self.assertEqualArray(expected, got, atol=1e-5)

            got = sess.run(None, {"x": _x.detach().numpy()})[0]
            self.assertEqualArray(expected, got, atol=1e-5)

    @classmethod
    def get_custom_model_2(cls):
        import torch

        class Bad2Fixed(torch.nn.Module):
            def forward(self, x):
                def true_fn(x):
                    return torch.sin(x), torch.cos(x)

                def false_fn(x):
                    return torch.cos(x), torch.sin(x)

                return torch.cond(x.sum() > 0, true_fn, false_fn, [x])

        return Bad2Fixed, torch.rand(5, 3)

    @skipif_ci_windows("not yet supported on Windows")
    @requires_torch("2.4")
    def test_controlflow_custom_if_2(self):
        cls, x = self.get_custom_model_2()
        model = cls()
        onx = to_onnx(model, (x,), inline=False)
        co = Counter([n.op_type for n in onx.graph.node])
        self.assertEqual(co, {"ReduceSum": 1, "Greater": 1, "If": 1})
        self.assertEqual(len(onx.functions), 2)
        ref = ExtendedReferenceEvaluator(onx)

        for _x in (x, -x):
            expected = model(_x)
            got = ref.run(None, {"x": _x.detach().numpy()})
            self.assertEqual(len(expected), len(got))
            for e, g in zip(expected, got):
                self.assertEqualArray(e, g, atol=1e-5)

    @skipif_ci_windows("not yet supported on Windows")
    @requires_torch("2.4")
    @hide_stdout()
    def test_controlflow_custom_if_inline(self):
        cls, x = self.get_custom_model()
        model = cls()
        onx = to_onnx(model, (x,), inline=True, verbose=2)
        self.assertEqual(len(onx.functions), 0)
        co = Counter([n.op_type for n in onx.graph.node])
        self.assertEqual(co, {"ReduceSum": 1, "Greater": 1, "If": 1})
        self.assertEqual(len(onx.functions), 0)
        ref = ExtendedReferenceEvaluator(onx)

        for _x in (x, -x):
            expected = model(_x)
            got = ref.run(None, {"x": _x.detach().numpy()})[0]
            self.assertEqualArray(expected, got, atol=1e-5)

    @skipif_ci_windows("not yet supported on Windows")
    @requires_torch("2.4")
    def test_controlflow_custom_if_two_inputs(self):
        import torch

        class TwoInputs(torch.nn.Module):
            def forward(self, x, y):
                def true_fn(x, y):
                    return torch.sin(x), torch.cos(x) + y

                def false_fn(x, y):
                    return torch.cos(x), torch.sin(x) + y

                return torch.cond(x.sum() > 0, true_fn, false_fn, [x, y])

        x, y = torch.rand(5, 3), torch.rand(5, 3)
        model = TwoInputs()
        onx = to_onnx(model, (x, y), inline=False)
        co = Counter([n.op_type for n in onx.graph.node])
        self.assertEqual(co, {"ReduceSum": 1, "Greater": 1, "If": 1})
        self.assertEqual(len(onx.functions), 2)
        ref = ExtendedReferenceEvaluator(onx)

        for _x in (x, -x):
            expected = model(_x, y)
            got = ref.run(None, {"x": _x.detach().numpy(), "y": y.detach().numpy()})
            self.assertEqual(len(expected), len(got))
            for e, g in zip(expected, got):
                self.assertEqualArray(e, g, atol=1e-5)

    @skipif_ci_windows("not yet supported on Windows")
    @requires_torch("2.4")
    def test_controlflow_custom_if_two_cond(self):
        import torch

        class TwoConds(torch.nn.Module):
            def forward(self, x):
                def true_fn2(x):
                    def true_fn1(x):
                        return torch.sin(x)

                    def false_fn1(x):
                        return torch.cos(x)

                    return torch.cond(x.sum() < 0, true_fn1, false_fn1, [x])

                def false_fn2(x):
                    return -x

                return torch.cond(x.sum() > 0, true_fn2, false_fn2, [x])

        x = torch.rand(5, 3)
        model = TwoConds()
        model(x)
        model(-x)
        torch.export.export(model, (x,))
        onx = to_onnx(model, (x,), inline=False)
        co = Counter([n.op_type for n in onx.graph.node])
        self.assertEqual(co, {"ReduceSum": 1, "Greater": 1, "If": 1})
        self.assertEqual(len(onx.functions), 4)
        ref = ExtendedReferenceEvaluator(onx)

        for _x in (x, -x):
            expected = model(_x)
            got = ref.run(None, {"x": _x.detach().numpy()})
            self.assertEqualArray(expected, got[0], atol=1e-5)

    @skipif_ci_windows("not yet supported on Windows")
    @requires_torch("2.4")
    def test_controlflow_custom_raw_test(self):
        import torch

        class RawTest(torch.nn.Module):
            def forward(self, x):
                def true_fn(x):
                    return torch.sin(x)

                def false_fn(x):
                    return torch.cos(x)

                if x.sum() > 0:
                    return true_fn(x)
                return false_fn(x)

        x = torch.rand(5, 3)
        model = RawTest()
        filename = "test_controlflow_custom_raw_test.onnx"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(model, (x,), filename, input_names=["x"], opset_version=18)
            onx = to_onnx(model, (x,), export_options=ExportOptions(jit=True))
        co = Counter([n.op_type for n in onx.graph.node])
        self.assertEqual(co, {"Sin": 1})
        self.assertEqual(len(onx.functions), 0)
        ref = ExtendedReferenceEvaluator(onx)

        for _x in (x,):
            expected = model(_x)
            got = ref.run(None, {"x": _x.detach().numpy()})
            self.assertEqualArray(expected, got[0], atol=1e-5)

    @skipif_ci_windows("not yet supported on Windows")
    @requires_torch("2.4")
    def test_controlflow_custom_fallback(self):
        import torch

        class RawTest(torch.nn.Module):
            def forward(self, x):
                def true_fn(x):
                    return torch.sin(x)

                def false_fn(x):
                    return torch.cos(x)

                if x.sum() > 0:
                    return true_fn(x)
                return false_fn(x)

        x = torch.rand(5, 3)
        model = RawTest()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            onx = to_onnx(model, (x,), export_options=ExportOptions(strategy="fallback"))
        co = Counter([n.op_type for n in onx.graph.node])
        self.assertEqual(co, {"Sin": 1})
        self.assertEqual(len(onx.functions), 0)
        ref = ExtendedReferenceEvaluator(onx)

        for _x in (x,):
            expected = model(_x)
            got = ref.run(None, {"x": _x.detach().numpy()})
            self.assertEqualArray(expected, got[0], atol=1e-5)

    @skipif_ci_windows("not yet supported on Windows")
    @requires_torch("2.4")
    def test_controlflow_custom_initializer(self):
        import torch

        class RawTest(torch.nn.Module):
            def forward(self, x):
                def true_fn(x):
                    return torch.sin(x) - torch.ones(x.shape, dtype=x.dtype)

                def false_fn(x):
                    return torch.cos(x) + torch.ones((1, 1024), dtype=x.dtype)

                return torch.cond(x.sum() > 0, true_fn, false_fn, [x])

        x = torch.rand(1024, 1024)
        model = RawTest()

        # not inlined
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            onx = to_onnx(model, (x,), inline=False)
        co = Counter([n.op_type for n in onx.graph.node])
        self.assertEqual(co, {"ReduceSum": 1, "Greater": 1, "If": 1})
        self.assertEqual(len(onx.functions), 2)
        ref = ExtendedReferenceEvaluator(onx)

        for _x in (x, -x):
            expected = model(_x)
            got = ref.run(None, {"x": _x.detach().numpy()})
            self.assertEqualArray(expected, got[0], atol=1e-5)

        # inlined
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            onx = to_onnx(model, (x,), inline=True)
        co = Counter([n.op_type for n in onx.graph.node])
        self.assertEqual(co, {"ReduceSum": 1, "Greater": 1, "If": 1})
        self.assertEqual(len(onx.functions), 0)
        ref = ExtendedReferenceEvaluator(onx)

        for _x in (x, -x):
            expected = model(_x)
            got = ref.run(None, {"x": _x.detach().numpy()})
            self.assertEqualArray(expected, got[0], atol=1e-5)

    def test_nested_cond(self):
        import onnxruntime
        import torch

        class Submodule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Nested weight
                self.weight = torch.nn.Parameter(torch.tensor([100.0]))

            def forward(self, x):
                def true_fn(x):
                    return x * self.weight

                def false_fn(x):
                    return x / self.weight

                y = torch.cond(torch.abs(x).sum() > 100, true_fn, false_fn, [x])
                return y

        class CondModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.submodule = Submodule()
                self.weight = torch.nn.Parameter(torch.tensor([42.0]))

            def forward(self, x):
                def true_fn(x):
                    return self.submodule(x)

                def false_fn(x):
                    return x - self.weight

                y = torch.cond(x.sum() > 0, true_fn, false_fn, [x])
                return y

        x = torch.tensor([-1, 2])
        model = CondModel()
        onx = to_onnx(model, (x,), inline=False)
        names = [(f.domain, f.name) for f in onx.functions]
        self.assertEqual(len(names), len(set(names)))
        ref = ExtendedReferenceEvaluator(onx)
        sess = onnxruntime.InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )

        for _x in (x, -x, -x * 1000, x * 1000):
            expected = model(_x)
            got = ref.run(None, {"x": _x.detach().numpy()})
            self.assertEqualArray(expected, got[0], atol=1e-5)
            got = sess.run(None, {"x": _x.detach().numpy()})
            self.assertEqualArray(expected, got[0], atol=1e-5)

    def test_cond_llm_image_embedding(self):
        import torch

        class Model(torch.nn.Module):
            def forward(
                self,
                input_ids,
                image_features,
                vocab_size,
            ):
                if image_features.numel():
                    input_shape = input_ids.size()
                    input_ids = input_ids.view(-1, input_shape[-1])

                    # positions for image tokens
                    condition = (input_ids < 0) & (input_ids > -int(1e9))
                    positions = torch.where(condition)
                    # has_image = len(positions[0].tolist()) > 0
                    input_ids = input_ids.clamp_min(0).clamp_max(vocab_size).detach()

                    return (input_ids, *positions)

                return (input_ids, *torch.where(torch.zeros((1, 1), dtype=torch.bool)))

        inputs = [
            (
                (torch.arange(24) - 8).reshape((2, -1)).to(torch.int64),
                torch.arange(32).reshape((2, -1)).to(torch.float32),
                torch.tensor(1025, dtype=torch.int64),
            ),
            (
                (torch.arange(24) - 8).reshape((2, -1)).to(torch.int64),
                torch.tensor([[], []], dtype=torch.float32),
                torch.tensor(1025, dtype=torch.int64),
            ),
        ]
        model = Model()
        expected = [model(*inp) for inp in inputs]

        self.assertEqual(
            string_type(expected, with_shape=True),
            "#2[(T7s2x12,T7s8,T7s8),(T7s2x12,T7s0,T7s0)]",
        )

        class Model2(torch.nn.Module):
            def forward(self, input_ids, image_features, vocab_size):
                def then_branch(input_ids, image_features, vocab_size):
                    input_shape = input_ids.size()
                    input_ids = input_ids.view(-1, input_shape[-1])

                    # positions for image tokens
                    condition = (input_ids < 0) & (input_ids > -int(1e9))
                    positions = torch.nonzero(condition, as_tuple=True)
                    input_ids = input_ids.clamp_min(0).clamp_max(vocab_size).detach()
                    return (input_ids, positions[0], positions[1])

                def else_branch(input_ids, image_features, vocab_size):
                    r = torch.where(torch.zeros((1, 1), dtype=torch.bool))
                    return (input_ids, r[0], r[1])

                a, b, c = torch.cond(
                    image_features.numel() > 0,
                    then_branch,
                    else_branch,
                    [input_ids, image_features, vocab_size],
                )
                return a, b, c

        model2 = Model2()
        new_out = [model2(*inp) for inp in inputs]
        self.assertEqualAny(expected, new_out)

        batch = torch.export.Dim("batch")
        seq_length = torch.export.Dim("seq_length")
        dynamic_shapes = ({0: batch}, {0: batch, 1: seq_length}, None)

        # print(
        #     torch.export.export(model2, inputs[0],
        #       dynamic_shapes=dynamic_shapes, strict=False)
        # )
        # torch.onnx.export(model2, (*inputs[0][:2], 1025),
        #   "test_cond_llm_image_embedding_dynamo.onnx",
        #   dynamic_shapes=dynamic_shapes, dynamo=True)
        # torch.onnx.export(model2, inputs[0], "test_cond_llm_image_embedding_dynamo.onnx",
        #   dynamic_shapes=dynamic_shapes, dynamo=True)

        from experimental_experiment.torch_interpreter.tracing import CustomTracer

        graph = CustomTracer().trace(
            model2,
            concrete_args=dict(zip(["input_ids", "image_features", "vocab_size"], inputs[0])),
        )
        self.assertNotEmpty(graph)

        onx = to_onnx(
            model2,
            inputs[0],
            dynamic_shapes=dynamic_shapes,
            export_options=ExportOptions(tracing=True, allow_untyped_output=True),
            inline=False,
        )
        with open("test_cond_llm_image_embedding_tracing.onnx", "wb") as f:
            f.write(onx.SerializeToString())

        # still does not work
        # onx = to_onnx(
        #     model2,
        #     inputs[0],
        #     dynamic_shapes=dynamic_shapes,
        #     export_options=ExportOptions(strict=False),
        # )
        # with open("test_cond_llm_image_embedding.onnx", "wb") as f:
        #     f.write(onx.SerializeToString())
        self.assertIn("If", {n.op_type for n in onx.graph.node})

        sess = ExtendedReferenceEvaluator(onx)
        for exp, inp in zip(expected, inputs):
            with self.subTest(input2_shape=inp[1].shape):
                feeds = dict(
                    zip(
                        [_.name for _ in onx.graph.input],
                        [_.detach().cpu().numpy() for _ in inp],
                    )
                )
                got = sess.run(None, feeds)
                self.assertEqual(len(got), 3)
                self.assertEqual(len(exp), 3)
                self.assertEqualArray(exp[2], got[2])
                self.assertEqualArray(exp[1], got[1])
                self.assertEqualArray(exp[0], got[0])

        # same with onnxruntime
        import onnxruntime

        sess = onnxruntime.InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        for exp, inp in zip(expected, inputs):
            with self.subTest(input2_shape=inp[1].shape):
                feeds = dict(
                    zip(
                        [_.name for _ in onx.graph.input],
                        [_.detach().cpu().numpy() for _ in inp],
                    )
                )
                got = sess.run(None, feeds)
                self.assertEqual(len(got), 3)
                self.assertEqual(len(exp), 3)
                self.assertEqualArray(exp[2], got[2])
                self.assertEqualArray(exp[1], got[1])
                self.assertEqualArray(exp[0], got[0])

    @requires_torch("2.7", "export of torch.cond")
    def test_controlflow_cond_submodule_1(self):
        import torch

        class SubThen(torch.nn.Module):
            def forward(self, x):
                return x * x

        class SubElse(torch.nn.Module):
            def forward(self, x):
                return torch.abs(x)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sub_then = SubThen()
                self.sub_else = SubElse()

            def forward(self, x):
                return torch.cond(x.sum() > 0, self.sub_then, self.sub_else, [x])

        model = Model()
        x = torch.rand((5, 4))
        model(x)
        torch.export.export(
            model,
            (x,),
            dynamic_shapes=({0: torch.export.Dim.DYNAMIC, 1: torch.export.Dim.DYNAMIC},),
        )

    @requires_torch("2.7", "export of torch.cond")
    def test_controlflow_cond_submodule_args(self):
        import torch

        class SubThen(torch.nn.Module):
            def forward(self, *args):
                x = args[0]
                return x * x

        class SubElse(torch.nn.Module):
            def forward(self, x):
                return torch.abs(x)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sub_then = SubThen()
                self.sub_else = SubElse()

            def forward(self, x):
                return torch.cond(x.sum() > 0, self.sub_then, self.sub_else, [x])

        model = Model()
        x = torch.rand((5, 4))
        model(x)
        torch.export.export(
            model,
            (x,),
            dynamic_shapes=({0: torch.export.Dim.DYNAMIC, 1: torch.export.Dim.DYNAMIC},),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
