import unittest
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
    requires_onnx_diagnostic,
)
from experimental_experiment.reference import ExtendedReferenceEvaluator
from experimental_experiment.torch_interpreter import to_onnx, ExportOptions


class TestOnnxExportTracing(ExtTestCase):
    def test_export_with_option_tracing(self):
        import torch

        class Neuron(torch.nn.Module):
            def __init__(self, n_dims: int, n_targets: int):
                super(Neuron, self).__init__()
                self.linear = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x):
                return torch.relu(self.linear(x))

        model = Neuron(5, 3)
        x = torch.rand(2, 5)
        expected = model(x)
        onx = to_onnx(
            model,
            (x,),
            dynamic_shapes=({0: "batch"},),
            export_options=ExportOptions(tracing=True),
            verbose=0,
        )
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, {"x": x.numpy()})
        self.assertEqualArray(expected, got[0], atol=1e-5)

    def test_export_with_option_tracing_shape(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.ones(x.shape)

        model = Model()
        x = torch.rand(2, 5)
        expected = model(x)
        onx = to_onnx(
            model,
            (x,),
            dynamic_shapes=({0: "batch"},),
            export_options=ExportOptions(tracing=True),
            verbose=0,
        )
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, {"x": x.numpy()})
        self.assertEqualArray(expected, got[0], atol=1e-5)

    def test_export_with_option_tracing_dtype(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.ones(x.shape, dtype=x.dtype)

        model = Model()
        x = torch.rand(2, 5)
        expected = model(x)
        onx = to_onnx(
            model,
            (x,),
            dynamic_shapes=({0: "batch"},),
            export_options=ExportOptions(tracing=True),
            verbose=0,
        )
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, {"x": x.numpy()})
        self.assertEqualArray(expected, got[0], atol=1e-5)

    def test_export_with_option_tracing_device(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.ones(x.shape, device=x.device)

        model = Model()
        x = torch.rand(2, 5)
        expected = model(x)
        onx = to_onnx(
            model,
            (x,),
            dynamic_shapes=({0: "batch"},),
            export_options=ExportOptions(tracing=True),
            verbose=0,
        )
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, {"x": x.numpy()})
        self.assertEqualArray(expected, got[0], atol=1e-5)

    def test_export_with_option_tracing_2(self):
        import torch

        class Neuron(torch.nn.Module):
            def __init__(self, n_dims: int, n_targets: int):
                super(Neuron, self).__init__()
                self.linear = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x, y):
                return torch.relu(self.linear(x)) + torch.relu(self.linear(y))

        model = Neuron(5, 3)
        x, y = torch.rand(2, 5), torch.rand(2, 5)
        expected = model(x, y)
        onx = to_onnx(
            model,
            (x, y),
            dynamic_shapes=({0: "batch"}, {0: "batch"}),
            export_options=ExportOptions(tracing=True),
            verbose=0,
        )
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, {"x": x.numpy(), "y": y.numpy()})
        self.assertEqualArray(expected, got[0], atol=1e-5)

    @requires_onnx_diagnostic("0.8.8")
    @ignore_warnings(UserWarning)
    def test_tracing_dynamic_cache1(self):
        import torch
        from transformers.cache_utils import DynamicCache
        from onnx_diagnostic.helpers import flatten_object
        from onnx_diagnostic.helpers.cache_helper import CacheKeyValue, make_dynamic_cache
        from onnx_diagnostic.torch_export_patches import torch_export_patches

        a_cache = make_dynamic_cache([(torch.ones((5, 6, 5, 6)), torch.ones((5, 6, 5, 6)) + 2)])
        cls_cache = a_cache.__class__

        class SubModelCache(torch.nn.Module):
            def forward(self, cache):
                d = cls_cache()
                dc = CacheKeyValue(cache)
                d.update(dc.key_cache[0] + 1, dc.value_cache[0] + 2, 0)
                return d

        class SubModel(torch.nn.Module):
            def forward(self, x, cache):
                dc = CacheKeyValue(cache)
                return x + dc.key_cache[0] + dc.value_cache[0]

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sub = SubModel()
                self.subcache = SubModelCache()

            def forward(self, x, cache: DynamicCache):
                return self.sub(x, self.subcache(cache))

        model = Model()
        inputs = torch.rand((5, 6, 5, 6)), make_dynamic_cache(
            [(torch.ones((5, 6, 5, 6)), torch.ones((5, 6, 5, 6)) + 2)]
        )
        expected = model(*inputs)
        with torch_export_patches(patch_transformers=True):
            onx = to_onnx(
                model,
                inputs,
                dynamic_shapes=(
                    ({0: "batch", 2: "DIM"}, [{0: "batch", 2: "DIM"}, {0: "batch", 2: "DIM"}])
                ),
                export_options=ExportOptions(tracing=True),
                verbose=10,
            )
        feeds = dict(
            zip(
                [i.name for i in onx.graph.input],
                [t.numpy() for t in flatten_object(inputs, drop_keys=True)],
            )
        )
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0], atol=1e-5)

    @requires_onnx_diagnostic("0.8.8")
    @ignore_warnings(UserWarning)
    def test_tracing_dynamic_cache0(self):
        import torch
        from onnx_diagnostic.helpers import flatten_object
        from onnx_diagnostic.helpers.cache_helper import CacheKeyValue, make_dynamic_cache
        from onnx_diagnostic.torch_export_patches import torch_export_patches

        a_cache = make_dynamic_cache([(torch.ones((5, 6, 5, 6)), torch.ones((5, 6, 5, 6)) + 2)])
        cls_cache = a_cache.__class__

        class SubModelCache(torch.nn.Module):
            def forward(self, cache):
                d = cls_cache()
                dc = CacheKeyValue(cache)
                d.update(dc.key_cache[0] + 1, dc.value_cache[0] + 2, 0)
                return d

        class SubModel(torch.nn.Module):
            def forward(self, x, cache):
                dc = CacheKeyValue(cache)
                return x + dc.key_cache[0] + dc.value_cache[0]

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sub = SubModel()
                self.subcache = SubModelCache()

            def forward(self, x, keys, values):
                cache = make_dynamic_cache([(keys, values)])
                return self.sub(x, self.subcache(cache))

        model = Model()
        inputs = torch.rand((5, 6, 5, 6)), torch.ones((5, 6, 5, 6)), torch.ones((5, 6, 5, 6)) + 2
        expected = model(*inputs)
        with torch_export_patches(patch_transformers=True):
            onx = to_onnx(
                model,
                inputs,
                dynamic_shapes=(
                    {0: "batch", 2: "DIM"},
                    {0: "batch", 2: "DIM"},
                    {0: "batch", 2: "DIM"},
                ),
                export_options=ExportOptions(tracing=True),
                verbose=0,
            )
        feeds = dict(
            zip(
                [i.name for i in onx.graph.input],
                [t.numpy() for t in flatten_object(inputs, drop_keys=True)],
            )
        )
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0], atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
