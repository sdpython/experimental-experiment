import unittest
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    requires_transformers,
    skipif_ci_windows,
)
from onnx_diagnostic.torch_export_patches import bypass_export_some_errors


class TestOnnxExportErrors(ExtTestCase):
    @requires_transformers("4.49.999")
    @skipif_ci_windows("not working on Windows")
    def test_pytree_flatten_mamba_cache(self):
        import torch
        import torch.utils._pytree as py_pytree
        from transformers.cache_utils import MambaCache

        class _config:
            def __init__(self):
                self.intermediate_size = 8
                self.state_size = 16
                self.conv_kernel = 32
                self.num_hidden_layers = 64
                self.dtype = torch.float16

        cache = MambaCache(_config(), max_batch_size=1, device="cpu")

        with bypass_export_some_errors():
            values, spec = py_pytree.tree_flatten(cache)
            cache2 = py_pytree.tree_unflatten(values, spec)
            self.assertEqual(cache.max_batch_size, cache2.max_batch_size)
            self.assertEqual(cache.intermediate_size, cache2.intermediate_size)
            self.assertEqual(cache.ssm_state_size, cache2.ssm_state_size)
            self.assertEqual(cache.conv_kernel_size, cache2.conv_kernel_size)
            self.assertEqualArrayAny(cache.conv_states, cache2.conv_states)
            self.assertEqualArrayAny(cache.ssm_states, cache2.ssm_states)
            self.assertEqual(cache.ssm_states[0].dtype, cache2.ssm_states[0].dtype)

    def test_exportable_dynamic_shapes_constraints(self):
        import torch

        class CustomCache:
            def __init__(self, shape=None):
                self.cache = [torch.zeros((shape)), torch.zeros((shape))] if shape else []

        def flatten_cache(cache):
            return [cache.cache], ["cache"]

        def unflatten_cache(values, context, output_type=None):
            cache = CustomCache()
            cache.cache = values[0]
            return cache

        def flatten_with_keys_cache(d):
            values, context = flatten_cache(d)
            return [
                (torch.utils._pytree.MappingKey(k), v) for k, v in zip(context, values)
            ], context

        torch.utils._pytree.register_pytree_node(
            CustomCache,
            flatten_cache,
            unflatten_cache,
            serialized_type_name=f"{CustomCache.__module__}.{CustomCache.__name__}",
            flatten_with_keys_fn=flatten_with_keys_cache,
        )

        class Model(torch.nn.Module):
            def forward(self, x, cache):
                return cache.cache[0][0, :] + x

        model = Model()
        model.eval()
        x, cache = torch.rand((2, 4)), CustomCache((2, 4))
        model(x, cache)
        DYN = torch.export.Dim.DYNAMIC
        torch.export.export(
            model, (x, cache), dynamic_shapes=({0: DYN}, [[{0: DYN}, {0: DYN}]])
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
