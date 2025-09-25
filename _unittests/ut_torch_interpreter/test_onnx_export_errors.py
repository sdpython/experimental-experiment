import unittest
from experimental_experiment.ext_test_case import ExtTestCase


class TestOnnxExportErrors(ExtTestCase):
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
        torch.export.export(model, (x, cache), dynamic_shapes=({0: DYN}, [[{0: DYN}, {0: DYN}]]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
