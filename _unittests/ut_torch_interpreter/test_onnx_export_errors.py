import unittest
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    requires_transformers,
    skipif_ci_windows,
)
from experimental_experiment.torch_interpreter.onnx_export_errors import (
    bypass_export_some_errors,
)


class TestOnnxExportErrors(ExtTestCase):
    @requires_transformers("4.43")
    @skipif_ci_windows("not working on Windows")
    def test_pytree_flatten(self):
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

        cache = MambaCache(_config(), batch_size=1)

        with bypass_export_some_errors():
            values, spec = py_pytree.tree_flatten(cache)
            cache2 = py_pytree.tree_unflatten(values, spec)
            self.assertEqual(cache.dtype, cache2.dtype)
            self.assertEqual(cache.batch_size, cache2.batch_size)
            self.assertEqual(cache.intermediate_size, cache2.intermediate_size)
            self.assertEqual(cache.ssm_state_size, cache2.ssm_state_size)
            self.assertEqual(cache.conv_kernel_size, cache2.conv_kernel_size)
            self.assertEqualArray(cache.conv_states, cache2.conv_states)
            self.assertEqualArray(cache.ssm_states, cache2.ssm_states)

    @requires_transformers("4.43")
    @skipif_ci_windows("not working on Windows")
    def test_exportable_mamba_cache(self):
        import torch
        from transformers.models.mamba.modeling_mamba import MambaCache

        class _config:
            def __init__(self):
                self.intermediate_size = 8
                self.state_size = 16
                self.conv_kernel = 32
                self.num_hidden_layers = 64
                self.dtype = torch.float16

        class Model(torch.nn.Module):
            def forward(self, x, cache: MambaCache):
                return cache.conv_states + x

        with bypass_export_some_errors():
            cache = MambaCache(_config(), batch_size=1)
            torch.export.export(Model(), (torch.ones(16, 16), cache))


if __name__ == "__main__":
    unittest.main(verbosity=2)
