import unittest
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    requires_torch,
    requires_transformers,
    skipif_ci_windows,
)
from onnx_diagnostic.torch_export_patches import bypass_export_some_errors
from experimental_experiment.helpers import string_type


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

    @requires_transformers("4.43")
    @requires_torch("2.7")
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
            def forward(self, x: torch.Tensor, cache: MambaCache):
                x1 = cache.ssm_states[0] + x
                x2 = cache.conv_states[0][:, :, ::2] + x1
                return x2

        cache = MambaCache(_config(), max_batch_size=1, device="cpu")
        self.assertEqual(
            string_type(cache), "MambaCache(conv_states=[T10r3,...], ssm_states=[T10r3,...])"
        )
        x = torch.ones(2, 8, 16).to(torch.float16)
        model = Model()
        model(x, cache)

        with bypass_export_some_errors():
            cache = MambaCache(_config(), max_batch_size=1, device="cpu")
            torch.export.export(Model(), (x, cache))

    @requires_transformers("4.49.999")
    @skipif_ci_windows("not working on Windows")
    def test_exportable_mamba_cache_dynamic(self):
        import torch
        from transformers.models.mamba.modeling_mamba import MambaCache

        class _config:
            def __init__(self):
                self.intermediate_size = 8
                self.state_size = 16
                self.conv_kernel = 32
                self.num_hidden_layers = 2
                self.dtype = torch.float16

        class Model(torch.nn.Module):
            def forward(self, x: torch.Tensor, cache: MambaCache):
                x1 = cache.ssm_states[0] + x
                x2 = cache.conv_states[0][:, :, ::2] + x1
                return x2

        cache = MambaCache(_config(), max_batch_size=1, device="cpu")
        self.assertEqual(
            string_type(cache),
            "MambaCache(conv_states=#2[T10r3,T10r3], ssm_states=#2[T10r3,T10r3])",
        )
        x = torch.ones(2, 8, 16).to(torch.float16)
        model = Model()
        model(x, cache)
        DYN = torch.export.Dim.DYNAMIC

        with bypass_export_some_errors():
            cache = MambaCache(_config(), max_batch_size=1, device="cpu")
            torch.export.export(
                Model(),
                (x, cache),
                dynamic_shapes=({0: DYN}, [[{0: DYN}, {0: DYN}], [{0: DYN}, {0: DYN}]]),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
