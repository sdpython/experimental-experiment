import copy
import unittest
from onnx_diagnostic.helpers import max_diff, string_type
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
    skipif_ci_windows,
    requires_torch,
    requires_transformers,
    long_test,
)
from experimental_experiment.torch_interpreter import to_onnx


class TestLlmModelHelperSerialization(ExtTestCase):
    @ignore_warnings("TracerWarning")
    @ignore_warnings(UserWarning)
    @skipif_ci_windows("not supported")
    @requires_torch("2.9")  # for torch.export.Dim.DYNAMIC
    @requires_transformers("4.49.9999")
    def test_phi2_output_order_export(self):
        import torch
        from onnx_diagnostic.torch_export_patches import torch_export_patches
        from experimental_experiment.torch_models.llm_model_helper import get_phi2

        res = get_phi2(
            num_hidden_layers=2,
            input_cache=True,
            common_dynamic_shapes=True,
            intermediate_size=5120,
            # hidden_size=1280,
            batch_size=2,
        )
        model, model_inputs, dyn_shapes = (res["model"], res["inputs"], res["dynamic_shapes"])
        expected = model(**copy.deepcopy(model_inputs))
        with torch_export_patches(patch_transformers=True) as modificator:
            modified_inputs = modificator(copy.deepcopy(model_inputs))
            ep = torch.export.export(
                model,
                (),
                kwargs=modified_inputs,
                dynamic_shapes=dyn_shapes,
                strict=False,
            )
            mod = ep.module()
            got = mod(**copy.deepcopy(model_inputs))

            # We check that should be the same order.
            self.assertNotIn("patched_DynamicCache", string_type(expected, with_shape=True))
            self.assertIn("patched_DynamicCache", string_type(got, with_shape=True))
            self.assertEqualAny(expected, got)
            flatten_got = torch.utils._pytree.tree_flatten(got)[0]

        flatten_expected = torch.utils._pytree.tree_flatten(expected)[0]
        self.assertEqualAny(flatten_expected, flatten_got)
        diff = max_diff(expected, got)
        self.assertLess(diff["abs"], 1e-5)
        diff = max_diff(flatten_expected, flatten_got)
        self.assertLess(diff["abs"], 1e-5)

    @ignore_warnings("TracerWarning")
    @ignore_warnings((UserWarning, DeprecationWarning))
    @skipif_ci_windows("not supported")
    @requires_torch("2.6")  # for torch.export.Dim.DYNAMIC
    @long_test()
    def test_phi2_output_order_onnx_dynamo(self):
        import torch
        from onnx_diagnostic.torch_export_patches import torch_export_patches
        from experimental_experiment.torch_models.llm_model_helper import get_phi2

        res = get_phi2(
            num_hidden_layers=2,
            input_cache=True,
            common_dynamic_shapes=True,
            intermediate_size=5120,
            # hidden_size=1280,
            batch_size=2,
        )
        model, model_inputs, dyn_shapes = (res["model"], res["inputs"], res["dynamic_shapes"])
        expected = model(**copy.deepcopy(model_inputs))
        with torch_export_patches(patch_transformers=True) as modificator:
            modified_inputs = modificator(model_inputs)
            ep = torch.onnx.export(
                model,
                (),
                kwargs=modified_inputs,
                dynamic_shapes=dyn_shapes,
                dynamo=True,
            )
            flatten_inputs = torch.utils._pytree.tree_flatten(model_inputs)[0]
            flatten_expected = torch.utils._pytree.tree_flatten(expected)[0]

        from onnxruntime import InferenceSession

        sess = InferenceSession(
            ep.model_proto.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        feeds = dict(
            zip(
                [i.name for i in sess.get_inputs()],
                [t.detach().cpu().numpy() for t in flatten_inputs],
            )
        )
        got = sess.run(None, feeds)
        self.assertEqualAny([_.detach().numpy() for _ in flatten_expected], got, atol=1e-5)
        diff = max_diff(flatten_expected, got)
        self.assertLess(diff["abs"], 1e-5)
        diff = max_diff(expected, got)
        self.assertLess(diff["abs"], 1e-5)

    @ignore_warnings("TracerWarning")
    @ignore_warnings(UserWarning)
    @skipif_ci_windows("not supported")
    @requires_torch("2.6")  # for torch.export.Dim.DYNAMIC
    @long_test()
    def test_phi2_output_order_custom(self):
        import torch
        from onnx_diagnostic.torch_export_patches import torch_export_patches
        from experimental_experiment.torch_models.llm_model_helper import get_phi2

        res = get_phi2(
            num_hidden_layers=2,
            input_cache=True,
            common_dynamic_shapes=True,
            intermediate_size=5120,
            # hidden_size=1280,
            batch_size=2,
        )
        model, model_inputs, dyn_shapes = (res["model"], res["inputs"], res["dynamic_shapes"])
        expected = model(**copy.deepcopy(model_inputs))
        with torch_export_patches(patch_transformers=True) as modificator:
            flatten_inputs = torch.utils._pytree.tree_flatten(model_inputs)[0]
            modified_inputs = modificator(model_inputs)
            onx = to_onnx(
                model,
                (),
                kwargs=modified_inputs,
                dynamic_shapes=dyn_shapes,
            )
            flatten_expected = torch.utils._pytree.tree_flatten(expected)[0]

        from onnxruntime import InferenceSession

        sess = InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
        feeds = dict(
            zip(
                [i.name for i in sess.get_inputs()],
                [t.detach().cpu().numpy() for t in flatten_inputs],
            )
        )
        got = sess.run(None, feeds)
        self.assertEqualAny([_.detach().numpy() for _ in flatten_expected], got, atol=1e-5)
        diff = max_diff(flatten_expected, got)
        self.assertLess(diff["abs"], 1e-5)
        diff = max_diff(expected, got)
        self.assertLess(diff["abs"], 1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
