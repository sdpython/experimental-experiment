import unittest
import torch
from onnx_diagnostic.helpers import max_diff
from onnx_diagnostic.helpers.torch_helper import torch_deepcopy
from onnx_diagnostic.helpers.rt_helper import make_feeds
from onnx_diagnostic.helpers.cache_helper import make_dynamic_cache
from onnx_diagnostic.torch_models.hghub import get_untrained_model_with_inputs
from onnx_diagnostic.torch_export_patches import torch_export_patches
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    hide_stdout,
    requires_onnx_diagnostic,
)
from experimental_experiment.torch_interpreter import to_onnx
from experimental_experiment.xbuilder import OptimizationOptions


class TestOptimizationUntrainedModel(ExtTestCase):
    @hide_stdout()
    @requires_onnx_diagnostic("0.7.17")
    def test_tiny_llm_to_onnx(self):
        import onnxruntime

        data = get_untrained_model_with_inputs("arnir0/Tiny-LLM")
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        b1 = data["inputs_batch1"]
        filename = self.get_dump_file("test_tiny_llm_to_onnx-custom.onnx")
        del inputs["position_ids"]
        del ds["position_ids"]
        del b1["position_ids"]

        expected = model(**torch_deepcopy(b1))

        with torch_export_patches(patch_transformers=True):
            onx = to_onnx(
                model,
                kwargs=inputs,
                dynamic_shapes=ds,
                filename=filename,
                verbose=1,
                large_model=True,
                options=OptimizationOptions(patterns="default+onnxruntime"),
            )

        outputs = [o.name for o in onx.model_proto.graph.output]
        self.assertEqual(
            ["output_0", "present_key_values_key_0", "present_key_values_value_0"], outputs
        )
        unique_ops = {n.op_type for n in onx.model_proto.graph.node}
        self.assertNotIn("HalfRotaryEmbedding", unique_ops)
        self.assertIn("RotaryEmbedding", unique_ops)
        self.assertIn("SimplifiedLayerNormalization", unique_ops)
        self.assertIn("SkipSimplifiedLayerNormalization", unique_ops)
        self.assertIn("QuickGelu", unique_ops)
        self.assertIn("CausalMaskMulAdd", unique_ops)
        self.assertIn("CausalMask", unique_ops)
        self.assertIn("LocalAttentionGQAsQ_to1", unique_ops)
        self.assertInOr(("CosSinCache_p1", "CosSinCacheWithRange"), unique_ops)
        sess = onnxruntime.InferenceSession(filename, providers=["CPUExecutionProvider"])
        feeds = make_feeds(sess, b1, use_numpy=True)
        got = sess.run(None, feeds)
        diff = max_diff(expected, got)
        assert diff["abs"] <= 1e-5, f"diff={diff}"

        problem = dict(
            input_ids=torch.tensor([[24320]], dtype=torch.int64),
            attention_mask=torch.tensor([[1, 1, 1, 1]], dtype=torch.int64),
            past_key_values=make_dynamic_cache(
                [
                    torch.rand((1, 1, 3, 96), dtype=torch.float32),
                    torch.rand((1, 1, 3, 96), dtype=torch.float32),
                ]
            ),
        )

        expected = model(**torch_deepcopy(problem))
        sess = onnxruntime.InferenceSession(filename, providers=["CPUExecutionProvider"])
        feeds = make_feeds(sess, problem, use_numpy=True)
        got = sess.run(None, feeds)
        diff = max_diff(expected, got)
        assert diff["abs"] <= 1e-5, f"diff={diff}"


if __name__ == "__main__":
    unittest.main(verbosity=2)
