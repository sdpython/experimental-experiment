import unittest
from onnx_diagnostic.helpers import flatten_object
from onnx_diagnostic.helpers.torch_helper import torch_dtype_to_onnx_dtype, torch_deepcopy
from onnx_diagnostic.torch_models.hghub import get_untrained_model_with_inputs
from experimental_experiment.ext_test_case import ExtTestCase
from experimental_experiment.convert.model_builder_base import ModelBuilderBase


class TestModelBuilderHelper(ExtTestCase):
    def test_model_builder_attention(self):
        import torch

        model_id = "arnir0/Tiny-LLM"
        input_ids = torch.randint(31000, size=(1, 7), dtype=torch.int64)
        attention_mask = torch.ones((1, 7), dtype=torch.bool)
        attention_mask[:, 2] = 0
        data = get_untrained_model_with_inputs(model_id)
        model = data["model"]

        self.assertEqual(model.model.layers[0].__class__.__name__, "LlamaDecoderLayer")
        forward_method = model.model.layers[0].forward
        store = []

        def steal_layer0(*args, **kwargs):
            store.append(torch_deepcopy((args, kwargs)))
            res = forward_method(*args, **kwargs)
            store.append(torch_deepcopy(res))
            return res

        model.model.layers[0].forward = steal_layer0
        model.generate(input_ids=input_ids, attention_mask=attention_mask)

        # for i, aa in enumerate(store):
        #     print(f"{i}: {self.string_type(aa, with_shape=True)}")
        args, kwargs = store[2]
        output = store[3]
        self.assertEqual(len(args), 1)
        self.assertIsInstance(output, torch.Tensor)
        n_layers = len(model.model.layers)
        new_inputs = {"hidden_states": args[0]}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                new_inputs[k] = v
            elif v is not None and not isinstance(v, (int, float, bool, tuple)):
                self.assertIn("Cache", v.__class__.__name__)
                flat = flatten_object(v, drop_keys=True)
                if not flat:
                    continue
                self.assertEqual(len(flat), 2 * n_layers)
                for i in range(0, len(flat), 2):
                    new_inputs[f"past_key_values_keys_{i//2}"] = flat[i]
                    new_inputs[f"past_key_values_values_{i//2}"] = flat[i + 1]
        self.assertEqual(
            [
                "hidden_states",
                "attention_mask",
                "position_ids",
                "past_key_values_keys_0",
                "past_key_values_values_0",
                "cache_position",
            ],
            list(new_inputs),
        )
        cache_dtype = torch_dtype_to_onnx_dtype(new_inputs["past_key_values_keys_0"].dtype)
        present_shape = store[4][1]["past_key_values"].layers[0].keys.shape
        builder = ModelBuilderBase(
            input_names=list(new_inputs),
            input_types={k: torch_dtype_to_onnx_dtype(t.dtype) for k, t in new_inputs.items()},
            input_shapes={k: tuple(int(i) for i in t.shape) for k, t in new_inputs.items()},
            output_names=["output", "present_key_values_keys_0", "present_key_values_values_0"],
            output_types={
                "output": torch_dtype_to_onnx_dtype(output.dtype),
                "present_key_values_keys_0": cache_dtype,
                "present_key_values_values_0": cache_dtype,
            },
            output_shapes={
                "output": tuple(int(i) for i in output.shape),
                "present_key_values_keys_0": present_shape,
                "present_key_values_values_0": present_shape,
            },
            num_attn_heads=present_shape[1],
        )
        builder.make_model(model.model.layers[0])
        onx = builder.build_model()
        self.dump_onnx("test_model_builder_attention.onnx", onx)


if __name__ == "__main__":
    unittest.main(verbosity=2)
