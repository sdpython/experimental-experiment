import unittest
import onnx
import onnx_ir
from onnx_diagnostic.helpers import flatten_object
from onnx_diagnostic.helpers.torch_helper import torch_dtype_to_onnx_dtype, torch_deepcopy
from onnx_diagnostic.torch_models.hghub import get_untrained_model_with_inputs
from onnx_diagnostic.torch_export_patches.patch_inputs import use_dyn_not_str
from onnx_diagnostic.torch_export_patches import torch_export_patches
from experimental_experiment.ext_test_case import ExtTestCase
from experimental_experiment.convert.model_builder_base import ModelBuilderBase
from experimental_experiment.torch_interpreter import to_onnx


def torch_dtype_to_ir_data_type(dt):
    itype = torch_dtype_to_onnx_dtype(dt)
    return {
        onnx.TensorProto.FLOAT: onnx_ir.DataType.FLOAT,
        onnx.TensorProto.FLOAT16: onnx_ir.DataType.FLOAT16,
        onnx.TensorProto.BFLOAT16: onnx_ir.DataType.BFLOAT16,
        onnx.TensorProto.INT64: onnx_ir.DataType.INT64,
        onnx.TensorProto.BOOL: onnx_ir.DataType.BOOL,
    }[itype]


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

        layer_export = model.model.layers[0]
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
                    new_inputs[f"past_key_values.{i//2}.key"] = flat[i]
                    new_inputs[f"past_key_values.{i//2}.value"] = flat[i + 1]
        self.assertEqual(
            [
                "hidden_states",
                "attention_mask",
                "position_ids",
                "past_key_values.0.key",
                "past_key_values.0.value",
                "cache_position",
            ],
            list(new_inputs),
        )
        cache_dtype = torch_dtype_to_ir_data_type(new_inputs["past_key_values.0.key"].dtype)
        present_shape = store[4][1]["past_key_values"].layers[0].keys.shape
        builder = ModelBuilderBase(
            input_names=list(new_inputs),
            input_types={k: torch_dtype_to_ir_data_type(t.dtype) for k, t in new_inputs.items()},
            input_shapes={k: tuple(int(i) for i in t.shape) for k, t in new_inputs.items()},
            output_names=["output", "present_key_values.0.key", "present_key_values.0.value"],
            output_types={
                "output": torch_dtype_to_ir_data_type(output.dtype),
                "present_key_values.0.key": cache_dtype,
                "present_key_values.0.value": cache_dtype,
            },
            output_shapes={
                "output": tuple(int(i) for i in output.shape),
                "present_key_values.0.key": present_shape,
                "present_key_values.0.value": present_shape,
            },
            num_attn_heads=args[0].shape[1],
            num_kv_heads=present_shape[1],
            head_size=present_shape[3],
            op_attention_type="GroupQueryAttention",
            rms_norm_eps=layer_export.post_attention_layernorm.variance_epsilon,
            context_length=model.model.rotary_emb.max_seq_len_cached,
            original_context_length=model.model.rotary_emb.original_max_seq_len,
            activation=layer_export.mlp.act_fn,
            intermediate_size=layer_export.mlp.intermediate_size,
        )
        builder.make_model(layer_export)
        onx = builder.build_model()
        model_proto = onnx_ir.serde.serialize_model(onx)
        self.dump_onnx("test_model_builder_attention.onnx", model_proto)

    def test_custom_export_with_layer_exposed_tiny_llm(self):
        model_id = "arnir0/Tiny-LLM"
        data = get_untrained_model_with_inputs(model_id)
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        layer_class = type(model.model.layers[0])
        filename = self.get_dump_file("test_custom_export_with_layer_exposed_tiny_llm.onnx")
        with torch_export_patches(patch_transformers=True):
            to_onnx(
                model,
                inputs,
                dynamic_shapes=ds,
                export_modules_as_functions={layer_class},
                inline=False,
                optimize=False,
                filename=filename,
                verbose=1,
            )

    def test_dynamo_export_with_layer_exposed_tiny_llm(self):
        import torch

        model_id = "arnir0/Tiny-LLM"
        data = get_untrained_model_with_inputs(model_id)
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        dsr = use_dyn_not_str(ds)
        layer_class = type(model.model.layers[0])
        names_to_preserve = []
        for name, mod in model.named_modules():
            if isinstance(mod, layer_class):
                names_to_preserve.append(name)
        self.assertEqual(["model.layers.0"], names_to_preserve)
        with torch_export_patches(patch_transformers=True):
            # case 1: does not work
            # ep = torch.export.export(
            #    model,
            #    (),
            #    kwargs=inputs,
            #    dynamic_shapes=dsr,
            #    preserve_module_call_signature=tuple(names_to_preserve),
            # )
            # filename = self.get_dump_file(
            #   "test_dynamo_export_with_layer_exposed_tiny_llm.1.onnx")
            # torch.onnx.export(ep, (), filename, kwargs=inputs, dynamic_shapes=ds)

            # case 2
            ep = torch.export.export(model, (), kwargs=inputs, dynamic_shapes=dsr)
            new_ep = torch.export.unflatten(ep)
            # does not work
            # filename = self.get_dump_file(
            #   "test_dynamo_export_with_layer_exposed_tiny_llm.2.onnx")
            # torch.onnx.export(new_ep, (), filename, kwargs=inputs, dynamic_shapes=ds)

            # case 3: works but does not produce what is expected
            class Model(torch.nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m

                def forward(self, input_ids, attention_mask, position_ids, past_key_values):
                    return self.m(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                    )

            model = Model(new_ep)
            filename = self.get_dump_file("test_dynamo_export_with_layer_exposed_tiny_llm.3.onnx")
            torch.onnx.export(model, (), filename, kwargs=inputs, dynamic_shapes=ds)


if __name__ == "__main__":
    unittest.main(verbosity=2)
