"""
.. _l-plot-exporter-recipes-custom-phi2:

to_onnx and Phi-2
=================

Exports model `Phi-2 <https://huggingface.co/microsoft/phi-2>`_.
We use a dummy model. The main difficulty is to set the dynamic shapes properly.
If there is an issue, you can go to the following line:
`torch/fx/experimental/symbolic_shapes.py#L5965
<https://github.com/pytorch/pytorch/blob/main/torch/fx/experimental/symbolic_shapes.py#L5965>`_
and look for ``log.info("set_replacement %s = %s (%s) %s", a, tgt, msg, tgt_bound)`` and add
before or after, something like:

::

    if isinstance(tgt, int):
        raise AssertionError(
            f"dynamic shape becomes a constant "
            f"{[a, tgt, type(tgt), msg, tgt_bound]}"
        )

Adding ``TORCH_LOGS="+dynamo" TORCHDYNAMO_VERBOSE=1`` prints out more information
about dynamic shapes.

Model
+++++
"""

import copy
from typing import Any, Dict
import onnx
import torch
import transformers
from onnx_array_api.plotting.graphviz_helper import plot_dot
from experimental_experiment.cache_helpers import make_dynamic_cache
from experimental_experiment.helpers import string_type
from experimental_experiment.xbuilder import GraphBuilder, InferShapesOptions
from experimental_experiment.torch_interpreter import to_onnx, ExportOptions


def get_phi2_untrained(batch_size: int = 2, **kwargs) -> Dict[str, Any]:
    """
    Gets a non initialized model with its inputs

    :param batch_size: batch size
    :param kwargs: to overwrite the configuration, example ``num_hidden_layers=1``
    :return: dictionary

    See `Phi-2/config.json
    <https://huggingface.co/microsoft/phi-2/blob/main/config.json>`_.
    """
    config = {
        "_name_or_path": "microsoft/phi-2",
        "architectures": ["PhiForCausalLM"],
        "attention_dropout": 0.0,
        "bos_token_id": 50256,
        "embd_pdrop": 0.0,
        "eos_token_id": 50256,
        "hidden_act": "gelu_new",
        "hidden_size": 2560,
        "initializer_range": 0.02,
        "intermediate_size": 10240,
        "layer_norm_eps": 1e-05,
        "max_position_embeddings": 2048,
        "model_type": "phi",
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "num_key_value_heads": 32,
        "partial_rotary_factor": 0.4,
        "qk_layernorm": False,
        "resid_pdrop": 0.1,
        "rope_scaling": None,
        "rope_theta": 10000.0,
        "tie_word_embeddings": False,
        "torch_dtype": "float16",
        "transformers_version": "4.37.0",
        "use_cache": True,
        "vocab_size": 51200,
    }
    config.update(**kwargs)
    conf = transformers.PhiConfig(**config)
    model = transformers.PhiForCausalLM(conf)
    model.eval()

    batch = torch.export.Dim("batch", min=1, max=1024)
    seq_length = torch.export.Dim("seq_length", min=1, max=4096)
    shapes = {}

    cache = make_dynamic_cache(
        [
            (torch.randn(batch_size, 32, 30, 80), torch.randn(batch_size, 32, 30, 80))
            for i in range(config["num_hidden_layers"])
        ]
    )
    cache2 = make_dynamic_cache(
        [
            (torch.randn(batch_size + 1, 32, 31, 80), torch.randn(batch_size + 1, 32, 31, 80))
            for i in range(config["num_hidden_layers"])
        ]
    )

    inputs = dict(
        input_ids=torch.randint(0, 50285, (batch_size, 3)).to(torch.int64),
        attention_mask=torch.ones((batch_size, 33)).to(torch.int64),
        past_key_values=cache,
    )
    inputs2 = dict(
        input_ids=torch.randint(0, 50285, (batch_size + 1, 4)).to(torch.int64),
        attention_mask=torch.ones((batch_size + 1, 35)).to(torch.int64),
        past_key_values=cache2,
    )
    n = len(cache.key_cache)
    cache_length = torch.export.Dim("cache_length", min=1, max=4096)
    shapes.update(
        {
            "input_ids": {0: batch, 1: seq_length},
            "attention_mask": {
                0: batch,
                1: torch.export.Dim.DYNAMIC,  # cache_length + seq_length
            },
            "past_key_values": [
                [{0: batch, 2: cache_length} for _ in range(n)],  # 0: batch,
                [{0: batch, 2: cache_length} for _ in range(n)],  # 0: batch,
            ],
        }
    )

    return dict(inputs=inputs, model=model, dynamic_shapes=shapes, inputs2=inputs2)


data = get_phi2_untrained(num_hidden_layers=2)
model = data["model"]
inputs = data["inputs"]
dynamic_shapes = data["dynamic_shapes"]

print("inputs", string_type(inputs, with_shape=True))
print("dynamic_shapes", dynamic_shapes)


# %%
# Let's check it is working.
# We need to copy the input before calling the model
# because it modifies the inputs and they are not properly
# set up when the export starts.
model(**copy.deepcopy(inputs))

# %%
# Export
# ++++++
#
# We try to export with :func:`experimental_experiment.torch_interpreter.to_onnx`.
#
try:
    to_onnx(
        copy.deepcopy(model),
        (),
        kwargs=copy.deepcopy(inputs),
        dynamic_shapes=dynamic_shapes,
    )
except Exception as e:
    print(f"export failed due to {e}")

# %%
# The export fails for a couple of reason but it is possible to patch the
# code to make it work. All those modifications are put in place by
# :func:`bypass_export_some_errors
# <onnx_diagnostic.torch_export_patches.bypass_export_some_errors>`
# and reverted after the export is done. Among other things, this function registers
# serialization functions as shown in example
# :ref:`l-plot-torch-export-with-dynamic-cache-201`.

from onnx_diagnostic.torch_export_patches import bypass_export_some_errors

with bypass_export_some_errors(patch_transformers=True, verbose=1) as modificator:
    print("inputs before", string_type(inputs, with_shape=True))
    inputs = modificator(inputs)
    print("inputs after", string_type(inputs, with_shape=True))
    # ep = torch.export.export(model, (), inputs, dynamic_shapes=dynamic_shapes, strict=False)
    large_onx = to_onnx(
        model,
        (),
        inputs,
        dynamic_shapes=dynamic_shapes,
        export_options=ExportOptions(strict=False),
        large_model=True,
    )
    large_onx.save("plot_exporter_recipes_c_phi2.onnx", all_tensors_to_one_file=True)

# %%
# Exported Model
# ++++++++++++++
#
# Let's display the model.

onx = onnx.load("plot_exporter_recipes_c_phi2.onnx")
gr = GraphBuilder(onx, infer_shapes_options=InferShapesOptions.NONE)
print(gr.pretty_text())

# %%
# Visually.

plot_dot(onx)
