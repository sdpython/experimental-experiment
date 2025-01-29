"""
.. _l-plot-exporter-exporter-phi35-piece:

Export Phi-3.5-mini-instruct piece by piece
===========================================

:func:`torch.export.export` often breaks on big models because there
are control flows or instructions breaking the propagation of
dynamic shapes (see ...). The function usually gives an indication where
the model implementation can be fixed but in case, that is not possible,
we can try to export the model piece by piece: every module
is converted separately from its submodule. A model can be exported even
if one of its submodules cannot.

Model
+++++
"""

import pprint
from typing import Any, Dict
import torch
import transformers
from experimental_experiment.helpers import string_type
from experimental_experiment.torch_interpreter.piece_by_piece import (
    CustomOpStrategy,
    trace_execution_piece_by_piece,
)
from experimental_experiment.torch_interpreter.onnx_export_errors import (
    register_additional_serialization_functions,
)


def get_phi35_untrained(batch_size: int = 2, **kwargs) -> Dict[str, Any]:
    """
    Gets a non initialized model with two sets of inputs and different shapes.

    :param batch_size: batch size
    :param kwargs: to overwrite the configuration, example ``num_hidden_layers=1``
    :return: dictionary

    See `Phi-3.5-mini-instruct/config.json
    <https://huggingface.co/microsoft/Phi-3.5-mini-instruct/blob/main/config.json>`_.
    """
    config = {
        "_name_or_path": "Phi-3.5-mini-instruct",
        "architectures": ["Phi3ForCausalLM"],
        "attention_dropout": 0.0,
        "auto_map": {
            "AutoConfig": "configuration_phi3.Phi3Config",
            "AutoModelForCausalLM": "modeling_phi3.Phi3ForCausalLM",
        },
        "bos_token_id": 1,
        "embd_pdrop": 0.0,
        "eos_token_id": 32000,
        "hidden_act": "silu",
        "hidden_size": 3072,
        "initializer_range": 0.02,
        "intermediate_size": 8192,
        "max_position_embeddings": 131072,
        "model_type": "phi3",
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "num_key_value_heads": 32,
        "original_max_position_embeddings": 4096,
        "pad_token_id": 32000,
        "resid_pdrop": 0.0,
        "rms_norm_eps": 1e-05,
        "rope_scaling": {
            "long_factor": [
                1.0800000429153442,
                1.1100000143051147,
                1.1399999856948853,
                1.340000033378601,
                1.5899999141693115,
                1.600000023841858,
                1.6200000047683716,
                2.620000123977661,
                3.2300000190734863,
                3.2300000190734863,
                4.789999961853027,
                7.400000095367432,
                7.700000286102295,
                9.09000015258789,
                12.199999809265137,
                17.670000076293945,
                24.46000099182129,
                28.57000160217285,
                30.420001983642578,
                30.840002059936523,
                32.590003967285156,
                32.93000411987305,
                42.320003509521484,
                44.96000289916992,
                50.340003967285156,
                50.45000457763672,
                57.55000305175781,
                57.93000411987305,
                58.21000289916992,
                60.1400032043457,
                62.61000442504883,
                62.62000274658203,
                62.71000289916992,
                63.1400032043457,
                63.1400032043457,
                63.77000427246094,
                63.93000411987305,
                63.96000289916992,
                63.970001220703125,
                64.02999877929688,
                64.06999969482422,
                64.08000183105469,
                64.12000274658203,
                64.41000366210938,
                64.4800033569336,
                64.51000213623047,
                64.52999877929688,
                64.83999633789062,
            ],
            "short_factor": [
                1.0,
                1.0199999809265137,
                1.0299999713897705,
                1.0299999713897705,
                1.0499999523162842,
                1.0499999523162842,
                1.0499999523162842,
                1.0499999523162842,
                1.0499999523162842,
                1.0699999332427979,
                1.0999999046325684,
                1.1099998950958252,
                1.1599998474121094,
                1.1599998474121094,
                1.1699998378753662,
                1.2899998426437378,
                1.339999794960022,
                1.679999828338623,
                1.7899998426437378,
                1.8199998140335083,
                1.8499997854232788,
                1.8799997568130493,
                1.9099997282028198,
                1.9399996995925903,
                1.9899996519088745,
                2.0199997425079346,
                2.0199997425079346,
                2.0199997425079346,
                2.0199997425079346,
                2.0199997425079346,
                2.0199997425079346,
                2.0299997329711914,
                2.0299997329711914,
                2.0299997329711914,
                2.0299997329711914,
                2.0299997329711914,
                2.0299997329711914,
                2.0299997329711914,
                2.0299997329711914,
                2.0299997329711914,
                2.0799996852874756,
                2.0899996757507324,
                2.189999580383301,
                2.2199995517730713,
                2.5899994373321533,
                2.729999542236328,
                2.749999523162842,
                2.8399994373321533,
            ],
            "type": "longrope",
        },
        "rope_theta": 10000.0,
        "sliding_window": 262144,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "use_cache": True,
        "attention_bias": False,
        "vocab_size": 32064,
    }
    config.update(**kwargs)
    conf = transformers.Phi3Config(**config)
    model = transformers.Phi3ForCausalLM(conf)
    model.eval()

    cache = transformers.cache_utils.DynamicCache(config["num_hidden_layers"])
    for i in range(config["num_hidden_layers"]):
        cache.update(
            torch.randn(batch_size, 32, 30, 96), torch.randn(batch_size, 32, 30, 96), i
        )
    cache2 = transformers.cache_utils.DynamicCache(config["num_hidden_layers"])
    for i in range(config["num_hidden_layers"]):
        cache2.update(
            torch.randn(batch_size + 1, 32, 31, 96),
            torch.randn(batch_size + 1, 32, 31, 96),
            i,
        )

    inputs = dict(
        input_ids=torch.randint(0, 32064, (batch_size, 3)).to(torch.int64),
        attention_mask=torch.ones((batch_size, 33)).to(torch.int64),
        past_key_values=cache,
    )
    inputs2 = dict(
        input_ids=torch.randint(0, 32064, (batch_size + 1, 4)).to(torch.int64),
        attention_mask=torch.ones((batch_size + 1, 35)).to(torch.int64),
        past_key_values=cache2,
    )
    return dict(inputs=inputs, model=model, inputs2=inputs2)


data = get_phi35_untrained(num_hidden_layers=2)
model, inputs, inputs2 = data["model"], data["inputs"], data["inputs2"]

print(string_type(inputs, with_shape=True))

# %%
# Dynamic Shapes
# ++++++++++++++
#
# We want to infer the dynamic shapes from the two sets of inputs we gave.
# For that, we use a function to trace the execution of the model
# including its submodules. It is going to execute the model twice
# with the two sets of inputs and stores every intermediate input and output.

diag = trace_execution_piece_by_piece(model, [inputs, inputs2], verbose=2)

# %%
# Now we keep in memory every input/output for the submodules,
# we can guess the dynamic shapes for every of them.
# The final ones:
dynamic_shapes = diag.guess_dynamic_shapes()
print("The dynamic shapes are:")
pprint.pprint(dynamic_shapes)

# %%
# And all the dynamic shapes all along the traced submodules.
print(
    diag.pretty_text(
        with_dynamic_shape=True,
        with_shape=False,
        with_min_max=False,
        with_device=False,
        with_inputs=False,
    ).replace("<_DimHint.DYNAMIC: 3>", "DYN")
)

# %%
# Evaluate the export
# +++++++++++++++++++
#
# In many cases, the export (to :class:`torch.fx.Graph`, to ONNX)
# does not work on the first try. We need a way to understand
# how much the model can be exported. It can be used to evaluate
# the how much code needs to be rewritten or patched to be exportable.
# The verbosity can be increase to show dynamic shapes, results
# of the discrepancies.
# Let's display the module and its submodule first.

print(
    diag.pretty_text(
        with_dynamic_shape=False,
        with_shape=False,
        with_min_max=False,
        with_device=False,
        with_inputs=False,
    )
)

# %%
# The we try to export to see the submodule failing the whole model.
# We can pickle the failing model and restore it to speedup
# the refactoring to make it work.
print("----------------------")
ep = diag.try_export(
    exporter="fx",
    use_dynamic_shapes=True,
    exporter_kwargs=dict(strict=False),
    bypass_kwargs=dict(patch_transformers=True, replace_dynamic_cache=True),
    verbose=1,
)
print(f"success: {ep.status}")
print(diag.get_export_report())

# %%
# Export piece by piece
# +++++++++++++++++++++
#
# The main module is not exportable because one piece cannot be exported.
# But maybe if we assume it works, maybe everything else is working.
# By using ``replace_by_custom_op=CustomOpStrategy.LOCAL``, the function
# replaces every submodule by a custom operator so that it can
# the exported program for every module without its submodules.
#
# It does not work yet because it does not know how to automatically produce
# a function producing a shape based on the input ones.
# This function needs to be written by the user for
# class Phi3RotaryEmbedding.


def result_of_same_shape(*args, **kwargs):
    "Returns the shape of one element of the cache based on the inputs."
    return torch.empty((*args[3].shape[:2], args[1].shape[1], args[3].shape[-1])).to(
        args[3].dtype
    )


with register_additional_serialization_functions():
    ep = diag.try_export(
        exporter="fx",
        use_dynamic_shapes=True,
        exporter_kwargs=dict(strict=False),
        # bypass_kwargs=dict(patch_transformers=True, replace_dynamic_cache=True),
        verbose=10,
        replace_by_custom_op=CustomOpStrategy.LOCAL,
        quiet=0,
        shape_functions={
            "Phi3Model": {
                1: result_of_same_shape,
                2: result_of_same_shape,
                3: result_of_same_shape,
                4: result_of_same_shape,
            }
        },
    )
print(f"success: {ep.status}")

# %%
# Let's print a readable report.
print(diag.get_export_report(fx=True))
