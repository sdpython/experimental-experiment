"""
.. _l-plot-torch-export-onnx-101:

102: Tweak onnx export
======================

A Llm model
+++++++++++
"""

import pprint
import torch
import torch.export._swap
from transformers.models.phi3.modeling_phi3 import (
    Phi3Attention,
    Phi3LongRoPEScaledRotaryEmbedding,
    Phi3RMSNorm,
)
from experimental_experiment.torch_test_helper import string_type
from experimental_experiment.torch_models.llm_model_helper import get_phi_35_mini_instruct
from experimental_experiment.torch_interpreter.onnx_export_errors import (
    bypass_export_some_errors,
)
from experimental_experiment.bench_run import max_diff


phi, inputs = get_phi_35_mini_instruct(num_hidden_layers=1, _attn_implementation="eager")

print("model type", type(phi))
print("inputs:", string_type(inputs))

expected = phi(**inputs)

###################################
# The module it contains
# ++++++++++++++++++++++

types = set(type(inst) for _, inst in phi.named_modules())
print("interesting types:")
pprint.pprint(list(types))

###################################
# The interesting module
# ++++++++++++++++++++++


attentions = {
    name: inst
    for name, inst in phi.named_modules()
    if isinstance(inst, (Phi3RMSNorm, Phi3Attention, Phi3LongRoPEScaledRotaryEmbedding))
}

print("names found:", tuple(sorted(attentions)))

###################################
# The exported program.


with bypass_export_some_errors():
    exported_program = torch.export.export(
        phi,
        tuple(),
        inputs,
        strict=False,
        preserve_module_call_signature=tuple(attentions),
    )
print(exported_program.graph)

###################################
# Checking they are the same.

new_outputs = exported_program.module()(**inputs)

print("--", string_type(expected), string_type(new_outputs))

for k in expected:
    print(
        f"-- max_diff for {k}: types "
        f"{string_type(expected[k])}, {string_type(new_outputs[k])}: "
        f"max_diff={max_diff(expected[k], new_outputs[k])}"
    )

############################
# With attention preserved

swapped_gm = torch.export._swap._swap_modules(exported_program, attentions)

print("--- the new graph")
print(swapped_gm.graph)


###################################
# Checking again they are the same.

print(type(swapped_gm))
new_outputs = swapped_gm(**inputs)
print("--", string_type(expected), string_type(new_outputs))
for k in expected:
    print(
        f"-- max_diff for {k}: types "
        f"{string_type(expected[k])}, {string_type(new_outputs[k])}: "
        f"max_diff={max_diff(expected[k], new_outputs[k])}"
    )
