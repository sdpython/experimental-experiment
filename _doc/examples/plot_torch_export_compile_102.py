"""
.. _l-plot-torch-export-compile-101:

======================
102: Tweak onnx export
======================

export or compile
=================
"""

import pprint
import torch
from experimental_experiment.helpers import string_type
from experimental_experiment.torch_models.llm_model_helper import get_phi_35_mini_instruct

# from experimental_experiment.bench_run import max_diff


class SubNeuron(torch.nn.Module):
    def __init__(self, n_dims: int = 5, n_targets: int = 3):
        super().__init__()
        self.linear = torch.nn.Linear(n_dims, n_targets)

    def forward(self, x):
        z = self.linear(x)
        return torch.sigmoid(z)


class Neuron(torch.nn.Module):
    def __init__(self, n_dims: int = 5, n_targets: int = 3):
        super().__init__()
        self.neuron = SubNeuron(n_dims, n_targets)

    def forward(self, x):
        z = self.neuron(x)
        return torch.relu(z)


model = Neuron()
inputs = (torch.randn(1, 5),)
expected = model(*inputs)
exported_program = torch.export.export(model, inputs)

print("-- fx graph with torch.export.export")
print(exported_program.graph)

####################################
# The export keeps track of the submodules calls.

print("-- module_call_graph", type(exported_program.module_call_graph))
print(exported_program.module_call_graph)

#########################################
# That information can be converted back into a exported program.

ep = torch.export.unflatten(exported_program)
print("-- unflatten", type(exported_program.graph))
print(ep.graph)


#######################################
# There is no trace left of the sub modules.
#
# And through compile
# ===================


def my_compiler(gm, example_inputs):
    print("-- graph with torch.compile")
    print(gm.graph)
    return gm.forward


optimized_mod = torch.compile(model, fullgraph=True, backend=my_compiler)
optimized_mod(*inputs)

################################################
# Applied to Phi
# ==============

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
# The exported program.
# +++++++++++++++++++++

from transformers.models.phi3.modeling_phi3 import logger

logger.warning_once = lambda *_, **__: None
torch.export._EXPORT_FLAGS = {"non_script"}
optimized_mod = torch.compile(phi, fullgraph=True, backend=my_compiler)
optimized_mod(**inputs)
