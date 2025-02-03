"""
.. _l-plot-torch-export-compile-101:

======================
102: Tweak onnx export
======================

export, unflatten and compile
=============================
"""

import torch
from experimental_experiment.helpers import pretty_onnx
from experimental_experiment.torch_interpreter import to_onnx


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

# %%
# The export keeps track of the submodules calls.

print("-- module_call_graph", type(exported_program.module_call_graph))
print(exported_program.module_call_graph)

# %%
# That information can be converted back into a exported program.

ep = torch.export.unflatten(exported_program)
print("-- unflatten", type(exported_program.graph))
print(ep.graph)

# %%
# Another graph obtained with torch.compile.


def my_compiler(gm, example_inputs):
    print("-- graph with torch.compile")
    print(gm.graph)
    return gm.forward


optimized_mod = torch.compile(model, fullgraph=True, backend=my_compiler)
optimized_mod(*inputs)

# %%
# Unflattened
# ===========


class SubNeuron2(torch.nn.Module):
    def __init__(self, n_dims: int = 5, n_targets: int = 3):
        super().__init__()
        self.linear = torch.nn.Linear(n_dims, n_targets)

    def forward(self, x):
        z = self.linear(x)
        return torch.sigmoid(z)


class Neuron2(torch.nn.Module):
    def __init__(self, n_dims: int = 5, n_targets: int = 3):
        super().__init__()
        self.neuron = SubNeuron2(n_dims, n_targets)

    def forward(self, x):
        z = self.neuron(x)
        return torch.relu(z)


model = Neuron2()
inputs = (torch.randn(1, 5),)
expected = model(*inputs)

onx = to_onnx(model, inputs)
print(pretty_onnx(onx))

# %%
# Let's preserve the module.


onx = to_onnx(model, inputs, export_modules_as_functions=True)
print(pretty_onnx(onx))
