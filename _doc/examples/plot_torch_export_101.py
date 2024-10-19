"""
.. _l-plot-torch-export-101:

101: Some dummy examples with torch.export.export
=================================================

Easy Case
+++++++++
"""

import torch


class Neuron(torch.nn.Module):
    def __init__(self, n_dims: int = 5, n_targets: int = 3):
        super().__init__()
        self.linear = torch.nn.Linear(n_dims, n_targets)

    def forward(self, x):
        z = self.linear(x)
        return torch.sigmoid(z)


exported_program = torch.export.export(Neuron(), (torch.randn(1, 5),))
print(exported_program.graph)

######################################
# With an integer as input
# ++++++++++++++++++++++++
#
# As `torch.export.export <https://pytorch.org/docs/stable/export.html>`_
# documentation, integer do not show up on the graph.


class NeuronIInt(torch.nn.Module):
    def __init__(self, n_dims: int = 5, n_targets: int = 3):
        super().__init__()
        self.linear = torch.nn.Linear(n_dims, n_targets)

    def forward(self, x: torch.Tensor, i_input: int):
        z = self.linear(x)
        return torch.sigmoid(z)[:, i_input]


exported_program = torch.export.export(NeuronIInt(), (torch.randn(1, 5), 2))
print(exported_program.graph)

######################################
# With an integer as input
# ++++++++++++++++++++++++


class NeuronIInt(torch.nn.Module):
    def __init__(self, n_dims: int = 5, n_targets: int = 3):
        super().__init__()
        self.linear = torch.nn.Linear(n_dims, n_targets)

    def forward(self, x: torch.Tensor, i_input):
        z = self.linear(x)
        return torch.sigmoid(z)[:, i_input]


exported_program = torch.export.export(
    NeuronIInt(), (torch.randn(1, 5), torch.Tensor([2]).to(torch.int32))
)
print(exported_program.graph)


######################################
# Wrapped
# +++++++


class WrappedNeuronIInt(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)


exported_program = torch.export.export(
    WrappedNeuronIInt(NeuronIInt()), (torch.randn(1, 5), torch.Tensor([2]).to(torch.int32))
)
print(exported_program.graph)
