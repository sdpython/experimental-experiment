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
# An exporter based on :func:`torch.export.export` cannot consider
# the integer as an input.


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
#
# But if the integer is wrapped into a Tensor, it works.


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
#
# Wrapped, it continues to work.


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


###########################################
# List
# ++++
#
# The last one does not export. An exporter based on
# :func:`torch.export.export` cannot work.


class NeuronNoneListInt(torch.nn.Module):
    def __init__(self, n_dims: int = 5, n_targets: int = 3):
        super().__init__()
        self.linear = torch.nn.Linear(n_dims, n_targets)

    def forward(self, x, yz, i_input):
        z = self.linear(x + yz[0] * yz[3])
        return torch.sigmoid(z)[:i_input]


try:
    exported_program = torch.export.export(
        NeuronNoneListInt(),
        (
            torch.randn(1, 5),
            [torch.randn(1, 5), None, None, torch.randn(1, 5)],
            torch.Tensor([2]).to(torch.int32),
        ),
    )
    print(exported_program.graph)
except torch._dynamo.exc.Unsupported as e:
    print("-- an error occured:")
    print(e)


###########################################
# Loops
# +++++
#
# Loops are not captured.


class NeuronLoop(torch.nn.Module):
    def __init__(self, n_dims: int = 5, n_targets: int = 3):
        super().__init__()
        self.linear = torch.nn.Linear(n_dims, n_targets)

    def forward(self, x, xs):
        z = self.linear(x)
        for i in range(len(xs)):
            x += xs[i] * (i + 1)
        return z


exported_program = torch.export.export(
    NeuronLoop(),
    (
        torch.randn(1, 5),
        [torch.randn(1, 5), torch.randn(1, 5)],
    ),
)
print(exported_program.graph)
