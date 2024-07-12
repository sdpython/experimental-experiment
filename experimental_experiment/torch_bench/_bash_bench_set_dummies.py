import torch
from ._bash_bench_model_runner import MakeConfig


class Neuron(torch.nn.Module):
    def __init__(self, n_dims: int = 5, n_targets: int = 3):
        super(Neuron, self).__init__()
        self.linear = torch.nn.Linear(n_dims, n_targets)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

    def _get_random_inputs(self, device: str):
        return (torch.randn(1, 5).to(device),)

    config = MakeConfig(download=False, to_tuple=False)


class Neuron16(Neuron):
    def __init__(self, n_dims: int = 5, n_targets: int = 3):
        super(Neuron, self).__init__()
        self.linear = torch.nn.Linear(n_dims, n_targets, dtype=torch.float16)
        assert self.linear.weight.dtype == torch.float16
        assert self.linear.bias.dtype == torch.float16

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

    def _get_random_inputs(self, device: str):
        return (torch.randn(1, 5).to(torch.float16).to(device),)


class NeuronTuple(torch.nn.Module):
    def __init__(self, n_dims: int = 5, n_targets: int = 3):
        super(NeuronTuple, self).__init__()
        self.linear = torch.nn.Linear(n_dims, n_targets)

    def forward(self, x):
        y = self.linear(x)
        return (torch.sigmoid(y), (x, y))

    def _get_random_inputs(self, device: str):
        return (torch.randn(1, 5).to(device),)

    config = MakeConfig(download=False, to_tuple=False)
