import torch
from ._bash_bench_model_runner import MakeConfig


class Neuron(torch.nn.Module):
    def __init__(self, n_dims: int = 5, n_targets: int = 3):
        super().__init__()
        self.linear = torch.nn.Linear(n_dims, n_targets)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

    def _get_random_inputs(self, device: str):
        return (torch.randn(2, 5).to(device),)

    config = MakeConfig(download=False, to_tuple=False)


class Neuron2Outputs(torch.nn.Module):
    def __init__(self, n_dims: int = 1000, n_targets: int = 100):
        super().__init__()
        self.linear = torch.nn.Linear(n_dims, n_targets)

    def forward(self, x):
        return torch.sigmoid(self.linear(x + 10)), torch.softmax(self.linear(x), dim=1)

    def _get_random_inputs(self, device: str):
        return (torch.randn(1, 1000).to(device),)

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
        super().__init__()
        self.linear = torch.nn.Linear(n_dims, n_targets)

    def forward(self, x):
        y = self.linear(x)
        return (torch.sigmoid(y), (x, y))

    def _get_random_inputs(self, device: str):
        return (torch.randn(1, 5).to(device),)

    config = MakeConfig(download=False, to_tuple=False)


class Neuron2Inputs(torch.nn.Module):
    def __init__(self, n_dims: int = 5, n_targets: int = 3):
        super().__init__()
        self.linear = torch.nn.Linear(n_dims, n_targets)

    def forward(self, x, y):
        z = self.linear(x) + y
        return torch.sigmoid(z)

    def _get_random_inputs(self, device: str):
        return (torch.randn(1, 5).to(device), torch.randn(1, 3).to(device))

    config = MakeConfig(download=False, to_tuple=False)


class NeuronNamed1(torch.nn.Module):
    def __init__(self, n_dims: int = 5, n_targets: int = 3):
        super().__init__()
        self.linear = torch.nn.Linear(n_dims, n_targets)

    def forward(self, input_x=None, input_y=None):
        if input_x is None:
            return torch.sigmoid(self.linear(input_y))
        return torch.sigmoid(self.linear(input_x))

    def _get_random_inputs(self, device: str):
        return (torch.randn(1, 5).to(device),)

    config = MakeConfig(download=False, to_tuple=False)


class NeuronNamed2(torch.nn.Module):
    def __init__(self, n_dims: int = 5, n_targets: int = 3):
        super().__init__()
        self.linear = torch.nn.Linear(n_dims, n_targets)

    def forward(self, input_x=None, input_y=None):
        if input_x is None:
            return torch.sigmoid(self.linear(input_y))
        return torch.sigmoid(self.linear(input_x))

    def _get_random_inputs(self, device: str):
        return (
            None,
            torch.randn(1, 5).to(device),
        )

    config = MakeConfig(download=False, to_tuple=False)


class NeuronNamedDict(torch.nn.Module):
    def __init__(self, n_dims: int = 5, n_targets: int = 3):
        super().__init__()
        self.linear = torch.nn.Linear(n_dims, n_targets)

    def forward(self, input_x=None, input_y=None):
        if input_x is None:
            return torch.sigmoid(self.linear(input_y))
        return torch.sigmoid(self.linear(input_x))

    def _get_random_inputs(self, device: str):
        return {"input_y": torch.randn(1, 5).to(device)}

    config = MakeConfig(download=False, to_tuple=False)


class NeuronIList(torch.nn.Module):
    def __init__(self, n_dims: int = 5, n_targets: int = 3):
        super().__init__()
        self.linear = torch.nn.Linear(n_dims, n_targets)

    def forward(self, x, yz):
        z = self.linear(x + yz[0] + yz[1])
        return torch.sigmoid(z)

    def _get_random_inputs(self, device: str):
        return (
            torch.randn(1, 5).to(device),
            [torch.randn(1, 5).to(device), torch.randn(1, 5).to(device)],
        )

    config = MakeConfig(download=False, to_tuple=False)
