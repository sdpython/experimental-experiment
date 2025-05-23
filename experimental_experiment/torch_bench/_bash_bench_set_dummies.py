import torch
from onnx_diagnostic.helpers.cache_helper import make_dynamic_cache
from ._bash_bench_model_runner import MakeConfig


class Neuron(torch.nn.Module):
    "Dummy module with one input."

    def __init__(self, n_dims: int = 5, n_targets: int = 3):
        super().__init__()
        self.linear = torch.nn.Linear(n_dims, n_targets)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

    def _get_random_inputs(self, device: str):
        return (torch.randn(2, 5).to(device),)

    config = MakeConfig(download=False, to_tuple=False)


class Neuron2Outputs(torch.nn.Module):
    "Dummy module with 2 outputs."

    def __init__(self, n_dims: int = 1000, n_targets: int = 100):
        super().__init__()
        self.linear = torch.nn.Linear(n_dims, n_targets)

    def forward(self, x):
        return torch.sigmoid(self.linear(x + 10)), torch.softmax(self.linear(x), dim=1)

    def _get_random_inputs(self, device: str):
        return (torch.randn(1, 1000).to(device),)

    config = MakeConfig(download=False, to_tuple=False)


class Neuron16(Neuron):
    "Dummy module with 1 input in float16."

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
    "Dummy module with a tuple as input."

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
    "Dummy module with 2 inputs."

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
    "Dummy module with named inputs."

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
    "Dummy module with named inputs and none inputs."

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
    "Dummy module with as dictionary as input."

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
    "Dummy module with a list as input."

    def __init__(self, n_dims: int = 5, n_targets: int = 3):
        super().__init__()
        self.linear = torch.nn.Linear(n_dims, n_targets)

    def forward(self, x, yz):
        z = self.linear(x + yz[0] * yz[1])
        return torch.sigmoid(z)

    def _get_random_inputs(self, device: str):
        return (
            torch.randn(1, 5).to(device),
            [torch.randn(1, 5).to(device), torch.randn(1, 5).to(device)],
        )

    config = MakeConfig(download=False, to_tuple=False)


class NeuronIInt(torch.nn.Module):
    "Dummy module with an integer."

    def __init__(self, n_dims: int = 5, n_targets: int = 3):
        super().__init__()
        self.linear = torch.nn.Linear(n_dims, n_targets)

    def forward(self, x, i_input):
        z = self.linear(x)
        return torch.sigmoid(z)[:, i_input]

    def _get_random_inputs(self, device: str):
        return (torch.randn(1, 5).to(device), 2)

    config = MakeConfig(download=False, to_tuple=False)


class NeuronNoneInt(torch.nn.Module):
    "Dummy module with an empty input and an integer as input."

    def __init__(self, n_dims: int = 5, n_targets: int = 3):
        super().__init__()
        self.linear = torch.nn.Linear(n_dims, n_targets)

    def forward(self, x, empty_input, i_input):
        z = self.linear(x)
        return torch.sigmoid(z)[:, i_input]

    def _get_random_inputs(self, device: str):
        return (torch.randn(1, 5).to(device), None, 2)

    config = MakeConfig(download=False, to_tuple=False)


class NeuronNoneListInt(torch.nn.Module):
    "Dummy module with a list and an integrer as inputs."

    def __init__(self, n_dims: int = 5, n_targets: int = 3):
        super().__init__()
        self.linear = torch.nn.Linear(n_dims, n_targets)

    def forward(self, x, yz, i_input):
        z = self.linear(x + yz[0] * yz[3])
        return torch.sigmoid(z)[:i_input]

    def _get_random_inputs(self, device: str):
        return (
            torch.randn(1, 5).to(device),
            [torch.randn(1, 5).to(device), None, None, torch.randn(1, 5).to(device)],
            2,
        )

    config = MakeConfig(download=False, to_tuple=False)


class NeuronNoneIntDefault(torch.nn.Module):
    "Dummy module with an optional integer and a list as inputs."

    def __init__(self, n_dims: int = 5, n_targets: int = 3):
        super().__init__()
        self.linear = torch.nn.Linear(n_dims, n_targets)

    def forward(self, x, empty_input, i_input=10):
        z = self.linear(x)
        return torch.sigmoid(z)[:, i_input]

    def _get_random_inputs(self, device: str):
        return (torch.randn(1, 5).to(device), None, 2)

    config = MakeConfig(download=False, to_tuple=False)


class NeuronNoneIntDict(torch.nn.Module):
    "Dummy module with an optional integer and dictionary as inputs."

    def __init__(self, n_dims: int = 5, n_targets: int = 3):
        super().__init__()
        self.linear = torch.nn.Linear(n_dims, n_targets)

    def forward(self, x, empty_input=None, i_input=2):
        z = self.linear(x)
        return torch.sigmoid(z)[:, i_input]

    def _get_random_inputs(self, device: str):
        return {"x": torch.randn(1, 5).to(device)}

    config = MakeConfig(download=False, to_tuple=False)


class NeuronDynamicCache(torch.nn.Module):
    "Dummy module with a :class:`transformers.cache_utils.DynamicCache`."

    def forward(self, x, dc):
        return x @ (
            (torch.cat(dc.key_cache, axis=1) + torch.cat(dc.value_cache, axis=1)).reshape(
                (-1, x.shape[1])
            )
        ).transpose(1, 0)

    def _get_random_inputs(self, device: str):
        cache = make_dynamic_cache(
            [(torch.ones((3, 8, 3, 8)).to(device), (torch.ones((3, 8, 3, 8)) * 2).to(device))]
        )
        return {"x": torch.randn(3, 8, 3, 8).to(device), "dc": cache}

    config = MakeConfig(download=False, to_tuple=False)


class NeuronMambaCache(torch.nn.Module):
    "Dummy module with a :class:`transformers.cache_utils.MambaCache`."

    def forward(self, x, dc):
        assert (
            x.dtype == dc.conv_states[0].dtype and x.dtype == dc.ssm_states[0].dtype
        ), f"dtypes are {x.dtype}, {dc.conv_states[0].dtype}, {dc.ssm_states[0].dtype}"
        return x @ (torch.cat([*dc.conv_states, *dc.ssm_states], axis=-1))

    def _get_random_inputs(self, device: str):
        import transformers

        class _config:
            def __init__(self):
                self.intermediate_size = 8
                self.state_size = 16
                self.conv_kernel = 32
                self.num_hidden_layers = 64
                self.dtype = torch.float32

        cache = transformers.cache_utils.MambaCache(_config(), max_batch_size=1, device="cpu")
        cache.conv_states[0] += 1
        cache.ssm_states[0] += 2
        if isinstance(cache.conv_states, list):
            cache.conv_states = [t.to(device).to(torch.float32) for t in cache.conv_states]
            cache.ssm_states = [t.to(device).to(torch.float32) for t in cache.ssm_states]
        else:
            cache.conv_states = cache.conv_states.to(device).to(torch.float32)
            cache.ssm_states = cache.ssm_states.to(device).to(torch.float32)
        return {"x": torch.randn(1, 1, 3, 8).to(device), "dc": cache}

    config = MakeConfig(download=False, to_tuple=False)
