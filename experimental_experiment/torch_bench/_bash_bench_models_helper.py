import random
from typing import Any, Callable, Tuple
import torch


def get_dummy_model() -> Tuple[Callable, Tuple[Any, ...]]:
    """Returns a dummy model used to validate the command line."""

    class Neuron(torch.nn.Module):
        def __init__(self, n_dims: int = 5, n_targets: int = 3):
            super().__init__()
            self.linear = torch.nn.Linear(n_dims, n_targets)

        def forward(self, x):
            return torch.sigmoid(self.linear(x))

    return Neuron, (torch.randn(2, 5),)


def get_dummy_model_fail() -> Tuple[Callable, Tuple[Any, ...]]:
    """Returns a dummy model failing."""

    class NeuronFail(torch.nn.Module):
        def __init__(self, n_dims: int = 5, n_targets: int = 3):
            super().__init__()
            self.linear = torch.nn.Linear(n_dims, n_targets)

        def forward(self, x):
            return torch.sigmoid(self.linear(x) * torch.Tensor([0, 1]))

    return NeuronFail, (torch.randn(1, 5),)


def get_dummy_model_fail_convert() -> Tuple[Callable, Tuple[Any, ...]]:
    """Returns a dummy model."""

    class NeuronF(torch.nn.Module):
        def __init__(self, n_dims: int = 5, n_targets: int = 3):
            super().__init__()
            self.linear = torch.nn.Linear(n_dims, n_targets)

        def forward(self, x):
            return torch.sigmoid(self.linear(x))

    return NeuronF, (torch.randn(1, 5),)


def ids_tensor(shape, vocab_size, rng=None, name=None):
    """Creates a random int32 tensor of the shape within the vocab size."""
    if rng is None:
        rng = random.Random()

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.randint(0, vocab_size - 1))

    return torch.tensor(data=values, dtype=torch.long).view(shape).contiguous()


def get_llama_model_layer(
    num_hidden_layers: int = 1,
) -> Tuple[Callable, Tuple[Any, ...]]:
    """Returns a llama model with a specific number of layers."""
    import torch
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaModel

    vocab_size = 32000
    config = LlamaConfig(
        num_hidden_layers=num_hidden_layers,
        hidden_size=4096,
        vocab_size=vocab_size,
        intermediate_size=11008,
        max_position_embeddings=2048,
        num_attention_heads=32,
    )
    config._attn_implementation = "eager"

    class LlamaModelWrapper(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.model = LlamaModel(config)

        def forward(self, input_ids, attention_mask):
            model_output = self.model(input_ids, attention_mask=attention_mask, use_cache=False)
            return model_output.to_tuple()

    def generate_example_inputs(batch: int, seq: int, vocab_size: int):
        input_ids = ids_tensor([batch, seq], vocab_size)
        input_mask = torch.tril(torch.ones(batch, seq, dtype=torch.float32))
        assert input_mask.dtype == torch.float32
        return input_ids, input_mask

    shape = (2, 1024)
    return (lambda: LlamaModelWrapper(config)), generate_example_inputs(
        shape[0], shape[1], vocab_size
    )
