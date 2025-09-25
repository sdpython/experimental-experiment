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


def get_speech2text2_causal_ml_not_trained_model() -> Tuple[Callable, Tuple[Any, ...]]:
    """
    Returns a model.
    See `Speech2Text2Config
    <https://huggingface.co/docs/transformers/model_doc/speech_to_text_2.Speech2Text2Config>`_.
    """
    from transformers import Speech2Text2Config, Speech2Text2ForCausalLM

    config = Speech2Text2Config()
    example = torch.tensor(
        [
            [
                0,
                145,
                336,
                147,
                147,
                175,
                145,
                145,
                3,
                7738,
                144,
                336,
                161,
                131,
                531,
                160,
                175,
                7738,
                114,
                160,
                7464,
                2221,
                117,
                216,
                160,
                9469,
                216,
                9764,
                531,
                9570,
                130,
                531,
                114,
                160,
                162,
                7738,
                114,
                147,
                9161,
                114,
                9469,
                175,
                9348,
                144,
                114,
                336,
                147,
                131,
                336,
                147,
                130,
                7738,
                114,
                147,
                9161,
                166,
                114,
                117,
                216,
                147,
                3,
                7738,
                175,
                1938,
                4626,
                531,
                336,
                117,
                336,
                131,
                7464,
                336,
                162,
                473,
                145,
                145,
                7738,
                114,
                160,
                7464,
                114,
                7738,
                147,
                114,
                131,
                336,
                216,
                147,
                114,
                9465,
                114,
                7738,
                2221,
                312,
                336,
                147,
                130,
                1932,
                144,
                216,
                175,
                9348,
                166,
                336,
                117,
                131,
                175,
                9094,
                115,
                336,
                160,
                78,
                175,
                9469,
                139,
                216,
                117,
                131,
                175,
                160,
                3,
                7738,
                145,
                114,
                147,
                162,
                117,
                161,
                114,
                144,
                175,
                7738,
                117,
                166,
                336,
                145,
                7464,
                114,
                9469,
                216,
                147,
                7464,
                166,
                531,
                161,
                9388,
                336,
                9258,
                131,
                141,
                7464,
                117,
                114,
                166,
                7464,
                136,
                114,
                9767,
                131,
                141,
                114,
                9469,
                166,
                336,
                117,
                131,
                175,
                9094,
                161,
                114,
                160,
                78,
                175,
                9094,
                5025,
                175,
                9161,
                131,
                1932,
                139,
                145,
                114,
                117,
                9388,
                141,
                336,
                7738,
                131,
                175,
                175,
                131,
                9388,
                114,
                147,
                9161,
                166,
                336,
                117,
                131,
                175,
                9094,
                312,
                216,
                141,
                9258,
                161,
                216,
                145,
                145,
                336,
                175,
                9094,
                130,
                336,
                293,
                175,
                7738,
                141,
                336,
                7738,
                117,
                336,
                131,
                131,
                175,
                9094,
                2221,
                161,
                141,
                175,
                175,
                160,
                139,
                531,
                9465,
                117,
                145,
                114,
                9570,
                216,
                9258,
                131,
                141,
                7464,
                115,
                114,
                161,
                9498,
                115,
                175,
                139,
                216,
                160,
                7464,
                141,
                7464,
                117,
                114,
                473,
                7738,
                145,
                336,
                78,
                7464,
                2221,
                117,
                141,
                114,
                166,
                144,
                216,
                216,
                175,
                9094,
                336,
                9258,
                2221,
                131,
                531,
                160,
                78,
                336,
                117,
                9388,
                115,
                114,
                131,
                9388,
                147,
                175,
                1938,
                9469,
                166,
                114,
            ]
        ],
        dtype=torch.int64,
    )
    return (lambda: Speech2Text2ForCausalLM(config)), (example,)
