"""
Code modified from different sources:

* https://github.com/huggingface/transformers/blob/main/tests/models/llama/test_modeling_llama.py
* https://github.com/pytorch/pytorch/pull/117009
"""

import random
from typing import Sequence, Tuple


def get_llama_decoder(
    input_dims: Sequence[Tuple[int, int]] = ((2, 8), (4, 7), (9, 15)),
    hidden_size=16,
    num_hidden_layers=1,
    vocab_size=1024,
    intermediate_size=16,
    max_position_embeddings=1024,
    num_attention_heads=2,
    _attn_implementation="eager",
):
    """
    Returns the decoder part.
    See :func:`experimental_experiment.torch_models.llama_helper.get_llama_model`.
    """
    import torch
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer

    config = LlamaConfig(
        num_hidden_layers=num_hidden_layers,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings,
        num_attention_heads=num_attention_heads,
    )
    if _attn_implementation:
        config._attn_implementation = _attn_implementation

    class LlamaDecoderWrapper(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.decoder = LlamaDecoderLayer(config, layer_idx=0)

        def forward(self, hidden_states, attention_mask, position_ids):
            (decoder_output,) = self.decoder(hidden_states, attention_mask, position_ids)
            return decoder_output

    def generate_example_inputs(batch: int, seq: int, hidden_size: int):
        # shape: batch x seq x hidden_size
        hidden_state = torch.randn(batch, seq, hidden_size)
        attention_mask = torch.zeros(batch, 1, seq, seq, dtype=torch.float)
        position_ids = torch.arange(0, seq, dtype=torch.int64)
        position_ids = position_ids.unsqueeze(0).view(-1, seq)
        return hidden_state, attention_mask, position_ids

    example_args_collection = []
    for b, s in input_dims:
        example_args_collection.append(generate_example_inputs(b, s, hidden_size))

    return LlamaDecoderWrapper(config), example_args_collection


def get_llama_attention(
    input_dims: Sequence[Tuple[int, int]] = ((2, 8), (4, 7), (9, 15)),
    hidden_size=16,
    num_hidden_layers=1,
    vocab_size=1024,
    intermediate_size=16,
    max_position_embeddings=1024,
    num_attention_heads=2,
    _attn_implementation="eager",
):
    """
    Returns the attention part.
    See :func:`experimental_experiment.torch_models.llama_helper.get_llama_model`.
    """
    import torch
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaAttention

    config = LlamaConfig(
        num_hidden_layers=num_hidden_layers,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings,
        num_attention_heads=num_attention_heads,
    )
    if _attn_implementation:
        config._attn_implementation = _attn_implementation

    class LlamaAttentionWrapper(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.attention = LlamaAttention(config, layer_idx=0)

        def forward(self, hidden_states, attention_mask, position_ids):
            attn_output, _, _ = self.attention(hidden_states, attention_mask, position_ids)
            return attn_output

    def generate_example_inputs(batch: int, seq: int, hidden_size: int):
        hidden_state = torch.randn(batch, seq, hidden_size)
        attention_mask = torch.zeros(batch, 1, seq, seq, dtype=torch.float)
        position_ids = torch.arange(0, seq, dtype=torch.int64)
        position_ids = position_ids.unsqueeze(0).view(-1, seq)

        return hidden_state, attention_mask, position_ids

    example_args_collection = []
    for b, s in input_dims:
        example_args_collection.append(generate_example_inputs(b, s, hidden_size))

    return LlamaAttentionWrapper(config), example_args_collection


def ids_tensor(shape, vocab_size, rng=None, name=None):
    #  Creates a random int32 tensor of the shape within the vocab size
    import torch

    if rng is None:
        rng = random.Random()

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.randint(0, vocab_size - 1))

    return torch.tensor(data=values, dtype=torch.long).view(shape).contiguous()


def get_llama_model(
    input_dims: Sequence[Tuple[int, int]] = ((2, 8), (4, 7), (9, 15)),
    hidden_size: int = 16,
    num_hidden_layers: int = 1,
    vocab_size: int = 1024,
    intermediate_size: int = 16,
    max_position_embeddings: int = 1024,
    num_attention_heads: int = 2,
    _attn_implementation: str = "eager",  # needed value to remove graph breaks
    with_mask: bool = True,
    dynamic_shapes: bool = False,
):
    """
    Returns a model.
    See `LlamaConfig
    <https://huggingface.co/docs/transformers/main/en/model_doc/llama#transformers.LlamaConfig>`_.
    The parameters are chosen for a unit test configuration.
    """
    import torch
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaModel

    _dynamic_shapes = {0: {0: "batch", 1: "length"}}
    if with_mask:
        _dynamic_shapes.update({1: {0: "batch", 1: "length"}})

    config = LlamaConfig(
        num_hidden_layers=num_hidden_layers,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings,
        num_attention_heads=num_attention_heads,
    )
    if _attn_implementation:
        config._attn_implementation = _attn_implementation

    if with_mask:

        class LlamaModelWrapper(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                self.model = LlamaModel(config)

            def forward(self, input_ids, attention_mask):
                model_output = self.model(
                    input_ids, attention_mask=attention_mask, use_cache=False
                )
                return model_output.to_tuple()

        def generate_example_inputs(batch: int, seq: int, vocab_size: int):
            input_ids = ids_tensor([batch, seq], vocab_size)
            input_mask = torch.tril(torch.ones(batch, seq, dtype=torch.float32))
            assert input_mask.dtype == torch.float32
            return input_ids, input_mask

        example_args_collection = []
        for b, s in input_dims:
            example_args_collection.append(generate_example_inputs(b, s, vocab_size))

        if not dynamic_shapes:
            return LlamaModelWrapper(config), example_args_collection
        return LlamaModelWrapper(config), example_args_collection, _dynamic_shapes

    # no mask

    class LlamaModelWrapper(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.model = LlamaModel(config)

        def forward(self, input_ids):
            model_output = self.model(input_ids, use_cache=False)
            return model_output.to_tuple()

    def generate_example_inputs(batch: int, seq: int, vocab_size: int):
        input_ids = ids_tensor([batch, seq], vocab_size)
        return (input_ids,)

    example_args_collection = []
    for b, s in input_dims:
        example_args_collection.append(generate_example_inputs(b, s, vocab_size))

    if not dynamic_shapes:
        return LlamaModelWrapper(config), example_args_collection
    return LlamaModelWrapper(config), example_args_collection, _dynamic_shapes
