import random
from typing import Any, Sequence, Tuple


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


def _prepare_config_and_inputs(
    batch_size: int,
    seq_length: int,
    vocab_size: int,
    type_sequence_label_size: int = 2,
    type_vocab_size: int = 16,
    num_labels: int = 3,
    num_choices: int = 4,
    use_input_mask: bool = False,
    use_token_type_ids: bool = False,
    use_labels: bool = False,
) -> Tuple[Any]:
    import torch

    input_ids = ids_tensor([batch_size, seq_length], vocab_size)

    input_mask = None
    if use_input_mask:
        input_mask = torch.tril(torch.ones(batch_size, seq_length))

    token_type_ids = None
    if use_token_type_ids:
        assert type_vocab_size > 0, "type_vocab_size is null"
        token_type_ids = ids_tensor([batch_size, seq_length], type_vocab_size)

    sequence_labels = None
    token_labels = None
    choice_labels = None
    if use_labels:
        assert type_sequence_label_size > 0, "type_sequence_label_size is null"
        assert num_labels > 0, "num_labels is null"
        assert num_choices > 0, "num_choices is null"
        sequence_labels = ids_tensor([batch_size], type_sequence_label_size)
        token_labels = ids_tensor([batch_size, seq_length], num_labels)
        choice_labels = ids_tensor([batch_size], num_choices)

    return (
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    )


def get_mistral_model(
    input_dims: Sequence[Tuple[int, int]] = ((13, 7), (14, 7), (15, 8)),
    hidden_size=32,
    num_hidden_layers=2,
    vocab_size=99,
    intermediate_size=16,
    max_position_embeddings=512,
    num_attention_heads=2,
    num_key_value_heads=2,
    sliding_window=4096,
    _attn_implementation="eager",  # needed value to remove graph breaks
    with_mask: bool = True,
):
    """
    Returns a model.
    See `MistralConfig
    <https://huggingface.co/docs/transformers/main/en/model_doc/mistral#transformers.MistralConfig>`_.
    The parameters are chosen for a unit test configuration.
    """
    import torch
    from transformers import MistralConfig
    from transformers.models.mistral.modeling_mistral import MistralModel

    config = MistralConfig(
        num_hidden_layers=num_hidden_layers,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        sliding_window=sliding_window,
    )
    if _attn_implementation:
        config._attn_implementation = _attn_implementation

    def generate_example_inputs(batch: int, seq: int, vocab_size: int, with_mask: bool):
        (
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = _prepare_config_and_inputs(
            batch_size=batch,
            seq_length=seq,
            vocab_size=vocab_size,
            use_input_mask=with_mask,
        )
        if with_mask:
            return input_ids, input_mask
        return (input_ids,)

    if with_mask:

        class MistralModelWrapper(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                self.model = MistralModel(config)

            def forward(self, input_ids, attention_mask):
                model_output = self.model(input_ids, attention_mask=attention_mask)
                return model_output.to_tuple()

    else:

        class MistralModelWrapper(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                self.model = MistralModel(config)

            def forward(self, input_ids):
                model_output = self.model(input_ids)
                return model_output.to_tuple()

    example_args_collection = []
    for b, s in input_dims:
        example_args_collection.append(
            generate_example_inputs(b, s, vocab_size, with_mask)
        )

    return MistralModelWrapper(config), example_args_collection
