import random
from typing import Any, Sequence, Tuple


def has_phi3() -> bool:
    """Tells if package :epkg:`transformers` is recent enough."""
    try:
        from transformers import Phi3Config

        assert Phi3Config
        return True
    except ImportError:
        return False


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


def get_phi3_model(
    input_dims: Sequence[Tuple[int, int]] = ((13, 7), (14, 7), (15, 8)),
    hidden_size=32,
    num_hidden_layers=2,
    vocab_size=99,
    intermediate_size=16,
    max_position_embeddings=512,
    num_attention_heads=4,
    num_key_value_heads=2,
    _attn_implementation="eager",  # needed value to remove graph breaks
    with_mask: bool = True,
):
    """
    Returns a model.
    See `PhiConfig
    <https://huggingface.co/docs/transformers/main/en/model_doc/phi#transformers.PhiConfig>`_.
    The parameters are chosen for a unit test configuration from `test_modeling_phi.py
    <https://github.com/huggingface/transformers/blob/main/tests/models/phi/test_modeling_phi.py>`_.
    """
    import torch
    from transformers import Phi3Config, Phi3Model

    config = Phi3Config(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        vocab_size=vocab_size,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        pad_token_id=min(32000, vocab_size - 1),
    )
    if _attn_implementation:
        config._attn_implementation = _attn_implementation

    if with_mask:

        class Phi3ModelWrapper(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                self.model = Phi3Model(config)

            def forward(self, input_ids, attention_mask):
                model_output = self.model(
                    input_ids, attention_mask=attention_mask, use_cache=False
                )
                return model_output.to_tuple()

        def generate_example_inputs(batch: int, seq: int, vocab_size: int):
            (
                input_ids,
                _token_type_ids,
                input_mask,
                _sequence_labels,
                _token_labels,
                _choice_labels,
            ) = _prepare_config_and_inputs(
                batch_size=batch,
                seq_length=seq,
                vocab_size=vocab_size,
                use_input_mask=True,
            )
            return input_ids, input_mask

        example_args_collection = []
        for b, s in input_dims:
            example_args_collection.append(generate_example_inputs(b, s, vocab_size))

        return Phi3ModelWrapper(config), example_args_collection

    # no mask

    class Phi3ModelWrapper(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.model = Phi3Model(config)

        def forward(self, input_ids):
            model_output = self.model(input_ids, use_cache=False)
            return model_output.to_tuple()

    def generate_example_inputs(batch: int, seq: int, vocab_size: int):
        (
            input_ids,
            _token_type_ids,
            _input_mask,
            _sequence_labels,
            _token_labels,
            _choice_labels,
        ) = _prepare_config_and_inputs(
            batch_size=batch,
            seq_length=seq,
            vocab_size=vocab_size,
            use_input_mask=True,
        )
        return (input_ids,)

    example_args_collection = []
    for b, s in input_dims:
        example_args_collection.append(generate_example_inputs(b, s, vocab_size))

    return Phi3ModelWrapper(config), example_args_collection
