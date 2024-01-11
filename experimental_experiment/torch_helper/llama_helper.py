from typing import Sequence, Tuple


def get_llama_decoder(
    input_dims: Sequence[Tuple[int, int]] = ((2, 8), (4, 7), (9, 15)),
    hidden_size=16,
    num_hidden_layers=1,
    vocab_size=1024,
    intermediate_size=16,
    max_position_embeddings=256,
    num_attention_heads=2,
    hidden_dropout_prob=0.0,
    attention_dropout_prob=0.0,
    same_shape: bool = False,
):
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
        hidden_dropout_prob=hidden_dropout_prob,
        attention_dropout_prob=attention_dropout_prob,
    )

    class LlamaDecoderWrapper(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            try:
                self.decoder = LlamaDecoderLayer(config, layer_idx=0)
            except TypeError:
                self.decoder = LlamaDecoderLayer(config)

        def forward(self, hidden_states, attention_mask, position_ids):
            (decoder_output,) = self.decoder(
                hidden_states, attention_mask, position_ids
            )
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
