=======
Mistral
=======

`Mistral <https://huggingface.co/docs/transformers/en/model_doc/mistral>`_

.. runpython::
    :showcode:

    import numpy as np
    from onnx_array_api.plotting.text_plot import onnx_simple_text_plot
    import torch
    from transformers import MistralConfig
    from transformers.models.mistral.modeling_mistral import MistralModel
    from experimental_experiment.torch_interpreter import to_onnx


    def ids_tensor(shape, vocab_size):
        total_dims = 1
        for dim in shape:
            total_dims *= dim

        values = []
        for _ in range(total_dims):
            values.append(np.random.randint(0, vocab_size - 1))

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
    ):
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


    config = MistralConfig(
        hidden_size=32,
        num_hidden_layers=2,
        vocab_size=1024,
        intermediate_size=16,
        max_position_embeddings=512,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    config._attn_implementation = "eager"
    model = MistralModel(config)

    batch, seq, vocab_size = 2, 1024, 1024

    input_ids = ids_tensor([batch, seq], vocab_size)
    input_mask = torch.tril(torch.ones(batch, seq, dtype=torch.float32))

    model(input_ids, input_mask)

    onx = to_onnx(model, (input_ids, input_mask))
    print(onnx_simple_text_plot(onx))
