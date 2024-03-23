
=======================
Tries with Undocumented
=======================

Example about torch._dynamo.export
==================================

.. runpython::
    :showcode:
    :process:

    import numpy as np
    from onnx_array_api.plotting.text_plot import onnx_simple_text_plot
    import torch
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaModel
    from experimental_experiment.torch_interpreter import to_onnx


    def ids_tensor(shape, vocab_size):
        total_dims = 1
        for dim in shape:
            total_dims *= dim

        values = []
        for _ in range(total_dims):
            values.append(np.random.randint(0, vocab_size - 1))

        return torch.tensor(data=values, dtype=torch.long).view(shape).contiguous()


    config = LlamaConfig(
        hidden_size=16,
        num_hidden_layers=1,
        vocab_size=1024,
        intermediate_size=16,
        max_position_embeddings=1024,
        num_attention_heads=2,
    )
    config._attn_implementation = "eager"

    model = LlamaModel(config)

    batch, seq, vocab_size = 2, 1024, 1024

    input_ids = ids_tensor([batch, seq], vocab_size)
    input_mask = torch.tril(torch.ones(batch, seq, dtype=torch.float32))

    model(input_ids, input_mask)

    from torch.export.dynamic_shapes import (
        _process_constraints,
        _process_dynamic_shapes,
        Constraint,
        dims,
        dynamic_dim,
    )

    args = input_ids, input_mask

    constraints = _process_dynamic_shapes(model, args, {}, None)
    print(constraints)

    gm, _ = torch._dynamo.export(
        model,
        aten_graph=True,
        tracing_mode="symbolic",
        decomposition_table={},
        constraints=constraints,
    )(*args)

    print(gm.graph)

Example about custom ops in onnxrt
==================================

Look into unit test file
`test_custom_ops.py <https://github.com/sdpython/experimental-experiment/blob/main/_unittests/ut_onnx_script/test_custom_ops.py>_`.
