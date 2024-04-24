=====
LLaMa
=====

Dummy Example
=============

`LLaMa <https://huggingface.co/docs/transformers/en/model_doc/llama>`_

.. runpython::
    :showcode:

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

    with torch.no_grad():
    
        model = LlamaModel(config)

        batch, seq, vocab_size = 2, 1024, 1024

        input_ids = ids_tensor([batch, seq], vocab_size)
        input_mask = torch.tril(torch.ones(batch, seq, dtype=torch.float32))

        model(input_ids, input_mask)

        onx = to_onnx(model, (input_ids, input_mask))
        print(onnx_simple_text_plot(onx))

Full Example
============

::

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    location = "meta-llama/Llama-2-7b-hf"
    cahce_dir = "_cache"
    l_config = AutoConfig.from_pretrained(
        location, use_auth_token=use_auth_token, cache_dir=cache_dir
    )
    l_config.use_cache = True
    llama = AutoModelForCausalLM.from_pretrained(
        location,
        use_auth_token=use_auth_token,
        config=l_config,
        torch_dtype=torch.float32,
        cache_dir=cache_dir=cache_dir,
    )
