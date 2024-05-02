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

.. code-block:: python

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

Llama 3
=======

See `Llama3 <https://huggingface.co/docs/transformers/main/en/model_doc/llama3>`_.

.. code-block:: python

    import os
    import time
    import onnxruntime
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from experimental_experiment.torch_interpreter import to_onnx

    model_id = "meta-llama/Meta-Llama-3-8B"

    with torch.no_grad():
        model = AutoModelForCausalLM.from_pretrained(model_id).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        base_prompt = "Is the conversion to onnx going to work?"
        base_inputs = tokenizer(base_prompt, return_tensors="pt")  # .to("cpu")
        input_ids = base_inputs.input_ids
        expected = model(input_ids)

        print(f"output type: {type(expected)}")
        print(f"logits: {expected.logits.shape}, {expected.logits.dtype}")

        print(
            "start conversion... with input_ids", input_ids.dtype, input_ids.shape
        )
        begin = time.perf_counter()
        large_onx = to_onnx(
            model,
            (input_ids,),
            input_names=["x"],
            verbose=1,
            large_model=True,
            # dynamic_shapes fails with transformers==4.37.2
            # TypeError: scaled_dot_product_attention(): argument 'is_causal' must be bool, not SymBool
            # dynamic_shapes={"x": {1: torch.export.Dim("length", min=2)}},
        )
        duration = time.perf_counter() - begin
        print(f"conversion done in {duration}s")

    folder = "test_zoo_export_llama3"
    if not os.path.exists(folder):
        os.mkdir(folder)
    else:
        for _ in os.listdir(folder):
            os.remove(os.path.join(folder, _))

    print(f"start saving in {folder!r}")
    begin = time.perf_counter()
    filename = os.path.join(folder, "llama3.onnx")
    large_onx.save(filename)
    duration = time.perf_counter() - begin
    print(f"saving done in {duration}s with {len(os.listdir(folder))} files")

    print(f"loading model {filename!r} with onnxruntime.")
    begin = time.perf_counter()
    sess = onnxruntime.InferenceSession(
        filename, providers=["CPUExecutionProvider"]
    )
    print(f"done in {time.perf_counter() - begin}s")

    print("running the first iteration")
    begin = time.perf_counter()
    name = large_onx.model_proto.graph.input[0].name
    np_input = input_ids.detach().cpu().numpy()
    got = sess.run(None, {name: np_input})
    print(f"done in {time.perf_counter() - begin}s")
    self.assertEqualArray(expected.logits, got[0], atol=1e-4)

    N = 5
    print(f"running {N} iterations with torch")
    begin = time.perf_counter()
    for i in range(N):
        model(input_ids)
    d = time.perf_counter() - begin
    print(f"done in {d}s for torch")

    print(f"running {N} iterations with onnxruntime")
    begin = time.perf_counter()
    for i in range(N):
        sess.run(None, {name: np_input})
    d = time.perf_counter() - begin
    print(f"done in {d}s for onnxruntime")
