
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

.. runpython::
    :showcode:
    :process:

    import os
    import warnings
    from typing import List
    import numpy as np
    import onnx
    # from onnx_array_api.plotting.text_plot import onnx_simple_text_plot
    import torch
    import torch.onnx
    import onnxscript
    from experimental_experiment.torch_helper.training_helper import (
        make_aot_ort,
        train_loop,
    )
    from experimental_experiment.torch_helper.dump_helper import dump_onnx
    # from experimental_experiment.torch_interpreter import to_onnx

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from transformers import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaModel


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
    config._attn_implementation = "sdpa"

    model = LlamaModel(config)

    batch, seq, vocab_size = 2, 1024, 1024

    input_ids = ids_tensor([batch, seq], vocab_size)
    input_mask = torch.tril(torch.ones(batch, seq, dtype=torch.float32))

    model(input_ids, input_mask)

    # onx = to_onnx(model, (input_ids, input_mask))
    # print(onnx_simple_text_plot(onx))


    op = onnxscript.opset18
    aten_opset = onnxscript.values.Opset("aten", 1)


    @onnxscript.script(aten_opset, default_opset=op)
    def scaled_dot_product_efficient_attention(
        query,
        key,
        value,
        attn_bias,
        compute_log_sumexp: bool,
        dropout_p: float,
        is_causal: bool,
    ):
        output, log_sumexp, philox_seed, philox_offset = aten_opset.ATen(
            query,
            key,
            value,
            attn_bias,
            compute_log_sumexp,
            dropout_p,
            is_causal,
            1.0,
            operator="_scaled_dot_product_efficient_attention",
        )
        return output, log_sumexp, philox_seed, philox_offset


    @onnxscript.script(aten_opset, default_opset=op)
    def scaled_dot_product_attention_backward(
        grad,
        query,
        key,
        value,
        attn_bias,
        output,
        logsumexp,
        philox_seed,
        philox_offset,
        dropout_p,
        grad_input_mask: List[bool],
        is_causal: bool,
    ):
        grad_query, grad_key, grad_value, grad_attn_bias = aten_opset.ATen(
            grad,
            query,
            key,
            value,
            attn_bias,
            output,
            logsumexp,
            philox_seed,
            philox_offset,
            dropout_p,
            grad_input_mask,
            is_causal,
            1.0,
            operator="_scaled_dot_product_efficient_attention_backward",
        )
        return grad_query, grad_key, grad_value, grad_attn_bias


    aten_conversion_changes = {
        (
            scaled_dot_product_efficient_attention,
            "_scaled_dot_product_efficient_attention"
        ),
        (
            scaled_dot_product_attention_backward,
            "_scaled_dot_product_efficient_attention_backward",
        ),
    }

    local_aot_ort, _ = make_aot_ort(
        dynamic=True,
        rewrite=True,
        aten_conversion_changes=aten_conversion_changes,
        verbose=1,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        optimized_mod = torch.compile(model, backend=local_aot_ort, fullgraph=True)
        with dump_onnx("dort-llama-ort", folder="dump_llama", clean=True):
            train_loop(optimized_mod, input_ids, input_mask)

    names = [_ for _ in os.listdir("dump_llama") if _.endswith(".onnx")]
    print("------------------------------------------")
    print(f"exported model: {names}")
    for name in names:
        print()
        print("NODES in {name!r}")
        onx = onnx.load(os.path.join("dump_llama", name))
        for i, node in enumerate(onx.graph.node):
            print(
                f"{i+1}/{len(onx.graph.node)}: "
                f"{node.op_type} {node.input} -> {node.output}"
            )

