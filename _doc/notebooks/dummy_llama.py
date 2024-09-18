import pprint
import time
import numpy as np
import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaModel
from experimental_experiment.xbuilder import OptimizationOptions
from experimental_experiment.torch_dynamo import onnx_custom_backend


def ids_tensor(shape, vocab_size):
    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(np.random.randint(0, vocab_size - 1))

    return torch.tensor(data=values, dtype=torch.long).view(shape).contiguous()


batch, seq, vocab_size = 2, 1025, 1024

config = LlamaConfig(
    hidden_size=4096,
    num_hidden_layers=1,
    vocab_size=vocab_size,  # 32000,
    intermediate_size=1024,  # 11008,
    max_position_embeddings=2048,
    num_attention_heads=32,
)
config._attn_implementation = "eager"

N = 10

with torch.no_grad():

    model = LlamaModel(config)

    input_ids = ids_tensor([batch, seq], vocab_size)
    input_mask = torch.tril(torch.ones(batch, seq, dtype=torch.float32))

    # warmup
    times = []
    print("warmup eager")
    for _ in range(3):
        # model(input_ids, input_mask)
        model(input_ids)

    # repeat
    print("repeat eager")
    begin = time.perf_counter()
    for _ in range(N):
        # model(input_ids, input_mask)
        model(input_ids)
    d = (time.perf_counter() - begin) / N
    times.append(("eager", d))
    print("avg time eager", d)

    for optim in [""]:  # , "default+onnxruntime", "default"]:
        options = OptimizationOptions(
            constant_folding=True,
            patterns=None if optim == "" else optim,
            verbose=0,
            processor="CUDA",
        )

        custom_custom_backend = (  # noqa: E731
            lambda *args, optim=optim, options=options, **kwargs: onnx_custom_backend(
                *args,
                target_opset=18,
                verbose=0,
                options=options,
                optimize=optim != "",
                dump_prefix=f"dump_onx_llama_{optim.replace('+', '_')}",
                **kwargs,
            )
        )

        compiled_model = torch.compile(
            model, backend=custom_custom_backend, fullgraph=True, dynamic=False
        )

        # warmup
        print("warmup compiled model")
        for _ in range(3):
            # compiled_model(input_ids, input_mask)
            compiled_model(input_ids)

        # repeat
        print("repeat compiled_model")
        begin = time.perf_counter()
        for _ in range(N):
            # compiled_model(input_ids, input_mask)
            compiled_model(input_ids)
        d = (time.perf_counter() - begin) / N
        times.append((optim, d))
        print(f"avg time custom backend with optimization={optim!r}", d)

print("-----------------")
pprint.pprint(times)
