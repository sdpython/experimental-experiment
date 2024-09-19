"""
========================================
102: Fuse kernels in a small Llama Model
========================================

This example leverages the function :epkg:`torch.compile` and the ability
to use a custom backend to test the optimization of a model by fusing
simple element-wise kernels.

It takes a small Llama model and uses a backend based on :epkg:`onnxruntime`.
The model is converted into ONNX and then optimized by fusing element-wise
kernels.

::

    python plot_custom_backend_llama --config large
"""

from experimental_experiment.args import get_parsed_args

script_args = get_parsed_args(
    "plot_custom_backend_llama",
    config=("medium", "large or medium depending, large means closer to the real model"),
    num_hidden_layers=(1, "number of hidden layers"),
    with_mask=(0, "tries with a mask as a secondary input"),
    description=__doc__,
    expose="config,num_hidden_layers,with_mask",
)

print("config={script_args.config!r}")
print("num_hidden_layers={script_args.num_hidden_layers!r}")

#################################
# Imports.

import time
import numpy as np
import pandas
import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaModel
from experimental_experiment.xbuilder import OptimizationOptions
from experimental_experiment.torch_dynamo import onnx_custom_backend

########################################
# The dummy model
# ===============


def ids_tensor(shape, vocab_size):
    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(np.random.randint(0, vocab_size - 1))

    return torch.tensor(data=values, dtype=torch.long).view(shape).contiguous()


################################
# The size of the input.
if script_args.config == "large":
    batch, seq, vocab_size = 2, 1024, 32000
    intermediate_size = 11008
else:
    batch, seq, vocab_size = 2, 1024, 1024
    intermediate_size = 1024

################################
# The configuration of the model.

config = LlamaConfig(
    hidden_size=4096,
    num_hidden_layers=int(script_args.num_hidden_layers),
    vocab_size=vocab_size,
    intermediate_size=intermediate_size,
    max_position_embeddings=2048,
    num_attention_heads=32,
)
config._attn_implementation = "eager"

######################################
# The number of time we run the model to measure
# the inference.
N = 10

###########################################
# Let's create the model with dummy inputs.
model = LlamaModel(config)

input_ids = ids_tensor([batch, seq], vocab_size)
inputs = (input_ids,)
if script_args.with_mask in (1, "1"):
    input_mask = torch.tril(torch.ones(batch, seq, dtype=torch.float32))
    inputs = (input_ids, input_mask)


##########################################
# Measure of eager mode
# =====================

times = []

with torch.no_grad():

    # warmup
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
    times.append(dict(optium="eager", avg_time=d, N=N))
    print("avg time eager", d)

############################################
# Measure with the custom backend
# ===============================
#
# Three kind of optimization:
#
# - **default**: the onnx model is optimized with less onnx operators
# - **default+onnxruntime**: the onnx model is optimized with fused kernels
#   implemented by onnxruntime
# - **default+onnxruntime+experimental**: the onnx model is optimized with fused kernels
#   implemented by onnxruntime and also custom kernels

with torch.no_grad():

    for optim in ["default", "default+onnxruntime", "default+onnxruntime+experimental"]:
        print("----------------------")
        print(f"optim={optim}")
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
        times.append(dict(optium=optim, avg_time=d, N=N))
        print(f"avg time custom backend with optimization={optim!r}", d)

###############################################
# Final results
# =============

df = pandas.DataFrame(times)
print(df)
