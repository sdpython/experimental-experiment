"""
.. _l-plot-custom-backend-llama-102:

========================================
102: Fuse kernels in a small Llama Model
========================================

This example leverages the function :epkg:`torch.compile` and the ability
to use a custom backend (see :epkg:`Custom Backends`)
to test the optimization of a model by fusing simple element-wise kernels.

It takes a small Llama model and uses a backend based on :epkg:`onnxruntime`.
The model is converted into ONNX and then optimized by fusing element-wise
kernels.

::

    python plot_custom_backend_llama --config large

The script requires the following packages beside pytorch,
:epkg:`onnxruntime-training` (for GPU), :epkg:`onnx-extended`
(compiled for GPU) and :epkg:`transformers`.
"""

from experimental_experiment.args import get_parsed_args

script_args = get_parsed_args(
    "plot_custom_backend_llama",
    config=("medium", "large or medium depending, large means closer to the real model"),
    num_hidden_layers=(1, "number of hidden layers"),
    with_mask=(0, "tries with a mask as a secondary input"),
    optim=("", "Optimization to apply, empty string for all"),
    description=__doc__,
    expose="config,num_hidden_layers,with_mask,optim",
)

print(f"config={script_args.config!r}")
print(f"num_hidden_layers={script_args.num_hidden_layers!r}")
print(f"with_mask={script_args.with_mask!r}")
print(f"optim={script_args.optim!r}")

#################################
# Imports.

import time
import numpy as np
import pandas
from tqdm import tqdm
import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaModel
from experimental_experiment.xbuilder import OptimizationOptions
from experimental_experiment.torch_dynamo import onnx_custom_backend
from experimental_experiment.bench_run import get_machine
from experimental_experiment.ext_test_case import unit_test_going

has_cuda = torch.cuda.is_available()
machine = get_machine()
print(f"has_cuda={has_cuda}")
print(f"processor: {machine['processor_name']}")
print(f"device: {machine.get('device_name', '?')}")

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
    hidden_size = 4096
    num_attention_heads = 32
else:
    batch, seq, vocab_size = 2, 1024, 1024
    intermediate_size = 1024
    hidden_size = 512
    num_attention_heads = 8

################################
# The configuration of the model.

config = LlamaConfig(
    hidden_size=hidden_size,
    num_hidden_layers=int(script_args.num_hidden_layers),
    vocab_size=vocab_size,
    intermediate_size=intermediate_size,
    max_position_embeddings=2048,
    num_attention_heads=num_attention_heads,
)
config._attn_implementation = "eager"

######################################
# The number of time we run the model to measure
# the inference.
warmup = 10 if script_args.config == "medium" else 5
N = 50 if script_args.config == "medium" else 25

###########################################
# Let's create the model with dummy inputs.
print("creates the model")
model = LlamaModel(config)

inputs = (ids_tensor([batch, seq], vocab_size),)
if script_args.with_mask in (1, "1"):
    input_mask = torch.tril(torch.ones(batch, seq, dtype=torch.float32))
    inputs = (*inputs, input_mask)

processor = "cuda" if has_cuda else "cpu"
print(f"moving model and inputs to processor={processor!r}")
model = model.to(processor)
inputs = tuple(i.to(processor) for i in inputs)


##########################################
# Measure of eager mode
# =====================

times = []

with torch.no_grad():

    # warmup
    print("warmup eager")
    for _ in tqdm(range(warmup)):
        # model(input_ids, input_mask)
        model(*inputs)
        if has_cuda:
            torch.cuda.synchronize()

    # repeat
    print("repeat eager")
    begin = time.perf_counter()
    for _ in tqdm(range(N)):
        model(*inputs)
        if has_cuda:
            torch.cuda.synchronize()
    d = (time.perf_counter() - begin) / N
    baseline = d
    times.append(dict(optim="eager", processor=processor, avg_time=d, warmup=warmup, N=N))
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
#   implemented by onnxruntime and also custom kernels, this does not work on
#   CPU.
#
# Some links:
#
# * :class:`experimental_experiment.xbuilder.OptimizationOptions`:
#   that class defines the optimizations to apply after the model
#   is converted to onnx,
# * :func:`experimental_experiment.torch_dynamo.onnx_custom_backend`:
#   that function implements the custom backend based on :epkg:`onnxruntime`,
#   it converts the model into ONNX, optimizes and runs it,
#   it does not support :epkg:`graph break`,
#   it does not work well with dynamic shapes yet.
# * The CUDA kernels are implemented at
#   `onnx_extended/ortops/optim/cuda
#   <https://github.com/sdpython/onnx-extended/tree/main/onnx_extended/ortops/optim/cuda>`_
# * Section :ref:`l-custom-op-patterns` covers the implemented patterns fusing nodes
#   in onnx models. See :ref:`l-design-pattern-optimizer` to understand how
#   these are applied to modify an onnx model.
#
# The GPU memory is not fully freed before two iterations. Only one scenario
# should be handled in the same process.
# Results may be very different with a different chip.

optimization = (
    [script_args.optim]
    if script_args.optim
    else ["default", "default+onnxruntime", "default+onnxruntime+experimental"]
)

if unit_test_going():
    # It is too long.
    optimization = []
    times = []


with torch.no_grad():

    for optim in optimization:
        print("----------------------")
        print(f"optim={optim}")

        # This variable is used to retrieve the onnx models created by the backend.
        # It can be set to None if it is not needed.
        # Graph are usually small as they do not contain weights.
        storage = None  # {}

        options = OptimizationOptions(
            constant_folding=True,
            patterns=None if optim == "" else optim,
            verbose=0,
            processor=processor.upper(),
        )

        # The backend used here overwrite some of the parameters provided by
        # function onnx_custom_backend.
        custom_custom_backend = lambda *args, optim=optim, options=options, storage=storage, **kwargs: onnx_custom_backend(  # noqa: E731, E501
            *args,
            target_opset=18,
            verbose=0,
            options=options,
            optimize=optim != "",
            storage=storage,
            dump_prefix=f"dump_onx_llama_{optim.replace('+', '_')}",
            **kwargs,
        )

        # The function setting the backend.
        compiled_model = torch.compile(
            model, backend=custom_custom_backend, fullgraph=True, dynamic=False
        )

        # warmup
        print("warmup compiled model")
        for _ in tqdm(range(warmup)):
            compiled_model(*inputs)
            if has_cuda:
                torch.cuda.synchronize()

        # repeat
        print("repeat compiled_model")
        begin = time.perf_counter()
        for _ in tqdm(range(N)):
            compiled_model(*inputs)
            if has_cuda:
                torch.cuda.synchronize()
        d = (time.perf_counter() - begin) / N

        # let's measure the number of custom ops
        n_custom_ops = None
        if storage is not None:
            onnx_model = storage["instance"][0]["onnx"]
            n_custom_ops = len([node for node in onnx_model.graph.node if node.domain != ""])

        times.append(
            dict(
                optim=optim,
                processor=processor,
                avg_time=d,
                warmup=warmup,
                N=N,
                n_custom_ops=n_custom_ops,
                speedup=baseline / d,
            )
        )
        print(f"avg time custom backend with optimization={optim!r}", d)

###############################################
# Final results
# =============
#
# avg_time, lower is better,
# speedup compare to eager mode, higher is better.

if times:
    df = pandas.DataFrame(times)
    print(df)

######################################
# Plot

if times:
    df.set_index("optim")[["speedup"]].plot.bar(
        title="Speedup for different optimization scenario"
    )
