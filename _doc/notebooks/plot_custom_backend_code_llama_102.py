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
    with_mask=(0, "tries with a mask as a secondary input"),
    optim=("default", "Optimization to apply, empty string for all"),
    description=__doc__,
    expose="config,num_hidden_layers,with_mask,optim",
)

assert script_args.optim, "optim must be specified."
assert script_args.with_mask in (0, "0"), "with_mask is not implemented."


print(f"with_mask={script_args.with_mask!r}")
print(f"optim={script_args.optim!r}")

#################################
# Imports.

import os
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

has_cuda = torch.cuda.is_available()
machine = get_machine()
print(f"has_cuda={has_cuda}")
print(f"processor: {machine['processor_name']}")
print(f"device: {machine.get('device_name', '?')}")

########################################
# The dummy model
# ===============

######################################
# The number of time we run the model to measure
# the inference.
warmup = 3
N = 10

###########################################
# Let's create the model.

# see https://huggingface.co/docs/transformers/en/model_doc/code_llama
if os.path.exists("CodeLlama-7b-model"):
    print("load the model")
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("./CodeLlama-7b-tokenizer")
    model = AutoModelForCausalLM.from_pretrained("./CodeLlama-7b-model")
else:
    print("retrieve the model")
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
    tokenizer.save_pretrained("CodeLlama-7b-tokenizer")
    model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-hf")
    model.save_pretrained("CodeLlama-7b-model")


##########################################
# Small example on how to generate an answer.

processor = "cuda" if has_cuda else "cpu"

PROMPT = '''def remove_non_ascii(s: str) -> str:
    """ <FILL_ME> """
    return result
'''

with torch.no_grad():
    print("tokenize the input")
    input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
    input_ids = input_ids.to(processor)
    print("run the model")
    model = model.to(processor)
    generated_ids = model.generate(input_ids, max_new_tokens=128).to(processor)
    print("interpret the answer")
    filling = tokenizer.batch_decode(
        generated_ids[:, input_ids.shape[1] :], skip_special_tokens=True
    )[0]
    print("---")
    print(PROMPT.replace("<FILL_ME>", filling))
    print("done")


# We use those inputs to benchmark the models.
inputs = (input_ids,)

# Just to make sure everything is ok.

print(f"moving model and inputs to processor={processor!r}")
model = model.to(processor)
inputs = tuple(i.to(processor) for i in inputs)

##########################################
# Measure of eager mode
# =====================


print("------------------------------------")
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
    times.append(dict(optium="eager", processor=processor, avg_time=d, warmup=warmup, N=N))
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
#
# The GPU memory is not fully freed before two iterations. Only one scenario
# should be handled in the same process.
# Results may be very different with a different chip.

optimization = [script_args.optim]

with torch.no_grad():

    for optim in optimization:
        print("----------------------")
        print(f"optim={optim}")

        # This variable is used to retrieve the onnx models created by the backend.
        # It can be set to None if it is not needed.
        # Graph are usually small as they do not contain weights.
        storage = {}

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
                optium=optim,
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

df = pandas.DataFrame(times)
print(df)
