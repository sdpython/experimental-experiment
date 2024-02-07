"""
.. _l-plot-onnxrt-diff:

Compares LLAMA exporters for onnxrt backend
===========================================

The script compares exported models in :epkg:`pytorch`
using :epkg:`onnxrt backend`.

To run the script:

::

    python _doc/examples/plot_llama_diff_dort --help

Some helpers
++++++++++++
"""

import os
import warnings
import logging

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import onnxruntime

        has_cuda = "CUDAExecutionProvider" in onnxruntime.get_available_providers()
except ImportError:
    print("onnxruntime not available.")
    import sys

    sys.exit(0)

import onnx
from onnx_array_api.reference import compare_onnx_execution, ExtendedReferenceEvaluator
import torch
from experimental_experiment.ext_test_case import get_parsed_args, unit_test_going
from experimental_experiment.convert.convert_helper import (
    optimize_model_proto,
    ort_optimize,
)
from experimental_experiment.torch_helper.llama_helper import (
    get_llama_model,
    get_llama_attention,
    get_llama_decoder,
)
from experimental_experiment.torch_helper.dump_helper import (
    assert_all_close,
    dump_onnx,
    onnx_debug_backend,
    reorder_functions_in_proto,
    inputs_from_onnx_model,
    build_matching_inputs,
)

has_cuda = has_cuda and torch.cuda.is_available()
logging.disable(logging.ERROR)
provider = "cuda" if has_cuda else "cpu"


#####################################
# The exporting functions
# +++++++++++++++++++++++


script_args = get_parsed_args(
    "plot_llama_diff_export",
    description=__doc__,
    part=("attention", "one value among attention, decoder, model"),
    ortopt=(1, "run onnxruntime optimization"),
    backward=(0, "does one operator for backward"),
    expose="part,exporter,ortopt",
)

print(f"part={script_args.part}")
ortopt = script_args.ortopt in (1, "1")
print(f"ortopt={ortopt}")
backward = script_args.backward in (1, "1")
print(f"backward={backward}")

###################################
# Model and data
# ++++++++++++++

if unit_test_going():
    kwargs = dict(input_dims=[(2, 1024)] * 2)
else:
    kwargs = dict(
        input_dims=[(2, 1024)] * 2,
        _attn_implementation="eager",
        num_hidden_layers=1,
        hidden_size=512,
        vocab_size=4000,
        intermediate_size=2000,
        max_position_embeddings=2048,
        num_attention_heads=8,
    )

if script_args.part == "attention":
    model, inputs = get_llama_attention(**kwargs)
elif script_args.part == "decoder":
    model, inputs = get_llama_decoder(**kwargs)
elif script_args.part == "model":
    model, inputs = get_llama_model(**kwargs)
else:
    raise RuntimeError(f"Unexpected value for part={script_args.part!r}")

print(f"simple run with {len(inputs)} inputs")
expected = model(*inputs[0])
print(f"eager mode worked {expected.shape}, {expected.dtype}")


###################################
# Exporting
# +++++++++

folder = "dump_models"
storage = {}

if backward:
    raise NotImplementedError()
else:
    # onnxrt backend
    optimized_mod = torch.compile(model, backend="onnxrt", fullgraph=True)
    with dump_onnx("llama_onnxrt", folder=folder, clean=True):
        expected_onnxrt = optimized_mod(*inputs[0])
    assert_all_close(expected, expected_onnxrt)

    # debugging backend
    onnx_mod = torch.compile(
        model,
        backend=lambda *args, **kwargs: onnx_debug_backend(
            *args,
            dump_prefix=os.path.join(folder, "llama_debug"),
            target_opset=18,
            storage=storage,
            **kwargs,
        ),
        fullgraph=True,
    )
    got = onnx_mod(*inputs[0])
    assert_all_close(expected, got)

#############################
# For forward, there are two files, one onnx model and the graph module
# printed in a txt file. For backward, there are two onnx models.
# Then it is multiplied by the number of backends.

models = os.listdir(folder)
print(f"exported models: {models}")

##############################
# Inputs used by the debug backend

feeds = storage["instance"][0]["inputs"][0]
for k, v in feeds.items():
    print(f"-- {k} {v.dtype} {v.shape}")

################################
# Let's the first line of the graph module

graph_module = storage["instance"][0]["graph_module"]
print("\n".join(str(graph_module.graph).split("\n")[:10]))


#########################################
# Comparison and execution
# ++++++++++++++++++++++++

if backward:
    assert False, "Not implemented yet"
else:
    onnx_models = list(sorted([m for m in models if m.endswith(".onnx")]))
    assert len(onnx_models) == 2, f"unexpected value {onnx_models}"
    model_onnxrt = os.path.join(folder, onnx_models[1])
    model_debug = os.path.join(folder, onnx_models[0])

print(f"model_onnxrt={model_onnxrt}")
print(f"model_debug={model_debug}")

############################
# The inputs of both models

print("onnxrt:", inputs_from_onnx_model(model_onnxrt))
print("debug:", inputs_from_onnx_model(model_debug))

#################################
# Inputs are not the same. The first model has more and some inputs were
# moved into the initializer list into for `model_debug`.

print("debug:", inputs_from_onnx_model(model_debug, init=True))

#####################################
# Optimization and Verification
# +++++++++++++++++++++++++++++
#
# Let's try the model with a python backend (reference implementation).
# First step, onnx-script uses many functions. The reference evaluation expects
# every function to be defined so the order of functions in the model matters.
# No recursivity is allowed by this runtime. We need to reorder as function Rank is usually placed
# at the end of the model.

reorder_functions_in_proto(model_onnxrt)

#################################
# For what's following, we need to build two lists of matching inputs.

feedsrt = build_matching_inputs(model_debug, feeds, model_onnxrt)

####################################
# Let's load the model and optimize them.

onnxrt = optimize_model_proto(onnx.load(model_onnxrt))
debug = onnx.load(model_debug)

###################################
# Let's apply onnxruntime optimization

if ortopt:
    print(f"run onnxruntime optimization on {model_onnxrt}")
    optimized = model_onnxrt.replace(".onnx", ".opt.onnx")
    ort_optimize(onnxrt, output=optimized)
    onnxrt = onnx.load(optimized)

    print(f"run onnxruntime optimization on {model_debug}")
    optimized = model_debug.replace(".onnx", ".opt.onnx")
    ort_optimize(debug, output=optimized)
    debug = onnx.load(optimized)


#######################
# We check both models are running.

out_onnxrt = ExtendedReferenceEvaluator(onnxrt).run(None, feedsrt)
out_debug = ExtendedReferenceEvaluator(debug).run(None, feeds)
assert out_onnxrt
assert out_debug

# assert_all_close(out_onnxrt, out_debug)

####################################
# Side by side


res1, res2, align, dc = compare_onnx_execution(
    onnxrt,
    debug,
    verbose=1,
    raise_exc=True,
    inputs=(feedsrt, feeds),
)
text = dc.to_str(res1, res2, align, column_size=90)
print(text)
