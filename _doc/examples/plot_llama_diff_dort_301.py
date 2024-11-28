"""
.. _l-plot-onnxrt-diff:

301: Compares LLAMA exporters for onnxrt backend
================================================

The script compares exported models in :epkg:`pytorch`
using :epkg:`onnxrt backend`. It tries to do a side by side
of the execution of both models.

To run the script:

::

    python _doc/examples/plot_llama_diff_dort --help


The following example compares the forward step for mixed precision on cuda
and produces all the intermediate onnx graphs.

::

    python _doc/examples/plot_llama_diff_dort.py --part model --ortopt 1 \\
            --cuda 1 --backward 0 --mixed 1

You may use ``--mixed=1`` to compare the backward graphs.

Some helpers
++++++++++++
"""

from experimental_experiment.args import get_parsed_args

script_args = get_parsed_args(
    "plot_llama_diff_export",
    description=__doc__,
    part=("attention", "one value among attention, decoder, model"),
    ortopt=(1, "run onnxruntime optimization"),
    backward=(0, "does one operator for backward"),
    cuda=(0, "use cuda or not"),
    mixed=(0, "use miwed precision"),
    opset=(18, "onnx opset"),
    expose="part,exporter,ortopt,cuda,mixed,opset",
)


import copy
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
from torch._dynamo.backends.common import aot_autograd
from experimental_experiment.ext_test_case import unit_test_going
from experimental_experiment.convert.convert_helper import (
    optimize_model_proto_oxs,
    ort_optimize,
)
from experimental_experiment.torch_models.llama_helper import (
    get_llama_model,
    get_llama_attention,
    get_llama_decoder,
)
from experimental_experiment.torch_models.dump_helper import (
    assert_all_close,
    dump_onnx,
    reorder_functions_in_proto,
    inputs_from_onnx_model,
    build_matching_inputs,
    results_to_string,
)
from experimental_experiment.torch_models.training_helper import (
    train_loop,
    make_aot_ort,
)
from experimental_experiment.torch_dynamo import (
    onnx_debug_backend,
    get_decomposition_table,
)

has_cuda = has_cuda and torch.cuda.is_available()
logging.disable(logging.ERROR)
provider = "cuda" if has_cuda else "cpu"


#####################################
# The exporting functions
# +++++++++++++++++++++++

print(f"part={script_args.part}")
ortopt = script_args.ortopt in (1, "1")
print(f"ortopt={ortopt}")
backward = script_args.backward in (1, "1")
print(f"backward={backward}")
use_cuda = script_args.cuda in (1, "1")
print(f"cuda={use_cuda}")
use_mixed = script_args.mixed in (1, "1")
print(f"mixed={use_mixed}")
opset = int(script_args.opset)
print(f"opset={opset}")

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

if use_cuda:
    model = model.to("cuda")
    inputs = [[i.to("cuda") for i in inp] for inp in inputs]

print(f"simple run with {len(inputs)} inputs")
if backward:
    if use_mixed:
        assert use_cuda, "mixed precision only works with cuda"
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            torch.cuda.synchronize()
            expected = train_loop(copy.deepcopy(model), *inputs[0])
            torch.cuda.synchronize()
    else:
        expected = train_loop(copy.deepcopy(model), *inputs[0])
    print(
        f"-- eager mode worked, {len(expected)} gradients, first one is "
        f"{expected[0].shape}, {expected[0].dtype}"
    )
else:
    if use_mixed:
        assert use_cuda, "mixed precision only works with cuda"
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            torch.cuda.synchronize()
            expected = model(*inputs[0])
            torch.cuda.synchronize()
    else:
        expected = model(*inputs[0])
    print(results_to_string(expected))


###################################
# Exporting
# +++++++++

if hasattr(torch._dynamo.variables.misc, "LoggingLoggerVariable"):
    # A tweak to make torch.export.export work.
    torch._dynamo.variables.misc.LoggingLoggerVariable.call_method = lambda *_, **__: None


folder = "dump_models"
storage = {}

if backward:
    # onnxrt backend
    local_aot_ort, _ = make_aot_ort(dynamic=False, rewrite=True)

    optimized_mod = torch.compile(
        copy.deepcopy(model), backend=local_aot_ort, dynamic=False, fullgraph=True
    )

    with dump_onnx("llama_onnxrt", folder=folder, clean=True):
        if use_mixed:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                torch.cuda.synchronize()
                expected_onnxrt = train_loop(optimized_mod, *inputs[0])
                torch.cuda.synchronize()
        else:
            expected_onnxrt = train_loop(optimized_mod, *inputs[0])
    assert_all_close(expected[0], expected_onnxrt[0], atol=1e-3)
    print(
        f"-- onnxrt backend worked, {len(expected_onnxrt)} gradients, first one is "
        f"{expected_onnxrt[0].shape}, {expected_onnxrt[0].dtype}"
    )

    # debugging backend
    aot_compiler = aot_autograd(
        fw_compiler=lambda *args, **kwargs: onnx_debug_backend(
            *args,
            dump_prefix=os.path.join(folder, "llama_debug"),
            target_opset=opset,
            storage=storage,
            **kwargs,
        ),
        decompositions=get_decomposition_table(),
    )
    onnx_mod = torch.compile(copy.deepcopy(model), backend=aot_compiler, fullgraph=True)

    if use_mixed:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            torch.cuda.synchronize()
            got = train_loop(onnx_mod, *inputs[0])
            torch.cuda.synchronize()
    else:
        got = train_loop(onnx_mod, *inputs[0])
    assert_all_close(expected[0], got[0], atol=1e-2 if use_mixed else 1e-4)
    print(
        f"-- debug backend worked, {len(got)} gradients, first one is "
        f"{got[0].shape}, {got[0].dtype}"
    )

else:
    # onnxrt backend
    local_aot_ort, _ = make_aot_ort(dynamic=True, rewrite=True)
    optimized_mod = torch.compile(model, backend=local_aot_ort, fullgraph=True)
    with dump_onnx("llama_onnxrt", folder=folder, clean=True):
        if use_mixed:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                torch.cuda.synchronize()
                expected_onnxrt = optimized_mod(*inputs[0])
                torch.cuda.synchronize()
        else:
            expected_onnxrt = optimized_mod(*inputs[0])
    assert_all_close(expected, expected_onnxrt, atol=1e-2)

    # debugging backend
    aot_compiler = aot_autograd(
        fw_compiler=lambda *args, **kwargs: onnx_debug_backend(
            *args,
            dump_prefix=os.path.join(folder, "llama_debug"),
            target_opset=17,
            storage=storage,
            **kwargs,
        )
    )

    onnx_mod = torch.compile(model, backend=aot_compiler, fullgraph=True)
    if use_mixed:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            got = onnx_mod(*inputs[0])
    else:
        try:
            got = onnx_mod(*inputs[0])
        except Exception as e:
            print(f"ERROR: {e}")
            got = None
    if got is not None:
        assert_all_close(expected, got, atol=1 if use_mixed else 1e-3)

#############################
# For forward, there are two files, one onnx model and the graph module
# printed in a txt file. For backward, there are two onnx models.
# Then it is multiplied by the number of backends.

models = os.listdir(folder)
print(f"exported models: {models}")

##############################
# Inputs used by the debug backend

if "instance" in storage:
    feeds = storage["instance"][0]["inputs"][0]
    for k, v in feeds.items():
        print(f"-- {k} {v.dtype} {v.shape}")

################################
# Let's the first line of the graph module

if "instance" in storage:
    graph_module = storage["instance"][0]["graph_module"]
    print("\n".join(str(graph_module.graph).split("\n")[:10]))


#########################################
# Comparison and execution
# ++++++++++++++++++++++++

if "instance" in storage:
    if backward:
        print(f"-- {len(storage['instance'])} onnx models were creates")
        for i, inst in enumerate(storage["instance"]):
            print(f"  model {i}: {len(inst['inputs'])} runs")

        # deal with backward
        onnx_models = list(sorted([m for m in models if m.endswith(".onnx")]))
        assert len(onnx_models) == 4, f"unexpected value {onnx_models}"
        onnx_models = list(sorted([m for m in models if m.endswith(".onnx") and "_1" in m]))
        assert len(onnx_models) == 2, f"unexpected value {onnx_models}"
        model_onnxrt = os.path.join(folder, onnx_models[1])
        model_debug = os.path.join(folder, onnx_models[0])
    else:
        onnx_models = list(sorted([m for m in models if m.endswith(".onnx")]))
        if len(onnx_models) == 2:
            model_onnxrt = os.path.join(folder, onnx_models[1])
            model_debug = os.path.join(folder, onnx_models[0])
        else:
            model_debug = os.path.join(folder, onnx_models[0])
            # the following error may appear:
            # Node type 'Rank' from domain 'pkg.onnxscript.torch_lib.common' is unknown
            print(f"One model is missing, onnx_models={onnx_models}")
            model_onnxrt = model_debug

    print(f"model_onnxrt={model_onnxrt}")
    print(f"model_debug={model_debug}")

############################
# The inputs of both models

if "instance" in storage:
    print("onnxrt:", inputs_from_onnx_model(model_onnxrt))
    print("debug:", inputs_from_onnx_model(model_debug))

#################################
# Inputs are not the same. The first model has more and some inputs were
# moved into the initializer list into for `model_debug`.

if "instance" in storage:
    print("debug:", inputs_from_onnx_model(model_debug, init=True))

#####################################
# Optimization and Verification
# +++++++++++++++++++++++++++++
#
# Let's try the model with a python backend (reference implementation).
# First step, onnxscript uses many functions. The reference evaluation expects
# every function to be defined so the order of functions in the model matters.
# No recursivity is allowed by this runtime.
# We need to reorder as function Rank is usually placed
# at the end of the model.

if "instance" in storage:
    reorder_functions_in_proto(model_onnxrt)

####################################
# Let's load the model and optimize them.

if "instance" in storage:
    debug = onnx.load(model_debug)
    try:
        onnxrt = optimize_model_proto_oxs(onnx.load(model_onnxrt))
    except ImportError as e:
        print("missing library", e)
        onnxrt = debug

###################################
# Let's apply onnxruntime optimization

if "instance" in storage and ortopt:
    providers = (
        [("CUDAExecutionProvider", {}), ("CPUExecutionProvider", {})]
        if use_cuda
        else ["CPUExecutionProvider"]
    )
    with open(model_onnxrt.replace(".onnx", ".before.opt.onnx"), "wb") as f:
        f.write(onnxrt.SerializeToString())
    print(f"run onnxruntime optimization on {model_onnxrt}")
    optimized = model_onnxrt.replace(".onnx", ".opt.onnx")
    ort_optimize(onnxrt, output=optimized, providers=providers)
    onnxrt = onnx.load(optimized)

    print(f"run onnxruntime optimization on {model_debug}")
    optimized = model_debug.replace(".onnx", ".opt.onnx")
    ort_optimize(debug, output=optimized, disable_aot=True, providers=providers)
    debug = onnx.load(optimized)

#################################
# For what's following, we need to build two lists of matching inputs.

if "instance" in storage:
    print("build_matching_inputs")
    feedsrt = build_matching_inputs(model_debug, feeds, model_onnxrt)
    print("done")


#######################
# We check both models are running.

if "instance" in storage:
    out_onnxrt = ExtendedReferenceEvaluator(onnxrt).run(None, feedsrt)
    out_debug = ExtendedReferenceEvaluator(debug).run(None, feeds)
    assert out_onnxrt
    assert out_debug

# assert_all_close(out_onnxrt, out_debug)

####################################
# Side by side


if "instance" in storage:
    res1, res2, align, dc = compare_onnx_execution(
        onnxrt,
        debug,
        verbose=1,
        raise_exc=True,
        inputs=(feedsrt, feeds),
    )
    text = dc.to_str(res1, res2, align, column_size=90)
    print(text)
