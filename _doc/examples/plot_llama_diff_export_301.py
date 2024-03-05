"""
.. _l-plot-llama-diff-export:

301: Compares LLAMA exporters
=============================

The script compares the two exporters implemented in :epkg:`pytorch`
for a part of llama model. The model are compared after all optimizations
were made with :epkg:`onnx-rewriter` and :epkg:`onnxruntime`.

* `TorchScript-based ONNX Exporter
  <https://pytorch.org/docs/stable/onnx.html#torchscript-based-onnx-exporter>`_,
  let's call it **script**
* `TorchDynamo-based ONNX Exporter
  <https://pytorch.org/docs/stable/onnx.html#torchdynamo-based-onnx-exporter>`_,
  let's call it **dynamo**

To run the script:

::

    python _doc/examples/plot_llama_diff_export --help

Some helpers
++++++++++++
"""

import contextlib
import os
import io
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

import numpy as np
import onnx
from onnx_array_api.reference import compare_onnx_execution, ExtendedReferenceEvaluator
import torch
from experimental_experiment.ext_test_case import unit_test_going
from experimental_experiment.args import get_parsed_args
from experimental_experiment.torch_interpreter import to_onnx
from experimental_experiment.xbuilder import OptimizationOptions
from experimental_experiment.convert.convert_helper import (
    optimize_model_proto,
    ort_optimize,
)
from experimental_experiment.torch_helper.llama_helper import (
    get_llama_model,
    get_llama_attention,
    get_llama_decoder,
)
from experimental_experiment.torch_helper.dump_helper import reorder_functions_in_proto

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
    exporter=("dynamo", "one value among dynamo, custom"),
    ortopt=(1, "run onnxruntime optimization"),
    expose="part,exporter,ortopt",
)

print(f"part={script_args.part}")
print(f"exporter={script_args.exporter}")
ortopt = script_args.ortopt in (1, "1")
print(f"ortopt={ortopt}")


def opt_filename(filename: str) -> str:
    name, ext = os.path.splitext(filename)
    return f"{name}.opt{ext}"


def export_script(filename, model, *args):
    with contextlib.redirect_stdout(io.StringIO()):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(model, args, filename, input_names=["input"])
    if ortopt:
        onx = onnx.load(filename)
        ort_optimize(onx, opt_filename(filename), providers=provider)


def export_dynamo(filename, model, *args):
    with contextlib.redirect_stdout(io.StringIO()):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            export_output = torch.onnx.dynamo_export(model, *args)
            model = export_output.model_proto
    try:
        new_model = optimize_model_proto(model)
    except ImportError as e:
        print("skipping optimization, missing package:", e)
        new_model = model
    with open(filename, "wb") as f:
        f.write(new_model.SerializeToString())
    if ortopt:
        ort_optimize(new_model, opt_filename(filename), providers=provider)


def export_custom(filename, model, *args):
    new_model = to_onnx(
        model,
        tuple(args),
        input_names=[f"input{i}" for i in range(len(args))],
        options=OptimizationOptions(
            remove_unused=True,
            constant_folding=False,
        ),
    )
    with open(filename, "wb") as f:
        f.write(new_model.SerializeToString())
    if ortopt:
        ort_optimize(new_model, opt_filename(filename), providers=provider)


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

exporter = script_args.exporter
file1 = f"llama.{script_args.part}.script.onnx"
file2 = f"llama.{script_args.part}.{exporter}.onnx"

print("torch script exporter")
export_script(file1, model, *inputs[0])

if exporter == "dynamo":
    print("torch dynamo exporter")
    export_dynamo(file2, model, *inputs[0])
elif exporter == "custom":
    print("torch custom exporter")
    export_custom(file2, model, *inputs[0])
else:
    raise AssertionError(f"Unexpected value for exporter={exporter!r}.")

#########################################
# Verification
# ++++++++++++

if ortopt:
    print("Using models optimized by onnxruntime")
    file1 = f"llama.{script_args.part}.script.opt.onnx"
    file2 = f"llama.{script_args.part}.{exporter}.opt.onnx"


providers = (
    ["CPUExecutionProvider"]
    if provider == "cpu"
    else [("CUDAExecutionProvider", {}), ("CPUExecutionProvider", {})]
)

model1 = onnx.load(file1)
model2 = onnx.load(file2)

feeds1, feeds2 = {}, {}
for i in range(len(inputs[0])):
    x = inputs[0][i].detach().numpy()
    feeds1[model1.graph.input[i].name] = x
    feeds2[model2.graph.input[i].name] = x

if ortopt:
    sess1 = onnxruntime.InferenceSession(file1, providers=providers)
    sess2 = onnxruntime.InferenceSession(file2, providers=providers)

    got1 = sess1.run(None, feeds1)
    got2 = sess2.run(None, feeds2)

    diff1 = np.abs(expected.detach().numpy() - got1[0]).max()
    diff2 = np.abs(expected.detach().numpy() - got2[0]).max()

    print(f"Error with the eager model and onnxruntime: {diff1}, {diff2}")

#########################################
# Verification with the reference evaluator
# +++++++++++++++++++++++++++++++++++++++++

reorder_functions_in_proto(file1)
reorder_functions_in_proto(file2)

sess1 = ExtendedReferenceEvaluator(file1)
try:
    sess2 = ExtendedReferenceEvaluator(file2)
except NotImplementedError as e:
    print(e)
    sess2 = None

got1 = sess1.run(None, feeds1)
got2 = got1 if sess2 is None else sess2.run(None, feeds2)

diff1 = np.abs(expected.detach().numpy() - got1[0]).max()
diff2 = np.abs(expected.detach().numpy() - got2[0]).max()

print(f"Error with the eager model and the reference evaluator: {diff1}, {diff2}")

#########################################
# Comparison and execution
# ++++++++++++++++++++++++


def clean_name(name):
    return name.replace(
        "_inlfunc_transformers_models_llama_modeling_llama_LlamaAttention", ""
    ).replace("_inlfunc_torch_nn_modules_linear_Linear", "")


if sess2 is not None:
    try:
        np_inputs = [i.detach().numpy() for i in inputs[0]]
        res1, res2, align, dc = compare_onnx_execution(
            model1, model2, inputs=np_inputs, verbose=1, raise_exc=False
        )
        for r in res2:
            r.name = clean_name(r.name)
        text = dc.to_str(res1, res2, align, column_size=90)
        print(text)
    except AssertionError as e:
        if (
            "Unexpected type <class 'list'> for value, it must be a numpy array."
            not in str(e)
        ):
            raise
        print(e)

#################################
# See :ref:`l-long-outputs-llama-diff-export` for a better view.
