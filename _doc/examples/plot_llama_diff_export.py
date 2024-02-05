"""
Compares LLAMA exporters
========================

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
    expose="part",
)

print(f"part={script_args.part}")


def opt_filename(filename: str) -> str:
    name, ext = os.path.splitext(filename)
    return f"{name}.opt{ext}"


def export_script(filename, model, *args):
    with contextlib.redirect_stdout(io.StringIO()):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(model, args, filename, input_names=["input"])
    onx = onnx.load(filename)
    ort_optimize(onx, opt_filename(filename), providers=provider)


def export_dynamo(filename, model, *args):
    with contextlib.redirect_stdout(io.StringIO()):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            export_output = torch.onnx.dynamo_export(model, *args)
            model = export_output.model_proto
    new_model = optimize_model_proto(model)
    with open(filename, "wb") as f:
        f.write(new_model.SerializeToString())
    ort_optimize(new_model, opt_filename(filename), providers=provider)


###################################
# Model and data

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
# Export

file1 = f"llama.{script_args.part}.script.onnx"
file2 = f"llama.{script_args.part}.dynamo.onnx"

print("torch script exporter")
export_script(file1, model, *inputs[0])

print("torch dynamo exporter")
export_dynamo(file2, model, *inputs[0])

#########################################
# Verification

file1 = f"llama.{script_args.part}.script.opt.onnx"
file2 = f"llama.{script_args.part}.dynamo.opt.onnx"


providers = (
    ["CPUExecutionProvider"]
    if provider == "cpu"
    else ["CUDAExecutionProvider", "CPUExecutionProvider"]
)
sess1 = onnxruntime.InferenceSession(file1, providers=providers)
sess2 = onnxruntime.InferenceSession(file2, providers=providers)


model1 = onnx.load(file1)
model2 = onnx.load(file2)

feeds1, feeds2 = {}, {}
for i in range(len(inputs[0])):
    x = inputs[0][i].detach().numpy()
    feeds1[sess1.get_inputs()[i].name] = x
    feeds2[sess2.get_inputs()[i].name] = x

got1 = sess1.run(None, feeds1)
got2 = sess2.run(None, feeds2)

diff1 = np.abs(expected.detach().numpy() - got1[0]).max()
diff2 = np.abs(expected.detach().numpy() - got2[0]).max()

print(f"Error with the eager model: {diff1}, {diff2}")

#########################################
# With the reference evaluator

sess1 = ExtendedReferenceEvaluator(file1)
sess2 = ExtendedReferenceEvaluator(file2)


got1 = sess1.run(None, feeds1)
got2 = sess2.run(None, feeds2)

diff1 = np.abs(expected.detach().numpy() - got1[0]).max()
diff2 = np.abs(expected.detach().numpy() - got2[0]).max()

print(f"Error with the eager model: {diff1}, {diff2}")

#########################################
# Comparison and execution
# ++++++++++++++++++++++++


def clean_name(name):
    return name.replace(
        "_inlfunc_transformers_models_llama_modeling_llama_LlamaAttention", ""
    ).replace("_inlfunc_torch_nn_modules_linear_Linear", "")


np_inputs = [i.detach().numpy() for i in inputs[0]]
res1, res2, align, dc = compare_onnx_execution(
    model1, model2, inputs=np_inputs, verbose=1
)
for r in res2:
    r.name = clean_name(r.name)
text = dc.to_str(res1, res2, align, column_size=90)
print(text)
