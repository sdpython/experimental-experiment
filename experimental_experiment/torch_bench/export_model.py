"""
Export a model
==============

The script export a model with different options.

::

    python -m experimental_experiment.torch_bench.export_model --help

Example, run llama model with onnxrt backend on cuda.

::

    python -m experimental_experiment.torch_bench.export_model --exporter script --device cuda
    
"""

from experimental_experiment.torch_bench._dort_cmd_common import export_args

args = export_args(
    "experimental_experiment.torch_bench.export_model", description=__doc__
)

import os
import time
import onnxruntime  # noqa: F401
import onnx
import torch
import torch._dynamo.backends.registry
from experimental_experiment.convert.convert_helper import (
    ort_optimize,
    optimize_model_proto,
)
from experimental_experiment.torch_bench._dort_cmd_common import (
    create_configuration_for_benchmark,
    create_model,
)
from experimental_experiment.xbuilder import OptimizationOptions
from experimental_experiment.torch_interpreter import to_onnx
from experimental_experiment.xoptim.patterns import get_pattern_list

config_dict = create_configuration_for_benchmark(
    model=args.model,
    config=args.config,
    repeat=1,
    warmup=1,
    num_hidden_layers=args.num_hidden_layers,
    implementation=args.implementation,
    with_mask=args.with_mask,
)

verbose = int(args.verbose)
optimize = args.optimize in (True, 1, "1", "True")
with_mask = args.with_mask in (True, 1, "1", "True")
disable_pattern = [_ for _ in args.disable_pattern.split(",") if _]
enable_pattern = [_ for _ in args.enable_pattern.split(",") if _]
print(f"model={args.model}")
print(f"model config={config_dict}")
print(f"exporter={args.exporter}")
print(f"verbose={verbose}")
print(f"optimize={args.optimize}")
print(f"implementation={args.implementation}")
print(f"with_mask={args.with_mask}")
print(f"mixed={args.mixed}")

if args.exporter == "custom":
    print(f"disable_pattern={disable_pattern!r}")
    print(f"enable_pattern={enable_pattern!r}")


is_cuda = args.device == "cuda"
if is_cuda:
    print(
        f"CUDA no model: memory allocated={torch.cuda.memory_allocated(0)}, "
        f"reserved={torch.cuda.memory_reserved(0)}"
    )

model, example_args_collection = create_model(args.model, config_dict)

device = args.device
model = model.eval().to(device)
example_inputs = example_args_collection[0]
inputs = tuple([t.to("cuda") for t in example_inputs]) if is_cuda else example_inputs

if is_cuda:
    print(
        f"CUDA model loaded: memory allocated={torch.cuda.memory_allocated(0)}, "
        f"reserved={torch.cuda.memory_reserved(0)}"
    )

print(f"Exporter model with exporter={args.exporter}, n_inputs={len(inputs)}")
use_dynamic = args.dynamic in (1, "1", True, "True")
print(f"dynamic={use_dynamic}")
use_mixed = args.mixed in (1, "1", True, "True")
print(f"mixed={use_mixed}")

folder = "dump_model"
if not os.path.exists(folder):
    os.mkdir(folder)

filename = os.path.join(
    folder,
    (
        f"export_{args.model}_{args.exporter}_{'dyn' if use_dynamic else 'static'}"
        f"{'_mixed' if use_mixed else ''}_{args.config}-v{args.target_opset}"
        f".{args.implementation}.onnx"
    ),
)
print(f"start exporting in {filename!r}")
begin = time.perf_counter()
if args.exporter == "script":
    torch.onnx.export(
        model,
        inputs,
        filename,
        do_constant_folding=False,
        input_names=[f"input{i}" for i in range(len(inputs))],
        opset_version=args.target_opset,
    )
elif args.exporter == "dynamo":
    with torch.no_grad():
        prog = torch.onnx.dynamo_export(model, *inputs)
    onx = prog.to_model_proto()
    if optimize:
        onx = optimize_model_proto(onx)
    with open(filename, "wb") as f:
        f.write(onx.SerializeToString())
elif args.exporter == "custom":
    patterns = get_pattern_list(enable_pattern, disable_pattern, verbose=args.verbose)
    onx = to_onnx(
        model,
        inputs,
        input_names=[f"input{i}" for i in range(len(inputs))],
        options=OptimizationOptions(patterns=patterns),
        verbose=args.verbose,
        target_opset=args.target_opset,
        optimize=optimize,
    )
    print([i.name for i in onx.graph.input])
    with open(filename, "wb") as f:
        f.write(onx.SerializeToString())
else:
    raise RuntimeError(f"Unknown exporter {args.exporter!r}")

print(f"exporter done in {time.perf_counter() - begin}s")

with open(filename, "rb") as f:
    onx = onnx.load(filename)
print(f"size of the export: {os.stat(filename).st_size / 2**20} Mb")
for i, init in enumerate(onx.graph.initializer):
    print(
        f"initializer {i+1}/{len(onx.graph.initializer)}:"
        f"{init.data_type}:{tuple(init.dims)}:{init.name}"
    )

if args.ort in (1, "1", "True", True):
    ort_filename = filename.replace(".onnx", ".opt.onnx")
    print(f"Optimize with onnxruntime={args.exporter}")
    ort_optimize(filename, ort_filename, providers=args.device)

print("done.")
