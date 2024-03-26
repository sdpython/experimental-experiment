"""
Run llama model with DORT
=========================

The script runs a few iterations of a dummy llama model.

::

    python -m experimental_experiment.torch_bench.dort_bench --help

Example, run llama model with onnxrt backend on cuda.

::

    python -m experimental_experiment.torch_bench.dort_bench --backend ort --device cuda
    
Other example, same script but dumps the produces models.

::

    ONNXRT_DUMP_PATH="llama_dort_" python -m experimental_experiment.torch_bench.dort_bench --backend ort --device cuda

Or simply this one:

::

    python -m experimental_experiment.torch_bench.dort_bench --backend custom --device cuda --export a -w 1
"""

from experimental_experiment.torch_bench._dort_cmd_common import dort_args

args = dort_args("experimental_experiment.torch_bench.dort_bench", description=__doc__)

import os
import time
import onnxruntime  # noqa: F401
import numpy as np
import torch
import torch._dynamo.backends.registry
import transformers
from experimental_experiment.convert.convert_helper import ort_optimize
from experimental_experiment.torch_models.dump_helper import dump_onnx
from experimental_experiment.torch_bench._dort_cmd_common import (
    create_compiled_model,
    create_configuration_for_benchmark,
    create_model,
)

config_dict = create_configuration_for_benchmark(
    model=args.model,
    config=args.config,
    repeat=args.repeat,
    warmup=args.warmup,
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
print(f"backend={args.backend}")
print(f"verbose={verbose}")
print(f"optimize={args.optimize}")
print(f"with_mask={args.with_mask}")
print(f"implementation={args.implementation}")
print(f"mixed={args.mixed}")

if args.backend == "custom":
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

if is_cuda:
    print(
        f"CUDA model loaded: memory allocated={torch.cuda.memory_allocated(0)}, "
        f"reserved={torch.cuda.memory_reserved(0)}"
    )

print(f"Build the compile model with backend={args.backend}")
use_dynamic = args.dynamic in (1, "1", True, "True")
print(f"dynamic={use_dynamic}")
if verbose:
    print(f"-- debug backend, opset={args.target_opset}")
    for a in example_args_collection[0]:
        print(f"  input: {a.dtype}:{a.shape}")

compiled_model = create_compiled_model(
    model,
    backend=args.backend,
    use_dynamic=use_dynamic,
    target_opset=args.target_opset,
    verbose=verbose,
    enable_pattern=enable_pattern,
    disable_pattern=disable_pattern,
    optimize=optimize,
)


def loop_iteration(is_cuda, inputs, compiled_model, loss):
    if args.mixed in (1, "1", True, "True") and is_cuda:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            result = compiled_model(*inputs)
    else:
        assert args.mixed not in (
            1,
            "1",
            True,
            "True",
        ), f"not implemented with is_cuda={is_cuda}, mixed={args.mixed}"
        if is_cuda:
            torch.cuda.nvtx.range_push("DORT-FORWARD")
        result = compiled_model(*inputs)
        if is_cuda:
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()

    # dummy_target = torch.ones_like(result[0], memory_format=torch.contiguous_format)
    if is_cuda:
        torch.cuda.nvtx.range_push("DORT-ERROR")
    error = result[0].sum()  # loss(result[0], dummy_target)
    if is_cuda:
        torch.cuda.nvtx.range_pop()
    if is_cuda:
        torch.cuda.nvtx.range_push("DORT-BACKWARD")
    error.backward()
    if is_cuda:
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()


print(f"warmup on device={args.device}")
if is_cuda:
    print(
        f"CUDA memory allocated={torch.cuda.memory_allocated(0)}, "
        f"reserved={torch.cuda.memory_reserved(0)}"
    )

warmup_times = []
loss = torch.nn.MSELoss()
for i in range(args.warmup):
    example_inputs = example_args_collection[i]
    inputs = [t.to("cuda") for t in example_inputs] if is_cuda else example_inputs
    if is_cuda:
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    if args.backend in ("ort", "custom", "debug", "plug") and i == 0 and args.export:
        with dump_onnx(
            f"dort-{args.export}-{args.backend}", folder="dump_dort_bench", clean=True
        ):
            loop_iteration(is_cuda, inputs, compiled_model, loss)

        for onx in os.listdir("dump_dort_bench"):
            if not onx.endswith(".onnx"):
                continue
            new_onx = onx.replace(".onnx", ".opt.onnx")
            print(f"  ort_optimize {onx} -> {new_onx}")
            ort_optimize(
                os.path.join("dump_dort_bench", onx),
                output=os.path.join("dump_dort_bench", new_onx),
                providers=(
                    [("CUDAExecutionProvider", {}), ("CPUExecutionProvider", {})]
                    if is_cuda
                    else ["CPUExecutionProvider"]
                ),
            )
    else:
        if is_cuda:
            torch.cuda.nvtx.range_push("DORT-ITERATION")
        loop_iteration(is_cuda, inputs, compiled_model, loss)
        if is_cuda:
            torch.cuda.nvtx.range_pop()

    warmup_times.append(time.perf_counter() - start_time)

warmup_time = sum(warmup_times)
print(f"warmup done in {warmup_time}s.")
if is_cuda:
    print(
        f"memory allocated={torch.cuda.memory_allocated(0)}, "
        f"reserved={torch.cuda.memory_reserved(0)}"
    )

print("measures")
times = []
for example_inputs in example_args_collection[args.warmup :]:
    inputs = [t.to("cuda") for t in example_inputs] if is_cuda else example_inputs
    start_time = time.perf_counter()
    loop_iteration(is_cuda, inputs, compiled_model, loss)
    times.append(time.perf_counter() - start_time)

print("measures done.")
print(f"dynamic={args.dynamic}")
print(f"mixed={args.mixed}")
print(f"backend={args.backend}")
print(f"num_hidden_layers={args.num_hidden_layers}")
print(f"mixed={args.mixed}")
print(f"repeat={args.repeat}")
print(f"device={args.device}")
print(f"avg={np.mean(times)}")
print(f"times={times}")
print(f"warmup_times={warmup_times}")
print("-----------")

idims = "x".join(map(str, config_dict["input_dims"][0]))
del config_dict["input_dims"]
vals = "-".join(map(str, config_dict.values()))
print(f":{args.model},{idims}-{vals};")
print(f":config,{args.config};")
print(f":mixed,{args.mixed};")
print(f":dynamic,{use_dynamic};")
print(f":optimize,{optimize};")
print(f":backend,{args.backend};")
print(f":repeat,{args.repeat};")
print(f":warmup,{args.warmup};")
print(f":with_mask,{args.with_mask};")
print(f":implementation,{args.implementation};")
print(f":torch,{torch.__version__};")
print(f":transformers,{transformers.__version__};")
if args.backend in {"custom"}:
    print(f":patterns,+{args.enable_pattern}-{args.disable_pattern};")
print(f":warmup_time,{sum(warmup_times)};")
print(f":time,{np.mean(times)};")
