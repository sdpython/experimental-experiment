"""
Run llama model with DORT
=========================

The script runs a few iterations of a dummy llama model.

::

    python -m experimental_experiment.llama.dort_bench --help

Example, run llama model with onnxrt backend on cuda.

::

    python -m experimental_experiment.torch_bench.dort_bench --backend ort --device cuda
    
Other example, same script but dumps the produces models.

::

    ONNXRT_DUMP_PATH="llama_dort_" python -m experimental_experiment.torch_bench.dort_bench --backend ort --device cuda
"""

from experimental_experiment.args import get_parsed_args

args = get_parsed_args(
    "experimental_experiment.torch_bench.dort_bench",
    description=__doc__,
    backend=("ort", "'ort' or 'inductor' or 'eager' or 'custom'"),
    device=("cpu", "'cpu' or 'cuda'"),
    num_hidden_layers=(1, "number of hidden layers"),
    warmup=5,
    repeat=5,
    mixed=(0, "mixed precision (based on autocast)"),
    export=("", "export the dynamo models"),
    target_opset=(18, "opset to convert into, use with backend=custom"),
    config=("default", "default or small to test"),
    expose="backend,repeat,warmup,device,num_hidden_layers,"
    "mixed,export,config,target_opset",
)

import os
import time
import onnxruntime  # noqa: F401
import numpy as np
import torch
import torch._dynamo.backends.registry
from torch._dynamo.backends.common import aot_autograd
from experimental_experiment.convert.convert_helper import ort_optimize
from experimental_experiment.torch_helper.llama_helper import get_llama_model
from experimental_experiment.torch_helper.training_helper import make_aot_ort
from experimental_experiment.torch_helper.dump_helper import dump_onnx
from experimental_experiment.torch_dynamo import get_decomposition_table
from experimental_experiment.torch_dynamo import onnx_custom_backend, onnx_debug_backend


if args.config == "small":
    config_dict = dict(
        input_dims=[(2, 1024)] * (args.repeat + args.warmup),
        hidden_size=16,
        num_hidden_layers=args.num_hidden_layers,
        vocab_size=1024,
        intermediate_size=16,
        max_position_embeddings=1024,
        num_attention_heads=2,
        _attn_implementation="eager",
    )
else:
    config_dict = dict(
        input_dims=[(2, 1024)] * (args.repeat + args.warmup),
        hidden_size=4096,
        num_hidden_layers=args.num_hidden_layers,
        vocab_size=32000,
        intermediate_size=11008,
        max_position_embeddings=2048,
        num_attention_heads=32,
        _attn_implementation="eager",
    )

print(f"llama config={config_dict}")
print(f"backend={args.backend}")
model, example_args_collection = get_llama_model(**config_dict)


device = args.device
model = model.eval().to(device)


if args.backend == "ort":
    local_aot_ort, local_ort = make_aot_ort(dynamic=True)
    compiled_model = torch.compile(model, backend=local_ort)
elif args.backend == "inductor":
    compiled_model = torch.compile(model, backend="inductor")
elif args.backend == "eager":
    compiled_model = model
elif args.backend == "custom":
    get_decomposition_table
    target_opset = args.target_opset
    aot_compiler = aot_autograd(
        fw_compiler=lambda *args, **kwargs: onnx_custom_backend(
            *args, target_opset=target_opset, **kwargs
        ),
        decompositions=get_decomposition_table(),
    )
    compiled_model = torch.compile(model, backend=aot_compiler, fullgraph=True)
elif args.backend == "debug":
    target_opset = args.target_opset
    print(f"-- debug backend, opset={target_opset}")
    for a in example_args_collection[0]:
        print(f"  input: {a.dtype}:{a.shape}")
    aot_compiler = aot_autograd(
        fw_compiler=lambda *args, **kwargs: onnx_debug_backend(
            *args, target_opset=target_opset, backend="ref", **kwargs
        ),
        decompositions=get_decomposition_table(),
    )
    compiled_model = torch.compile(model, backend=aot_compiler, fullgraph=True)
else:
    raise ValueError(f"Unexpected backend={args.backend!r}.")


def loop_iteration(is_cuda, inputs, compiled_model, loss):
    if args.mixed and is_cuda:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            result = compiled_model(*inputs)
    else:
        result = compiled_model(*inputs)

    # dummy_target = torch.ones_like(result[0], memory_format=torch.contiguous_format)
    error = result[0].sum()  # loss(result[0], dummy_target)
    error.backward()
    if is_cuda:
        torch.cuda.synchronize()


print("warmup")
warmup_times = []
is_cuda = args.device == "cuda"
loss = torch.nn.MSELoss()
for i in range(args.warmup):
    example_inputs = example_args_collection[i]
    inputs = [t.to("cuda") for t in example_inputs] if is_cuda else example_inputs
    start_time = time.perf_counter()
    if args.backend in ("ort", "custom", "debug") and i == 0 and args.export:
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
                    ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if is_cuda
                    else ["CPUExecutionProvider"]
                ),
            )
    else:
        loop_iteration(is_cuda, inputs, compiled_model, loss)
    warmup_times.append(time.perf_counter() - start_time)

warmup_time = sum(warmup_times)
print(f"warmup done in {warmup_time}s.")

print("measures")
times = []
for example_inputs in example_args_collection[args.warmup :]:
    inputs = [t.to("cuda") for t in example_inputs] if is_cuda else example_inputs
    start_time = time.perf_counter()
    loop_iteration(is_cuda, inputs, compiled_model, loss)
    times.append(time.perf_counter() - start_time)

print("measures done.")

print(f"backend={args.backend}")
print(f"num_hidden_layers={args.num_hidden_layers}")
print(f"mixed={args.mixed}")
print(f"repeat={args.repeat}")
print(f"device={args.device}")
print(f"avg={np.mean(times)}")
print(f"times={times}")
print(f"warmup_times={warmup_times}")
print(f":time,{np.mean(times)};")
print(f":warmup_time,{sum(warmup_times)};")
print(f":torch,{torch.__file__};")
