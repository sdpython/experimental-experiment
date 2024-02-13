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

import time
import onnxruntime
import numpy as np
import torch
import torch._dynamo.backends.registry
from experimental_experiment.llama.llama_helper import get_llama_model
from experimental_experiment.args import get_parsed_args
from experimental_experiment.convert_helper import optimize_model_proto
from experimental_experiment.torch_dynamo import onnx_custom_backend
from experimental_experiment.torch_helper.training_helper import make_aot_ort


args = get_parsed_args(
    "experimental_experiment.torch_bench.dort_bench",
    description=__doc__,
    backend=("ort", "'ort' or 'inductor' or 'eager' or 'custom'"),
    device=("cpu", "'cpu' or 'cuda'"),
    num_hidden_layers=(1, "number of hidden layers"),
    warmup=5,
    repeat=5,
    mixed=(0, "mixed precision (based on autocast)"),
    export=(
        "",
        "export the model with dynamo and torch.script, " "use this as a prefix",
    ),
    config=("default", "default or small to test"),
    expose="backend,repeat,warmup,device,num_hidden_layers,mixed,export,config",
)


if args.config == "small":
    model, example_args_collection = get_llama_model(
        input_dims=[(2, 1024)] * (args.repeat + args.warmup),
        _attn_implementation="eager",
        num_hidden_layers=args.num_hidden_layers,
        hidden_size=16,
        vocab_size=1024,
        intermediate_size=16,
        max_position_embeddings=1024,
        num_attention_heads=2,
    )
else:
    model, example_args_collection = get_llama_model(
        input_dims=[(2, 1024)] * (args.repeat + args.warmup),
        _attn_implementation="eager",
        num_hidden_layers=args.num_hidden_layers,
        hidden_size=4096,
        vocab_size=32000,
        intermediate_size=11008,
        max_position_embeddings=2048,
        num_attention_heads=32,
    )


device = args.device
model = model.eval().to(device)


local_aot_ort, local_ort = make_aot_ort(dynamic=True)

if args.backend == "ort":
    compiled_model = torch.compile(model, backend=local_ort)
elif args.backend == "inductor":
    compiled_model = torch.compile(model, backend="inductor")
elif args.backend == "eager":
    compiled_model = model
elif args.backend == "custom":
    compiled_model = torch.compile(model, backend=onnx_custom_backend)
else:
    raise ValueError(f"Unexpected backend={args.backend!r}.")


def loop_iteration(is_cuda, inputs, compiled_model):
    if args.mixed and is_cuda:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            result = compiled_model(*inputs)
    else:
        result = compiled_model(*inputs)

    dummy_loss = torch.ones_like(result[0], memory_format=torch.contiguous_format)
    result[0].backward(dummy_loss)
    if is_cuda:
        torch.cuda.synchronize()


if args.export:
    providers = (
        ["CPUExecutionProvider"]
        if device == "cpu"
        else ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    filename = f"{args.export}.script.onnx"
    print("export with torch.onnx.export to {filename!r}")
    input_names = ["input{i}" for i in range(len(example_args_collection[0]))]
    torch.onnx.export(model, *example_args_collection[0], filename, input_names)

    ofilename = f"{args.export}.script.opt.onnx"
    print("onnxruntime optimization to {ofilename!r}")
    opts = onnxruntime.SessionOptions()
    opts.optimized_model_filepath = ofilename
    sess = onnxruntime.InferenceSession(filename, opts, providers=providers)

    filename = f"{args.export}.dynamo.onnx"
    print("export with torch.onnx.dynamo_export to {filename!r}")
    export_output = torch.onnx.dynamo_export(model, *args)
    optimized_model = optimize_model_proto(export_output.model_proto)
    with open(filename, "wb") as f:
        f.write(optimized_model.SerializeToString())

    ofilename = f"{args.export}.dynamo.opt.onnx"
    print("onnxruntime optimization to {ofilename!r}")
    opts = onnxruntime.SessionOptions()
    opts.optimized_model_filepath = ofilename
    sess = onnxruntime.InferenceSession(filename, opts, providers=providers)

print("warmup")
warmup_times = []
is_cuda = args.device == "cuda"
for i in range(args.warmup):
    example_inputs = example_args_collection[i]
    inputs = [t.to("cuda") for t in example_inputs] if is_cuda else example_inputs
    start_time = time.perf_counter()
    loop_iteration(is_cuda, inputs, compiled_model)
    warmup_times.append(time.perf_counter() - start_time)

warmup_time = time.perf_counter() - start_time
print(f"warmup done in {warmup_time}s.")

print("measures")
times = []
for example_inputs in example_args_collection[args.warmup :]:
    inputs = [t.to("cuda") for t in example_inputs] if is_cuda else example_inputs
    start_time = time.perf_counter()
    loop_iteration(is_cuda, inputs, compiled_model)
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
