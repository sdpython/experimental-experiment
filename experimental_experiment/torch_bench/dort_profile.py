"""
Profile python execution for DORT
=================================

The script runs a few iterations of a dummy llama model.

::

    python -m experimental_experiment.torch_bench.dort_profile --help

Example, run llama model with onnxrt backend on cuda.

::

    python -m experimental_experiment.torch_bench.dort_profile --backend ort --device cuda

"""


def main():
    """
    Main function for command line
    ``python -m experimental_experiment.torch_bench.dort_profile``.
    """
    from experimental_experiment.torch_bench._dort_cmd_common import dort_args

    args = dort_args("experimental_experiment.torch_bench.dort_profile", description=__doc__)

    import os
    import time
    import numpy as np
    from onnx_array_api.profiling import profile, profile2graph
    import torch
    import torch._dynamo.backends.registry
    import transformers
    from experimental_experiment.convert.convert_helper import ort_optimize
    from experimental_experiment.torch_bench import BOOLEAN_VALUES
    from experimental_experiment.torch_models.llama_helper import get_llama_model
    from experimental_experiment.torch_models.dump_helper import dump_onnx
    from experimental_experiment.torch_bench._dort_cmd_common import (
        create_compiled_model,
        create_configuration_for_benchmark,
    )

    config_dict = create_configuration_for_benchmark(
        model="llama",
        config=args.config,
        repeat=args.repeat,
        warmup=args.warmup,
        num_hidden_layers=args.num_hidden_layers,
        implementation=args.implementation,
    )

    verbose = int(args.verbose)
    disable_pattern = [_ for _ in args.disable_pattern.split("+") if _]
    enable_pattern = [_ for _ in args.enable_pattern.split("+") if _]
    print(f"model config={config_dict}")
    print(f"backend={args.backend}")
    print(f"verbose={args.verbose}")
    print(f"implementation={args.implementation}")
    print(f"mixed={args.mixed}")

    if args.backend == "custom":
        print(f"disable_pattern={disable_pattern!r}")
        print(f"enable_pattern={enable_pattern!r}")

    is_cuda = args.device.startswith("cuda")
    if is_cuda:
        print(
            f"CUDA no model: memory allocated={torch.cuda.memory_allocated(0)}, "
            f"reserved={torch.cuda.memory_reserved(0)}"
        )

    model, example_args_collection = get_llama_model(**config_dict)

    device = args.device
    model = model.eval().to(device)

    if is_cuda:
        print(
            f"CUDA model loaded: memory allocated={torch.cuda.memory_allocated(0)}, "
            f"reserved={torch.cuda.memory_reserved(0)}"
        )

    print(f"Build the compile model with backend={args.backend}")
    use_dynamic = args.dynamic in BOOLEAN_VALUES
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
        ort_optimize=args.ort_optimize,
    )

    def loop_iteration(is_cuda, inputs, compiled_model, loss):
        if args.mixed in BOOLEAN_VALUES and is_cuda:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                result = compiled_model(*inputs)
        else:
            assert (
                args.mixed not in BOOLEAN_VALUES
            ), f"not implemented with is_cuda={is_cuda}, mixed={args.mixed}"
            result = compiled_model(*inputs)

        # dummy_target = torch.ones_like(result[0], memory_format=torch.contiguous_format)
        error = result[0].sum()  # loss(result[0], dummy_target)
        error.backward()
        if is_cuda:
            torch.cuda.synchronize()

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
            loop_iteration(is_cuda, inputs, compiled_model, loss)

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

    def main_loop():
        for example_inputs in example_args_collection[args.warmup :]:
            inputs = [t.to("cuda") for t in example_inputs] if is_cuda else example_inputs
            start_time = time.perf_counter()
            loop_iteration(is_cuda, inputs, compiled_model, loss)
            times.append(time.perf_counter() - start_time)

    ps = profile(main_loop)[0]

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
    print(f":llama,{idims}-{vals};")
    print(f":config,{args.config};")
    print(f":mixed,{args.mixed};")
    print(f":dynamic,{use_dynamic};")
    print(f":backend,{args.backend};")
    print(f":repeat,{args.repeat};")
    print(f":warmup,{args.warmup};")
    print(f":torch,{torch.__version__};")
    print(f":transformers,{transformers.__version__};")
    if args.backend in {"custom"}:
        print(f":patterns,+{args.enable_pattern}-{args.disable_pattern};")
    print(f":warmup_time,{sum(warmup_times)};")
    print(f":time,{np.mean(times)};")

    print("--------------------------------------------------------------------------")
    root, nodes = profile2graph(ps, clean_text=lambda x: "/".join(x.split("/")[-2:]))
    text = root.to_text()
    print(text)


if __name__ == "__main__":
    main()
