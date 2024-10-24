"""
Run llama model with DORT
=========================

The script runs a few iterations of a dummy llama model.

::

    python -m experimental_experiment.torch_bench.dort_bench --help

Example, run llama model with onnxrt backend on cuda.

::

    python -m experimental_experiment.torch_bench.dort_bench \\
           --backend ort --device cuda --config medium

To export the models:

::

    python -m experimental_experiment.torch_bench.dort_bench \\
           --backend custom --device cuda --export a -w 3


Profiling:

::

    nsys profile python -m experimental_experiment.torch_bench.dort_bench \\
                        --device cuda -w 3 -r 5 --mixed 1 --config large \\
                        --backend eager --enable_pattern=default+onnxruntime

With experimental optimizers:

::

    python -m experimental_experiment.torch_bench.dort_bench --backend custom \\
           --device cuda --mixed=1 --export model -w 3 \\
           --enable_pattern=default+onnxruntime+experimental

Or:

::

    python -m experimental_experiment.torch_bench.dort_bench --backend ort+ \\
          --device cuda --mixed=1 --export model -w 3 \\
          --enable_pattern=default+onnxruntime+experimental
"""

import os
import pprint


def main(args=None):
    from experimental_experiment.torch_bench._dort_cmd_common import dort_args

    args = dort_args(
        "experimental_experiment.torch_bench.dort_bench",
        description=__doc__,
        new_args=args,
    )

    from ..bench_run import (
        multi_run,
        make_configs,
        make_dataframe_from_benchmark_data,
        run_benchmark,
    )

    if multi_run(args):
        configs = make_configs(args)
        data = run_benchmark(
            "experimental_experiment.torch_bench.dort_bench",
            configs,
            args.verbose,
            stop_if_exception=False,
        )
        if args.verbose > 2:
            pprint.pprint(data if args.verbose > 3 else data[:2])
        if args.output_data:
            df = make_dataframe_from_benchmark_data(data, detailed=False)
            df.to_csv(args.output_data, index=False, errors="ignore")
            df.to_excel(args.output_data + ".xlsx", index=False)
            if args.verbose:
                print(df)
    else:
        import logging
        import time
        import onnxruntime  # noqa: F401
        import numpy as np
        import torch
        import torch._dynamo.backends.registry
        import transformers
        from experimental_experiment.torch_bench import BOOLEAN_VALUES
        from experimental_experiment.convert.convert_helper import (
            ort_optimize as run_ort_optimize,
        )
        from experimental_experiment.torch_models.dump_helper import dump_onnx
        from experimental_experiment.torch_bench._dort_cmd_common import (
            create_compiled_model,
            create_configuration_for_benchmark,
            create_model,
        )
        from experimental_experiment.memory_peak import start_spying_on, flatten

        config_dict = create_configuration_for_benchmark(
            model=args.model,
            config=args.config,
            repeat=args.repeat,
            warmup=args.warmup,
            num_hidden_layers=args.num_hidden_layers,
            implementation=args.implementation,
            with_mask=args.with_mask,
            shape_scenario=args.shape_scenario,
        )

        verbose = int(args.verbose)
        optimize = args.optimize in BOOLEAN_VALUES
        ort_optimize = args.ort_optimize in BOOLEAN_VALUES
        with_mask = args.with_mask in BOOLEAN_VALUES
        disable_pattern = [_ for _ in args.disable_pattern.split("+") if _]
        enable_pattern = [_ for _ in args.enable_pattern.split("+") if _]
        print(f"model={args.model}")
        print(f"model config={config_dict}")
        print(f"backend={args.backend}")
        print(f"verbose={verbose}")
        print(f"optimize={args.optimize}")
        print(f"ort_optimize={ort_optimize}")
        print(f"order_algorithm={args.order}")
        print(f"with_mask={with_mask}")
        print(f"implementation={args.implementation}")
        print(f"mixed={args.mixed}")
        print(f"shape_scenario={args.shape_scenario}")
        dump_patterns = args.dump_patterns in BOOLEAN_VALUES

        if args.backend == "custom":
            print(f"disable_pattern={disable_pattern!r}")
            print(f"enable_pattern={enable_pattern!r}")
        assert not dump_patterns or args.export, (
            f"optimization patterns cannot be dumped if export is not set "
            f"dump_patterns={dump_patterns!r}, export={args.export}"
        )

        is_cuda = args.device.startswith("cuda")
        if is_cuda:
            print(
                f"CUDA no model: memory allocated={torch.cuda.memory_allocated(0)}, "
                f"reserved={torch.cuda.memory_reserved(0)}"
            )

        device = args.device
        model, example_args_collection = create_model(args.model, config_dict)

        if args.backend != "ortmodule":
            model = model.eval()
        model = model.to(device)

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

        dump_folder = args.dump_folder

        if args.export and dump_folder and not os.path.exists(dump_folder):
            os.makedirs(dump_folder)

        if dump_patterns:
            dump_patterns_folder = os.path.join(dump_folder, "patterns")
            if os.path.exists(dump_patterns_folder):
                for _ in os.listdir(dump_patterns_folder):
                    if _.endswith(".onnx"):
                        os.remove(os.path.join(dump_patterns_folder, _))
        else:
            dump_patterns_folder = None
        if verbose:
            if dump_patterns:
                print(
                    f"dump models and patterns in {dump_folder!r} "
                    f"and {dump_patterns_folder!r}"
                )
            else:
                print(f"dump models in {dump_folder!r}")

        logger = logging.getLogger("onnxscript.optimizer.constant_folding")
        logger.setLevel(logging.ERROR)

        compiled_model = create_compiled_model(
            model,
            backend=args.backend,
            use_dynamic=use_dynamic,
            target_opset=args.target_opset,
            verbose=verbose,
            enable_pattern=enable_pattern,
            disable_pattern=disable_pattern,
            optimize=optimize,
            ort_optimize=ort_optimize,
            use_fused_aten_ops=args.implementation == "sdpa",
            dump_prefix=(
                f"{dump_folder}/{args.export}-{args.model}-{args.backend}"
                if args.export
                else None
            ),
            dump_patterns=dump_patterns_folder,
            processor=device.upper() if device.upper() == "CPU" else "CPU,CUDA",
            order_algorithm=args.order,
        )
        del model

        print(f"type of compiled_model={type(compiled_model)}")

        def loop_iteration(is_cuda, inputs, compiled_model, loss):
            torch.set_grad_enabled(True)

            mixed = args.mixed in BOOLEAN_VALUES
            if mixed and is_cuda:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    torch.cuda.nvtx.range_push("DORT-FORWARD-MIXED")
                    result = compiled_model(*inputs)
                    torch.cuda.synchronize()
                    torch.cuda.nvtx.range_pop()
            elif is_cuda:
                torch.cuda.nvtx.range_push("DORT-FORWARD")
                result = compiled_model(*inputs)
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
            else:
                result = compiled_model(*inputs)

            # dummy_target = torch.ones_like(result[0],
            # memory_format=torch.contiguous_format)
            if mixed and is_cuda:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    torch.cuda.nvtx.range_push("DORT-ERROR-MIXED")
                    error = result[0].sum()  # loss(result[0], dummy_target)
                    torch.cuda.nvtx.range_pop()
                    torch.cuda.nvtx.range_push("DORT-BACKWARD-MIXED")
                    error.backward()
                    torch.cuda.synchronize()
                    torch.cuda.nvtx.range_pop()
            elif is_cuda:
                torch.cuda.nvtx.range_push("DORT-ERROR")
                error = result[0].sum()  # loss(result[0], dummy_target)
                torch.cuda.nvtx.range_pop()
                torch.cuda.nvtx.range_push("DORT-BACKWARD")
                error.backward()
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
            else:
                error = result[0].sum()  # loss(result[0], dummy_target)
                error.backward()

        print(f"warmup on device={args.device}")
        if is_cuda:
            print(
                f"CUDA memory allocated={torch.cuda.memory_allocated(0)}, "
                f"reserved={torch.cuda.memory_reserved(0)}"
            )

        if args.memory_spy in ("1", 1, "True", "true", True):
            memory = start_spying_on(cuda=is_cuda)
        else:
            memory = None

        warmup_times = []
        loss = torch.nn.MSELoss()
        for i in range(args.warmup):
            example_inputs = example_args_collection[i]
            inputs = [t.to("cuda") for t in example_inputs] if is_cuda else example_inputs
            if is_cuda:
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            if (
                args.backend in ("ort", "custom", "debug", "plug", "ort+")
                and i == 0
                and args.export
            ):
                with dump_onnx(
                    f"dort-{args.export}-{args.model}-{args.backend}",
                    folder=dump_folder,
                    clean=True,
                ):
                    loop_iteration(is_cuda, inputs, compiled_model, loss)

                for onx in os.listdir(dump_folder):
                    if not onx.endswith(".onnx"):
                        continue
                    if ".opt." in onx:
                        continue
                    new_onx = onx.replace(".onnx", ".opt.onnx")
                    print(f"  ort_optimize {onx} -> {new_onx}")
                    run_ort_optimize(
                        os.path.join(dump_folder, onx),
                        output=os.path.join(dump_folder, new_onx),
                        providers=(
                            [
                                ("CUDAExecutionProvider", {}),
                                ("CPUExecutionProvider", {}),
                            ]
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
        if memory is not None:
            stat_memory = flatten(memory.stop(), prefix="memory_")
            print(stat_memory)
        else:
            stat_memory = None

        i_shapes = set(config_dict["input_dims"])
        if len(i_shapes) == 1:
            idims = "x".join(str(i) for i in i_shapes)
        else:
            idims = "|".join("x".join(map(str, shs)) for shs in list(i_shapes)[:2])
        del config_dict["input_dims"]
        vals = "-".join(map(str, config_dict.values()))
        print(f":{args.model},{idims}-{vals};")
        print(f":config,{args.config};")
        print(f":mixed,{args.mixed};")
        print(f":dynamic,{use_dynamic};")
        print(f":optimize,{optimize};")
        print(f":order,{args.order};")
        print(f":ort_optimize,{ort_optimize};")
        print(f":backend,{args.backend};")
        print(f":repeat,{args.repeat};")
        print(f":warmup,{args.warmup};")
        print(f":with_mask,{args.with_mask};")
        print(f":implementation,{args.implementation};")
        print(f":torch,{torch.__version__};")
        print(f":transformers,{transformers.__version__};")
        if stat_memory:
            for k, v in stat_memory.items():
                print(f":{k},{v};")
        if args.backend in {"custom", "ort+", "debug"}:
            suffix = "+oo" if args.ort_optimize else ""
            print(f":patterns,+{args.enable_pattern}-{args.disable_pattern}{suffix};")
        print(f":warmup_time,{sum(warmup_times)};")
        print(f":time,{np.mean(times)};")


if __name__ == "__main__":
    main()
