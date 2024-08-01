"""
Export a model
==============

The script export a model with different options.

::

    python -m experimental_experiment.torch_bench.export_model --help

Example, run llama model with onnxrt backend on cuda.

::

    python -m experimental_experiment.torch_bench.export_model --exporter script --device cuda --config medium
    
"""

import pprint


def main(args=None):

    from experimental_experiment.torch_bench._dort_cmd_common import export_args

    args = export_args(
        "experimental_experiment.torch_bench.export_model",
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
            "experimental_experiment.torch_bench.export_model",
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

        import os
        import time
        import torch
        import torch._dynamo.backends.registry
        from experimental_experiment.torch_bench._dort_cmd_common import (
            create_configuration_for_benchmark,
            create_model,
        )
        from experimental_experiment.memory_peak import start_spying_on, flatten
        from .export_model_helper import (
            run_onnx_inference,
            run_inference,
            common_export,
        )

        verbose = int(args.verbose)
        use_dynamic = args.dynamic in (1, "1", True, "True")
        with_mask = args.with_mask in (True, 1, "1", "True")
        order = args.order in (True, 1, "1", "True")
        large_model = args.large_model in (True, 1, "1", "True")
        disable_pattern = [_ for _ in args.disable_pattern.split("+") if _]
        enable_pattern = [_ for _ in args.enable_pattern.split("+") if _]

        config_dict = create_configuration_for_benchmark(
            model=args.model,
            config=args.config,
            repeat=1,
            warmup=1,
            num_hidden_layers=args.num_hidden_layers,
            implementation=args.implementation,
            with_mask=with_mask,
            dynamic_shapes=use_dynamic,
        )

        print(f"model={args.model}")
        print(f"model config={config_dict}")
        print(f"exporter={args.exporter}")
        print(f"verbose={verbose}")
        print(f"optimize={args.optimize}")
        print(f"dtype={args.dtype}")
        print(f"implementation={args.implementation}")
        print(f"with_mask={args.with_mask}")
        print(f"order={args.order}")
        print(f"dump_folder={args.dump_folder}")

        if args.exporter == "custom":
            print(f"disable_pattern={disable_pattern!r}")
            print(f"enable_pattern={enable_pattern!r}")

        is_cuda = args.device.startswith("cuda")
        if is_cuda:
            print(
                f"CUDA no model: memory allocated={torch.cuda.memory_allocated(0)}, "
                f"reserved={torch.cuda.memory_reserved(0)}"
            )

        begin = time.perf_counter()
        res = create_model(args.model, config_dict, dtype=args.dtype)
        model, example_args_collection = res[:2]
        dynamic_shapes = None if len(res) == 2 else res[2]
        print(f"[export_model] model created in {time.perf_counter() - begin}")

        device = args.device
        model = model.eval().to(device)
        example_inputs = [
            tuple(t.to(device) for t in example_inputs)
            for example_inputs in example_args_collection
        ]

        if is_cuda:
            print(
                f"CUDA model loaded: memory allocated={torch.cuda.memory_allocated(0)}, "
                f"reserved={torch.cuda.memory_reserved(0)}"
            )

        print(f"dynamic={use_dynamic}")

        folder = args.dump_folder
        if folder and not os.path.exists(folder):
            os.makedirs(folder)

        filename = (
            f"export_{args.model}_{args.exporter}_{'dyn' if use_dynamic else 'static'}"
            f"{args.dtype}_{args.config}-v{args.target_opset}"
            f".{args.implementation}.onnx"
        )

        if args.dynamic:
            print(f"[export_model] dynamic_shapes={dynamic_shapes}")
        msg = [tuple(i.shape for i in inp) for inp in example_inputs]
        print(f"[export_model] input_shapes={msg}")
        conversion = {}
        memory_stats = {}

        print(
            f"Exporter model with exporter={args.exporter}, n_inputs={len(example_inputs[0])}"
        )
        if args.exporter == "eager":
            print("[export_model] start benchmark")
            begin = time.perf_counter()
            result = run_inference(
                model,
                example_inputs[0],
                warmup=args.warmup,
                repeat=args.repeat,
                verbose=args.verbose,
            )
            print(f"[export_model] benchmark done in {time.perf_counter() - begin}")
        else:
            print(f"[export_model] export to onnx with exporter={args.exporter!r}")
            begin = time.perf_counter()

            memory_session = (
                start_spying_on(cuda=args.device.startswith("cuda"))
                if args.memory_peak
                else None
            )
            print(f"[export_model] start memory peak monitoring {memory_session}")
            proto = common_export(
                model=model,
                inputs=example_inputs[0],
                exporter=args.exporter,
                target_opset=args.target_opset,
                folder=args.dump_folder,
                filename=filename,
                dynamic_shapes=dynamic_shapes if args.dynamic else None,
                ort_optimize=args.ort in ("1", 1, "True", True),
                optimize_oxs=args.optimize in ("1", 1, "True", True),
                enable_pattern=args.enable_pattern,
                disable_pattern=args.disable_pattern,
                verbose=args.verbose,
                stats=conversion,
                large_model=large_model,
                order=order,
            )
            print(
                f"[export_model] export to onnx done in {time.perf_counter() - begin}"
            )
            if memory_session is not None:
                memory_results = memory_session.stop()
                print(f"[export_model] ends memory monitoring {memory_results}")
                memory_stats = flatten(memory_results, prefix="memory_")
            else:
                memory_stats = {}

            result = run_onnx_inference(
                proto,
                example_inputs,
                warmup=args.warmup,
                repeat=args.repeat,
                verbose=args.verbose,
                ort_optimize=args.ort_optimize,
                torch_model=model,
            )

        print("[export_model] end")
        print("------------------------------")
        for k, v in sorted(args.__dict__.items()):
            print(f":{k},{v};")
        for k, v in sorted(conversion.items()):
            print(f":{k},{v};")
        if memory_stats:
            for k, v in memory_stats.items():
                print(f":{k},{v};")
        for k, v in sorted(result.items()):
            print(f":{k},{v};")


if __name__ == "__main__":
    main()
