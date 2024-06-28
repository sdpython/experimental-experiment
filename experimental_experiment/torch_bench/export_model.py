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
            df.to_csv(args.output_data, index=False)
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
        with_mask = args.with_mask in (True, 1, "1", "True")
        disable_pattern = [_ for _ in args.disable_pattern.split("+") if _]
        enable_pattern = [_ for _ in args.enable_pattern.split("+") if _]
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

        use_dynamic = args.dynamic in (1, "1", True, "True")
        begin = time.perf_counter()
        model, example_args_collection, dynamic_shapes = create_model(
            args.model, config_dict, dynamic_shapes=use_dynamic, with_mask=with_mask
        )
        print(f"[export_model] model created in {time.perf_counter() - begin}")

        device = args.device
        model = model.eval().to(device)
        example_inputs = example_args_collection[0]
        inputs = (
            tuple([t.to("cuda") for t in example_inputs]) if is_cuda else example_inputs
        )

        if is_cuda:
            print(
                f"CUDA model loaded: memory allocated={torch.cuda.memory_allocated(0)}, "
                f"reserved={torch.cuda.memory_reserved(0)}"
            )

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

        if args.dynamic:
            print(f"[export_model] dynamic_shapes={dynamic_shapes}")
        msg = [tuple(i.shape for i in inp) for inp in example_inputs]
        print(f"[export_model] input_shapes={msg}")
        conversion = {}
        memory_stats = {}

        print(f"Exporter model with exporter={args.exporter}, n_inputs={len(inputs)}")
        if args.exporter == "eager":
            print("[export_model] start benchmark")
            begin = time.perf_counter()
            result = run_inference(
                model,
                example_inputs,
                warmup=args.warmup,
                repeat=args.repeat,
                verbose=args.verbose,
            )
            print(f"[export_model] benchmark done in {time.perf_counter() - begin}")
        else:
            print(
                f"[export_model] export to onnx with exporter={args.exporter!r} "
                f"and optimization={args.optimization!r}"
            )
            begin = time.perf_counter()

            memory_session = (
                start_spying_on(cuda=args.device == "cuda")
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
                ort_optimize=args.ort_optimize,
                optimization=args.optimization,
                verbose=args.verbose,
                stats=conversion,
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
