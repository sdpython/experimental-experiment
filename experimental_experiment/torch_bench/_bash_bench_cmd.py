import os
import pprint
from typing import Optional, List


def bash_bench_parse_args(name: str, doc: str, new_args: Optional[List[str]] = None):
    """
    Returns parsed arguments.
    """
    from experimental_experiment.args import get_parsed_args

    args = get_parsed_args(
        f"experimental_experiment.torch_bench.{name}",
        description=doc,
        model=(
            "101Dummy",
            "if empty, prints the list of models, "
            "all for all models, a list of indices works as well",
        ),
        exporter=(
            "custom",
            "export, custom, dynamo, dynamo2, script",
        ),
        process=("0", "run every run in a separate process"),
        device=("cpu", "'cpu' or 'cuda'"),
        dynamic=("0", "use dynamic shapes"),
        target_opset=("18", "opset to convert into, use with backend=custom"),
        verbose=("0", "verbosity"),
        opt_patterns=("", "a list of optimization patterns to disable"),
        dump_folder=("dump_bash_bench", "where to dump the exported model"),
        quiet=("1", "catch exception and go on or fail"),
        dtype=(
            "",
            "converts the model using this type, empty for no change, "
            "possible value, float16, float32, ...",
        ),
        output_data=(
            f"output_data_{name}.csv",
            "when running multiple configuration, save the results in that file",
        ),
        memory_peak=(
            "0",
            "measure the memory peak during exporter, "
            "it starts another process to monitor the memory",
        ),
        new_args=new_args,
        expose="repeat,warmup",
    )
    return args


def bash_bench_main(name: str, doc: str, args: Optional[List[str]] = None):
    """
    Main command line for all bash_bench script.

    :param name: suffix for the bash
    :param doc: documentation
    :param args: optional arguments
    """

    args = bash_bench_parse_args("bash_bench_huggingface.py", __doc__, new_args=args)
    print(f"[{name}] start")
    for k, v in sorted(args.__dict__.items()):
        print(f"{k}={v}")

    from experimental_experiment.bench_run import (
        multi_run,
        make_configs,
        make_dataframe_from_benchmark_data,
        run_benchmark,
    )

    if name == "bash_bench_huggingface":
        from ._bash_bench_set_huggingface import HuggingfaceRunner

        runner = HuggingfaceRunner(device=args.device)
    else:
        raise AssertionError(f"Unexpected bash_bench name {name!r}.")
    names = runner.get_model_name_list()

    if not args.model and args.model not in ("0", 0):
        # prints the list of models.
        print(f"list of models for device={args.device} (args.model={args.model!r})")
        print("--")
        print("\n".join([f"{i: 3d} - {n}" for i, n in enumerate(names)]))
        print("--")

    else:
        if args.model == "all":
            args.model = ",".join(names)
        elif args.model == "All":
            args.model = ",".join(n for n in names if not n.startswith("101"))

        if multi_run(args):
            configs = make_configs(args)
            if args.output_data:
                name, ext = os.path.splitext(args.output_data)
                temp_output_data = f"{name}.temp{ext}"
            else:
                temp_output_data = None
            data = run_benchmark(
                "experimental_experiment.torch_bench.bash_bench_huggingface",
                configs,
                args.verbose,
                stop_if_exception=False,
                temp_output_data=temp_output_data,
                dump_std="dump_test_models",
            )
            if args.verbose > 2:
                pprint.pprint(data if args.verbose > 3 else data[:2])
            if args.output_data:
                df = make_dataframe_from_benchmark_data(data, detailed=False)
                print("Prints the results into file {args.output_data!r}")
                df.to_csv(args.output_data, index=False)
                df.to_excel(args.output_data + ".xlsx", index=False)
                if args.verbose:
                    print(df)

        else:
            try:
                indice = int(args.model)
                name = names[indice]
            except (TypeError, ValueError):
                name = args.model

            if args.verbose:
                print(f"Running model {name!r}")

            runner = HuggingfaceRunner(
                include_model_names={name},
                verbose=args.verbose,
                device=args.device,
                target_opset=args.target_opset,
                repeat=args.repeat,
                warmup=args.warmup,
                dtype=args.dtype,
            )
            data = list(
                runner.enumerate_test_models(
                    process=args.process in ("1", 1, "True", True),
                    exporter=args.exporter,
                    quiet=args.quiet in ("1", 1, "True", True),
                    folder="dump_test_models",
                    optimization=args.opt_patterns,
                    memory_peak=args.memory_peak in ("1", 1, "True", True),
                )
            )
            if len(data) == 1:
                for k, v in sorted(data[0].items()):
                    print(f":{k},{v};")
            else:
                print(f":model_name,{name};")
                print(f":device,{args.device};")
                print(f":ERROR,unexpected number of data {len(data)};")
