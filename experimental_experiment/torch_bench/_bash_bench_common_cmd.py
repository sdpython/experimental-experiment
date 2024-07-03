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
            "dummy",
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
        disable_pattern=("", "a list of optimization patterns to disable"),
        enable_pattern=("default", "list of optimization patterns to enable"),
        dump_folder=("dump_bash_bench", "where to dump the exported model"),
        output_data=(
            f"output_data_{name}.csv",
            "when running multiple configuration, save the results in that file",
        ),
        new_args=new_args,
    )
    return args


def bash_bench_main(name: str, doc: str, args: Optional[List[str]] = None):
    """
    Main command line for all bash_bench script.

    :param name: suffix for the bash
    :param doc: documentation
    :param args: optional arguments
    """

    args = bash_bench_parse_args("bash_bench_hugginfface.py", __doc__, new_args=args)

    from experimental_experiment.bench_run import (
        multi_run,
        make_configs,
        make_dataframe_from_benchmark_data,
        run_benchmark,
    )

    if name == "bash_bench_huggingface":
        from ._bash_bench_huggingface import HuggingfaceRunner

        runner = HuggingfaceRunner(device=args.device)
    else:
        raise AssertionError(f"Unexpected bash_bench name {name!r}.")
    names = runner.get_model_name_list()

    if not args.model:
        # prints the list of models.
        print(f"list of models for device={args.device}")
        print("--")
        print("\n".join([f"{i+1: 3d} - {n}" for i, n in enumerate(names)]))
        print("--")

    else:
        if args.model == "all":
            args.model = ",".join(names)

        if multi_run(args):
            configs = make_configs(args)
            data = run_benchmark(
                "experimental_experiment.torch_bench.bash_bench_huggingface",
                configs,
                args.verbose,
                stop_if_exception=False,
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

            runner = HuggingfaceRunner(
                include_model_names={name},
                verbose=args.verbose,
                device=args.device,
                target_opset=args.target_opset,
            )
            data = list(
                runner.enumerate_test_models(
                    process=args.process in ("1", 1, "True", True),
                    exporter=args.exporter,
                )
            )
            if len(data) == 1:
                for k, v in data[0].items():
                    print(f":{k},{v};")
            else:
                print(f"::model_name,{name};")
                print(f":device,{args.device};")
                print(f":ERROR,unexpected number of data {len(data)};")
