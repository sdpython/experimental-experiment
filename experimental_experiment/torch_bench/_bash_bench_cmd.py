import os
import pprint
import time
from datetime import datetime
from typing import List, Optional
import numpy as np


def bash_bench_parse_args(name: str, doc: str, new_args: Optional[List[str]] = None):
    """Returns parsed arguments."""
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
            "export, export-nostrict, export-default, "
            "inductor, "
            "custom, custom-fallback, custom-nostrict, custom-tracing, "
            "onnx_dynamo, onnx_dynamo-fallback, torch_script",
        ),
        process=("0", "run every run in a separate process"),
        device=("cpu", "'cpu' or 'cuda'"),
        dynamic=("0", "use dynamic shapes"),
        target_opset=(
            "18",  # from ..torch_interpreter import DEFAULT_TARGET_OPSET
            "opset to convert into, use with backend=custom",
        ),
        verbose=("0", "verbosity"),
        opt_patterns=("", "a list of optimization patterns to disable"),
        dump_folder=("dump_bash_bench", "where to dump the exported model"),
        quiet=("1", "catch exception and go on or fail"),
        start=("0", "first model to run (to continue a bench)"),
        rtopt=("1", "runtime optimization are enabled"),
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
        nvtx=("0", "add events to profile"),
        profile=(
            "0",
            "run a profiling to see which python function is taking the most time",
        ),
        dump_ort=("0", "dump the onnxruntime optimized graph"),
        split_process=("0", "run exporter and the inference in two separate processes"),
        part=("", "which part to run, 0, or 1"),
        tag=("", "add a version tag when everything else did not change"),
        timeout=("600", "timeout for subprocesses"),
        shape2=("0", "redo the shape inference"),
        patch=("1", "patches torch and transformers before exporting"),
        attn_impl=("eager", "attention implementation"),
        new_args=new_args,
        expose="repeat,warmup",
    )
    return args


def _clean_text(text):
    import onnx
    import onnxruntime
    import onnxscript
    import torch
    import experimental_experiment

    pathes = [
        os.path.abspath(os.path.normpath(os.path.join(os.path.dirname(m.__file__), "..")))
        for m in [onnx, onnxruntime, onnxscript, np, torch, experimental_experiment]
    ]
    for p in pathes:
        text = text.replace(p, "")
    text = text.replace("experimental_experiment", "experimental_experiment".upper())
    return text


def bash_bench_main(script_name: str, doc: str, args: Optional[List[str]] = None):
    """
    Main command line for all bash_bench script.

    :param script_name: suffix for the bash
    :param doc: documentation
    :param args: optional arguments
    """
    args = bash_bench_parse_args(f"{script_name}.py", __doc__, new_args=args)
    print(f"[{script_name}] start")
    for k, v in sorted(args.__dict__.items()):
        print(f"{k}={v}")

    from experimental_experiment.torch_bench import BOOLEAN_VALUES
    from experimental_experiment.torch_bench._bash_bench_benchmark_runner_agg import (
        merge_benchmark_reports,
    )
    from experimental_experiment.torch_bench._bash_bench_model_runner import ModelRunner
    from experimental_experiment.bench_run import (
        make_configs,
        make_dataframe_from_benchmark_data,
        multi_run,
        run_benchmark,
    )

    if script_name == "bash_bench_huggingface":
        from ._bash_bench_set_huggingface import HuggingfaceRunner

        runner = HuggingfaceRunner(device=args.device, verbose=int(args.verbose))
    elif script_name == "bash_bench_huggingface_big":
        from ._bash_bench_set_huggingface_big import HuggingfaceBigRunner

        runner = HuggingfaceBigRunner(device=args.device, verbose=int(args.verbose))
    elif script_name == "bash_bench_issues":
        from ._bash_bench_set_issues import IssueRunner

        runner = IssueRunner(device=args.device, verbose=int(args.verbose))
    elif script_name == "bash_bench_torchbench":
        from ._bash_bench_set_torchbench import TorchBenchRunner

        runner = TorchBenchRunner(device=args.device, verbose=int(args.verbose))
    elif script_name == "bash_bench_torchbench_ado":
        from ._bash_bench_set_torchbench_ado import TorchBenchAdoRunner

        runner = TorchBenchAdoRunner(device=args.device, verbose=int(args.verbose))
    elif script_name == "bash_bench_timm":
        from ._bash_bench_set_timm import TimmRunner

        runner = TimmRunner(device=args.device, verbose=int(args.verbose))
    elif script_name == "bash_bench_explicit":
        from ._bash_bench_set_explicit import ExplicitRunner

        runner = ExplicitRunner(device=args.device, verbose=int(args.verbose))
    elif script_name == "bash_bench_untrained":
        from ._bash_bench_untrained import UntrainedRunner

        runner = UntrainedRunner(device=args.device, verbose=int(args.verbose))
    else:
        raise AssertionError(f"Unexpected bash_bench name {script_name!r}.")
    names = runner.get_model_name_list()

    def _name(name, names):
        if isinstance(name, int):
            name = names[name]
        return name

    missing = {
        "suite": runner.SUITE,
        "model_task": "",
        "time_latency": lambda missing, config: {
            "model_name": _name(config["model"], names),
            "ERR_crash": "INFERENCE failed",
        },
        "time_export_success": lambda missing, config: {
            "model_name": _name(config["model"], names),
            "ERR_crash": "EXPORT failed",
        },
        "time_latency_eager": lambda missing, config: {
            "model_name": _name(config["model"], names),
            "ERR_crash": "EAGER is missing",
        },
    }

    if not args.model and args.model not in ("0", 0):
        # prints the list of models.
        print(f"list of models for device={args.device} (args.model={args.model!r})")
        print("--")
        print("\n".join([f"{i: 3d}/{len(names)} - {n}" for i, n in enumerate(names)]))
        print("--")
    elif args.model == "Refresh":
        names = "\n".join(sorted(runner.refresh_model_names()))
        print("Refresh the list with:")
        print(names)
    else:
        if args.model == "all":
            args.model = ",".join(names)
        elif args.model == "All":
            args.model = ",".join(n for n in names if not n.startswith("101"))
        elif args.model == "Head":
            args.model = ",".join([n for n in names if not n.startswith("101")][:10])
        elif args.model == "Tail":
            args.model = ",".join(n for n in names[-10:] if not n.startswith("101"))
        elif isinstance(args.model, str) and "-" in args.model and "/" not in args.model:
            ms = []
            for v in args.model.split(","):
                if "-" not in v:
                    ms.append(v)
                    continue
                spl = v.split("-")
                assert len(spl) == 2, f"Unexpected value {v!r} in {args.model!r}"
                a, b = spl
                ia = int(a)
                ib = int(b)
                for i in range(ia, ib + 1):
                    ms.append(str(i))
            args.model = ",".join(ms)

        assert (
            "," not in args.tag
        ), f"Parameter tag={args.tag!r} does not support multiple values."
        if (
            multi_run(args)
            or args.process in BOOLEAN_VALUES
            or (args.split_process in BOOLEAN_VALUES and args.part in (None, ""))
        ):
            assert args.part == "", f"part={args.part} must be empty"
            args_output_data = args.output_data
            if args.output_data:
                name, ext = os.path.splitext(args.output_data)
                temp_output_data = f"{name}.temp{ext}"
            else:
                temp_output_data = None
            split_process = args.split_process in BOOLEAN_VALUES
            if split_process and args.part == "":
                args.part = "0,1"
                if args.verbose:
                    print("Running export and inference in two different processes")
            configs = make_configs(
                args,
                drop={"process"},
                replace={"output_data": ""},
                last={"part"} if split_process else None,
                filter_function=lambda kwargs: ModelRunner.allowed_configuration(
                    exporter=kwargs["exporter"],
                    optimization=kwargs.get("opt_patterns", None),
                ),
            )
            assert configs, f"No configuration configs={configs} for args={args}"
            data = run_benchmark(
                f"experimental_experiment.torch_bench.{script_name}",
                configs,
                args.verbose,
                stop_if_exception=False,
                temp_output_data=temp_output_data,
                dump_std=args.dump_folder,
                start=args.start,
                summary=merge_benchmark_reports,
                timeout=int(args.timeout),
                missing=missing,
            )
            if args.verbose > 2:
                pprint.pprint(data if args.verbose > 3 else data[:2])
            if args_output_data:
                df = make_dataframe_from_benchmark_data(data, detailed=False)
                filename = args_output_data
                if os.path.exists(filename):
                    # Let's avoid losing data.
                    name, ext = os.path.splitext(filename)
                    i = 2
                    while os.path.exists(filename):
                        filename = f"{name}.m{i}{ext}"
                        i += 1
                print(f"Prints out the merged results into file {filename!r}")
                fold, _ = os.path.split(filename)
                if fold and not os.path.exists(fold):
                    os.makedirs(fold)
                df.to_csv(filename, index=False, errors="ignore")
                df.to_excel(filename + ".xlsx", index=False)
                if args.verbose:
                    print(df)

                # also write a summary
                fn = f"{filename}.summary.xlsx"
                print(f"Prints out the merged summary into file {fn!r}")
                merge_benchmark_reports(df, excel_output=fn)
        else:
            try:
                indice = int(args.model)
                name = names[indice]
            except (TypeError, ValueError):
                name = args.model

            if args.verbose:
                print(f"Running model {name!r}")

            do_profile = args.profile in BOOLEAN_VALUES

            runner = runner.__class__(
                include_model_names={name},
                verbose=args.verbose,
                device=args.device,
                target_opset=args.target_opset,
                repeat=args.repeat,
                warmup=args.warmup,
                dtype=args.dtype,
                nvtx=args.nvtx in BOOLEAN_VALUES,
                dump_ort=args.dump_ort in BOOLEAN_VALUES,
                attn_impl=args.attn_impl,
            )

            if do_profile:
                import cProfile

                pr = cProfile.Profile()
                pr.enable()

            split_process = args.split_process in BOOLEAN_VALUES
            begin = time.perf_counter()
            assert args.rtopt in (
                "0",
                "1",
                0,
                1,
                9,
                "9",
            ), f"rtopt={args.rtopt!r} must be in (0, 1, 9)"
            data = list(
                runner.enumerate_test_models(
                    process=args.process in BOOLEAN_VALUES,
                    exporter=args.exporter,
                    quiet=args.quiet in BOOLEAN_VALUES,
                    dynamic=args.dynamic in BOOLEAN_VALUES,
                    folder=args.dump_folder,
                    optimization=args.opt_patterns if args.opt_patterns != "none" else "",
                    memory_peak=args.memory_peak in BOOLEAN_VALUES,
                    part=int(args.part) if split_process else None,
                    pickled_name="temp_pickled_file.pkl" if split_process else None,
                    rtopt=int(args.rtopt),
                    shape_again=args.shape2 in BOOLEAN_VALUES,
                    apply_patches=args.patch in BOOLEAN_VALUES,
                )
            )
            duration = time.perf_counter() - begin

            if do_profile:
                import io
                import pstats
                from onnx_array_api.profiling import profile2graph

                pr.disable()
                s = io.StringIO()
                sortby = pstats.SortKey.CUMULATIVE
                ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
                root, _ = profile2graph(ps, clean_text=_clean_text)
                text = root.to_text(fct_width=100)
                filename = (
                    f"{args.output_data}.profile.txt" if args.output_data else "profile.txt"
                )
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(text)

            if args.tag:
                for d in data:
                    d["version_tag"] = args.tag
            if len(data) == 1:
                for k, v in sorted(data[0].items()):
                    print(f":{k},{v};")
            else:
                print(f":model_name,{name};")
                print(f":device,{args.device};")
                print(f":ERROR,unexpected number of data {len(data)};")

            if args.output_data:
                df = make_dataframe_from_benchmark_data(data, detailed=False)

                df["DATE"] = f"{datetime.now():%Y-%m-%d}"
                df["ITER"] = 0
                df["TIME_ITER"] = duration
                df["PART"] = int(args.part) if args.part in (0, 1, "0", "1") else np.nan

                filename = args.output_data
                if os.path.exists(filename):
                    # Let's avoid losing data.
                    name, ext = os.path.splitext(filename)
                    i = 2
                    while os.path.exists(filename):
                        filename = f"{name}.i{i}{ext}"
                        i += 1

                print(f"Prints out the results into file {filename!r}")
                fold, _ = os.path.split(filename)
                if fold and not os.path.exists(fold):
                    os.makedirs(fold)
                df.to_csv(filename, index=False, errors="ignore")
                df.to_excel(filename + ".xlsx", index=False)
                if args.verbose:
                    print(df)

                # also write a summary
                if args.part in (None, "", 1, "1"):
                    fn = f"{filename}.summary-one.xlsx"
                    print(f"Prints out the summary into file {fn!r}")
                    merge_benchmark_reports(df, excel_output=fn)
