"""
.. _l-plot-llama-bench:

102: Measure LLAMA speed
========================

The script is calling many times the script ``experimental_experiment.torch_bench.dort_bench.py``.

::

    python _doc/examples/plot_llama_bench_102.py --help
    
For exemple, to check mixed precision on multiple backend:

::

    python _doc/examples/plot_llama_bench_102.py --device=cuda --num_hidden_layers=2 --mixed=1

::

    python _doc/examples/plot_llama_bench_102.py --device=cuda --num_hidden_layers=2 --mixed=1 --backend=eager,dynger,ortmodule,inductor,ort+,custom --config=large

With 32Gb GPU memory, the script runs with 6 layers.

::

    python _doc/examples/plot_llama_bench_102.py --device=cuda --num_hidden_layers=6 --mixed=1 --backend=eager,dynger,ortmodule,inductor,ort+,custom --config=large

    python _doc/examples/plot_llama_bench_102.py --device=cuda --num_hidden_layers=2 --mixed=1 --backend=eager,ort+,custom --config=large

Run the following command to run one experiment and get the available options:

::

    python -m experimental_experiment.torch_bench.dort_bench --help

"""

from experimental_experiment.args import get_parsed_args, check_cuda_availability

parsed_args = get_parsed_args(
    "plot_llama_bench",
    description=__doc__,
    warmup=5,
    repeat=10,
    model=("llama", "model to benchmark"),
    backend=(
        "eager,dynger,inductor,ort,ort+,custom,ortmodule",
        "backend to test, among eager,dynger,inductor,ort,ort+,custom,plug,ortmodule,backort",
    ),
    device=("cuda" if check_cuda_availability() else "cpu", "device to test"),
    num_hidden_layers=("1", "hidden layers to test"),
    mixed=("0", "boolean value to test (mixed precision or not)"),
    dynamic=("0", "boolean value to test dynamic shapes or not"),
    script_name=("experimental_experiment.torch_bench.dort_bench", "script to run"),
    dump=(0, "dump the models with env ONNXRT_DUMP_PATH"),
    check=(0, "just check the script is working, ignores all other parameters"),
    config=("medium", "configuration to use, default or medium"),
    patterns=(
        "none,default,default+onnxruntime," "default+onnxruntime+experimental",
        "optimization patterns to use",
    ),
    implementation=("eager", "eager or sdpa or both values comma separated value"),
    with_mask=(1, "with or without a second input (mask"),
    disable_pattern=("none", "pattern or patterns to disable"),
    ort_optimize=(
        "0,1",
        "enable or disable onnxruntime optimization, " "by default, tries both",
    ),
    order=("none", "optimization order see class OrderAlgorithm, none by default"),
    verbose=(1, "verbosity"),
    expose="backend,device,num_hidden_layers,mixed,scipt_name,repeat,"
    "warmup,dump,check,config,patterns,dynamic,disable_pattern,model"
    "implementation,with_mask,ort_optimize,verbose,order",
)

import onnxruntime  # noqa: F401
import numpy as np
import pandas
import matplotlib.pyplot as plt
import itertools
import torch
from experimental_experiment.ext_test_case import unit_test_going
from experimental_experiment.bench_run import run_benchmark, get_machine, BenchmarkError

script_name = "experimental_experiment.torch_bench.dort_bench"
machine = {} if unit_test_going() else get_machine()


repeat = parsed_args.repeat
warmup = parsed_args.warmup


def make_config(
    model,
    backend,
    device,
    num_hidden_layers,
    repeat,
    mixed,
    dynamic,
    config,
    warmup,
    pattern,
    disable_pattern,
    implementation,
    with_mask,
    ort_optimize,
    order,
    verbose,
    existing=None,
):
    if backend not in ("custom", "ort+"):
        ort_optimize = None
        pattern = None
        disable_pattern = None
    cf = dict(
        model=model,
        backend=backend,
        device=device,
        num_hidden_layers=num_hidden_layers,
        repeat=repeat,
        mixed=mixed,
        dynamic=dynamic,
        config=config,
        warmup=warmup,
        implementation=implementation,
        with_mask=with_mask,
        ort_optimize=ort_optimize,
        order=order,
        verbose=verbose,
    )
    cf = {k: v for k, v in cf.items() if v is not None}

    if existing and backend not in ("custom", "ort+"):
        for ex in existing:
            if not ex:
                continue
            equal = True
            for k in cf:
                if cf[k] != ex[k]:
                    equal = False
                    break
            if equal:
                return None

    if pattern is None:
        opt = {}
    elif pattern == "none":
        opt = dict(enable_pattern="default", disable_pattern="default")
    elif pattern in "default" or "+" in pattern:
        opt = dict(enable_pattern=pattern)
    else:
        raise AssertionError(f"unexpected value for pattern={pattern!r}")
    cf.update(opt)
    if disable_pattern not in ("none", None):
        if "disable_pattern" in cf:
            cf["disable_pattern"] += f",{disable_pattern}"
        else:
            cf["disable_pattern"] = disable_pattern
    if "enable_pattern" in cf and "+experimental" in cf["enable_pattern"]:
        try:
            import onnx_extended  # noqa: F401
        except ImportError:
            return None
    elif not ort_optimize and backend in ("custom", "ort+"):
        return None
    assert (
        cf["backend"] != "eager" or cf.get("ort_optimize", None) is None
    ), f"Wrong configuration {cf}"
    return cf


if parsed_args.check not in (1, "1"):
    verbose = parsed_args.verbose
    configs = []
    for (
        backend,
        device,
        num_hidden_layers,
        mixed,
        dynamic,
        pattern,
        impl,
        ort_optimize,
    ) in itertools.product(
        parsed_args.backend.split(","),
        parsed_args.device.split(","),
        list(map(int, parsed_args.num_hidden_layers.split(","))),
        list(map(int, parsed_args.mixed.split(","))),
        list(map(int, parsed_args.dynamic.split(","))),
        parsed_args.patterns.split(","),
        parsed_args.implementation.split(","),
        list(map(int, parsed_args.ort_optimize.split(","))),
    ):
        if mixed == 1 and device == "cpu":
            continue
        if machine.get("capability", (0, 0)) < (7, 0) and backend == "inductor":
            continue
        configs.append(
            make_config(
                model=parsed_args.model,
                backend=backend,
                device=device,
                num_hidden_layers=num_hidden_layers,
                repeat=repeat,
                mixed=mixed,
                dynamic=dynamic,
                config=parsed_args.config,
                warmup=warmup,
                pattern=pattern,
                disable_pattern=parsed_args.disable_pattern,
                existing=configs,
                implementation=impl,
                with_mask=parsed_args.with_mask,
                ort_optimize=ort_optimize,
                order=parsed_args.order,
                verbose=verbose,
            )
        )
else:
    verbose = 5
    device = "cuda" if torch.cuda.is_available() else "cpu"
    configs = [
        dict(
            model=parsed_args.model,
            backend="custom",
            device=device,
            num_hidden_layers=1,
            repeat=1,
            mixed=0,
            dynamic=0,
            warmup=1,
            config="small",
        ),
    ]

################################
# All configurations to consider.

configs = [cf for cf in configs if cf]
if verbose:
    for i, cf in enumerate(configs):
        print(f"config {i+1}: {cf}")

################################
# Running configuration.


try:
    data = run_benchmark(
        parsed_args.script_name,
        configs,
        verbose=verbose,
        stop_if_exception=False,
        dump=parsed_args.dump in ("1", 1),
    )
    data_collected = True
except BenchmarkError as e:
    if verbose:
        print(e)
    data_collected = False

#########################
# Let's process the data.

prefix = (
    f"plot_{parsed_args.model}-{parsed_args.with_mask}-"
    f"m{parsed_args.mixed}d{parsed_args.dynamic}-"
    f"{parsed_args.implementation}"
)

if data_collected:

    def clean_pattern(s):
        s = s.replace("+default-default", "")
        return s

    def make_legend(row):
        row = row.to_dict()
        val = [
            row["device"],
            f"h{row['num_hidden_layers']}",
            row["implementation"],
            row["backend"],
        ]
        if row["mixed"]:
            val.append("mix")
        if row["dynamic"]:
            val.append("dyn")
        if "patterns" in row and row["patterns"] and "nan" not in str(row["patterns"]):
            val.append(f"({clean_pattern(row['patterns'])})")
        s = "-".join(map(str, val))
        assert "nan" not in s, f"Legend {s!r} is wrong, row={row}"
        return s

    df = pandas.DataFrame(data)
    df = df.drop(["OUTPUT", "ERROR"], axis=1)
    df["legend"] = df.apply(make_legend, axis=1)
    df["time"] = df["time"].astype(float)
    df_eager = df[(df["implementation"] == "eager") & (df["backend"] == "eager")][
        "time"
    ].dropna()
    if df_eager.shape[0] > 0:
        min_eager = df_eager.min()
        df["increase"] = df["time"] / min_eager - 1
        # df["ERROR"] = df["ERROR"].apply(lambda s: s.replace("\n", " "))
    filename = f"plot_{prefix}_bench_with_cmd.csv"
    df.to_csv(filename, index=False)
    filename = f"plot_{prefix}_bench_with_cmd.xlsx"
    df.to_excel(filename, index=False)

    df = df.drop(["CMD"], axis=1)
    filename = f"plot_{prefix}_bench.csv"
    df.to_csv(filename, index=False)
    df = pandas.read_csv(filename)  # to cast type
    print(df)

    # summary
    cs = [
        c
        for c in ["backend", "patterns", "warmup_time", "time", "increase"]
        if c in df.columns
    ]
    dfs = df[cs]
    filename = f"plot_{prefix}_summary.xlsx"
    dfs.to_excel(filename, index=False)
    filename = f"plot_{prefix}_summary.csv"
    dfs.to_csv(filename, index=False)
    print(dfs)

########################
# First lines.

print(df.head(2).T)

################################
# More simple

for c in ["time", "warmup_time"]:
    if c not in df.columns:
        df[c] = np.nan

########################################
# Simplified data

print(df.sort_values("legend"))

###############################
# Plot warmup time.

torch_version = list(set(df["torch"].dropna()))
transformers_version = list(set(df["transformers"].dropna()))
ver = f"{torch_version[0]} - {transformers_version[0]}"
model = parsed_args.model
modeldf = list(set(df[model].dropna()))[0]
title_prefix = (
    f"lower better\n"
    f"{parsed_args.model} - {ver} - mask{parsed_args.with_mask}"
    f"\n<device>-h<hidden-layers>-<implementation>-<backend>-(optimization)"
)


if data_collected:
    fig, ax = plt.subplots(1, 1, figsize=(12, df.shape[0] // 3 + 1))

    df = df.sort_values("time").set_index("legend")
    df[["warmup_time"]].plot.barh(ax=ax, title=f"warmup time\n{title_prefix}")
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(f"plot_{prefix}_bench_warmup_time.png")

###############################
# Plot time.

if data_collected:
    fig, ax = plt.subplots(1, 1, figsize=(12, df.shape[0] // 3 + 1))

    df[["time"]].plot.barh(ax=ax, title=f"computation time\n{title_prefix}")
    mi, ma = df["time"].min(), df["time"].max()
    mi = mi - (ma - mi) / 10
    ax.set_xlim(left=mi)
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(f"plot_{prefix}_bench_time.png")

###############################
# Plot increase.

if data_collected:
    fig, ax = plt.subplots(1, 1, figsize=(12, df.shape[0] // 3 + 1))

    df[["increase"]].plot.barh(ax=ax, title=f"comparison to eager %\n{title_prefix}")
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(f"plot_{prefix}_bench_relative.png")
