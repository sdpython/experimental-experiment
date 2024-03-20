"""
.. _l-plot-llama-bench:

102: Measure LLAMA speed
========================

The script is calling many times the script ``experimental_experiment.torch_bench.dort_bench.py``.

::

    python _doc/examples/plot_llama_bench.py --help
    
For exemple, to check mixed precision on multiple backend:

::

    python _doc/examples/plot_llama_bench.py --device=cuda --num_hidden_layers=1 --mixed=1


Run the following command to run one experiment and get the available options:

::

    python -m experimental_experiment.torch_bench.dort_bench --help

"""

from experimental_experiment.args import get_parsed_args, check_cuda_availability

parsed_args = get_parsed_args(
    "plot_llama_bench",
    description=__doc__,
    warmup=3,
    repeat=5,
    model=("llama", "model to benchmark"),
    backend=("eager,dynger,inductor,ort,custom,plug", "backend to test"),
    device=("cuda" if check_cuda_availability() else "cpu", "device to test"),
    num_hidden_layers=("2", "hidden layers to test"),
    mixed=("0", "boolean value to test (mixed precision or not)"),
    dynamic=("0", "boolean value to test dynamic shapes or not"),
    script_name=("experimental_experiment.torch_bench.dort_bench", "script to run"),
    dump=(0, "dump the models with env ONNXRT_DUMP_PATH"),
    check=(0, "just check the script is working, ignores all other parameters"),
    config=("medium", "configuration to use, default or medium"),
    patterns=("none,default,default+onnxruntime", "optimization patterns to use"),
    disable_pattern=("none", "pattern or patterns to disable"),
    expose="backend,device,num_hidden_layers,mixed,scipt_name,repeat,"
    "warmup,dump,check,config,patterns,dynamic,disable_pattern,model",
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
    existing=None,
):
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
    )

    if existing and backend != "custom":
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

    if pattern == "none":
        opt = dict(disable_pattern="default")
    elif pattern in ("default", "default+onnxruntime"):
        opt = dict(enable_pattern=pattern)
    else:
        raise AssertionError(f"unexpected value for pattern={pattern!r}")
    cf.update(opt)
    if disable_pattern != "none":
        if "disable_pattern" in cf:
            cf["disable_pattern"] += f",{disable_pattern}"
        else:
            cf["disable_pattern"] = disable_pattern
    return cf


if parsed_args.check not in (1, "1"):
    verbose = 1
    configs = []
    for (
        backend,
        device,
        num_hidden_layers,
        mixed,
        dynamic,
        pattern,
    ) in itertools.product(
        parsed_args.backend.split(","),
        parsed_args.device.split(","),
        list(map(int, parsed_args.num_hidden_layers.split(","))),
        list(map(int, parsed_args.mixed.split(","))),
        list(map(int, parsed_args.dynamic.split(","))),
        parsed_args.patterns.split(","),
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
            )
        )
else:
    verbose = 5
    device = "cuda" if torch.cuda.is_available() else "cpu"
    configs = [
        dict(
            model=parsed_args.model,
            backend="ort",
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
    print(e)
    data_collected = False

#########################
# Let's process the data.

if data_collected:

    def clean_pattern(s):
        if "+default" in s:
            s = s.replace("ConstantOfShapeScatterND", "")
        s = s.replace("+default-default", "")
        return s

    def make_legend(row):
        row = row.to_dict()
        val = [row["device"], row["backend"], f"h{row['num_hidden_layers']}"]
        if row["mixed"]:
            val.append("mixed")
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
    min_eager = df[df.legend.str.contains("eager")]["time"].dropna().min()
    df["increase"] = df["time"] / min_eager - 1
    # df["ERROR"] = df["ERROR"].apply(lambda s: s.replace("\n", " "))
    filename = f"plot_{parsed_args.model}_bench_with_cmd.csv"
    df.to_csv(filename, index=False)

    df = df.drop(["CMD"], axis=1)
    filename = f"plot_{parsed_args.model}_bench.csv"
    df.to_csv(filename, index=False)
    df = pandas.read_csv(filename)  # to cast type
    print(df)

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

if data_collected:
    fig, ax = plt.subplots(1, 1, figsize=(12, df.shape[0] // 3 + 1))

    df = df.sort_values("time").set_index("legend")
    df[["warmup_time"]].plot.barh(
        ax=ax, title=f"lower better\n{parsed_args.model}\nwarmup time\n{ver}"
    )
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(f"plot_{parsed_args.model}_bench_warmup_time.png")

###############################
# Plot time.

if data_collected:
    fig, ax = plt.subplots(1, 1, figsize=(12, df.shape[0] // 3 + 1))

    df[["time"]].plot.barh(
        ax=ax, title=f"lower better\n{parsed_args.model}\niteration time\n{ver}"
    )
    mi, ma = df["time"].min(), df["time"].max()
    mi = mi - (ma - mi) / 10
    ax.set_xlim(left=mi)
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(f"plot_{parsed_args.model}_bench_time.png")

###############################
# Plot increase.

if data_collected:
    fig, ax = plt.subplots(1, 1, figsize=(12, df.shape[0] // 3 + 1))

    df[["increase"]].plot.barh(
        ax=ax, title=f"lower better\n{parsed_args.model}\ncomparison to eager %"
    )
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(f"plot_{parsed_args.model}_bench_relative.png")
