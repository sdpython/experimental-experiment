"""
.. _l-plot-llama-bench:

Measure LLAMA speed
===================

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
    backend=("eager,inductor,ort,custom", "backend to test"),
    device=("cpu,cuda" if check_cuda_availability() else "cpu", "device to test"),
    num_hidden_layers=("2", "hidden layers to test"),
    mixed=("0,1", "boolean value to test (mixed precision or not)"),
    dynamic=("0,1", "boolean value to test dynamic shapes or not"),
    script_name=("experimental_experiment.torch_bench.dort_bench", "script to run"),
    dump=(0, "dump the models with env ONNXRT_DUMP_PATH"),
    check=(0, "just check the script is working, ignores all other parameters"),
    config=("medium", "configuration to use, default or medium"),
    patterns=("none,default,onnxruntime", "optimization patterns to use"),
    expose="backend,device,num_hidden_layers,mixed,scipt_name,repeat,"
    "warmup,dump,check,config,patterns,dynamic",
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
    backend, device, num_hidden_layers, repeat, mixed, dynamic, config, warmup, pattern
):
    cf = dict(
        backend=backend,
        device=device,
        num_hidden_layers=num_hidden_layers,
        repeat=repeat,
        mixed=mixed,
        dynamic=dynamic,
        config=config,
        warmup=warmup,
    )
    if pattern == "none":
        opt = dict(disable_pattern="default")
    elif pattern == "default":
        opt = dict(enable_pattern="default")
    elif pattern == "onnxruntime":
        opt = dict(enable_pattern="onnxruntime")
    else:
        raise AssertionError(f"unexpected value for pattern={pattern!r}")
    cf.update(opt)
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
        if machine.get("capability", (0, 0)) >= (7, 0) and backend == "inductor":
            continue
        configs.append(
            make_config(
                backend=backend,
                device=device,
                num_hidden_layers=num_hidden_layers,
                repeat=repeat,
                mixed=mixed,
                dynamic=dynamic,
                config=parsed_args.config,
                warmup=warmup,
                pattern=pattern,
            )
        )
        print(f"config {len(configs)}: {configs[-1]}")
else:
    verbose = 5
    device = "cuda" if torch.cuda.is_available() else "cpu"
    configs = [
        dict(
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

    def make_legend(row):
        row = row.to_dict()
        val = [row["device"], row["backend"], f"h{row['num_hidden_layers']}"]
        if row["mixed"]:
            val.append("mixed")
        if row["dynamic"]:
            val.append("dyn")
        if row["patterns"] and "nan" not in str(row["patterns"]):
            val.append(f"({row['patterns']})")
        s = "-".join(map(str, val))
        assert "nan" not in s, f"Legend {s!r} is wrong, row={row}"
        return s

    df = pandas.DataFrame(data)
    df = df.drop(["OUTPUT"], axis=1)
    df["ERROR"] = df["ERROR"].apply(lambda s: s.replace("\n", " "))
    df["legend"] = df.apply(make_legend, axis=1)
    filename = "plot_llama_bench_with_errors.csv"
    df.to_csv(filename, index=False)

    df = df.drop(["ERROR", "CMD"], axis=1)
    filename = "plot_llama_bench.csv"
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
# Plot.

if data_collected:
    min_eager = df[df.legend.str.contains("eager")]["time"].min()
    torch_version = list(set(df["torch"]))
    transformers_version = list(set(df["transformers"]))
    ver = f"{torch_version[0]} - {transformers_version[0]}"
    llama = list(set(df["llama"]))[0]

    fig, ax = plt.subplots(3, 1, figsize=(3 * df.shape[0] * 2, 5))

    # warmup time
    df = df.sort_values("time").set_index("legend")
    df[["warmup_time"]].plot.barh(ax=ax[0], title=f"warmup time\n{ver}")

    # time
    df[["time"]].plot.barh(ax=ax[1], title=f"iteration time\n{ver}")
    mi, ma = df["time"].min(), df["time"].max()
    mi = mi - (ma - mi) / 10
    ax[1].set_xlim(left=mi)

    # comparison
    df["time"]
    df["increase"] = (df["time"] / min_eager - 1) * 100
    df[["increase"]].plot.barh(ax=ax[2], title="comparison to eager %")

    fig.suptitle(f"lower better\n{llama}")
    fig.tight_layout()
    fig.savefig("plot_llama_bench.png")
