"""
.. _l-plot-llama-bench:

Measure LLAMA speed
===================

The script is calling many times the script ``onnxrt_backend_dev.llama.dort_bench.py``.

::

    python docs/examples/plot_llama_bench.py --help
    
For exemple, to check mixed precision on multiple backend:

::

    python docs/examples/plot_llama_bench.py --device=cuda --num_hidden_layers=1 --mixed=1


Run the following command to run one experiment and get the available options:

::

    python -m onnxrt_backend_dev.llama.dort_bench --help

"""

import onnxruntime  # noqa: F401
import numpy as np
import pandas
import matplotlib.pyplot as plt
import itertools
import torch
from onnxrt_backend_dev.ext_test_case import unit_test_going
from onnxrt_backend_dev.bench_run import run_benchmark, get_machine, BenchmarkError
from onnxrt_backend_dev.args import get_parsed_args

script_name = "onnxrt_backend_dev.llama.dort_bench"
machine = {} if unit_test_going() else get_machine()


parsed_args = get_parsed_args(
    "plot_llama_bench",
    description=__doc__,
    warmup=5,
    repeat=5,
    backend=("eager,inductor,ort", "backend to test"),
    device=("cpu,cuda" if torch.cuda.is_available() else "cpu", "device to test"),
    num_hidden_layers=("1,2", "hidden layers to test"),
    mixed=("0,1", "boolean value to test (mixed precision or not)"),
    script_name=("onnxrt_backend_dev.llama.dort_bench", "script to run"),
    dump=(0, "dump the models with env ONNXRT_DUMP_PATH"),
    check=(0, "just check the script is working, ignores all other parameters"),
    expose="backend,device,num_hidden_layers,mixed,scipt_name,repeat,warmup,dump,check",
)
repeat = parsed_args.repeat
warmup = parsed_args.warmup

if machine.get("capability", (0, 0)) >= (7, 0) and parsed_args.check not in (1, "1"):
    verbose = 1
    configs = []
    for backend, device, num_hidden_layers, mixed in itertools.product(
        parsed_args.backend.split(","),
        parsed_args.device.split(","),
        list(map(int, parsed_args.num_hidden_layers.split(","))),
        list(map(int, parsed_args.mixed.split(","))),
    ):
        if mixed == 1 and device == "cpu":
            continue
        configs.append(
            dict(
                backend=backend,
                device=device,
                num_hidden_layers=num_hidden_layers,
                repeat=repeat,
                mixed=mixed,
                warmup=warmup,
            )
        )
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
            warmup=1,
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

if data_collected:
    df = pandas.DataFrame(data)
    df = df.drop(["ERROR", "OUTPUT"], axis=1)
    filename = "plot_llama_bench.csv"
    df.to_csv(filename, index=False)
    df = pandas.read_csv(filename)  # to cast type
    print(df)

################################
# More simple

for c in ["time", "warmup_time"]:
    if c not in df.columns:
        df[c] = np.nan
columns = ["backend", "num_hidden_layers", "mixed", "time", "device", "warmup_time"]
if data_collected:
    try:
        dfs = df[columns]
    except KeyError as e:
        raise RuntimeError(f"Missing columns in {df.columns}\n{df.head().T}") from e
    print(dfs)

###############################
# Plot.

if data_collected:
    fig, ax = plt.subplots(2, 3, figsize=(10, 9))

    # warmup time

    piv = dfs[(dfs.device == "cpu") & (dfs.mixed == 0)].pivot(
        index="num_hidden_layers", columns="backend", values="warmup_time"
    )
    if len(piv) > 0:
        piv.plot(title="llama with dort on cpu\nwarmup time", ax=ax[0, 0])

    piv = dfs[(dfs.device == "cuda") & (dfs.mixed == 0)].pivot(
        index="num_hidden_layers", columns="backend", values="warmup_time"
    )
    if len(piv) > 0:
        piv.plot(title="llama with dort on cuda\nwarmup time", ax=ax[0, 1])

    piv = dfs[(dfs.device == "cuda") & (dfs.mixed == 0)].pivot(
        index="num_hidden_layers", columns="backend", values="warmup_time"
    )
    if len(piv) > 0:
        piv.plot(title="llama with dort on cuda (mixed)\nwarmup time", ax=ax[0, 2])

    # time

    piv = dfs[(dfs.device == "cpu") & (dfs.mixed == 0)].pivot(
        index="num_hidden_layers", columns="backend", values="time"
    )
    if len(piv) > 0:
        piv.plot(
            title=f"llama with dort on cpu\ntraining time for {repeat} iterations",
            ax=ax[1, 0],
        )

    piv = dfs[(dfs.device == "cuda") & (dfs.mixed == 0)].pivot(
        index="num_hidden_layers", columns="backend", values="time"
    )
    if len(piv) > 0:
        piv.plot(
            title=f"llama with dort on cuda\ntraining time for {repeat} iterations",
            ax=ax[1, 1],
        )

    piv = dfs[(dfs.device == "cuda") & (dfs.mixed == 1)].pivot(
        index="num_hidden_layers", columns="backend", values="time"
    )
    if len(piv) > 0:
        piv.plot(
            title=f"llama with dort on cuda (mixed)\ntraining time for {repeat} iterations",
            ax=ax[1, 2],
        )

    fig.tight_layout()
    fig.savefig("plot_llama_bench.png")
