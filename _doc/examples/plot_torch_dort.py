"""
Evaluate DORT
=============

It compares DORT to eager mode and the default backend.

To run the script:

::

    python _doc/examples/plot_torch_dort_dort --help

Some helpers
++++++++++++
"""
import torch._dynamo
import contextlib
import itertools
import os
import gc
import platform

# import pickle
import pprint
import multiprocessing
import time
import cProfile
import pstats
import io
import warnings
import logging
from pstats import SortKey

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import onnxruntime

        has_cuda = "CUDAExecutionProvider" in onnxruntime.get_available_providers()
except ImportError:
    print("onnxruntime not available.")
    import sys

    sys.exit(0)

import numpy as np
import matplotlib.pyplot as plt
import pandas
import onnx
from onnx_array_api.profiling import profile2graph
import torch
from torch import nn
import torch.nn.functional as F
import experimental_experiment
from experimental_experiment.plotting.memory import memory_peak_plot
from experimental_experiment.ext_test_case import get_parsed_args, measure_time
from experimental_experiment.memory_peak import start_spying_on
from tqdm import tqdm

logging.disable(logging.ERROR)


def system_info():
    obs = {}
    obs["processor"] = platform.processor()
    obs["cores"] = multiprocessing.cpu_count()
    try:
        obs["cuda"] = 1 if torch.cuda.is_available() else 0
        obs["cuda_count"] = torch.cuda.device_count()
        obs["cuda_name"] = torch.cuda.get_device_name()
        obs["cuda_capa"] = torch.cuda.get_device_capability()
    except (RuntimeError, AssertionError):
        # no cuda
        pass
    return obs


pprint.pprint(system_info())

#####################################
# Scripts arguments


script_args = get_parsed_args(
    "plot_torch_dort",
    description=__doc__,
    scenarios={
        "small": "small model to test",
        "middle": "55Mb model",
        "large": "1Gb model",
    },
    warmup=5,
    repeat=5,
    repeat1=(1, "repeat for the first iteration"),
    maxtime=(
        2,
        "maximum time to run a model to measure the computation time, "
        "it is 0.1 when scenario is small",
    ),
    expose="scenarios,repeat,repeat1,warmup",
)

if script_args.scenario in (None, "small"):
    script_args.maxtime = 0.1
print(f"scenario={script_args.scenario or 'small'}")
print(f"warmup={script_args.warmup}")
print(f"repeat={script_args.repeat}")
print(f"repeat1={script_args.repeat1}")
print(f"maxtime={script_args.maxtime}")

############################
# The model
# +++++++++
#
# A simple model to convert.


class MyModelClass(nn.Module):
    def __init__(self, scenario=script_args.scenario):
        super(MyModelClass, self).__init__()
        if scenario == "middle":
            self.large = False
            self.conv1 = nn.Conv2d(1, 32, 5)
            # self.conv2 = nn.Conv2d(128, 16, 5)
            self.fc1 = nn.Linear(30752, 1024)
            self.fcs = []
            self.fc2 = nn.Linear(1024, 128)
            self.fc3 = nn.Linear(128, 10)
        elif scenario in (None, "small"):
            self.large = False
            self.conv1 = nn.Conv2d(1, 16, 5)
            # self.conv2 = nn.Conv2d(16, 16, 5)
            self.fc1 = nn.Linear(144, 512)
            self.fcs = []
            self.fc2 = nn.Linear(512, 128)
            self.fc3 = nn.Linear(128, 10)
        elif scenario in (None, "large"):
            self.large = True
            self.conv1 = nn.Conv2d(1, 32, 5)
            # self.conv2 = nn.Conv2d(128, 16, 5)
            self.fc1 = nn.Linear(30752, 4096)
            # torch script does not support loops.
            self.fca = nn.Linear(4096, 4096)
            self.fcb = nn.Linear(4096, 4096)
            self.fcc = nn.Linear(4096, 4096)
            self.fcd = nn.Linear(4096, 4096)
            self.fce = nn.Linear(4096, 4096)
            self.fcf = nn.Linear(4096, 4096)
            self.fcg = nn.Linear(4096, 4096)
            self.fch = nn.Linear(4096, 4096)
            self.fci = nn.Linear(4096, 4096)
            # end of the unfolded loop.
            self.fc2 = nn.Linear(4096, 128)
            self.fc3 = nn.Linear(128, 10)
        else:
            raise ValueError(f"Unsupported scenario={scenario!r}.")

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (4, 4))
        # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        if self.large:
            # loop
            x = F.relu(self.fca(x))
            x = F.relu(self.fcb(x))
            x = F.relu(self.fcc(x))
            x = F.relu(self.fcd(x))
            x = F.relu(self.fce(x))
            x = F.relu(self.fcf(x))
            x = F.relu(self.fcg(x))
            x = F.relu(self.fch(x))
            x = F.relu(self.fci(x))
            # end of the loop
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def create_model_and_input(scenario=script_args.scenario):
    if scenario == "middle":
        shape = [1, 1, 128, 128]
    elif scenario in (None, "small"):
        shape = [1, 1, 16, 16]
    elif scenario == "large":
        shape = [1, 1, 128, 128]
    else:
        raise ValueError(f"Unsupported scenario={scenario!r}.")
    input_tensor = torch.rand(*shape).to(torch.float32)
    model = MyModelClass(scenario=scenario)
    assert model(input_tensor) is not None
    return model, input_tensor


def torch_model_size(model):
    size_model = 0
    for param in model.parameters():
        size = param.numel() * torch.finfo(param.data.dtype).bits / 8
        size_model += size
    return size_model


model, input_tensor = create_model_and_input()
model_size = torch_model_size(model)
print(f"model size={model_size / 2 ** 20} Mb")

#######################################
# Backends
# ++++++++


def get_torch_eager(model, *args):
    def my_compiler(gm, example_inputs):
        return gm.forward

    with contextlib.redirect_stdout(io.StringIO()):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            optimized_mod = torch.compile(model, fullgraph=True, backend=my_compiler)
            optimized_mod(*args)
            return optimized_mod


def get_torch_default(model, *args):
    with contextlib.redirect_stdout(io.StringIO()):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            optimized_mod = torch.compile(model, fullgraph=True, mode="reduce-overhead")
            optimized_mod(*args)
            return optimized_mod


def get_torch_dort(model, *args):
    with contextlib.redirect_stdout(io.StringIO()):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            optimized_mod = torch.compile(model, backend="onnxrt", fullgraph=True)
            optimized_mod(*args)
            return optimized_mod


#########################################
# Let's check they are working.

export_functions = [
    get_torch_eager,
    get_torch_default,
    get_torch_dort,
]

exporters = {f.__name__.replace("get_", ""): f for f in export_functions}

supported_exporters = {}
for k, v in exporters.items():
    print(f"run function {k}")
    filename = f"plot_torch_dort_{k}.onnx"
    torch._dynamo.reset()
    model, input_tensor = create_model_and_input()
    try:
        v(model, input_tensor)
    except Exception as e:
        print(f"skipped due to {str(e)[:1000]}")
        continue
    supported_exporters[k] = v
    del model
    gc.collect()
    time.sleep(1)


#################################
# Compile and Memory
# ++++++++++++++++++


def flatten(ps):
    obs = ps["cpu"].to_dict(unit=2**20)
    if "gpus" in ps:
        for i, g in enumerate(ps["gpus"]):
            for k, v in g.to_dict(unit=2**20).items():
                obs[f"gpu{i}_{k}"] = v
    return obs


data = []

for k, v in supported_exporters.items():
    print(f"run compile for memory {k} on cpu")
    filename = f"plot_torch_dort_{k}.onnx"
    if has_cuda:
        torch.cuda.set_device(0)
    torch._dynamo.reset()
    # CPU
    model, input_tensor = create_model_and_input()
    stat = start_spying_on(cuda=1 if has_cuda else 0)
    v(model, input_tensor)
    obs = flatten(stat.stop())
    print("done.")
    obs.update(dict(export=k, p="cpu"))
    data.append(obs)
    del model
    gc.collect()
    time.sleep(1)

    torch._dynamo.reset()
    # CUDA
    model, input_tensor = create_model_and_input()
    model = model.cuda()
    input_tensor = input_tensor.cuda()
    print(f"run compile for memory {k} on cuda")
    stat = start_spying_on(cuda=1 if has_cuda else 0)
    v(model, input_tensor)
    obs = flatten(stat.stop())
    print("done.")
    obs.update(dict(export=k, p="cuda"))
    data.append(obs)
    del model
    gc.collect()
    time.sleep(1)

#############################
# The result.
df1 = pandas.DataFrame(data)
df1.to_csv("plot_torch_dort_1_memory.csv", index=False)
df1.to_excel("plot_torch_dort_1_memory.xlsx", index=False)
print(df1)

for p in ["cpu", "cuda"]:
    ax = memory_peak_plot(
        df1[df1["p"] == p],
        key=("export",),
        bars=[model_size * i / 2**20 for i in range(1, 5)],
        suptitle=f"Memory Consumption of the Compilation on {p}\n"
        f"model size={model_size / 2**20:1.0f} Mb",
    )
    ax[0, 0].get_figure().savefig(f"plot_torch_dort_1_memory_{p}.png")

#################################
# dort first iteration speed
# ++++++++++++++++++++++++++

data = []

for k, v in supported_exporters.items():
    print(f"run dort cpu {k}: {script_args.repeat1}")
    times = []
    for i in range(int(script_args.repeat1)):
        model, input_tensor = create_model_and_input()
        torch._dynamo.reset()
        begin = time.perf_counter()
        v(model, input_tensor)
        duration = time.perf_counter() - begin
        times.append(duration)
        del model
        gc.collect()
        time.sleep(1)

    print(f"done: {times[-1]}")
    data.append(
        dict(
            export=k,
            time=np.mean(times),
            min=min(times),
            max=max(times),
            first=times[0],
            last=times[-1],
            std=np.std(times),
            p="cpu",
        )
    )

    print(f"run dort cuda {k}: {script_args.repeat1}")
    times = []
    for i in range(int(script_args.repeat1)):
        model, input_tensor = create_model_and_input()
        model = model.cuda()
        input_tensor = input_tensor.cuda()
        torch._dynamo.reset()
        begin = time.perf_counter()
        v(model, input_tensor)
        duration = time.perf_counter() - begin
        times.append(duration)
        del model
        gc.collect()
        time.sleep(1)

    print(f"done: {times[-1]}")
    data.append(
        dict(
            export=k,
            time=np.mean(times),
            min=min(times),
            max=max(times),
            first=times[0],
            last=times[-1],
            std=np.std(times),
            p="cuda",
        )
    )

#############################
# The result.
df1 = pandas.DataFrame(data)
df1.to_csv("plot_torch_dort_1_time.csv", index=False)
df1.to_excel("plot_torch_dort_1_time.xlsx", index=False)
print(df1)

fig, ax = plt.subplots(1, 1)
dfi = df1[["export", "p", "time", "std"]].set_index(["export", "p"])
dfi["time"].plot.bar(ax=ax, title="Compilation time", yerr=dfi["std"], rot=30)
fig.tight_layout()
fig.savefig("plot_torch_dort_1_time.png")

####################################
# Compilation Profiling
# +++++++++++++++++++++


def clean_text(text):
    pathes = [
        os.path.abspath(
            os.path.normpath(os.path.join(os.path.dirname(torch.__file__), ".."))
        ),
        os.path.abspath(
            os.path.normpath(os.path.join(os.path.dirname(onnx.__file__), ".."))
        ),
        os.path.abspath(
            os.path.normpath(
                os.path.join(os.path.dirname(experimental_experiment.__file__), "..")
            )
        ),
    ]
    for p in pathes:
        text = text.replace(p, "")
    text = text.replace("experimental_experiment", "experimental_experiment".upper())
    return text


def profile_function(
    name, export_function, with_args=True, verbose=False, suffix="export"
):
    if verbose:
        print(f"profile {name}: {export_function}")
    if with_args:
        model, input_tensor = create_model_and_input()
        pr = cProfile.Profile()
        pr.enable()
        for i in range(int(script_args.repeat1)):
            export_function(model, input_tensor)
        pr.disable()
    else:
        pr = cProfile.Profile()
        pr.enable()
        for i in range(int(script_args.repeat1)):
            export_function()
        pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    # with open(f"plot_torch_dort_profile_{name}_{suffix}.pickle", "wb") as f:
    #     pickle.dump(ps, f)

    raw = s.getvalue()
    text = "\n".join(raw.split("\n")[:200])
    if verbose:
        print(text)
    with open(f"plot_torch_dort_profile_{name}_{suffix}.txt", "w") as f:
        f.write(raw)

    root, nodes = profile2graph(ps, clean_text=clean_text)
    text = root.to_text()
    with open(f"plot_torch_dort_profile_{name}_{suffix}_h.txt", "w") as f:
        f.write(text)
    if verbose:
        print("done.")


model, input_tensor = create_model_and_input()


def function_to_profile(model=model, input_tensor=input_tensor):
    return get_torch_dort(model, input_tensor)


profile_function("dort", function_to_profile, verbose=True, suffix="1")


######################################
# Benchmark exported models with ORT
# ++++++++++++++++++++++++++++++++++


def benchmark(shape):
    data = []
    data_mem_first_run = []
    data_mem_run = []
    confs = list(
        itertools.product(
            export_functions,
            ["CPU", "CUDA"],
        )
    )
    loop = tqdm(confs)
    print(f"number of experiments: {len(loop)}")
    for export_fct, p in loop:
        name = export_fct.__name__.replace("get_torch_", "")
        obs = {}  # system_info()
        obs["name"] = name
        obs["compute"] = p
        obs["export"] = name

        model, input_tensor = create_model_and_input()
        if p == "CUDA":
            model = model.cuda()
            input_tensor = input_tensor.cuda()
        exported_model = export_fct(model, input_tensor)

        def call_model(exported_model=exported_model, input_tensor=input_tensor):
            return exported_model(input_tensor).sum()

        stat = start_spying_on(cuda=1 if has_cuda else 0)
        try:
            call_model()
        except Exception as e:
            loop.set_description(f"ERROR-run: {name} {e}")
            obs.update({"error": e, "step": "load"})
            data.append(obs)
            stat.stop()
            continue
        memobs = flatten(stat.stop())
        memobs.update(obs)
        data_mem_first_run.append(memobs)

        # memory consumption
        stat = start_spying_on(cuda=1 if has_cuda else 0)
        for i in range(0, script_args.warmup):
            call_model()
        memobs = flatten(stat.stop())
        memobs.update(obs)
        data_mem_run.append(memobs)

        obs.update(
            measure_time(
                call_model,
                max_time=script_args.maxtime,
                repeat=script_args.repeat,
                number=1,
            )
        )

        profile_function(name, call_model, with_args=False, suffix=f"run_{p}")

        loop.set_description(f"{obs['average']} {name} {p}")
        data.append(obs)
        del model
        del exported_model
        gc.collect()
        time.sleep(1)

    df = pandas.DataFrame(data)
    df.to_csv("plot_torch_dort_ort_time.csv", index=False)
    df.to_excel("plot_torch_dort_ort_time.xlsx", index=False)
    dfmemr = pandas.DataFrame(data_mem_run)
    dfmemr.to_csv("plot_torch_dort_ort_run_mem.csv", index=False)
    dfmemr.to_excel("plot_torch_dort_ort_run_mem.xlsx", index=False)
    dfmemfr = pandas.DataFrame(data_mem_first_run)
    dfmemfr.to_csv("plot_torch_dort_ort_first_run_mem.csv", index=False)
    dfmemfr.to_excel("plot_torch_dort_ort_first_run_mem.xlsx", index=False)
    return df, dfmemfr, dfmemr


df, dfmemfr, dfmemr = benchmark(list(input_tensor.shape))
print(df)

#####################################
# Other view


def view_time(df, title, suffix="time"):
    piv = pandas.pivot_table(df, index="export", columns=["compute"], values="average")
    print(piv)
    piv.to_csv(f"plot_torch_dort_{suffix}_compute.csv")
    piv.to_excel(f"plot_torch_dort_{suffix}_compute.xlsx")

    piv_gpu = pandas.pivot_table(
        df[df.compute == "CUDA"],
        index="export",
        columns=["compute"],
        values="average",
    )
    piv_cpu = pandas.pivot_table(
        df[df.compute == "CPU"],
        index="export",
        columns=["compute"],
        values="average",
    )

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title)
    piv_cpu.plot.barh(ax=ax[0], title="CPU", logx=True)
    piv_gpu.plot.barh(ax=ax[1], title="CUDA", logx=True)
    fig.tight_layout()
    fig.savefig(f"plot_torch_dort_{suffix}.png")
    return ax


view_time(df, "Compares processing time on backends")


########################################
# Memory First Running Time (ORT)
# +++++++++++++++++++++++++++++++

for compute in ["CPU", "CUDA"]:
    ax = memory_peak_plot(
        dfmemfr[dfmemfr.compute == compute],
        ("export",),
        suptitle=f"Memory Consumption of backend, first running time"
        f"\nrunning on {compute}",
        bars=[model_size * i / 2**20 for i in range(1, 3)],
        figsize=(18, 6),
    )
    ax[0, 0].get_figure().savefig(f"plot_torch_dort_first_run_mem_{compute}.png")

########################################
# Memory Running Time (ORT)
# +++++++++++++++++++++++++

for compute in ["CPU", "CUDA"]:
    ax = memory_peak_plot(
        dfmemr[dfmemr.compute == compute],
        ("export",),
        suptitle=f"Memory Consumption of backens, running time"
        f"\nrunning on {compute}",
        bars=[model_size * i / 2**20 for i in range(1, 3)],
        figsize=(18, 6),
    )
    ax[0, 0].get_figure().savefig(f"plot_torch_dort_run_mem_{compute}.png")
