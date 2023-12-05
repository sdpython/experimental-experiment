"""
Evaluate different ways to export a torch model to ONNX
=======================================================

The example evaluates the performance of onnxruntime of a simple
torch model after it was converted into ONNX through different processes:

* `TorchScript-based ONNX Exporter
  <https://pytorch.org/docs/stable/onnx.html#torchscript-based-onnx-exporter>`_,
  let's call it **script**
* `TorchDynamo-based ONNX Exporter
  <https://pytorch.org/docs/stable/onnx.html#torchdynamo-based-onnx-exporter>`_,
  let's call it **dynamo**
* if available, the previous model but optimized, **dynopt**
* a custom exporter **custom**, this exporter supports a very limited
  set of models, as **dynamo**, it relies on
  `torch.fx <https://pytorch.org/docs/stable/fx.html>`_ but the design is closer to
  what tensorflow-onnx does.
* the same exporter but unused nodes were removed, **cusopt**

Some helpers
++++++++++++
"""
import itertools
import os
import platform
import pprint
import multiprocessing
import time
import cProfile
import pstats
import io
from pstats import SortKey
import numpy as np
import matplotlib.pyplot as plt
import pandas
import onnx
from onnx_extended.ext_test_case import measure_time
import torch
from torch import nn
import torch.nn.functional as F
import experimental_experiment
from experimental_experiment.torch_exp.onnx_export import to_onnx
from tqdm import tqdm


def system_info():
    obs = {}
    obs["processor"] = platform.processor()
    obs["cores"] = multiprocessing.cpu_count()
    obs["cuda"] = 1 if torch.cuda.is_available() else 0
    obs["cuda_count"] = torch.cuda.device_count()
    obs["cuda_name"] = torch.cuda.get_device_name()
    obs["cuda_capa"] = torch.cuda.get_device_capability()
    return obs


pprint.pprint(system_info())


############################
# The model
# +++++++++
#
# A simple model to convert.


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, 5)
        self.conv2 = nn.Conv2d(128, 16, 5)
        self.fc1 = nn.Linear(13456, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


#######################################
# The exporters
# +++++++++++++


def export_script(filename, model, *args):
    torch.onnx.export(model, *args, filename, input_names=["input"])


def export_dynamo(filename, model, *args):
    export_output = torch.onnx.dynamo_export(model, *args)
    export_output.save(filename)


def export_dynopt(filename, model, *args):
    export_output = torch.onnx.dynamo_export(model, *args)
    export_output.save(filename)
    model_onnx = onnx.load(filename)

    from onnxrewriter.optimizer import optimize

    optimized_model = optimize(model_onnx)
    with open(filename, "wb") as f:
        f.write(optimized_model.SerializeToString())


def export_custom(filename, model, *args):
    onx = to_onnx(model, tuple(args), input_names=["input"])
    with open(filename, "wb") as f:
        f.write(onx.SerializeToString())


def export_cus_p1(filename, model, *args):
    onx = to_onnx(model, tuple(args), input_names=["input"], remove_unused=True)
    with open(filename, "wb") as f:
        f.write(onx.SerializeToString())


def export_cus_p2(filename, model, *args):
    onx = to_onnx(
        model,
        tuple(args),
        input_names=["input"],
        remove_unused=True,
        constant_folding=True,
    )
    with open(filename, "wb") as f:
        f.write(onx.SerializeToString())


#########################################
# Let's check they are working.

export_functions = [
    export_script,
    export_dynamo,
    export_dynopt,
    export_custom,
    export_cus_p1,
    export_cus_p2,
]

exporters = {f.__name__.replace("export_", ""): f for f in export_functions}
shape = [1, 1, 128, 128]
input_tensor = torch.rand(*shape).to(torch.float32)
model = MyModel()

supported_exporters = {}
for k, v in exporters.items():
    print(f"run exporter {k}")
    filename = f"plot_torch_export_{k}.onnx"
    try:
        v(filename, model, input_tensor)
    except Exception as e:
        print(f"skipped due to {e}")
        continue
    supported_exporters[k] = v
    print("done.")


#################################
# Exporter speed
# ++++++++++++++

data = []

for k, v in supported_exporters.items():
    print(f"run exporter {k}")
    filename = f"plot_torch_export_{k}.onnx"
    times = []
    for i in range(5):
        begin = time.perf_counter()
        v(filename, model, input_tensor)
        duration = time.perf_counter() - begin
        times.append(duration)
    times.sort()
    onx = onnx.load(filename)
    print("done.")
    data.append(
        dict(
            export=k,
            time=np.mean(duration),
            min=times[0],
            max=times[-1],
            std=np.std(times),
            nodes=len(onx.graph.node),
        )
    )


#########################################
# The last export to measure time torch spends in export the model
# before any other export can begin the translation
# except the first one.

begin = time.perf_counter()
for i in range(5):
    exported_mod = torch.export.export(model, (input_tensor,))
duration = time.perf_counter() - begin
data.append(dict(export="torch", time=duration / 5))

#############################
# The result.
df1 = pandas.DataFrame(data)
print(df1)

fig, ax = plt.subplots(1, 1)
df1[["export", "time"]].set_index("export")["time"].plot.barh(
    ax=ax, title="Export time", yerr=df1["std"]
)
fig.tight_layout()
fig.savefig("plot_torch_export.png")

####################################
# Profiling
# +++++++++

pr = cProfile.Profile()
pr.enable()
for i in range(5):
    export_custom("dummy.onnx", model, input_tensor)
pr.disable()
s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()


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
    return text


text = "\n".join(s.getvalue().split("\n")[:200])
print(clean_text(text))

############################################
# The following display helps to understand.
# Most of the tiume added by the custom converter is used to
# converter the initializer and build the onnx model once the conversion
# is complete.

# from onnx_array_api.profiling import profile2graph
# root, nodes = profile2graph(ps, clean_text=clean_text)
# text = root.to_text()
# print(text)

######################################
# Benchmark
# +++++++++


def benchmark():
    from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel

    shape = [1, 1, 128, 128]
    data = []
    confs = list(
        itertools.product(
            [_ for _ in os.listdir(".") if ".onnx" in _ and _.startswith("plot_torch")],
            [
                ["CPUExecutionProvider"],
                ["CUDAExecutionProvider", "CPUExecutionProvider"],
            ],
            ["0", "1"],
        )
    )
    loop = tqdm(confs)
    print(f"number of experiments: {len(loop)}")
    for name, ps, aot in loop:
        root = os.path.split(name)[-1]
        _, ext = os.path.splitext(root)
        if ext != ".onnx":
            continue

        obs = {}  # system_info()
        obs["name"] = name
        obs["providers"] = ",".join(ps)
        p = "CUDA" if "CUDA" in obs["providers"] else "CPU"
        obs["compute"] = p
        obs["aot"] = 1 if aot == "0" else 0
        obs["export"] = name.replace("plot_torch_export_", "").replace(".onnx", "")

        onx = onnx.load(name)
        obs["n_nodes"] = len(onx.graph.node)
        obs["n_function"] = len(onx.functions or [])
        obs["n_sub"] = len([n for n in onx.graph.node if n.op_type == "Sub"])

        opts = SessionOptions()
        opts.add_session_config_entry("session.disable_aot_function_inlining", aot)
        opts.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.optimized_model_filepath = (
            f"ort-{name.replace('.onnx', '')}-{p.lower()}-aot{aot}.onnx"
        )

        try:
            sess = InferenceSession(name, opts, providers=ps)
        except Exception as e:
            loop.set_description(f"ERROR-load: {name} {e}")
            obs.update({"error": e, "step": "run"})
            data.append(obs)
            continue

        input_name = sess.get_inputs()[0].name
        feeds = {input_name: np.random.rand(*shape).astype(np.float32)}
        try:
            for i in range(0, 5):
                sess.run(None, feeds)
        except Exception as e:
            loop.set_description(f"ERROR-run: {name} {e}")
            obs.update({"error": e, "step": "load"})
            data.append(obs)
            continue
        obs.update(measure_time(lambda: sess.run(None, feeds), max_time=1))

        loop.set_description(f"{obs['average']} {name} {ps}")
        data.append(obs)

    df = pandas.DataFrame(data)
    df.to_csv("benchmark.csv", index=False)
    df.to_excel("benchmark.xlsx", index=False)
    return df


df = benchmark()
print(df)

#####################################
# Other view

piv = pandas.pivot_table(
    df, index="export", columns=["compute", "aot"], values="average"
)
print(piv)

fig, ax = plt.subplots()
piv.plot.barh(ax=ax, title="Compares onnxruntime time on exported models")
fig.tight_layout()
fig.savefig("plot_torch_export_ort.png")
