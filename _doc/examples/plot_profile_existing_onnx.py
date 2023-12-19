"""
Profile an existing model
=========================

Profiles any onnx model on CPU.

Preparation
+++++++++++
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from onnx_extended.args import get_parsed_args
from onnx_extended.tools.js_profile import (
    js_profile_to_dataframe,
    plot_ort_profile,
)

filename = os.path.join(os.path.dirname(__file__ or ""), "example_4700-CPUep-opt.onnx")

script_args = get_parsed_args(
    "plot_profile_existing_onnx",
    filename=(filename, "input file"),
    repeat=10,
    expose="",
)


for att in "filename,repeat".split(","):
    print(f"{att}={getattr(script_args, att)}")

########################################
# Random inputs.


def create_random_input(sess):
    feeds = {}
    for i in sess.get_inputs():
        shape = i.shape
        ot = i.type
        if ot == "tensor(float)":
            dtype = np.float32
        else:
            raise ValueError(f"Unsupposed onnx type {ot}.")
        t = np.random.rand(*shape).astype(dtype)
        feeds[i.name] = t
    return feeds


def create_session(filename, profiling=False):
    from onnxruntime import InferenceSession, SessionOptions

    if not profiling:
        return InferenceSession(filename, providers=["CPUExecutionProvider"])
    opts = SessionOptions()
    opts.enable_profiling = True
    return InferenceSession(filename, opts, providers=["CPUExecutionProvider"])


sess = create_session(script_args.filename)
feeds = create_random_input(sess)
sess.run(None, feeds)


#######################################
# Profiling
# +++++++++

sess = create_session(script_args.filename, profiling=True)

for i in range(script_args.repeat):
    sess.run(None, feeds)

prof = sess.end_profiling()
df = js_profile_to_dataframe(prof, first_it_out=True)
df.to_csv("plot_profile_existing_onnx.csv")
df.to_excel("plot_profile_existing_onnx.xlsx")
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
plot_ort_profile(df, ax[0], ax[1], "dort")
fig.savefig("plot_profile_existing_onnx.png")
