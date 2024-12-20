"""
.. _l-plot-optimize-101:

=======================================================
101: Onnx Model Optimization based on Pattern Rewriting
=======================================================

This example shows how to optimize a graph using pattern optimization.
The graph was obtained by running a dummy llama model.
It is the backward graph.

A model
=======
"""

import os
import onnx
import pandas
from experimental_experiment.helpers import pretty_onnx
from onnx_array_api.plotting.graphviz_helper import plot_dot
from experimental_experiment.xbuilder.graph_builder import (
    GraphBuilder,
    OptimizationOptions,
)

filename = (
    os.path.join(os.path.dirname(__file__), "data", "dort-c-custom__1.onnx")
    if "__file__" in globals()
    else "data/dort-c-custom__1.onnx"
)
proto = onnx.load(filename)

print(f"number of nodes: {len(proto.graph.node)}")


print(pretty_onnx(proto))

##############################
# And visually.

plot_dot(proto)

###############################
# Optimization
# ============

gr = GraphBuilder(
    proto,
    infer_shapes_options=True,
    optimization_options=OptimizationOptions(
        patterns="default",
        verbose=1,  # a higher value increases the verbosity when optimizations for patterns
    ),
)
stats = gr.optimize()
df = pandas.DataFrame(stats)
df.to_csv("plot_optimize.csv")
df.to_excel("plot_optimize.xlsx")
df

##############################
# Summary

for c in df.columns:
    if "time" not in c and "pattern" not in c:
        df[c] = df[c].fillna(0).astype(int)

aggs = {
    "time_in": "sum",
    "added": "sum",
    "removed": "sum",
    "iteration": "max",
    "match_index": "max",
    "instances": "sum",
}
print(df.groupby("pattern").agg(aggs))

##############################
# The total is:

diff = df["added"].sum() - df["removed"].sum()

print(f"number of removed nodes: {-diff}")

##############################
# Conversion to onnx.
optimized_proto = gr.to_onnx(optimize=False)
with open("plot_optimize_101.onnx", "wb") as f:
    f.write(optimized_proto.SerializeToString())

print(f"number of new nodes: {len(optimized_proto.graph.node)}")


##########################################
# It gives the following.

print(pretty_onnx(optimized_proto))

##############################
# And visually.

plot_dot(optimized_proto)

#################################
# The first list of patterns optimizes the graph with only
# standard onnx operators: :ref:`l-pattern-optimization-onnx`.
# The second list is specific to :epkg:`onnxruntime`:
# :ref:`l-pattern-optimization-ort`.
#
# Focus on one optimizer
# ======================

gr = GraphBuilder(
    optimized_proto,
    infer_shapes_options=True,
    optimization_options=OptimizationOptions(
        patterns="SwitchOrderBinary",
        verbose=10,
    ),
)
stats = gr.optimize()
df = pandas.DataFrame(stats)
df.to_csv("plot_optimize.csv")
df.to_excel("plot_optimize.xlsx")
df
