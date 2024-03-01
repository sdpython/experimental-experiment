"""
==================
Graph Optimization
==================

This example shows how to optimize a graph using pattern optimization.
The graph was obtained by running a dummy llama model.
It is the backward graph.

A model
=======
"""

import os
import onnx
import pandas
from onnx_array_api.plotting.text_plot import onnx_simple_text_plot
from onnx_array_api.plotting.graphviz_helper import plot_dot
from experimental_experiment.torch_exp.graph_builder import (
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


print(onnx_simple_text_plot(proto))

##############################
# And visually.

plot_dot(proto)

###############################
# Optimization
# ============

gr = GraphBuilder(
    proto,
    infer_shapes=True,
    optimization_options=OptimizationOptions(patterns="default"),
)
stats = gr.optimize()
df = pandas.DataFrame(stats)
df.to_csv("plot_optimize.csv")
df.to_excel("plot_optimize.xlsx")
df

##############################
# Summary

print(df[["pattern", "added", "removed"]].groupby("pattern").sum())

##############################
# The total is:

diff = df["added"].sum() - df["removed"].sum()

print(f"number of removed nodes: {-diff}")

##############################
# Conversion to onnx.
optimized_proto = gr.to_onnx(optimize=False)

print(f"number of new nodes: {len(optimized_proto.graph.node)}")


##########################################
# It gives the following.

print(onnx_simple_text_plot(optimized_proto))

##############################
# And visually.

plot_dot(optimized_proto)

#################################
# The first list of patterns optimizes the graph with only
# standard onnx operators: :ref:`l-pattern-optimization-onnx`.
# The second list is specific to :pekg:`onnxruntime`:
# :ref:`l-pattern-optimization-ort`.
