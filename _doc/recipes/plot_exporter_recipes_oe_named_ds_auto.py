"""
.. _l-plot-exporter-recipes-onnx-exporter-modules:

torch.onnx.export: Rename Dynamic Shapes
========================================

Example given in :ref:`l-plot-exporter-dynamic_shapes` can only be exported
with dynamic shapes using ``torch.export.Dim.AUTO``. As a result, the exported
onnx models have dynamic dimensions with unpredictable names.

Model with unpredictable names for dynamic shapes
+++++++++++++++++++++++++++++++++++++++++++++++++
"""

import torch
from experimental_experiment.helpers import pretty_onnx


class Model(torch.nn.Module):
    def forward(self, x, y, z):
        return torch.cat((x, y), axis=1) + z[:, ::2]


model = Model()
x = torch.randn(2, 3)
y = torch.randn(2, 5)
z = torch.randn(2, 16)
model(x, y, z)

# %%
# Let's export it.

AUTO = torch.export.Dim.AUTO
ep = torch.export.export(
    model,
    (x, y, z),
    dynamic_shapes=({0: AUTO, 1: AUTO}, {0: AUTO, 1: AUTO}, {0: AUTO, 1: AUTO}),
)

# %%
# Let's convert it into ONNX.

onx = torch.onnx.export(ep).model_proto

for inp in onx.graph.input:
    print(f" input: {pretty_onnx(inp)}")
for out in onx.graph.output:
    print(f"output: {pretty_onnx(out)}")

# %%
# Rename the dynamic shapes
# +++++++++++++++++++++++++
#
# We just need to give the onnx exporter the same information
# :func:`torch.export.export` was given but we replace ``AUTO``
# by the name this dimension should have.

onx = torch.onnx.export(
    ep,
    dynamic_shapes=(
        {0: "batch", 1: "dx"},
        {0: "batch", 1: "dy"},
        {0: "batch", 1: "dx+dy"},
    ),
).model_proto

for inp in onx.graph.input:
    print(f" input: {pretty_onnx(inp)}")
for out in onx.graph.output:
    print(f"output: {pretty_onnx(out)}")

# %%
# A model with an unknown output shape
# ++++++++++++++++++++++++++++++++++++


class UnknownOutputModel(torch.nn.Module):
    def forward(self, x):
        return torch.nonzero(x)


model = UnknownOutputModel()
x = torch.randint(0, 2, (10, 2))
model(x)

# %%
# Let's export it.

ep = torch.export.export(
    model,
    (x,),
    dynamic_shapes=({0: torch.export.Dim("batch"), 1: AUTO},),
)
print(ep)

# %%
# Let's export it into ONNX.

onx = torch.onnx.export(ep, dynamic_shapes=({0: "batch", 1: "dx"},), dynamo=True).model_proto

for inp in onx.graph.input:
    print(f" input: {pretty_onnx(inp)}")
for out in onx.graph.output:
    print(f"output: {pretty_onnx(out)}")

# %%
# The exporter has detected a dimension could not be infered
# from the input shape somewhere in the graph and introduced a
# new dimension name.
# Let's rename it as well. Let's also change the output name
# because the functionality may not be implemented yet when
# the output dynamic shapes are given as a tuple.

try:
    onx = torch.onnx.export(
        ep,
        dynamic_shapes=({0: "batch", 1: "dx"},),
        output_dynamic_shapes={"zeros": {0: "num_zeros"}},
        output_names=["zeros"],
        dynamo=True,
    ).model_proto
    raise AssertionError(
        "able to rename output dynamic dimensions, please update the tutorial"
    )
except (TypeError, torch.onnx._internal.exporter._errors.ConversionError) as e:
    print(f"unable to rename output dynamic dimensions due to {e}")
    onx = None

if onx is not None:
    for inp in onx.graph.input:
        print(f" input: {pretty_onnx(inp)}")
    for out in onx.graph.output:
        print(f"output: {pretty_onnx(out)}")
