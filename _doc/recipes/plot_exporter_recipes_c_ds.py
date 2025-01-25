"""
.. _l-plot-exporter-recipes-custom-ds:

to_onnx and infer dynamic shapes
================================

We try to make it easier to export with dynamic shapes.
To do that, we run the model at least twice with a different
set of inputs and we try to guess the dynamic shapes found
along the way.


Infer dynamic shapes
++++++++++++++++++++
"""

import onnx
from onnx_array_api.plotting.graphviz_helper import plot_dot
import torch
from experimental_experiment.torch_interpreter import to_onnx
from experimental_experiment.torch_interpreter.piece_by_piece import (
    trace_execution_piece_by_piece,
)


class MA(torch.nn.Module):
    def forward(self, x, y):
        return x + y


class MM(torch.nn.Module):
    def forward(self, x, y):
        return x * y


class MASMM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ma = MA()
        self.mm = MM()

    def forward(self, x, y, z):
        return self.ma(x, y) - self.mm(y, z)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ma = MA()
        self.masmm = MASMM()

    def forward(self, x):
        return self.ma(x, self.masmm(x, x, x))


# %%
# The model.
model = Model()

# %%
# Two sets of inputs.
inputs = [
    ((torch.randn((5, 6)),), {}),
    ((torch.randn((6, 6)),), {}),
]

# %%
# Then we run the model, stores intermediates inputs and outputs,
# to finally guess the dynamic shapes.
diag = trace_execution_piece_by_piece(model, inputs, verbose=0)
pretty = diag.pretty_text(with_dynamic_shape=True)
print(pretty)

# %%
# The dynamic shapes are obtained with:
ds = diag.guess_dynamic_shapes()
print(ds)


# %%
# Export
# ++++++
#
# We use these dynamic shapes to export.

onx, builder = to_onnx(
    model, inputs[0][0], kwargs=inputs[0][1], dynamic_shapes=ds[0], return_builder=True
)
onnx.save(onx, "plot_exporter_recipes_c_ds.onnx")
print(builder.pretty_text())


####################################
# And visually.

plot_dot(onx)
