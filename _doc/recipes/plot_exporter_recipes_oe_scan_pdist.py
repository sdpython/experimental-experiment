"""
.. _l-plot-exporter-recipes-onnx-exporter-pdist:

torch.onnx.export and a model with a loop (scan)
================================================

Control flow cannot be exported with a change.
The code of the model can be changed or patched
to introduce function :func:`torch.ops.higher_order.scan`.

Pairwise Distance
+++++++++++++++++

We appy loops to the pairwise distances (:class:`torch.nn.PairwiseDistance`).
"""

import scipy.spatial.distance as spd
import torch
from onnx_array_api.plotting.graphviz_helper import plot_dot
from experimental_experiment.helpers import pretty_onnx


class ModuleWithControlFlowLoop(torch.nn.Module):
    def forward(self, x, y):
        dist = torch.empty((x.shape[0], y.shape[0]), dtype=x.dtype)
        for i in range(x.shape[0]):
            sub = y - x[i : i + 1]
            d = torch.sqrt((sub * sub).sum(axis=1))
            dist[i, :] = d
        return dist


model = ModuleWithControlFlowLoop()
x = torch.randn(3, 4)
y = torch.randn(5, 4)
pwd = spd.cdist(x.numpy(), y.numpy())
expected = torch.from_numpy(pwd)
print(f"shape={pwd.shape}, discrepancies={torch.abs(expected - model(x,y)).max()}")

# %%
# :func:`torch.export.export` works because it unrolls the loop.
# It works if the input size never change.


ep = torch.export.export(model, (x, y))
print(ep.graph)

# %%
# However, with dynamic shapes, that's another story.

x_rows = torch.export.Dim("x_rows")
y_rows = torch.export.Dim("y_rows")
dim = torch.export.Dim("dim")
try:
    ep = torch.export.export(
        model, (x, y), dynamic_shapes={"x": {0: x_rows, 1: dim}, "y": {0: y_rows, 1: dim}}
    )
    print(ep.graph)
except Exception as e:
    print(e)

# %%
# Suggested Patch
# +++++++++++++++
#
# We need to rewrite the module with function
# :func:`torch.ops.higher_order.scan`.


def dist(y: torch.Tensor, scanned_x: torch.Tensor):
    sub = y - scanned_x.reshape((1, -1))
    sq = sub * sub
    rd = torch.sqrt(sq.sum(axis=1))
    # clone --> UnsupportedAliasMutationException:
    # Combine_fn might be aliasing the input!
    return [y.clone(), rd]


class ModuleWithControlFlowLoopScan(torch.nn.Module):

    def forward(self, x, y):
        carry, out = torch.ops.higher_order.scan(
            dist,
            [y],
            [x],
            dim=0,
            reverse=False,
            additional_inputs=[],
        )
        return out


model = ModuleWithControlFlowLoopScan()
print(f"shape={pwd.shape}, discrepancies={torch.abs(expected - model(x,y)).max()}")

# %%
# That works. Let's export again.

ep = torch.export.export(
    model, (x, y), dynamic_shapes={"x": {0: x_rows, 1: dim}, "y": {0: y_rows, 1: dim}}
)
print(ep.graph)

# %%
# The graph shows some unused results and this might confuse the exporter.
# We need to run :meth:`torch.export.ExportedProgram.run_decompositions`.
ep = ep.run_decompositions({})
print(ep.graph)

# %%
# Let's export again with ONNX.

onx = torch.onnx.export(
    model,
    (x, y),
    dynamic_shapes={"x": {0: x_rows, 1: dim}, "y": {0: y_rows, 1: dim}},
    dynamo=True,
)
print(pretty_onnx(onx))

# %%
# And visually.

plot_dot(onx)
