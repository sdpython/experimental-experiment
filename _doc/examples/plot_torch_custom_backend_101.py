"""
.. _l-plot-custom-backend:

===============================
101: A custom backend for torch
===============================

This example leverages the examples introduced on this page
`Custom Backends <https://docs.pytorch.org/stable/torch.compiler_custom_backends.html>`_.
It uses backend :func:`experimental_experiment.torch_dynamo.onnx_custom_backend`
based on :epkg:`onnxruntime` and running on CPU or CUDA.
It could easily replaced by
:func:`experimental_experiment.torch_dynamo.onnx_debug_backend`.
This one based on the reference implemented from onnx
can show the intermediate results if needed. It is very slow.

A model
=======
"""

import copy
from experimental_experiment.helpers import pretty_onnx
from onnx_array_api.plotting.graphviz_helper import plot_dot
import torch
from torch._dynamo.backends.common import aot_autograd

# from torch._functorch._aot_autograd.utils import make_boxed_func
from experimental_experiment.torch_dynamo import (
    onnx_custom_backend,
    get_decomposition_table,
)
from experimental_experiment.torch_interpreter import ExportOptions


class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(10, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.layers(x)


x = torch.randn(3, 10, dtype=torch.float32)

mlp = MLP()
print(mlp(x))

# %%
# A custom backend
# ================
#
# This backend leverages :epkg:`onnxruntime`.
# It is available through function
# :func:`experimental_experiment.torch_dynamo.onnx_custom_backend`
# and implemented by class :class:`OrtBackend
# <experimental_experiment.torch_dynamo.fast_backend.OrtBackend>`.

compiled_model = torch.compile(
    copy.deepcopy(mlp),
    backend=lambda *args, **kwargs: onnx_custom_backend(*args, target_opset=18, **kwargs),
    dynamic=False,
    fullgraph=True,
)

print(compiled_model(x))

# %%
# Training
# ========
#
# It can be used for training as well. The compilation may not
# be working if the model is using function the converter does not know.
# Maybe, there exist a way to decompose this new function into
# existing functions. A recommended list is returned by
# with function :func:`get_decomposition_table
# <experimental_experiment.torch_dynamo.get_decomposition_table>`.
# An existing list can be filtered out from some inefficient decompositions
# with function :func:`filter_decomposition_table
# <experimental_experiment.torch_dynamo.filter_decomposition_table>`.


aot_compiler = aot_autograd(
    fw_compiler=lambda *args, **kwargs: onnx_custom_backend(
        *args,
        target_opset=18,
        export_options=ExportOptions(decomposition_table=get_decomposition_table()),
        **kwargs,
    ),
)

compiled_model = torch.compile(
    copy.deepcopy(mlp),
    backend=aot_compiler,
    fullgraph=True,
    dynamic=False,
)

print(compiled_model(x))

# %%
# Let's see an iteration loop.

from sklearn.datasets import load_diabetes


class DiabetesDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X / 10).to(torch.float32)
        self.y = torch.from_numpy(y).to(torch.float32).reshape((-1, 1))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def trained_model(max_iter=5, dynamic=False, storage=None):
    aot_compiler = aot_autograd(
        fw_compiler=lambda *args, **kwargs: onnx_custom_backend(
            *args, target_opset=18, storage=storage, **kwargs
        ),
        decompositions=get_decomposition_table(),
    )

    compiled_model = torch.compile(
        MLP(),
        backend=aot_compiler,
        fullgraph=True,
        dynamic=dynamic,
    )

    trainloader = torch.utils.data.DataLoader(
        DiabetesDataset(*load_diabetes(return_X_y=True)),
        batch_size=5,
        shuffle=True,
        num_workers=0,
    )

    loss_function = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(compiled_model.parameters(), lr=1e-1)

    for epoch in range(0, max_iter):
        current_loss = 0.0

        for _, data in enumerate(trainloader, 0):
            X, y = data

            optimizer.zero_grad()
            p = compiled_model(X)
            loss = loss_function(p, y)
            loss.backward()

            optimizer.step()

            current_loss += loss.item()

        print(f"Loss after epoch {epoch+1}: {current_loss}")

    print("Training process has finished.")
    return compiled_model


trained_model(3)

# %%
# What about the ONNX model?
# ==========================
#
# The backend converts the model into ONNX then runs it with :epkg:`onnxruntime`.
# Let's see what it looks like.

storage = {}

trained_model(3, storage=storage)

print(f"{len(storage['instance'])} were created.")

for i, inst in enumerate(storage["instance"][:2]):
    print()
    print(f"-- model {i} running on {inst['providers']}")
    print(pretty_onnx(inst["onnx"]))


# %%
# The forward graph.

plot_dot(storage["instance"][0]["onnx"])


# %%
# The backward graph.

plot_dot(storage["instance"][1]["onnx"])


# %%
# What about dynamic shapes?
# ==========================
#
# Any input or output having `_dim_` in its name is a dynamic dimension.
# Any output having `_NONE_` in its name is replace by None.
# It is needed by pytorch.

storage = {}

trained_model(3, storage=storage, dynamic=True)

print(f"{len(storage['instance'])} were created.")

for i, inst in enumerate(storage["instance"]):
    print()
    print(f"-- model {i} running on {inst['providers']}")
    print()
    print(pretty_onnx(inst["onnx"]))

# %%
# The forward graph.

plot_dot(storage["instance"][0]["onnx"])


# %%
# The backward graph.

plot_dot(storage["instance"][1]["onnx"])


# %%
# Pattern Optimizations
# =====================
#
# By default, once exported into onnx, a model is optimized by
# looking for patterns. Each of them locally replaces a couple of
# nodes to optimize the computation
# (see :ref:`l-pattern-optimization-onnx` and
# :ref:`l-pattern-optimization-ort`).
