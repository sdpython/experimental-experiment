"""
.. _l-plot-exporter-lost_dynamic_dimension:

A lost dyanmic dimension
=================================


A dynamic dimension is replaced by a constant
+++++++++++++++++++++++++++++++++++++++++++++
"""

import torch


def dummy_function(idx, x_len):
    # [1, 2, 3] becomes [1, 2, 3, x_len]
    return torch.nn.functional.pad(idx, (0, 1), value=x_len)


class Model(torch.nn.Module):
    def forward(self, x, y):
        padded = dummy_function(x, y.shape[0])
        return padded.reshape((-1, 1)) + torch.arange(padded.max()).reshape((1, -1))


model = Model()
inputs = (
    (torch.arange(3) + 1).to(torch.int64),
    torch.tensor([0, 5], dtype=torch.int64),
)
print(model(*inputs))

# %%
# Let's export.
AUTO = torch.export.Dim.AUTO
ep = torch.export.export(model, inputs, dynamic_shapes={"x": {0: AUTO}, "y": {0: AUTO}})

# %%
# Let's check it works.
print(ep.module()(*inputs))

# %%
# Let's print the graph.
print(ep.graph)

# %%
# It shows the following line
# ``[torch.ops.aten.pad.default](args = (%x, [0, 1], constant, 2.0)``
# which corresponds to ``torch.nn.functional.pad(idx, (0, 1), value=x_len)``.
# But in this case, ``x_len`` is equal to ``y.shape[0]`` which was defined
# as a dynamic dimension. Se if we choose something like the following:

inputs2 = (
    (torch.arange(3) + 1).to(torch.int64),
    torch.tensor([0, 5, 6], dtype=torch.int64),
)

# %%
# The original model works.
print(model(*inputs2))

# %%
# But the exported program does not.
try:
    print(ep.module()(*inputs2))
except Exception as e:
    print(e)

# %%
# How to fix it?
# ++++++++++++++
#
# In this particular case, function is not the only way ``pad``
# to produce the desired result.


def dummy_function(idx, x_len):
    # [1, 2, 3] becomes [1, 2, 3, x_len]
    return torch.cat([idx, torch.tensor([x_len], dtype=torch.int64)], dim=0)


class ModelCat(torch.nn.Module):
    def forward(self, x, y):
        padded = dummy_function(x, y.shape[0])
        return padded.reshape((-1, 1)) + torch.arange(padded.max()).reshape((1, -1))


modelcat = ModelCat()
print(modelcat(*inputs))

# %%
# Let's export.
epcat = torch.export.export(modelcat, inputs, dynamic_shapes={"x": {0: AUTO}, "y": {0: AUTO}})

# %%
# Let's check it works.
print(epcat.module()(*inputs))

# %%
# Let's print the graph.
print(epcat.graph)

# %%
# And the final verification.
print(epcat.module()(*inputs2))

# %%
# It finally works.
