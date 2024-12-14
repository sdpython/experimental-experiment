"""
.. _l-plot-exporter-dynamic_shapes:

A few tricks about dynamic shapes
=================================

Settings the dynamic shapes is not always easy.
Here are a few tricks to make it work.

dx + dy not allowed?
++++++++++++++++++++
"""

import torch


class Model(torch.nn.Module):
    def forward(self, x, y, z):
        return torch.cat((x, y), axis=1) + z


model = Model()
x = torch.randn(2, 3)
y = torch.randn(2, 4)
z = torch.randn(2, 7)
model(x, y, z)


print(torch.export.export(model, (x, y, z)).graph)

############################################
# Everything is fine so far. With dynamic shapes now.
# dx + dy is not allowed...

batch = torch.export.Dim("batch")
dx = torch.export.Dim("dz")
dy = torch.export.Dim("dy")

try:
    dz = dx + dy
except Exception as e:
    print(f"unable to add dimension because {e}")

##########################################
# Then we could make it a different one.

dz = torch.export.Dim("dz")
try:
    torch.export.export(
        model,
        (x, y, z),
        dynamic_shapes={
            "x": {0: batch, 1: dx},
            "y": {0: batch, 1: dy},
            "z": {0: batch, 1: dz},
        },
    )
except Exception as e:
    print(e)

########################################
# Still no luck but with ``torch.export.Dim.DYNAMIC``.

ep = torch.export.export(
    model,
    (x, y, z),
    dynamic_shapes={
        "x": {0: batch, 1: dx},
        "y": {0: batch, 1: dy},
        "z": {0: batch, 1: torch.export.Dim.DYNAMIC},
    },
)

print(ep)

#####################################
# Still no luck but with ``torch.export.Dim.AUTO``.

print(
    torch.export.export(
        model,
        (x, y, z),
        dynamic_shapes=(
            {0: batch, 1: torch.export.Dim.STATIC},
            {0: batch, 1: torch.export.Dim.AUTO},
            {0: batch, 1: torch.export.Dim.AUTO},
        ),
    )
)
