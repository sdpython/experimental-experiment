"""
.. _l-plot-exporter-recipes-custom-cond:

to_onnx and a model with a test
===============================

Control flow cannot be exported with a change.
The code of the model can be changed or patched
to introduce function :func:`torch.cond`.

A model with a test
+++++++++++++++++++
"""

import torch
from experimental_experiment.helpers import pretty_onnx
from experimental_experiment.torch_interpreter import to_onnx


#################################
# We define a model with a control flow (-> graph break)


class ModuleWithControlFlow(torch.nn.Module):
    def forward(self, x):
        if x.sum():
            return x * 2
        return -x


class ModelWithControlFlow(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(3, 2),
            torch.nn.Linear(2, 1),
            ModuleWithControlFlow(),
        )

    def forward(self, x):
        out = self.mlp(x)
        return out


model = ModelWithControlFlow()

######################################
# Let's check it runs.
x = torch.randn(1, 3)
model(x)

######################################
# As as expected, it does not export.
try:
    torch.export.export(model, (x,))
    raise AssertionError("This export should failed unless pytorch now supports this model.")
except Exception as e:
    print(e)

####################################
# The exporter fails with the same eror as it expects torch.export.export to work.

try:
    to_onnx(model, (x,))
except Exception as e:
    print(e)


####################################
# Suggested Patch
# +++++++++++++++
#
# Let's avoid the graph break by replacing the forward.


def new_forward(x):
    def identity2(x):
        return x * 2

    def neg(x):
        return -x

    return torch.cond(x.sum() > 0, identity2, neg, (x,))


print("the list of submodules")
for name, mod in model.named_modules():
    print(name, type(mod))
    if isinstance(mod, ModuleWithControlFlow):
        mod.forward = new_forward

####################################
# Let's export again.

onx = to_onnx(model, (x,))
print(pretty_onnx(onx))

####################################
# We can also inline the local function.

onx = to_onnx(model, (x,), inline=True)
print(pretty_onnx(onx))
