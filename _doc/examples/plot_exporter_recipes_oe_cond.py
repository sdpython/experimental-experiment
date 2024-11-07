"""
.. _l-plot-exporter-recipes-onnx-exporter-cond:

torch.onnx.export and a model with a test
=========================================

Control flow cannot be exported with a change.
The code of the model can be changed or patched
to introduce function :func:`torch.cond`.

A model with a test
+++++++++++++++++++
"""

from onnx.printer import to_text
import torch


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
x = torch.randn(3)
model(x)

######################################
# As as expected, it does not export.
try:
    torch.export.export(model, (x,))
    raise AssertionError("This export should failed unless pytorch now supports this model.")
except Exception as e:
    print(e)

####################################
# It does export with torch.onnx.export because it uses JIT to trace the execution.
# But the model is not exactly the same as the initial model.
ep = torch.onnx.export(model, (x,), dynamo=True)
print(to_text(ep.model_proto))


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

ep = torch.onnx.export(model, (x,), dynamo=True)
print(to_text(ep.model_proto))


####################################
# Let's optimize to see a small model.

ep = torch.onnx.export(model, (x,), dynamo=True)
ep.optimize()
print(to_text(ep.model_proto))
