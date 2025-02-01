"""
.. _l-plot-exporter-recipes-onnx-exporter-custom-ops-fct:

torch.onnx.export and a custom operator registered with a function
==================================================================

This example shows how to convert a custom operator, inspired from
`Python Custom Operators
<https://pytorch.org/tutorials/advanced/python_custom_ops.html#python-custom-ops-tutorial>`_.

A model with a custom ops
+++++++++++++++++++++++++
"""

import numpy as np
from onnx.printer import to_text
import onnxscript
import torch


# %%
# We define a model with a custom operator.


def numpy_sin(x: torch.Tensor) -> torch.Tensor:
    assert x.device.type == "cpu"
    x_np = x.numpy()
    return torch.from_numpy(np.sin(x_np))


class ModuleWithACustomOperator(torch.nn.Module):
    def forward(self, x):
        return numpy_sin(x)


model = ModuleWithACustomOperator()

# %%
# Let's check it runs.
x = torch.randn(1, 3)
model(x)

# %%
# As expected, it does not export.
try:
    torch.export.export(model, (x,))
    raise AssertionError("This export should failed unless pytorch now supports this model.")
except Exception as e:
    print(e)

# %%
# The exporter fails with the same eror as it expects torch.export.export to work.

try:
    torch.onnx.export(model, (x,), dynamo=True)
except Exception as e:
    print(e)


# %%
# Registration
# ++++++++++++
#
# The exporter how to convert the new exporter into ONNX.
# This must be defined. The first piece is to tell the exporter
# that the shape of the output is the same as x.
# input names must be the same.
# We also need to rewrite the module to be able to use it.


def register(fct, fct_shape, namespace, fname):
    schema_str = torch.library.infer_schema(fct, mutates_args=())
    custom_def = torch.library.CustomOpDef(namespace, fname, schema_str, fct)
    custom_def.register_kernel("cpu")(fct)
    custom_def._abstract_fn = fct_shape


register(numpy_sin, lambda x: torch.empty_like(x), "mylib", "numpy_sin")

# %%
# We also need to rewrite the module to be able to use it.


class ModuleWithACustomOperator(torch.nn.Module):
    def forward(self, x):
        return torch.ops.mylib.numpy_sin(x)


model = ModuleWithACustomOperator()

# %%
# Let's check it runs again.
model(x)

# %%
# Let's see what the fx graph looks like.

print(torch.export.export(model, (x,)).graph)

# %%
# Next is the conversion to onnx.

op = onnxscript.opset18


@onnxscript.script()
def numpy_sin_to_onnx(x):
    return op.Sin(x)


# %%
# And we convert again.

ep = torch.onnx.export(
    model,
    (x,),
    custom_translation_table={torch.ops.mylib.numpy_sin.default: numpy_sin_to_onnx},
    dynamo=True,
)
print(to_text(ep.model_proto))
