"""
.. _l-plot-exporter-recipes-onnx-exporter-custom-ops-inplace:

torch.onnx.export and a custom operator inplace
===============================================

This example shows how to convert a custom operator as defined
in the tutorial `Python Custom Operators
<https://pytorch.org/tutorials/advanced/python_custom_ops.html#python-custom-ops-tutorial>`_.

Inplace modification are not supported by onnx.

A model with a custom ops
+++++++++++++++++++++++++
"""

import numpy as np
from onnx.printer import to_text
import onnxscript
import torch


#################################
# We define a model with a custom operator.


@torch.library.custom_op("mylib::numpy_sin", mutates_args={"output"}, device_types="cpu")
def numpy_sin(x: torch.Tensor, output: torch.Tensor) -> None:
    assert x.device == output.device
    assert x.device.type == "cpu"
    x_np = x.numpy()
    output_np = output.numpy()
    np.sin(x_np, out=output_np)


class ModuleWithACustomOperator(torch.nn.Module):
    def forward(self, x):
        out = torch.zeros(x.shape)
        numpy_sin(x, out)
        return out


model = ModuleWithACustomOperator()

######################################
# Let's check it runs.
x = torch.randn(1, 3)
model(x)

######################################
# As expected, it does not export.
try:
    torch.export.export(model, (x,))
    raise AssertionError("This export should failed unless pytorch now supports this model.")
except Exception as e:
    print(e)

####################################
# The exporter fails with the same eror as it expects torch.export.export to work.

try:
    torch.onnx.export(model, (x,), dynamo=True)
except Exception as e:
    print(e)


####################################
# Registration
# ++++++++++++
#
# The exporter how to convert the new exporter into ONNX.
# This must be defined. The first piece is to tell the exporter
# that the shape of the output is the same as x.
# input names must be the same.


@numpy_sin.register_fake
def numpy_sin_shape(x, output):
    pass


#####################################
# Next is the conversion to onnx.
T = str  # a tensor name


op = onnxscript.opset18

#####################################
# Let's convert the custom op into onnx.


@onnxscript.script()
def numpy_sin_to_onnx(x):
    return op.Sin(x)


#####################################
# And we convert again.

ep = torch.onnx.export(
    model,
    (x,),
    custom_translation_table={torch.ops.mylib.numpy_sin: numpy_sin_to_onnx},
    dynamo=True,
)
print(to_text(ep.model_proto))
