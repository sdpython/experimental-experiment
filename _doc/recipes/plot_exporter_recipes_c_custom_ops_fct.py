"""
.. _l-plot-exporter-recipes-custom-custom-ops-fct:

to_onnx and a custom operator registered with a function
========================================================

This example shows how to convert a custom operator, inspired from
`Python Custom Operators
<https://pytorch.org/tutorials/advanced/python_custom_ops.html#python-custom-ops-tutorial>`_.

A model with a custom ops
+++++++++++++++++++++++++
"""

from typing import Any, Dict, List
import numpy as np
import torch
from onnx_array_api.plotting.graphviz_helper import plot_dot
from experimental_experiment.xbuilder import GraphBuilder
from experimental_experiment.helpers import pretty_onnx
from experimental_experiment.torch_interpreter import to_onnx, Dispatcher


#################################
# We define a model with a custom operator.


def numpy_sin(x: torch.Tensor) -> torch.Tensor:
    assert x.device.type == "cpu"
    x_np = x.numpy()
    return torch.from_numpy(np.sin(x_np))


class ModuleWithACustomOperator(torch.nn.Module):
    def forward(self, x):
        return numpy_sin(x)


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
    to_onnx(model, (x,))
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


def register(fct, fct_shape, namespace, fname):
    schema_str = torch.library.infer_schema(fct, mutates_args=())
    custom_def = torch.library.CustomOpDef(namespace, fname, schema_str, fct)
    custom_def.register_kernel("cpu")(fct)
    custom_def._abstract_fn = fct_shape


register(numpy_sin, lambda x: torch.empty_like(x), "mylib", "numpy_sin")


class ModuleWithACustomOperator(torch.nn.Module):
    def forward(self, x):
        return torch.ops.mylib.numpy_sin(x)


model = ModuleWithACustomOperator()

###########################
# Let's check it runs again.
model(x)

####################################
# Let's see what the fx graph looks like.

print(torch.export.export(model, (x,)).graph)

#####################################
# Next is the conversion to onnx.
T = str  # a tensor name


def numpy_sin_to_onnx(
    g: GraphBuilder,
    sts: Dict[str, Any],
    outputs: List[str],
    x: T,
    name: str = "mylib.numpy_sin",
) -> T:
    # name= ... lets the user know when the node comes from
    return g.op.Sin(x, name=name, outputs=outputs)


####################################
# We create a :class:`Dispatcher <experimental_experiment.torch_interpreter.Dispatcher>`.

dispatcher = Dispatcher({"mylib::numpy_sin": numpy_sin_to_onnx})

#####################################
# And we convert again.

onx = to_onnx(model, (x,), dispatcher=dispatcher, optimize=False)
print(pretty_onnx(onx))

#####################################
# And we convert again with optimization this time.

onx = to_onnx(model, (x,), dispatcher=dispatcher, optimize=True)
print(pretty_onnx(onx))

###################################
# Let's make sure the node was produce was the user defined converter for numpy_sin.
# The name should be 'mylib.numpy_sin'.

print(onx.graph.node[0])

####################################
# And visually.

plot_dot(onx)
