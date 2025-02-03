"""
.. _l-plot-onnxscript-102:

=============================
102: Examples with onnxscript
=============================

This script gathers a couple of examples based on :epkg:`onnxscript`.

Custom Opset and Local Functions
================================
"""

import onnx
import onnxscript

op = onnxscript.opset18
my_opset = onnxscript.values.Opset("m_opset.ml", version=1)


@onnxscript.script(my_opset, default_opset=op)
def do_this(x, y):
    return op.Add(x, y)


@onnxscript.script(my_opset, default_opset=op)
def do_that(x, y):
    return op.Sub(x, y)


@onnxscript.script(my_opset, default_opset=op)
def do_this_or_do_that(x, y, do_this_or_do_that: bool = True):
    if do_this_or_do_that:
        ret = my_opset.do_this(x, y)
    else:
        ret = my_opset.do_that(x, y)
    return ret


# %%
# Then we export the model into ONNX.

proto = do_this_or_do_that.to_model_proto(functions=[do_this, do_that])
print(onnx.printer.to_text(proto))
