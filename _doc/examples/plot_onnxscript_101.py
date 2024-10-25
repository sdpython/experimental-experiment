"""
.. _l-plot-onnxscript-101:

=============================
101: Examples with onnxscript
=============================


Custom Opset
============
"""

import onnx
import onnxscript

op = onnxscript.opset18
my_opset = onnxscript.values.Opset("m_opset.ml", version=1)


@onnxscript.script(my_opset, default_opset=op)
def do_this(x, y, do_this_or_do_that: bool = True):
    return op.Add(x, y)


@onnxscript.script(my_opset, default_opset=op)
def do_that(x, y, do_this_or_do_that: bool = True):
    return op.Sub(x, y)


@onnxscript.script(my_opset, default_opset=op)
def do_this_or_do_that(x, y, do_this_or_do_that: bool = True):
    if do_this_or_do_that:
        ret = my_opset.do_this(x, y=y)
    else:
        ret = my_opset.do_that(x, y=y)
    return ret


# proto = do_this_or_do_that.to_function_proto()
# print(onnx.printer.to_text(proto))


proto = do_this_or_do_that.to_model_proto(functions=[do_this, do_that])
print(onnx.printer.to_text(proto))
