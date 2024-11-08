Exporter Recipes
================

A model can be converted to ONNX if :func:`torch.export.export` is
able to convert that model into a graph. This is not always
possible but usually possible with some code changes.
These changes may not be desired as they may hurt the performance
and make the code more complex than it should be.
The conversion is a necessary step to be able to use
ONNX. Next examples shows some recurrent code patterns and
ways to rewrite them so that the exporter works.
