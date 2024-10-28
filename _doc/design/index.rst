.. _l-design:

======
Design
======

This module was started to experiment around function :func:`torch.export.export`
and see what kind of issues occur when leveraging that function to convert
a :class:`torch.nn.Module` into :epkg:`ONNX`.
The design is organized around the following main pieces:

* :class:`GraphBuilder <experimental_experiment.xbuilder.GraphBuilder>`:
  simplifies the creation of an ONNX model with the inner API
  available in :epkg:`onnx`. It also tracks type and shape information.
  It supports shape inference and constant computation with
  numpy arrays or torch tensors.
* :class:`DynamoInterpreter <experimental_experiment.torch_interpreter.interpreter.DynamoInterpreter>`:
  walks through a torch model and converts every piece into ONNX nodes,
  the first step is to get an :class:`torch.fx.Graph`, then convert every of its nodes
  into ONNX operators.
* :mod:`_aten_functions <experimental_experiment.torch_interpreter._aten_functions>` and
  :mod:`_aten_methods <experimental_experiment.torch_interpreter._aten_methods>`:
  a collection of functions converting every node from :class:`torch.fx.Graph` instance
  into ONNX operators. There is one function per node type.
* :class:`GraphBuilderPatternOptimization <experimental_experiment.xoptim.GraphBuilderPatternOptimization>`:
  once the model is converted, this class looks for sequence of ONNX nodes which can
  be rewritten in a more efficient way. This can be applied on graphs produced by
  the exporter and any existing graphs.

Function :func:`to_onnx <experimental_experiment.torch_interpreter.to_onnx>`
calls these pieces to produce an ONNX models. Next sections gives more details
about how it is implemented. It may not be fully up to date.

.. toctree::
    :maxdepth: 1
    
    exporter
    optimizer
    backends
