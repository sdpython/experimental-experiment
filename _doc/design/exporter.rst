===============
Custom Exporter
===============

The exporter implemented in this package is built upon a classic
architecture with two main classes:

* a :class:`GraphBuilder <experimental_experiment.xbuilder.GraphBuilder>`,
  it is a container for created nodes and initializers,
  it stores additional information such as shapes, types, constants,
  it providers methods to easily create nodes, provides unique names.
* a :class:`DynamoInterpreter
  <experimental_experiment.torch_interpreter.interpreter.DynamoInterpreter>`,
  this class goes through the model described as a :epkg:`GraphModule` and
  calls the appropriate converting functions to translate every call
  into an equivalent ONNX graph.

Both classes are usually not seen by the user. They are called either by
function :function:`to_onnx <experimental_experiment.torch_interpreter.to_onnx>`
which converts a model into an ONNX graph or through a custom backend:

* :func:`onnx_custom_backend <experimental_experiment.torch_dynamo.onnx_custom_backend>`,
  this backend leverages :epkg:`onnxruntime` to run the inference, it is fast,
* :func:`onnx_debug_backend <experimental_experiment.torch_dynamo.onnx_debug_backend>`,
  a backend using :epkg:`numpy`, meant to debug, it is slow.

This second backend calls the reference implementation through class
:class:`ExtendedReferenceEvaluator
<experimental_experiment.reference.ExtendedReferenceEvaluator>`.
This class extends :class:`ReferenceEvaluator <onnx.reference.ReferenceEvaluator>`
from package :epkg:`onnx.`

Exporter Logic
==============

