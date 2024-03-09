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

One objective: SPEED
====================

The only objective for this exporter is **speed**. It must fast as the
size of the model to convert grows fast. The exporter may be one piece
of the backend calling :epkg:`onnxruntime`. This only objective implies
a few constraints.

**multi-opset support**

The converter must support the conversion to different
`opset <https://onnx.ai/onnx/intro/concepts.html#what-is-an-opset-version>`_
to avoid using the :func:`onnx.version_converter.convert_version` which
does not fully work when the model includes other domain than the main one.

**use shape and type information**

The :epkg:`GraphModule` comes with the shape and type information
of the tensor it manipulates. It must be used to optimize
the onnx graph rather than using an optimizer after the conversion
happens.

**no decorators, no code interpretation**

Writing efficient code is easier when the code you is the code you get.
A decorator hides some logic a developper must take into account
to avoid writing non efficient code. On the same level, translating
a python code into ONNX requires extra logic the developper does not
control.

**no fallback**

The implementation fails if it cannot find a solution to convert
the model into ONNX. There are some ways to go around that but
there are not enabled by default. The user must know if the exporter
follows a different way to produce the model.

GraphBuilder
============

:class:`GraphBuilder <experimental_experiment.xbuilder.GraphBuilder>`
start from empty or take an existing graph as an input.
In that case, the builder is usually used by an optimizer.

Internal containers
+++++++++++++++++++

Beside the onnx structure, the builder holds information about
the requested opsets and the dynamic shapes.
During the conversion, it stores informations about

- `_unique_names`: names already taken for results
- `_unique_node_names`: names already taken for node node

- `_known_names`: existing names
- `_known_types`: known type for every result, it must exist
- `_known_shapes`: known shape for every result, either shape or rank is known
- `_known_ranks`: declared ranks
- `_known_value_shape`: results known as shapes, the implementation tries
  to capture the logic with string, :epkg:`sympy` could be used

The model stores some constant, the builder assumes every node
taking only constant as inputs produces a new constant.

- `constants_`: constant values
- `constants_computed_`: computed constant values, constant built from constant,
  every computed constant is cached,

The builder tries to minimize the number of intializers to create.
It stores a unique value for the small one:

- `_values`: cache initializer value to merge those which are equal

The forward/backward graphs may dynamic dimension as input.
Some results are reshaped based on this inputs.
The following container keep track of this information.

- `dynamic_objects`: list of dynamic dimensions coming as inputs
- `dynamic_objects_rev`: reverse dictionary to fasten lookups
- `_dynamic_alias`: used when the user gives a different
    name to the dynamic shapes

Next container store dynamic shapes.

- `_cache_shape`: cache concatenation of shapes

API
+++

The following methods are used to add onnx elements to the graph.

* :meth:`get_opset <experimental_experiment.xbuilder.GraphBuilder.get_opset>`:
  get the value for a domain
* :meth:`make_tensor_input <experimental_experiment.xbuilder.GraphBuilder.make_tensor_input>`:
  adds an input to the graph, `is_dimension` specifies if this input is a dynamic
  dimension, a single integer,
* :meth:`make_tensor_output <experimental_experiment.xbuilder.GraphBuilder.make_tensor_output>`:
  adds an output to the graph, `is_dimension` specifies if this output is a dynamic
  dimension, a single integer,
* :meth:`make_initializer <experimental_experiment.xbuilder.GraphBuilder.make_initializer>`:
  this method is used to add initializer to the graph,
* :meth:`make_node <experimental_experiment.xbuilder.GraphBuilder.make_node>`:
  add a node to the graph
* :meth:`to_onnx <experimental_experiment.xbuilder.GraphBuilder.to_onnx>`:
  produces the final ONNX

Some needs are very common and deserve a dedicated method.

* :meth:`make_nodes <experimental_experiment.xbuilder.GraphBuilder.make_nodes>`:
  adds many nodes in one row, it renames the intermediate result if needed.
* :meth:`from_array <experimental_experiment.xbuilder.GraphBuilder.from_array>`:
  converts a torch Tensor into a TensorProto,
* :meth:`get_attribute <experimental_experiment.xbuilder.GraphBuilder.get_attribute>`:
  retrieve an attribute from a NodeProto
* :meth:`make_shape_from_results <experimental_experiment.xbuilder.GraphBuilder.make_shape_from_results>`:
  makes a shape from a tuple having integer, string, or `torch.SymInt`

It is important to update the shape the information is available.

* :meth:`has_type <experimental_experiment.xbuilder.GraphBuilder.has_type>`
* :meth:`has_shape <experimental_experiment.xbuilder.GraphBuilder.has_shape>`
* :meth:`has_rank <experimental_experiment.xbuilder.GraphBuilder.has_rank>`
* :meth:`has_dynamic_object <experimental_experiment.xbuilder.GraphBuilder.has_dynamic_object>`
* :meth:`is_constant <experimental_experiment.xbuilder.GraphBuilder.is_constant>`
* :meth:`value_as_shape <experimental_experiment.xbuilder.GraphBuilder.value_as_shape>`

Get the information:

* :meth:`get_type <experimental_experiment.xbuilder.GraphBuilder.get_type>`
* :meth:`get_shape <experimental_experiment.xbuilder.GraphBuilder.get_shape>`
* :meth:`get_rank <experimental_experiment.xbuilder.GraphBuilder.get_rank>`
* :meth:`get_constant <experimental_experiment.xbuilder.GraphBuilder.get_constant>`
* :meth:`value_as_shape <experimental_experiment.xbuilder.GraphBuilder.value_as_shape>`

Set the information:

* :meth:`set_type <experimental_experiment.xbuilder.GraphBuilder.set_type>`
* :meth:`set_shape <experimental_experiment.xbuilder.GraphBuilder.set_shape>`
* :meth:`set_rank <experimental_experiment.xbuilder.GraphBuilder.set_rank>`
* :meth:`set_value_shape <experimental_experiment.xbuilder.GraphBuilder.set_value_shape>`

A function used to provide information to the user and calls in most of the error message:

* :meth:`get_debug_msg <experimental_experiment.xbuilder.GraphBuilder.get_debug_msg>`

::

  assert name in self._known_ranks, (
    f"Rank is unknown for result {name!r}, "
    f"known_shapes={self._known_ranks}{self.get_debug_msg()}"
  )

Example
+++++++

.. runpython::
    :showcode:

    import numpy as np
    from onnx import TensorProto
    from experimental_experiment.xbuilder import GraphBuilder
    from experimental_experiment.reference import ExtendedReferenceEvaluator
    from onnx_array_api.plotting.text_plot import onnx_simple_text_plot


    gr = GraphBuilder(18, ir_version=9)
    gr.make_tensor_input("X", TensorProto.FLOAT, ("a", "b"), is_dimension=False)
    weight = gr.make_initializer("", np.array([[0.4, 0.5, 0.6]], dtype=np.float32).T)
    bias = gr.make_initializer("", np.array([[0.4, 0.5, 0.6]], dtype=np.float32))
    mm = gr.make_node("MatMul", ["X", weight])
    out = gr.make_node("Add", [mm, bias], ["Y"])
    gr.make_tensor_output(out, TensorProto.FLOAT, ("a",), indexed=False, is_dimension=False)
    onx = gr.to_onnx()

    ref = ExtendedReferenceEvaluator(onx)
    x = np.random.rand(5, 3).astype(np.float32)
    y = ref.run(None, {"X": x})[0]

    print(y)
    print("----------")
    print(onnx_simple_text_plot(onx))


