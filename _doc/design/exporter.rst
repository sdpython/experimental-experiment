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
function :func:`to_onnx <experimental_experiment.torch_interpreter.to_onnx>`
which converts a model into an ONNX graph or through a custom backend:

* :func:`onnx_custom_backend <experimental_experiment.torch_dynamo.onnx_custom_backend>`,
  this backend leverages :epkg:`onnxruntime` to run the inference, it is fast,
* :func:`onnx_debug_backend <experimental_experiment.torch_dynamo.onnx_debug_backend>`,
  a backend using :epkg:`numpy`, meant to debug, it is slow.

This second backend calls the reference implementation through class
:class:`ExtendedReferenceEvaluator
<experimental_experiment.reference.ExtendedReferenceEvaluator>`.
This class extends :class:`ReferenceEvaluator <onnx.reference.ReferenceEvaluator>`
from package :epkg:`onnx`.

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

    print(onnx_simple_text_plot(onx))

    print("Without any information, the known shapes are:")
    print(gr._known_shapes)

    print("Without any information, the known shapes are:")
    print(gr.constants_)

    print("The constant are not converted into TensorProto until the very end:")
    print(gr.initializers_dict)

The constant are only computed on demand. Their conversion to TensorProto
only happens when method
:meth:`to_onnx <experimental_experiment.xbuilder.GraphBuilder.to_onnx>`
is called.

Debugging
+++++++++

An exception is raised an error is detected and it displays the result
of :meth:`get_debug_msg <experimental_experiment.xbuilder.GraphBuilder.get_debug_msg>`.

.. runpython::
    :showcode:
    :exception:

    import numpy as np
    from onnx import TensorProto
    from experimental_experiment.xbuilder import GraphBuilder
    from experimental_experiment.reference import ExtendedReferenceEvaluator
    from onnx_array_api.plotting.text_plot import onnx_simple_text_plot


    gr = GraphBuilder(18, ir_version=9)
    gr.make_tensor_input("X", TensorProto.FLOAT, ("a", "b"), is_dimension=False)
    weight = gr.make_initializer("", np.array([[0.4, 0.5, 0.6]], dtype=np.float32).T)
    bias = gr.make_initializer("", np.array([[0.4, 0.5, 0.6]], dtype=np.float32))
    mm = gr.make_node("MatMul", ["X", weight], name="N1")
    out = gr.make_node("Add", [mm, "bias"], ["Y"], name="N2")
    gr.make_tensor_output(out, TensorProto.FLOAT, ("a",), indexed=False, is_dimension=False)
    onx = gr.to_onnx()

It shows the information currently available while building the model.
At the end the following lines appear.

::

    [GraphBuilder-EAQ.make_node] N1              [##:-  ] MatMul:['X', 'init1_s3x1_']->['_onx_matmul0']

It says one node named `N1` was created. `##` means the shape and type are
known for the two inputs it has. `-` means nothing is known for the output.
When the type is specified, it shows the following.

.. runpython::
    :showcode:
    :exception:

    import numpy as np
    from onnx import TensorProto
    from experimental_experiment.xbuilder import GraphBuilder
    from experimental_experiment.reference import ExtendedReferenceEvaluator
    from onnx_array_api.plotting.text_plot import onnx_simple_text_plot


    gr = GraphBuilder(18, ir_version=9)
    gr.make_tensor_input("X", TensorProto.FLOAT, ("a", "b"), is_dimension=False)
    weight = gr.make_initializer("", np.array([[0.4, 0.5, 0.6]], dtype=np.float32).T)
    bias = gr.make_initializer("", np.array([[0.4, 0.5, 0.6]], dtype=np.float32))
    mm = gr.make_node("MatMul", ["X", weight], name="N1")
    gr.set_type(mm, TensorProto.FLOAT)
    out = gr.make_node("Add", [mm, "bias"], ["Y"], name="N2")
    gr.make_tensor_output(out, TensorProto.FLOAT, ("a",), indexed=False, is_dimension=False)
    onx = gr.to_onnx()

It shows `U` when the type and rank are known, `#` if the type and shape are known.

::

    [GraphBuilder-MJG.make_node] N1              [##:U  ] MatMul:['X', 'init1_s3x1_']->['_onx_matmul0']

Simplified API
++++++++++++++

For the most common nodes, there exists a shortcut
to make the syntax shorter.

.. runpython::
    :showcode:

    import numpy as np
    from onnx import TensorProto
    from experimental_experiment.xbuilder import GraphBuilder
    from experimental_experiment.reference import ExtendedReferenceEvaluator
    from onnx_array_api.plotting.text_plot import onnx_simple_text_plot


    gr = GraphBuilder(18, ir_version=9)
    gr.make_tensor_input("X", TensorProto.FLOAT, ("a", "b"), is_dimension=False)
    mm = gr.op.MatMul("X", np.array([[0.4, 0.5, 0.6]], dtype=np.float32).T)
    out = gr.op.Add(mm, np.array([0.4, 0.5, 0.6], dtype=np.float32), outputs=["Y"])
    gr.make_tensor_output(
        out, TensorProto.FLOAT, ("a",), indexed=False, is_dimension=False
    )
    onx = gr.to_onnx()

    ref = ExtendedReferenceEvaluator(onx)
    x = np.random.rand(5, 3).astype(np.float32)
    y = ref.run(None, {"X": x})[0]

    print(y)

    print(onnx_simple_text_plot(onx))

Optimizations
+++++++++++++

:class:`GraphBuilder <experimental_experiment.xbuilder.GraphBuilder>`
implements three basic optimizations algorithms not using patterns.
Except constant folding, they are called by default.

* :meth:`remove_unused <experimental_experiment.xbuilder.GraphBuilder.remove_unused>`:
  removes unused nodes 
* :meth:`remove_identity_nodes <experimental_experiment.xbuilder.GraphBuilder.remove_identity_nodes>`:
  removes identity nodes
* :meth:`constant_folding <experimental_experiment.xbuilder.GraphBuilder.constant_folding>`:
  replaces constant whenever it is possible and it makes sense

DynamoInterpreter
=================

Class :class:`DynamoInterpreter
<experimental_experiment.torch_interpreter.interpreter.DynamoInterpreter>`
walks through a graph module and selects the best translation
for every part. It is a sequence of calls to internal functions
called :epkg:`aten functions`. It looks like the following:

.. runpython::
    :showcode:

    import torch

    class Neuron(torch.nn.Module):
        def __init__(self, n_dims: int, n_targets: int):
            super(Neuron, self).__init__()
            self.linear = torch.nn.Linear(n_dims, n_targets)

        def forward(self, x):
            return torch.sigmoid(self.linear(x))


    x = torch.rand(5, 3)
    model = Neuron(3, 1)
    graph = torch.export.export(model, (x,))
    print(graph)    

The called function such as `torch.ops.aten.addmm.default` are well
identified and those cannot be converted into ONNX.
The interpret just maps this string to a function creating
the onnx implementation :func:`aten_addmm
<experimental_experiment.torch_interpreter._aten_functions.aten_addmm>`
inside a dispatcher
:meth:`run_node <experimental_experiment.torch_interpreter.interpreter.DynamoInterpreter.run_node>`
which includes the following piece of code:

::

    if node.op == "placeholder":
        return self.placeholder(node)
    if node.op == "call_function":
        return self.call_function(node)
    if node.op == "output":
        return self.output(node)
    if node.op == "call_module":
        return self.call_module(node)
    if node.op == "get_attr":
        return self.get_attr(node)
    if node.op == "call_method":
        return self.call_method(node)

A converting function
+++++++++++++++++++++

Let's consider the easy converting following function.

::

    def aten_addmm(
        g: GraphBuilder,
        sts: bool,
        outputs: List[str],

        a: T,
        b: T,
        c: T,
        beta: float = 1.0,
        alpha: float = 1.0,
    ) -> T:
        "gemm"
        res = g.op.Gemm(
            b, c, a, alpha=float(alpha), beta=float(beta), outputs=outputs, name="addmm"
        )
        if sts:
            g.set_type(res, g.get_type(b))
            g.set_rank(res, 2)
        return res


The three first arguments are the
:class:`GraphBuilder <experimental_experiment.xbuilder.GraphBuilder>`,
a boolean asking the function to set the shape and rank,
the output names to make sure the name are the same than the one in the graph
provided by torch. It helps debugging.

Shapes And Types
++++++++++++++++

The function can assume the type is always filled.
The shapes should be set but in this case, only the rank is provided.
It is not mandatory but it helps the following functions to take
the right decision. 
:class:`GraphBuilder <experimental_experiment.xbuilder.GraphBuilder>`
is setting the type and shape for a limited number of operator type
such as `Identity`. It should be better in the next versions.
Some helpers were already implemented to set shape or types
as shown in this function.

::

    def aten_asin(g: GraphBuilder, sts: bool, outputs: List[str], x: T) -> T:
        "asin"
        res = g.make_node("Asin", [x], outputs)
        if sts:
            set_type_shape_unary_op(g, outputs[0], x)
        return res  

The boolean `sts` is False when the graph given by torch contains
information about shape and type. In that case, the interpreter will
give them to the graph builder.

Different Implementations
+++++++++++++++++++++++++

In the following case, the function adds a node `Identity`
or `CastLike` depending on the types. `CastLike` is only needed when types
are different. And the graph builder will remove the Identity node.

::

    def aten_copy(
        g: GraphBuilder,
        sts: bool,
        outputs: List[str],
        x: T,
        src: T,
        non_blocking: bool = False,
    ) -> T:
        "identity"
        assert not non_blocking, "copy implemented when non_blocking is True"
        if g.get_type(x) == g.get_type(src):
            return g.op.Identity(src, name="copy")
        return g.op.CastLike(src, x, name="copy")

Conventions
+++++++++++

The node should be given name based on the aten functions they
are part of. Doing that helps the developper to find where a failing
node comes from.

Functions
+++++++++

All the avaialable functions are listed in one the those three pages:

* :ref:`l-aten-functions`: functions

* :ref:`l-aten-methods`: methods

* :ref:`l-aten-prims`: primitives

Every function added to these modules is automatically added
to the list of known converter functions.

Example
+++++++

.. runpython::
    :showcode:

    import torch
    from onnx_array_api.plotting.text_plot import onnx_simple_text_plot
    from experimental_experiment.torch_interpreter import to_onnx

    class Neuron(torch.nn.Module):
        def __init__(self, n_dims: int, n_targets: int):
            super(Neuron, self).__init__()
            self.linear = torch.nn.Linear(n_dims, n_targets)

        def forward(self, x):
            return torch.sigmoid(self.linear(x))


    model = Neuron(3, 1)

    x = torch.rand(5, 3)

    onx = to_onnx(model, (x,), input_names=["x"])

    print(onnx_simple_text_plot(onx))

And visually:

.. gdot::
    :script: DOT-SECTION
    :process:

    import torch
    from onnx_array_api.plotting.dot_plot import to_dot
    from experimental_experiment.torch_interpreter import to_onnx

    class Neuron(torch.nn.Module):
        def __init__(self, n_dims: int, n_targets: int):
            super(Neuron, self).__init__()
            self.linear = torch.nn.Linear(n_dims, n_targets)

        def forward(self, x):
            return torch.sigmoid(self.linear(x))


    x = torch.rand(5, 3)
    model = Neuron(3, 1)
    onx = to_onnx(model, (x,), input_names=["x"])
    print("DOT-SECTION", to_dot(onx))

Debugging
+++++++++

There is no fallback by default. The converter fails if
the conversion to ONNX cannot happen. In that case, it tries to
give you some information why it failed.
(The example might succeed in the future.)

.. runpython::
    :showcode:
    :exception:

    import torch
    from experimental_experiment.torch_interpreter import to_onnx

    class Neuron(torch.nn.Module):
        def __init__(self, n_dims: int, n_targets: int):
            super(Neuron, self).__init__()
            self.linear = torch.nn.Linear(n_dims, n_targets)

        def forward(self, x):
            return torch.celu(self.linear(x))


    x = torch.rand(5, 3)
    model = Neuron(3, 1)


    onx = to_onnx(model, (x,), input_names=["x"])    

In particular, the first line of the error message. This one tells you there is currently no known
conversion of function `aten.celu`. A function `aten_celu` must be added
to the file `experimental_experiment.torch_interpreter._aten_functions`.

::

    Unable to interpret function <class 'torch._ops.OpOverload'>: <OpOverload(op='aten.celu', overload='default')>,
    searched for ['aten::celu', 'celu_default'] and attributes ['__qualname__', '__name__'], args=(addmm,), kwargs={}

Below is the graph module:

::

    -- process.graph_module --
    graph():
        %arg0_1 : [num_users=1] = placeholder[target=arg0_1]
        %arg1_1 : [num_users=1] = placeholder[target=arg1_1]
        %arg2_1 : [num_users=1] = placeholder[target=arg2_1]
        %t : [num_users=1] = call_function[target=torch.ops.aten.t.default](args = (%arg0_1,), kwargs = {})
        %addmm : [num_users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%arg1_1, %arg2_1, %t), kwargs = {})
        %celu : [num_users=1] = call_function[target=torch.ops.aten.celu.default](args = (%addmm,), kwargs = {})
        return (celu,)
    -- process.progress --
    node 5/7 

The last line tells you, it stopped at line 5/7 which helps to find what functions were called
before. Next is the information of all nodes added so far.
We can see that except this function, everything looks good and all shapes
are known.

::

    [GraphBuilder-BQU.make_tensor_input] x[1:5x3]
    [GraphBuilder-BQU.make_initializer] arg0_1[torch.float32:torch.Size([1, 3]):[-0.44980645179748535, 0.29780903458595276, -0.32629191875457764]]
    [GraphBuilder-BQU.make_initializer] arg1_1[torch.float32:torch.Size([1]):[0.2905656397342682]]
    [GraphBuilder-BQU.make_node]                 [#:#   ] Identity:['x']->['arg2_1']
    [GraphBuilder-BQU.make_node] t               [#:#   ] Transpose:['arg0_1']->['t']
    [GraphBuilder-BQU.make_node] addmm           [###:# ] Gemm:['arg2_1', 't', 'arg1_1']->['addmm']

There is also this section starting with `--TORCH-SHAPE--`
which shows which shapes are given by torch.

::

    --TORCH-SHAPES--
    arg0_1: ('run_node', (('example_value', torch.float32, torch.Size([5, 1])), ('val', torch.float32, torch.Size([1, 3])))) --- 1:2:(1, 3):
    arg1_1: ('run_node', (('example_value', torch.float32, torch.Size([5, 1])), ('val', torch.float32, torch.Size([1])))) --- 1:1:(1,):
    arg2_1: ('run_node', ('', ('val', torch.float32, torch.Size([5, 3])))) --- 1:2:(5, 3):
    t: ('run_node', ('', ('val', torch.float32, torch.Size([3, 1])))) --- 1:2:(3, 1):
    addmm: ('run_node', ('', ('val', torch.float32, torch.Size([5, 1])))) --- 1:2:(5, 1):
    celu: ('run_node', ('', ('val', torch.float32, torch.Size([5, 1])))) --- :::

Dynamic Shapes
++++++++++++++

It just needs to be added when calling function
:func:`to_onnx <experimental_experiment.torch_interpreter.to_onnx>`:
``dynamic_shapes={"x": {0: torch.export.Dim("batch")}}``.

.. runpython::
    :showcode:

    import torch
    from onnx_array_api.plotting.text_plot import onnx_simple_text_plot
    from experimental_experiment.torch_interpreter import to_onnx

    class Neuron(torch.nn.Module):
        def __init__(self, n_dims: int, n_targets: int):
            super(Neuron, self).__init__()
            self.linear = torch.nn.Linear(n_dims, n_targets)

        def forward(self, x):
            return torch.sigmoid(self.linear(x))


    model = Neuron(3, 1)

    x = torch.rand(5, 3)

    onx = to_onnx(
        model,
        (x,),
        input_names=["x"],
        dynamic_shapes={"x": {0: torch.export.Dim("batch")}},
    )

    print(onnx_simple_text_plot(onx))
