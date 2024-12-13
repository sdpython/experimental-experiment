.. _l-design-pattern-optimizer:

=================
Pattern Optimizer
=================

The pattern optimizer is implemented by class :class:`GraphBuilderPatternOptimization
<experimental_experiment.xoptim.GraphBuilderPatternOptimization>`.
It searches for a specific sequence of nodes in the graph and
replaces it by another one without changing the inputs or the long_outputs
of the graph. The goal of the optimizer is to make the whole computation
graph more efficient. The goal of this implementation is to make this
optimization as fast as possible. 
Assuming the nodes in an onnx graph are ordered in a way every input of a
node was created by previous nodes, the optimizer must not require
any global reordering. The cost should be in :math:`O(N P I)` in the worst 
case where *N* is the number of nodes, *P* is the number of patterns,
*I* is the number of iterations.

It is difficult to foresee what a pattern needs in order to rewrite a part
of the graph. This API tries to give as much freedom as it can without
leaving too much to do to the developper which tries to add a new pattern.

Patterns
========

Patterns must inherit from :class:`PatternOptimization
<experimental_experiment.xoptim.PatternOptimization>`.
This class defines two methods.

PatternOptimization.match
+++++++++++++++++++++++++

::

    def match(
        self,
        g: "GraphBuilderPatternOptimization",
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:

* ``g`` is a :class:`GraphBuilderPatternOptimization
  <experimental_experiment.xoptim.GraphBuilderPatternOptimization>`,
  it holds all the existing nodes, is able to return any information
  about type, shape, the node before, the node after another one.
* ``node``: the matching must determine if some nodes around this one
  are part of set of nodes this pattern optimizer can rewrite.
  From there, the function explores wherever it needs,
  checking any condition it needs.
* ``matched``: usually unused, it returns of nodes already matching
  a pattern

The method must not modify the graph.
The method returns None if no match is found or an instance of class :class:`MatchResult
<experimental_experiment.xoptim.MatchResult>`. It must contain:

* a list of nodes involved in the rewriting. It does not mean all of them will be
  removed but all of them are needed to do the rewriting and must
  not be impacted by other pattern optimizer.
* A function doing the rewriting (usually method *apply* of the pattern class).
* An existing node where the rewritten nodes can be inserted.
  Knowing it makes it faster to rewriter. If not specified, the optimizer
  will automatically determine the position of the new nodes.

*Debugging: method none*

::

    def none(
        self,
        node: Optional[NodeProto] = None,
        lineno: Optional[int] = None,
        msg: str = "",
    ):

It may be useful which reason made a pattern matching fail.
Instead of returning None, method *match* can return the following
expression:

::

    return self.none(node, inspect.currentframe().f_lineno)

By setting the verbosity (see next Section), the user may then know
which lines in the code returned None and which condition failed.

PatternOptimization.apply
+++++++++++++++++++++++++

::

    @classmethod
    def apply(
        cls, g: "GraphBuilder", *nodes: Sequence[NodeProto]
    ) -> List[NodeProto]:

The method does the rewriting. It assumes it can happen.
It takes a list of nodes impacted by the rewriting. It assumes no other
pattern optimizer modified them or will modify them.
It receives the list of nodes
returned by method *apply*. Since it is a list of argument, method
*match* can include None values. The method returns the new nodes.
The optimizer considers that any node given to this function is removed
from the graph, and any node returned by it are added.
If a received node must be kept, it must be added to the list of returned node.

Optimization Algorithm
======================

It is implemented in method :meth:`optimize
<experimental_experiment.xoptim.GraphBuilderPatternOptimization.optimize>`

::

    def optimize(
        self, max_iter=-1, remove_identity: bool = True
    ) -> List[Dict[str, Any]]:


The algorithm runs multiple iteration until the graph is not evolving
or `max_iter` is reached. By default, it is equal to the number of nodes.
An iteration is:

::

    matches = []

    builds all successors and predecessors

    # Step 1: match

    for all patterns P:

        for all nodes n:

            r = p.match(n) 
            if r:
                if no node already scheduled to be rewritten by another match:
                    matches.append(r)
    
    # Step 2: apply

    for all matches r:
        apply the match r

    # Step 3: clean

    remove unused nodes
    remove identity nodes

This algorithm may apply more than one rewriting at each iteration
but it guarantees the local structure when applying the rewriting was
not altered by another one.

Adding a pattern
================

See :pr:`80` about the addition of a new pattern.

Example
=======

Simple API
++++++++++

We consider the following simple model:

.. runpython::
    :showcode:

    import torch
    from experimental_experiment.helpers import pretty_onnx
    from experimental_experiment.xbuilder import OptimizationOptions
    from experimental_experiment.torch_interpreter import to_onnx


    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(10, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 1),
            )

        def forward(self, x):
            return self.layers(x)


    x = torch.rand(3, 10)
    onx = to_onnx(
        MLP(), (x,), input_names=["x"], options=OptimizationOptions(patterns=None)
    )
    with open("temp_doc_mlp.onnx", "wb") as f:
        f.write(onx.SerializeToString())
    print(pretty_onnx(onx))

Which we can renders as follows:

.. gdot::
    :script: DOT-SECTION

    import onnx
    from onnx_array_api.plotting.dot_plot import to_dot

    onx = onnx.load("temp_doc_mlp.onnx")

    print("DOT-SECTION", to_dot(onx))

We then apply the optimizations by writing the following code:

.. runpython::
    :showcode:

    import onnx
    from experimental_experiment.helpers import pretty_onnx
    from experimental_experiment.xbuilder import GraphBuilder

    onx = onnx.load("temp_doc_mlp.onnx")

    # The model is placed in a GraphBuilder.
    # It creates dictionnaires to store shapes, ranks, types
    # to make it easier to the optimizers to find the information
    # they need. It still uses NodeProto to store nodes
    gr = GraphBuilder(onx, infer_shapes_options=True)

    # Let's optimize.
    opt_onx = gr.to_onnx(optimize=True)
    with open("temp_doc_mlp_opt.onnx", "wb") as f:
        f.write(opt_onx.SerializeToString())
    print(pretty_onnx(opt_onx))

Which renders as follows:

.. gdot::
    :script: DOT-SECTION

    import onnx
    from onnx_array_api.plotting.dot_plot import to_dot

    onx = onnx.load("temp_doc_mlp_opt.onnx")

    print("DOT-SECTION", to_dot(onx))

Verbosity
+++++++++

.. runpython::
    :showcode:

    import onnx
    from experimental_experiment.xbuilder import GraphBuilder

    onx = onnx.load("temp_doc_mlp.onnx")

    gr = GraphBuilder(onx, infer_shapes_options=True, verbose=1)
    opt_onx = gr.to_onnx(optimize=True)

With more verbosity:

.. runpython::
    :showcode:

    import onnx
    from experimental_experiment.xbuilder import GraphBuilder

    onx = onnx.load("temp_doc_mlp.onnx")

    gr = GraphBuilder(onx, infer_shapes_options=True, verbose=11)
    opt_onx = gr.to_onnx(optimize=True)

Select the pattern to use
+++++++++++++++++++++++++

Class :class:`OptimizationOptions <experimental_experiment.xbuilder.OptimizationOptions>`
is used to enable or disable patterns.

.. runpython::
    :showcode:

    import onnx
    from experimental_experiment.xbuilder import GraphBuilder, OptimizationOptions

    onx = onnx.load("temp_doc_mlp.onnx")

    gr = GraphBuilder(
        onx,
        infer_shapes_options=True,
        optimization_options=OptimizationOptions(
            patterns="TransposeTranspose,TransposeMatMul", verbose=1
        ),
    )
    opt_onx = gr.to_onnx(optimize=True)

There exists some predefined lists of patterns:

* ``default``: includes all patterns using only standard onnx patterns.
* ``onnxruntime``: patterns specific to :epkg:`onnxruntime`, the final model
  may be executed by onnxruntime and possibly only onnxruntime as it may
  introduce patterns from :epkg:`Supported Operators and Data Types`.

.. runpython::
    :showcode:

    import onnx
    from experimental_experiment.xbuilder import GraphBuilder, OptimizationOptions

    onx = onnx.load("temp_doc_mlp.onnx")

    gr = GraphBuilder(
        onx,
        infer_shapes_options=True,
        optimization_options=OptimizationOptions(
            patterns="default+onnxruntime", verbose=1
        ),
    )
    opt_onx = gr.to_onnx(optimize=True)

Statistics
++++++++++

This can be used to see when a pattern is applied and how long it takes.

.. runpython::
    :showcode:

    import pandas
    import onnx
    from experimental_experiment.xbuilder import GraphBuilder, OptimizationOptions

    onx = onnx.load("temp_doc_mlp.onnx")

    gr = GraphBuilder(
        onx,
        infer_shapes_options=True,
        optimization_options=OptimizationOptions(patterns="default"),
    )
    stat = gr.optimize()

    print(pandas.DataFrame(stat))

It can be aggregated:

.. runpython::
    :showcode:

    import pandas
    import onnx
    from experimental_experiment.xbuilder import GraphBuilder, OptimizationOptions

    onx = onnx.load("temp_doc_mlp.onnx")

    gr = GraphBuilder(
        onx,
        infer_shapes_options=True,
        optimization_options=OptimizationOptions(patterns="default"),
    )
    stat = gr.optimize()

    df = pandas.DataFrame(stat)
    for c in df.columns:
        if "time" not in c and "pattern" not in c:
            df[c] = df[c].fillna(0).astype(int)
    aggs = {
        "time_in": "sum",
        "added": "sum",
        "removed": "sum",
        "iteration": "max",
        "match_index": "max",
        "instances": "sum",
    }
    print(df.groupby("pattern").agg(aggs))

Shape inference
===============

The optimizers require to know the shapes to ensure they can rewrite
some nodes and avoid producing a model which does not return the
same results. If it is missing, some patterns cannot match for sure
and they will not match.

This information can be built by running shape inference
on the onnx models. That's what is done is the previous examples.
However, the best case is when this information comes from torch.

Function :func:`to_onnx <experimental_experiment.torch_interpreter.to_onnx>`
converts a torch model into ONNX. While doing so, it stores the shape
information coming from torch. There is no need to run shape inference
on the onnx model it generates before optimizing it.

Available Patterns and API
==========================

All patterns may be found at :ref:`l-pattern-optimization-onnx`
and :ref:`l-pattern-optimization-ort`.

When writing a pattern, walking along the graph or checking the shape
is very common. Class :class:`GraphBuilderPatternOptimization
<experimental_experiment.xoptim.GraphBuilderPatternOptimization>`
provides the following methods.

Opsets
++++++

Patterns must rewrite using the nodes of the opset defined in the model.

* :attr:`main_opset <experimental_experiment.xoptim.GraphBuilderPatternOptimization.main_opset>`: returns the opset

Shapes, Types
+++++++++++++

* :meth:`has_type <experimental_experiment.xoptim.GraphBuilderPatternOptimization.has_type>`: tells if a result type is known
* :meth:`get_type <experimental_experiment.xoptim.GraphBuilderPatternOptimization.get_type>`: returns a result type, fails if not known
* :meth:`has_shape <experimental_experiment.xoptim.GraphBuilderPatternOptimization.has_shape>`: tells if a result shape is known
* :meth:`get_shape <experimental_experiment.xoptim.GraphBuilderPatternOptimization.get_shape>`: returns a result shape, fails if not known
* :meth:`has_rank <experimental_experiment.xoptim.GraphBuilderPatternOptimization.has_rank>`: tells if a result rank is known
* :meth:`get_rank <experimental_experiment.xoptim.GraphBuilderPatternOptimization.get_rank>`: returns a result rank, fails if not known
* :meth:`try_infer_type <experimental_experiment.xoptim.GraphBuilderPatternOptimization.try_infer_type>`: returns a type if it can be guessed
* :meth:`try_infer_shape <experimental_experiment.xoptim.GraphBuilderPatternOptimization.try_infer_shape>`: returns a shape if it can be guessed

Constants
+++++++++

* :meth:`is_constant <experimental_experiment.xoptim.GraphBuilderPatternOptimization.is_constant>`:
  tells if a node is a constant (it may be a constant, an initializer or any value built on other constants)
* :meth:`is_constant_scalar <experimental_experiment.xoptim.GraphBuilderPatternOptimization.is_constant_scalar>`:
  checks a constant is a scalar and compares its value to a number
* :meth:`get_computed_constant <experimental_experiment.xoptim.GraphBuilderPatternOptimization.get_computed_constant>`:
  returns the constant, computes it is a constant built from other constants
* :meth:`get_attribute <experimental_experiment.xoptim.GraphBuilderPatternOptimization.get_attribute>`:
  returns an attribute of a node

Graph
+++++

* :meth:`next_node <experimental_experiment.xoptim.GraphBuilderPatternOptimization.next_node>`:
  returns the next node only if there is only one
* :meth:`next_nodes <experimental_experiment.xoptim.GraphBuilderPatternOptimization.next_nodes>`:
  returns the node consuming this result
* :meth:`node_before <experimental_experiment.xoptim.GraphBuilderPatternOptimization.node_before>`:
  returns the node producing the result
* :meth:`is_output <experimental_experiment.xoptim.GraphBuilderPatternOptimization.is_output>`:
  tells if a result is an output
* :meth:`is_used_by_subgraph <experimental_experiment.xoptim.GraphBuilderPatternOptimization.is_used_by_subgraph>`:
  tells if a result is used by a subgraph
* :meth:`is_used_more_than_once <experimental_experiment.xoptim.GraphBuilderPatternOptimization.is_used_more_than_once>`:
  tells if a result is used more than once
* :meth:`is_used_only_by <experimental_experiment.xoptim.GraphBuilderPatternOptimization.is_used_only_by>`:
  tells if a result is only used by specific nodes

Nodes
+++++

* :meth:`make_node <experimental_experiment.xoptim.GraphBuilderPatternOptimization.make_node>`:
  creates a node without adding it to the graph
* :meth:`make_node_check_opset <experimental_experiment.xoptim.GraphBuilderPatternOptimization.make_node_check_opset>`:
  creates a node without adding it to the graph, deals with some constraints
  related to opset version
