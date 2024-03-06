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
<experimental_experiment.xoptim.patterns.PatternOptimization>`.
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
<experimental_experiment.xoptim.patterns.MatchResult>`. It must contain:

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
It takes a list of nodes impacted by the rewriting assumes no other
pattern optimizer will be modify them. It receives the list of nodes
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
    from onnx_array_api.plotting.text_plot import onnx_simple_text_plot
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
    print(onnx_simple_text_plot(onx))

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
    from onnx_array_api.plotting.text_plot import onnx_simple_text_plot
    from experimental_experiment.xbuilder import GraphBuilder

    onx = onnx.load("temp_doc_mlp.onnx")

    # The model is placed in a GraphBuilder.
    # It creates dictionnaires to store shapes, ranks, types
    # to make it easier to the optimizers to find the information
    # they need. It still uses NodeProto to store nodes
    gr = GraphBuilder(onx, infer_shapes=True)

    # Let's optimize.
    opt_onx = gr.to_onnx(optimize=True)
    with open("temp_doc_mlp_opt.onnx", "wb") as f:
        f.write(opt_onx.SerializeToString())
    print(onnx_simple_text_plot(opt_onx))

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
    from onnx_array_api.plotting.text_plot import onnx_simple_text_plot
    from experimental_experiment.xbuilder import GraphBuilder

    onx = onnx.load("temp_doc_mlp.onnx")

    gr = GraphBuilder(onx, infer_shapes=True, verbose=1)
    opt_onx = gr.to_onnx(optimize=True)

With more verbosity:

.. runpython::
    :showcode:

    import onnx
    from onnx_array_api.plotting.text_plot import onnx_simple_text_plot
    from experimental_experiment.xbuilder import GraphBuilder

    onx = onnx.load("temp_doc_mlp.onnx")

    gr = GraphBuilder(onx, infer_shapes=True, verbose=11)
    opt_onx = gr.to_onnx(optimize=True)

Select the pattern to use
+++++++++++++++++++++++++

Class :class:`OptimizationOptions <experimental_experiment.xbuilder.OptimizationOptions>`
is used to enable or disable patterns.

.. runpython::
    :showcode:

    import onnx
    from onnx_array_api.plotting.text_plot import onnx_simple_text_plot
    from experimental_experiment.xbuilder import GraphBuilder, OptimizationOptions

    onx = onnx.load("temp_doc_mlp.onnx")

    gr = GraphBuilder(
        onx,
        infer_shapes=True,
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
    from onnx_array_api.plotting.text_plot import onnx_simple_text_plot
    from experimental_experiment.xbuilder import GraphBuilder, OptimizationOptions

    onx = onnx.load("temp_doc_mlp.onnx")

    gr = GraphBuilder(
        onx,
        infer_shapes=True,
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
    from onnx_array_api.plotting.text_plot import onnx_simple_text_plot
    from experimental_experiment.xbuilder import GraphBuilder, OptimizationOptions

    onx = onnx.load("temp_doc_mlp.onnx")

    gr = GraphBuilder(
        onx,
        infer_shapes=True,
        optimization_options=OptimizationOptions(patterns="default"),
    )
    stat = gr.optimize()

    print(pandas.DataFrame(stat))

Shape inference
===============

The optimizers require to know the shape to ensure they can rewrite
some nodes and avoid producing a model which does not return the
same results. If it is missing, some patterns cannot match for sure.
They won't match.

This information can be built by running shape inference
on the onnx models. That's what is done is the previous examples.
However, the best case is when this information comes from torch.

Function :func:`to_onnx <experimental_experiment.torch_interpreter.to_onnx>`
converts a torch model into ONNX. While doing so, it stores the shape
information coming from torch. There is no need to run shape inference
on the onnx model it generates before optimizing it.

Available Patterns
==================

They may be found at :ref:`l-pattern-optimization-onnx`
and :ref:`l-pattern-optimization-ort`.
