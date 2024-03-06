=================
Pattern Optimizer
=================

The pattern optimizer is implemented by class :class:`GraphBuilderPatternOptimization
<experimental_experiment.xoptim.graph_builder_optim>`.
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

Patterns must inherit from class:`PatternOptimization
<experimental_experiment.xoptim.patterns.PatternOptimization>`.
This class defines two methods.

PatternOptimization.match
+++++++++++++++++++++++++

::

    def match(
            self,
            g: "GraphBuilderPatternOptimization",  # noqa: F821
            node: NodeProto,
            matched: List[MatchResult],
        ) -> Optional[MatchResult]:

* ``g`` is a :class:`GraphBuilderPatternOptimization
  <experimental_experiment.xoptim.graph_builder_optim>`,
  it holds all the existing nodes, is able to return any information
  about type, shape, the node before, the node after another one.
* ``node``: the matching must determine if some nodes around this one
  are part of set of nodes this pattern optmizer can rewrite.
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
        cls, g: "GraphBuilder", *nodes: Sequence[NodeProto]  # noqa: F821
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

    for all patterns P:

        for all nodes n:

            r = p.match(n) 
            if r:
                if no node already scheduled to be rewritten by another match:
                    matches.append(r)
    
        for all matches r:
            apply the match r

This algorithm may apply more than one rewriting at each iteration
but it guarantees the local structure when applying the rewriting was
not altered by another one.

Example
=======

    .. gdot::
        :script: DOT-SECTION

        from onnx_array_api.plotting.dot_plot import to_dot

        opset = onnx_opset_version() - 2
        X, y = make_regression(100, n_features=10, bias=2, random_state=0)
        X = X.astype(numpy.float32)
        y = y.astype(numpy.float32)
        w = (numpy.random.rand(y.shape[0]) + 1).astype(X.dtype)
        X_train, _, y_train, __, w_train, ___ = train_test_split(X, y, w)
        reg = LinearRegression()
        reg.fit(X_train, y_train, sample_weight=w_train)
        reg.coef_ = reg.coef_.reshape((1, -1))
        onx = to_onnx(reg, X_train, target_opset=opset, black_op={"LinearRegressor"})

        onx_loss = add_loss_output(
            onx, weight_name="weight", score_name="elastic", l1_weight=0.1, l2_weight=0.9
        )

        print("DOT-SECTION", to_dot(onx_loss))
