import inspect
import os
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple
from onnx import NodeProto


class MatchResult:
    """
    Returns matching results.

    :param pattern: object detecting the pattern
    :param nodes: nodes to be replaced
    :param apply: node computing the replacements
    :param insert_at: insert the new nodes at this point if specified
    """

    def __init__(
        self,
        pattern: "PatternOptimization",
        nodes: List[NodeProto],
        apply: Callable,
        insert_at: Optional[NodeProto] = None,
    ):
        self.pattern = pattern
        self.nodes = nodes
        self.apply = apply
        self.insert_at = insert_at

    def to_string(self, short: bool = True) -> str:
        types = [n.op_type for n in self.nodes if n is not None]
        if short:
            return f"MatchResult: {self.pattern} replaces {types}"
        inputs = set()
        outputs = set()
        for node in self.nodes:
            if node is None:
                continue
            inputs |= set(node.input)
            outputs |= set(node.output)
        return (
            f"MatchResult: {self.pattern} replaces {types}, "
            f"inputs: {inputs}, outputs: {outputs}"
        )

    def __str__(self) -> str:
        return self.to_string(short=True)


class PatternOptimization:
    """
    Defines an optimization pattern.
    Function match should return None if the match does not happen
    or better ``self.none(node, inspect.currentframe().f_lineno)``.
    That allows the user to know which line rejected a specific pattern
    by setting environment variable ``LOG_PATTERN_OPTIMIZE=10``.

    :param verbose: determine the verbosity, this can be also dermine by setting up
        environment variable ``LOG_PATTERN_OPTIMIZE=10``
    """

    def __init__(self, verbose: int = 0):
        value = os.environ.get("LOG_PATTERN_OPTIMIZE", "0")
        self.verbose = max(verbose, int(value))

    def __str__(self) -> str:
        return self.__class__.__name__

    def __eq__(self, o: "PatternOptimization"):
        """
        Basic comparison based on the class name.
        """
        return type(o) == type(self)

    def enumerate_matches(
        self, g: "GraphBuilderPatternOptimization"  # noqa: F821
    ) -> Iterator:
        """
        Enumerates all the
        """
        matched = []
        for node in g.iter_nodes():
            res = self.match(g, node, matched)
            if res:
                matched.append(res)
                yield res

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        """
        Determines nodes around *node* which can be rewritten.

        :param g: is a :class:`GraphBuilderPatternOptimization
            <experimental_experiment.xoptim.GraphBuilderPatternOptimization>`,
            it holds all the existing nodes, is able to return any information
            about type, shape, the node before, the node after another one.
        :param node: the matching must determine if some nodes around this one
            are part of set of nodes this pattern optmizer can rewrite.
            From there, the function explores wherever it needs,
            checking any condition it needs.
        :param matched: usually unused, it returns of nodes already matching
            a pattern

        The method must not modify the graph.
        The method returns None if no match is found or an instance
        of class :class:`MatchResult
        <experimental_experiment.xoptim.patterns.MatchResult>`. It must contain:

        * a list of nodes involved in the rewriting. It does not mean all
          of them will be removed but all of them are needed to do the rewriting
          and must not be impacted by other pattern optimizer.
        * A function doing the rewriting (usually method *apply* of the pattern class).
        * An existing node where the rewritten nodes can be inserted.
          Knowing it makes it faster to rewriter. If not specified, the optimizer
          will automatically determine the position of the new nodes.
        """
        raise NotImplementedError(
            f"This function must be overloaded in class {self.__class__}."
        )

    def none(
        self,
        node: Optional[NodeProto] = None,
        lineno: Optional[int] = None,
        msg: str = "",
    ):
        """
        It may be useful which reason made a pattern matching fail.
        Instead of returning None, method *match* can return the following
        expression:

        ::

            return self.none(node, inspect.currentframe().f_lineno)

        By setting the verbosity (see next Section), the user may then know
        which lines in the code returned None and which condition failed.
        """
        if node and self.verbose:
            if self.verbose >= 10:
                print(
                    f"[{self.__class__.__name__}.match] NONE - line: {lineno}:"
                    f"{os.path.split(self.__class__.__module__)[-1]}, op_type={node.op_type}{msg}"
                )

    @classmethod
    def apply(
        cls, g: "GraphBuilder", *nodes: Sequence[NodeProto]  # noqa: F821
    ) -> List[NodeProto]:
        """
        The method does the rewriting. It assumes it can happen.
        It takes a list of nodes impacted by the rewriting assumes no other
        pattern optimizer will be modify them. It receives the list of nodes
        returned by method *apply*. Since it is a list of argument, method
        *match* can include None values. The method returns the new nodes.
        The optimizer considers that any node given to this function is removed
        from the graph, and any node returned by it are added.
        If a received node must be kept, it must be added to the list of returned node.

        :param nodes: nodes returned by method *match*, there are then removed
        :return: nodes to add to graph.
        """
        raise NotImplementedError(f"This function must be overloaded in class {cls}.")


class EasyPatternOptimization(PatternOptimization):
    """
    Implements a pattern optimization for quick experimentation.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose=verbose)
        self._cache = {}

    def match_pattern(
        self,
        g: "GraphBuilder",  # noqa: F821
        *args: List[str],
        **kwargs: Dict[str, Any],
    ):
        """
        Builds the pattern to match.
        """
        raise NotImplementedError(
            f"Class {self.__class__.__name__!r} must overwrite method match_pattern."
        )

    def _get_pattern(self, g):
        from .graph_builder_optim import GraphBuilderPatternOptimization

        kwargs = {}
        args = []

        # There should be a better way.
        sig = inspect.signature(self.match_pattern)
        for i, p in enumerate(sig.parameters.values()):
            if i == 0:
                continue
            if p.default is not inspect._empty:
                # an attribute
                kwargs[p.name] = p.default
            else:
                args.append(p.name)

        g2 = g.builder.empty_copy(as_function=True)
        for name in args:
            g2.make_tensor_input(name, 0, None, False)
        output = self.match_pattern(g2, *args, **kwargs)
        if isinstance(output, str):
            g2.make_tensor_output(output, 0, None, is_dimension=False)
        else:
            for name in output:
                g2.make_tensor_output(name, 0, None)
        pat = GraphBuilderPatternOptimization(g2, verbose=g.verbose)
        pat._build()
        return pat

    @classmethod
    def _match_nodes_io(cls, n1: NodeProto, n2: NodeProto) -> bool:
        if n1.op_type != n2.op_type:
            return False
        if len(n1.input) != len(n2.input):
            return False
        if len(n1.output) != len(n2.output):
            return False
        return True

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:

        # We cache the pattern.
        if id(g) in self._cache:
            pat = self._cache[id(g)]
        else:
            pat = self._get_pattern(g)
            self._cache[id(g)] = pat

        # Let's match the first node.
        # Then we need to match successors and predecessors.
        p_node = pat.nodes[-1]  # the last one
        if not self._match_nodes_io(node, p_node):
            return self.none()

        marked = {id(p_node): (p_node, node)}
        stacked = [id(p_node)]
        while stacked:
            idn = stacked.pop()
            n, pn = marked[idn]

            # predecessor
            # loop over input
            pred = g.node_before(n)
            ppred = pat.node_before(pn)
            if ppred is not None:
                if pred is None:
                    return self.none(node, inspect.currentframe().f_lineno)
                if pred.op_type != ppred.op_type:
                    return self.none(node, inspect.currentframe().f_lineno)
                # matching backward
                key = id(pred)
                if key not in marked:
                    marked[key] = ppred, pred
                    stacked.append(id(ppred))

            # successors
            ns = g.next_nodes(n)
            pns = pat.next_nodes(pn)
            if len(ns) != len(pns):
                return self.none(node, inspect.currentframe().f_lineno)

            # matching forward is more difficult, we first check the node operators
            if len(ns) != len(pns):
                return self.none(node, inspect.currentframe().f_lineno)

            s1 = set(_.op_type for _ in ns)
            s2 = set(_.op_type for _ in pns)
            if s1 != s2:
                return self.none(node, inspect.currentframe().f_lineno)

            if len(s2) == len(pns):
                if len(s1) != len(s2):
                    return self.none(node, inspect.currentframe().f_lineno)
                # unique node type, this is easy
                d1 = {_.op_type: _ for _ in ns}
                d2 = {_.op_type: _ for _ in pns}
                for k, v in d1.items():
                    vv = d2[k]  # it exists
                    key = id(vv)
                    if key not in marked:
                        marked[key] = (vv, v)
                        stacked.append(key)
            else:
                # otherwise this is less easy but the number of possibilities
                # is usually small
                # we need to check the marked operator first
                assert False, "not implemented yet"

    @classmethod
    def apply_pattern(cls, g: "GraphBuilder", *args, **kwargs):  # noqa: F821
        """
        Applies the replacements.
        """
        raise NotImplementedError(
            f"Class {cls.__name__!r} must overwrite method apply_pattern."
        )

    @classmethod
    def apply(
        cls,
        g: "GraphBuilder",  # noqa: F821
        pat: "GraphBuilderPatternOptimization",  # noqa: F821
        mapping: List[Tuple[NodeProto, NodeProto]],
        *nodes: Sequence[NodeProto],
    ) -> List[NodeProto]:
        raise NotImplementedError(
            f"Class {cls.__name__!r} must overwrite method apply_pattern."
        )
