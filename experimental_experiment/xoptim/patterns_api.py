import inspect
import os
import pprint
import textwrap
from collections import Counter
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union
import numpy as np
from onnx import AttributeProto, FunctionProto, ModelProto, NodeProto, TensorProto


def string_to_elem_type(name: str) -> int:
    """
    Converts a string into an element type.
    INT64 -> TensorProto.INT64
    """
    assert hasattr(TensorProto, name), f"Unable to interpret type {name!r}"
    return getattr(TensorProto, name)


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
        assert hasattr(
            pattern, "verbose"
        ), f"Class {type(pattern)} has not attribute 'verbose'"
        if pattern.verbose >= 10:
            print(
                f"[{self.__class__.__name__}.match] MATCH {pattern.__class__.__name__} "
                f"with {len(nodes)} nodes and types "
                f"{[n.op_type for n in nodes if n is not None]}"
            )

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

    def debug_string(self, g: Optional["GraphBuilder"] = None) -> str:  # noqa: F821
        """
        Returns a string showing the matched nodes.
        """

        def _p(i, g=g):
            if g.has_shape(i):
                return f"{i}:{g.get_type(i)}:{g.get_shape(i)}"
            return f"{i}:{g.get_type(i)}:R{g.get_rank(i)}"

        rows = []
        for ind, node in enumerate(self.nodes):
            if node is None:
                rows.append(f"{ind} -")
                continue
            rows.append(f"{ind} - {node.op_type}({node.input}) -> {node.output}")
        if g:
            rows.append("--------")
            for ind, node in enumerate(self.nodes):
                if node is None:
                    rows.append(f"{ind} -")
                    continue
                rows.append(
                    f"{ind} - {node.op_type}({', '.join(map(_p, node.input))}) "
                    f"-> {', '.join(map(_p, node.output))}"
                )
        return "\n".join(rows)


class PatternOptimization:
    """
    Defines an optimization pattern.
    Function match should return None if the match does not happen
    or better ``self.none(node, inspect.currentframe().f_lineno)``.
    That allows the user to know which line rejected a specific pattern
    by setting environment variable ``LOG_PATTERN_OPTIMIZE=10``.
    An environment variable equal to the class name can be set as well to
    track this specific pattern.

    :param verbose: determine the verbosity, this can be also dermine by setting up
        environment variable ``LOG_PATTERN_OPTIMIZE=10``
    :param priority: at each iteration,
        all patterns whose priority is below one threshold
        are executed, if none of them matches, the priority is increase
    :param min_opset: can be applied if main opset is > min_opset

    Example :ref:`l-plot-model-to-code` shows a way to find or build a skeleton
    for a pattern. If environment variable ``PATTERN`` is set to a specific pattern,
    verbosity is enable for this pattern only.
    """

    def __init__(self, verbose: int = 0, priority: int = 1, min_opset: int = 1):
        value = os.environ.get("LOG_PATTERN_OPTIMIZE", "0")
        self.verbose = max(verbose, int(value))
        value = os.environ.get(self.__class__.__name__, "0")
        self.verbose = max(self.verbose, int(value))
        self.priority = priority
        self.min_opset = min_opset
        pattern = os.environ.get("PATTERN", "")
        if pattern in (
            self.__class__.__name__,
            self.__class__.__name__.replace("Pattern", ""),
        ):
            self.verbose = max(self.verbose, 10)

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __eq__(self, o: "PatternOptimization"):
        """Basic comparison based on the class name."""
        return type(o) == type(self)  # noqa: E721

    def enumerate_matches(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
    ) -> Iterator:
        """Enumerates all the possible matches."""
        if self.verbose >= 10:
            print(
                f"[PatternOptimization.enumerate_matches] start {self.__class__.__name__} "
                f"with main_opset={g.main_opset} and min_opset={self.min_opset}"
            )
        if g.main_opset >= self.min_opset:
            matched = []
            # g.iter_nodes() iterates on g.builder.nodes: ->
            #   too slow to have a secondary iterator
            for node in g.builder.nodes:
                # This expression seems awkard but it saves 10% just by looking into
                # the first item of the list and then, if necessary, walking through the
                # rest of the outputs.
                if g.is_used(node.output[0]) or any(g.is_used(o) for o in node.output[1:]):
                    # We avoid processing a node which is not used.
                    res = self.match(g, node, matched)
                    if res:
                        matched.append(res)
                        yield res
                elif self.verbose >= 10:
                    print(
                        f"[PatternOptimization.enumerate_matches] result no output "
                        f"in {','.join(node.output)} is used "
                        f"{[g.is_used(o) for o in node.output]}"
                    )

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
        <experimental_experiment.xoptim.MatchResult>`. It must contain:

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

    def _debug_print(self) -> str:
        return ""

    def none(
        self,
        node: Optional[NodeProto] = None,
        lineno: Optional[int] = None,
        msg: Optional[Union[Callable, str]] = None,
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
            if msg is None:
                msg = ""
            elif callable(msg):
                msg = msg()
            if msg:
                msg = f"\n{msg}"
            if self.verbose >= 10 and hasattr(self, "_debug"):
                msg2 = self._debug_print()
                if msg2:
                    msg2 = f"\n{textwrap.indent(msg2, '    ')}"
                print(
                    f"[{self.__class__.__name__}.match] NONE - line: {lineno}:"
                    f"{os.path.split(self.__class__.__module__)[-1]}, "
                    f"op_type={node.op_type}, name={node.name}{msg}{msg2}"
                )
            elif self.verbose >= 9:
                print(
                    f"[{self.__class__.__name__}.match] NONE - line: {lineno}:"
                    f"{os.path.split(self.__class__.__module__)[-1]}, "
                    f"op_type={node.op_type}, name={node.name}, "
                    f"inputs={','.join(node.input)}{msg}"
                )

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        *nodes: Sequence[NodeProto],
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
        raise NotImplementedError(
            f"This function must be overloaded in class {self.__class__.__name__!r}."
        )


class EasyPatternOptimization(PatternOptimization):
    """
    Implements a pattern optimization for quick experimentation.
    The current implementation does not match on domain name.
    It does not compares attributes either.
    The environment variable ``AMBIGUITIES=1`` can be set to one to
    raise an exception when this case happens.
    """

    def __init__(self, verbose: int = 0, priority: int = 0, min_opset: int = 1):
        super().__init__(verbose=verbose, priority=priority, min_opset=min_opset)
        self._cache = {}
        self._validate_parameters = {}
        self._debug_ambiguities = int(os.environ.get("AMBIGUITIES", 0)) == 1

    def add_validate_param(self, key: str, value: Any):
        """
        Stores a value to retrieve when apply_pattern is called.
        """
        self._validate_parameters[key] = value

    def get_validate_param(self, key: str) -> Any:
        assert (
            key in self._validate_parameters
        ), f"Unable to find key {key!r} in {sorted(self._validate_parameters)}"
        return self._validate_parameters[key]

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

    def _build_pattern(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        fct: Callable,
    ) -> "GraphBuilderPatternOptimization":  # noqa: F821
        from .graph_builder_optim import GraphBuilderPatternOptimization

        kwargs = {}
        args = []

        # There should be a better way.
        sig = inspect.signature(fct)
        anns = []
        for i, p in enumerate(sig.parameters.values()):
            if i == 0:
                continue
            if p.default is not inspect._empty:
                # an attribute
                kwargs[p.name] = p.default
            else:
                args.append(p.name)
                anns.append(p.annotation)

        assert len(kwargs) == 0, f"Attributes are not supported yet but kwargs={kwargs}"

        g2 = g.builder.empty_copy(as_function=True, constant_size=2**30, _shapable=False)
        for name, ann in zip(args, anns):
            if ann is None or ann is str or ann is inspect._empty:
                g2.make_tensor_input(name, 0, None, False, marker=f"_build_pattern1_{name}")
                # Type is unknown
                g2.set_type(name, -1)
                continue
            assert isinstance(
                ann, str
            ), f"Annotation for {name!r} must be a string or None but ann={ann!r}"
            itype = string_to_elem_type(ann)
            g2.make_tensor_input(name, itype, None, False, marker=f"_build_pattern2_{name}")

        output = fct(g2, *args, **kwargs)
        if isinstance(output, str):
            g2.make_tensor_output(output, 0, None, is_dimension=False)
        else:
            for name in output:
                g2.make_tensor_output(name, 0, None, is_dimension=False)
        pat = GraphBuilderPatternOptimization(
            g2, verbose=max(0, g.verbose - 1), processor=g.processor
        )
        pat._build()
        return pat

    def _get_match_pattern(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
    ) -> "GraphBuilderPatternOptimization":  # noqa: F821
        cache_key = 0, tuple(sorted(g.opsets.items()))
        if cache_key in self._cache:
            return self._cache[cache_key]

        pat = self._build_pattern(g, self.match_pattern)
        self._cache[cache_key] = pat
        return pat

    def _get_apply_pattern(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
    ) -> "GraphBuilderPatternOptimization":  # noqa: F821
        cache_key = 1, tuple(sorted(g.opsets.items()))
        if cache_key in self._cache:
            return self._cache[cache_key]

        pat = self._build_pattern(g, self.apply_pattern)
        self._cache[cache_key] = pat
        return pat

    def display_pattern(self, g, fct) -> str:
        """
        Shows the pattern to match or to apply.
        """
        pat = self._build_pattern(g, fct)
        rows = []
        rows.append(
            f"{fct.__name__}({', '.join(pat.input_names)}) -> {', '.join(pat.output_names)}"
        )
        for node in pat.nodes:
            rows.append(f"{node.op_type}({', '.join(node.input)}) -> {', '.join(node.output)}")
        return "\n".join(rows)

    def _match_backward(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        pat: "GraphBuilderPatternOptimization",  # noqa: F821
        marked: Dict[int, Tuple[NodeProto, NodeProto]],
        pair_results_names: Dict[str, str],
        stacked: List[int],
        n: NodeProto,
        pn: NodeProto,
    ) -> Optional[int]:
        """
        Matches backward.

        :param g: graph
        :param node: root node (the node the matched began with,
            used only for debugging)
        :param pat: pattern
        :param marked: nodes of the pattern marked as already matched
        :param stacked: next node to look into
        :param n: node coming from the graph
        :param ns: node coming from the pattern
        :return: number of matched nodes, None or False to indicate a failed match
        """
        if self.verbose >= 10:
            print(
                f"[EasyPatternOptimization._match_backward] starts with "
                f"pair_results_names={pair_results_names}"
            )
        res = 0

        # predecessors
        if len(n.input) != len(pn.input):
            # not the same number of inputs
            self._hint(
                "BACKWARD: not the same number of inputs",
                "-- pattern",
                pn,
                "-- model",
                n,
            )
            return self.none(node, inspect.currentframe().f_lineno)

        pattern_input_names = set(pat.input_names)
        for nr, pnr in zip(n.input, pn.input):
            if (
                pnr not in pattern_input_names
                and not g.is_constant(nr)
                and len(g.next_nodes(nr)) != len(pat.next_nodes(pnr))
            ):
                self._hint(
                    "BACKWARD: one input is used outside the pattern",
                    "-- pattern input and pattern node",
                    pnr,
                    pn,
                    "-- model input and model node",
                    nr,
                    n,
                    "-- len(pat.next_nodes(pnr))",
                    len(pat.next_nodes(pnr)),
                    *pat.next_nodes(pnr),
                    type(pn),
                    "-- len(g.next_nodes(nr)))",
                    len(g.next_nodes(nr)),
                    *g.next_nodes(nr),
                    type(n),
                )
                return self.none(node, inspect.currentframe().f_lineno)

        for i, pi in zip(n.input, pn.input):
            ppred = pat.node_before(pi)
            if ppred is None:
                # ppred is None means the pattern ends here.
                continue
            pred = g.node_before(i)
            if pred is None:
                # No node in the graph.
                self._hint(
                    "BACKWARD: no node in the graph",
                    "-- pred",
                    pred,
                    "-- ppred",
                    ppred,
                )
                return self.none(node, inspect.currentframe().f_lineno)
            if pred.op_type != ppred.op_type or len(pred.input) != len(ppred.input):
                # Distinct type
                self._hint(
                    "BACKWARD: distinct types or distinct number of inputs",
                    "-- pred",
                    pred,
                    "-- ppred",
                    ppred,
                )
                return self.none(node, inspect.currentframe().f_lineno)

            # matching backward
            key = id(ppred)
            if key not in marked:
                # check for ambiguities
                amb = self._has_ambiguities(pair_results_names, pred, ppred)
                if amb:
                    self._hint(
                        "BACKWARD: ambiguities with names",
                        "-- ambiguities",
                        pred,
                        ppred,
                        "-- pairs",
                        pair_results_names,
                        "-- pattern",
                        self._pattern_to_string(g),
                    )
                    return self.none(node, inspect.currentframe().f_lineno)

                self._update_ambiguities(pair_results_names, pred, ppred)

                marked[key] = pred, ppred
                stacked.append(key)
                res += 1
        if self.verbose > 5 and res > 0:
            print(f"[EasyPatternOptimization._match_backward] add {res} nodes")
        return res

    def _match_forward(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        pat: "GraphBuilderPatternOptimization",  # noqa: F821
        marked: Dict[int, Tuple[NodeProto, NodeProto]],
        pair_results_names: Dict[str, str],
        stacked: List[int],
        n: Union[NodeProto, str],
        pn: Union[NodeProto, str],
    ) -> Optional[int]:
        """
        Matches forward.

        :param g: graph
        :param node: root node (the node the matched began with,
            used only for debugging),
        :param pat: pattern
        :param marked: nodes of the pattern marked as already matched
        :param stacked: next node to look into
        :param n: node coming from the graph,
            it can be a string to start from a result
        :param ns: node coming from the pattern,
            it can be a string to start from a result
        :return: number of matched nodes to continue,
            None or False to indicate a failed match
        """
        if self.verbose >= 10:
            print(
                f"[EasyPatternOptimization._match_forward] starts with "
                f"pair_results_names={pair_results_names}"
            )
        res = 0

        # successors
        if isinstance(n, NodeProto) and isinstance(pn, NodeProto):
            if len(n.output) != len(pn.output):
                # not the same number of outputs
                self._hint(
                    "FORWARD: not the same number of outputs",
                    "-- pattern",
                    pn,
                    "-- model",
                    n,
                )
                return self.none(node, inspect.currentframe().f_lineno)
            matched_results = list(zip(n.output, pn.output))
        elif isinstance(n, str) and isinstance(pn, str):
            matched_results = [(n, pn)]
        else:
            raise AssertionError(f"Unexpected types for n: {type(n)} and pn: {type(pn)}.")

        for o, op in matched_results:
            ns = g.next_nodes(o)
            pns = pat.next_nodes(op)
            if len(pns) == 0:
                # The pattern has no node forward, the matching stops.
                continue
            if len(ns) < len(pns):
                # Not enough nodes in the graph to match the pattern,
                # the result is known.
                self._hint(
                    "FORWARD: not enough nodes in the graph to match the pattern",
                    "-- o",
                    o,
                    "-- po",
                    op,
                    "-- len(ns)",
                    len(ns),
                    "-- len(pns)",
                    len(pns),
                )
                return self.none(node, inspect.currentframe().f_lineno)

            # Here comes the fun part, there is the same number of successors or more
            # nodes in the graph to match with the pattern.
            # And we have to handle the nodes already marked as found.
            # Hopefully, there is only one option.

            if len(ns) == len(pns) == 1:
                # Let's deal with the simple case
                if ns[0].op_type != pns[0].op_type or len(ns[0].input) != len(pns[0].input):
                    self._hint(
                        "FORWARD: distinct types or distinct number of inputs",
                        "-- pred",
                        ns[0],
                        "-- ppred",
                        pns[0],
                    )
                    return self.none(node, inspect.currentframe().f_lineno)
                amb = self._has_ambiguities(pair_results_names, ns[0], pns[0])
                if amb:
                    self._hint(
                        "BACKWARD: ambiguities with names",
                        "-- ambiguities",
                        ns[0],
                        pns[0],
                        "-- pairs",
                        pair_results_names,
                        "-- pattern",
                        self._pattern_to_string(g),
                    )
                    if self._debug_ambiguities:
                        raise AssertionError(
                            f"An ambiguities was detected, ns[0]="
                            f"{g.builder.pretty_node(ns[0], short=True)}, "
                            f"pns[0]={g.builder.pretty_node(pns[0], short=True)},\n"
                            f"pairs={pprint.pformat(pair_results_names)}\n-- pattern -- \n"
                            f"{self._pattern_to_string(g)}\n-- graph --\n"
                            f"{g.builder.pretty_text()}"
                        )
                    return self.none(node, inspect.currentframe().f_lineno)

                key = id(pns[0])
                if key not in marked:
                    marked[key] = ns[0], pns[0]
                    self._update_ambiguities(pair_results_names, ns[0], pns[0])
                    stacked.append(key)
                    res += 1
                continue

            # Let's remove the nodes already marked.
            p_marked = [_ for _ in pns if id(_) not in marked]
            id_marked = [id(marked[id(_)][0]) for _ in pns if id(_) in marked]
            assert len(id_marked) + len(p_marked) == len(pns), (
                f"Unexpected, id_marked={id_marked}, "
                f"id_p_marked={set(map(id, p_marked))}, "
                f"pns_ids={set(map(id, pns))}, "
                f"ns_ids={set(map(id, ns))}, o={o!r}, op={op!r}, "
                f"n.op_type={n.op_type!r}, "
                f"n.output={n.output}, np.output={pn.output}, "
                f"ns_types={set(_.op_type for _ in ns)}, "
                f"pns_types={set(_.op_type for _ in pns)}"
            )
            free = [_ for _ in ns if id(_) not in id_marked]
            if len(p_marked) == 0:
                # Everything is already marked.
                continue
            if len(free) < len(p_marked):
                # Not enough successors to match the remaining patterns.
                return self.none(node, inspect.currentframe().f_lineno)
            if len(p_marked) == len(free) == 1:
                # Only one option again.
                if p_marked[0].op_type != free[0].op_type or len(p_marked[0].input) != len(
                    free[0].input
                ):
                    self._hint(
                        "FORWARD: distinct types or distinct number of inputs",
                        "-- pred",
                        p_marked[0],
                        "-- ppred",
                        free[0],
                    )
                    return self.none(node, inspect.currentframe().f_lineno)
                amb = self._has_ambiguities(pair_results_names, free[0], p_marked[0])
                if amb:
                    self._hint(
                        "FORWARD: ambiguities with names",
                        "-- ambiguities",
                        free[0],
                        p_marked[0],
                        "-- pairs",
                        pair_results_names,
                    )
                    if self._debug_ambiguities:
                        raise AssertionError(
                            f"An ambiguities was detected, free[0]="
                            f"{g.builder.pretty_node(free[0], short=True)}, "
                            f"p_marked[0]={g.builder.pretty_node(p_marked[0], short=True)}, "
                            f"pairs={pprint.pformat(pair_results_names)}\n-- pattern -- \n"
                            f"{self._pattern_to_string(g)}\n-- graph --\n"
                            f"{g.builder.pretty_text()}"
                        )
                    return self.none(node, inspect.currentframe().f_lineno)

                key = id(p_marked[0])
                if key not in marked:
                    marked[key] = free[0], p_marked[0]
                    self._update_ambiguities(
                        pair_results_names,
                        free[0],
                        p_marked[0],
                        debug_msg=lambda: textwrap.indent(
                            self.display_pattern(g, self.match_pattern), "    "
                        ),
                    )
                    stacked.append(key)
                    res += 1
                continue

            # And now another fun part, let's try to handle the case when there
            # is only one option, matching on node type only returns one option.
            expected_op_type = [_.op_type for _ in p_marked]

            ec = Counter(expected_op_type)
            gc = Counter(_.op_type for _ in free)
            if len(ec) != len(gc) or set(ec) != set(gc):
                # number of unique operator types is different.
                self._hint(
                    "FORWARD: number of unique operator types is different",
                    "-- pattern",
                    ec,
                    pn,
                    "-- model",
                    gc,
                    n,
                    "-- model-marked",
                    id_marked,
                )
                return self.none(node, inspect.currentframe().f_lineno)
            for k, v in ec.items():
                if gc[k] < v:
                    # Not enough types to match.
                    return self.none(node, inspect.currentframe().f_lineno)

            # At this stage, we know matching the types is possible.
            # We first mark whatever is possible.
            ptype_to_node = {_.op_type: _ for _ in p_marked}
            gtype_to_node = {_.op_type: _ for _ in free}
            missing = []
            for k, v in ec.items():
                if gc[k] == v == 1:
                    key = id(ptype_to_node[k])
                    amb = self._has_ambiguities(
                        pair_results_names, gtype_to_node[k], ptype_to_node[k]
                    )
                    if not amb and key not in marked:
                        self._update_ambiguities(
                            pair_results_names, gtype_to_node[k], ptype_to_node[k]
                        )
                        marked[key] = gtype_to_node[k], ptype_to_node[k]
                        stacked.append(key)
                        res += 1
                else:
                    missing.append(k)

            if not missing:
                continue

            # At this stage, there are mutiple options for matching. We can:
            # 1. make assumptions and continue
            # 2. mark the node as incomplete matching, we could end up stuck anyway.
            assert True, (
                f"There are more than one option, this will be implemented later, "
                f"ec={ec}, gc={gc}"
            )
        if self.verbose > 5 and res > 0:
            print(f"[EasyPatternOptimization._match_forward] add {res} nodes")
        return res

    def _debug_print(self) -> str:
        if not hasattr(self, "_debug"):
            return ""

        def _s(s):
            if len(s) <= 30:
                return s
            return f"{s[:15]}...{s[-15:]}"

        def _p(n, full=False):
            if isinstance(n, NodeProto):
                if full:
                    return (
                        f"{n.op_type}({', '.join(map(_s, n.input))}) "
                        f"-> ({', '.join(map(_s, n.output))})"
                    )
                return f"{n.op_type}({','.join(map(_s, n.input))})"
            return str(n)

        rows = []
        for k, v in sorted(self._debug.items()):
            if k == "stacked":
                rows.append(f"len({k})={len(v)}:{v}")
                continue
            if k == "iteration":
                rows.append(f"{k}={v}")
                continue
            if k == "marked":
                rows.append(f"--marked-- #{len(v)}")
                for i, tu in v.items():
                    rows.append(f"  {_p(tu[0])} ~ {_p(tu[1])} [{id(tu[0])}-{i}]")
                continue
            if k == "hint":
                rows.append(f"--hint--: {v[0]}")
                for i in v[1:]:
                    rows.append("  " + _p(i, full=True))
                continue
            if k in {"node", "pattern", "pattern_node", "pattern_nodes"}:
                continue
            rows.append(f"-- not shown {k}")

        return "\n".join(rows)

    def _hint(self, *args: Sequence[Any]):
        """
        Add debugging information to help users.
        """
        if self.verbose >= 5:
            self._debug["hint"] = args

    def validate_mapping(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        deleted_nodes: List[NodeProto],
        pattern_nodes: Optional[List[NodeProto]] = None,
    ) -> bool:
        """
        Validates the mapping.

        :param g: GraphBuilder
        :param deleted_nodes: matched nodes from the model (to be deleted)
        :param pattern_nodes: matched nodes coming from the pattern
        :return: validate the mapping or not, default is True
        """
        return True

    def validate_attribute_mapping(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        deleted_nodes: List[NodeProto],
        pattern_nodes: Optional[List[NodeProto]] = None,
    ) -> bool:
        """
        Validates the mapping of the attributes

        :param g: GraphBuilder
        :param deleted_nodes: matched nodes from the model (to be deleted)
        :param pattern_nodes: matched nodes coming from the pattern
        :return: validate the mapping or not, default is True
        """
        assert len(deleted_nodes) == len(pattern_nodes), (
            f"Mismatched number of nodes len(deleted_nodes)={len(deleted_nodes)}, "
            f"len(pattern_nodes)={len(pattern_nodes)}"
        )
        for i, (node, pat_node) in enumerate(zip(deleted_nodes, pattern_nodes)):
            assert node.op_type == pat_node.op_type or node.domain != pat_node.domain, (
                f"Node type mismatch at position {i}, {node.op_type!r} != "
                f"{pat_node.op_type!r} or {node.domain!r} != {pat_node.domain!r}"
            )
            in_graph = {att.name: att for att in node.attribute}
            for att in pat_node.attribute:
                if att.name not in in_graph:
                    if self.verbose >= 5:
                        print(
                            f"[EasyPatternOptimization.validate_attribute_mapping] failed "
                            f"attribute {att.name!r} (missing), nodes: "
                            f"{g.builder.pretty_node(node, short=True)} / "
                            f"{g.builder.pretty_node(pat_node, short=True)}"
                        )
                    return False
                n_att = in_graph[att.name]
                if att.type != n_att.type:
                    if self.verbose >= 5:
                        print(
                            f"[EasyPatternOptimization.validate_attribute_mapping] failed "
                            f"attribute {att.name!r} (type), "
                            f"nodes: {g.builder.pretty_node(node, short=True)} / "
                            f"{g.builder.pretty_node(pat_node, short=True)}"
                        )
                    return False
                if att.type == AttributeProto.INT and att.i != n_att.i:
                    if (
                        att.name == "axis"
                        and node.op_type in {"Split", "Concat"}
                        and g.has_rank(node.input[0])
                    ):
                        # Let's compare negative value.
                        rk = g.get_rank(node.input[0])
                        i1 = (att.i + rk) % rk
                        i2 = (n_att.i + rk) % rk
                    if i1 != i2:
                        if self.verbose >= 5:
                            print(
                                f"[EasyPatternOptimization.validate_attribute_mapping] failed "
                                f"attribute {att.name!r} (value int), nodes: "
                                f"{g.builder.pretty_node(node, short=True)} / "
                                f"{g.builder.pretty_node(pat_node, short=True)}"
                            )
                        return False
                if att.type == AttributeProto.FLOAT and att.f != n_att.f:
                    if self.verbose >= 5:
                        print(
                            f"[EasyPatternOptimization.validate_attribute_mapping] failed "
                            f"attribute {att.name!r} (value float), nodes: "
                            f"{g.builder.pretty_node(node, short=True)} / "
                            f"{g.builder.pretty_node(pat_node, short=True)}"
                        )
                    return False
                if att.type == AttributeProto.STRING and att.s != n_att.s:
                    if self.verbose >= 5:
                        print(
                            f"[EasyPatternOptimization.validate_attribute_mapping] "
                            f"failed attribute {att.name!r} (value string), "
                            f"nodes: {g.builder.pretty_node(node, short=True)} / "
                            f"{g.builder.pretty_node(pat_node, short=True)}"
                        )
                    return False
                assert att.type in {
                    AttributeProto.INT,
                    AttributeProto.FLOAT,
                    AttributeProto.STRING,
                }, (
                    f"Attribute comparison not implemented for data_type={att.type}, "
                    f"att={att} in node {pat_node}"
                )
        return True

    def _update_ambiguities(
        self, pair_results_names, node: NodeProto, pattern_node: NodeProto, debug_msg=Callable
    ):
        for a, b in zip(node.input, pattern_node.input):
            if b in pair_results_names:
                assert pair_results_names[b] == a, (
                    f"Ambiguity {b!r} is mapped to {pair_results_names[b]!r} and {a!r} "
                    f"pair_results_names={pair_results_names}, pattern is\n"
                    f"{debug_msg()}"
                )

            else:
                pair_results_names[b] = a
        for a, b in zip(node.output, pattern_node.output):
            if b in pair_results_names:
                assert pair_results_names[b] == a, (
                    f"Ambiguity {b!r} is mapped to {pair_results_names[b]!r} and {a!r} "
                    f"pair_results_names={pair_results_names}, pattern is\n"
                    f"{debug_msg()}"
                )
            else:
                pair_results_names[b] = a

    def _has_ambiguities(
        self, pair_results_names, node: NodeProto, pattern_node: NodeProto
    ) -> bool:
        for a, b in zip(node.input, pattern_node.input):
            if b in pair_results_names and pair_results_names[b] != a:
                return True
        for a, b in zip(node.output, pattern_node.output):
            if b in pair_results_names and pair_results_names[b] != a:
                return True
        return False

    def _pattern_to_string(self, g: "GraphBuilder"):  # noqa: F821
        return textwrap.indent(self.display_pattern(g, self.match_pattern), "    ")

    def pretty_matched_pairs(self, pairs: List[Tuple[NodeProto, NodeProto]]) -> str:
        "Pretty display for paired nodes."
        rows = []
        for a, b in pairs:
            sa = f"- {a.op_type}({', '.join(a.input)}) -> {', '.join(a.output)}"
            sb = f"+ {b.op_type}({', '.join(b.input)}) -> {', '.join(b.output)}"
            rows.extend([sa, sb])
        return "\n".join(rows)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        pat = self._get_match_pattern(g)

        # Let's match the first node.
        # Then we need to match successors and predecessors.
        p_node = pat.nodes[-1]  # the last one
        if node.op_type != p_node.op_type:
            # The first node does not have the same type.
            return self.none()
        if len(node.input) != len(p_node.input):
            return self.none(node, inspect.currentframe().f_lineno)

        check_ids = set(id(n) for n in pat.nodes)
        if self.verbose > 5:
            print(
                f"[EasyPatternOptimization.match] -- starts with "
                f"{node.op_type}({', '.join(node.input)})"
            )
            if self.verbose >= 10:
                print("[EasyPatternOptimization.match] match pattern")
                print(self._pattern_to_string(g))

        pair_results_names = {}
        self._update_ambiguities(pair_results_names, node, p_node)
        marked = {id(p_node): (node, p_node)}
        stacked = [id(p_node)]
        iteration = 0

        if self.verbose > 5:
            self._debug = dict(
                pattern=pat,
                marked=marked,
                stacked=stacked,
                iteration=iteration,
                node=node,
                pattern_node=p_node,
                pattern_nodes=pat.nodes,
            )

        # to avoid infinite loops.
        max_iter = len(pat.nodes) * 2
        while stacked and iteration < max_iter:
            assert all(id(b[1]) in check_ids for b in marked.values()), (
                f"At least one id is not part of the pattern ids={check_ids}, "
                f"marked={set(id(b[1]) for b in marked.values())}"
            )

            iteration += 1
            if self.verbose > 5:
                print(
                    f"[EasyPatternOptimization.match] iteration={iteration} "
                    f"n_marked={len(marked)}, n_stacked={len(stacked)}, "
                    f"marked_types={Counter(_[1].op_type for _ in marked.values())}"
                )
            idn = stacked.pop()
            n, pn = marked[idn]

            fall_back_candidates = None
            if any(pat.node_before(i) is not None for i in pn.input):
                # There are backward nodes in the pattern.
                res = self._match_backward(
                    g, node, pat, marked, pair_results_names, stacked, n, pn
                )
                if res is None:
                    if self.verbose > 5:
                        print("[EasyPatternOptimization.match] done. backward failed.")
                    return res
            else:
                # We check then if an input or pn has an unmatched node.
                for x in pn.input:
                    psuccessors = pat.next_nodes(x)
                    if len(psuccessors) == 1:
                        # It is itself.
                        continue
                    for pnn in psuccessors:
                        if id(pnn) not in marked:
                            # One unmarked node is consuming the input.
                            # The potential list of candidates.
                            fall_back_candidates = list(zip(n.input, pn.input))
                            break

            assert all(id(b[1]) in check_ids for b in marked.values()), (
                f"At least one id is not part of the pattern ids={check_ids}, "
                f"marked={set(id(b[1]) for b in marked.values())}"
            )

            res = self._match_forward(g, node, pat, marked, pair_results_names, stacked, n, pn)
            if res is None:
                if self.verbose > 5:
                    print("[EasyPatternOptimization.match] done. forward failed.")
                return res

            if res == 0 and fall_back_candidates:
                # No backward possible, no forward either.
                # We make sure that one of pattern inputs is not linked to another
                # node in the pattern itself.
                for candidate in fall_back_candidates:
                    res = self._match_forward(
                        g, node, pat, marked, pair_results_names, stacked, *candidate
                    )
                    if res is None or res == 0:
                        continue
                    break

            assert all(id(b[1]) in check_ids for b in marked.values()), (
                f"At least one id is not part of the pattern ids={check_ids}, "
                f"marked={set(id(b[1]) for b in marked.values())}"
            )

            if self.verbose > 5:
                self._debug["iteration"] = iteration

        if iteration >= max_iter and stacked:
            self.hint("reached {iteration}>={max_iter} iterations")
            return self.none(node, inspect.currentframe().f_lineno)

        # At this point, the pattern is matched but let's make sure.
        assert len(stacked) == 0, f"There are still {len(stacked)} nodes to explore."
        if len(marked) != len(pat.nodes):
            # This should matched in most cases but when there are
            # multiple outputs,
            self._hint(
                "MATCH: not enough matched nodes",
                "-- len(marked)",
                len(marked),
                "-- len(pat.nodes)",
                len(pat.nodes),
            )
            return self.none(node, inspect.currentframe().f_lineno)

        # We order the matched nodes in the same order than the pattern
        # to let next functions to be able to build the matching again.
        matched_nodes = [marked[id(n)][0] for i, n in enumerate(pat.nodes)]

        if not self.validate_attribute_mapping(g, matched_nodes, pat.nodes):
            if self.verbose >= 2:
                print(
                    f"[EasyPatternOptimization.match] attribute validation failed-1 "
                    f"{len(marked)} marked nodes with {iteration} iterations"
                )
            return None

        if not self.validate_mapping(g, matched_nodes, pat.nodes):
            if self.verbose >= 2:
                print(
                    f"[EasyPatternOptimization.match] validation failed-2 "
                    f"{len(marked)} marked nodes with {iteration} iterations"
                )
            return None

        if self.verbose > 5:
            print(
                f"[EasyPatternOptimization.match] done = matched. "
                f"{len(marked)} marked nodes with {iteration} iterations"
            )
            if self.verbose >= 10:
                for node, pat_node in zip(matched_nodes, pat.nodes):
                    sleft = f"{node.op_type}({node.input})->{node.output}"
                    print(
                        f"    {sleft}{' ' * (60 - len(sleft))}"
                        f"MATCHED  {pat_node.op_type}"
                        f"({pat_node.input})->{pat_node.output}"
                    )

        return MatchResult(self, matched_nodes, self.apply)

    def apply_pattern(self, g: "GraphBuilder", *args, **kwargs):  # noqa: F821
        """
        Applies the replacement.
        """
        raise NotImplementedError(
            f"Class {self.__class__.__name__!r} must overwrite method 'apply_pattern'."
        )

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        *nodes: Sequence[NodeProto],
    ) -> List[NodeProto]:
        # Why build the pattern gain
        pat = self._get_match_pattern(g)
        assert len(nodes) == len(pat.nodes), (
            f"Mismatch matched nodes pattern has {len(pat.nodes)} != {len(nodes)} = "
            f"the number of matched nodes"
        )
        new_pat = self._build_pattern(g, self.apply_pattern)
        assert len(new_pat.inputs) == len(pat.inputs), (
            f"Not the same number of inputs, matched inputs={len(new_pat.inputs)}, "
            f"got {len(pat.inputs)} in the applied pattern."
        )
        assert len(new_pat.outputs) == len(pat.outputs), (
            f"Not the same number of outputs, matched outputs={new_pat.output_names}, "
            f"got {pat.output_names} in the applied pattern."
        )

        if g.verbose > 5:
            print(
                f"[EasyPatternOptimization.apply] replace {len(nodes)} nodes: "
                f"{self.display_pattern(g, self.apply_pattern)}"
            )

        matched_pattern_to_applied_pattern = {}
        for i, j in zip(pat.input_names, new_pat.input_names):
            matched_pattern_to_applied_pattern[i] = j
        for i, j in zip(pat.output_names, new_pat.output_names):
            matched_pattern_to_applied_pattern[i] = j

        matched_pattern_to_graph_name = {}
        input_names = set(pat.input_names)
        output_names = set(pat.output_names)

        matched_pairs = list(zip(nodes, pat.nodes))
        for gn, pn in matched_pairs:
            assert (
                gn.op_type == pn.op_type
            ), f"Unexpected type mismatch {gn.op_type!r} != {pn.op_type!r}"
            assert len(gn.input) == len(
                pn.input
            ), f"Unexpected number of inputs for type {gn.op_type}"
            for a, b in zip(gn.input, pn.input):
                if b not in input_names or b == "":
                    # optional input or not an interesting input
                    continue
                if b in matched_pattern_to_graph_name:
                    assert matched_pattern_to_graph_name[b] == a, (
                        f"Ambiguities, pattern {self.__class__.__name__} "
                        f"name {b!r} means {a!r} or "
                        f"{matched_pattern_to_graph_name[b]!r}\n"
                        f"{self.pretty_matched_pairs(matched_pairs)}"
                    )
                else:
                    matched_pattern_to_graph_name[b] = a

            assert len(gn.output) == len(
                pn.output
            ), f"Unexpected number of outputs for type {gn.op_type}"
            for a, b in zip(gn.output, pn.output):
                if b not in output_names or b == "":
                    # Only final outputs are interesting.
                    continue
                assert a != "", f"{a!r} cannot be optional"
                if b in matched_pattern_to_graph_name:
                    assert matched_pattern_to_graph_name[b] == a, (
                        f"Ambiguities, pattern name {b!r} means "
                        f"{a!r} or {matched_pattern_to_graph_name[b]}"
                    )
                else:
                    matched_pattern_to_graph_name[b] = a

        replacements = {}
        for k, v in matched_pattern_to_graph_name.items():
            replacements[matched_pattern_to_applied_pattern[k]] = v

        # Creation of the new initializers
        for name, init in new_pat.builder.initializers_dict.items():
            # We add them to the graph, they will be removed if unused.
            new_name = g.make_initializer(
                name, init, source=f"EasyPatternOptimization.init/from({name})"
            )
            replacements[new_name] = name

        # Creation of the new node.
        new_nodes = []
        for node in new_pat.nodes:
            new_inputs = []
            for i in node.input:
                assert i in replacements, f"Unable to find {i!r} in {replacements}"
                ni = replacements[i]
                new_inputs.append(ni)
            new_outputs = []
            for o in node.output:
                if o in replacements:
                    new_outputs.append(replacements[o])
                else:
                    # We give it a new name.
                    n = g.unique_name(o)
                    replacements[o] = n
                    new_outputs.append(n)

            if (
                node.op_type == "Constant"
                and node.domain == ""
                and len(node.attribute) == 1
                and node.attribute[0].name == "value"
            ):
                value = node.attribute[0].t
                size = np.prod(value.dims)
                if size >= g.builder.optimization_options.constant_size:
                    # We check the size to convert it into initializer if needed.
                    name = g.make_initializer(
                        new_outputs[0],
                        value,
                        source=f"EasyPatternOptimization.constant/from({new_outputs[0]})",
                    )
                    assert name == new_outputs[0], f"Name mismatch {name} != {new_outputs[0]}"
                    continue

            new_node = g.make_node(
                node.op_type,
                new_inputs,
                new_outputs,
                domain=node.domain,
                name=node.name,
            )
            new_node.attribute.extend(node.attribute)
            new_nodes.append(new_node)

        if g.verbose > 5:
            print(f"[EasyPatternOptimization.apply] done with {len(new_nodes)} nodes")

        self.post_apply_pattern(g, *nodes)
        return new_nodes

    def post_apply_pattern(self, g, *nodes):
        """
        Method to overload to apply as step after the pattern was applied.
        """


class OnnxEasyPatternOptimization(EasyPatternOptimization):
    """
    Implementations pattern matching with onnx models.

    :param match_model: model expressing the pattern to match
    :param apply_model: model expression the replacement pattern
    """

    def __init__(
        self,
        match_model: Union[ModelProto, FunctionProto],
        apply_model: Union[ModelProto, FunctionProto],
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self._match_model = match_model
        self._apply_model = apply_model

    def _build_pattern(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        fct: Callable,
    ) -> "GraphBuilderPatternOptimization":  # noqa: F821
        if fct == self.match_pattern:
            onx = self._match_model
        elif fct == self.apply_pattern:
            onx = self._apply_model
        else:
            raise AssertionError(f"Cannot return pattern for unknown method {fct!r}.")

        from ..xbuilder import GraphBuilder
        from .graph_builder_optim import GraphBuilderPatternOptimization

        gb = GraphBuilder(onx)
        g2 = GraphBuilderPatternOptimization(
            gb, verbose=max(0, gb.verbose - 1), processor=g.processor
        )
        g2._build()
        return g2


def make_pattern_from_onnx(
    name: str,
    match_model: Union[ModelProto, FunctionProto],
    apply_model: Union[ModelProto, FunctionProto],
    verbose: int = 0,
) -> OnnxEasyPatternOptimization:
    """
    Dynamically create a new class inheriting from
    :class:`EasyPatternOptimization`.

    :param name: class name
    :param match_model: model expressing the pattern to match
    :param apply_model: model expression the replacement pattern
    :param verbose: verbosity
    :return: instance of a new class
    """
    new_type = type(name, (OnnxEasyPatternOptimization,), {})
    return new_type(match_model, apply_model, verbose=verbose)


def pattern_table_doc(
    pattern_list: List[PatternOptimization], as_rst: bool = False
) -> Union[str, List[Dict[str, Any]]]:
    """
    Builds a table for with some information about patterns.
    See :func:`experimental_experiment.xoptim.get_pattern_list`
    for an example.
    """
    data = []
    for pat in pattern_list:
        data.append(
            dict(
                name=pat.__class__.__name__,
                short_name=pat.__class__.__name__.replace("Pattern", ""),
                priority=pat.priority,
                doc=pat.__class__.__doc__.split("::", maxsplit=1)[0].replace("\n", " "),
            )
        )
    if as_rst:
        import pandas

        df = pandas.DataFrame(data)
        return df.to_markdown(tablefmt="rst")
    return data
