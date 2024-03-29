import inspect
import os
import textwrap
from collections import Counter
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple
import numpy as np
from onnx import NodeProto
from ..xbuilder._dtype_helper import string_to_elem_type


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

    def _debug_print(self) -> str:
        return ""

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
            if self.verbose >= 10 and hasattr(self, "_debug"):
                msg2 = self._debug_print()
                if msg2:
                    msg2 = f"\n{textwrap.indent(msg2, '    ')}"
                print(
                    f"[{self.__class__.__name__}.match] NONE - line: {lineno}:"
                    f"{os.path.split(self.__class__.__module__)[-1]}, "
                    f"op_type={node.op_type}{msg}{msg2}"
                )
            elif self.verbose >= 9:
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
    The current implementation does not match on domain name.
    It does not compares attributes either.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose=verbose)
        self._cache = {}

    @classmethod
    def match_pattern(
        cls,
        g: "GraphBuilder",  # noqa: F821
        *args: List[str],
        **kwargs: Dict[str, Any],
    ):
        """
        Builds the pattern to match.
        """
        raise NotImplementedError(
            f"Class {cls.__name__!r} must overwrite method match_pattern."
        )

    @classmethod
    def _build_pattern(
        cls, g: "GraphBuilderPatternOptimization", fct: Callable  # noqa: F821
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

        g2 = g.builder.empty_copy(as_function=True, constant_size=2**30)
        for name, ann in zip(args, anns):
            if ann is None or ann is str or ann is inspect._empty:
                g2.make_tensor_input(name, 0, None, False)
                continue
            assert isinstance(
                ann, str
            ), f"Annotation for {name!r} must be a string or None but ann={ann!r}"
            itype = string_to_elem_type(ann)
            g2.make_tensor_input(name, itype, None, False)

        output = fct(g2, *args, **kwargs)
        if isinstance(output, str):
            g2.make_tensor_output(output, 0, None, is_dimension=False)
        else:
            for name in output:
                g2.make_tensor_output(name, 0, None, is_dimension=False)
        pat = GraphBuilderPatternOptimization(g2, verbose=max(0, g.verbose - 1))
        pat._build()
        return pat

    def _get_match_pattern(
        self, g: "GraphBuilderPatternOptimization"  # noqa: F821
    ) -> "GraphBuilderPatternOptimization":  # noqa: F821
        cache_key = 0, tuple(sorted(g.opsets.items()))
        if cache_key in self._cache:
            return self._cache[cache_key]

        pat = self._build_pattern(g, self.match_pattern)
        self._cache[cache_key] = pat
        return pat

    def _get_apply_pattern(
        self, g: "GraphBuilderPatternOptimization"  # noqa: F821
    ) -> "GraphBuilderPatternOptimization":  # noqa: F821
        cache_key = 1, tuple(sorted(g.opsets.items()))
        if cache_key in self._cache:
            return self._cache[cache_key]

        pat = self._build_pattern(g, self.apply_pattern)
        self._cache[cache_key] = pat
        return pat

    @classmethod
    def display_pattern(cls, g, fct) -> str:
        """
        Shows the pattern to match or to apply.
        """
        pat = cls._build_pattern(g, fct)
        rows = []
        rows.append(
            f"{fct.__name__}({', '.join(pat.input_names)}) -> {', '.join(pat.output_names)}"
        )
        for node in pat.nodes:
            rows.append(
                f"{node.op_type}({', '.join(node.input)}) -> {', '.join(node.output)}"
            )
        return "\n".join(rows)

    def _match_backward(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        pat: "GraphBuilderPatternOptimization",  # noqa: F821
        marked: Dict[int, Tuple[NodeProto, NodeProto]],
        stacked: List[int],
        n: NodeProto,
        pn: NodeProto,
    ) -> Optional[int]:
        """
        Matches backward.

        :param g: graph
        :param node: root node (the node the matched begain with,
            used only for debugging)
        :param pat: pattern
        :param marked: nodes of the pattern marked as already matched
        :param stacked: next node to look into
        :param n: node coming from the graph
        :param ns: node coming from the pattern
        :return: number of matched nodes, None or False to indicate a failed match
        """
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
        for i, pi in zip(n.input, pn.input):
            ppred = pat.node_before(pi)
            if ppred is None:
                # ppred is None means the pattern ends here.
                continue
            pred = g.node_before(i)
            if pred is None:
                # No node in the graph.
                return self.none(node, inspect.currentframe().f_lineno)
            if pred.op_type != ppred.op_type:
                return self.none(node, inspect.currentframe().f_lineno)
            # matching backward
            key = id(ppred)
            if key not in marked:
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
        stacked: List[int],
        n: NodeProto,
        pn: NodeProto,
    ) -> Optional[int]:
        """
        Matches forward.

        :param g: graph
        :param node: root node (the node the matched begain with,
            used only for debugging)
        :param pat: pattern
        :param marked: nodes of the pattern marked as already matched
        :param stacked: next node to look into
        :param n: node coming from the graph
        :param ns: node coming from the pattern
        :return: number of matched nodes to continue, None or False to indicate a failed match
        """
        res = 0

        # successors
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

        for o, op in zip(n.output, pn.output):
            ns = g.next_nodes(o)
            pns = pat.next_nodes(op)
            if len(pns) == 0:
                # The pattern has no node forward, the matching stops.
                continue
            if len(ns) < len(pns):
                # Not enough node in the graph to match the pattern,
                # the result is known.
                return self.none(node, inspect.currentframe().f_lineno)

            # Here comes the fun part, there is the same number of successors or more
            # nodes in the graph to match with the pattern.
            # And we have to handle the nodes already marked as found.
            # Hopefully, there is only one option.

            if len(ns) == len(pns) == 1:
                # Let's deal with the simple case
                if ns[0].op_type != pns[0].op_type:
                    return self.none(node, inspect.currentframe().f_lineno)

                key = id(pns[0])
                if key not in marked:
                    marked[key] = ns[0], pns[0]
                    stacked.append(key)
                    res += 1
                continue

            # Let's remove the nodes already marked.
            p_marked = [_ for _ in pns if id(_) not in marked]
            id_marked = list(id(marked[id(_)][0]) for _ in pns if id(_) in marked)
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
                if p_marked[0].op_type != free[0].op_type:
                    return self.none(node, inspect.currentframe().f_lineno)

                key = id(p_marked[0])
                if key not in marked:
                    marked[key] = free[0], p_marked[0]
                    stacked.append(key)
                    res += 1
                continue

            # And now another fun part, let's try to handle the case when
            # there is only one option, matching on node type only returns one
            # option.
            expected_op_type = [_.op_type for _ in p_marked]
            got_op_type = [_.op_type for _ in free]

            ec = Counter(expected_op_type)
            gc = Counter(got_op_type)
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
            gtype_to_node = {_.op_type: _ for _ in got_op_type}
            missing = []
            for k, v in ec.items():
                if gc[k] == v == 1:
                    key = id(ptype_to_node[k])
                    if key not in marked:
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
            print(f"[EasyPatternOptimization._match_foward] add {res} nodes")
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
        self._debug["hint"] = args

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

        check_ids = set(id(n) for n in pat.nodes)
        if self.verbose > 5:
            print(
                f"[EasyPatternOptimization.match] starts with "
                f"{node.op_type}({', '.join(node.input)})"
            )
            if self.verbose >= 10:
                print("[EasyPatternOptimization.match] match pattern")
                print(
                    textwrap.indent(self.display_pattern(g, self.match_pattern), "    ")
                )

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

        while stacked:
            assert all(map(lambda b: id(b[1]) in check_ids, marked.values())), (
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

            res = self._match_backward(g, node, pat, marked, stacked, n, pn)
            if res is None:
                if self.verbose > 5:
                    print("[EasyPatternOptimization.match] done. backward failed.")
                return res

            assert all(map(lambda b: id(b[1]) in check_ids, marked.values())), (
                f"At least one id is not part of the pattern ids={check_ids}, "
                f"marked={set(id(b[1]) for b in marked.values())}"
            )

            res = self._match_forward(g, node, pat, marked, stacked, n, pn)
            if res is None:
                if self.verbose > 5:
                    print("[EasyPatternOptimization.match] done. forward failed.")
                return res

            assert all(map(lambda b: id(b[1]) in check_ids, marked.values())), (
                f"At least one id is not part of the pattern ids={check_ids}, "
                f"marked={set(id(b[1]) for b in marked.values())}"
            )

        if self.verbose > 5:
            print(f"[EasyPatternOptimization.match] done. {len(marked)} marked nodes")

        # At this point, the pattern is matched but let's make sure.
        assert len(marked) == len(pat.nodes), (
            f"Number of marked nodes is different, {len(marked)} marked nodes, "
            f"and {len(pat.nodes)} nodes in the pattern, marked is {marked}"
        )
        assert len(stacked) == 0, f"There are still {len(stacked)} nodes to explore."

        # We order the matched nodes in the same order than the pattern
        # to let next functions to be able to build the matching again.
        matched_nodes = [marked[id(n)][0] for i, n in enumerate(pat.nodes)]
        return MatchResult(self, matched_nodes, self.apply)

    @classmethod
    def apply_pattern(cls, g: "GraphBuilder", *args, **kwargs):  # noqa: F821
        """
        Applies the replacement.
        """
        raise NotImplementedError(
            f"Class {cls.__name__!r} must overwrite method 'apply_pattern'."
        )

    @classmethod
    def apply(
        cls,
        g: "GraphBuilder",  # noqa: F821
        *nodes: Sequence[NodeProto],
    ) -> List[NodeProto]:
        pat = cls._build_pattern(g, cls.match_pattern)
        assert len(nodes) == len(pat.nodes), (
            f"Mismatch matched nodes pattern has {len(pat.nodes)} != {len(nodes)} = "
            f"the number of matched nodes"
        )
        new_pat = cls._build_pattern(g, cls.apply_pattern)
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
                f"{cls.display_pattern(g, cls.apply_pattern)}"
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
                        f"Ambiguities, pattern name {b!r} means "
                        f"{a!r} or {matched_pattern_to_graph_name[b]}"
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
            new_name = g.make_initializer(name, init)
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
                    name = g.make_initializer(new_outputs[0], value)
                    assert (
                        name == new_outputs[0]
                    ), f"Name mismatch {name} != {new_outputs[0]}"
                    continue

            new_node = g.make_node(
                node.op_type, new_inputs, new_outputs, domain=node.domain
            )
            new_node.attribute.extend(node.attribute)
            new_nodes.append(new_node)

        if g.verbose > 5:
            print(f"[EasyPatternOptimization.apply] done with {len(new_nodes)} nodes")

        return new_nodes
