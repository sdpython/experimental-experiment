import os
import pprint
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Union
import numpy as np
from onnx import AttributeProto, NodeProto
import onnx.helper as oh
from ..xbuilder._onnx_helper import enumerate_subgraphs
from ..xbuilder.type_inference import infer_types
from .patterns import MatchResult, PatternOptimization, get_default_patterns


def _count(matches):
    stats = {}
    for n in matches:
        cl = n[0].__class__.__name__
        if cl in stats:
            stats[cl] += 1
        else:
            stats[cl] = 1
    return ", ".join([f"{v}*{k}" for k, v in stats.items()])


class GraphBuilderPatternOptimization:
    """
    Implements optimization after the conversion is done.
    The differences between the two models can be display with a
    command line such as:

    ::

        python -m onnx_array_api compare -m1 <model.onnx> -m2 <optimized.onnx> -m nodes -c 80

    This class assumes a pattern cannot reuse an existing name.
    """

    def __init__(
        self,
        builder: "GraphBuilder",  # noqa: F821
        patterns: Optional[List[PatternOptimization]] = None,
        recursive: bool = False,
        verbose: int = 0,
    ):
        self.builder = builder
        self.verbose = max(verbose, int(os.environ.get("LOG_PATTERN_OPTIMIZE", "0")))
        self.patterns = patterns or get_default_patterns(self.verbose)
        self.recursive = recursive
        self._build()
        # This assume a name is given once and
        # no constant can replace an existing one.
        # _build method should not change it.
        self._cache_computed_constant = {}

    def iter_nodes(self) -> Iterator:
        for node in self.builder.nodes:
            yield node

    def _build(self):
        """
        Builds successor and predecessor.
        """
        self.nodes_ = {}
        self.outputs_ = {o.name for o in self.builder.outputs}
        for node in self.iter_nodes():
            self.nodes_[id(node)] = node

        self.predecessors_ = {}
        self.successors_ = {}
        self.used_ = {}
        for k, v in self.nodes_.items():
            assert isinstance(v, NodeProto), f"Unexpected type {type(v)} for node {k}"
            for o in v.output:
                self.predecessors_[o] = k
            for i in v.input:
                if i not in self.successors_:
                    self.successors_[i] = []
                self.successors_[i].append(k)

            for sub in enumerate_subgraphs(v):
                g = sub[-1]
                sub_knowns = set()
                for n in g.input:
                    sub_knowns.add(n.name)
                for n in g.initializer:
                    sub_knowns.add(n.name)
                for n in g.sparse_initializer:
                    sub_knowns.add(n.name)
                for n in g.node:
                    for i in n.input:
                        if i not in sub_knowns:
                            # an input coming from the parent
                            self.used_.add(i)
                    for i in n.output:
                        sub_knowns.add(i)

    def is_used_by_subgraph(self, name: str) -> bool:
        """
        Tells if a result is used by a subgraphs.
        """
        return name in self.used_

    def is_output(self, name: str) -> bool:
        """
        Tells if a result is an output.
        """
        return name in self.outputs_

    def is_used_more_than_once(self, name: str) -> bool:
        """
        Tells if a result is used more than once in the current graph or in a subgraph
        or if it is an output.
        """
        if self.is_used_by_subgraph(name):
            return True
        if self.is_output(name):
            return True
        suc = self.successors_[name]
        return len(suc) > 1

    def is_used_only_by(self, name, *nodes: List[NodeProto]) -> bool:
        """
        Tells if a result is only used by a specific set of nodes.
        """
        next_nodes = self.next_nodes(name)
        allowed = set(id(n) for n in nodes)
        return all(map(lambda n: id(n) in allowed, next_nodes))

    def is_constant(self, name: str) -> bool:
        """
        Tells if a result is a constant.
        """
        return self.builder.is_constant(name)

    def is_constant_scalar(self, name: str, value: Optional[Any] = None) -> bool:
        """
        Tells if a constant is a scalar

        :param name: name
        :param value: value to compare to if specified
        :return: boolean
        """
        if not self.is_constant(name):
            return False
        cst = self.get_computed_constant(name)
        if hasattr(cst, "numpy"):
            # This could fail xith bfloat16, ...
            cst = cst.detach().cpu().numpy()
        assert isinstance(
            cst, np.ndarray
        ), f"Unexpected type for constant {name}!r, type is {type(cst)}"
        shape = cst.shape
        if shape not in (tuple(), (1,)):
            return False
        if value is None:
            return True
        return all(cst == value)

    def get_constant_scalar(self, name: str) -> Union[int, float]:
        """
        Returns a scalar as a constant.

        :param name: name
        :return: int or float
        """
        cst = self.get_computed_constant(name)
        if hasattr(cst, "numpy"):
            # This could fail xith bfloat16, ...
            cst = cst.detach().cpu().numpy()
        assert isinstance(
            cst, np.ndarray
        ), f"Unexpected type for constant {name}!r, type is {type(cst)}"
        shape = cst.shape
        value = cst[0] if shape == (1,) else cst
        if value.dtype in {np.float32, np.float16, np.float64}:
            return float(value)
        return int(value)

    def get_computed_constant(
        self, name: str, statistics: Optional[List[str]] = None
    ) -> Any:
        """
        Returns the value for the constant `name`.
        """
        if name in self._cache_computed_constant:
            value = self._cache_computed_constant[name]
        else:
            value = self.builder.get_constant(name, computed_value=True)
            self._cache_computed_constant[name] = value
        if statistics is None:
            return value
        stats = []
        for st in statistics:
            key = name, st
            if key in self._cache_computed_constant:
                stat = self._cache_computed_constant[key]
            else:
                if st == "min":
                    stat = value.min()
                elif st == "max":
                    stat = value.max()
                else:
                    raise RuntimeError(f"Unknown statistics {st!r} for {name!r}.")
                self._cache_computed_constant[key] = stat
            stats.append(stat)
        return stats

    def get_attribute(
        self, node: NodeProto, att_name: str, exc: bool = True
    ) -> AttributeProto:
        """
        Returns an attribute for a node.
        """
        return self.builder.get_attribute(node, att_name, exc=exc)

    def get_axis(self, node: NodeProto, default_axis: Optional[int] = None) -> int:
        """
        Retrieves the axis for many operators.
        """
        att = self.get_attribute(node, "axis", exc=False)
        if att is None:
            assert (
                default_axis is not None
            ), f"Node {node.op_type} has no axis and no default value."
            return default_axis
        return att.i

    def get_constant_or_attribute(
        self,
        node: NodeProto,
        attribute: str,
        input_index: int,
        cvt: Optional[Callable] = None,
    ) -> Any:
        """
        Returns an input or the value of an attribute.
        Some attributes became inputs in more recent opsets.
        The function checks both.

        :param node: node
        :param attribute: attribute name
        :param input_index: input index
        :param cvt: if not None, called this conversion function before
            returning the result
        :return: value
        """
        found = None
        for att in node.attribute:
            if att.name == attribute:
                found = att
        assert (
            found is None
        ), f"get_constant_or_attribute not implemented for attribute={attribute!r} and node={node}."
        assert input_index < len(
            node.input
        ), f"Input {input_index} does not exist in node {node}."
        val = self.get_computed_constant(node.input[input_index])
        if cvt is None:
            return val
        try:
            return cvt(val)
        except (ValueError, TypeError) as e:
            raise RuntimeError(f"Unable to convert val={val} with cvt={cvt}") from e

    def has_type(self, name: str) -> bool:
        """
        Tells if a result has a type.
        """
        return self.builder.has_type(name)

    def get_type(self, name: str) -> int:
        """
        Returns the type of a result.
        """
        return self.builder.get_type(name)

    def has_rank(self, name: str) -> int:
        """
        Tells if a result has a rank.
        """
        return self.builder.has_rank(name)

    def get_rank(self, name: str) -> int:
        """
        Returns the rank of a result.
        """
        return self.builder.get_rank(name)

    def has_shape(self, name: str) -> bool:
        """
        Tells if a result has a shape.
        """
        return self.builder.has_shape(name)

    def get_shape(self, name: str) -> int:
        """
        Returns the shape of a result.
        """
        return self.builder.get_shape(name)

    def node_before(self, name: str) -> NodeProto:
        """
        Returns the node producing this output.
        Returns None if it is an input or an initializer.
        """
        if name not in self.predecessors_:
            return None
        predecessor = self.predecessors_[name]
        return self.nodes_[predecessor]

    def try_infer_type(self, name: str, exc: bool = False) -> int:
        """
        Tries to infer the type of a result.

        :param name: name of the result for which to infer the type
        :param exc: if True, raises an exception if something goes wrong
        :return: type
        """
        if self.has_type(name):
            it = self.get_type(name)
            if exc and it == 0:
                raise RuntimeError(
                    f"Unable to guess type for {name!r}, "
                    f"knowns types are {pprint.pformat(self.builder._known_types)}"
                )
            return it

        assert (
            name not in self.builder.initializers_dict
        ), f"name {name!r} has no type but it is an initializer"
        assert name not in self.builder.input_names, (
            f"name {name!r} has no type but it is an input, "
            f"known_types={pprint.pformat(self.builder._known_types)}"
        )
        node = self.node_before(name)
        input_types = [
            (self.get_type(i) if self.has_type(i) else 0) for i in node.input
        ]
        output_type = infer_types(node, input_types, name)
        if output_type > 0:
            return output_type

        # second try with more depth
        input_types = [self.try_infer_type(i, exc=exc) for i in node.input]
        output_type = infer_types(node, input_types, name)
        if output_type > 0:
            return output_type

        # no luck
        if exc:
            raise RuntimeError(
                f"Unable to guess type for {name!r}, "
                f"knowns types are {pprint.pformat(self.builder._known_types)}"
            )
        return 0

    def try_infer_shape(self, name: str, exc: bool = False) -> int:
        """
        Tries to infer the type of a result.

        :param name: name of the result for which to infer the type
        :param exc: if True, raises an exception if something goes wrong
        :return: type
        """
        if self.has_shape(name):
            return self.get_shape(name)
        if exc:
            raise RuntimeError(
                f"Unable to guess shape for {name!r}, "
                f"knowns shapes are {pprint.pformat(self.builder._known_shapes)}"
            )
        return None

    def next_node(self, name: str) -> NodeProto:
        """
        Returns the next node if it is unique, otherwise fails.
        """
        res = self.next_nodes(name)
        assert len(res) == 1, f"Unexpected number of successors {len(res)} for {name!r}"
        return res[0]

    def next_nodes(self, name: str) -> List[NodeProto]:
        """
        Returns the node consuming the given results.
        """
        if name not in self.successors_:
            return []
        return [self.nodes_[i] for i in self.successors_[name]]

    @property
    def main_opset(self):
        "Returns the opset for the main domain (assuming it is used)."
        return self.builder.opsets[""]

    def make_initializer(
        self, name: str, value: Any, external: bool = False, msg: str = ""
    ) -> str:
        new_name = self.builder.make_initializer(
            name, value, external=external, msg=msg
        )
        return new_name

    def unique_name(self, prefix: str) -> str:
        return self.builder.unique_name(prefix)

    def make_node_check_opset(
        self,
        op_type: str,
        inputs: Union[str, List[str]],
        outputs: Union[int, List[str], str] = 1,
        domain: str = "",
        attributes: Optional[List[AttributeProto]] = None,
        name: Optional[str] = None,
        **kwargs,
    ):
        """
        Creates a node without adding it to the graph but
        adapt for some known operators changing over
        multiple opets.

        :param op_type: operator type
        :param inputs: input names
        :param outputs: outputs names, if one integer, creates n unique names,
            if str, creates one unique names, if a list, use the name
        :param domain: node domain
        :param attributes: list of attributes
        :param name: node name
        :param kwargs: other attributes
        :return: a node
        """
        assert domain == "", f"The method only supports the main domain not {domain!r}"
        if op_type in {"Squeeze", "Unsqueeze"}:
            if self.builder.main_opset < 13:
                assert (
                    len(inputs) == 1
                ), f"axis must be given as an attribute for {op_type!r}"
                return self.make_node(
                    op_type,
                    inputs,
                    outputs,
                    domain=domain,
                    attributes=attributes,
                    name=name,
                    **kwargs,
                )
            if len(inputs) == 1 and "axes" in kwargs:
                axes = kwargs["axes"]
                axes_name = self.make_initializer("", np.array([axes], dtype=np.int64))
                inputs.append(axes_name)
                del kwargs["axes"]
            return self.make_node(
                op_type,
                inputs,
                outputs,
                domain=domain,
                attributes=attributes,
                name=name,
                **kwargs,
            )

        raise RuntimeError(f"Operator {op_type!r} not supported yet.")

    def make_node(
        self,
        op_type: str,
        inputs: Union[str, List[str]],
        outputs: Union[int, List[str], str] = 1,
        domain: str = "",
        attributes: Optional[List[AttributeProto]] = None,
        name: Optional[str] = None,
        **kwargs,
    ) -> NodeProto:
        """
        Creates a node without adding it to the graph.

        :param op_type: operator type
        :param inputs: input names
        :param outputs: outputs names, if one integer, creates n unique names,
            if str, creates one unique names, if a list, use the name
        :param domain: node domain
        :param attributes: list of attributes
        :param name: node name
        :param kwargs: other attributes
        :return: a node
        """
        name = self.builder.unique_node_name(name)
        if isinstance(outputs, int):
            if outputs == 1:
                outputs = [self.unique_name(f"{op_type.lower()}-{inputs[0]}")]
            else:
                outputs = [
                    self.unique_name(f"{op_type.lower()}-{inputs[0]}-{i}")
                    for i in range(outputs)
                ]
        elif isinstance(outputs, str):
            outputs = [self.unique_name(outputs)]
        proto = oh.make_node(
            op_type,
            inputs,
            outputs,
            domain=domain,
            name=name,
            **kwargs,
        )
        if attributes:
            proto.attribute.extend(attributes)
        return proto

    def apply_match(self, match: MatchResult) -> List[NodeProto]:
        """
        Applies one match.
        Returns the new nodes.
        """
        idn = [id(n) for n in match.nodes if n is not None]

        assert all(
            map(lambda i: i in self.nodes_, idn)
        ), f"One node in {idn} is not referenced"

        positions = {id(n): i for i, n in enumerate(self.iter_nodes())}

        assert all(
            map(lambda i: i in positions, idn)
        ), f"One node in {idn} is not referenced"

        removed = [positions[i] for i in idn]
        position_insert = (
            None if match.insert_at is None else positions[id(match.insert_at)]
        )
        new_nodes = match.apply(self, *match.nodes)

        if self.verbose >= 10:
            print(f"[GraphBuilderPatternOptimization.apply_match] {match}")
            for node in match.nodes:
                if node is None:
                    continue
                print(f"  - {node.op_type}: {node.input} -> {node.output}")
            for node in new_nodes:
                if node is None:
                    continue
                print(f"  + {node.op_type}: {node.input} -> {node.output}")

        self.builder.insert_and_remove_nodes(position_insert, new_nodes, removed)
        if self.verbose >= 10:
            print(f"[GraphBuilderPatternOptimization.apply_match] {match} applied.")
        return new_nodes

    def _check_graph(self, statistics, step, iteration, code):
        begin = time.perf_counter()
        assert (
            len(self.builder.nodes) > 0
        ), f"The onnx model is empty (step {step}, no node)"
        known = set(n.name for n in self.builder.inputs)
        known |= set(self.builder.initializers_dict)
        for p, node in enumerate(self.builder.nodes):
            for i in node.input:
                if i == "":
                    continue
                assert i in known, (
                    f"Unknown input {i!r}, step {step!r} at position {p} "
                    f"in node {node.op_type} "
                    f"[{node.name}]: {node.input} -> {node.output}"
                )
            known |= set(node.output)
        for o in self.builder.outputs:
            assert o.name in known, f"Unknown output {o.name!r}, step {step!r}"
        statistics.append(
            dict(
                pattern=f"check_pattern_{code}",
                time_in=time.perf_counter() - begin,
                iteration=iteration,
            )
        )

    def do_not_remove(self, node: NodeProto) -> bool:
        """Tells if a node can be removed."""
        return self.builder.do_not_remove(node)

    def optimize(
        self, max_iter=-1, remove_identity: bool = True, stop_after: int = -1
    ) -> List[Dict[str, Any]]:
        """
        Optimizes the based on the given list of patterns.

        :param max_iter: maximum number of iterations
        :param remove_identity: remove identity nodes, it is better to keep it True,
            not doing it might prevent other patterns to find a set of nodes to optimize
        :param sopt_after: stop after this number of replacements (to debug),
            -1 not to stop
        :return: the method returns informations about the applied processes.

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
        """
        assert (
            not self.recursive
        ), "GraphBuilderPatternOptimization.optimize does not implement recursivity"
        continue_optimization = True
        if max_iter == -1:
            max_iter = len(self.builder.nodes)
        if self.verbose > 0:
            print(
                f"[GraphBuilderPatternOptimization.optimize] start with "
                f"{len(self.builder.nodes)} nodes and {len(self.patterns)} patterns"
            )
            for i, pattern in enumerate(self.patterns):
                print(
                    f"[GraphBuilderPatternOptimization.optimize] "
                    f"use pattern {i+1}/{len(self.patterns)} - {pattern}"
                )

        begin_all = time.perf_counter()
        statistics = []

        n_applied = 0
        last_it = 0
        for it in range(max_iter):
            if not continue_optimization:
                break
            if self.verbose > 0:
                print(
                    f"[GraphBuilderPatternOptimization.optimize] iteration {it}: "
                    f"{len(self.builder.nodes)} nodes"
                )

            # detects patterns

            found = False
            marked = set()
            matches = []
            durations = {}
            for pattern in self.patterns:
                if not continue_optimization:
                    break
                begin = time.perf_counter()
                before = len(matches)
                for match in pattern.enumerate_matches(self):

                    # bypass this node if the name contains some specific name
                    fail_match = False
                    for n in match.nodes:
                        if n and self.do_not_remove(n):
                            fail_match = True
                            break
                    if fail_match:
                        continue

                    # checks that a node is not already part of another pattern
                    bypass = False
                    for n in match.nodes:
                        if id(n) in marked:
                            # a node is already marked for replacements
                            bypass = True
                            break
                    if bypass:
                        continue
                    for n in match.nodes:
                        marked.add(id(n))
                    found = True
                    if self.verbose > 2:
                        print(
                            f"[GraphBuilderPatternOptimization.optimize] match={match}"
                        )
                    matches.append((pattern, match))
                    if stop_after > 0 and len(matches) + n_applied >= stop_after:
                        continue_optimization = False
                        if self.verbose > 0:
                            print(
                                f"[GraphBuilderPatternOptimization.optimize] stop after with "
                                f"{len(matches)} as stop_after={stop_after} and n_applied={n_applied}"
                            )
                        break

                d = time.perf_counter() - begin
                statistics.append(
                    dict(
                        pattern=f"match_{pattern}",
                        iteration=it,
                        instances=len(matches) - before,
                        time_in=d,
                        match_index=len(matches),
                    )
                )
                durations[pattern.__class__.__name__] = (
                    durations.get(pattern.__class__.__name__, 0) + d
                )

            if self.verbose > 0 and matches:
                if durations:
                    rev = max([(v, k) for k, v in durations.items()])
                    revs = f"{rev[-1]}:{rev[0]:.3f}"
                    if len(matches) == 1:
                        print(
                            f"[GraphBuilderPatternOptimization.optimize] applies "
                            f"{len(matches)} matches, [0]={str(matches[0][-1])} - "
                            f"time={sum(durations.values()):.3f} | max_time={revs}"
                        )
                    else:
                        print(
                            f"[GraphBuilderPatternOptimization.optimize] applies "
                            f"{len(matches)} matches, {_count(matches)} - "
                            f"time={sum(durations.values()):.3f} | max_time={revs}"
                        )
                elif len(matches) == 1:
                    print(
                        f"[GraphBuilderPatternOptimization.optimize] applies "
                        f"{len(matches)} matches, [0]={str(matches[0][-1])}"
                    )
                else:
                    print(
                        f"[GraphBuilderPatternOptimization.optimize] applies "
                        f"{len(matches)} matches, {_count(matches)}"
                    )

            # applies patterns (they must be disjoined)

            n_added = 0
            n_removed = 0
            for im, (pattern, match) in enumerate(matches):
                if self.verbose > 3:
                    print(
                        f"[GraphBuilderPatternOptimization.optimize] "
                        f"apply {match.to_string(short=False)}"
                    )

                begin = time.perf_counter()
                added_nodes = self.apply_match(match)

                if self.verbose > 3:
                    print(
                        f"[GraphBuilderPatternOptimization.optimize] - add "
                        f"{[n.op_type for n in added_nodes]}"
                    )
                add = len(added_nodes)
                added_outputs = set()
                for n in added_nodes:
                    added_outputs |= set(n.output)

                rem = len([n for n in match.nodes if n is not None])
                removed_outputs = set()
                for n in match.nodes:
                    if n is None:
                        continue
                    removed_outputs |= set(n.output)

                full_removed = set(i for i in removed_outputs if i not in added_outputs)
                for i in full_removed:
                    assert not self.is_output(i), (
                        f"Output {i!r} must not be removed, added_outputs={added_outputs},"
                        f"removed_outputs={removed_outputs}"
                    )

                if self.verbose > 3:
                    print(
                        f"[GraphBuilderPatternOptimization.optimize] done {match}: -{rem} +{add} nodes"
                    )
                    if full_removed and self.verbose > 4:
                        print(
                            f"[GraphBuilderPatternOptimization.optimize] removed outputs {full_removed}"
                        )

                obs = dict(
                    pattern=f"apply_{pattern}",
                    added=add,
                    removed=rem,
                    iteration=it,
                    match_index=im,
                    instances=1,
                    time_in=time.perf_counter() - begin,
                )
                statistics.append(obs)
                self._check_graph(statistics, str(match), it, "A")

                n_added += add
                n_removed += rem
                n_applied += 1
            if self.verbose > 2:
                print(
                    f"[GraphBuilderPatternOptimization.optimize] done all: -{n_removed} +{n_added} nodes"
                )

            # remove unnecessary identity nodes

            begin = time.perf_counter()
            id_removed = self.builder.remove_identity_nodes()
            statistics.append(
                dict(
                    pattern="remove_identity_nodes",
                    iteration=it,
                    removed=id_removed,
                    time_in=time.perf_counter() - begin,
                )
            )
            self._check_graph(statistics, "remove_identity", it, "B")

            # rebuild the graph structure

            begin = time.perf_counter()
            self._build()
            statistics.append(
                dict(
                    pattern="build_for_pattern",
                    iteration=it,
                    removed=id_removed,
                    time_in=time.perf_counter() - begin,
                )
            )

            # next iteration

            last_it = it + 1
            if not found:
                break

        if self.verbose > 0:
            duration = time.perf_counter() - begin_all
            print(
                f"[GraphBuilderPatternOptimization.optimize] done after {last_it} iterations with "
                f"{len(self.builder.nodes)} nodes in {duration:.3f}"
            )

        return statistics
