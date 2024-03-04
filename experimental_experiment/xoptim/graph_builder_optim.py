import os
import pprint
import time
from typing import Any, Dict, Iterator, List, Optional, Union
from onnx import AttributeProto, NodeProto
import onnx.helper as oh
from ..xbuilder._onnx_helper import enumerate_subgraphs
from ..xbuilder.type_inference import infer_types
from .patterns import MatchResult, PatternOptimization, get_default_patterns


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
        self.patterns = patterns or get_default_patterns()
        self.verbose = max(verbose, int(os.environ.get("LOG_PATTERN_OPTIMIZE", "0")))
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

    def is_constant(self, name: str) -> bool:
        """
        Tells if a result is a constant.
        """
        return self.builder.is_constant(name)

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

    def get_constant_or_attribute(
        self, node: NodeProto, attribute: str, input_index: int
    ) -> Any:
        """
        Returns an input or the value of an attribute.
        Some attributes became inputs in more recent opsets.
        The function checks both.

        :param node: node
        :param attribute: attribute name
        :input_index: input index
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
        return self.get_computed_constant(node.input[input_index])

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
        name = self.builder.unique_node_name(name)
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
        self.builder.insert_and_remove_nodes(position_insert, new_nodes, removed)

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

        return new_nodes

    def optimize(
        self, max_iter=-1, remove_identity: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Optimizes the based on the given list of patterns.

        :param max_iter: maximum number of iterations
        :param remove_identity: remove identity nodes, it is better to keep it True,
            not doing it might prevent other patterns to find a set of nodes to optimize
        :return: the method returns informations about the applied processes.
        """

        def _check(step):
            assert (
                len(self.builder.nodes) > 0
            ), f"The onnx model is empty (step {step}, no node)"
            known = set(n.name for n in self.builder.inputs)
            known |= set(self.builder.initializers_dict)
            for node in self.builder.nodes:
                for i in node.input:
                    assert (
                        i in known
                    ), f"Unknown input {i!r}, step {step!r} in node {node}"
                known |= set(node.output)
            for o in self.builder.outputs:
                assert o.name in known, f"Unknown output {o.name!r}, step {step!r}"

        assert (
            not self.recursive
        ), "GraphBuilderPatternOptimization.optimize does not implement recursivity"
        if max_iter == -1:
            max_iter = len(self.builder.nodes)
        if self.verbose > 0:
            print(
                f"[GraphBuilderPatternOptimization.optimize] start with "
                f"{len(self.builder.nodes)} nodes and {len(self.patterns)} patterns"
            )

        begin_all = time.perf_counter()
        statistics = []

        last_it = 0
        for it in range(max_iter):
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
                begin = time.perf_counter()
                for match in pattern.enumerate_matches(self):
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
                durations[str(pattern)] = time.perf_counter() - begin

            if self.verbose > 0 and matches:
                rev = max([(v, k) for k, v in durations.items()])
                revs = f"{rev[-1]}:{rev[0]:.3f}"
                print(
                    f"[GraphBuilderPatternOptimization.optimize] applies {len(matches)} matches, "
                    f"[0]={str(matches[0][-1])} - time={sum(durations.values()):.3f} | max_time={revs}"
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

                added_nodes = self.apply_match(match)
                _check(str(match))
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
                    pattern=str(pattern),
                    added=add,
                    removed=rem,
                    iteration=it,
                    match_index=im,
                    time_in=durations[str(pattern)],
                )
                statistics.append(obs)

                n_added += add
                n_removed += rem
            if self.verbose > 2:
                print(
                    f"[GraphBuilderPatternOptimization.optimize] done all: -{n_removed} +{n_added} nodes"
                )

            # remove unncessary identity nodes

            id_removed = self.builder.remove_identity_nodes()
            if id_removed:
                statistics.append(
                    dict(
                        pattern="remove_identity_nodes",
                        iteration=it,
                        removed=id_removed,
                    )
                )
            _check("remove_identity")

            # rebuild the graph structure

            self._build()

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
