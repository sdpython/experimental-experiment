from typing import Any, Iterator, List, Optional, Union
from onnx import AttributeProto, NodeProto
import onnx.helper as oh
from ._onnx_helper import enumerate_subgraphs
from .optimization_patterns import (
    MatchResult,
    PatternOptimization,
    get_default_patterns,
)


class GraphBuilderPatternOptimization:
    """
    Implements optimization after the conversion is done.
    """

    def __init__(
        self,
        builder: "GraphBuilder",  # noqa: F821
        patterns: Optional[List[PatternOptimization]] = None,
        verbose: int = 0,
    ):
        self.builder = builder
        self.patterns = patterns or get_default_patterns()
        self.verbose = verbose
        self._build()

    def iter_nodes(self) -> Iterator:
        for node in self.builder.nodes:
            yield node

    def _build(self):
        """
        Builds successor and predecessor.
        """
        self.nodes_ = {}
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
        Tells if a results is used by a subgraphs.
        """
        return name in self.used_

    def is_constant(self, name: str) -> bool:
        """
        Tells if a result is a constant.
        """
        return self.builder.is_constant(name)

    def get_constant(self, name: str) -> Any:
        """
        Returns the value for the constant `name`.
        """
        return self.builder.get_constant(name)

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
        return self.get_constant(node.input[input_index])

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
            op_type, inputs, outputs, domain=domain, name=name, **kwargs
        )
        if attributes:
            proto.attribute.extend(attributes)
        return proto

    def apply_match(self, match: MatchResult) -> List[NodeProto]:
        """
        Applies one match.
        Returns the new nodes.
        """
        idn = [id(n) for n in match.nodes]
        assert all(
            map(lambda i: i in self.nodes_, idn)
        ), f"One node in {idn} is not referenced"
        positions = {id(n): i for i, n in enumerate(self.iter_nodes())}
        assert all(
            map(lambda i: i in positions, idn)
        ), f"One node in {idn} is not referenced"
        insert_at = min(positions[i] for i in idn)
        new_nodes = match.apply(self, *match.nodes)
        removed = [positions[i] for i in idn]
        self.builder.insert_and_remove_nodes(insert_at, new_nodes, removed)
        return new_nodes

    def optimize(self, max_iter=-1):
        """
        Optimizes the based on the given list of patterns.

        :param max_iter: maximum number of iterations
        """
        if max_iter == -1:
            max_iter = len(self.builder.nodes)
        if self.verbose > 0:
            print(
                f"[GraphBuilderPatternOptimization.optimize] start with "
                f"{len(self.builder.nodes)} nodes and {len(self.patterns)} patterns"
            )
        last_it = 0
        for it in range(max_iter):
            if self.verbose > 0:
                print(
                    f"[GraphBuilderPatternOptimization.optimize] iteration {it}: "
                    f"{len(self.builder.nodes)} nodes"
                )
            found = False
            marked = set()
            matches = []
            for pattern in self.patterns:
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
                    matches.append(match)
            if self.verbose > 1:
                print(
                    f"[GraphBuilderPatternOptimization.optimize] applies {len(matches)} matches"
                )

            n_added = 0
            n_removed = 0
            for match in matches:
                if self.verbose > 2:
                    print(f"[GraphBuilderPatternOptimization.optimize] apply {match}")
                add = len(self.apply_match(match))
                rem = len(match.nodes)
                if self.verbose > 2:
                    print(
                        f"[GraphBuilderPatternOptimization.optimize] done {match}, - {rem} + {add} nodes"
                    )
                n_added += add
                n_removed -= rem
            if self.verbose > 1:
                print(
                    f"[GraphBuilderPatternOptimization.optimize] done all - {rem} + {add} nodes"
                )
            self._build()

            last_it = it + 1
            if not found:
                break

        if self.verbose > 0:
            print(
                f"[GraphBuilderPatternOptimization.optimize] done after {last_it} iterations with "
                f"{len(self.builder.nodes)} nodes"
            )
