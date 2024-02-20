from typing import Callable, Iterator, List, Optional, Union
import numpy as np
from onnx import NodeProto


class MatchResult:
    """
    Returns matching results.

    :param pattern: object detecting the pattern
    :param nodes: nodes to be replaced
    :param apply: node computing the replacements
    """

    def __init__(
        self, pattern: "PatternOptimization", nodes: List[NodeProto], apply: Callable
    ):
        self.pattern = pattern
        self.nodes = nodes
        self.apply = apply

    def __str__(self) -> str:
        types = [n.op_type for n in self.nodes]
        return f"MatchResult: {self.pattern} replaces {types}"


class PatternOptimization:
    """
    Defines an optimization pattern.

    :param description:
    """

    def __init__(self):
        pass

    def __str__(self) -> str:
        return self.__class__.__name__

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
        raise NotImplementedError(
            f"This function must be overloaded in class {self.__class__}."
        )


class CastPattern(PatternOptimization):
    """
    Checks that a Cast is really needeD.
    """

    def __init__(self):
        PatternOptimization.__init__(self)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Cast":
            return None
        if not g.has_type(node.input[0]):
            itype = g.try_infer_type(node.input[0])
            if itype == 0:
                return None
        else:
            itype = g.get_type(node.input[0])
        att = g.get_attribute(node, "to")
        if att.i != itype:
            return None

        def apply(g: "GraphBuilder", node: NodeProto) -> List[NodeProto]:  # noqa: F821
            new_node = g.make_node(
                "Identity", node.input, node.output, name=self.__class__.__name__
            )
            return [new_node]

        return MatchResult(self, [node], apply)


class UnsqueezeUnsqueezePattern(PatternOptimization):
    """
    Replaces the sequence Unsqueeze, Unsqueeze by Unsqueeze.
    """

    def __init__(self):
        PatternOptimization.__init__(self)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Unsqueeze":
            return None
        if g.is_used_by_subgraph(node.output[0]):
            return None
        next_nodes = g.next_nodes(node.output[0])
        if len(next_nodes) != 1:
            return None
        next_node = next_nodes[0]
        if next_node.op_type != "Unsqueeze":
            return None

        def apply(
            g: "GraphBuilder", node: NodeProto, next_node: NodeProto  # noqa: F821
        ) -> List[NodeProto]:
            axis1 = g.get_constant_or_attribute(node, "axis", 1)
            axis2 = g.get_constant_or_attribute(next_node, "axis", 1)
            new_axis = g.make_initializer("", np.hstack([axis1, axis2]))
            new_node = g.make_node(
                "Unsqueeze",
                [node.input[0], new_axis],
                next_node.output,
                name=self.__class__.__name__,
            )
            return [new_node]

        return MatchResult(self, [node, next_node], apply)


def get_default_patterns() -> List[PatternOptimization]:
    """
    Returns a default list of optimization patters.
    It is equal to the following list.

    .. runpython::
        :showcode:

        import pprint
        from experimental_experiment.torch_exp.optimization_patterns import get_default_patterns
        pprint.pprint(get_default_patterns())
    """
    return [CastPattern(), UnsqueezeUnsqueezePattern()]


def get_pattern(obj: Union[PatternOptimization, str]) -> PatternOptimization:
    """
    Returns an optimization pattern based on its name.
    """
    if isinstance(obj, PatternOptimization):
        return obj
    if obj == "UnsqueezeUnsqueeze":
        return UnsqueezeUnsqueezePattern()
    if obj == "Cast":
        return CastPattern()
    raise RuntimeError(f"Unable to find pattern for {obj!r}.")
