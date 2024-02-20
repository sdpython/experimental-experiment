from typing import Callable, Iterator, List, Optional, Union
import numpy as np
from onnx import NodeProto, helper as oh


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
    Checks that a Cast is really needed.
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Cast" or node.domain != "":
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


class LayerNormalizationPattern(PatternOptimization):
    """
    Replaces the sequence Pow(., 2) + ReduceMean + Add + Sqrt + Reciprocal + Mul by LayerNormalization
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "ReduceMean" or node.domain != "":
            return None
        axis = g.get_constant_or_attribute(node, "axes", input_index=1)
        print("**", axis)
        node_pow = g.node_before(node.input[0])
        if node_pow.op_type != "Pow" or node.domain != "":
            return None
        exponent = g.get_constant(node_pow.input[1])
        if not g.is_value_equal_to_number(exponent, 2):
            return None
        node_add = g.next_node(node.output[0])
        if node_add.op_type != "Add" or node_add.domain != "":
            return None
        epsilon = g.get_constant(node_add.input[1])
        if not g.is_real_number(epsilon):
            return None
        node_sqrt = g.next_node(node_add.output[0])
        if node_sqrt.op_type != "Sqrt" or node_sqrt.domain != "":
            return None
        node_reciprocal = g.next_node(node_sqrt.output[0])
        if node_reciprocal.op_type != "Reciprocal" or node_reciprocal.domain != "":
            return None
        node_mul = g.next_node(node_reciprocal.output[0])
        if node_mul.op_type != "Mul" or node_mul.domain != "":
            return None

        if (
            g.n_successors(node_pow.output[0]) != 1
            or g.n_successors(node.output[0]) != 1
            or g.n_successors(node_add.output[0]) != 1
            or g.n_successors(node_sqrt.output[0]) != 1
        ):
            # intermediate results are used
            return None

        def apply(
            g: "GraphBuilder",  # noqa: F821
            node_pow: NodeProto,
            node_mean: NodeProto,
            node_add: NodeProto,
            node_sqrt: NodeProto,
            node_reciprocal: NodeProto,
            node_mul: NodeProto,
        ) -> List[NodeProto]:
            itype = g.get_type(node.input[0])
            dtype = oh.tensor_dtype_to_np_dtype(itype)
            axis = g.get_constant_or_attribute(node, "axes", input_index=1)
            epsilon = g.get_constant(node_add.input[1])
            scale = g.make_initializer("", np.array([1], dtype=dtype))
            bias = g.make_initializer("", np.array([0], dtype=dtype))
            new_node = g.make_node(
                "LayerNormalization",
                [node_pow.input[0], scale, bias],
                [node_mul.output[0], "", node_reciprocal],
                axis=axis,
                epsilon=epsilon,
            )
            return [new_node]

        nodes = [node_pow, node, node_add, node_sqrt, node_reciprocal, node_mul]
        return MatchResult(self, nodes, apply)


class ReshapeMatMulReshapePattern(PatternOptimization):
    """
    Replaces the sequence Reshape, Matmul, Reshape by Matmul.
    """

    def __init__(self):
        PatternOptimization.__init__(self)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "MatMul" or node.domain != "":
            return None
        if g.is_used_by_subgraph(node.output[0]):
            return None
        if g.is_used_by_subgraph(node.input[0]):
            return None
        if g.is_used_by_subgraph(node.input[1]):
            return None

        next_nodes = g.next_nodes(node.output[0])
        if len(next_nodes) == 0:
            return None
        next_node = next_nodes[0]
        if next_node.op_type != "Reshape" or node.domain != "":
            return None

        node_before_left = g.node_before(node.input[0])
        node_before_right = g.node_before(node.input[1])
        if (
            node_before_left.op_type != "Reshape"
            or node_before_left.domain != ""
            or node_before_right.op_type != "Reshape"
            or node_before_right.domain != ""
        ):
            return None

        # condition on shapes
        shape_left = tuple(g.get_constant(node_before_left.input[1]))
        shape_right = tuple(g.get_constant(node_before_right.input[1]))
        shape_final = tuple(g.get_constant(next_node.input[1]))
        if len(shape_final) < 4:
            return None
        ndim = len(shape_final)
        if len(shape_left) != 3 or len(shape_right) != 3:
            return None

        mshape_left = g.get_shape(node_before_left.input[0])
        mshape_right = g.get_shape(node_before_right.input[0])
        if len(mshape_left) != ndim or len(mshape_right) != ndim:
            return None
        if mshape_left[-2:] != shape_left[-2:] or mshape_right[-2:] != shape_right[-2:]:
            return None

        # At this stage, both Reshape before MatMul reduces the rank by 1
        # without changing the two last dimensions
        # and the Reshape after restores it. They can safely be removed.

        def apply(
            g: "GraphBuilder",  # noqa: F821
            node_before_left: NodeProto,
            node_before_right: NodeProto,
            node: NodeProto,
            next_node: NodeProto,
        ) -> List[NodeProto]:
            new_node = g.make_node(
                "MatMul",
                [node_before_left.input[0], node_before_right.input[0]],
                next_node.output,
                name=self.__class__.__name__,
            )
            return [new_node]

        return MatchResult(
            self, [node_before_left, node_before_right, node, next_node], apply
        )


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
        if node.op_type != "Unsqueeze" or node.domain != "":
            return None
        if g.is_used_by_subgraph(node.output[0]):
            return None
        next_nodes = g.next_nodes(node.output[0])
        if len(next_nodes) != 1:
            return None
        next_node = next_nodes[0]
        if next_node.op_type != "Unsqueeze" or node.domain != "":
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
    return [
        CastPattern(),
        LayerNormalizationPattern(),
        ReshapeMatMulReshapePattern(),
        UnsqueezeUnsqueezePattern(),
    ]


def get_pattern(obj: Union[PatternOptimization, str]) -> PatternOptimization:
    """
    Returns an optimization pattern based on its name.
    """
    if isinstance(obj, PatternOptimization):
        return obj

    mapping = {
        v.__class__.__name__.replace("Pattern", ""): v for v in get_default_patterns()
    }
    if obj in mapping:
        return mapping[obj]
    raise RuntimeError(f"Unable to find pattern for {obj!r}.")
