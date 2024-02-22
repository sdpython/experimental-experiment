from typing import Callable, Iterator, List, Optional, Tuple, Union
import numpy as np
from onnx import NodeProto
from .annotations import all_int, compatible_shapes, compatible_dimensions


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
                "Identity",
                node.input,
                node.output,
                name=f"{self.__class__.__name__}--{node.name}",
            )
            return [new_node]

        return MatchResult(self, [node], apply)


class ExpandPattern(PatternOptimization):
    """
    Checks that a Expand is really needed.
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Expand" or node.domain != "":
            return None
        if not g.has_shape(node.input[0]):
            return None
        shape = g.get_shape(node.input[0])
        if not all_int(shape):
            return None
        new_shape = tuple(g.get_constant(node.input[1]))
        if shape != new_shape:
            return

        def apply(g: "GraphBuilder", node: NodeProto) -> List[NodeProto]:  # noqa: F821
            new_node = g.make_node(
                "Identity",
                node.input,
                node.output,
                name=f"{self.__class__.__name__}--{node.name}",
            )
            return [new_node]

        return MatchResult(self, [node], apply)


class ReshapeMatMulReshapePattern(PatternOptimization):
    """
    Replaces the sequence Reshape, Matmul, Reshape by Matmul.
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "MatMul" or node.domain != "":
            return None
        if g.is_used_more_than_once(node.output[0]):
            return None

        next_nodes = g.next_nodes(node.output[0])
        if len(next_nodes) == 0:
            return None
        next_node = next_nodes[0]
        if next_node.op_type != "Reshape" or node.domain != "":
            return None

        node_before_left = g.node_before(node.input[0])
        node_before_right = g.node_before(node.input[1])
        if node_before_left is None or node_before_right is None:
            return None
        if (
            node_before_left.op_type != "Reshape"
            or node_before_left.domain != ""
            or node_before_right.op_type != "Reshape"
            or node_before_right.domain != ""
        ):
            return None

        # condition on shapes
        shape_left = tuple(int(i) for i in g.get_constant(node_before_left.input[1]))
        shape_right = tuple(int(i) for i in g.get_constant(node_before_right.input[1]))
        shape_final = tuple(int(i) for i in g.get_constant(next_node.input[1]))
        if len(shape_final) < 4:
            return None
        ndim = len(shape_final)
        if len(shape_left) != 3 or len(shape_right) != 3:
            return None

        mshape_left = g.get_shape(node_before_left.input[0])
        mshape_right = g.get_shape(node_before_right.input[0])
        if len(mshape_left) != ndim or len(mshape_right) != ndim:
            return None
        if (
            not compatible_shapes(mshape_left[-2:], shape_left[-2:])
            or not compatible_shapes(mshape_right[-2:], shape_right[-2:])
            or not compatible_dimensions(
                mshape_left[-1], shape_left[-1], mshape_right[-2], shape_right[-2]
            )
        ):
            return None

        # At this stage, both Reshape before MatMul reduces the rank by 1
        # without changing the two last dimensions
        # and the Reshape after restores it. They can safely be removed.
        if g.verbose > 3:
            print(
                f"[ReshapeMatMulReshapePattern] compatible shapes: mshape_left={mshape_left} "
                f"shape_left={shape_left} | mshape_left={mshape_right} shape_left={shape_right}"
            )

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
                name=f"{self.__class__.__name__}--{node.name}",
            )
            res = [new_node]
            if g.is_used_more_than_once(node_before_left.output[0]):
                res.append(node_before_left)
            if g.is_used_more_than_once(node_before_right.output[0]):
                res.append(node_before_right)
            return res

        return MatchResult(
            self,
            [node_before_left, node_before_right, node, next_node],
            apply,
            insert_at=node,
        )


class ReshapeReshapePattern(PatternOptimization):
    """
    Replaces the sequence Reshape, Reshape by Reshape.
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Reshape" or node.domain != "":
            return None
        if g.is_used_more_than_once(node.output[0]):
            return None
        next_nodes = g.next_nodes(node.output[0])
        if len(next_nodes) != 1:
            return None
        next_node = next_nodes[0]
        if next_node.op_type != "Reshape" or node.domain != "":
            return None
        if next_node.input[0] != node.output[0]:
            return None

        def apply(
            g: "GraphBuilder", node: NodeProto, next_node: NodeProto  # noqa: F821
        ) -> List[NodeProto]:
            new_node = g.make_node(
                "Reshape",
                [node.input[0], next_node.input[1]],
                next_node.output,
                name=f"{self.__class__.__name__}--{node.name}",
            )
            return [new_node]

        return MatchResult(self, [node, next_node], apply)


class TransposeMatMulPattern(PatternOptimization):
    """
    Replaces the sequence Transpose, Matmul or Gemm into Gemm
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type not in {"MatMul", "Gemm"} or node.domain != "":
            return None
        if not g.has_rank(node.input[0]) or not g.has_rank(node.input[1]):
            return None
        if g.get_rank(node.input[0]) != 2 or g.get_rank(node.input[1]) != 2:
            return None

        nodes_before = [g.node_before(node.input[0]), g.node_before(node.input[1])]
        ns = [
            (
                n
                if n is not None and n.op_type == "Transpose" and n.domain == ""
                else None
            )
            for n in nodes_before
        ]
        if len([_ for _ in ns if _ is not None]) == 0:
            return None

        for n in ns:
            if n is None:
                continue
            perm = tuple(g.get_attribute(n, "perm").ints)
            if perm != (1, 0):
                # unexpected transpose
                return None

        # At this stage, one or two inputs are transposed before being used.
        # MatMul or Gemm are operating on 2D tensors.
        nodes = [*ns, node]

        def apply(
            g: "GraphBuilder",  # noqa: F821
            node_before_left: Optional[NodeProto],
            node_before_right: Optional[NodeProto],
            nodes: NodeProto,
        ) -> List[NodeProto]:

            inputs = [
                (
                    node.input[0]
                    if node_before_left is None
                    else node_before_left.input[0]
                ),
                (
                    node.input[1]
                    if node_before_right is None
                    else node_before_right.input[0]
                ),
                *node.input[2:],
            ]

            transA = 0 if node_before_left is None else 1
            transB = 0 if node_before_right is None else 1
            keep = []
            for att in node.attribute:
                if att.name in {"alpha", "beta"}:
                    keep.append(att)
                elif att.name == "transA":
                    transA = (att.i + transA) % 2
                elif att.name == "transB":
                    transB = (att.i + transB) % 2
                else:
                    raise NotImplementedError(
                        f"Unexpected attribute {att.name!r}={att} for node={node}"
                    )

            new_node = g.make_node(
                "Gemm",
                inputs,
                node.output,
                name=f"{self.__class__.__name__}--{node.name}",
                transA=transA,
                transB=transB,
            )
            new_node.attribute.extend(keep)
            res = [new_node]
            if node_before_left is not None and g.is_used_more_than_once(
                node_before_left.output[0]
            ):
                res.append(node_before_left)
            if node_before_right is not None and g.is_used_more_than_once(
                node_before_right.output[0]
            ):
                res.append(node_before_right)
            return res

        return MatchResult(self, nodes, apply, insert_at=node)


class TransposeTransposePattern(PatternOptimization):
    """
    Removes two consecutive transpose if the second one put the tensor in origin shape.
    """

    @classmethod
    def apply_transpose(cls, perm: Tuple[int, ...], on: List[int]) -> List[int]:
        assert len(perm) == len(on), "length mismatch"
        res = [None for i in on]
        for i, p in enumerate(perm):
            res[i] = p
        return res

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Transpose" or node.domain != "":
            return None
        next_nodes = g.next_nodes(node.output[0])
        next_node = None
        for n in next_nodes:
            if n.op_type == "Transpose":
                next_node = n
        if next_node is None:
            return None

        # Three consecutive transpose are not expected but let's continue
        # as if it could be possible.
        nodes = [node, next_node]
        perms = [tuple(g.get_attribute(n, "perm").ints) for n in nodes]
        lens = [len(p) for p in perms]
        assert min(lens) == max(lens), (
            f"Consecutive Transpose should apply on tensors with "
            f"the same rank but perms={perms}."
        )
        on = list(range(lens[0]))
        first = on.copy()
        for p in perms:
            self.apply_transpose(p, on)
        if on != first:
            return None

        def apply(
            g: "GraphBuilder", node: NodeProto, next_node: NodeProto  # noqa: F821
        ) -> List[NodeProto]:
            new_nodes = [
                g.make_node(
                    "Identity",
                    [node.input[0]],
                    next_node.output,
                    name=f"{self.__class__.__name__}--{node.name}",
                )
            ]
            if g.is_used_more_than_once(node.output[0]):
                new_nodes.append(node)
            return new_nodes

        return MatchResult(self, [node, next_node], apply)


class UnsqueezeUnsqueezePattern(PatternOptimization):
    """
    Replaces the sequence Unsqueeze, Unsqueeze by Unsqueeze.
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Unsqueeze" or node.domain != "":
            return None
        if g.is_used_more_than_once(node.output[0]):
            return None
        next_nodes = g.next_nodes(node.output[0])
        if len(next_nodes) != 1:
            return None
        next_node = next_nodes[0]
        if next_node.op_type != "Unsqueeze" or node.domain != "":
            return None
        if next_node.input[0] != node.output[0]:
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
                name=f"{self.__class__.__name__}--{node.name}",
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
        ExpandPattern(),
        ReshapeMatMulReshapePattern(),
        ReshapeReshapePattern(),
        TransposeMatMulPattern(),
        TransposeTransposePattern(),
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
