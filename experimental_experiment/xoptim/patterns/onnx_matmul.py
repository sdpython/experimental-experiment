import inspect
from typing import List, Optional, Tuple
import numpy as np
from onnx import NodeProto
from ...xbuilder.shape_helper import (
    compatible_shapes,
    compatible_dimensions,
    is_static_shape,
)
from ..patterns_api import MatchResult, PatternOptimization


class MatMulReshape2Of3Pattern(PatternOptimization):
    """
    Replaces the reshapes around a matmul
    It can be 3 or 2 out of 3.
    It is similar to
    :class:`experimental_experiment.xoptim.patterns.onnx_reshape.Reshape2Of3Pattern`.
    """

    def same_size(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821,
        sh1: Tuple[int, ...],
        sh2: Tuple[int, ...],
    ) -> bool:
        # We cannot handle all the case.
        if is_static_shape(sh1) and is_static_shape(sh2):
            return np.prod(sh1) == np.prod(sh2)
        return sh1 == sh2

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "MatMul" or node.domain != "":
            return self.none()

        if (
            not g.has_shape(node.output[0])
            or not g.has_shape(node.input[0])
            or not g.has_shape(node.input[1])
        ):
            # Shapes are missing. They should be populated as much as possible.
            return self.none(node, inspect.currentframe().f_lineno)

        next_nodes = g.next_nodes(node.output[0])
        if len(next_nodes) > 1 or (
            len(next_nodes) == 0 and not g.is_output(node.output[0])
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        next_node = None if len(next_nodes) == 0 else next_nodes[0]
        node_left = g.node_before(node.input[0])
        node_right = g.node_before(node.input[1])

        type_left = None if node_left is None else node_left.op_type
        type_right = None if node_right is None else node_right.op_type
        type_out = None if next_node is None else next_node.op_type

        types = [type_left, type_right, type_out]
        n_reshape = len([_ for _ in types if _ == "Reshape"])
        if n_reshape < 2:
            return self.none(node, inspect.currentframe().f_lineno)

        if node_left is not None and node_left.op_type != "Reshape":
            node_left = None
        if node_right is not None and node_right.op_type != "Reshape":
            node_right = None
        if next_node is not None and next_node.op_type != "Reshape":
            next_node = None

        shape_left_left = None if node_left is None else g.get_shape(node_left.input[0])
        shape_right_right = (
            None if node_right is None else g.get_shape(node_right.input[0])
        )

        shape_left = g.get_shape(node.input[0])
        shape_right = g.get_shape(node.input[1])

        if (
            shape_left_left is not None
            and not self.same_size(g, shape_left[-2:], shape_left_left[-2:])
        ) or (
            shape_right_right is not None
            and not self.same_size(g, shape_right[-2:], shape_right_right[-2:])
        ):
            # last dimension are the same
            return self.none(node, inspect.currentframe().f_lineno)

        the_shape_left = shape_left_left or shape_left
        the_shape_right = shape_right_right or shape_right
        if not self.same_size(g, the_shape_left[:-2], the_shape_right[:-2]):
            # first dimension are the same
            return self.none(node, inspect.currentframe().f_lineno)

        if next_node is not None:
            next_shape = g.get_shape(next_node.output[0])
            matmul_shape = the_shape_left[:-1] + (shape_right[-1],)
            if matmul_shape[-2:] != next_shape[-2:] or not self.same_size(
                g, matmul_shape[:-2], next_shape[:-2]
            ):
                return self.none(node, inspect.currentframe().f_lineno)
            first_dims = set(
                [next_shape[:-2], the_shape_left[:-2], the_shape_right[:-2]]
            )
            if len(first_dims) == 3:
                # All shapes are different. It is not worth it.
                return self.none(node, inspect.currentframe().f_lineno)

        # The pattern is not handling the reshape after the matmul,
        # ReshapeReshapePattern will do it.

        nodes = [node_left, node_right, node, next_node]

        return MatchResult(self, nodes, self.apply)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node_left: NodeProto,
        node_right: NodeProto,
        node: NodeProto,
        next_node: NodeProto,
    ) -> List[NodeProto]:

        res = []

        shape_left_left = None if node_left is None else g.get_shape(node_left.input[0])
        shape_right_right = (
            None if node_right is None else g.get_shape(node_right.input[0])
        )

        shape_left = g.get_shape(node.input[0])
        shape_right = g.get_shape(node.input[1])

        the_shape_left = shape_left_left or shape_left
        the_shape_right = shape_right_right or shape_right

        # node left
        if node_left is None:
            expected_shape = the_shape_right[:-2] + shape_left[-2:]
            if the_shape_left != expected_shape:
                shape_name = g.make_initializer(
                    "", np.array(expected_shape, dtype=np.int64)
                )
                left_name = g.unique_name(f"{self.__class__.__name__}L_{node.input[0]}")
                res.append(
                    g.make_node(
                        "Reshape",
                        [node.input[0], shape_name],
                        [left_name],
                        name=f"{self.__class__.__name__}--{node.name}",
                    )
                )
            else:
                left_name = node.input[0]
        elif g.is_used_more_than_once(node_left.output[0]):
            res.append(node_left)
            left_name = node_left.input[0]
        else:
            left_name = node_left.input[0]

        # node right
        if node_right is None:
            expected_shape = the_shape_left[:-2] + shape_right[-2:]
            if the_shape_right != expected_shape:
                shape_name = g.make_initializer(
                    "", np.array(expected_shape, dtype=np.int64)
                )
                right_name = g.unique_name(
                    f"{self.__class__.__name__}L_{node.input[0]}"
                )
                res.append(
                    g.make_node(
                        "Reshape",
                        [node.input[1], shape_name],
                        [right_name],
                        name=f"{self.__class__.__name__}--{node.name}",
                    )
                )
            else:
                right_name = node.input[1]
        elif g.is_used_more_than_once(node_right.output[0]):
            res.append(node_right)
            right_name = node_right.input[0]
        else:
            right_name = node_right.input[0]

        if next_node is None:
            # Reshape is needed.
            previous_shape = shape_left[:-1] + (shape_right[-1],)
            new_shape = the_shape_left[:-1] + (the_shape_right[-1],)
            if previous_shape != new_shape:
                new_name = g.unique_name(f"{self.__class__.__name__}L_{node.output[0]}")
                previous_shape_name = g.make_initializer(
                    "", np.array(previous_shape, dtype=np.int64)
                )
                res.extend(
                    [
                        g.make_node(
                            node.op_type,
                            [left_name, right_name],
                            [new_name],
                            name=f"{self.__class__.__name__}--{node.name}",
                        ),
                        g.make_node(
                            "Reshape",
                            [new_name, previous_shape_name],
                            [node.output[0]],
                            name=f"{self.__class__.__name__}--{node.name}",
                        ),
                    ]
                )
            else:
                res.appens(
                    g.make_node(
                        node.op_type,
                        [left_name, right_name],
                        [node.output[0]],
                        name=f"{self.__class__.__name__}--{node.name}",
                    )
                )
        else:
            main_node = g.make_node(
                node.op_type,
                [left_name, right_name],
                [next_node.output[0]],
                name=f"{self.__class__.__name__}--{node.name}",
            )
            res.append(main_node)

            if g.is_used_more_than_once(node.output[0]):
                previous_shape = shape_left[:-1] + (shape_right[-1],)
                previous_shape_name = g.make_initializer(
                    "", np.array(previous_shape, dtype=np.int64)
                )
                res.append(
                    g.make_node(
                        "Reshape",
                        [main_node.output[0], previous_shape_name],
                        [node.output[0]],
                        name=f"{self.__class__.__name__}--{node.name}",
                    )
                )

        return res


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
            return self.none()
        if g.is_used_more_than_once(node.output[0]):
            return self.none(node, inspect.currentframe().f_lineno)

        next_nodes = g.next_nodes(node.output[0])
        if len(next_nodes) == 0:
            return self.none(node, inspect.currentframe().f_lineno)
        next_node = next_nodes[0]
        if next_node.op_type != "Reshape" or node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        node_before_left = g.node_before(node.input[0])
        node_before_right = g.node_before(node.input[1])
        if node_before_left is None or node_before_right is None:
            return self.none(node, inspect.currentframe().f_lineno)
        if (
            node_before_left.op_type != "Reshape"
            or node_before_left.domain != ""
            or node_before_right.op_type != "Reshape"
            or node_before_right.domain != ""
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        # condition on shapes
        if not g.is_constant(node_before_left.input[1]):
            return
        shape_left = tuple(
            int(i) for i in g.get_computed_constant(node_before_left.input[1])
        )
        if not g.is_constant(node_before_right.input[1]):
            return
        shape_right = tuple(
            int(i) for i in g.get_computed_constant(node_before_right.input[1])
        )
        if not g.is_constant(next_node.input[1]):
            return
        shape_final = tuple(int(i) for i in g.get_computed_constant(next_node.input[1]))
        if len(shape_final) < 4:
            return self.none(node, inspect.currentframe().f_lineno)
        ndim = len(shape_final)
        if len(shape_left) != 3 or len(shape_right) != 3:
            return self.none(node, inspect.currentframe().f_lineno)

        mshape_left = g.get_shape(node_before_left.input[0])
        mshape_right = g.get_shape(node_before_right.input[0])
        if len(mshape_left) != ndim or len(mshape_right) != ndim:
            return self.none(node, inspect.currentframe().f_lineno)
        if (
            not compatible_shapes(mshape_left[-2:], shape_left[-2:])
            or not compatible_shapes(mshape_right[-2:], shape_right[-2:])
            or not compatible_dimensions(
                mshape_left[-1], shape_left[-1], mshape_right[-2], shape_right[-2]
            )
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        # At this stage, both Reshape before MatMul reduces the rank by 1
        # without changing the two last dimensions
        # and the Reshape after restores it. They can safely be removed.
        if g.verbose > 3:
            print(
                f"[ReshapeMatMulReshapePattern] compatible shapes: mshape_left={mshape_left} "
                f"shape_left={shape_left} | mshape_left={mshape_right} shape_left={shape_right}"
            )

        return MatchResult(
            self,
            [node_before_left, node_before_right, node, next_node],
            self.apply,
        )

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node_before_left: NodeProto,
        node_before_right: NodeProto,
        node: NodeProto,
        next_node: NodeProto,
    ) -> List[NodeProto]:
        res = []
        if g.is_used_more_than_once(node_before_left.output[0]):
            res.append(node_before_left)
        if g.is_used_more_than_once(node_before_right.output[0]):
            res.append(node_before_right)
        new_node = g.make_node(
            "MatMul",
            [node_before_left.input[0], node_before_right.input[0]],
            next_node.output,
            name=f"{self.__class__.__name__}--{node.name}",
            doc_string=next_node.doc_string,
        )
        res.append(new_node)
        return res


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
            return self.none()
        if not g.has_rank(node.input[0]) or not g.has_rank(node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        if g.get_rank(node.input[0]) != 2 or g.get_rank(node.input[1]) != 2:
            return self.none(node, inspect.currentframe().f_lineno)

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
            return self.none(node, inspect.currentframe().f_lineno)

        if g.processor == "CUDA":
            nns = []
            for n in ns:
                if n is None:
                    nns.append(n)
                    continue
                if g.is_used_more_than_once(n.output[0]):
                    nns.append(None)
                    continue
                nns.append(n)
            if len([_ for _ in ns if _ is not None]) == 0:
                return self.none(node, inspect.currentframe().f_lineno)
            ns = nns

        for n in ns:
            if n is None:
                continue
            perm = tuple(g.get_attribute(n, "perm").ints)
            if perm != (1, 0):
                # unexpected transpose
                return self.none(node, inspect.currentframe().f_lineno)

        # At this stage, one or two inputs are transposed before being used.
        # MatMul or Gemm are operating on 2D tensors.
        nodes = [*ns, node]

        return MatchResult(self, nodes, self.apply)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node_before_left: Optional[NodeProto],
        node_before_right: Optional[NodeProto],
        node: NodeProto,
    ) -> List[NodeProto]:

        inputs = [
            (node.input[0] if node_before_left is None else node_before_left.input[0]),
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
            doc_string=node.doc_string,
        )
        new_node.attribute.extend(keep)
        res = [new_node]
        if node_before_left is not None and g.is_used_more_than_once(
            node_before_left.output[0]
        ):
            # This is not efficient on CUDA.
            res.append(node_before_left)
        if node_before_right is not None and g.is_used_more_than_once(
            # This is not efficient on CUDA.
            node_before_right.output[0]
        ):
            res.append(node_before_right)
        return res


class TransposeReshapeMatMulPattern(PatternOptimization):
    """
    Replaces the sequence Transpose, Reshape, Matmul into
    Reshape, Transpose, Matmul if possible. Another optimizer
    will optimizes this sequence by using Gemm or better.
    """

    def check_transpose_node(self, g: "GraphBuilder", name: str) -> bool:  # noqa: F821
        if g.is_used_more_than_once(name):
            return False
        node = g.node_before(name)
        if node is None or node.op_type != "Reshape":
            return False
        if g.is_used_more_than_once(node.input[0]):
            return False
        node_node = g.node_before(node.input[0])
        if node_node is None or node_node.op_type != "Transpose":
            return False
        perm = tuple(g.get_attribute(node_node, "perm").ints)
        id_perm = tuple(range(len(perm)))
        if perm[:-2] != id_perm[:-2] or (perm[-1], perm[-2]) != id_perm[-2:]:
            return False
        return True

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
        left_first: bool = True,
    ) -> Optional[MatchResult]:
        if node.op_type != "MatMul" or node.domain != "":
            return self.none()

        left = self.check_transpose_node(g, node.input[0])
        right = self.check_transpose_node(g, node.input[1])
        if left and left_first:
            # even right is ok, it will be handled by another call to the optimizer.
            side = "left"
        elif right:
            side = "right"
        else:
            return self.none(node, inspect.currentframe().f_lineno)

        if side == "left":
            node_left = g.node_before(node.input[0])
            node_left_tr = g.node_before(node_left.input[0])
            node_right = None
            node_right_tr = None
            shape_name = node_left.input[1]
        else:
            node_left = None
            node_left_tr = None
            node_right = g.node_before(node.input[1])
            node_right_tr = g.node_before(node_right.input[0])
            shape_name = node_right.input[1]

        if not g.is_constant(shape_name):
            if left_first and right:
                return self.match(g, node, matched, left_first=False)
            return self.none(node, inspect.currentframe().f_lineno)

        shape_before = g.get_shape((node_left or node_right).input[0])
        shape_after = g.get_shape((node_left or node_right).output[0])
        if shape_before[-2:] != shape_after[-2:]:
            # the two last dimension are not modified by the reshape
            if left_first and right:
                return self.match(g, node, matched, left_first=False)
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(
            self,
            [node, node_left, node_left_tr, node_right, node_right_tr],
            self.apply,
            insert_at=node,
        )

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node: NodeProto,
        node_left: Optional[NodeProto],
        node_left_tr: Optional[NodeProto],
        node_right: Optional[NodeProto],
        node_right_tr: Optional[NodeProto],
    ) -> List[NodeProto]:

        shape = list(g.get_computed_constant((node_left or node_right).input[1]))
        shape[-2], shape[-1] = shape[-1], shape[-2]
        shape_name = g.make_initializer("", np.array(shape, dtype=np.int64))

        if node_right is None:
            # left side

            perm = list(range(g.get_rank(node.input[0])))
            perm[-2], perm[-1] = perm[-1], perm[-2]
            left_name = g.unique_name(
                f"{self.__class__.__name__}L_{node_left_tr.input[0]}"
            )
            res = [
                g.make_node(
                    "Reshape",
                    [node_left_tr.input[0], shape_name],
                    [left_name],
                    name=f"{self.__class__.__name__}--{node.name}",
                ),
                g.make_node(
                    "Transpose",
                    [left_name],
                    [node.input[0]],
                    perm=tuple(perm),
                    name=f"{self.__class__.__name__}--{node.name}",
                ),
                node,
            ]

        else:
            # right side
            perm = list(range(g.get_rank(node.input[1])))
            perm[-2], perm[-1] = perm[-1], perm[-2]
            right_name = g.unique_name(
                f"{self.__class__.__name__}L_{node_right_tr.input[0]}"
            )
            res = [
                g.make_node(
                    "Reshape",
                    [node_right_tr.input[0], shape_name],
                    [right_name],
                    name=f"{self.__class__.__name__}--{node.name}",
                ),
                g.make_node(
                    "Transpose",
                    [right_name],
                    [node.input[1]],
                    perm=tuple(perm),
                    name=f"{self.__class__.__name__}--{node.name}",
                ),
                node,
            ]

        return res
