import inspect
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
from onnx import NodeProto
from ...xbuilder._shape_helper import (
    compatible_shapes,
    compatible_dimensions,
    is_static_shape,
    all_int,
)
from ..patterns_api import MatchResult, PatternOptimization


class MatMulAddPattern(PatternOptimization):
    """
    Replaces the sequence MatMul, Add into Gemm
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
        if g.get_rank(node.input[0]) < 2 or g.get_rank(node.input[1]) != 2:
            # If node.op_type is Gemm, this condition is useless,
            # if node.op_type is MatMul we reshape the matrix if
            # it the rank is > 2 and the last dimension known,
            # but then no bias should be allowed a Gemm does not support
            # broadcast.
            return self.none(node, inspect.currentframe().f_lineno)
        if g.get_rank(node.input[0]) > 2:
            sh1 = g.get_shape(node.input[0]) if g.has_shape(node.input[0]) else None
            sh2 = g.get_shape(node.input[1]) if g.has_shape(node.input[1]) else None
            if (sh1 is None or not isinstance(sh1[-1], int)) and (
                sh2 is None or not isinstance(sh2[0], int)
            ):
                # unkown k for the matrix multiplication
                return self.none(node, inspect.currentframe().f_lineno)

        next_nodes = g.next_nodes(node.output[0])
        if len(next_nodes) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        add_node = next_nodes[0]
        if add_node.op_type != "Add":
            return self.none(node, inspect.currentframe().f_lineno)

        # Gemm does not allow broadcasting.
        bias2 = add_node.input[0 if add_node.input[1] == node.output[0] else 1]
        if not g.has_shape(node.input[1]) or not g.has_shape(bias2):
            return self.none(node, inspect.currentframe().f_lineno)
        transB = (
            g.get_attributes_with_default(node, transB=0).get("transB", 0)
            if node.op_type == "Gemm"
            else 0
        )
        shape_2 = g.get_shape(node.input[1])
        last_dim = shape_2[-1 - transB]
        shape_bias = g.get_shape(bias2)
        if last_dim != shape_bias[-1]:
            return self.none(node, inspect.currentframe().f_lineno)
        if len(shape_bias) > 1:
            shape_node_out = (
                g.get_shape(node.output[0]) if g.has_shape(node.output[0]) else None
            )
            if shape_node_out is not None:
                if len(shape_node_out) != len(shape_bias):
                    return self.none(node, inspect.currentframe().f_lineno)
                elif shape_node_out != shape_bias:
                    return self.none(node, inspect.currentframe().f_lineno)
            elif min(shape_bias[:-1]) <= 1:
                return self.none(node, inspect.currentframe().f_lineno)

        if node.op_type == "MatMul" or len(node.input) == 2:
            return MatchResult(self, [node, add_node], self.apply, insert_at=add_node)

        bias = node.input[2]
        if (
            not g.has_shape(bias)
            or not g.has_shape(bias2)
            or g.get_shape(bias) != g.get_shape(bias2)
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [node, add_node], self.apply, insert_at=add_node)

    def _apply_matmmul(
        self,
        g: "GraphBuilder",  # noqa: F821
        matmul_node: NodeProto,
        add_node: NodeProto,
    ) -> List[NodeProto]:
        bias2 = add_node.input[0 if add_node.input[1] == matmul_node.output[0] else 1]
        if g.get_rank(matmul_node.input[0]) > 2:
            rk_bias = g.get_rank(bias2)
            # get k
            sh1 = (
                g.get_shape(matmul_node.input[0])
                if g.has_shape(matmul_node.input[0])
                else None
            )
            sh2 = (
                g.get_shape(matmul_node.input[1])
                if g.has_shape(matmul_node.input[1])
                else None
            )
            k = sh1[-1] if sh1 is not None and isinstance(sh1[-1], int) else sh2[0]
            new_shape = g.make_initializer(
                "",
                np.array([-1, k], dtype=np.int64),
                source="MatMulAddPattern.new_shape.1",
            )
            reshaped = g.unique_name(f"{self.__class__.__name__}--{matmul_node.input[0]}")
            reshape_node = g.make_node(
                "Reshape",
                [matmul_node.input[0], new_shape],
                [reshaped],
                name=f"{self.__class__.__name__}--{matmul_node.name}",
            )
            reshape_nodes = [reshape_node]
            if rk_bias > 2:
                if g.has_shape(bias2) and isinstance(g.get_shape(bias2)[-1], int):
                    new_shape_bias = g.make_initializer(
                        "",
                        np.array([-1, g.get_shape(bias2)[-1]], dtype=np.int64),
                        source="MatMulAddPattern.new_shape.3",
                    )
                else:
                    that_shape_bias = g.unique_name(
                        f"{self.__class__.__name__}--{matmul_node.input[0]}"
                    )
                    reshape_nodes.append(
                        g.make_node(
                            "Shape",
                            [bias2],
                            [that_shape_bias],
                            start=-1,
                            name=f"{self.__class__.__name__}--{matmul_node.name}",
                        )
                    )
                    new_shape_bias = g.unique_name(
                        f"{self.__class__.__name__}--{matmul_node.input[0]}"
                    )
                    minus1 = g.make_initializer(
                        "",
                        np.array([-1], dtype=np.int64),
                        source="MatMulAddPattern.new_shape.7",
                    )
                    reshape_nodes.append(
                        g.make_node(
                            "Concat",
                            [minus1, that_shape_bias],
                            [new_shape_bias],
                            axis=0,
                            name=f"{self.__class__.__name__}--{matmul_node.name}",
                        )
                    )

                reshaped_bias = g.unique_name(
                    f"{self.__class__.__name__}--{matmul_node.input[0]}"
                )
                reshape_bias_node = g.make_node(
                    "Reshape",
                    [bias2, new_shape_bias],
                    [reshaped_bias],
                    name=f"{self.__class__.__name__}--{matmul_node.name}",
                )
                reshape_nodes.append(reshape_bias_node)
                bias_gemm_name = reshaped_bias
            else:
                bias_gemm_name = bias2

            inputs = [reshaped, matmul_node.input[1]]
            unshaped = g.unique_name(f"{self.__class__.__name__}--{matmul_node.input[0]}")
            outputs = [unshaped]

            # last reshape
            if g.has_shape(matmul_node.input[0]) and all_int(
                g.get_shape(matmul_node.input[0])
            ):
                shape_back = g.make_initializer(
                    "",
                    np.array([*g.get_shape(matmul_node.input[0])[:-1], -1], dtype=np.int64),
                    source="MatMulAddPattern.new_shape.2",
                )
            else:
                # We extract the shape.
                that_shape = g.unique_name(
                    f"{self.__class__.__name__}--{matmul_node.input[0]}"
                )
                reshape_nodes.append(
                    g.make_node(
                        "Shape",
                        [matmul_node.input[0]],
                        [that_shape],
                        start=0,
                        end=-1,
                        name=f"{self.__class__.__name__}--{matmul_node.name}",
                    )
                )
                shape_back = g.unique_name(
                    f"{self.__class__.__name__}--{matmul_node.input[0]}"
                )
                minus1 = g.make_initializer(
                    "",
                    np.array([-1], dtype=np.int64),
                    source="MatMulAddPattern.new_shape.3",
                )
                reshape_nodes.append(
                    g.make_node(
                        "Concat",
                        [that_shape, minus1],
                        [shape_back],
                        axis=0,
                        name=f"{self.__class__.__name__}--{matmul_node.name}",
                    )
                )
            reshape_back = g.make_node(
                "Reshape",
                [unshaped, shape_back],
                add_node.output,
                name=f"{self.__class__.__name__}--{matmul_node.name}",
            )
        else:
            inputs = matmul_node.input
            outputs = add_node.output
            reshape_node = None
            bias_gemm_name = bias2

        new_node = g.make_node(
            "Gemm",
            [*inputs, bias_gemm_name],
            outputs,
            name=f"{self.__class__.__name__}--{matmul_node.name}",
            doc_string=matmul_node.doc_string,
        )
        if matmul_node.op_type == "Gemm":
            new_node.attribute.extend(matmul_node.attribute)
        if reshape_node:
            return [*reshape_nodes, new_node, reshape_back]
        return [new_node]

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        matmul_node: NodeProto,
        add_node: NodeProto,
    ) -> List[NodeProto]:

        if matmul_node.op_type == "MatMul" or len(matmul_node.input) == 2:
            return self._apply_matmmul(g, matmul_node, add_node)

        bias2 = add_node.input[0 if add_node.input[1] == matmul_node.output[0] else 1]
        # Two bias we need to add first.
        bias_all = g.unique_name(f"{self.__class__.__name__}--{matmul_node.input[2]}")
        new_add_node = g.make_node(
            "Add",
            [bias2, matmul_node.input[2]],
            [bias_all],
            name=f"{self.__class__.__name__}--{matmul_node.name}",
        )
        new_node = g.make_node(
            "Gemm",
            [*matmul_node.input[:2], bias_all],
            add_node.output,
            name=f"{self.__class__.__name__}--{matmul_node.name}",
            doc_string=matmul_node.doc_string,
        )
        new_node.attribute.extend(matmul_node.attribute)
        return [new_add_node, new_node]


class GemmTransposePattern(PatternOptimization):
    """
    Replaces Gemm (., constant) by Gemm(., constant', transB=1)
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Gemm" or node.domain != "":
            return self.none()
        if not g.is_constant(node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        if node.op_type == "Gemm":
            atts = g.get_attributes_with_default(node, transA=0, transB=0, beta=1.0)
            if atts.get("beta", 1) != 1:
                return self.none(node, inspect.currentframe().f_lineno)
            if atts.get("transB", 0) or atts.get("transA", 0):
                return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, [node], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node: NodeProto,
    ) -> List[NodeProto]:
        tr = g.unique_name(f"{self.__class__.__name__}--{node.input[1]}")
        return [
            g.make_node(
                "Transpose",
                [node.input[1]],
                [tr],
                perm=[1, 0],
                name=f"{self.__class__.__name__}--{node.name}",
                doc_string=node.doc_string,
            ),
            g.make_node(
                "Gemm",
                [node.input[0], tr, *node.input[2:]],
                node.output,
                transB=1,
                name=f"{self.__class__.__name__}--{node.name}",
                doc_string=node.doc_string,
            ),
        ]


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
        constraints: Dict[str, Set[Union[int, str]]],
    ) -> bool:
        # We cannot handle all the case.
        if is_static_shape(sh1) and is_static_shape(sh2):
            return np.prod(sh1) == np.prod(sh2)
        if sh1 == sh2:
            return True
        # The constraints should be applied here.
        return False

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if (node.op_type != "MatMul" or node.domain != "") and (
            node.op_type != "FusedMatMul" or node.domain != "com.microsoft"
        ):
            return self.none()

        if node.op_type == "FusedMatMul":
            tA = g.get_attribute(node, "transBatchA", exc=False)
            if tA is not None and tA.i != 0:
                return self.none(node, inspect.currentframe().f_lineno)
            tB = g.get_attribute(node, "transBatchB", exc=False)
            if tB is not None and tB.i != 0:
                return self.none(node, inspect.currentframe().f_lineno)

        if (
            not g.has_shape(node.output[0])
            or not g.has_shape(node.input[0])
            or not g.has_shape(node.input[1])
        ):
            # Shapes are missing. They should be populated as much as possible.
            return self.none(node, inspect.currentframe().f_lineno)

        next_nodes = g.next_nodes(node.output[0])
        if len(next_nodes) > 1 or (len(next_nodes) == 0 and not g.is_output(node.output[0])):
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
        shape_right_right = None if node_right is None else g.get_shape(node_right.input[0])

        shape_left = g.get_shape(node.input[0])
        shape_right = g.get_shape(node.input[1])

        if (
            shape_left_left is not None
            and not self.same_size(
                g, shape_left[-2:], shape_left_left[-2:], g.get_registered_constraints()
            )
        ) or (
            shape_right_right is not None
            and not self.same_size(
                g, shape_right[-2:], shape_right_right[-2:], g.get_registered_constraints()
            )
        ):
            # last dimension are the same
            return self.none(node, inspect.currentframe().f_lineno)

        the_shape_left = shape_left_left or shape_left
        the_shape_right = shape_right_right or shape_right
        if not is_static_shape(the_shape_left) or not is_static_shape(the_shape_right):
            return self.none(node, inspect.currentframe().f_lineno)
        if not self.same_size(
            g, the_shape_left[:-2], the_shape_right[:-2], g.get_registered_constraints()
        ):
            # first dimension are the same
            return self.none(node, inspect.currentframe().f_lineno)

        if next_node is not None:
            next_shape = g.get_shape(next_node.output[0])
            matmul_shape = the_shape_left[:-1] + (shape_right[-1],)
            if matmul_shape[-2:] != next_shape[-2:] and not self.same_size(
                g, matmul_shape[:-2], next_shape[:-2], g.get_registered_constraints()
            ):
                return self.none(node, inspect.currentframe().f_lineno)
            first_dims = {next_shape[:-2], the_shape_left[:-2], the_shape_right[:-2]}
            if len(first_dims) == 3:
                # All shapes are different. It is not worth it.
                return self.none(node, inspect.currentframe().f_lineno)

            if len(next_shape) != len(the_shape_left) and len(next_shape) != len(
                the_shape_right
            ):
                return self.none(node, inspect.currentframe().f_lineno)

            if matmul_shape[-1] != next_shape[-1]:
                # 1x9x64, 1x64x9 -> 1x9x9 -> 1x81
                # The last dimension changed.
                return self.none(node, inspect.currentframe().f_lineno)
        else:
            if len(the_shape_left) != len(the_shape_right):
                return self.none(node, inspect.currentframe().f_lineno)

        # The pattern is not handling the reshape after the matmul,
        # ReshapeReshapePattern will do it.
        nodes = [node_left, node_right, node, next_node]

        return MatchResult(self, nodes, self.apply)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node_left: Optional[NodeProto],
        node_right: Optional[NodeProto],
        node: NodeProto,
        next_node: Optional[NodeProto],
    ) -> List[NodeProto]:
        res = []

        shape_left_left = None if node_left is None else g.get_shape(node_left.input[0])
        shape_right_right = None if node_right is None else g.get_shape(node_right.input[0])

        shape_left = g.get_shape(node.input[0])
        shape_right = g.get_shape(node.input[1])

        the_shape_left = shape_left_left or shape_left
        the_shape_right = shape_right_right or shape_right

        # If the first dimensions are not the same, we may assume
        # the size is the same but a reshape is still needed.
        add_right, add_left = False, False
        one_more_reshape = the_shape_left[:-2] != the_shape_right[:-2]
        if one_more_reshape:
            expected_shape = g.get_shape(
                node.output[0] if next_node is None else next_node.output[0]
            )

            assert node_left is not None or node_right is not None, (
                f"Shapes are not consistent, one node Reshape should be there, "
                f"node.name={node.name!r}, "
                f"shape_left={shape_left}, shape_right={shape_right}, "
                f"the_shape_left={shape_left_left}, "
                f"the_shape_right={the_shape_right}, "
                f"node_left is None={node_left is None}, "
                f"node_right is None={node_right is None}, "
                f"next_node is None={next_node is None}, "
                f"expected_shape={expected_shape}"
            )
            if node_left is not None and the_shape_left[:-2] != expected_shape[:-2]:
                add_left = True
            elif node_right is not None and the_shape_right[:-2] != expected_shape[:-2]:
                add_right = True
            elif node_left is not None and node_right is not None:
                raise AssertionError(
                    f"Case still not implemented, shapes are not consistent, "
                    f"one node Reshape should be there, "
                    f"node.name={node.name!r}, "
                    f"shape_left={shape_left}, shape_right={shape_right}, "
                    f"the_shape_left={shape_left_left}, "
                    f"the_shape_right={the_shape_right}, "
                    f"node_left is None={node_left is None}, "
                    f"node_right is None={node_right is None}, "
                    f"next_node is None={next_node is None}, "
                    f"expected_shape={expected_shape}"
                )

        # node left
        if node_left is None:
            expected_shape = the_shape_right[:-2] + shape_left[-2:]
            if the_shape_left != expected_shape:
                shape_name = g.make_initializer(
                    "",
                    np.array(expected_shape, dtype=np.int64),
                    source="MatMulReshape2Of3Pattern.apply.shape.1",
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
                    "",
                    np.array(expected_shape, dtype=np.int64),
                    source="MatMulReshape2Of3Pattern.apply.shape.2",
                )
                right_name = g.unique_name(f"{self.__class__.__name__}L_{node.input[0]}")
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
            assert not add_right and not add_left, (
                f"add_right={add_right}, add_left={add_left} "
                f"are not implemented yet in this case."
            )
            # Reshape is needed.
            previous_shape = shape_left[:-1] + (shape_right[-1],)
            new_shape = the_shape_left[:-1] + (the_shape_right[-1],)
            if previous_shape != new_shape:
                new_name = g.unique_name(f"{self.__class__.__name__}L_{node.output[0]}")
                previous_shape_name = g.make_initializer(
                    "",
                    np.array(previous_shape, dtype=np.int64),
                    source="MatMulReshape2Of3Pattern.shape.3",
                )
                mm = g.make_node(
                    node.op_type,
                    [left_name, right_name],
                    [new_name],
                    name=f"{self.__class__.__name__}--{node.name}",
                    domain=node.domain,
                )
                if node.attribute:
                    mm.attribute.extend(node.attribute)
                res.extend(
                    [
                        mm,
                        g.make_node(
                            "Reshape",
                            [new_name, previous_shape_name],
                            [node.output[0]],
                            name=f"{self.__class__.__name__}--{node.name}",
                        ),
                    ]
                )
            else:
                mm = g.make_node(
                    node.op_type,
                    [left_name, right_name],
                    [node.output[0]],
                    name=f"{self.__class__.__name__}--{node.name}",
                    domain=node.domain,
                )
                if node.attribute:
                    mm.attribute.extend(node.attribute)
                res.append(mm)
        else:
            if add_left:
                new_left_name = g.unique_name(f"{self.__class__.__name__}AL_{left_name}")
                new_sh = (
                    g.get_shape(next_node.output[0])[:-2] + g.get_shape(node.input[0])[-2:]
                )
                sh = g.make_initializer(
                    "",
                    np.array(new_sh, dtype=np.int64),
                    source="MatMulReshape2Of3Pattern.apply.sh.1",
                )
                add = g.make_node(
                    "Reshape",
                    [left_name, sh],
                    [new_left_name],
                    name=f"{self.__class__.__name__}--AL--{node.name}",
                )
                res.append(add)
                left_name = new_left_name
            if add_right:
                new_right_name = g.unique_name(f"{self.__class__.__name__}AR_{right_name}")
                new_sh = (
                    g.get_shape(next_node.output[0])[:-2] + g.get_shape(node.input[1])[-2:]
                )
                sh = g.make_initializer(
                    "",
                    np.array(new_sh, dtype=np.int64),
                    source="MatMulReshape2Of3Pattern.apply.sh.2",
                )
                add = g.make_node(
                    "Reshape",
                    [right_name, sh],
                    [new_right_name],
                    name=f"{self.__class__.__name__}--AR--{node.name}",
                )
                res.append(add)
                right_name = new_right_name

            main_node = g.make_node(
                node.op_type,
                [left_name, right_name],
                [next_node.output[0]],
                name=f"{self.__class__.__name__}--{node.name}",
                domain=node.domain,
            )
            if node.attribute:
                main_node.attribute.extend(node.attribute)
            res.append(main_node)

            if g.is_used_more_than_once(node.output[0]):
                previous_shape = shape_left[:-1] + (shape_right[-1],)
                previous_shape_name = g.make_initializer(
                    "",
                    np.array(previous_shape, dtype=np.int64),
                    source="MatMulReshape2Of3Pattern.apply.shape.4",
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


class MulMulMatMulPattern(PatternOptimization):
    """
    Replaces ``MatMul(a*c, b*d)``
    where c and d are constant scalar
    by ``MatMul(a,b) * (c,d)``.
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "MatMul" or node.domain != "":
            return self.none()

        node_before = [g.node_before(i) for i in node.input]
        if None in node_before:
            return self.none(node, inspect.currentframe().f_lineno)
        types = set(_.op_type for _ in node_before)
        if types != {"Mul"}:
            return self.none(node, inspect.currentframe().f_lineno)
        cst = [i for i in [*node_before[0].input, *node_before[1].input] if g.is_constant(i)]
        if len(cst) != 2 or not all(g.is_constant_scalar(c) for c in cst):
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [*node_before, node], self.apply)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        mul1: NodeProto,
        mul2: NodeProto,
        node: NodeProto,
    ) -> List[NodeProto]:
        cst = [i for i in [*mul1.input, *mul2.input] if g.is_constant(i)]
        not_cst = [i for i in [*mul1.input, *mul2.input] if i not in cst]
        assert len(cst) == 2, f"impossible cst={cst!r}"
        assert len(not_cst) == 2, f"impossible not_cst={not_cst!r}"
        cs = [g.get_computed_constant(c) for c in cst]
        c = (cs[0] * cs[1]).astype(cs[0].dtype)

        ccc = g.make_initializer("", c, source="MulMulMatMulPattern.apply.ccc")
        mul_name = g.unique_name(f"{self.__class__.__name__}_{node.output[0]}")

        return [
            g.make_node(
                "MatMul",
                not_cst,
                [mul_name],
                name=f"{self.__class__.__name__}--{node.name}-1",
            ),
            g.make_node(
                "Mul",
                [mul_name, ccc],
                node.output,
                name=f"{self.__class__.__name__}--{node.name}-2",
            ),
        ]


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
        shape_left = tuple(int(i) for i in g.get_computed_constant(node_before_left.input[1]))
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
                f"[ReshapeMatMulReshapePattern] compatible shapes: "
                f"mshape_left={mshape_left} "
                f"shape_left={shape_left} | mshape_left={mshape_right} "
                f"shape_left={shape_right}"
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
            (n if n is not None and n.op_type == "Transpose" and n.domain == "" else None)
            for n in nodes_before
        ]
        if len([_ for _ in ns if _ is not None]) == 0:
            return self.none(node, inspect.currentframe().f_lineno)

        if g.has_processor("CUDA"):
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

        if len([_ for _ in ns if _ is not None]) == 0:
            return self.none(node, inspect.currentframe().f_lineno)

        # At this stage, one or two inputs are transposed before being used.
        # MatMul or Gemm are operating on 2D tensors.
        nodes = [*ns, node]

        if node.op_type == "Gemm":
            if nodes[1] is not None:  # nodes_before_right
                atts = g.get_attributes_with_default(node, transA=0, transB=0)
                if atts.get("transB", 0) != atts.get("transA", 0) and g.is_constant(
                    node.input[1]
                ):
                    # it is better to do constant folding rather than changing transB
                    return self.none(node, inspect.currentframe().f_lineno)
            if nodes[0] is not None:  # nodes_before_left
                atts = g.get_attributes_with_default(node, transA=0, transB=0)
                if atts.get("transB", 0) != atts.get("transA", 0) and g.is_constant(
                    node.input[0]
                ):
                    # it is better to do constant folding rather than changing transB
                    return self.none(node, inspect.currentframe().f_lineno)

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
            (node.input[1] if node_before_right is None else node_before_right.input[0]),
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
            node_before_right.output[0]
        ):
            # This is not efficient on CUDA.
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
        shape_name = g.make_initializer(
            "",
            np.array(shape, dtype=np.int64),
            source="TransposeReshapeMatMulPattern.apply.shape_name",
        )

        if node_right is None:
            # left side

            perm = list(range(g.get_rank(node.input[0])))
            perm[-2], perm[-1] = perm[-1], perm[-2]
            left_name = g.unique_name(f"{self.__class__.__name__}L_{node_left_tr.input[0]}")
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
            right_name = g.unique_name(f"{self.__class__.__name__}L_{node_right_tr.input[0]}")
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


class SwitchReshapeActivationPattern(PatternOptimization):
    """
    Swiches Gelu and Reshape after a Gemm or a MatMul.
    Gelu can also be Gelu, Exp, Elu, Relu, Tan,
    Tanh, Cos, Cosh, Sin, Sinh, Erf, LeakyRelu, PRelu,
    Selu, Softmax, Softplus.
    Reshape can also be Transpose.
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
        left_first: bool = True,
    ) -> Optional[MatchResult]:
        if (
            node.op_type
            not in {
                "Cos",
                "Cosh",
                "Elu",
                "Erf",
                "Exp",
                "Gelu",
                "LeakyRelu",
                "PRelu",
                "Relu",
                "Selu",
                "Sin",
                "Sinh",
                "Softmax",
                "Softplus",
                "Tan",
                "Tanh",
            }
            or node.domain != ""
        ):
            return self.none()
        if g.is_used_more_than_once(node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        before = g.node_before(node.input[0])
        if g.is_used_more_than_once(before.input[0]):
            return self.none(before, inspect.currentframe().f_lineno)
        if before.op_type not in {"Reshape", "Transpose"} or node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        before_before = g.node_before(before.input[0])
        if before_before.op_type not in {"Gemm", "MatMul"} or before_before.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(
            self,
            [before_before, before, node],
            self.apply,
            insert_at=before if before.op_type == "Reshape" else before_before,
        )

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        mm_node: NodeProto,
        tr_node: NodeProto,
        f_node: NodeProto,
    ) -> List[NodeProto]:
        name1 = g.unique_name(f"{self.__class__.__name__}L_{mm_node.output[0]}")
        name2 = g.unique_name(f"{self.__class__.__name__}L_{tr_node.output[0]}")
        nodes = [
            g.make_node(
                mm_node.op_type,
                mm_node.input,
                [name1],
                domain=mm_node.domain,
                name=f"{self.__class__.__name__}--{mm_node.name}",
            ),
            g.make_node(
                f_node.op_type,
                [name1],
                [name2],
                domain=f_node.domain,
                name=f"{self.__class__.__name__}--{f_node.name}",
            ),
            g.make_node(
                tr_node.op_type,
                [name2, *tr_node.input[1:]],
                f_node.output,
                domain=tr_node.domain,
                name=f"{self.__class__.__name__}--{tr_node.name}",
            ),
        ]
        nodes[1].attribute.extend(f_node.attribute)
        nodes[2].attribute.extend(tr_node.attribute)
        return nodes
