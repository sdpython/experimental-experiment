import inspect
from typing import List, Optional, Sequence
import numpy as np
from onnx import NodeProto
from ...helpers import tensor_dtype_to_np_dtype
from ..patterns_api import MatchResult, PatternOptimization
from ..patterns.onnx_attention import FunctionAttentionPattern
from ..patterns.onnx_rotary import FunctionHalfRotaryEmbeddingPattern


class ContribRotaryEmbeddingPattern(PatternOptimization):
    """
    Very similar to
    :class:`experimental_experiment.xoptim.patterns.onnx_rotary.RotaryEmbeddingPattern`.
    """

    _operator_name = FunctionHalfRotaryEmbeddingPattern._operator_name
    _domain_name = FunctionHalfRotaryEmbeddingPattern._domain_name

    def __init__(self, verbose: int = 0, priority: int = 2):
        super().__init__(verbose, priority)
        self._info = []

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != self._operator_name or node.domain != self._domain_name:
            return self.none()
        if not g.has_shape(node.input[0]) or g.get_rank(node.input[0]) != 4:
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.has_shape(node.input[1]) or not g.has_shape(node.input[2]):
            return self.none(node, inspect.currentframe().f_lineno)
        shape_cos = g.get_shape(node.input[1])
        shape_sin = g.get_shape(node.input[2])
        if shape_cos != shape_sin:
            return self.none(node, inspect.currentframe().f_lineno)
        if len(shape_cos) != 4:
            return self.none(node, inspect.currentframe().f_lineno)
        if shape_cos[1] != 1 or shape_sin[1] != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        # if shape_cos[0] != 1 or shape_sin[0] != 1:
        # batch size is not 1 because position_ids was involved in the
        # computation of cos/sin caches.
        #    return self.none(node, inspect.currentframe().f_lineno)

        concat_cos = g.node_before(node.input[1])
        if concat_cos is None or concat_cos.op_type != "Concat" or concat_cos.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if concat_cos.input[0] != concat_cos.input[1]:
            return self.none(node, inspect.currentframe().f_lineno)
        if g.get_attribute(concat_cos, "axis").i != -1:
            return self.none(node, inspect.currentframe().f_lineno)

        concat_sin = g.node_before(node.input[2])
        if concat_sin is None or concat_sin.op_type != "Concat" or concat_sin.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if concat_sin.input[0] != concat_sin.input[1]:
            return self.none(node, inspect.currentframe().f_lineno)
        if g.get_attribute(concat_sin, "axis").i != -1:
            return self.none(node, inspect.currentframe().f_lineno)

        if g.is_used_more_than_once(node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)

        # If cos_cache[-1] + sin_cache[-1] == X.shape[-1],
        # then there is no split before.
        split_node = g.node_before(node.input[0])
        if split_node is None or split_node.op_type != "Split" or split_node.domain != "":
            if not g.has_shape(concat_cos.input[0]) or not g.has_shape(concat_sin.input[0]):
                return self.none(node, inspect.currentframe().f_lineno)
            cos_shape = g.get_shape(concat_cos.input[0])
            sin_shape = g.get_shape(concat_sin.input[0])
            input_shape = g.get_shape(node.input[0])
            if g.builder.evaluate_dimension_equality_with_constraints(
                input_shape[-1], cos_shape[-1], "+", sin_shape[-1]
            ):
                shape = g.get_shape(node.input[0])
                self._info.append((node.input[0], shape))
                if not isinstance(shape[1], int):
                    # Number of heads is not fixed"
                    return self.none(node, inspect.currentframe().f_lineno)
                # No split before, no concat after but there could be still position ids
                return self._match_last_part(
                    g,
                    concat_cos,
                    concat_sin,
                    None,
                    node,
                    None,
                    comment="path with no split before, no concat after",
                )
                # return MatchResult(
                #    self,
                #    [None, concat_cos, concat_sin, None, node, None],
                #    self.apply,
                #    comment="path with no split before, no concat after",
                # )

        if split_node is None or split_node.op_type != "Split" or split_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.has_shape(split_node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        shape_input = g.get_shape(split_node.input[0])
        if not isinstance(shape_input[1], int):
            # Not a fixed number of heads.
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant(split_node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        cst = g.get_computed_constant(split_node.input[1])
        if cst.shape != (2,):
            return self.none(node, inspect.currentframe().f_lineno)

        next_nodes = g.next_nodes(node.output[0])
        if len(next_nodes) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        concat_node = next_nodes[0]
        if concat_node.op_type != "Concat" or concat_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if split_node.output[1] != concat_node.input[1]:
            return self.none(node, inspect.currentframe().f_lineno)
        axis = g.get_attribute(concat_node, "axis").i
        if axis != -1:
            return self.none(node, inspect.currentframe().f_lineno)

        input_name = node.input[0] if split_node is None else split_node.input[0]
        shape = g.get_shape(input_name)
        self._info.append((input_name, shape))
        if not isinstance(shape[1], int):
            # Number of heads is not fixed"
            return self.none(node, inspect.currentframe().f_lineno)

        return self._match_last_part(
            g,
            concat_cos,
            concat_sin,
            split_node,
            node,
            concat_node,
            comment="path with split before, concat after",
        )

    def _match_last_part(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        concat_cos: NodeProto,
        concat_sin: NodeProto,
        split_node: Optional[NodeProto],
        node: NodeProto,
        concat_node: Optional[NodeProto],
        comment: str,
    ) -> Optional[MatchResult]:
        # Finally, we need to check if position_ids exists or it is given
        # a default value.
        common = self._find_common_ancestor(g, concat_cos, concat_sin)
        if common is not None and not common:
            # cos/sin are switched. The pattern cannot match.
            return self.none(node, inspect.currentframe().f_lineno)
        if (
            common
            and common[0].op_type == "Mul"
            and {"Sin", "Cos"} & set(n.op_type for n in common)
        ):
            # pattern FunctionCosSinCache has yet to be triggered first.
            return self.none(node, inspect.currentframe().f_lineno)

        if (
            common
            and common[0].op_type.startswith("CosSinCache")
            and common[0].domain == self._domain_name
        ):
            # Finally, we need to check if position_ids exists or if it is given
            # a default value.
            cos_sin = common[0]
            if not g.has_shape(cos_sin.input[0]) or not g.has_shape(cos_sin.input[1]):
                return self.none(node, inspect.currentframe().f_lineno)
            expand_node = g.node_before(cos_sin.input[1])
            if expand_node is None:
                return self.none(node, inspect.currentframe().f_lineno)
            shape_expand = g.builder.value_as_shape(expand_node.input[1])
            if shape_expand is None or len(shape_expand) != 3 or shape_expand[1:] != (1, 1):
                return self.none(node, inspect.currentframe().f_lineno)
            if not g.has_shape(expand_node.input[0]):
                return self.none(node, inspect.currentframe().f_lineno)
            wei_shape = g.get_shape(expand_node.input[0])
            if wei_shape[0] != 1:
                return self.none(node, inspect.currentframe().f_lineno)
            position_ids_shape = g.get_shape_renamed(cos_sin.input[0])
            weights_shape = g.get_shape_renamed(cos_sin.input[1])
            if (
                len(position_ids_shape) != 2
                or len(weights_shape) != 3
                or position_ids_shape[0] != weights_shape[0]
            ):
                return self.none(node, inspect.currentframe().f_lineno)

            # Then we need to add those nodes to the matched nodes.
            return MatchResult(
                self,
                [expand_node, concat_cos, concat_sin, split_node, node, concat_node, *common],
                self.apply,
                comment=f"{comment} / with CosSinCache",
            )

        return MatchResult(
            self,
            [None, concat_cos, concat_sin, split_node, node, concat_node],
            self.apply,
            insert_at=None if g.is_used_more_than_once(concat_cos.output[0]) else concat_node,
            comment=f"{comment} / without CosSinCache",
        )

    def _find_common_ancestor(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        concat_cos: NodeProto,
        concat_sin: NodeProto,
    ) -> Optional[List[NodeProto]]:
        anc_cos, anc_sin = concat_cos, concat_sin
        nodes = []
        for _it in range(5):
            cos_name, sin_name = anc_cos.input[0], anc_sin.input[0]
            anc_cos = g.node_before(cos_name)
            anc_sin = g.node_before(sin_name)
            if anc_cos is None or anc_sin is None:
                return None
            if (
                anc_cos.input[0] == anc_sin.input[0]
                and id(anc_cos) == id(anc_sin)
                and len(anc_cos.output) == 2
            ):
                if cos_name != anc_cos.output[0] or sin_name != anc_cos.output[1]:
                    # cos/sin were switched, the pattern should not match at all.
                    return []
                nodes.append(anc_cos)
                return nodes[::-1]
            nodes.extend([anc_cos, anc_sin])
        return None

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        expand_node: Optional[NodeProto],
        concat_cos: NodeProto,
        concat_sin: NodeProto,
        split_node: NodeProto,
        half_node: NodeProto,
        concat_node: NodeProto,
        *prefix_nodes: Sequence[NodeProto],
    ) -> List[NodeProto]:
        if split_node is None:
            rotary_dim = None
            shape = g.get_shape(half_node.input[0])
            main_input = half_node.input[0]
            main_output = half_node.output[0]
        else:
            cst = g.get_computed_constant(split_node.input[1])
            rotary_dim = int(cst[0])
            shape = g.get_shape(split_node.input[0])
            main_input = split_node.input[0]
            main_output = concat_node.output[0]

        assert isinstance(shape[1], int), (
            f"Number of heads is not fixed, shape("
            f"{split_node.input[0] if split_node is not None else half_node.input[0]}"
            f")={shape}, info={self._info}"
        )
        num_heads = shape[1]

        used_twice_cos = g.is_used_more_than_once(concat_cos.output[0])
        used_twice_sin = g.is_used_more_than_once(concat_sin.output[0])
        rotary_nodes = [expand_node, *prefix_nodes] if used_twice_cos or used_twice_sin else []
        if used_twice_cos:
            rotary_nodes.append(concat_cos)
        if used_twice_sin:
            rotary_nodes.append(concat_sin)

        batch_name = g.unique_name(f"{self.__class__.__name__}--{half_node.input[0]}--batch")
        zeroone = g.make_initializer(
            "", np.array([0, 1], dtype=np.int64), source=f"{self.__class__.__name__}.01"
        )
        one = g.make_initializer("", g.ONE, source=f"{self.__class__.__name__}.1")
        one_no_dim = g.make_initializer("", g.ONE_NO_DIM, source=f"{self.__class__.__name__}.1d")

        # position_ids
        zero_no_dim = g.make_initializer(
            "", g.ZERO_NO_DIM, source=f"{self.__class__.__name__}.0d"
        )

        added_nodes = []
        if prefix_nodes:
            assert expand_node is not None, "expand node is missing, pattern should not match"
            assert prefix_nodes[0].op_type.startswith(
                "CosSinCache"
            ), f"Unexpected first node {prefix_nodes[0]}"
            cos_sin = prefix_nodes[0]
            position_ids = cos_sin.input[0]
            (max_ids, max_ids_1, new_positions_ids, cos_out, sin_out, range_ids) = [
                g.unique_name(f"{self.__class__.__name__}--{position_ids}") for i in range(6)
            ]
            zero = g.make_initializer("", g.ZERO, source=f"{self.__class__.__name__}.0")
            added_nodes = [
                g._make_node("ReduceMax", [position_ids], [max_ids], keepdims=0),
                g._make_node("Add", [max_ids, one_no_dim], [max_ids_1]),
                g._make_node("Range", [zero_no_dim, max_ids_1, one_no_dim], [range_ids]),
                g._make_node("Unsqueeze", [range_ids, zero], [new_positions_ids]),
                g._make_node(
                    cos_sin.op_type,
                    [new_positions_ids, expand_node.input[0]],
                    [cos_out, sin_out],
                    domain=cos_sin.domain,
                ),
            ]
            cos_cur, sin_cur = cos_out, sin_out
            for i in range(1, len(prefix_nodes), 2):
                ncos, nsin = prefix_nodes[i : i + 2]
                if ncos.op_type == "Concat":
                    break
                rcos, rsin = [
                    g.unique_name(f"{self.__class__.__name__}--{position_ids}") for i in range(2)
                ]
                added_nodes.extend(
                    [
                        g._make_node(ncos.op_type, [cos_cur, *ncos.input[1:]], [rcos]),
                        g._make_node(ncos.op_type, [sin_cur, *nsin.input[1:]], [rsin]),
                    ]
                )
                if ncos.attribute:
                    added_nodes[-2].attribute.extend(ncos.attribute)
                if nsin.attribute:
                    added_nodes[-1].attribute.extend(nsin.attribute)
                cos_cur, sin_cur = rcos, rsin
            cos_input, sin_input = cos_cur, sin_cur
            range_nodes = []
        else:
            assert expand_node is None, f"Unexpected expand node {expand_node}"
            position_ids = g.unique_name(
                f"{self.__class__.__name__}--{half_node.input[0]}_position_ids"
            )
            seq_length = g.unique_name(
                f"{self.__class__.__name__}--{half_node.input[0]}--seq_length"
            )
            seq_length_squeezed = g.unique_name(
                f"{self.__class__.__name__}--{half_node.input[0]}--seqsq"
            )
            exp_shape = g.unique_name(f"{self.__class__.__name__}--{half_node.input[0]}_pshape")
            flat_pids = g.unique_name(
                f"{self.__class__.__name__}--{half_node.input[0]}_flat_pids"
            )
            cos_input, sin_input = concat_cos.input[0], concat_sin.input[0]
            range_nodes = [
                g._make_node("Shape", [main_input], [batch_name], start=0, end=1),
                g._make_node("Shape", [main_input], [seq_length], start=2, end=3),
                g._make_node("Squeeze", [seq_length], [seq_length_squeezed]),
                g._make_node(
                    "Range", [zero_no_dim, seq_length_squeezed, one_no_dim], [flat_pids]
                ),
                g._make_node("Concat", [batch_name, one], [exp_shape], axis=0),
                g._make_node("Expand", [flat_pids, exp_shape], [position_ids]),
            ]

        cos_name = g.unique_name(f"{self.__class__.__name__}--{half_node.input[1]}")
        sin_name = g.unique_name(f"{self.__class__.__name__}--{half_node.input[2]}")
        rotary_nodes.extend(
            [
                *added_nodes,
                g._make_node("Squeeze", [cos_input, zeroone], [cos_name]),
                g._make_node("Squeeze", [sin_input, zeroone], [sin_name]),
                *range_nodes,
            ]
        )
        rotary_nodes = [n for n in rotary_nodes if n]
        for node in rotary_nodes:
            if not node.name:
                node.name = g.builder.unique_node_name(
                    f"{self.__class__.__name__}--{half_node.name}"
                )

        kwargs = {} if rotary_dim is None else {"rotary_embedding_dim": rotary_dim}
        rotary_node = g.make_node(
            "RotaryEmbedding",
            [main_input, position_ids, cos_name, sin_name],
            [main_output],
            name=f"{self.__class__.__name__}--{half_node.name}",
            num_heads=num_heads,
            domain="com.microsoft",
            **kwargs,
        )
        rotary_nodes.append(rotary_node)
        return rotary_nodes


class ContribRotaryEmbedding3DPattern(PatternOptimization):
    """
    Extension to
    :class:`experimental_experiment.xoptim.patterns_ort.llm_optim.ContribRotaryEmbeddingPattern`,
    turn the operator into a 3D operator including the transpose.
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "RotaryEmbedding" or node.domain != "com.microsoft":
            return self.none()
        transpose = g.node_before(node.input[0])
        if transpose is None or transpose.op_type != "Transpose" or transpose.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        perm = tuple(g.get_attribute(transpose, "perm").ints)
        if perm != (0, 2, 1, 3):
            return self.none(node, inspect.currentframe().f_lineno)
        if g.is_used_more_than_once(node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, [transpose, node], self.apply, insert_at=node)

    def apply(
        self, g: "GraphBuilder", transpose: NodeProto, rotary: NodeProto  # noqa: F821
    ) -> List[NodeProto]:
        last_dim = g.unique_name(f"{transpose.input[0]}::Shape3")
        new_shape2 = g.unique_name(f"{transpose.input[0]}::Shape+1")
        new_shape = g.make_initializer(
            "", np.array([0, 0, -1], dtype=np.int64), source=f"{self.__class__.__name__}.00_1"
        )
        reshaped = g.unique_name(f"{transpose.input[0]}::3D")
        rot_name = g.unique_name(f"{transpose.input[0]}::3Dr")
        reshaped2 = g.unique_name(f"{transpose.input[0]}::4D")
        nodes = [
            g._make_node("Reshape", [transpose.input[0], new_shape], [reshaped]),
            g._make_node(
                rotary.op_type, [reshaped, *rotary.input[1:]], [rot_name], domain=rotary.domain
            ),
            g._make_node("Shape", [transpose.input[0]], [last_dim], start=3),
            g._make_node("Concat", [new_shape, last_dim], [new_shape2], axis=0),
            g._make_node("Reshape", [rot_name, new_shape2], [reshaped2]),
            g._make_node("Transpose", [reshaped2], [rotary.output[0]], perm=[0, 2, 1, 3]),
        ]
        if rotary.attribute:
            nodes[1].attribute.extend(rotary.attribute)
        for node in nodes:
            if not node.name:
                node.name = g.builder.unique_node_name(
                    f"{self.__class__.__name__}--{rotary.name}"
                )
        return nodes


class MultiHeadAttention3DPattern(PatternOptimization):
    """
    Merges multiple nodes into MultiHeadAttention. It assumes pattern
    :class:`experimental_experiment.xoptim.patterns.onnx_attention.FunctionAttentionPattern`
    was triggered before.
    """

    _prefix_operator_name = f"{FunctionAttentionPattern._operator_name}_to"

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if (
            not node.op_type.startswith(self._prefix_operator_name)
            or node.domain != FunctionAttentionPattern._domain_name
            or len(node.input) != 5
        ):
            return self.none()
        if not g.is_constant_scalar(node.input[4]):
            return self.none(node, inspect.currentframe().f_lineno)

        q_transpose = g.node_before(node.input[0])
        expected_perm = (0, 2, 1, 3)
        if (
            q_transpose is None
            or q_transpose.op_type != "Transpose"
            or tuple(g.get_attribute(q_transpose, "perm").ints) != expected_perm
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.has_shape(q_transpose.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        shape = g.get_shape(q_transpose.input[0])
        if not isinstance(shape[2], int):
            return self.none(node, inspect.currentframe().f_lineno)

        k_concat = g.node_before(node.input[1])
        if (
            k_concat is None
            or k_concat.op_type != "Concat"
            or g.get_attribute(k_concat, "axis").i != -2
            or len(k_concat.input) != 2
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        k_transpose = g.node_before(k_concat.input[1])
        if (
            k_transpose is None
            or k_transpose.op_type != "Transpose"
            or tuple(g.get_attribute(k_transpose, "perm").ints) != expected_perm
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        v_concat = g.node_before(node.input[2])
        if (
            v_concat is None
            or v_concat.op_type != "Concat"
            or g.get_attribute(v_concat, "axis").i != -2
            or len(v_concat.input) != 2
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        v_transpose = g.node_before(v_concat.input[1])
        if (
            v_transpose is None
            or v_transpose.op_type != "Transpose"
            or tuple(g.get_attribute(v_transpose, "perm").ints) != expected_perm
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        transposes = g.next_nodes(node.output[0])
        if len(transposes) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        transpose = transposes[0]
        if (
            transpose is None
            or transpose.op_type != "Transpose"
            or tuple(g.get_attribute(transpose, "perm").ints) != expected_perm
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        if (
            not g.has_shape(q_transpose.input[0])
            or g.get_rank(q_transpose.input[0]) != 4
            or not isinstance(g.get_shape(q_transpose.input[0])[-1], int)
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        for n in [q_transpose, k_transpose, v_transpose, node]:
            if g.is_used_more_than_once(n.output[0]):
                return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(
            self,
            [q_transpose, k_transpose, k_concat, v_transpose, v_concat, node, transpose],
            self.apply,
        )

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        q_transpose: NodeProto,
        k_transpose: NodeProto,
        k_concat: NodeProto,
        v_transpose: NodeProto,
        v_concat: NodeProto,
        attention: NodeProto,
        transpose: NodeProto,
    ) -> List[NodeProto]:
        query = q_transpose.input[0]
        keys = k_transpose.input[0]
        values = v_transpose.input[0]
        mask = attention.input[3]
        past_keys = k_concat.input[0]
        past_values = v_concat.input[0]
        num_heads = g.get_shape(query)[2]

        scale = float(g.get_constant_scalar(attention.input[4])) ** 2
        dtype = tensor_dtype_to_np_dtype(g.get_type(query))
        zero = g.make_initializer(
            "", np.array([0], dtype=dtype), source=f"{self.__class__.__name__}.0"
        )
        minfty = g.make_initializer(
            "", np.array([-np.inf], dtype=dtype), source=f"{self.__class__.__name__}._inf"
        )
        init_00_1 = g.make_initializer(
            "", np.array([0, 0, -1], dtype=np.int64), source=f"{self.__class__.__name__}.00_1"
        )
        last = g.get_shape(query)[-1]
        init_00_1l = g.make_initializer(
            "",
            np.array([0, 0, -1, last], dtype=np.int64),
            source=f"{self.__class__.__name__}.00_1l",
        )

        r_query = g.unique_name(f"{self.__class__.__name__}--{query}")
        r_keys = g.unique_name(f"{self.__class__.__name__}--{keys}")
        r_values = g.unique_name(f"{self.__class__.__name__}--{values}")
        attention_bias = g.unique_name(f"{self.__class__.__name__}--{mask}")
        r_output = g.unique_name(f"{self.__class__.__name__}--{transpose.output[0]}")

        nodes = [
            g._make_node("Reshape", [query, init_00_1], [r_query]),
            g._make_node("Reshape", [keys, init_00_1], [r_keys]),
            g._make_node("Reshape", [values, init_00_1], [r_values]),
            g._make_node("Where", [mask, zero, minfty], [attention_bias]),
            g._make_node(
                "MultiHeadAttention",
                [r_query, r_keys, r_values, "", "", attention_bias, past_keys, past_values],
                [r_output, k_concat.output[0], v_concat.output[0]],
                num_heads=num_heads,
                scale=scale,
                domain="com.microsoft",
            ),
            g._make_node("Reshape", [r_output, init_00_1l], [transpose.output[0]]),
        ]
        for node in nodes:
            if not node.name:
                node.name = g.builder.unique_node_name(
                    f"{self.__class__.__name__}--{attention.name}"
                )
        return nodes
