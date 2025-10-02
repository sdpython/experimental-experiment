import inspect
from typing import List, Optional, Sequence
import numpy as np
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization
from ..patterns.onnx_rotary import FunctionHalfRotaryEmbeddingPattern


class ContribRotaryEmbeddingPattern(PatternOptimization):
    """
    Very similar to
    :class:`experimental_experimental.xoptim.patterns.onnx_rotary.RotaryEmbeddingPattern`.
    """

    _operator_name = FunctionHalfRotaryEmbeddingPattern._operator_name
    _domain_name = FunctionHalfRotaryEmbeddingPattern._domain_name

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != self._operator_name or node.domain != self._domain_name:
            # Not ready in opset 23.
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

        assert isinstance(shape[1], int), f"Number of heads is not fixed, shape(X)={shape}"
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
