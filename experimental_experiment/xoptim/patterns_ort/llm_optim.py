import inspect
from typing import List, Optional, Sequence, Tuple
import numpy as np
from onnx import NodeProto, TensorProto
from ...helpers import tensor_dtype_to_np_dtype
from ..patterns_api import MatchResult, PatternOptimization
from ..patterns.onnx_attention import FunctionAttentionPattern, FunctionAttentionGQAPattern
from ..patterns.onnx_rotary import FunctionHalfRotaryEmbeddingPattern


class ContribRotaryEmbeddingPattern(PatternOptimization):
    """
    Very similar to
    :class:`experimental_experiment.xoptim.patterns.onnx_rotary.RotaryEmbeddingPattern`.

    Model with nodes to be fused:

    .. gdot::
        :script: DOT-SECTION
        :process:

        from experimental_experiment.doc import to_dot
        import numpy as np
        import ml_dtypes
        import onnx
        import onnx.helper as oh
        import onnx.numpy_helper as onh

        opset_imports = [
            oh.make_opsetid("", 20),
            oh.make_opsetid("intermediate", 1),
            oh.make_opsetid("com.microsoft", 1),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape=("a", 2, "c", "2*e"))
        )
        inputs.append(
            oh.make_tensor_value_info("m1", onnx.TensorProto.FLOAT, shape=(1, 1, "c", "e"))
        )
        inputs.append(
            oh.make_tensor_value_info("m2", onnx.TensorProto.FLOAT, shape=(1, 1, "c", "e"))
        )
        nodes.append(oh.make_node("Concat", ["m2", "m2"], ["m2x2"], axis=-1))
        nodes.append(oh.make_node("Concat", ["m1", "m1"], ["m1x2"], axis=-1))
        nodes.append(
            oh.make_node(
                "HalfRotaryEmbedding", ["X", "m2x2", "m1x2"], ["Y"], domain="intermediate"
            )
        )
        outputs.append(
            oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape=("a", "b", "c", "2*e"))
        )
        graph = oh.make_graph(
            nodes,
            "pattern",
            inputs,
            outputs,
            initializers,
            sparse_initializer=sparse_initializers,
        )
        model = oh.make_model(graph, functions=functions, opset_imports=opset_imports)

        print("DOT-SECTION", to_dot(model))

    Outcome of the fusion:

    .. gdot::
        :script: DOT-SECTION
        :process:

        from experimental_experiment.doc import to_dot
        import numpy as np
        import ml_dtypes
        import onnx
        import onnx.helper as oh
        import onnx.numpy_helper as onh

        opset_imports = [
            oh.make_opsetid("", 20),
            oh.make_opsetid("intermediate", 1),
            oh.make_opsetid("com.microsoft", 1),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape=("a", 2, "c", "2*e"))
        )
        inputs.append(
            oh.make_tensor_value_info("m1", onnx.TensorProto.FLOAT, shape=(1, 1, "c", "e"))
        )
        inputs.append(
            oh.make_tensor_value_info("m2", onnx.TensorProto.FLOAT, shape=(1, 1, "c", "e"))
        )
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["init7_s2_0_1"],
                value=onh.from_array(np.array([0, 1], dtype=np.int64), name="value"),
            )
        )
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["init7_s_0"],
                value=onh.from_array(np.array(0, dtype=np.int64), name="value"),
            )
        )
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["init7_s_1"],
                value=onh.from_array(np.array(1, dtype=np.int64), name="value"),
            )
        )
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["init7_s1_1"],
                value=onh.from_array(np.array([1], dtype=np.int64), name="value"),
            )
        )
        nodes.append(
            oh.make_node(
                "Squeeze", ["m2", "init7_s2_0_1"], ["ContribRotaryEmbeddingPattern--m2x2"]
            )
        )
        nodes.append(
            oh.make_node(
                "Squeeze", ["m1", "init7_s2_0_1"], ["ContribRotaryEmbeddingPattern--m1x2"]
            )
        )
        nodes.append(
            oh.make_node(
                "Shape", ["X"], ["ContribRotaryEmbeddingPattern--X--batch"], end=1, start=0
            )
        )
        nodes.append(
            oh.make_node(
                "Shape", ["X"], ["ContribRotaryEmbeddingPattern--X--seq_length"], end=3, start=2
            )
        )
        nodes.append(
            oh.make_node(
                "Squeeze",
                ["ContribRotaryEmbeddingPattern--X--seq_length"],
                ["ContribRotaryEmbeddingPattern--X--seqsq"],
            )
        )
        nodes.append(
            oh.make_node(
                "Range",
                ["init7_s_0", "ContribRotaryEmbeddingPattern--X--seqsq", "init7_s_1"],
                ["ContribRotaryEmbeddingPattern--X_flat_pids"],
            )
        )
        nodes.append(
            oh.make_node(
                "Concat",
                ["ContribRotaryEmbeddingPattern--X--batch", "init7_s1_1"],
                ["ContribRotaryEmbeddingPattern--X_pshape"],
                axis=0,
            )
        )
        nodes.append(
            oh.make_node(
                "Expand",
                [
                    "ContribRotaryEmbeddingPattern--X_flat_pids",
                    "ContribRotaryEmbeddingPattern--X_pshape",
                ],
                ["ContribRotaryEmbeddingPattern--X_position_ids"],
            )
        )
        nodes.append(
            oh.make_node(
                "RotaryEmbedding",
                [
                    "X",
                    "ContribRotaryEmbeddingPattern--X_position_ids",
                    "ContribRotaryEmbeddingPattern--m2x2",
                    "ContribRotaryEmbeddingPattern--m1x2",
                ],
                ["Y"],
                domain="com.microsoft",
                num_heads=2,
            )
        )
        outputs.append(
            oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape=("a", "b", "c", "2*e"))
        )
        graph = oh.make_graph(
            nodes,
            "pattern",
            inputs,
            outputs,
            initializers,
            sparse_initializer=sparse_initializers,
        )
        model = oh.make_model(graph, functions=functions, opset_imports=opset_imports)

        print("DOT-SECTION", to_dot(model))
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
        #    return self.none(node, inspect.currentframe().f_lineno)
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
            and not common[0].op_type.startswith("CosSinCacheWithRange")
            and common[0].domain == self._domain_name
        ):
            # Finally, we need to check if position_ids exists or if it is given
            # a default value.
            cos_sin = common[0]
            if not g.has_shape(cos_sin.input[0]) or not g.has_shape(cos_sin.input[1]):
                return self.none(node, inspect.currentframe().f_lineno)
            expand_node = g.node_before(cos_sin.input[1])
            if expand_node is not None:
                # position_ids is expanded first
                shape_expand = g.builder.value_as_shape(expand_node.input[1])
                if shape_expand is None or len(shape_expand) != 3 or shape_expand[1:] != (1, 1):
                    # maybe position_ids is not given
                    return self.none(
                        node,
                        inspect.currentframe().f_lineno,
                        msg=lambda: (
                            f"op_type={expand_node.op_type!r} name={expand_node.input[1]!r} "
                            f"shape_expand={shape_expand}"
                        ),
                    )
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
                or (position_ids_shape[0] != weights_shape[0] and weights_shape[0] != 1)
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
                return self.none(concat_cos, inspect.currentframe().f_lineno)
            if anc_cos.input[0] == anc_sin.input[0] and id(anc_cos) == id(anc_sin):
                if len(anc_cos.output) == 2:
                    if (
                        cos_name != anc_cos.output[0]
                        or sin_name != anc_cos.output[1]
                        or not anc_cos.op_type.startswith("CosSinCache")
                    ):
                        # cos/sin were switched, the pattern should not match at all.
                        return []
                    nodes.append(anc_cos)
                    return nodes[::-1]
                # cos/sin are not produced the usual way (CosSinCache)
                return []
            nodes.extend([anc_cos, anc_sin])
        return self.none(concat_cos, inspect.currentframe().f_lineno)

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
            assert prefix_nodes[0].op_type.startswith(
                "CosSinCache"
            ), f"Unexpected first node {prefix_nodes[0]}"
            cos_sin = prefix_nodes[0]
            position_ids = cos_sin.input[0]
            max_ids, max_ids_1, new_positions_ids, cos_out, sin_out, range_ids = [
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
                    [
                        new_positions_ids,
                        expand_node.input[0] if expand_node is not None else cos_sin.input[1],
                    ],
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
                f"{self.__class__.__name__}--{half_node.input[0]}_seq_length"
            )
            seq_length_squeezed = g.unique_name(
                f"{self.__class__.__name__}--{half_node.input[0]}_seqsq"
            )
            exp_shape = g.unique_name(f"{self.__class__.__name__}_{half_node.input[0]}_pshape")
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

    Model with nodes to be fused:

    .. gdot::
        :script: DOT-SECTION
        :process:

        from experimental_experiment.doc import to_dot
        import numpy as np
        import ml_dtypes
        import onnx
        import onnx.helper as oh
        import onnx.numpy_helper as onh

        opset_imports = [
            oh.make_opsetid("", 20),
            oh.make_opsetid("intermediate", 1),
            oh.make_opsetid("com.microsoft", 1),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info(
                "ContribRotaryEmbeddingPattern--m2x2",
                onnx.TensorProto.FLOAT,
                shape=("NEWDIM_range", 2),
            )
        )
        inputs.append(
            oh.make_tensor_value_info("position_ids", onnx.TensorProto.INT64, shape=("a", "e"))
        )
        inputs.append(
            oh.make_tensor_value_info(
                "ContribRotaryEmbeddingPattern--m1x2",
                onnx.TensorProto.FLOAT,
                shape=("NEWDIM_range", 2),
            )
        )
        inputs.append(
            oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape=("a", "c", 2, "d"))
        )
        nodes.append(oh.make_node("Transpose", ["X"], ["Xt"], perm=[0, 2, 1, 3]))
        nodes.append(
            oh.make_node(
                "RotaryEmbedding",
                [
                    "Xt",
                    "position_ids",
                    "ContribRotaryEmbeddingPattern--m1x2",
                    "ContribRotaryEmbeddingPattern--m2x2",
                ],
                ["Y"],
                domain="com.microsoft",
                num_heads=2,
                rotary_embedding_dim=4,
            )
        )
        outputs.append(
            oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape=("a", "b", "c", "d"))
        )
        graph = oh.make_graph(
            nodes,
            "pattern",
            inputs,
            outputs,
            initializers,
            sparse_initializer=sparse_initializers,
        )
        model = oh.make_model(graph, functions=functions, opset_imports=opset_imports)

        print("DOT-SECTION", to_dot(model))

    Outcome of the fusion:

    .. gdot::
        :script: DOT-SECTION
        :process:

        from experimental_experiment.doc import to_dot
        import numpy as np
        import ml_dtypes
        import onnx
        import onnx.helper as oh
        import onnx.numpy_helper as onh

        opset_imports = [
            oh.make_opsetid("", 20),
            oh.make_opsetid("intermediate", 1),
            oh.make_opsetid("com.microsoft", 1),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info(
                "ContribRotaryEmbeddingPattern--m2x2",
                onnx.TensorProto.FLOAT,
                shape=("NEWDIM_range", 2),
            )
        )
        inputs.append(
            oh.make_tensor_value_info("position_ids", onnx.TensorProto.INT64, shape=("a", "e"))
        )
        inputs.append(
            oh.make_tensor_value_info(
                "ContribRotaryEmbeddingPattern--m1x2",
                onnx.TensorProto.FLOAT,
                shape=("NEWDIM_range", 2),
            )
        )
        inputs.append(
            oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape=("a", "c", 2, "d"))
        )
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["init7_s3_0_0_-1"],
                value=onh.from_array(np.array([0, 0, -1], dtype=np.int64), name="value"),
            )
        )
        nodes.append(oh.make_node("Reshape", ["X", "init7_s3_0_0_-1"], ["X::3D"]))
        nodes.append(
            oh.make_node(
                "RotaryEmbedding",
                [
                    "X::3D",
                    "position_ids",
                    "ContribRotaryEmbeddingPattern--m1x2",
                    "ContribRotaryEmbeddingPattern--m2x2",
                ],
                ["X::3Dr"],
                domain="com.microsoft",
                num_heads=2,
                rotary_embedding_dim=4,
            )
        )
        nodes.append(oh.make_node("Shape", ["X"], ["X::Shape3"], start=3))
        nodes.append(
            oh.make_node(
                "Concat", ["init7_s3_0_0_-1", "X::Shape3"], ["X::Shape+1"], axis=0
            )
        )
        nodes.append(oh.make_node("Reshape", ["X::3Dr", "X::Shape+1"], ["X::4D"]))
        nodes.append(oh.make_node("Transpose", ["X::4D"], ["Y"], perm=[0, 2, 1, 3]))
        outputs.append(
            oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape=("a", "b", "c", "d"))
        )
        graph = oh.make_graph(
            nodes,
            "pattern",
            inputs,
            outputs,
            initializers,
            sparse_initializer=sparse_initializers,
        )
        model = oh.make_model(graph, functions=functions, opset_imports=opset_imports)

        print("DOT-SECTION", to_dot(model))
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

    Model with nodes to be fused:

    .. gdot::
        :script: DOT-SECTION
        :process:

        from experimental_experiment.doc import to_dot
        import numpy as np
        import ml_dtypes
        import onnx
        import onnx.helper as oh
        import onnx.numpy_helper as onh

        opset_imports = [
            oh.make_opsetid("", 18),
            oh.make_opsetid("intermediate", 1),
            oh.make_opsetid("com.microsoft", 1),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info(
                "mask", onnx.TensorProto.BOOL, shape=("am", 1, "cm", "dm")
            )
        )
        inputs.append(
            oh.make_tensor_value_info(
                "past_values", onnx.TensorProto.FLOAT, shape=("pav", 8, "pcv", 64)
            )
        )
        inputs.append(
            oh.make_tensor_value_info(
                "values", onnx.TensorProto.FLOAT, shape=("av", "bv", 8, 64)
            )
        )
        inputs.append(
            oh.make_tensor_value_info(
                "query", onnx.TensorProto.FLOAT, shape=("aq", "bq", 8, 64)
            )
        )
        inputs.append(
            oh.make_tensor_value_info(
                "past_keys", onnx.TensorProto.FLOAT, shape=("pak", 8, "pck", 64)
            )
        )
        inputs.append(
            oh.make_tensor_value_info("keys", onnx.TensorProto.FLOAT, shape=("ak", "bk", 8, 64))
        )
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["scale_sqrt"],
                value=onh.from_array(
                    np.array([0.3162277638912201], dtype=np.float32), name="value"
                ),
            )
        )
        nodes.append(oh.make_node("Transpose", ["query"], ["t_query"], perm=[0, 2, 1, 3]))
        nodes.append(oh.make_node("Transpose", ["keys"], ["t_keys"], perm=[0, 2, 1, 3]))
        nodes.append(
            oh.make_node("Concat", ["past_keys", "t_keys"], ["ct_keys"], axis=-2)
        )
        nodes.append(
            oh.make_node("Transpose", ["values"], ["t_values"], perm=[0, 2, 1, 3])
        )
        nodes.append(
            oh.make_node("Concat", ["past_values", "t_values"], ["ct_values"], axis=-2)
        )
        nodes.append(
            oh.make_node(
                "LocalAttention_to1",
                ["t_query", "ct_keys", "ct_values", "mask", "scale_sqrt"],
                ["prob"],
                domain="intermediate",
            )
        )
        nodes.append(oh.make_node("Transpose", ["prob"], ["Y"], perm=[0, 2, 1, 3]))
        outputs.append(
            oh.make_tensor_value_info(
                "ct_values", onnx.TensorProto.FLOAT, shape=("pav", 8, "pcv+bv", 64)
            )
        )
        outputs.append(
            oh.make_tensor_value_info(
                "Y", onnx.TensorProto.FLOAT, shape=("ay", "by", "cy", "dy")
            )
        )
        outputs.append(
            oh.make_tensor_value_info(
                "ct_keys", onnx.TensorProto.FLOAT, shape=("pak", 8, "pck+bk", 64)
            )
        )
        graph = oh.make_graph(
            nodes,
            "pattern",
            inputs,
            outputs,
            initializers,
            sparse_initializer=sparse_initializers,
        )
        model = oh.make_model(graph, functions=functions, opset_imports=opset_imports)

        print("DOT-SECTION", to_dot(model))

    Outcome of the fusion:

    .. gdot::
        :script: DOT-SECTION
        :process:

        from experimental_experiment.doc import to_dot
        import numpy as np
        import ml_dtypes
        import onnx
        import onnx.helper as oh
        import onnx.numpy_helper as onh

        opset_imports = [
            oh.make_opsetid("", 18),
            oh.make_opsetid("intermediate", 1),
            oh.make_opsetid("com.microsoft", 1),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info(
                "mask", onnx.TensorProto.BOOL, shape=("am", 1, "cm", "dm")
            )
        )
        inputs.append(
            oh.make_tensor_value_info(
                "past_values", onnx.TensorProto.FLOAT, shape=("pav", 8, "pcv", 64)
            )
        )
        inputs.append(
            oh.make_tensor_value_info(
                "values", onnx.TensorProto.FLOAT, shape=("av", "bv", 8, 64)
            )
        )
        inputs.append(
            oh.make_tensor_value_info(
                "query", onnx.TensorProto.FLOAT, shape=("aq", "bq", 8, 64)
            )
        )
        inputs.append(
            oh.make_tensor_value_info(
                "past_keys", onnx.TensorProto.FLOAT, shape=("pak", 8, "pck", 64)
            )
        )
        inputs.append(
            oh.make_tensor_value_info("keys", onnx.TensorProto.FLOAT, shape=("ak", "bk", 8, 64))
        )
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["init7_s3_0_0_-1"],
                value=onh.from_array(np.array([0, 0, -1], dtype=np.int64), name="value"),
            )
        )
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["init1_s1_"],
                value=onh.from_array(np.array([0.0], dtype=np.float32), name="value"),
            )
        )
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["init1_s1_2"],
                value=onh.from_array(np.array([-np.inf], dtype=np.float32), name="value"),
            )
        )
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["init7_s4_0_0_-1_64"],
                value=onh.from_array(np.array([0, 0, -1, 64], dtype=np.int64), name="value"),
            )
        )
        nodes.append(
            oh.make_node(
                "Reshape", ["query", "init7_s3_0_0_-1"], ["MultiHeadAttention3DPattern--query"]
            )
        )
        nodes.append(
            oh.make_node(
                "Reshape", ["keys", "init7_s3_0_0_-1"], ["MultiHeadAttention3DPattern--keys"]
            )
        )
        nodes.append(
            oh.make_node(
                "Reshape",
                ["values", "init7_s3_0_0_-1"],
                ["MultiHeadAttention3DPattern--values"],
            )
        )
        nodes.append(
            oh.make_node(
                "Where",
                ["mask", "init1_s1_", "init1_s1_2"],
                ["MultiHeadAttention3DPattern--mask"],
            )
        )
        nodes.append(
            oh.make_node(
                "MultiHeadAttention",
                [
                    "MultiHeadAttention3DPattern--query",
                    "MultiHeadAttention3DPattern--keys",
                    "MultiHeadAttention3DPattern--values",
                    "",
                    "",
                    "MultiHeadAttention3DPattern--mask",
                    "past_keys",
                    "past_values",
                ],
                ["MultiHeadAttention3DPattern--Y", "ct_keys", "ct_values"],
                domain="com.microsoft",
                num_heads=8,
                scale=0.10000000149011612,
            )
        )
        nodes.append(
            oh.make_node(
                "Reshape", ["MultiHeadAttention3DPattern--Y", "init7_s4_0_0_-1_64"], ["Y"]
            )
        )
        outputs.append(
            oh.make_tensor_value_info(
                "ct_values", onnx.TensorProto.FLOAT, shape=("pav", 8, "pcv+bv", 64)
            )
        )
        outputs.append(
            oh.make_tensor_value_info(
                "Y", onnx.TensorProto.FLOAT, shape=("ay", "by", "cy", "dy")
            )
        )
        outputs.append(
            oh.make_tensor_value_info(
                "ct_keys", onnx.TensorProto.FLOAT, shape=("pak", 8, "pck+bk", 64)
            )
        )
        graph = oh.make_graph(
            nodes,
            "pattern",
            inputs,
            outputs,
            initializers,
            sparse_initializer=sparse_initializers,
        )
        model = oh.make_model(graph, functions=functions, opset_imports=opset_imports)

        print("DOT-SECTION", to_dot(model))
    """

    _prefixes_operator_name = (
        f"{FunctionAttentionPattern._operator_name}_to",
        f"{FunctionAttentionPattern._operator_name}sQ_to",
        f"{FunctionAttentionPattern._operator_name}SW_to",
        f"{FunctionAttentionPattern._operator_name}SWsQ_to",
    )

    def __init__(self, verbose: int = 0, priority: int = 2):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if (
            not node.op_type.startswith(self._prefixes_operator_name)
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
            if n and g.is_used_more_than_once(n.output[0]):
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
        switch_where = "SW" in attention.op_type

        nodes = [
            g._make_node("Reshape", [query, init_00_1], [r_query]),
            g._make_node("Reshape", [keys, init_00_1], [r_keys]),
            g._make_node("Reshape", [values, init_00_1], [r_values]),
            g._make_node(
                "Where",
                [mask, minfty, zero] if switch_where else [mask, zero, minfty],
                [attention_bias],
            ),
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
            if node.name:
                continue
            node.name = g.builder.unique_node_name(f"{self.__class__.__name__}--{attention.name}")
        return nodes


class GroupQueryAttention3DPattern(PatternOptimization):
    """
    Fuse LocalAttention into GroupQueryAttention.
    ``bias`` is not supported by this kernel on CUDA.

    .. gdot::
        :script: DOT-SECTION
        :process:

        from experimental_experiment.doc import to_dot
        import numpy as np
        import ml_dtypes
        import onnx
        import onnx.helper as oh
        import onnx.numpy_helper as onh

        opset_imports = [
            oh.make_opsetid("", 24),
            oh.make_opsetid("intermediate", 1),
            oh.make_opsetid("com.microsoft", 1),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info(
                "query", onnx.TensorProto.FLOAT, shape=("batch", 8, "seq_length", 32)
            )
        )
        inputs.append(
            oh.make_tensor_value_info(
                "past_value", onnx.TensorProto.FLOAT, shape=("batch", 4, "past_length", 32)
            )
        )
        inputs.append(
            oh.make_tensor_value_info(
                "key", onnx.TensorProto.FLOAT, shape=("batch", 4, "seq_length", 32)
            )
        )
        inputs.append(
            oh.make_tensor_value_info(
                "value", onnx.TensorProto.FLOAT, shape=("batch", 4, "seq_length", 32)
            )
        )
        inputs.append(
            oh.make_tensor_value_info(
                "past_key", onnx.TensorProto.FLOAT, shape=("batch", 4, "past_length", 32)
            )
        )
        inputs.append(
            oh.make_tensor_value_info(
                "bitwise_not", onnx.TensorProto.BOOL, shape=("seq_length", "total_length")
            )
        )
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["init1_s_::RSh1"],
                value=onh.from_array(
                    np.array([0.4204482138156891], dtype=np.float32), name="value"
                ),
            )
        )
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["init7_s5_1_1_2_1_1"],
                value=onh.from_array(np.array([1, 1, 2, 1, 1], dtype=np.int64), name="value"),
            )
        )
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["init7_s4_0_8_-1_32"],
                value=onh.from_array(np.array([0, 8, -1, 32], dtype=np.int64), name="value"),
            )
        )
        nodes.append(oh.make_node("Concat", ["past_key", "key"], ["cat"], axis=2))
        nodes.append(oh.make_node("Concat", ["past_value", "value"], ["cat_1"], axis=2))
        nodes.append(
            oh.make_node(
                "LocalAttentionGQASW_to1",
                [
                    "query",
                    "cat",
                    "cat_1",
                    "bitwise_not",
                    "init1_s_::RSh1",
                    "init7_s5_1_1_2_1_1",
                    "init7_s4_0_8_-1_32",
                ],
                ["output_0"],
                domain="intermediate",
            )
        )
        outputs.append(
            oh.make_tensor_value_info(
                "output_0", onnx.TensorProto.FLOAT, shape=("batch", 8, "seq_length", 32)
            )
        )
        outputs.append(
            oh.make_tensor_value_info(
                "cat_1",
                onnx.TensorProto.FLOAT,
                shape=("batch", 4, "past_length+seq_length", 32),
            )
        )
        outputs.append(
            oh.make_tensor_value_info(
                "cat", onnx.TensorProto.FLOAT, shape=("batch", 4, "past_length+seq_length", 32)
            )
        )
        graph = oh.make_graph(
            nodes,
            "pattern",
            inputs,
            outputs,
            initializers,
            sparse_initializer=sparse_initializers,
        )
        model = oh.make_model(graph, functions=functions, opset_imports=opset_imports)

        print("DOT-SECTION", to_dot(model))

    Outcome of the fusion:

    .. gdot::
        :script: DOT-SECTION
        :process:

        from experimental_experiment.doc import to_dot
        import numpy as np
        import ml_dtypes
        import onnx
        import onnx.helper as oh
        import onnx.numpy_helper as onh

        opset_imports = [
            oh.make_opsetid("", 24),
            oh.make_opsetid("intermediate", 1),
            oh.make_opsetid("com.microsoft", 1),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info(
                "query", onnx.TensorProto.FLOAT, shape=("batch", 8, "seq_length", 32)
            )
        )
        inputs.append(
            oh.make_tensor_value_info(
                "past_value", onnx.TensorProto.FLOAT, shape=("batch", 4, "past_length", 32)
            )
        )
        inputs.append(
            oh.make_tensor_value_info(
                "key", onnx.TensorProto.FLOAT, shape=("batch", 4, "seq_length", 32)
            )
        )
        inputs.append(
            oh.make_tensor_value_info(
                "value", onnx.TensorProto.FLOAT, shape=("batch", 4, "seq_length", 32)
            )
        )
        inputs.append(
            oh.make_tensor_value_info(
                "past_key", onnx.TensorProto.FLOAT, shape=("batch", 4, "past_length", 32)
            )
        )
        inputs.append(
            oh.make_tensor_value_info(
                "bitwise_not", onnx.TensorProto.BOOL, shape=("seq_length", "total_length")
            )
        )
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["init1_s1_3"],
                value=onh.from_array(
                    np.array([-3.4028234663852886e38], dtype=np.float32), name="value"
                ),
            )
        )
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["init1_s1_2"],
                value=onh.from_array(np.array([0.0], dtype=np.float32), name="value"),
            )
        )
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["init7_s2_0_1"],
                value=onh.from_array(np.array([0, 1], dtype=np.int64), name="value"),
            )
        )
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["init6_s1_"],
                value=onh.from_array(np.array([1], dtype=np.int32), name="value"),
            )
        )
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["init7_s3_0_0_-1"],
                value=onh.from_array(np.array([0, 0, -1], dtype=np.int64), name="value"),
            )
        )
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["init7_s4_0_0_-1_32"],
                value=onh.from_array(np.array([0, 0, -1, 32], dtype=np.int64), name="value"),
            )
        )
        nodes.append(
            oh.make_node(
                "Where",
                ["bitwise_not", "init1_s1_3", "init1_s1_2"],
                ["GroupQueryAttention3DPattern--bitwise_not2"],
            )
        )
        nodes.append(
            oh.make_node(
                "Shape", ["query"], ["GroupQueryAttention3DPattern--query3"], end=1, start=0
            )
        )
        nodes.append(
            oh.make_node(
                "Unsqueeze",
                ["GroupQueryAttention3DPattern--bitwise_not2", "init7_s2_0_1"],
                ["GroupQueryAttention3DPattern--bitwise_not"],
            )
        )
        nodes.append(
            oh.make_node(
                "Shape",
                ["GroupQueryAttention3DPattern--bitwise_not2"],
                ["GroupQueryAttention3DPattern--tl"],
                start=-1,
            )
        )
        nodes.append(
            oh.make_node(
                "Cast",
                ["GroupQueryAttention3DPattern--tl"],
                ["GroupQueryAttention3DPattern--tl2"],
                to=6,
            )
        )
        nodes.append(
            oh.make_node(
                "Sub",
                ["GroupQueryAttention3DPattern--tl2", "init6_s1_"],
                ["GroupQueryAttention3DPattern--tl_1"],
            )
        )
        nodes.append(
            oh.make_node(
                "Expand",
                ["GroupQueryAttention3DPattern--tl_1", "GroupQueryAttention3DPattern--query3"],
                ["GroupQueryAttention3DPattern--sl"],
            )
        )
        nodes.append(
            oh.make_node(
                "Transpose",
                ["query"],
                ["GroupQueryAttention3DPattern--query2"],
                perm=[0, 2, 1, 3],
            )
        )
        nodes.append(
            oh.make_node(
                "Transpose", ["key"], ["GroupQueryAttention3DPattern--cat2"], perm=[0, 2, 1, 3]
            )
        )
        nodes.append(
            oh.make_node(
                "Transpose",
                ["value"],
                ["GroupQueryAttention3DPattern--cat_12"],
                perm=[0, 2, 1, 3],
            )
        )
        nodes.append(
            oh.make_node(
                "Reshape",
                ["GroupQueryAttention3DPattern--query2", "init7_s3_0_0_-1"],
                ["GroupQueryAttention3DPattern--query"],
            )
        )
        nodes.append(
            oh.make_node(
                "Reshape",
                ["GroupQueryAttention3DPattern--cat2", "init7_s3_0_0_-1"],
                ["GroupQueryAttention3DPattern--cat"],
            )
        )
        nodes.append(
            oh.make_node(
                "Reshape",
                ["GroupQueryAttention3DPattern--cat_12", "init7_s3_0_0_-1"],
                ["GroupQueryAttention3DPattern--cat_1"],
            )
        )
        nodes.append(
            oh.make_node(
                "GroupQueryAttention",
                [
                    "GroupQueryAttention3DPattern--query",
                    "GroupQueryAttention3DPattern--cat",
                    "GroupQueryAttention3DPattern--cat_1",
                    "past_key",
                    "past_value",
                    "GroupQueryAttention3DPattern--sl",
                    "GroupQueryAttention3DPattern--tl2",
                    "",
                    "",
                    "",
                    "GroupQueryAttention3DPattern--bitwise_not",
                ],
                ["GroupQueryAttention3DPattern--output_0", "cat", "cat_1"],
                domain="com.microsoft",
                do_rotary=0,
                kv_num_heads=4,
                num_heads=8,
                rotary_interleaved=0,
                scale=0.1767767071723938,
            )
        )
        nodes.append(
            oh.make_node(
                "Reshape",
                ["GroupQueryAttention3DPattern--output_0", "init7_s4_0_0_-1_32"],
                ["GroupQueryAttention3DPattern--output_02"],
            )
        )
        nodes.append(
            oh.make_node(
                "Transpose",
                ["GroupQueryAttention3DPattern--output_02"],
                ["output_0"],
                perm=[0, 2, 1, 3],
            )
        )
        outputs.append(
            oh.make_tensor_value_info(
                "output_0", onnx.TensorProto.FLOAT, shape=("batch", 8, "seq_length", 32)
            )
        )
        outputs.append(
            oh.make_tensor_value_info(
                "cat_1",
                onnx.TensorProto.FLOAT,
                shape=("batch", 4, "past_length+seq_length", 32),
            )
        )
        outputs.append(
            oh.make_tensor_value_info(
                "cat", onnx.TensorProto.FLOAT, shape=("batch", 4, "past_length+seq_length", 32)
            )
        )
        graph = oh.make_graph(
            nodes,
            "pattern",
            inputs,
            outputs,
            initializers,
            sparse_initializer=sparse_initializers,
        )
        model = oh.make_model(graph, functions=functions, opset_imports=opset_imports)

        print("DOT-SECTION", to_dot(model))
    """

    _prefixes_operator_name = (
        f"{FunctionAttentionGQAPattern._operator_gqa_name}SW_to",
        f"{FunctionAttentionGQAPattern._operator_gqa_name}SWsQ_to",
        f"{FunctionAttentionGQAPattern._operator_gqa_name}_to",
        f"{FunctionAttentionGQAPattern._operator_gqa_name}sQ_to",
    )

    def __init__(self, verbose: int = 0, priority: int = 2):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if (
            not node.op_type.startswith(self._prefixes_operator_name)
            or node.domain != FunctionAttentionGQAPattern._domain_name
            or len(node.input) != 7
        ):
            return self.none()
        keys, values = node.input[1:3]
        concats = g.node_before(keys), g.node_before(values)
        if None in concats:
            return self.none(node, inspect.currentframe().f_lineno)
        if len(concats[0].input) != 2 or len(concats[1].input) != 2:
            return self.none(node, inspect.currentframe().f_lineno)
        if g.get_attribute_with_default(concats[0], "axis", 0) != g.get_attribute_with_default(
            concats[1], "axis", 0
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        if len(node.input) > 3 and node.input[3] and g.has_processor("CUDA"):
            # GroupQueryAttention does not work with a bias.
            return self.none()

        if len(node.input) > 3 and (
            not g.has_rank(node.input[3]) or g.get_rank(node.input[3]) < 2
        ):
            # Only 2D ranks allowed.
            return self.none(node, inspect.currentframe().f_lineno)

        if not g.is_constant_scalar(node.input[4]):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant(node.input[5]):
            return self.none(node, inspect.currentframe().f_lineno)
        cst = g.get_computed_constant(node.input[5])
        if cst is None:
            return self.none(node, inspect.currentframe().f_lineno)
        cst = tuple(cst)
        if len(cst) < 4:
            return self.none(node, inspect.currentframe().f_lineno)
        if cst[:2] != cst[3:] or cst[:2] != (1, 1):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant(node.input[6]):
            return self.none(node, inspect.currentframe().f_lineno)
        shape_or_axis = g.get_computed_constant(node.input[6])
        if shape_or_axis is None:
            return self.none(node, inspect.currentframe().f_lineno)
        if "sQ_to" in node.op_type:
            # This is an axis for a Squeeze node.
            if not g.get_shape(node.input[1]):
                # We need that shape to get kv_num_heads.
                return self.none(node, inspect.currentframe().f_lineno)
        else:
            # This is a shape for a Reshape node.
            if shape_or_axis[1] <= 0:
                return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, [*concats, node], self.apply)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        keys_concat_node: NodeProto,
        values_concat_node: NodeProto,
        local_attention_gqa: NodeProto,
    ) -> List[NodeProto]:
        query, _keys, _values, mask = local_attention_gqa.input[:4]
        scale = g.get_constant_scalar(local_attention_gqa.input[4])  # this scale ** 0.5
        expand_shape = g.get_computed_constant(local_attention_gqa.input[5])
        repeat = int(expand_shape[2])
        rk_mask = g.get_rank(mask)

        if "sQ_" in local_attention_gqa.op_type:
            k_shape = g.get_shape(local_attention_gqa.input[1])
            kv_num_heads = k_shape[1]
        else:
            reshape_shape = g.get_computed_constant(local_attention_gqa.input[6])
            kv_num_heads = reshape_shape[1] // repeat

        num_heads = kv_num_heads * repeat
        head_size = g.get_shape(query)[-1]

        shape00 = g.make_initializer(
            "", np.array([0, 0, -1], dtype=np.int64), source=f"{self.__class__.__name__}.00"
        )
        shape0000 = g.make_initializer(
            "",
            np.array([0, 0, -1, head_size], dtype=np.int64),
            source=f"{self.__class__.__name__}.00_1",
        )
        one32 = g.make_initializer(
            "", np.array([1], dtype=np.int32), source=f"{self.__class__.__name__}.1i"
        )

        query3D = g.unique_name(f"{self.__class__.__name__}--{query}")
        query4D = g.unique_name(f"{self.__class__.__name__}--{query}")
        keys3D = g.unique_name(f"{self.__class__.__name__}--{_keys}")
        keys4D = g.unique_name(f"{self.__class__.__name__}--{_keys}")
        values3D = g.unique_name(f"{self.__class__.__name__}--{_values}")
        values4D = g.unique_name(f"{self.__class__.__name__}--{_values}")
        attn3D = g.unique_name(f"{self.__class__.__name__}--{local_attention_gqa.output[0]}")
        attn4D = g.unique_name(f"{self.__class__.__name__}--{local_attention_gqa.output[0]}")

        seqlensk = g.unique_name(f"{self.__class__.__name__}--sl")
        total_length64 = g.unique_name(f"{self.__class__.__name__}--tl")
        total_length = g.unique_name(f"{self.__class__.__name__}--tl")
        total_length_1 = g.unique_name(f"{self.__class__.__name__}--tl_1")

        batch_shape = g.unique_name(f"{self.__class__.__name__}--{query}")

        nodes = []
        # mask is not mask if SW
        switch_where = "SW" in local_attention_gqa.op_type
        if g.get_type(mask) == TensorProto.BOOL:
            itype = g.get_type(query)
            dtype = tensor_dtype_to_np_dtype(itype)
            zero = g.make_initializer(
                "", np.array([0], dtype=dtype), source=f"{self.__class__.__name__}.0"
            )
            infty = g.make_initializer(
                "",
                np.array([np.finfo(dtype).min], dtype=dtype),
                source=f"{self.__class__.__name__}.inf{itype}",
            )
            float_mask = g.unique_name(f"{self.__class__.__name__}--{mask}")
            nodes.append(
                g._make_node(
                    "Where",
                    [mask, infty, zero] if switch_where else [mask, zero, infty],
                    [float_mask],
                )
            )
        else:
            raise NotImplementedError(
                f"float mask is not implemented yet for pattern {self.__class__.__name__!r}"
            )

        if rk_mask == 2:
            expanded_mask = g.unique_name(f"{self.__class__.__name__}--{mask}")
            cst01 = g.make_initializer(
                "", np.array([0, 1], dtype=np.int64), source=f"{self.__class__.__name__}.01"
            )
            nodes.append(g._make_node("Unsqueeze", [float_mask, cst01], [expanded_mask]))
        elif rk_mask == 3:
            expanded_mask = g.unique_name(f"{self.__class__.__name__}--{mask}")
            cst0 = g.make_initializer(
                "", np.array([0], dtype=np.int64), source=f"{self.__class__.__name__}.0"
            )
            nodes.append(g._make_node("Unsqueeze", [float_mask, cst0], [expanded_mask]))
        else:
            expanded_mask = float_mask

        attention_node = g._make_node(
            "GroupQueryAttention",
            [
                query3D,
                keys3D,
                values3D,
                keys_concat_node.input[0],
                values_concat_node.input[0],
                seqlensk,
                total_length,
                "",
                "",
                "",
                expanded_mask,
            ],
            [attn3D, keys_concat_node.output[0], values_concat_node.output[0]],
            num_heads=num_heads,
            kv_num_heads=kv_num_heads,
            scale=scale**2,
            do_rotary=0,
            rotary_interleaved=0,
            domain="com.microsoft",
            doc_string="This operator only accepts batch_size=1 "
            "and (past_length==0 or seq_length==1).",
        )

        nodes.extend(
            [
                g._make_node("Shape", [query], [batch_shape], start=0, end=1),
                g._make_node("Shape", [float_mask], [total_length64], start=-1),
                g._make_node("Cast", [total_length64], [total_length], to=TensorProto.INT32),
                g._make_node("Sub", [total_length, one32], [total_length_1]),
                g._make_node("Expand", [total_length_1, batch_shape], [seqlensk]),
                g._make_node("Transpose", [query], [query4D], perm=[0, 2, 1, 3]),
                g._make_node(
                    "Transpose", [keys_concat_node.input[1]], [keys4D], perm=[0, 2, 1, 3]
                ),
                g._make_node(
                    "Transpose", [values_concat_node.input[1]], [values4D], perm=[0, 2, 1, 3]
                ),
                g._make_node("Reshape", [query4D, shape00], [query3D]),
                g._make_node("Reshape", [keys4D, shape00], [keys3D]),
                g._make_node("Reshape", [values4D, shape00], [values3D]),
                attention_node,
                g._make_node("Reshape", [attn3D, shape0000], [attn4D]),
                g._make_node(
                    "Transpose", [attn4D], [local_attention_gqa.output[0]], perm=[0, 2, 1, 3]
                ),
            ]
        )
        for node in nodes:
            if not node.name:
                node.name = g.builder.unique_node_name(
                    f"{self.__class__.__name__}--{local_attention_gqa.name}"
                )
        return nodes


class Attention3DPattern(PatternOptimization):
    _prefixes_operator_name = (f"{FunctionAttentionGQAPattern._operator_gqa_name}_to",)

    def __init__(self, verbose: int = 0, priority: int = 2):
        super().__init__(verbose, priority)

    def _match_above(
        self, g: "GraphBuilderPatternOptimization", node: NodeProto, name: str  # noqa: F821
    ) -> Optional[Tuple[NodeProto, NodeProto, NodeProto]]:
        transpose = g.node_before(name)
        if not transpose or transpose.op_type != "Transpose":
            return self.none(node, inspect.currentframe().f_lineno)
        if tuple(g.get_attribute(transpose, "perm").ints) != (0, 2, 1, 3):
            return self.none(node, inspect.currentframe().f_lineno)
        reshape = g.node_before(transpose.input[0])
        if not reshape or reshape.op_type != "Reshape":
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant(reshape.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        cst = g.get_computed_constant(reshape.input[1])
        if cst is None:
            return self.none(node, inspect.currentframe().f_lineno)
        matmul = g.node_before(reshape.input[0])
        if matmul is None or matmul.op_type != "MatMul":
            return self.none(node, inspect.currentframe().f_lineno)
        return matmul, reshape, transpose

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if (
            not node.op_type.startswith(self._prefixes_operator_name)
            or node.domain != FunctionAttentionGQAPattern._domain_name
            or len(node.input) != 5
        ):
            return self.none()

        query, keys, values = node.input[:3]

        before_query = self._match_above(query)
        if before_query is None:
            return self.none(node, inspect.currentframe().f_lineno)
        before_keys = self._match_above(keys)
        if before_keys is None:
            return self.none(node, inspect.currentframe().f_lineno)
        before_values = self._match_above(values)
        if before_values is None:
            return self.none(node, inspect.currentframe().f_lineno)

        mm_q, re_q, tr_q = before_query
        mm_k, re_k, tr_k = before_keys
        mm_v, re_v, tr_v = before_values

        if mm_q.input[0] != mm_k.input[0] or mm_q.input[0] != mm_v.input[0]:
            return self.none(node, inspect.currentframe().f_lineno)

        cst_q = g.get_computed_constant(re_q.input[0])
        cst_k = g.get_computed_constant(re_k.input[0])
        cst_v = g.get_computed_constant(re_v.input[0])
        if tuple(cst_q) != tuple(cst_k) or tuple(cst_q) != tuple(cst_v):
            return self.none(node, inspect.currentframe().f_lineno)

        transposes = g.next_nodes(node.output[0])
        if len(transposes) != 1 or transposes[0].op_type != "Transpose":
            return self.none(node, inspect.currentframe().f_lineno)
        transpose = transposes[0]
        if tuple(g.get_attribute(transpose, "perm").ints) != (0, 2, 1, 3):
            return self.none(node, inspect.currentframe().f_lineno)
        reshapes = g.next_nodes(transpose.output[0])
        if len(reshapes) != 1 or reshapes[0].op_type != "Reshape":
            return self.none(node, inspect.currentframe().f_lineno)
        reshape = reshapes[0]
        if not g.is_constant(reshape.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        cst = g.get_computed_constant(reshape.input[1])
        if cst is None:
            return self.none(node, inspect.currentframe().f_lineno)

        nodes = [mm_q, re_q, tr_q, mm_k, re_k, tr_k, mm_v, re_v, tr_v, node, transpose, reshape]
        for n in nodes[:-1]:
            if g.is_used_more_than_once(n.output[0]):
                return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, nodes, self.apply)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        mm_q: NodeProto,
        re_q: NodeProto,
        tr_q: NodeProto,
        mm_k: NodeProto,
        re_k: NodeProto,
        tr_k: NodeProto,
        mm_v: NodeProto,
        re_v: NodeProto,
        tr_v: NodeProto,
        attention: NodeProto,
        transpose: NodeProto,
        reshape: NodeProto,
    ) -> List[NodeProto]:

        packed = g.unique_name(
            f"{self.__class__.__name__}--{mm_q.input[1]}-{mm_k.input[1]}-{mm_v.input[1]}"
        )
        concat_node = g.make_node(
            "Concat",
            [mm_q.input[1], mm_k.input[1], mm_v.input[1]],
            [packed],
            axis=-1,
            name=f"{self.__class__.__name__}--{attention.name}",
        )
        scale = float(g.get_constant_scalar(attention.input[4])) ** 2
        num_heads = g.get_shape(attention.input[0])[1]
        sizes = (
            g.get_shape(mm_q.input[1])[-1],
            g.get_shape(mm_k.input[1])[-1],
            g.get_shape(mm_k.input[1])[-1],
        )

        attention_node = g.make_node(
            "Attention",
            [
                mm_q.input[0],
                packed,
                "",
                "",
                "",
                attention.input[3],
            ],
            [reshape.output[0]],
            num_heads=num_heads,
            qkv_hidden_size=sizes,
            scale=scale**2,
            domain="com.microsoft",
            name=f"{self.__class__.__name__}--{attention.name}",
        )
        return [concat_node, attention_node]
