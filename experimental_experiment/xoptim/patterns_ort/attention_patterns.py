import inspect
from typing import List, Optional
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization


class AttentionPattern(PatternOptimization):
    """
    Replaces many nodes by Attention [com.microsoft].
    The first sketch of the pattern was generared from
    an onnx model and the following code:

    .. code-block:: python

        import onnx
        from onnx_array_api.translate_api import translate
        from experimental_experiment.xbuilder import GraphBuilder, OptimizationOptions
        from experimental_experiment.xbuilder.reverse_graph_builder import (
            to_graph_pattern_matching,
        )

        filename = "unfused_Attention.onnx"
        onx = onnx.load(filename)
        builder = GraphBuilder(
            onx, optimization_options=OptimizationOptions(patterns="default+onnxruntime")
        )
        new_onx = builder.to_onnx()

        onnx.save(new_onx, "new_onnx.onnx")
        print(to_graph_pattern_matching(new_onx))

        print("---------------------------------")
        print("---------------------------------")
        print("---------------------------------")

        print(translate(onx, api="onnx-short"))
    """

    def __init__(self, verbose: int = 0, priority: int = 3):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        node_22_Reshape = node
        if node_22_Reshape.op_type != "Reshape" or node_22_Reshape.domain != "":
            return self.none()
        transpose_5 = node_22_Reshape.input[0]
        ### val_132 = node_22_Reshape.input[1]

        # val_132 has no predecessor.

        if g.is_used_more_than_once(transpose_5):
            return self.none(node, inspect.currentframe().f_lineno)
        node_21_Transpose = g.node_before(transpose_5)
        if (
            node_21_Transpose is None
            or node_21_Transpose.op_type != "Transpose"
            or node_21_Transpose.domain != ""
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        matmul_1 = node_21_Transpose.input[0]

        if g.is_used_more_than_once(matmul_1):
            return self.none(node, inspect.currentframe().f_lineno)
        node_20_MatMul = g.node_before(matmul_1)
        if (
            node_20_MatMul is None
            or node_20_MatMul.op_type != "MatMul"
            or node_20_MatMul.domain != ""
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        masked_fill_3 = node_20_MatMul.input[0]
        transpose_3 = node_20_MatMul.input[1]

        if g.is_used_more_than_once(transpose_3):
            return self.none(node, inspect.currentframe().f_lineno)
        node_15_Transpose = g.node_before(transpose_3)
        if (
            node_15_Transpose is None
            or node_15_Transpose.op_type != "Transpose"
            or node_15_Transpose.domain != ""
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        view_2 = node_15_Transpose.input[0]

        if g.is_used_more_than_once(view_2):
            return self.none(node, inspect.currentframe().f_lineno)
        node_11_Reshape = g.node_before(view_2)
        if (
            node_11_Reshape is None
            or node_11_Reshape.op_type != "Reshape"
            or node_11_Reshape.domain != ""
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        linear_5 = node_11_Reshape.input[0]
        ### val_120 = node_11_Reshape.input[1]

        # val_120 has no predecessor.

        if g.is_used_more_than_once(linear_5):
            return self.none(node, inspect.currentframe().f_lineno)
        node_10_Add = g.node_before(linear_5)
        if node_10_Add is None or node_10_Add.op_type != "Add" or node_10_Add.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        val_115 = node_10_Add.input[0]
        ### encoder_encoders_0_self_attn_linear_v_bias = node_10_Add.input[1]

        # encoder_encoders_0_self_attn_linear_v_bias has no predecessor.

        if g.is_used_more_than_once(val_115):
            return self.none(node, inspect.currentframe().f_lineno)
        node_9_MatMul = g.node_before(val_115)
        if (
            node_9_MatMul is None
            or node_9_MatMul.op_type != "MatMul"
            or node_9_MatMul.domain != ""
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        ### layer_norm_1 = node_9_MatMul.input[0]
        ### encoder_encoders_0_self_attn_linear_v_weight = node_9_MatMul.input[1]

        # encoder_encoders_0_self_attn_linear_v_weight has no predecessor.

        # layer_norm_1 has no predecessor.

        if g.is_used_more_than_once(masked_fill_3):
            return self.none(node, inspect.currentframe().f_lineno)
        node_19_Where = g.node_before(masked_fill_3)
        if (
            node_19_Where is None
            or node_19_Where.op_type != "Where"
            or node_19_Where.domain != ""
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        eq_87 = node_19_Where.input[0]
        ### val_126 = node_19_Where.input[1]
        softmax = node_19_Where.input[2]

        if g.is_used_more_than_once(softmax):
            return self.none(node, inspect.currentframe().f_lineno)
        node_18_Softmax = g.node_before(softmax)
        if (
            node_18_Softmax is None
            or node_18_Softmax.op_type != "Softmax"
            or node_18_Softmax.domain != ""
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        masked_fill_2 = node_18_Softmax.input[0]

        if g.is_used_more_than_once(masked_fill_2):
            return self.none(node, inspect.currentframe().f_lineno)
        node_17_Where = g.node_before(masked_fill_2)
        if (
            node_17_Where is None
            or node_17_Where.op_type != "Where"
            or node_17_Where.domain != ""
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        eq_87 = node_17_Where.input[0]
        ### val_124 = node_17_Where.input[1]
        add_322 = node_17_Where.input[2]

        if g.is_used_more_than_once(add_322):
            return self.none(node, inspect.currentframe().f_lineno)
        node_16_Add = g.node_before(add_322)
        if node_16_Add is None or node_16_Add.op_type != "Add" or node_16_Add.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        matmul = node_16_Add.input[0]
        ### unsqueeze_9 = node_16_Add.input[1]

        # unsqueeze_9 has no predecessor.

        if g.is_used_more_than_once(matmul):
            return self.none(node, inspect.currentframe().f_lineno)
        node_14_FusedMatMul = g.node_before(matmul)
        if (
            node_14_FusedMatMul is None
            or node_14_FusedMatMul.op_type != "FusedMatMul"
            or node_14_FusedMatMul.domain != "com.microsoft"
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        transpose_1 = node_14_FusedMatMul.input[0]
        TransposeFusedMatMulBPattern__transpose_4 = node_14_FusedMatMul.input[1]

        if g.is_used_more_than_once(TransposeFusedMatMulBPattern__transpose_4):
            return self.none(node, inspect.currentframe().f_lineno)
        node_13_Transpose = g.node_before(TransposeFusedMatMulBPattern__transpose_4)
        if (
            node_13_Transpose is None
            or node_13_Transpose.op_type != "Transpose"
            or node_13_Transpose.domain != ""
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        view_1 = node_13_Transpose.input[0]

        if g.is_used_more_than_once(view_1):
            return self.none(node, inspect.currentframe().f_lineno)
        node_8_Reshape = g.node_before(view_1)
        if (
            node_8_Reshape is None
            or node_8_Reshape.op_type != "Reshape"
            or node_8_Reshape.domain != ""
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        linear_4 = node_8_Reshape.input[0]
        ### val_112 = node_8_Reshape.input[1]

        # val_112 has no predecessor.

        if g.is_used_more_than_once(linear_4):
            return self.none(node, inspect.currentframe().f_lineno)
        node_7_Add = g.node_before(linear_4)
        if node_7_Add is None or node_7_Add.op_type != "Add" or node_7_Add.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        val_107 = node_7_Add.input[0]
        ### encoder_encoders_0_self_attn_linear_k_bias = node_7_Add.input[1]

        # encoder_encoders_0_self_attn_linear_k_bias has no predecessor.

        if g.is_used_more_than_once(val_107):
            return self.none(node, inspect.currentframe().f_lineno)
        node_6_MatMul = g.node_before(val_107)
        if (
            node_6_MatMul is None
            or node_6_MatMul.op_type != "MatMul"
            or node_6_MatMul.domain != ""
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        ### layer_norm_1 = node_6_MatMul.input[0]
        ### encoder_encoders_0_self_attn_linear_k_weight = node_6_MatMul.input[1]

        # encoder_encoders_0_self_attn_linear_k_weight has no predecessor.

        # layer_norm_1 has no predecessor.

        if g.is_used_more_than_once(transpose_1):
            return self.none(node, inspect.currentframe().f_lineno)
        node_12_Transpose = g.node_before(transpose_1)
        if (
            node_12_Transpose is None
            or node_12_Transpose.op_type != "Transpose"
            or node_12_Transpose.domain != ""
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        view = node_12_Transpose.input[0]

        if g.is_used_more_than_once(view):
            return self.none(node, inspect.currentframe().f_lineno)
        node_5_Reshape = g.node_before(view)
        if (
            node_5_Reshape is None
            or node_5_Reshape.op_type != "Reshape"
            or node_5_Reshape.domain != ""
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        linear_3 = node_5_Reshape.input[0]
        ### val_104 = node_5_Reshape.input[1]

        # val_104 has no predecessor.

        if g.is_used_more_than_once(linear_3):
            return self.none(node, inspect.currentframe().f_lineno)
        node_4_Add = g.node_before(linear_3)
        if node_4_Add is None or node_4_Add.op_type != "Add" or node_4_Add.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        val_97 = node_4_Add.input[0]
        ### encoder_encoders_0_self_attn_linear_q_bias = node_4_Add.input[1]

        # encoder_encoders_0_self_attn_linear_q_bias has no predecessor.

        if g.is_used_more_than_once(val_97):
            return self.none(node, inspect.currentframe().f_lineno)
        node_3_MatMul = g.node_before(val_97)
        if (
            node_3_MatMul is None
            or node_3_MatMul.op_type != "MatMul"
            or node_3_MatMul.domain != ""
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        ### layer_norm_1 = node_3_MatMul.input[0]
        ### encoder_encoders_0_self_attn_linear_q_weight = node_3_MatMul.input[1]

        # encoder_encoders_0_self_attn_linear_q_weight has no predecessor.

        # layer_norm_1 has no predecessor.

        # val_124 has no predecessor.

        node_2_Equal = g.node_before(eq_87)
        if (
            node_2_Equal is None
            or node_2_Equal.op_type != "Equal"
            or node_2_Equal.domain != ""
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        convert_element_type_default = node_2_Equal.input[0]
        ### val_10 = node_2_Equal.input[1]

        # val_10 has no predecessor.

        if g.is_used_more_than_once(convert_element_type_default):
            return self.none(node, inspect.currentframe().f_lineno)
        node_1_Cast = g.node_before(convert_element_type_default)
        if node_1_Cast is None or node_1_Cast.op_type != "Cast" or node_1_Cast.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        unsqueeze_6 = node_1_Cast.input[0]

        if g.is_used_more_than_once(unsqueeze_6):
            return self.none(node, inspect.currentframe().f_lineno)
        node_0_Unsqueeze = g.node_before(unsqueeze_6)
        if (
            node_0_Unsqueeze is None
            or node_0_Unsqueeze.op_type != "Unsqueeze"
            or node_0_Unsqueeze.domain != ""
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        ### expand_1 = node_0_Unsqueeze.input[0]
        ### dim_0_7 = node_0_Unsqueeze.input[1]

        # dim_0_7 has no predecessor.

        # expand_1 has no predecessor.

        # val_126 has no predecessor.

        # eq_87 is already processed.

        # list of nodes
        nodes = [
            node_0_Unsqueeze,
            node_1_Cast,
            node_2_Equal,
            node_3_MatMul,
            node_4_Add,
            node_5_Reshape,
            node_12_Transpose,
            node_6_MatMul,
            node_7_Add,
            node_8_Reshape,
            node_13_Transpose,
            node_14_FusedMatMul,
            node_16_Add,
            node_17_Where,
            node_18_Softmax,
            node_19_Where,
            node_9_MatMul,
            node_10_Add,
            node_11_Reshape,
            node_15_Transpose,
            node_20_MatMul,
            node_21_Transpose,
            node_22_Reshape,
        ]
        return self.none()
        return MatchResult(self, nodes, self.apply, insert_at=nodes[-1])

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        *nodes: NodeProto,
    ) -> List[NodeProto]:
        #
        raise NotImplementedError("not yet")
        """
        node = nodes[0]
        fc = g.make_node(
            "FusedConv",
            node.input,
            node_act.output,
            domain="com.microsoft",
            activation=node_act.op_type,
            name=f"{self.__class__.__name__}--{node.name}",
        )
        #fc.attribute.extend(node.attribute)
        return [fc]
        """
