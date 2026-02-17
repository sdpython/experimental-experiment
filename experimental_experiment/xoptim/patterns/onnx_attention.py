import inspect
from typing import List, Optional, Tuple, Union
import numpy as np
from onnx import NodeProto
from ...helpers import tensor_dtype_to_np_dtype
from ...xbuilder import FunctionOptions, GraphBuilder
from ..patterns_api import MatchResult, PatternOptimization


class FunctionAttentionPattern(PatternOptimization):
    """
    Merges Attention nodes into a local function.
    That includes a version for GroupQueryAttention
    (see second pattern).

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
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info(
                "values", onnx.TensorProto.FLOAT, shape=("av", "bv", "cv", "dv")
            )
        )
        inputs.append(
            oh.make_tensor_value_info(
                "keys", onnx.TensorProto.FLOAT, shape=("ak", "bk", "ck", "dk")
            )
        )
        inputs.append(
            oh.make_tensor_value_info("scale_sqrt", onnx.TensorProto.FLOAT, shape=(1,))
        )
        inputs.append(
            oh.make_tensor_value_info(
                "mask", onnx.TensorProto.BOOL, shape=("am", "bm", "cm", "dm")
            )
        )
        inputs.append(
            oh.make_tensor_value_info(
                "query", onnx.TensorProto.FLOAT, shape=("aq", "bq", "cq", "dq")
            )
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
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["zero"],
                value=onh.from_array(np.array([0.0], dtype=np.float32), name="value"),
            )
        )
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["minfty"],
                value=onh.from_array(np.array([-np.inf], dtype=np.float32), name="value"),
            )
        )
        nodes.append(oh.make_node("Mul", ["query", "scale_sqrt"], ["query_scaled"]))
        nodes.append(oh.make_node("Mul", ["keys", "scale_sqrt"], ["keys_scaled"]))
        nodes.append(
            oh.make_node(
                "Transpose", ["keys_scaled"], ["keys_scaled_t"], perm=[0, 1, 3, 2]
            )
        )
        nodes.append(oh.make_node("MatMul", ["query_scaled", "keys_scaled_t"], ["qk"]))
        nodes.append(oh.make_node("Where", ["mask", "zero", "minfty"], ["bias"]))
        nodes.append(oh.make_node("Add", ["qk", "bias"], ["qkb"]))
        nodes.append(oh.make_node("Softmax", ["qkb"], ["qkbs"], axis=-1))
        nodes.append(oh.make_node("IsNaN", ["qkbs"], ["nans"]))
        nodes.append(oh.make_node("Where", ["nans", "zero", "qkbs"], ["filt"]))
        nodes.append(oh.make_node("MatMul", ["filt", "values"], ["Y"]))
        outputs.append(
            oh.make_tensor_value_info(
                "Y", onnx.TensorProto.FLOAT, shape=("ay", "by", "cy", "dy")
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
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info(
                "values", onnx.TensorProto.FLOAT, shape=("av", "bv", "cv", "dv")
            )
        )
        inputs.append(
            oh.make_tensor_value_info(
                "keys", onnx.TensorProto.FLOAT, shape=("ak", "bk", "ck", "dk")
            )
        )
        inputs.append(
            oh.make_tensor_value_info("scale_sqrt", onnx.TensorProto.FLOAT, shape=(1,))
        )
        inputs.append(
            oh.make_tensor_value_info(
                "mask", onnx.TensorProto.BOOL, shape=("am", "bm", "cm", "dm")
            )
        )
        inputs.append(
            oh.make_tensor_value_info(
                "query", onnx.TensorProto.FLOAT, shape=("aq", "bq", "cq", "dq")
            )
        )
        nodes.append(
            oh.make_node(
                "LocalAttention_to1",
                ["query", "keys", "values", "mask", "scale_sqrt"],
                ["Y"],
                domain="intermediate",
            )
        )
        outputs.append(
            oh.make_tensor_value_info(
                "Y", onnx.TensorProto.FLOAT, shape=("ay", "by", "cy", "dy")
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

    GroupQueryAttention (GQA):

    .. gdot::
        :script: DOT-SECTION
        :process:

        from experimental_experiment.doc import to_dot
        import numpy as np
        import ml_dtypes
        import onnx
        import onnx.helper as oh
        import onnx.numpy_helper as onh
        from onnx_array_api.translate_api.make_helper import make_node_extended

        opset_imports = [
            oh.make_opsetid("", 18),
            oh.make_opsetid("intermediate", 1),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info("init1_s_::RSh1", onnx.TensorProto.FLOAT, shape=(1,))
        )
        inputs.append(
            oh.make_tensor_value_info(
                "query", onnx.TensorProto.FLOAT, shape=("batch", 8, "seq_length", 32)
            )
        )
        inputs.append(
            oh.make_tensor_value_info(
                "cat_1",
                onnx.TensorProto.FLOAT,
                shape=("batch", 4, "past_length+seq_length", 32),
            )
        )
        inputs.append(
            oh.make_tensor_value_info(
                "cat", onnx.TensorProto.FLOAT, shape=("batch", 4, "past_length+seq_length", 32)
            )
        )
        inputs.append(
            oh.make_tensor_value_info(
                "to", onnx.TensorProto.BOOL, shape=("seq_length", "total_length")
            )
        )
        inputs.append(
            oh.make_tensor_value_info("init7_s4_0_8_-1_32", onnx.TensorProto.INT64, shape=(4,))
        )
        inputs.append(
            oh.make_tensor_value_info("init7_s5_1_1_2_1_1", onnx.TensorProto.INT64, shape=(5,))
        )
        nodes.append(
            make_node_extended(
                "Constant",
                [],
                ["init1_s_::RSh1"],
                value=onh.from_array(
                    np.array([0.4204482138156891], dtype=np.float32), name="value"
                ),
            )
        )
        nodes.append(
            make_node_extended(
                "Constant",
                [],
                ["init7_s1_2"],
                value=onh.from_array(np.array([2], dtype=np.int64), name="value"),
            )
        )
        nodes.append(
            make_node_extended(
                "Constant",
                [],
                ["init1_s_::RSh12"],
                value=onh.from_array(
                    np.array([0.4204482138156891], dtype=np.float32), name="value"
                ),
            )
        )
        nodes.append(
            make_node_extended(
                "Constant",
                [],
                ["init7_s5_1_1_2_1_1"],
                value=onh.from_array(np.array([1, 1, 2, 1, 1], dtype=np.int64), name="value"),
            )
        )
        nodes.append(
            make_node_extended(
                "Constant",
                [],
                ["init7_s4_0_8_-1_32"],
                value=onh.from_array(np.array([0, 8, -1, 32], dtype=np.int64), name="value"),
            )
        )
        nodes.append(
            make_node_extended(
                "Constant",
                [],
                ["init1_s1_"],
                value=onh.from_array(np.array([-inf], dtype=np.float32), name="value"),
            )
        )
        nodes.append(
            make_node_extended(
                "Constant",
                [],
                ["c_lifted_tensor_0"],
                value=onh.from_array(np.array(0.0, dtype=np.float32), name="value"),
            )
        )
        nodes.append(make_node_extended("Mul", ["query", "init1_s_::RSh1"], ["_onx_mul_query"]))
        nodes.append(make_node_extended("Unsqueeze", ["cat", "init7_s1_2"], ["cat::UnSq2"]))
        nodes.append(
            make_node_extended(
                "Mul",
                ["cat::UnSq2", "init1_s_::RSh12"],
                ["ShapeBasedExpandSwapPattern_SwapUnaryPattern--repeat_interleave_1"],
            )
        )
        nodes.append(
            make_node_extended(
                "Expand",
                [
                    "ShapeBasedExpandSwapPattern_SwapUnaryPattern--repeat_interleave_1",
                    "init7_s5_1_1_2_1_1",
                ],
                ["SwapUnaryPattern--repeat_interleave_1"],
            )
        )
        nodes.append(
            make_node_extended(
                "Reshape",
                ["SwapUnaryPattern--repeat_interleave_1", "init7_s4_0_8_-1_32"],
                ["SwapUnaryPattern--transpose"],
            )
        )
        nodes.append(
            make_node_extended(
                "Transpose",
                ["SwapUnaryPattern--transpose"],
                ["_onx_mul_transpose"],
                perm=[0, 1, 3, 2],
            )
        )
        nodes.append(
            make_node_extended("MatMul", ["_onx_mul_query", "_onx_mul_transpose"], ["matmul"])
        )
        nodes.append(
            make_node_extended("Where", ["to", "init1_s1_", "matmul"], ["masked_fill"])
        )
        nodes.append(make_node_extended("Softmax", ["masked_fill"], ["softmax"], axis=-1))
        nodes.append(make_node_extended("IsNaN", ["softmax"], ["isnan"]))
        nodes.append(
            make_node_extended("Where", ["isnan", "c_lifted_tensor_0", "softmax"], ["where"])
        )
        nodes.append(make_node_extended("Unsqueeze", ["cat_1", "init7_s1_2"], ["cat_1::UnSq2"]))
        nodes.append(
            make_node_extended(
                "Expand", ["cat_1::UnSq2", "init7_s5_1_1_2_1_1"], ["_onx_expand_cat_1::UnSq2"]
            )
        )
        nodes.append(
            make_node_extended(
                "Reshape",
                ["_onx_expand_cat_1::UnSq2", "init7_s4_0_8_-1_32"],
                ["repeat_interleave"],
            )
        )
        nodes.append(make_node_extended("MatMul", ["where", "repeat_interleave"], ["output_0"]))
        outputs.append(
            oh.make_tensor_value_info(
                "output_0", onnx.TensorProto.FLOAT, shape=("batch", 8, "seq_length", 32)
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
        from onnx_array_api.translate_api.make_helper import make_node_extended

        opset_imports = [
            oh.make_opsetid("", 18),
            oh.make_opsetid("intermediate", 1),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info("init1_s_::RSh1", onnx.TensorProto.FLOAT, shape=(1,))
        )
        inputs.append(
            oh.make_tensor_value_info(
                "query", onnx.TensorProto.FLOAT, shape=("batch", 8, "seq_length", 32)
            )
        )
        inputs.append(
            oh.make_tensor_value_info(
                "cat_1",
                onnx.TensorProto.FLOAT,
                shape=("batch", 4, "past_length+seq_length", 32),
            )
        )
        inputs.append(
            oh.make_tensor_value_info(
                "cat", onnx.TensorProto.FLOAT, shape=("batch", 4, "past_length+seq_length", 32)
            )
        )
        inputs.append(
            oh.make_tensor_value_info(
                "to", onnx.TensorProto.BOOL, shape=("seq_length", "total_length")
            )
        )
        inputs.append(
            oh.make_tensor_value_info("init7_s4_0_8_-1_32", onnx.TensorProto.INT64, shape=(4,))
        )
        inputs.append(
            oh.make_tensor_value_info("init7_s5_1_1_2_1_1", onnx.TensorProto.INT64, shape=(5,))
        )
        nodes.append(
            make_node_extended(
                "LocalAttentionGQASW_to1",
                [
                    "query",
                    "cat",
                    "cat_1",
                    "to",
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

    _operator_name = "LocalAttention"
    _domain_name = "intermediate"

    def __init__(self, verbose: int = 0, priority: int = 0):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Softmax" or node.domain != "":
            return self.none()
        axis = g.get_attribute(node, "axis").i
        if axis != -1:
            return self.none(node, inspect.currentframe().f_lineno)

        node_before = g.node_before(node.input[0])
        if node_before.op_type == "Add":
            # Add(X, Where(mask, 0, -inf))
            add_node = node_before
            where_node = g.node_before(add_node.input[1])
            if where_node is None or where_node.op_type != "Where":
                return self.none(node, inspect.currentframe().f_lineno)

            if not g.is_constant_scalar(where_node.input[1]):
                return self.none(node, inspect.currentframe().f_lineno)
            if not g.is_constant_scalar(where_node.input[2]):
                return self.none(node, inspect.currentframe().f_lineno)

            cst_zero = g.get_constant_scalar(where_node.input[1])
            if cst_zero != 0:
                return self.none(node, inspect.currentframe().f_lineno)
            cst_inf = g.get_constant_scalar(where_node.input[2])
            if not np.isinf(cst_inf):
                return self.none(node, inspect.currentframe().f_lineno)

            mat_qk = g.node_before(add_node.input[0])
            if mat_qk is None or mat_qk.op_type not in ("MatMul", "FusedMatMul"):
                return self.none(node, inspect.currentframe().f_lineno)
        elif node_before.op_type == "Where":
            # Where(mask, -inf, X)
            add_node = None
            where_node = node_before
            if not g.is_constant_scalar(where_node.input[1]):
                return self.none(node, inspect.currentframe().f_lineno)
            cst_zero = None
            cst_inf = g.get_constant_scalar(where_node.input[1])
            if not np.isinf(cst_inf):
                return self.none(node, inspect.currentframe().f_lineno)
            mat_qk = g.node_before(where_node.input[2])
            if mat_qk is None or mat_qk.op_type not in ("MatMul", "FusedMatMul"):
                return self.none(node, inspect.currentframe().f_lineno)
        else:
            return self.none(node, inspect.currentframe().f_lineno)

        mul1 = g.node_before(mat_qk.input[0])
        if mul1 is None or mul1.op_type != "Mul":
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant_scalar(mul1.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)

        if mat_qk.op_type == "MatMul":
            transpose = g.node_before(mat_qk.input[1])
            if transpose is None or transpose.op_type != "Transpose":
                return self.none(node, inspect.currentframe().f_lineno)
            perm = g.get_attribute(transpose, "perm").ints
            if tuple(perm) != (0, 1, 3, 2):
                return self.none(node, inspect.currentframe().f_lineno)
            mul2 = g.node_before(transpose.input[0])
        else:
            transA = g.get_attribute_with_default(mat_qk, "transA", 0)
            transB = g.get_attribute_with_default(mat_qk, "transB", 1)
            if transA != 0 or transB != 1:
                return self.none(node, inspect.currentframe().f_lineno)
            transpose = None
            mul2 = g.node_before(mat_qk.input[1])

        if mul2 is None:
            return self.none(node, inspect.currentframe().f_lineno)

        if mul2.op_type == "Mul":
            # This condition is verified for Attention or MultiHeadAttention.
            gqa_expand = gqa_reshape = gqa_unsqueeze = None
        elif mul2.op_type == "Reshape":
            # This condition is verified by GroupQueryAttention.
            gqa_reshape = mul2
            mul2 = None
            gqa_expand = g.node_before(gqa_reshape.input[0])
            if gqa_expand.op_type != "Expand":
                return self.none(node, inspect.currentframe().f_lineno)
            mul2 = g.node_before(gqa_expand.input[0])
            if mul2.op_type != "Mul":
                return self.none(node, inspect.currentframe().f_lineno)
            gqa_unsqueeze = g.node_before(mul2.input[0])
            if gqa_unsqueeze.op_type != "Unsqueeze":
                return self.none(node, inspect.currentframe().f_lineno)
            #
            if not g.is_constant(gqa_expand.input[1]):
                return self.none(node, inspect.currentframe().f_lineno)
            exp_shape = g.get_computed_constant(gqa_expand.input[1])
            if tuple(exp_shape[:2]) != (1, 1) or tuple(exp_shape[3:]) != (1, 1):
                return self.none(node, inspect.currentframe().f_lineno)
            if not g.is_constant(gqa_unsqueeze.input[1]):
                return self.none(node, inspect.currentframe().f_lineno)
            unsq_shape = g.get_computed_constant(gqa_unsqueeze.input[1])
            if tuple(unsq_shape) != (2,):
                return self.none(node, inspect.currentframe().f_lineno)
            if not g.is_constant(gqa_reshape.input[1]):
                return self.none(node, inspect.currentframe().f_lineno)
            resh_shape = g.get_computed_constant(gqa_reshape.input[1])
            if resh_shape.size != 4:
                return self.none(node, inspect.currentframe().f_lineno)
            if not g.has_shape(gqa_unsqueeze.input[0]) or not g.has_shape(gqa_reshape.output[0]):
                return self.none(node, inspect.currentframe().f_lineno)
            shape1 = g.get_shape_renamed(gqa_unsqueeze.input[0])
            shape2 = g.get_shape_renamed(gqa_reshape.output[0])
            if shape1[0] != shape2[0] or shape1[2] != shape2[2] or shape1[3] != shape2[3]:
                return self.none(node, inspect.currentframe().f_lineno)
        else:
            # No Attention, no MultiHeadAttention, no GroupQueryAttention
            return self.none(node, inspect.currentframe().f_lineno)

        if mul2.input[1] != mul1.input[1]:
            if not g.is_constant_scalar(mul1.input[1]) or not g.is_constant_scalar(mul2.input[1]):
                return self.none(node, inspect.currentframe().f_lineno)
            cst1 = g.get_constant_scalar(mul1.input[1])
            cst2 = g.get_constant_scalar(mul2.input[1])
            if cst1 != cst2:
                return self.none(node, inspect.currentframe().f_lineno)

        # after softmax
        next_nodes = g.next_nodes(node.output[0])
        if len(next_nodes) != 2:
            return self.none(node, inspect.currentframe().f_lineno)
        if {n.op_type for n in next_nodes} != {"Where", "IsNaN"}:
            return self.none(node, inspect.currentframe().f_lineno)
        isnan, where2 = next_nodes[:: (1 if next_nodes[0].op_type == "IsNaN" else -1)]
        if where2.input[0] != isnan.output[0]:
            return self.none(node, inspect.currentframe().f_lineno)
        if where2.input[2] != node.output[0]:
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant_scalar(where2.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        cst = g.get_constant_scalar(where2.input[1])
        if cst != 0:
            return self.none(node, inspect.currentframe().f_lineno)
        mat_qkvs = g.next_nodes(where2.output[0])
        if len(mat_qkvs) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        mat_qkv = mat_qkvs[0]
        if mat_qkv.op_type != "MatMul":
            return self.none(node, inspect.currentframe().f_lineno)

        if gqa_reshape:
            # We need to include the nodes repeating values,
            # the same one which repeated the keys.
            gqa_reshape_v = g.node_before(mat_qkv.input[1])
            if gqa_reshape_v.op_type != "Reshape":
                return self.none(node, inspect.currentframe().f_lineno)
            gqa_expand_v = g.node_before(gqa_reshape_v.input[0])
            if gqa_expand_v.op_type != "Expand":
                return self.none(node, inspect.currentframe().f_lineno)
            gqa_unsqueeze_v = g.node_before(gqa_expand_v.input[0])
            if gqa_unsqueeze_v.op_type != "Unsqueeze":
                return self.none(node, inspect.currentframe().f_lineno)
            #
            if not g.is_constant(gqa_expand.input[1]):
                return self.none(node, inspect.currentframe().f_lineno)
            exp_shape_v = g.get_computed_constant(gqa_expand_v.input[1])
            if tuple(exp_shape) != tuple(exp_shape_v):
                return self.none(node, inspect.currentframe().f_lineno)
            if not g.is_constant(gqa_unsqueeze_v.input[1]):
                return self.none(node, inspect.currentframe().f_lineno)
            unsq_shape_v = g.get_computed_constant(gqa_unsqueeze_v.input[1])
            if tuple(unsq_shape_v) != tuple(unsq_shape):
                return self.none(node, inspect.currentframe().f_lineno)
            if not g.is_constant(gqa_reshape_v.input[1]):
                return self.none(node, inspect.currentframe().f_lineno)
            resh_shape_v = g.get_computed_constant(gqa_reshape_v.input[1])
            if tuple(resh_shape_v) != tuple(resh_shape):
                return self.none(node, inspect.currentframe().f_lineno)
        else:
            gqa_expand_v = gqa_reshape_v = gqa_unsqueeze_v = None

        nodes = [
            mul1,
            gqa_unsqueeze,
            mul2,
            gqa_expand,
            gqa_reshape,
            transpose,
            mat_qk,
            where_node,
            add_node,
            node,
            isnan,
            where2,
            gqa_unsqueeze_v,
            gqa_expand_v,
            gqa_reshape_v,
            mat_qkv,
        ]

        for n in nodes[:-1]:
            if not n:
                continue
            if n.op_type == "Softmax":
                if len(g.next_nodes(n.output[0])) != 2:
                    return self.none(node, inspect.currentframe().f_lineno)
                continue
            if g.is_used_more_than_once(n.output[0]):
                return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, nodes, self.apply)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        mul1: NodeProto,
        gqa_unsqueeze: Optional[NodeProto],
        mul2: NodeProto,
        gqa_expand: Optional[NodeProto],
        gqa_reshape: Optional[NodeProto],
        transpose: Optional[NodeProto],
        mat_qk: NodeProto,
        where_node: NodeProto,
        add_node: Optional[NodeProto],
        softmax: NodeProto,
        isnan: NodeProto,
        where: NodeProto,
        gqa_unsqueeze_v: Optional[NodeProto],
        gqa_expand_v: Optional[NodeProto],
        gqa_reshape_v: Optional[NodeProto],
        mat_qkv: NodeProto,
    ) -> List[NodeProto]:
        itype = g.get_type(mul1.input[1])
        suffix = []
        if add_node is None:
            suffix.append("SW")
        if transpose is None:
            assert (
                mat_qk.op_type == "FusedMatMul"
            ), f"transpose is None but mat_qk={g.pretty_node(mat_qk)}"
            suffix.append("noT")
        if gqa_reshape:
            gqa = "GQA"
            gqa_args = [gqa_expand.input[1], gqa_reshape.input[1]]
        else:
            gqa = ""
            gqa_args = []

        name = f"{self._operator_name}{gqa}{''.join(suffix)}_to{itype}"
        attention_nodes = [
            g.make_node(
                name,
                [
                    mul1.input[0],
                    gqa_unsqueeze.input[0] if gqa_reshape else mul2.input[0],
                    gqa_unsqueeze_v.input[0] if gqa_reshape else mat_qkv.input[1],
                    where_node.input[0],
                    mul1.input[1],
                    *gqa_args,
                ],
                [mat_qkv.output[0]],
                name=f"{self.__class__.__name__}--{softmax.name}",
                domain=self._domain_name,
            )
        ]
        nodes_to_return = attention_nodes

        # Creates the local function
        if not g.builder.has_local_function(name, domain=self._domain_name):
            self._add_local_function(
                g.builder, name, itype=itype, gqa=gqa, switch_where=add_node is None
            )
        return nodes_to_return

    @classmethod
    def _add_local_function(
        cls, g: GraphBuilder, name: str, itype: int, gqa: bool, switch_where: bool
    ):
        lg = GraphBuilder(g.main_opset, as_function=True)
        lg.make_tensor_input("query")
        lg.make_tensor_input("keys")
        lg.make_tensor_input("values")
        lg.make_tensor_input("mask")
        lg.make_tensor_input("scale_sqrt")

        scaled_keys = lg.op.Mul("keys", "scale_sqrt", name=cls.__name__)
        if gqa:
            lg.make_tensor_input("expand_shape")
            lg.make_tensor_input("gqa_shape")

            two = np.array([2], dtype=np.int64)
            unsq_keys = lg.op.UnsqueezeAnyOpset(scaled_keys, two, name=cls.__name__)
            unsq_values = lg.op.UnsqueezeAnyOpset("values", two, name=cls.__name__)
            exp_keys = lg.op.Expand(unsq_keys, "expand_shape")
            exp_values = lg.op.Expand(unsq_values, "expand_shape")
            resh_keys = lg.op.Reshape(exp_keys, "gqa_shape")
            resh_values = lg.op.Reshape(exp_values, "gqa_shape")
            scaled_keys = resh_keys
            values = resh_values
        else:
            values = "values"

        scaled_query = lg.op.Mul("query", "scale_sqrt", name=cls.__name__)
        scaled_keys_t = lg.op.Transpose(scaled_keys, perm=(0, 1, 3, 2), name=cls.__name__)
        qk = lg.op.MatMul(scaled_query, scaled_keys_t, name=cls.__name__)
        dtype = tensor_dtype_to_np_dtype(itype)
        zero = np.array([0], dtype=dtype)
        minfty = np.array([-np.inf], dtype=dtype)
        where_args = (minfty, qk) if switch_where else (qk, minfty)
        masked_qk = lg.op.Where("mask", *where_args, name=cls.__name__)
        softmax = lg.op.Softmax(masked_qk, axis=-1, name=cls.__name__)
        filtered = lg.op.Where(
            lg.op.IsNaN(softmax, name=cls.__name__), zero, softmax, name=cls.__name__
        )
        lg.op.MatMul(filtered, values, outputs=["Y"], name=cls.__name__)

        lg.make_tensor_output("Y")

        function_options = FunctionOptions(
            export_as_function=True,
            name=name,
            domain=cls._domain_name,
            move_initializer_to_constant=True,
        )
        g.make_local_function(lg, function_options=function_options)
        assert g.has_local_function(
            name, domain=cls._domain_name
        ), f"The function {cls._domain_name}.{name} was not added to the builder."


class FunctionAttentionGQAPattern(FunctionAttentionPattern):
    """
    Merges onnx nodes equivalent to repeat interleave followed by function
    ``LocalAttention`` into ``LocalAttentionGQA`` (GQA for GroupQueryAttention).
    """

    _operator_gqa_name = f"{FunctionAttentionPattern._operator_name}GQA"

    def _match_keys_or_values(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        keys_or_values: str,
    ) -> Optional[Tuple[NodeProto, NodeProto, NodeProto, Tuple[Tuple[Union[int, str], ...]]]]:

        gqa_reshape = g.node_before(keys_or_values)
        if not gqa_reshape or gqa_reshape.op_type != "Reshape" or gqa_reshape.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        gqa_expand = g.node_before(gqa_reshape.input[0])
        if gqa_expand.op_type != "Expand":
            return self.none(node, inspect.currentframe().f_lineno)

        gqa_unsqueeze = g.node_before(gqa_expand.input[0])
        if gqa_unsqueeze.op_type != "Unsqueeze":
            return self.none(node, inspect.currentframe().f_lineno)
        #
        if not g.is_constant(gqa_expand.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        exp_shape = g.get_computed_constant(gqa_expand.input[1])
        if tuple(exp_shape[:2]) != (1, 1) or tuple(exp_shape[3:]) != (1, 1):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant(gqa_unsqueeze.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        unsq_shape = g.get_computed_constant(gqa_unsqueeze.input[1])
        if tuple(unsq_shape) != (2,):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant(gqa_reshape.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        resh_shape = g.get_computed_constant(gqa_reshape.input[1])
        if resh_shape.size != 4:
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.has_shape(gqa_unsqueeze.input[0]) or not g.has_shape(gqa_reshape.output[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        shape1 = g.get_shape_renamed(gqa_unsqueeze.input[0])
        shape2 = g.get_shape_renamed(gqa_reshape.output[0])
        if shape1[0] != shape2[0] or shape1[2] != shape2[2] or shape1[3] != shape2[3]:
            return self.none(node, inspect.currentframe().f_lineno)

        return (
            gqa_unsqueeze,
            gqa_expand,
            gqa_reshape,
            (tuple(unsq_shape), tuple(exp_shape), tuple(resh_shape)),
        )

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if (
            not node.op_type.startswith(FunctionAttentionPattern._operator_name)
            or node.op_type.startswith(FunctionAttentionGQAPattern._operator_gqa_name)
            or node.domain != FunctionAttentionGQAPattern._domain_name
        ):
            return self.none()

        keys, values = node.input[1:3]

        matched_keys = self._match_keys_or_values(g, node, keys)
        if not matched_keys:
            return self.none(node, inspect.currentframe().f_lineno)

        matched_values = self._match_keys_or_values(g, node, values)
        if not matched_values:
            return self.none(node, inspect.currentframe().f_lineno)

        gqa_unsqueeze, gqa_expand, gqa_reshape, shapes = matched_keys
        gqa_unsqueeze_v, gqa_expand_v, gqa_reshape_v, _shapes_v = matched_values

        unsq_shape, exp_shape, resh_shape = shapes
        unsq_shape_v, exp_shape_v, resh_shape_v = shapes

        if unsq_shape_v != unsq_shape:
            return self.none(node, inspect.currentframe().f_lineno)
        if exp_shape != exp_shape_v:
            return self.none(node, inspect.currentframe().f_lineno)
        if resh_shape_v != resh_shape:
            return self.none(node, inspect.currentframe().f_lineno)

        nodes = [
            gqa_unsqueeze,
            gqa_expand,
            gqa_reshape,
            gqa_unsqueeze_v,
            gqa_expand_v,
            gqa_reshape_v,
            node,
        ]
        for n in nodes[:-1]:
            if g.is_used_more_than_once(n.output[0]):
                return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, nodes, self.apply)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        gqa_unsqueeze: NodeProto,
        gqa_expand: NodeProto,
        gqa_reshape: NodeProto,
        gqa_unsqueeze_v: NodeProto,
        gqa_expand_v: NodeProto,
        gqa_reshape_v: NodeProto,
        attn: NodeProto,
    ) -> List[NodeProto]:
        itype = g.get_type(gqa_unsqueeze.input[0])
        name = f"{self._operator_gqa_name}{attn.op_type[len(self._operator_name):]}"
        attention_nodes = [
            g.make_node(
                name,
                [
                    attn.input[0],
                    gqa_unsqueeze.input[0],
                    gqa_unsqueeze_v.input[0],
                    attn.input[3] if len(attn.input) > 3 else "",
                    attn.input[4] if len(attn.input) > 4 else "",
                    gqa_expand.input[1],
                    gqa_reshape.input[1],
                ],
                [attn.output[0]],
                name=f"{self.__class__.__name__}--{attn.name}",
                domain=self._domain_name,
            )
        ]

        # Creates the local function
        if not g.builder.has_local_function(name, domain=self._domain_name):
            self._add_local_function(
                g.builder, name, itype=itype, gqa=True, switch_where="SW" in attn.op_type
            )
        return attention_nodes
