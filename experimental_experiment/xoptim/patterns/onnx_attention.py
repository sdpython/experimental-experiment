import inspect
from typing import List, Optional, Tuple, Union
import numpy as np
from onnx import NodeProto, TensorProto
from ...helpers import tensor_dtype_to_np_dtype
from ...xbuilder import FunctionOptions, GraphBuilder
from ..patterns_api import MatchResult, PatternOptimization


class FunctionAttentionPattern(PatternOptimization):
    """
    Merges Attention nodes into a local function.
    That includes a version for GroupQueryAttention
    (see second pattern).

    Main Pattern
    ++++++++++++

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

    GroupQueryAttention (GQA)
    +++++++++++++++++++++++++

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
                ["init7_s1_2"],
                value=onh.from_array(np.array([2], dtype=np.int64), name="value"),
            )
        )
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["init1_s_::RSh12"],
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
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["init1_s1_"],
                value=onh.from_array(np.array([-np.inf], dtype=np.float32), name="value"),
            )
        )
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["c_lifted_tensor_0"],
                value=onh.from_array(np.array(0.0, dtype=np.float32), name="value"),
            )
        )
        nodes.append(oh.make_node("Mul", ["query", "init1_s_::RSh1"], ["_onx_mul_query"]))
        nodes.append(oh.make_node("Unsqueeze", ["cat", "init7_s1_2"], ["cat::UnSq2"]))
        nodes.append(
            oh.make_node(
                "Mul",
                ["cat::UnSq2", "init1_s_::RSh12"],
                ["ShapeBasedExpandSwapPattern_SwapUnaryPattern--repeat_interleave_1"],
            )
        )
        nodes.append(
            oh.make_node(
                "Expand",
                [
                    "ShapeBasedExpandSwapPattern_SwapUnaryPattern--repeat_interleave_1",
                    "init7_s5_1_1_2_1_1",
                ],
                ["SwapUnaryPattern--repeat_interleave_1"],
            )
        )
        nodes.append(
            oh.make_node(
                "Reshape",
                ["SwapUnaryPattern--repeat_interleave_1", "init7_s4_0_8_-1_32"],
                ["SwapUnaryPattern--transpose"],
            )
        )
        nodes.append(
            oh.make_node(
                "Transpose",
                ["SwapUnaryPattern--transpose"],
                ["_onx_mul_transpose"],
                perm=[0, 1, 3, 2],
            )
        )
        nodes.append(
            oh.make_node("MatMul", ["_onx_mul_query", "_onx_mul_transpose"], ["matmul"])
        )
        nodes.append(
            oh.make_node("Where", ["to", "init1_s1_", "matmul"], ["masked_fill"])
        )
        nodes.append(oh.make_node("Softmax", ["masked_fill"], ["softmax"], axis=-1))
        nodes.append(oh.make_node("IsNaN", ["softmax"], ["isnan"]))
        nodes.append(
            oh.make_node("Where", ["isnan", "c_lifted_tensor_0", "softmax"], ["where"])
        )
        nodes.append(oh.make_node("Unsqueeze", ["cat_1", "init7_s1_2"], ["cat_1::UnSq2"]))
        nodes.append(
            oh.make_node(
                "Expand", ["cat_1::UnSq2", "init7_s5_1_1_2_1_1"], ["_onx_expand_cat_1::UnSq2"]
            )
        )
        nodes.append(
            oh.make_node(
                "Reshape",
                ["_onx_expand_cat_1::UnSq2", "init7_s4_0_8_-1_32"],
                ["repeat_interleave"],
            )
        )
        nodes.append(oh.make_node("MatMul", ["where", "repeat_interleave"], ["output_0"]))
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
            oh.make_node(
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

    3D Pattern
    ++++++++++

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
                "values_t", onnx.TensorProto.FLOAT, shape=("av", 8, "cv", 64)
            )
        )
        inputs.append(
            oh.make_tensor_value_info(
                "keys", onnx.TensorProto.FLOAT, shape=("ak", "ck", "bk*dk")
            )
        )
        inputs.append(
            oh.make_tensor_value_info("scale_sqrt", onnx.TensorProto.FLOAT, shape=(1,))
        )
        inputs.append(oh.make_tensor_value_info("shape0", onnx.TensorProto.INT64, shape=(4,)))
        inputs.append(
            oh.make_tensor_value_info(
                "mask", onnx.TensorProto.BOOL, shape=("am", "bm", "cm", "dm")
            )
        )
        inputs.append(
            oh.make_tensor_value_info(
                "query", onnx.TensorProto.FLOAT, shape=("aq", "cq", "bq*dq")
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
                ["shape0"],
                value=onh.from_array(np.array([0, 0, 8, 64], dtype=np.int64), name="value"),
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
                value=onh.from_array(np.array([-inf], dtype=np.float32), name="value"),
            )
        )
        nodes.append(
            oh.make_node(
                "Mul", ["query", "scale_sqrt"], ["SwapUnaryPattern--query_reshaped"]
            )
        )
        nodes.append(
            oh.make_node(
                "Transpose", ["SwapUnaryPattern--query_t"], ["query_scaled"], perm=[0, 2, 1, 3]
            )
        )
        nodes.append(
            oh.make_node(
                "Reshape",
                ["SwapUnaryPattern--query_reshaped", "shape0"],
                ["SwapUnaryPattern--query_t"],
            )
        )
        nodes.append(
            oh.make_node(
                "Mul", ["keys", "scale_sqrt"], ["SwapUnaryPattern--keys_reshaped"]
            )
        )
        nodes.append(
            oh.make_node(
                "Reshape",
                ["SwapUnaryPattern--keys_reshaped", "shape0"],
                ["SwapUnaryPattern--keys_t"],
            )
        )
        nodes.append(
            oh.make_node(
                "Transpose", ["SwapUnaryPattern--keys_t"], ["keys_scaled"], perm=[0, 2, 3, 1]
            )
        )
        nodes.append(oh.make_node("MatMul", ["query_scaled", "keys_scaled"], ["qk"]))
        nodes.append(oh.make_node("Where", ["mask", "zero", "minfty"], ["bias"]))
        nodes.append(oh.make_node("Add", ["qk", "bias"], ["qkb"]))
        nodes.append(oh.make_node("Softmax", ["qkb"], ["qkbs"], axis=-1))
        nodes.append(oh.make_node("IsNaN", ["qkbs"], ["nans"]))
        nodes.append(oh.make_node("Where", ["nans", "zero", "qkbs"], ["filt"]))
        nodes.append(oh.make_node("MatMul", ["filt", "values_t"], ["Y"]))
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
                "values_t", onnx.TensorProto.FLOAT, shape=("av", 8, "cv", 64)
            )
        )
        inputs.append(
            oh.make_tensor_value_info(
                "keys", onnx.TensorProto.FLOAT, shape=("ak", "ck", "bk*dk")
            )
        )
        inputs.append(
            oh.make_tensor_value_info("scale_sqrt", onnx.TensorProto.FLOAT, shape=(1,))
        )
        inputs.append(oh.make_tensor_value_info("shape0", onnx.TensorProto.INT64, shape=(4,)))
        inputs.append(
            oh.make_tensor_value_info(
                "mask", onnx.TensorProto.BOOL, shape=("am", "bm", "cm", "dm")
            )
        )
        inputs.append(
            oh.make_tensor_value_info(
                "query", onnx.TensorProto.FLOAT, shape=("aq", "cq", "bq*dq")
            )
        )
        nodes.append(
            oh.make_node(
                "Reshape",
                ["query", "shape0"],
                ["FunctionAttentionPattern--SwapUnaryPattern--query_t"],
            )
        )
        nodes.append(
            oh.make_node(
                "Reshape",
                ["keys", "shape0"],
                ["FunctionAttentionPattern--SwapUnaryPattern--keys_t"],
            )
        )
        nodes.append(
            oh.make_node(
                "Transpose",
                ["FunctionAttentionPattern--SwapUnaryPattern--query_t"],
                ["FunctionAttentionPattern--query"],
                perm=[0, 2, 1, 3],
            )
        )
        nodes.append(
            oh.make_node(
                "Transpose",
                ["FunctionAttentionPattern--SwapUnaryPattern--keys_t"],
                ["FunctionAttentionPattern--keys"],
                perm=[0, 2, 1, 3],
            )
        )
        nodes.append(
            oh.make_node(
                "LocalAttention_to1",
                [
                    "FunctionAttentionPattern--query",
                    "FunctionAttentionPattern--keys",
                    "values_t",
                    "mask",
                    "scale_sqrt",
                ],
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
    """

    _operator_name = "LocalAttention"
    _domain_name = "intermediate"

    def __init__(self, verbose: int = 0, priority: int = 0):
        super().__init__(verbose, priority)

    def _find_index_inf(self, g, where_node):
        for i in (1, 2):
            if g.is_constant_scalar(where_node.input[i]):
                cst = g.get_constant_scalar(where_node.input[i])
                if np.isinf(cst):
                    return i
        return None

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Softmax" or node.domain != "" or g.main_opset < 18:
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
            if not g.is_constant_scalar(where_node.input[1]) and not g.is_constant_scalar(
                where_node.input[2]
            ):
                return self.none(node, inspect.currentframe().f_lineno)
            cst_zero = None
            inf_index = 1 if g.is_constant_scalar(where_node.input[1]) else 2
            cst_inf = g.get_constant_scalar(where_node.input[inf_index])
            if not np.isinf(cst_inf) or cst_inf > 0:
                return self.none(node, inspect.currentframe().f_lineno)
            mat_qk = g.node_before(where_node.input[3 - inf_index])
            if mat_qk is None or mat_qk.op_type not in ("MatMul", "FusedMatMul"):
                return self.none(node, inspect.currentframe().f_lineno)
        else:
            return self.none(node, inspect.currentframe().f_lineno)

        mul1 = g.node_before(mat_qk.input[0])
        if mul1 is None:
            return self.none(node, inspect.currentframe().f_lineno)
        if mul1.op_type == "Transpose":
            transpose_mul1 = mul1
            reshape_mul1 = g.node_before(mul1.input[0])
            perm = tuple(g.get_attribute(transpose_mul1, "perm").ints)
            if perm != (0, 2, 1, 3):
                return self.none(node, inspect.currentframe().f_lineno)
            if reshape_mul1 is None:
                return self.none(node, inspect.currentframe().f_lineno)
            mul1 = g.node_before(reshape_mul1.input[0])
        else:
            reshape_mul1 = None
            transpose_mul1 = None
        if mul1 is None or mul1.op_type != "Mul":
            return self.none(node, inspect.currentframe().f_lineno)

        if not g.is_constant_scalar(mul1.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)

        if mat_qk.op_type == "MatMul":
            transpose = g.node_before(mat_qk.input[1])
            if transpose is None or transpose.op_type != "Transpose":
                return self.none(node, inspect.currentframe().f_lineno)
            perm = g.get_attribute(transpose, "perm").ints
            if transpose_mul1 is None:
                if tuple(perm) != (0, 1, 3, 2):
                    return self.none(node, inspect.currentframe().f_lineno)
                mul2 = g.node_before(transpose.input[0])
                reshape_mul2 = None
            else:
                if tuple(perm) != (0, 2, 3, 1):
                    return self.none(node, inspect.currentframe().f_lineno)
                reshape_mul2 = g.node_before(transpose.input[0])
                if reshape_mul2 is None:
                    return self.none(node, inspect.currentframe().f_lineno)
                mul2 = g.node_before(reshape_mul2.input[0])
                if not g.is_constant(reshape_mul1.input[1]) or not g.is_constant(
                    reshape_mul2.input[1]
                ):
                    return self.none(node, inspect.currentframe().f_lineno)
                shapem1 = g.get_computed_constant(reshape_mul1.input[1])
                shapem2 = g.get_computed_constant(reshape_mul2.input[1])
                if shapem1 is None or shapem2 is None:
                    return self.none(node, inspect.currentframe().f_lineno)
                if shapem1.tolist() != shapem2.tolist():
                    return self.none(node, inspect.currentframe().f_lineno)
        else:
            transA = g.get_attribute_with_default(mat_qk, "transA", 0)
            transB = g.get_attribute_with_default(mat_qk, "transB", 1)
            if transA != 0 or transB != 1:
                return self.none(node, inspect.currentframe().f_lineno)
            transpose = None
            mul2 = g.node_before(mat_qk.input[1])
            reshape_mul2 = None

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
                return self.none(
                    node,
                    inspect.currentframe().f_lineno,
                    msg=lambda: f"Shape mismatch {shape1=}, {shape2=}",
                )
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
            transpose_mul1,
            reshape_mul1,
            gqa_unsqueeze,
            mul2,
            reshape_mul2,
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
            if n.op_type == "Where" and id(n) == id(where_node):
                # The rewriting will add that to the list of rewritten nodes.
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
        transpose_mul1: Optional[NodeProto],
        reshape_mul1: Optional[NodeProto],
        gqa_unsqueeze: Optional[NodeProto],
        mul2: NodeProto,
        reshape_mul2: Optional[NodeProto],
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

        index_inf = self._find_index_inf(g, where_node)
        assert index_inf, (
            f"Could not any inf in node {g.pretty_node(where_node)}, "
            f"the pattern {self.__class__.__name__} should not have matched."
        )
        switch_where = index_inf == 1
        if switch_where:
            suffix.append("SW")

        if transpose is None:
            assert (
                mat_qk.op_type == "FusedMatMul"
            ), f"transpose is None but mat_qk={g.pretty_node(mat_qk)}"
            suffix.append("noT")
        if gqa_reshape:
            gqa = "GQA" if gqa_reshape.op_type == "Reshape" else "GQAsQ"
            gqa_args = [gqa_expand.input[1], gqa_reshape.input[1]]
        else:
            gqa = ""
            gqa_args = []

        # nodes to add
        attention_nodes = []
        if g.is_used_more_than_once(where_node.output[0]):
            # keep it if it used more than once
            attention_nodes.append(where_node)

        scale = mul1.input[1]
        if reshape_mul1 is not None:
            assert (
                reshape_mul2 is not None
                and transpose_mul1 is not None
                and transpose is not None
                and gqa_unsqueeze is None
            ), (
                f"Inconsistencies with {reshape_mul2=}, {transpose_mul1=}, "
                f"{transpose=}, {gqa_unsqueeze=}"
            )
            keys = g.unique_name(f"{self.__class__.__name__}--{mul1.input[0]}")
            values = g.unique_name(f"{self.__class__.__name__}--{mul2.input[0]}")
            keys_t = g.unique_name(f"{self.__class__.__name__}--{transpose_mul1.input[0]}")
            values_t = g.unique_name(f"{self.__class__.__name__}--{transpose.input[0]}")
            attention_nodes.extend(
                [
                    g.make_node(
                        "Reshape",
                        [mul1.input[0], reshape_mul1.input[1]],
                        [keys_t],
                        name=f"{self.__class__.__name__}--{reshape_mul1.name}",
                    ),
                    g.make_node(
                        "Reshape",
                        [mul2.input[0], reshape_mul2.input[1]],
                        [values_t],
                        name=f"{self.__class__.__name__}--{reshape_mul2.name}",
                    ),
                    g.make_node(
                        "Transpose",
                        [keys_t],
                        [keys],
                        perm=[0, 2, 1, 3],
                        name=f"{self.__class__.__name__}--{transpose_mul1.name}",
                    ),
                    g.make_node(
                        "Transpose",
                        [values_t],
                        [values],
                        perm=[0, 2, 1, 3],
                        name=f"{self.__class__.__name__}--{transpose.name}",
                    ),
                ]
            )
        else:
            keys = mul1.input[0]
            values = gqa_unsqueeze.input[0] if gqa_reshape else mul2.input[0]

        name = f"{self._operator_name}{gqa}{''.join(suffix)}_to{itype}"
        attention_nodes.append(
            g.make_node(
                name,
                [
                    keys,
                    values,
                    gqa_unsqueeze_v.input[0] if gqa_reshape else mat_qkv.input[1],
                    where_node.input[0],
                    scale,
                    *gqa_args,
                ],
                [mat_qkv.output[0]],
                name=f"{self.__class__.__name__}--{softmax.name}",
                domain=self._domain_name,
            )
        )

        nodes_to_return = attention_nodes

        # Creates the local function
        if not g.builder.has_local_function(name, domain=self._domain_name):
            self._add_local_function(
                g.builder,
                name,
                itype=itype,
                gqa=gqa,
                switch_where=switch_where,
                use_qga_squeeze=gqa_reshape and gqa_reshape.op_type == "Squeeze",
            )
        return nodes_to_return

    @classmethod
    def _add_local_function(
        cls,
        g: GraphBuilder,
        name: str,
        itype: int,
        gqa: bool,
        switch_where: bool,
        use_qga_squeeze: bool,
    ):
        lg = GraphBuilder(g.main_opset, as_function=True)
        lg.make_tensor_input("query")
        lg.make_tensor_input("keys")
        lg.make_tensor_input("values")
        mask_name = "not_mask" if switch_where else "mask"
        lg.make_tensor_input(mask_name)
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
            if use_qga_squeeze:
                resh_keys = lg.op.Squeeze(exp_keys, "gqa_shape")
                resh_values = lg.op.Squeeze(exp_values, "gqa_shape")
            else:
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
        masked_qk = lg.op.Where(mask_name, *where_args, name=cls.__name__)
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


class _CommonGQAMethods:
    def _match_keys_or_values(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        keys_or_values: str,
    ) -> Optional[Tuple[NodeProto, NodeProto, NodeProto, Tuple[Tuple[Union[int, str], ...]]]]:

        gqa_reshape = g.node_before(keys_or_values)
        if (
            not gqa_reshape
            or gqa_reshape.op_type not in ("Reshape", "Squeeze")
            or gqa_reshape.domain != ""
            or g.main_opset < 18
        ):
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
        if gqa_reshape.op_type == "Reshape":
            if resh_shape.size != 4:
                return self.none(node, inspect.currentframe().f_lineno)
        elif gqa_reshape.op_type == "Squeeze":
            if resh_shape.size != 1:
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


class FunctionAttentionGQAPattern(FunctionAttentionPattern, _CommonGQAMethods):
    """
    Merges onnx nodes equivalent to repeat interleave followed by function
    ``LocalAttention`` into ``LocalAttentionGQA`` (GQA for GroupQueryAttention).

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
                "cat", onnx.TensorProto.FLOAT, shape=("batch", 4, "past_length+seq_length", 32)
            )
        )
        inputs.append(
            oh.make_tensor_value_info("init1_s_::RSh1", onnx.TensorProto.FLOAT, shape=(1,))
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
        inputs.append(
            oh.make_tensor_value_info(
                "cat_1",
                onnx.TensorProto.FLOAT,
                shape=("batch", 4, "past_length+seq_length", 32),
            )
        )
        inputs.append(
            oh.make_tensor_value_info(
                "query", onnx.TensorProto.FLOAT, shape=("batch", 8, "seq_length", 32)
            )
        )
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["init7_s1_2"],
                value=onh.from_array(np.array([2], dtype=np.int64), name="value"),
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
        nodes.append(oh.make_node("Unsqueeze", ["cat", "init7_s1_2"], ["cat::UnSq2"]))
        nodes.append(
            oh.make_node(
                "Expand", ["cat::UnSq2", "init7_s5_1_1_2_1_1"], ["_onx_expand_cat::UnSq2"]
            )
        )
        nodes.append(
            oh.make_node(
                "Reshape",
                ["_onx_expand_cat::UnSq2", "init7_s4_0_8_-1_32"],
                ["repeat_interleave_1"],
            )
        )
        nodes.append(oh.make_node("Unsqueeze", ["cat_1", "init7_s1_2"], ["cat_1::UnSq2"]))
        nodes.append(
            oh.make_node(
                "Expand", ["cat_1::UnSq2", "init7_s5_1_1_2_1_1"], ["_onx_expand_cat_1::UnSq2"]
            )
        )
        nodes.append(
            oh.make_node(
                "Reshape",
                ["_onx_expand_cat_1::UnSq2", "init7_s4_0_8_-1_32"],
                ["repeat_interleave"],
            )
        )
        nodes.append(
            oh.make_node(
                "LocalAttentionSW_to1",
                ["query", "repeat_interleave_1", "repeat_interleave", "to", "init1_s_::RSh1"],
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
                "cat", onnx.TensorProto.FLOAT, shape=("batch", 4, "past_length+seq_length", 32)
            )
        )
        inputs.append(
            oh.make_tensor_value_info("init1_s_::RSh1", onnx.TensorProto.FLOAT, shape=(1,))
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
        inputs.append(
            oh.make_tensor_value_info(
                "cat_1",
                onnx.TensorProto.FLOAT,
                shape=("batch", 4, "past_length+seq_length", 32),
            )
        )
        inputs.append(
            oh.make_tensor_value_info(
                "query", onnx.TensorProto.FLOAT, shape=("batch", 8, "seq_length", 32)
            )
        )
        nodes.append(
            oh.make_node(
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

    _operator_gqa_name = f"{FunctionAttentionPattern._operator_name}GQA"

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

        # Final verification, let's check none the nodes is used outside the pattern.
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
            if n and g.is_used_more_than_once(n.output[0]):
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
        gqa = "" if gqa_reshape.op_type == "Reshape" else "sQ"
        name = f"{self._operator_gqa_name}{gqa}{attn.op_type[len(self._operator_name):]}"
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
                g.builder,
                name,
                itype=itype,
                gqa=True,
                switch_where="SW" in attn.op_type,
                use_qga_squeeze=gqa_reshape_v.op_type == "Squeeze",
            )
        return attention_nodes


class AttentionGQAPattern(PatternOptimization, _CommonGQAMethods):
    """
    Fuses LocalAttention into Attention.
    Opset must be >= 23 to do so.

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
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info("key", onnx.TensorProto.FLOAT, shape=("a", 2, "c", 8))
        )
        inputs.append(
            oh.make_tensor_value_info("mask", onnx.TensorProto.BOOL, shape=("a", 1, "c", "c+h"))
        )
        inputs.append(
            oh.make_tensor_value_info("value", onnx.TensorProto.FLOAT, shape=("a", 2, "c", 8))
        )
        inputs.append(
            oh.make_tensor_value_info(
                "past_key", onnx.TensorProto.FLOAT, shape=("a", 2, "h", 8)
            )
        )
        inputs.append(
            oh.make_tensor_value_info("query", onnx.TensorProto.FLOAT, shape=("a", 4, "c", 8))
        )
        inputs.append(
            oh.make_tensor_value_info(
                "past_value", onnx.TensorProto.FLOAT, shape=("a", 2, "h", 8)
            )
        )
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["two"],
                value=onh.from_array(np.array([2], dtype=np.int64), name="value"),
            )
        )
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["t11211"],
                value=onh.from_array(np.array([1, 1, 2, 1, 1], dtype=np.int64), name="value"),
            )
        )
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["resh"],
                value=onh.from_array(np.array([0, 4, -1, 8], dtype=np.int64), name="value"),
            )
        )
        nodes.append(oh.make_node("Concat", ["past_key", "key"], ["present_key"], axis=2))
        nodes.append(
            oh.make_node("Concat", ["past_value", "value"], ["present_value"], axis=2)
        )
        nodes.append(oh.make_node("Unsqueeze", ["present_key", "two"], ["key_u"]))
        nodes.append(oh.make_node("Expand", ["key_u", "t11211"], ["key_ue"]))
        nodes.append(oh.make_node("Reshape", ["key_ue", "resh"], ["key_ues"]))
        nodes.append(oh.make_node("Unsqueeze", ["present_value", "two"], ["value_u"]))
        nodes.append(oh.make_node("Expand", ["value_u", "t11211"], ["value_ue"]))
        nodes.append(oh.make_node("Reshape", ["value_ue", "resh"], ["value_ues"]))
        nodes.append(
            oh.make_node(
                "Attention",
                ["query", "key_ues", "value_ues", "mask"],
                ["Y"],
                scale=0.10999999940395355,
            )
        )
        outputs.append(
            oh.make_tensor_value_info(
                "present_value", onnx.TensorProto.FLOAT, shape=("a", 2, "c+h", 8)
            )
        )
        outputs.append(
            oh.make_tensor_value_info(
                "present_key", onnx.TensorProto.FLOAT, shape=("a", 2, "c+h", 8)
            )
        )
        outputs.append(
            oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape=("a", 4, "c_", 8))
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
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        inputs.append(
            oh.make_tensor_value_info("key", onnx.TensorProto.FLOAT, shape=("a", 2, "c", 8))
        )
        inputs.append(
            oh.make_tensor_value_info("mask", onnx.TensorProto.BOOL, shape=("a", 1, "c", "c+h"))
        )
        inputs.append(
            oh.make_tensor_value_info("value", onnx.TensorProto.FLOAT, shape=("a", 2, "c", 8))
        )
        inputs.append(
            oh.make_tensor_value_info(
                "past_key", onnx.TensorProto.FLOAT, shape=("a", 2, "h", 8)
            )
        )
        inputs.append(
            oh.make_tensor_value_info("query", onnx.TensorProto.FLOAT, shape=("a", 4, "c", 8))
        )
        inputs.append(
            oh.make_tensor_value_info(
                "past_value", onnx.TensorProto.FLOAT, shape=("a", 2, "h", 8)
            )
        )
        nodes.append(
            oh.make_node(
                "Attention",
                ["query", "key", "value", "mask", "past_key", "past_value"],
                ["Y", "present_key", "present_value"],
                is_causal=0,
                scale=0.10999999940395355,
            )
        )
        outputs.append(
            oh.make_tensor_value_info(
                "present_value", onnx.TensorProto.FLOAT, shape=("a", 2, "c+h", 8)
            )
        )
        outputs.append(
            oh.make_tensor_value_info(
                "present_key", onnx.TensorProto.FLOAT, shape=("a", 2, "c+h", 8)
            )
        )
        outputs.append(
            oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape=("a", 4, "c_", 8))
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
        f"{FunctionAttentionGQAPattern._operator_gqa_name}noT_to",
        f"{FunctionAttentionGQAPattern._operator_gqa_name}SWnoT_to",
        f"{FunctionAttentionGQAPattern._operator_gqa_name}sQnoT_to",
        f"{FunctionAttentionGQAPattern._operator_gqa_name}sQSWnoT_to",
    )

    def __init__(self, verbose: int = 0, priority: int = 2):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if g.main_opset < 23:
            return self.none()
        if (
            (node.op_type != "Attention" or node.domain != "")
            and (
                not node.op_type.startswith(self._prefixes_operator_name)
                or node.domain != FunctionAttentionGQAPattern._domain_name
                or len(node.input) != 7
            )
        ) or len(node.output) > 1:
            return self.none()

        if len(node.input) > 3 and (
            not g.has_rank(node.input[3]) or g.get_rank(node.input[3]) < 2
        ):
            # Only 2D ranks allowed.
            return self.none(node, inspect.currentframe().f_lineno)

        if node.op_type == "Attention":
            if not g.has_rank(node.input[0]) and g.get_rank(node.input[0]) != 4:
                # Only 4D Attention
                return self.none(node, inspect.currentframe().f_lineno)
            # Node Attention, we still need to check if there is some GQA node.
            gqa_keys = self._match_keys_or_values(g, node, node.input[1])
            if not gqa_keys:
                return self.none(node, inspect.currentframe().f_lineno)
            gqa_values = self._match_keys_or_values(g, node, node.input[2])
            if not gqa_values:
                return self.none(node, inspect.currentframe().f_lineno)
            gqa_unsqueeze, gqa_expand, gqa_reshape, shapes = gqa_keys
            gqa_unsqueeze_v, gqa_expand_v, gqa_reshape_v, shapes_v = gqa_values
            unsq_shape, exp_shape, resh_shape = shapes
            unsq_shape_v, exp_shape_v, resh_shape_v = shapes_v

            if unsq_shape_v != unsq_shape:
                return self.none(node, inspect.currentframe().f_lineno)
            if exp_shape != exp_shape_v:
                return self.none(node, inspect.currentframe().f_lineno)
            if resh_shape_v != resh_shape:
                return self.none(node, inspect.currentframe().f_lineno)
            gqa_nodes = [
                gqa_unsqueeze,
                gqa_expand,
                gqa_reshape,
                gqa_unsqueeze_v,
                gqa_expand_v,
                gqa_reshape_v,
            ]

            concats = g.node_before(gqa_unsqueeze.input[0]), g.node_before(
                gqa_unsqueeze_v.input[0]
            )
            if None in concats:
                return self.none(node, inspect.currentframe().f_lineno)
            if len(concats[0].input) != 2 or len(concats[1].input) != 2:
                return self.none(node, inspect.currentframe().f_lineno)
            if concats[0].op_type != "Concat" or concats[1].op_type != "Concat":
                return self.none(node, inspect.currentframe().f_lineno)
            if g.get_attribute_with_default(
                concats[0], "axis", 0
            ) != g.get_attribute_with_default(concats[1], "axis", 0):
                return self.none(node, inspect.currentframe().f_lineno)

        else:
            keys, values = node.input[1:3]
            concats = g.node_before(keys), g.node_before(values)
            if None in concats:
                return self.none(node, inspect.currentframe().f_lineno)
            if len(concats[0].input) != 2 or len(concats[1].input) != 2:
                return self.none(node, inspect.currentframe().f_lineno)
            if concats[0].op_type != "Concat" or concats[1].op_type != "Concat":
                return self.none(node, inspect.currentframe().f_lineno)
            if g.get_attribute_with_default(
                concats[0], "axis", 0
            ) != g.get_attribute_with_default(concats[1], "axis", 0):
                return self.none(node, inspect.currentframe().f_lineno)

            # Local function
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
            if "GQAsQ" in node.op_type:
                # This is an axis for a Squeeze node.
                if not g.get_shape(node.input[1]):
                    # We need that shape to get kv_num_heads.
                    return self.none(node, inspect.currentframe().f_lineno)
            else:
                # This is a shape for a Reshape node.
                if shape_or_axis[1] <= 0:
                    return self.none(node, inspect.currentframe().f_lineno)
            gqa_nodes = [None for _ in range(6)]

        # Final verification, let's check none the nodes is used outside the pattern.
        nodes = [*concats, *gqa_nodes, node]
        for n in nodes[2:-1]:
            if n and g.is_used_more_than_once(n.output[0]):
                return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, nodes, self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        keys_concat_node: NodeProto,
        values_concat_node: NodeProto,
        gqa_unsqueeze: Optional[NodeProto],
        gqa_expand: Optional[NodeProto],
        gqa_reshape: Optional[NodeProto],
        gqa_unsqueeze_v: Optional[NodeProto],
        gqa_expand_v: Optional[NodeProto],
        gqa_reshape_v: Optional[NodeProto],
        local_attention_gqa: Optional[NodeProto],
    ) -> List[NodeProto]:
        query, _keys, _values, mask = local_attention_gqa.input[:4]
        attn_kwargs = {}
        if local_attention_gqa.op_type == "Attention":
            scale = g.get_attribute_with_default(local_attention_gqa, "scale", None)
            if scale is not None:
                attn_kwargs["scale"] = scale
            attn_kwargs["is_causal"] = g.get_attribute_with_default(
                local_attention_gqa, "is_causal", 0
            )
        else:
            scale = g.get_constant_scalar(local_attention_gqa.input[4]) ** 2  # this scale ** 0.5
            attn_kwargs["scale"] = scale

        # In case we need the 3D pattern.
        # expand_shape = g.get_computed_constant(local_attention_gqa.input[5])
        # repeat = int(expand_shape[2])
        # if "sQ_" in local_attention_gqa.op_type:
        #    k_shape = g.get_shape(local_attention_gqa.input[1])
        #    kv_num_heads = k_shape[1]
        # else:
        #    reshape_shape = g.get_computed_constant(local_attention_gqa.input[6])
        #    kv_num_heads = reshape_shape[1] // repeat
        #
        # num_heads = kv_num_heads * repeat

        nodes = []

        final_mask = mask
        if mask:
            switch_where = "SW" in local_attention_gqa.op_type
            if switch_where:
                # mask is not mask if SW
                if g.get_type(mask) == TensorProto.BOOL:
                    final_mask = g.unique_name(f"{self.__class__.__name__}--{mask}")
                    nodes.append(g._make_node("Not", [mask], [final_mask]))
                else:
                    raise NotImplementedError(
                        f"float mask is not implemented yet for pattern "
                        f"{self.__class__.__name__!r}"
                    )

        nodes.extend(
            [
                g._make_node(
                    "Attention",
                    [
                        query,
                        keys_concat_node.input[1],
                        values_concat_node.input[1],
                        final_mask,
                        keys_concat_node.input[0],
                        values_concat_node.input[0],
                    ],
                    [
                        local_attention_gqa.output[0],
                        keys_concat_node.output[0],
                        values_concat_node.output[0],
                    ],
                    # q_num_heads=num_heads,
                    # kv_num_heads=kv_num_heads,
                    **attn_kwargs,
                ),
            ]
        )
        for node in nodes:
            if not node.name:
                node.name = g.builder.unique_node_name(
                    f"{self.__class__.__name__}--{local_attention_gqa.name}"
                )
        return nodes
