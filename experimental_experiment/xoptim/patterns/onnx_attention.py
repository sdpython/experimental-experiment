import inspect
from typing import List, Optional
import numpy as np
from onnx import NodeProto
from ...helpers import tensor_dtype_to_np_dtype
from ...xbuilder import FunctionOptions, GraphBuilder
from ..patterns_api import MatchResult, PatternOptimization


class FunctionAttentionPattern(PatternOptimization):
    """Merges Attention nodes into a local funtcion."""

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
        add_node = g.node_before(node.input[0])
        if add_node is None or add_node.op_type != "Add":
            return self.none(node, inspect.currentframe().f_lineno)
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
        if mat_qk is None or mat_qk.op_type != "MatMul":
            return self.none(node, inspect.currentframe().f_lineno)
        mul1 = g.node_before(mat_qk.input[0])
        if mul1 is None or mul1.op_type != "Mul":
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant_scalar(mul1.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)

        transpose = g.node_before(mat_qk.input[1])
        if transpose is None or transpose.op_type != "Transpose":
            return self.none(node, inspect.currentframe().f_lineno)
        perm = g.get_attribute(transpose, "perm").ints
        if tuple(perm) != (0, 1, 3, 2):
            return self.none(node, inspect.currentframe().f_lineno)
        mul2 = g.node_before(transpose.input[0])
        if mul2 is None or mul2.op_type != "Mul":
            return self.none(node, inspect.currentframe().f_lineno)
        if mul2.input[1] != mul1.input[1]:
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

        for n in [mul1, mul2, transpose, mat_qk, where_node, add_node, isnan, where2]:
            if g.is_used_more_than_once(n.output[0]):
                return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(
            self,
            [mul1, mul2, transpose, mat_qk, where_node, add_node, node, isnan, where2, mat_qkv],
            self.apply,
        )

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        mul1: NodeProto,
        mul2: NodeProto,
        transpose: NodeProto,
        mat_qk: NodeProto,
        where_node: NodeProto,
        add_node: NodeProto,
        softmax: NodeProto,
        isnan: NodeProto,
        where: NodeProto,
        mat_qkv: NodeProto,
    ) -> List[NodeProto]:
        itype = g.get_type(mul1.input[1])
        name = f"{self._operator_name}_to{itype}"
        attention_nodes = [
            g.make_node(
                name,
                [
                    mul1.input[0],
                    mul2.input[0],
                    mat_qkv.input[1],
                    where_node.input[0],
                    mul1.input[1],
                ],
                [mat_qkv.output[0]],
                name=f"{self.__class__.__name__}--{softmax.name}",
                domain=self._domain_name,
            )
        ]
        nodes_to_return = attention_nodes

        # Creates the local function
        if not g.builder.has_local_function(name, domain=self._domain_name):
            self._add_local_function(g.builder, name, itype=itype)
        return nodes_to_return

    @classmethod
    def _add_local_function(cls, g: GraphBuilder, name: str, itype: int):
        lg = GraphBuilder(g.main_opset, as_function=True)
        lg.make_tensor_input("query")
        lg.make_tensor_input("keys")
        lg.make_tensor_input("values")
        lg.make_tensor_input("mask")
        lg.make_tensor_input("scale_sqrt")

        scaled_query = lg.op.Mul("query", "scale_sqrt", name=cls.__name__)
        scaled_keys = lg.op.Mul("keys", "scale_sqrt", name=cls.__name__)
        scaled_keys_t = lg.op.Transpose(scaled_keys, perm=(0, 1, 3, 2), name=cls.__name__)
        qk = lg.op.MatMul(scaled_query, scaled_keys_t, name=cls.__name__)
        dtype = tensor_dtype_to_np_dtype(itype)
        zero = np.array([0], dtype=dtype)
        minfty = np.array([-np.inf], dtype=dtype)
        bias = lg.op.Where("mask", zero, minfty, name=cls.__name__)
        softmax = lg.op.Softmax(
            lg.op.Add(qk, bias, name=cls.__name__), axis=-1, name=cls.__name__
        )

        filtered = lg.op.Where(
            lg.op.IsNaN(softmax, name=cls.__name__), zero, softmax, name=cls.__name__
        )
        lg.op.MatMul(filtered, "values", outputs=["Y"], name=cls.__name__)

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
