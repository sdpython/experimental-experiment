import inspect
from typing import List, Optional
import numpy as np
from onnx import NodeProto
from ...helpers import tensor_dtype_to_np_dtype, from_array_extended
from ..patterns_api import MatchResult, PatternOptimization


class SimplifiedLayerNormalizationPattern(PatternOptimization):
    """Replaces the sequence Transpose, Matmul into FusedMatMul."""

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "ReduceMean" or node.domain != "":
            return self.none()
        if len(node.input) < 2:
            return self.none(node, inspect.currentframe().f_lineno)

        axis = g.get_constant_or_attribute(node, "axes", input_index=1, cvt=tuple)
        assert isinstance(axis, tuple), f"unexpected type {type(axis)} for axis"
        if len(axis) != 1:
            return self.none(node, inspect.currentframe().f_lineno)

        node_pow = g.node_before(node.input[0])
        if node_pow is None:
            return self.none(node, inspect.currentframe().f_lineno)
        if node_pow.op_type != "Pow" or node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant_scalar(node_pow.input[1], 2):
            return self.none(node, inspect.currentframe().f_lineno)

        node_add = g.next_node(node.output[0])
        if node_add.op_type != "Add" or node_add.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        if not g.is_constant_scalar(node_add.input[0]) and not g.is_constant_scalar(
            node_add.input[1]
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        node_sqrt = g.next_node(node_add.output[0])
        if node_sqrt.op_type != "Sqrt" or node_sqrt.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        node_reciprocal = g.next_node(node_sqrt.output[0])
        if (
            node_reciprocal.op_type not in ("Reciprocal", "Div")
            or node_reciprocal.domain != ""
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        if node_reciprocal.op_type == "Div":
            if node_reciprocal.input[1] != node_sqrt.output[0]:
                return self.none(node, inspect.currentframe().f_lineno)
            if not g.is_constant_scalar(node_reciprocal.input[0], 1):
                return self.none(node, inspect.currentframe().f_lineno)

        node_mul = g.next_node(node_reciprocal.output[0])
        if node_mul.op_type != "Mul" or node_mul.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        if (
            g.is_used_more_than_once(node_pow.output[0])
            or g.is_used_more_than_once(node.output[0])
            or g.is_used_more_than_once(node_add.output[0])
            or g.is_used_more_than_once(node_sqrt.output[0])
        ):
            # intermediate results are used
            return self.none(node, inspect.currentframe().f_lineno)

        mul_i = set(node_mul.input)
        cmp = {node_pow.input[0], node_reciprocal.output[0]}
        if mul_i != cmp:
            # We check the multiplication node takes the output of the div node
            # and the input of the pow node.
            return self.none(node, inspect.currentframe().f_lineno)

        nodes = [node_pow, node, node_add, node_sqrt, node_reciprocal, node_mul]
        return MatchResult(self, nodes, self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node_pow: NodeProto,
        node_reduce: NodeProto,
        node_add: NodeProto,
        node_sqrt: NodeProto,
        node_reciprocal: NodeProto,
        node_mul: NodeProto,
    ) -> List[NodeProto]:
        nname = node_reduce.name
        nodes = []
        epsilon = g.get_computed_constant(node_add.input[1])
        shape = (
            g.get_shape(node_reduce.input[0]) if g.has_shape(node_reduce.input[0]) else None
        )
        axis = g.get_constant_or_attribute(node_reduce, "axes", input_index=1)[0]
        assert shape is None or axis < len(
            shape
        ), f"axis={axis} and shape={shape} don't match for {node_reduce.input[0]!r}"
        stash_type = g.get_type(node_reduce.input[0])
        dtype = tensor_dtype_to_np_dtype(stash_type)
        if shape is not None and isinstance(shape[axis], int):
            # a constant
            scale = g.make_initializer(
                "",
                np.array([1] * shape[axis], dtype=dtype),
                source="SimplifiedLayerNormalizationPattern.apply.scale",
            )
        else:
            sh = g.make_node(
                "Shape", [node_pow.input[0]], name=f"{self.__class__.__name__}--{nname}"
            )
            axis_name = g.make_initializer(
                "",
                np.array([axis], dtype=np.int64),
                source="SimplifiedLayerNormalizationPattern.apply.axis",
            )
            ga = g.make_node(
                "Gather",
                [sh.output[0], axis_name],
                name=f"{self.__class__.__name__}--{nname}",
            )
            # sc = g.make_node_check_opset(
            #    "Unsqueeze", [ga.output[0]], axes=[0],
            #       name=f"{self.__class__.__name__}--{nname}"
            # )
            cc = g.make_node(
                "ConstantOfShape",
                [ga.output[0]],
                value=from_array_extended(np.array([1], dtype=dtype)),
                name=f"{self.__class__.__name__}--{nname}",
            )
            scale = cc.output[0]
            nodes.extend([sh, ga, cc])

        layer = g.make_node(
            "SimplifiedLayerNormalization",
            [node_pow.input[0], scale],
            [node_mul.output[0], node_reciprocal.output[0]],
            epsilon=float(epsilon[0] if epsilon.shape else epsilon),
            axis=int(axis),
            stash_type=stash_type,
            name=f"{self.__class__.__name__}--{nname}",
        )

        nodes.append(layer)
        return nodes


class SkipLayerNormalizationPattern(PatternOptimization):
    """Replaces the sequence Add + LayerNormalization into SkipLayerNormalization."""

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "LayerNormalization" or node.domain != "":
            return self.none()
        if not g.has_rank(node.input[0]) and g.get_rank(node.input[0]) not in (2, 3):
            return self.none(node, inspect.currentframe().f_lineno)
        axis = g.get_attribute(node, "axis", exc=False)
        axis = 0 if axis is None else axis.i
        if axis != -1:
            return self.none(node, inspect.currentframe().f_lineno)
        before = g.node_before(node.input[0])
        if before is None or before.op_type != "Add":
            return self.none(node, inspect.currentframe().f_lineno)
        nodes = [before, node]
        return MatchResult(self, nodes, self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        add_node: NodeProto,
        norm_node: NodeProto,
    ) -> List[NodeProto]:
        atts = []
        epsilon = g.get_attribute(norm_node, "epsilon", exc=False)
        if epsilon:
            atts.append(epsilon)
        u1 = g.unique_name("unused")
        u2 = g.unique_name("unused")
        layer = g.make_node(
            "SkipLayerNormalization",
            [*add_node.input, *norm_node.input[1:]],
            [norm_node.output[0], u1, u2, add_node.output[0]],
            name=f"{self.__class__.__name__}--{norm_node.name}",
            domain="com.microsoft",
        )
        if atts:
            layer.attribute.extend(atts)
        return [layer]
