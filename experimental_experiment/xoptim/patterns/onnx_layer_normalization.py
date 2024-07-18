import inspect
from typing import List, Optional
import numpy as np
from onnx import NodeProto
from onnx.helper import tensor_dtype_to_np_dtype
from ..patterns_api import MatchResult, PatternOptimization


class LayerNormalizationPattern(PatternOptimization):
    """
    Fuses node of a LayerNormalization.
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "ReduceMean" or node.domain != "":
            return self.none()

        # before

        pow = g.node_before(node.input[0])
        if pow is None:
            return self.none(node, inspect.currentframe().f_lineno)
        if pow.op_type != "Pow" or len(g.next_nodes(pow.output[0])) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        if (
            not g.is_constant_scalar(pow.input[1])
            or g.get_constant_scalar(pow.input[1]) != 2
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        sub = g.node_before(pow.input[0])
        if sub is None:
            return self.none(node, inspect.currentframe().f_lineno)
        if sub.op_type != "Sub" or len(g.next_nodes(sub.output[0])) != 2:
            return self.none(node, inspect.currentframe().f_lineno)
        red = g.node_before(sub.input[1])
        if red is None:
            return self.none(node, inspect.currentframe().f_lineno)
        if red.op_type != "ReduceMean" or len(g.next_nodes(red.output[0])) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        if sub.input[0] != red.input[0]:
            return self.none(node, inspect.currentframe().f_lineno)
        kp = g.get_attribute(red, "keepdims", exc=False)
        if kp is None or kp.i != 1:
            return self.none(node, inspect.currentframe().f_lineno)

        # after
        add = g.next_nodes(node.output[0])
        if len(add) != 1 or add[0].op_type != "Add":
            return self.none(node, inspect.currentframe().f_lineno)
        add = add[0]
        if not g.is_constant_scalar(add.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        sqrt = g.next_nodes(add.output[0])
        if len(sqrt) != 1 or sqrt[0].op_type != "Sqrt":
            return self.none(node, inspect.currentframe().f_lineno)
        sqrt = sqrt[0]
        div = g.next_nodes(sqrt.output[0])
        if len(div) != 1 or div[0].op_type != "Div":
            return self.none(node, inspect.currentframe().f_lineno)
        div = div[0]
        if div.input[0] != sub.output[0]:
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(
            self, [red, sub, pow, node, add, sqrt, div], self.apply, insert_at=node
        )

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        red: NodeProto,
        sub: NodeProto,
        pow: NodeProto,
        node: NodeProto,
        add: NodeProto,
        sqrt: NodeProto,
        div: NodeProto,
    ) -> List[NodeProto]:
        eps = g.get_constant_scalar(add.input[1])
        dtype = tensor_dtype_to_np_dtype(g.get_type(red.input[0]))
        scale = g.make_initializer("", np.array([1], dtype=dtype))
        new_node = g.make_node(
            "LayerNormalization",
            [red.input[0], scale],
            [div.output[0], div.input[1]],
            epsilon=float(eps),
            name=f"{self.__class__.__name__}--{node.name}",
            doc_string=node.doc_string,
        )
        return [new_node]


class LayerNormalizationScalePattern(PatternOptimization):
    """
    Fused LayerNormalization, scale, bias just after.
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "LayerNormalization" or node.domain != "":
            return self.none()

        nodes = g.next_nodes(node.output[0])
        if len(nodes) != 1 or nodes[0].op_type != "Mul":
            return self.none(node, inspect.currentframe().f_lineno)
        mul_node = nodes[0]
        nodes = g.next_nodes(mul_node.output[0])
        if len(nodes) == 0:
            return MatchResult(
                self, [node, mul_node, None], self.apply, insert_at=mul_node
            )
        if len(nodes) == 1 and nodes[0].op_type == "Add":
            if len(node.input) != 2:
                return self.none(node, inspect.currentframe().f_lineno)
            return MatchResult(
                self, [node, mul_node, nodes[0]], self.apply, insert_at=nodes[0]
            )
        return self.none(node, inspect.currentframe().f_lineno)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        ln_node: NodeProto,
        mul_node: NodeProto,
        add_node: Optional[NodeProto],
    ) -> List[NodeProto]:

        # scale
        scale = (
            mul_node.input[1]
            if mul_node.input[0] == ln_node.output[0]
            else mul_node.input[0]
        )
        new_scale = None
        if g.is_constant_scalar(ln_node.input[1]):
            fscale = g.get_constant_scalar(ln_node.input[1])
            if fscale == 1:
                new_scale = scale
        new_nodes = []
        if new_scale is None:
            new_scale = g.unique_name(f"{self.__class__.__name__}_{ln_node.input[1]}")
            node = g.make_node(
                "Mul",
                [ln_node.input[1], scale],
                [new_scale],
                name=f"{self.__class__.__name__}--{ln_node.name}",
            )
            new_nodes.append(node)

        if add_node:
            assert len(ln_node.input) == 2, "Only two inputs are expected in node."
            new_bias = (
                add_node.input[1]
                if add_node.input[0] == mul_node.output[0]
                else add_node.input[0]
            )
        else:
            new_bias = ln_node.input[2] if len(ln_node.input) > 2 else None

        kwargs = {}
        axis = g.get_attribute(ln_node, "axis", exc=None)
        if axis:
            kwargs["axis"] = axis.i
        epsilon = g.get_attribute(ln_node, "epsilon", exc=None)
        if epsilon:
            kwargs["epsilon"] = epsilon.f
        stash_type = g.get_attribute(ln_node, "stash_type", exc=None)
        if stash_type:
            kwargs["stash_type"] = stash_type.i

        new_node = g.make_node(
            "LayerNormalization",
            (
                [ln_node.input[0], new_scale, new_bias]
                if new_bias
                else [ln_node.input[0], new_scale]
            ),
            [(add_node or mul_node).output[0], ln_node.input[1]],
            name=f"{self.__class__.__name__}--{ln_node.name}",
            doc_string=ln_node.doc_string,
            **kwargs,
        )
        return [*new_nodes, new_node]
