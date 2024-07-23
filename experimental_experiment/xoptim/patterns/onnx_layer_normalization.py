import inspect
from typing import List, Optional
import numpy as np
from onnx import NodeProto
from onnx.helper import tensor_dtype_to_np_dtype
from onnx.numpy_helper import from_array
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

        if not g.is_constant(node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        axis = g.get_computed_constant(node.input[1])
        if axis.tolist() != [-1]:
            if not g.has_rank(node.input[0]):
                return self.none(node, inspect.currentframe().f_lineno)
            rk = g.get_rank(node.input[0])
            al = axis.tolist()
            if al != list(range(rk - len(al), rk)):
                return self.none(node, inspect.currentframe().f_lineno)

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
        if not g.is_constant(red.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        axis2 = g.get_computed_constant(red.input[1])
        if axis.tolist() != axis2.tolist():
            return self.none(node, inspect.currentframe().f_lineno)
        if sub.input[0] != red.input[0]:
            return self.none(node, inspect.currentframe().f_lineno)
        kp = g.get_attribute(red, "keepdims", exc=False)
        if kp is None or kp.i != 1:
            return self.none(node, inspect.currentframe().f_lineno)

        # after
        add = g.next_nodes(node.output[0])
        if len(add) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        if add[0].op_type == "Add":
            add = add[0]
            if not g.is_constant_scalar(add.input[1]):
                return self.none(node, inspect.currentframe().f_lineno)
            sqrt = g.next_nodes(add.output[0])
        else:
            add = None
        if add is None:
            sqrt = g.next_nodes(node.output[0])
        if len(sqrt) != 1 or sqrt[0].op_type != "Sqrt":
            return self.none(node, inspect.currentframe().f_lineno)
        sqrt = sqrt[0]
        div = g.next_nodes(sqrt.output[0])
        if len(div) != 1 or div[0].op_type != "Div":
            return self.none(node, inspect.currentframe().f_lineno)
        div = div[0]
        if len(g.next_nodes(div.input[1])) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
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
        add: Optional[NodeProto],
        sqrt: NodeProto,
        div: NodeProto,
    ) -> List[NodeProto]:
        itype = g.get_type(red.input[0])
        dtype = tensor_dtype_to_np_dtype(itype)

        axis = g.get_computed_constant(red.input[1]).tolist()
        scale = None
        dtype = tensor_dtype_to_np_dtype(g.get_type(red.input[0]))
        new_nodes = []
        if axis == [-1]:
            ly_axis = -1
            if g.has_shape(red.input[0]):
                shape = g.get_shape(red.input[0])
                if isinstance(shape[-1], int):
                    scale = g.make_initializer("", np.ones((shape[-1],), dtype=dtype))
                    bias = g.make_initializer("", np.zeros((shape[-1],), dtype=dtype))
        else:
            ly_axis = min(axis)
        if scale is None:
            shape = g.unique_name(f"{self.__class__.__name__}_Sh_{red.input[0]}")
            new_nodes.append(
                g.make_node(
                    "Shape",
                    [red.input[0]],
                    [shape],
                    start=ly_axis,
                    name=f"{self.__class__.__name__}--{red.name}",
                )
            )
            scale = g.unique_name(f"{self.__class__.__name__}_Sc_{red.input[0]}")
            new_nodes.append(
                g.make_node(
                    "ConstantOfShape",
                    [shape],
                    [scale],
                    name=f"{self.__class__.__name__}--{red.name}",
                    value=from_array(np.array([1], dtype=dtype)),
                )
            )
            bias = g.unique_name(f"{self.__class__.__name__}_Bi_{red.input[0]}")
            new_nodes.append(
                g.make_node(
                    "ConstantOfShape",
                    [shape],
                    [bias],
                    name=f"{self.__class__.__name__}--{red.name}",
                    value=from_array(np.array([0], dtype=dtype)),
                )
            )

        eps = g.get_constant_scalar(add.input[1]) if add else 9.999999960041972e-13

        new_nodes.append(
            g.make_node(
                "LayerNormalization",
                [red.input[0], scale, bias],
                [div.output[0]],
                epsilon=float(eps),
                name=f"{self.__class__.__name__}--{node.name}",
                doc_string=node.doc_string,
                stash_type=1,  # itype,
                axis=ly_axis,
            )
        )
        return new_nodes


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

        if len(node.output) != 1:
            # No need for the scale.
            return self.none(node, inspect.currentframe().f_lineno)

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
            [(add_node or mul_node).output[0]],
            name=f"{self.__class__.__name__}--{ln_node.name}",
            doc_string=ln_node.doc_string,
            **kwargs,
        )
        return [*new_nodes, new_node]
