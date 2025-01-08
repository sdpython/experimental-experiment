import inspect
from typing import List, Optional
import numpy as np
from onnx import NodeProto
from ...helpers import tensor_dtype_to_np_dtype, from_array_extended
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
        if len(node.input) != 2:
            # Not defined for older opset than 18.
            return self.none(node, inspect.currentframe().f_lineno)
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
            not g.is_constant_scalar(pow.input[1], broadcast=True)
            or g.get_constant_scalar(pow.input[1], broadcast=True) != 2
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
            if not g.is_constant_scalar(add.input[1], broadcast=True):
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
        if len(div) != 1:
            return self.none(node, inspect.currentframe().f_lineno)

        div = div[0]
        if div.op_type == "Div":
            if len(g.next_nodes(div.input[1])) != 1:
                return self.none(node, inspect.currentframe().f_lineno)
            if div.input[0] != sub.output[0]:
                return self.none(node, inspect.currentframe().f_lineno)
        elif div.op_type == "Reciprocal":
            if div.input[0] != sub.output[0]:
                return self.none(node, inspect.currentframe().f_lineno)
        else:
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
                    scale = g.make_initializer(
                        "",
                        np.ones((shape[-1],), dtype=dtype),
                        source="LayerNormalizationPattern.apply.scale",
                    )
                    bias = g.make_initializer(
                        "",
                        np.zeros((shape[-1],), dtype=dtype),
                        source="LayerNormalizationPattern.apply.bias",
                    )
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
                    value=from_array_extended(np.array([1], dtype=dtype)),
                )
            )
            bias = g.unique_name(f"{self.__class__.__name__}_Bi_{red.input[0]}")
            new_nodes.append(
                g.make_node(
                    "ConstantOfShape",
                    [shape],
                    [bias],
                    name=f"{self.__class__.__name__}--{red.name}",
                    value=from_array_extended(np.array([0], dtype=dtype)),
                )
            )

        eps = (
            g.get_constant_scalar(add.input[1], broadcast=True)
            if add
            else 9.999999960041972e-13
        )

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
            return MatchResult(self, [node, mul_node, None], self.apply, insert_at=mul_node)

        index = 1 if mul_node.input[0] == node.output[0] else 0
        if not g.has_shape(mul_node.input[index]):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.has_shape(node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        if g.get_shape(mul_node.input[index]) != g.get_shape(node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)

        nodes = g.next_nodes(mul_node.output[0])
        if len(nodes) != 1 or nodes[0].op_type != "Add":
            return MatchResult(self, [node, mul_node, None], self.apply, insert_at=nodes[0])

        add_node = nodes[0]
        index = 1 if add_node.input[0] == mul_node.output[0] else 0
        if not g.has_shape(add_node.input[index]):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.has_shape(node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        if g.get_shape(add_node.input[index]) != g.get_shape(node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [node, mul_node, add_node], self.apply, insert_at=nodes[0])

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        ln_node: NodeProto,
        mul_node: NodeProto,
        add_node: Optional[NodeProto],
    ) -> List[NodeProto]:
        # scale
        scale = (
            mul_node.input[1] if mul_node.input[0] == ln_node.output[0] else mul_node.input[0]
        )
        new_scale = None
        if g.is_constant_scalar(ln_node.input[1], broadcast=True):
            fscale = g.get_constant_scalar(ln_node.input[1], broadcast=True)
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
            if len(ln_node.input) == 2:
                new_bias = (
                    add_node.input[1]
                    if add_node.input[0] == mul_node.output[0]
                    else add_node.input[0]
                )
            else:
                # there is an existing bias
                existing_bias = ln_node.input[2]
                mul_cst = (
                    mul_node.input[0]
                    if mul_node.input[1] == ln_node.output[0]
                    else mul_node.input[1]
                )
                add_cst = (
                    add_node.input[0]
                    if add_node.input[1] == mul_node.output[0]
                    else add_node.input[1]
                )

                # new_bias = existing_bias * mul_cst + add_cst
                temp = g.unique_name(f"{self.__class__.__name__}_{ln_node.input[1]}")
                new_bias = g.unique_name(f"{self.__class__.__name__}_{ln_node.input[1]}")
                new_nodes.extend(
                    [
                        g.make_node(
                            "Mul",
                            [mul_cst, existing_bias],
                            [temp],
                            name=f"{self.__class__.__name__}--{ln_node.name}",
                        ),
                        g.make_node(
                            "Add",
                            [temp, add_cst],
                            [new_bias],
                            name=f"{self.__class__.__name__}--{ln_node.name}",
                        ),
                    ]
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


class CastLayerNormalizationCastPattern(PatternOptimization):
    """
    Checks that a Cast is really needed around LayerNormalization
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:

        if node.op_type not in (
            "LayerNormalization",
            "SimplifiedLayerNormalization",
        ) or node.domain not in ("", "com.microsoft"):
            return self.none()

        if len(node.output) > 1 and g.is_used(node.output[1]):
            # No need for the scale.
            return self.none(node, inspect.currentframe().f_lineno)

        stash_type = g.get_attribute(node, "stash_type", exc=False)
        stash_itype = 1 if stash_type is None else stash_type.i

        cast_before = g.node_before(node.input[0])
        if cast_before is None or cast_before.op_type != "Cast" or cast_before.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        to = g.get_attribute(cast_before, "to")
        if to.i != stash_itype:
            return self.none(node, inspect.currentframe().f_lineno)
        if g.is_used_more_than_once(node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)

        cast_afters = g.next_nodes(node.output[0])
        if len(cast_afters) != 1:
            return self.none(node, inspect.currentframe().f_lineno)

        cast_after = cast_afters[0]
        if cast_after.op_type != "Cast" or cast_after.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        to = g.get_attribute(cast_after, "to")
        itype = g.get_type(cast_before.input[0])
        if to.i != itype:
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, [cast_before, node, cast_after], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        cast_before: NodeProto,
        node: NodeProto,
        cast_after: NodeProto,
    ) -> List[NodeProto]:
        itype = g.get_type(cast_before.input[0])
        other = []
        nodes = []
        for i in node.input[1:]:
            name = g.unique_name(f"{self.__class__.__name__}_{i}")
            other.append(name)
            nodes.append(
                g.make_node(
                    "Cast",
                    [i],
                    [name],
                    to=itype,
                    name=f"{self.__class__.__name__}--cast--{node.name}",
                )
            )

        new_node = g.make_node(
            node.op_type,
            [cast_before.input[0], *other],
            [cast_after.output[0], *node.output[1:]],
            name=f"{self.__class__.__name__}--{node.name}",
            doc_string=node.doc_string,
            domain=node.domain,
        )
        new_node.attribute.extend(node.attribute)
        return [*nodes, new_node]


class BatchNormalizationPattern(PatternOptimization):
    """
    Checks that a BatchNormalization is really needed.
    """

    def __init__(self, verbose: int = 0, priority: int = 0):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "BatchNormalization" or node.domain != "":
            return self.none()
        if len(node.output) > 1 and g.next_nodes(node.output[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        if len(node.output) > 2 and g.next_nodes(node.output[2]):
            return self.none(node, inspect.currentframe().f_lineno)

        momentum = 0.9
        epsilon = 1e-5
        training_mode = 0
        for att in node.attribute:
            if att.name == "momentum":
                momentum = att.f
            elif att.name == "epsilon":
                epsilon = att.f
            elif att.name == "training_mode":
                training_mode = att.i
        if training_mode and momentum != 0:
            return self.none(node, inspect.currentframe().f_lineno)
        if epsilon != 0:
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant(node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant(node.input[2]):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant(node.input[3]):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant(node.input[3]):
            return self.none(node, inspect.currentframe().f_lineno)

        # biases
        for z in node.input[2:4]:
            cst = g.get_computed_constant(z)
            if cst is None:
                return self.none(node, inspect.currentframe().f_lineno)
            if cst.min() == cst.max() == 0:
                continue
            return self.none(node, inspect.currentframe().f_lineno)

        # scales
        for z in [node.input[1], node.input[4]]:
            cst = g.get_computed_constant(z)
            if cst is None:
                return self.none(node, inspect.currentframe().f_lineno)
            if cst.min() == cst.max() == 1:
                continue
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [node], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node: NodeProto,
    ) -> List[NodeProto]:
        new_node = g.make_node(
            "Identity",
            node.input[:1],
            node.output[:1],
            name=f"{self.__class__.__name__}--{node.name}",
            doc_string=node.doc_string,
        )
        return [new_node]


class BatchNormalizationTrainingPattern(PatternOptimization):
    """
    Checks that a BatchNormalization in training mode can be avoided.
    """

    def __init__(self, verbose: int = 0, priority: int = 0):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "BatchNormalization" or node.domain != "":
            return self.none()
        if g.main_opset < 18:
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.has_rank(node.input[0]) or g.get_rank(node.input[0]) < 2:
            return self.none(node, inspect.currentframe().f_lineno)
        if len(node.output) > 1 and (
            not g.has_rank(node.input[1]) or g.next_nodes(node.output[1])
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        if len(node.output) > 2 and (
            not g.has_rank(node.input[2]) or g.next_nodes(node.output[2])
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        momentum = 0.9
        training_mode = 0
        for att in node.attribute:
            if att.name == "momentum":
                momentum = att.f
            elif att.name == "training_mode":
                training_mode = att.i
        if not training_mode and momentum != 1:
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [node], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node: NodeProto,
    ) -> List[NodeProto]:
        nname = f"{self.__class__.__name__}--{node.name}"
        rk = g.get_rank(node.input[0])
        axes = tuple(np.delete(np.arange(rk), 1))
        init_axes = g.make_initializer(
            "",
            np.array(list(axes), dtype=np.int64),
            source="BatchNormalizationTrainingPattern.apply.init_axes",
        )

        mean_name = g.unique_name(f"{self.__class__.__name__}_mean_{node.input[0]}")
        mean = g.make_node(
            "ReduceMean", [node.input[0], init_axes], [mean_name], keepdims=1, name=nname
        )
        centered_name = g.unique_name(f"{self.__class__.__name__}_center_{node.input[0]}")
        sub = g.make_node("Sub", [node.input[0], mean_name], [centered_name], name=nname)

        x2 = g.unique_name(f"{self.__class__.__name__}_x2_{node.input[0]}")
        mul2 = g.make_node("Mul", [centered_name, centered_name], [x2], name=nname)

        var_name = g.unique_name(f"{self.__class__.__name__}_var_{node.input[0]}")
        var = g.make_node("ReduceMean", [x2, init_axes], [var_name], keepdims=1, name=nname)

        dtype = tensor_dtype_to_np_dtype(g.get_type(node.input[0]))
        epsilon = g.get_attributes_with_default(node, epsilon=1e-5)["epsilon"]
        init_epsilon = g.make_initializer(
            "",
            np.array([epsilon], dtype=dtype),
            source="BatchNormalizationTrainingPattern.apply.init_epsilon",
        )
        vare_name = g.unique_name(f"{self.__class__.__name__}_vareps_{node.input[0]}")
        add = g.make_node("Add", [var_name, init_epsilon], [vare_name], name=nname)
        std_name = g.unique_name(f"{self.__class__.__name__}_vareps_{node.input[0]}")
        sqrt = g.make_node("Sqrt", [vare_name], [std_name], name=nname)

        new_shape = [1 for _ in range(rk)]
        new_shape[1] = -1
        new_shape = g.make_initializer(
            "",
            np.array(new_shape, dtype=np.int64),
            source="BatchNormalizationTrainingPattern.apply.new_shape",
        )

        if g.get_rank(node.input[1]) == 1:
            scale_name = g.unique_name(f"{self.__class__.__name__}_scale_{node.input[1]}")
            scale = g.make_node(
                "Reshape", [node.input[1], new_shape], [scale_name], name=nname
            )
        else:
            scale_name = node.input[1]
            scale = None

        if g.get_rank(node.input[2]) == 1:
            bias_name = g.unique_name(f"{self.__class__.__name__}_bias_{node.input[2]}")
            bias = g.make_node("Reshape", [node.input[2], new_shape], [bias_name], name=nname)
        else:
            bias_name = node.input[2]
            bias = None

        scaled_name = g.unique_name(f"{self.__class__.__name__}_scaled_{node.input[1]}")
        scaled = g.make_node("Div", [centered_name, std_name], [scaled_name], name=nname)

        scaled2_name = g.unique_name(f"{self.__class__.__name__}_scaled2_{node.input[2]}")
        scaled2 = g.make_node("Mul", [scaled_name, scale_name], [scaled2_name], name=nname)

        final = g.make_node("Add", [scaled2_name, bias_name], [node.output[0]], name=nname)

        return [
            _
            for _ in [mean, sub, mul2, var, add, sqrt, scale, bias, scaled, scaled2, final]
            if _ is not None
        ]
