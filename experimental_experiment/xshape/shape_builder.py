from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import onnx
import onnx.helper as oh
from onnx_diagnostic.helpers import string_type
from onnx_diagnostic.helpers.onnx_helper import dtype_to_tensor_dtype
from ..helpers import make_hash
from ._shape_helper import DYNAMIC_SHAPE
from .evaluate_expressions import evaluate_expression
from ._onnx_helper import (
    element_wise_binary_op_types,
    element_wise_op_cmp_types,
    unary_like_op_types,
)


class ShapeBuilder:
    """API for a class computing shapes in an ONNX model."""

    _op_type_element_wise_types = element_wise_binary_op_types()
    _op_type_element_wise_cmp_types = element_wise_op_cmp_types()
    _op_type_unary_like = unary_like_op_types()

    @property
    def input_names(self) -> List[str]:
        raise NotImplementedError(f"not overloaded in {self.__class__.__name__!r}")

    @property
    def output_names(self) -> List[str]:
        raise NotImplementedError(f"not overloaded in {self.__class__.__name__!r}")

    def get_shape(self, name: str) -> DYNAMIC_SHAPE:
        raise NotImplementedError(f"not overloaded in {self.__class__.__name__!r}")

    def set_shape(self, name: str, shape: DYNAMIC_SHAPE):
        raise NotImplementedError(f"not overloaded in {self.__class__.__name__!r}")

    def get_type(self, name: str) -> int:
        raise NotImplementedError(f"not overloaded in {self.__class__.__name__!r}")

    def set_type(self, name: str, itype: int):
        raise NotImplementedError(f"not overloaded in {self.__class__.__name__!r}")

    def get_rank(self, name: str) -> int:
        raise NotImplementedError(f"not overloaded in {self.__class__.__name__!r}")

    def set_rank(self, name: str, rank: int):
        raise NotImplementedError(f"not overloaded in {self.__class__.__name__!r}")

    def register_constraint_dimension(self, dim_name: str, value: Any):
        raise NotImplementedError(f"not overloaded in {self.__class__.__name__!r}")

    def _hash(self) -> str:
        return make_hash(self)

    def update_shapes(self, model: onnx.ModelProto):
        """Updates model shapes with the value stored inside this graph."""
        self._update_shapes_graph(model.graph)

    def _update_shapes_graph(self, graph: onnx.GraphProto):
        exclude = (
            set(i.name for i in graph.input)
            | set(i.name for i in graph.output)
            | set(i.name for i in graph.initializer)
            | set(i.name for i in graph.sparse_initializer)
        )
        include = set()
        for node in graph.node:
            include |= set(node.output)
        include -= exclude
        include -= set(i.name for i in graph.value_info)
        ordered_include = []
        for node in graph.node:
            for o in node.output:
                if o in include:
                    ordered_include.append(o)
        infos = []
        for k in ordered_include:
            if not self.has_shape(k):
                continue
            infos.append(oh.make_tensor_value_info(k, self.get_type(k), list(self.get_shape(k))))
        graph.value_info.extend(infos)

    def get_attribute(
        self, node: onnx.NodeProto, att_name: str, exc: bool = True
    ) -> Optional[onnx.AttributeProto]:
        """Returns an attribute for a node."""
        for att in node.attribute:
            if att.name == att_name:
                return att
        assert not exc, (
            f"Unable to find attribute {att_name!r} for node "
            f"type {node.op_type!r} in node {node}"
        )
        return None

    def get_attribute_with_default(
        self, node: onnx.NodeProto, name: str, default_value: Any
    ) -> Any:
        """
        Returns an attribute or its default value if missing.

        :param node: node
        :param name: attribute name
        :param default_value: default value
        :return: value
        """
        for att in node.attribute:
            if att.name == name:
                if att.type == onnx.AttributeProto.INT:
                    return att.i
                if att.type == onnx.AttributeProto.INTS:
                    return list(att.ints)
                if att.type == onnx.AttributeProto.FLOAT:
                    return att.f
                if att.type == onnx.AttributeProto.FLOATS:
                    return list(att.floats)
                if att.type == onnx.AttributeProto.STRING:
                    return att.s
                raise TypeError(
                    f"Not implemented for attribute name {att.name!r}, attribute={att}"
                )
        return default_value

    def get_attributes_with_default(
        self, node: onnx.NodeProto, **default_values
    ) -> Dict[str, Any]:
        """
        Returns int or float attributes. If missing, the default value is returned
        if it is not None.

        :param node: node
        :param default_values: default values
        """
        res = {}
        for att in node.attribute:
            if att.name in default_values:
                if att.type == onnx.AttributeProto.INT:
                    res[att.name] = att.i
                elif att.type == onnx.AttributeProto.INTS:
                    res[att.name] = list(att.ints)
                elif att.type == onnx.AttributeProto.FLOAT:
                    res[att.name] = att.f
                elif att.type == onnx.AttributeProto.FLOATS:
                    res[att.name] = list(att.floats)
                elif att.type == onnx.AttributeProto.STRING:
                    res[att.name] = att.s
                else:
                    raise TypeError(
                        f"Not implemented for attribute name {att.name!r}, attribute={att}"
                    )
        for k, v in default_values.items():
            if k not in res and v is not None:
                res[k] = v
        res = {k: v for k, v in res.items() if v is not None}
        return res

    def pretty_node(
        self,
        node: Optional[onnx.NodeProto],
        limit: int = 80,
        short: bool = True,
        shape: bool = False,
    ) -> str:
        """
        Pretty rendering for a node.

        :param node: node to render
        :param limit: to show type and shapes after the limit
        :param short: do not display shape information on the left
        :param shape: show shape information below
        :return: string
        """
        if node is None:
            return "None"
        if shape:
            st = []
            for i in node.input:
                dt = self.get_type(i) if self.has_type(i) else "-"
                sh = (
                    "x".join(str(_).replace(" ", "") for _ in self.get_shape(i))
                    if self.has_shape(i)
                    else (f"rk={self.get_rank(i)}" if self.has_rank(i) else "?")
                )
                st.append(f"{i}:{dt}|{sh}")
            st.append("->")
            for i in node.output:
                dt = self.get_type(i) if self.has_type(i) else "-"
                sh = (
                    "x".join(str(_).replace(" ", "") for _ in self.get_shape(i))
                    if self.has_shape(i)
                    else (f"rk={self.get_rank(i)}" if self.has_rank(i) else "?")
                )
                st.append(f"{i}:{dt}|{sh}")
            shape_info = " ".join(st)
        else:
            shape_info = ""
        text = (
            (
                f"{node.op_type}[{node.domain}]: "
                f"{', '.join(node.input)} -> {', '.join(node.output)}"
            )
            if node.domain
            else f"{node.op_type}: {', '.join(node.input)} -> {', '.join(node.output)}"
        )
        if shape_info:
            text = f"{text} ## {shape_info}"
        if short:
            return text
        add = " " * abs(80 - len(text))
        text += add
        info = []
        for o in node.output:
            t = f"T{self.get_type(o)}" if self.has_type(o) else ""
            s = " x ".join(map(str, self.get_shape(o))) if self.has_shape(o) else ""
            info.append(": ".join([t, s]))
        if node.name:
            s = f"{text}|{' '.join(info)}"
            return f"{s}{' ' * (110 - len(s))}- {node.name}"
        return f"{text}|{' '.join(info)}"

    def map_value_info_dimension_with_true_values(self, name: str, tensor: np.ndarray):
        assert self.has_type(name), f"Missing type for {name!r}."
        assert self.has_shape(name), f"Missing shape for {name!r}."
        dtype = dtype_to_tensor_dtype(tensor.dtype)
        assert dtype == self.get_type(name), (
            f"Type mismatch for {name!r}, expecting "
            f"{self.get_type(name)}, got {dtype} in "
            f"{string_type(tensor, with_shapes=True)}"
        )
        res = {}
        shape = self.get_shape(name)
        for i, (value, dim) in enumerate(zip(tensor.shape, shape)):
            if isinstance(dim, str):
                if dim in res:
                    assert res[dim] == value, (
                        f"Shape mismatch for {name!r} for dimension {i}, "
                        f"known dimensions are {shape}, got "
                        f"{string_type(tensor, with_shapes=True)}"
                    )
                res[dim] = value
            else:
                assert dim == value, (
                    f"Shape mismatch for {name!r} for dimension {i}, "
                    f"expecting {dim}, got {string_type(tensor, with_shapes=True)}"
                )
        return res

    def evaluate_shape(self, name: str, context: Dict[str, int]) -> Tuple[int, ...]:
        shape = self.get_shape(name)
        return tuple(evaluate_expression(s, context) for s in shape)

    def compare_computed_shape_with_tensor(
        self, name: str, tensor: np.ndarray, context: Dict[str, int]
    ) -> Tuple[Tuple[str, int, int], ...]:
        assert self.has_type(name), f"Missing type for {name!r}."
        assert self.has_shape(name), f"Missing shape for {name!r}."
        dtype = dtype_to_tensor_dtype(tensor.dtype)
        assert dtype == self.get_type(name), (
            f"Type mismatch for {name!r}, expecting "
            f"{self.get_type(name)}, got {dtype} in "
            f"{string_type(tensor, with_shapes=True)}"
        )
        computed = self.evaluate_shape(name, context=context)
        return tuple(zip(self.get_shape(name), tensor.shape, computed))

    def compare_with_true_inputs(
        self,
        inputs: Union[Dict[str, np.ndarray], List[np.ndarray]],
        outputs: Union[Dict[str, np.ndarray], List[np.ndarray]],
        exc: bool = True,
    ) -> Dict[str, Tuple[Tuple[str, int, int], ...]]:
        """
        Compares the shape of the outputs with what the output shapes would return.

        :param inputs: inputs
        :param outputs: outputs
        :param exc: raises an exception when a discrepancy is met
        :return: list of expression, expected value, computed value
        """
        if isinstance(inputs, list):
            inputs = dict(zip(self.input_names, inputs))
        if isinstance(outputs, list):
            outputs = dict(zip(self.output_names, outputs))

        context = {}
        for name in self.input_names:
            res = self.map_value_info_dimension_with_true_values(name, inputs[name])
            for k, v in res.items():
                if k not in context:
                    context[k] = v
                    continue
                if context[k] != res[k]:
                    assert not exc, (
                        f"Dimension mismatch for dimension {k!r}, previous value is "
                        f"{context[k]}, new value is {res[k]} for name={name!r}"
                    )

        final = {}
        for name, tensor in outputs.items():
            res = self.compare_computed_shape_with_tensor(name, tensor, context)
            for dim, expected, got in res:
                assert not exc or expected == got, (
                    f"Output dimension mismatch for {dim!r} for results {name!r}, "
                    f"expected is {expected!r}, got {got!r}."
                )
            final[name] = res
        return final
