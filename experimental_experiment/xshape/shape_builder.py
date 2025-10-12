import pprint
from typing import Dict, Optional, Tuple, Union
import onnx
from onnx_diagnostic.helpers import string_type
from ..helpers import make_hash
from ._shape_helper import DYNAMIC_SHAPE
from .shape_type_compute import set_shape_type_op_any, set_shape_type_custom


class ShapeBuilder:
    """API for a class computing shapes in an ONNX model."""

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

    def _hash(self) -> str:
        return make_hash(self)


class BasicShapeBuilder(ShapeBuilder):
    """Implements a basic class doing shape inference in an ONNX model."""

    def __init__(self):
        self._input_names = []
        self._output_names = []
        self._known_shapes = {}
        self._known_ranks = {}
        self._known_types = {}
        self.constraints_ = {}
        #
        self._known_value_shape = {}
        self.constants_ = {}
        self._known_torch_value = {}
        # self.dynamic_dimensions_source={}
        # self.dynamic_dimensions_source_flat={}
        # self._dynamic_examples={}

    def get_debug_msg(self, limit: int = 1000) -> str:
        """
        Returns a string providing as much information as possible
        to help the developper understand why a conversion failed.

        :param limit: limit the string if the model is big
        :return: many pieces of informations about the on going conversion
        """
        import numpy as np
        import onnx.numpy_helper as onh

        def assert_sorted(inputs):
            try:
                return sorted(inputs)
            except TypeError:
                return list(inputs)

        def _align(s, length):
            if len(s) < length:
                s += " " * (length - len(s))
            return s

        def _dtype(t):
            if hasattr(t, "dtype"):
                return t.dtype
            if hasattr(t, "data_type"):
                return t.data_type
            raise RuntimeError(f"dtype unknown for type {type(t)}-{t}.")

        def _shape(t):
            if hasattr(t, "shape"):
                return t.dtype
            if hasattr(t, "dims"):
                return tuple(t.dims)
            raise RuntimeError(f"dtype unknown for type {type(t)}-{t}.")

        def _size(t):
            if hasattr(t, "numel"):
                return t.numel()
            if hasattr(t, "size"):
                return t.size
            if hasattr(t, "dims"):
                return np.prod(tuple(t.dims))
            raise RuntimeError(f"Size unknown for type {type(t)}-{t}.")

        def _values(t):
            if hasattr(t, "detach"):

                def is_allow_non_fake_inputs_enabled():
                    from torch._guards import detect_fake_mode

                    return detect_fake_mode(t)

                if is_allow_non_fake_inputs_enabled():
                    return "FakeTensorMode enabled"
                return t.detach().cpu().flatten().tolist()
            if hasattr(t, "size"):
                return t.ravel().tolist()
            if hasattr(t, "dims"):
                a = onh.to_array(t)
                return a.ravel().tolist()
            raise RuntimeError(f"Values unknown for type {type(t)}-{t}.")

        rows = ["", "--DEBUG--"]
        hs = self._hash()
        rows.append(f"[ShapeBuilder-{hs}] Message starts")

        # if self._implicit_decisions:
        #    rows.append("--IMPLICIT DECISIONS--")
        #    rows.extend(map(str, self._implicit_decisions))
        if self.constraints_:
            rows.append("--CONSTRAINTS--")
            for a, b in assert_sorted(self.constraints_.items()):
                rows.append(f"    {a} = {b}")
        rows.append("--SHAPE--")
        # rows.append(f"dynamic_dimensions_source=
        #                   {pprint.pformat(self.dynamic_dimensions_source)}")
        # rows.append(
        #    f"dynamic_dimensions_source_flat="
        #    f"{pprint.pformat(self.dynamic_dimensions_source_flat)}"
        # )
        # rows.append(
        #    f"output_dynamic_dimensions_source_flat="
        #    f"{pprint.pformat(self.output_dynamic_dimensions_source_flat)}"
        # )
        # rows.append(f"dynamic_alias={pprint.pformat(self._dynamic_alias)[:10000]}")
        # rows.append(f"dynamic_shapes={pprint.pformat(self.dynamic_shapes)[:10000]}")
        rows.append(f"_known_shapes={pprint.pformat(self._known_shapes)[:10000]}")
        rows.append(f"_known_types={pprint.pformat(self._known_types)[:10000]}")
        short_sh = {
            k: (v if (isinstance(v, tuple) and len(v) < 10) else string_type(v, with_shape=True))
            for k, v in self._known_value_shape.items()
        }
        rows.append(f"_known_value_shape={pprint.pformat(short_sh)[:10000]}")
        rows.append(
            f"_known_constants={pprint.pformat(list(assert_sorted(self.constants_))[:10000])}"
        )
        reminaing_ranks = {
            k: v for k, v in self._known_ranks.items() if k not in self._known_shapes
        }
        rows.append(f"_known_ranks (with no shape)={pprint.pformat(reminaing_ranks )[:10000]}")

        rows.append("--TORCH-SHAPES--")
        for kk, vv in self._known_torch_value.items():
            rows.append(
                f"    {kk}: {vv} --- "
                f"{self.get_type(kk) if self.has_type(kk) else ''}:"
                f"{self.get_rank(kk) if self.has_rank(kk) else ''}:"
                f"{self.get_shape(kk) if self.has_shape(kk) else ''}:"
            )
            if len(rows) > limit:
                rows.append("...")
                break

        return "\n".join(rows)

    def run_node(self, node: onnx.NodeProto):
        """
        Uses shapes availables in the ShapeBuilder to infer the output shapes
        and types.
        """
        self._make_node_set_type_shape(node)

    def run_value_info(self, info: onnx.ValueInfoProto, is_input: bool):
        """Fills ShapeBuilder with information coming from an input or output."""
        assert info.type.tensor_type, f"info is not a tensor type: {info}"
        if is_input:
            self._input_names.append(info.name)
        else:
            self._output_names.append(info.name)
        self.set_type(info.name, info.type.tensor_type.elem_type)
        shape = info.type.tensor_type.shape
        value = tuple(d.dim_param or d.dim_value for d in shape.dim)
        self.set_shape(info.name, value)

    def run_model(
        self,
        model: Union[onnx.ModelProto, onnx.GraphProto],
        functions: Optional[Dict[Tuple[str, str], onnx.FunctionProto]] = None,
    ):
        """Runs inference over a model or a graph."""
        if isinstance(model, onnx.ModelProto):
            return self.run_model(
                model.graph, functions={(f.domain, f.name): f for f in model.functions}
            )
        assert isinstance(model, onnx.GraphProto), f"Unexpected type {type(model)} for model"
        graph = model
        for i in graph.input:
            self.run_value_info(i, True)
        for node in graph.node:
            self.run_node(node)
        for i in graph.output:
            self.run_value_info(i, False)

    def _make_node_set_type_shape(self, node: onnx.NodeProto):
        """Updates shapes for a node."""
        if node.domain == "":
            node.doc_string += "#Io1"
            set_shape_type_op_any(self, node)
        else:
            # Missing type means it is probably coming from an inlined function.
            node.doc_string += (
                "#Io3" if node.input and not self.has_type(node.input[0]) else "#Io2"
            )
            set_shape_type_custom(self, node)
