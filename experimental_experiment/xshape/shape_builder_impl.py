import os
import pprint
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
from onnx.external_data_helper import uses_external_data
from onnx_diagnostic.helpers import string_type
from onnx.reference import ReferenceEvaluator
from ._shape_helper import DYNAMIC_SHAPE, is_static_shape
from ._builder_runtime import _BuilderRuntime
from ._shape_runtime import _ShapeRuntime
from ._inference_runtime import _InferenceRuntime
from .rename_expressions import parse_expression_tokens
from ._onnx_helper import str_tensor_proto_type
from .shape_builder import ShapeBuilder


class BasicShapeBuilder(ShapeBuilder, _BuilderRuntime, _ShapeRuntime, _InferenceRuntime):
    """
    Implements a basic class doing shape inference in an ONNX model.

    A couple of environment variables can be set to help debugging any issue.

    * ``ONNXSTOPSHAPE=<name>``: raises an exception when ``name`` receives a shape.
    * ``ONNXSTOPTYPE=<name>``: raises an exception when ``name`` receives a type.
    * ``ONNXDYNDIM=<name>``: raises an exception when dimension ``name`` is used
    * ``ONNXCST=1``: shows which constant is requested
    * ``ONNXSHAPECOMPUTE=1``: raises an exception when a shape is missing
    * ``ONNXSTOPVALUESHAPE=<name>``: more information in function dealing with shapes
    """

    def __init__(self, verbose: int = 0):
        self.verbose = verbose
        self._input_names = []
        self._output_names = []
        self._known_shapes = {}
        self._known_ranks = {}
        self._known_types = {}
        self.constraints_ = {}
        self.dynamic_dimensions_ = {}
        self.constants_ = {}
        #
        self._known_value_shape = {}
        self.constants_computed_ = {}
        self._calls = []
        # self.dynamic_dimensions_source={}
        # self.dynamic_dimensions_source_flat={}
        # self._dynamic_examples={}
        self._debug_stop_shape = os.environ.get("ONNXSTOPSHAPE", "#?#")
        self._debug_stop_type = os.environ.get("ONNXSTOPTYPE", "#?#")
        self._debug_dyn_dim = set(os.environ.get("ONNXDYNDIM", "").split(","))
        self._debug_get_constant = int(os.environ.get("ONNXCST", "0"))
        self._debug_shape_missing = int(os.environ.get("ONNXSHAPECOMPUTE", "0"))
        self._debug_value_shape = os.environ.get("ONNXSTOPVALUESHAPE", "")
        self._debug_msg = []

    @property
    def input_names(self) -> List[str]:
        return self._input_names

    @property
    def output_names(self) -> List[str]:
        return self._output_names

    def is_constant(self, name: str) -> bool:
        """Tells if a result is a constant."""
        return name in self.constants_

    def get_constant(
        self,
        name: str,
        exc: bool = True,
        computed_value: bool = False,
        as_shape: bool = False,
        multiple_outputs: bool = False,
    ) -> Union[np.ndarray, onnx.NodeProto]:
        """
        The method returns the constant *name*. It is a tensor (numpy array)
        or a NodeProto which must be evaluated.
        If *computed_value* is True, the NodeProto is evaluated with the
        ReferenceEvaluator.

        :param name: constant name
        :param exc: raise an exception if anything is impossible to do
        :param computed_value: compute the value if not a constant
        :param as_shape: returns a tuple for a shape
        :param multiple_outputs: allow multiple outputs
        :return: value
        """
        assert self.is_constant(name), f"{name!r} is not a constant{self.get_debug_msg()}"

        if as_shape:
            assert not multiple_outputs, "multiple outputs not allowed with as_shape=True"
            res = self.get_constant(name, exc, computed_value=computed_value, as_shape=False)
            if res is None:
                assert not exc, (
                    f"No constant for name={name!r}, exc={exc}, "
                    f"computed_value={computed_value}, as_shape={as_shape}, "
                    f"multiple_outputs={multiple_outputs}{self.get_debug_msg()}"
                )
                if self._debug_get_constant:
                    print(f"[ShapeBuilder-{self._hash()}.get_constant] FAIL(1) name={name!r}")
                return None
            assert multiple_outputs or not isinstance(
                res, tuple
            ), f"Multiple outputs is not allowed but type is {type(res)} for name={name!r}"
            new_res = []
            for i in res:
                new_res.append(i if isinstance(i, str) else int(i))
            return tuple(new_res)

        if name in self.constants_computed_:
            value = self.constants_computed_[name]
            assert value is not None, f"Constant is empty for name={name!r}"
            assert multiple_outputs or not isinstance(
                value, tuple
            ), f"Multiple output is not allowed but type is {type(value)} for name={name!r}"
            assert (
                not exc or value is not None
            ), f"Unable to compute value {name!r}{self.get_debug_msg()}"
            return value

        possible_value = self.constants_[name]

        if isinstance(possible_value, onnx.TensorProto):
            if uses_external_data(possible_value):
                assert not exc, (
                    f"Tensor is using external data, data_type={possible_value.data_type}, "
                    f"dims={possible_value.dims}"
                )
                return None
            v = onh.to_array(possible_value)
            assert not multiple_outputs, f"Multiple outputs is not allowed for name={name!r}"
            self.constants_computed_[name] = v
            return v

        assert isinstance(
            possible_value, onnx.TensorProto
        ), f"Unexpected type {type(possible_value)} for a constant{self.get_debug_msg()}"
        res, _ = self.compute_constant(name, exc=exc)
        if res is None:
            # The constant is too big to be computed.
            if self._debug_get_constant:
                print(f"[ShapeBuilder-{self._hash()}.get_constant] FAIL(2) name={name!r}")
            return None

        assert multiple_outputs or not isinstance(
            res, tuple
        ), f"Multiple outputs is not allowed but type is {type(res)} for name={name!r}"
        assert (
            not multiple_outputs
        ), f"get_constants not implemented when multiple_outputs=True, name={name!r}"
        if not isinstance(res, tuple):
            return res

        if len(res) == 1:
            assert multiple_outputs or not isinstance(
                value, tuple
            ), f"Multiple output is not allowed but type is {type(value)} for name={name!r}"
            assert (
                not exc or res[0] is not None
            ), f"Unable to compute value {name!r}{self.get_debug_msg()}"
            return res[0]

        index = list(possible_value.output).index(name)
        value = res[index]
        assert value is not None, f"Constant is empty for name={name!r}"
        assert multiple_outputs or not isinstance(
            value, tuple
        ), f"Multiple outputs is not allowed but type is {type(value)} for name={name!r}"
        assert (
            not exc or value is not None
        ), f"Unable to compute value {name!r}{self.get_debug_msg()}"
        return value

    def set_constant(self, name: str, value: Union[onnx.TensorProto, onnx.NodeProto]) -> bool:
        """Tells if a result is a constant."""
        assert (
            name not in self.constants_
        ), f"Constant {name!r} is already defined{self.get_debug_msg()}"
        self.constants_[name] = value
        if isinstance(value, onnx.TensorProto):
            if not self.has_type(name):
                self.set_type(name, value.data_type)
            if not self.has_shape(name):
                self.set_shape(name, tuple(value.dims))
        elif isinstance(value, onnx.NodeProto):
            for att in value.attribute:
                if att.name == "value" and att.t:
                    self.constants_[name] = att.t
                    if not self.has_type(name):
                        self.set_type(name, att.t.data_type)
                    if not self.has_shape(name):
                        self.set_shape(name, tuple(att.t.dims))
                    return
            # Let's execute the node otherwise.
            ref = ReferenceEvaluator(value)
            val = ref.run(None, {})[0]
            self.constants_computed_[name] = val
            self.set_type(name, oh.np_dtype_to_tensor_dtype(val.dtype))
            self.set_shape(name, tuple(map(int, val.shape)))
        else:
            raise TypeError(f"Unexpected type {type(value)} for value.")

    def set_value_shape(self, name: str, value: Any, equal_to: Optional[Tuple[str, str]] = None):
        """
        Sets the value for a shape result.

        :param name: name
        :param value: it cannot be empty
        :param equal_to: if specified, the value is also equal to this value

        A value can be a string (for an unknown shape, a tuple for a shape,
        an integer for a single scalar.
        """
        if self._debug_value_shape and name == self._debug_value_shape:
            raise AssertionError(
                f"Requested stop, name={name!r}, value={value!r}, equal_to={equal_to!r}"
            )

        assert isinstance(
            name, str
        ), f"Unexpected type {type(name)} for name={name!r}{self.get_debug_msg()}"
        assert not isinstance(value, tuple) or all(isinstance(d, (str, int)) for d in value), (
            f"Unexpected value for shape {name!r}, value={value!r}, "
            f"types={string_type(value)}{self.get_debug_msg()}"
        )
        if not self.has_rank(name):
            self.set_shape(name, (len(value),) if isinstance(value, tuple) else tuple())
        assert self.has_rank(name), (
            f"name={name!r}, has no rank, but it should, value={value!r}"
            f"{self.get_debug_msg()}"
        )
        assert self.get_rank(name) in (0, 1), (
            f"name={name!r} is not a shape, its rank is {self.get_rank(name)}"
            f"{self.get_debug_msg()}"
        )
        assert not isinstance(value, (int, float)) or self.get_rank(name) == 0, (
            f"Mismatch between value={value!r} and rank="
            f"{self.get_rank(name)} for name={name!r}"
            f"{self.get_debug_msg()}"
        )
        if equal_to is None:
            if name in self._known_value_shape:
                existing = self._known_value_shape[name]
                if (
                    isinstance(existing, tuple)
                    and isinstance(value, tuple)
                    and len(existing) == len(value) == 1
                    and isinstance(existing[0], str)
                ):
                    self.register_constraint_dimension("existing", value)
                    return
            assert (
                name not in self._known_value_shape or self._known_value_shape[name] == value
            ), (
                f"Shape value for {name!r} (value={value!r}) is already "
                f"registered and is different from the existing "
                f"value={value!r} (equal_to={equal_to!r}), "
                f"existing value is {self._known_value_shape.get(name, None)!r}"
                f"{self.get_debug_msg()}"
            )
            if self.verbose > 2:
                print(f"[GraphBuilder-{self._hash()}.set_value_shape] {name}:{value}")
            self._known_value_shape[name] = value
            return

        assert (
            name in equal_to
        ), f"Unexpected name={name!r}, it should be in equal_to={equal_to!r}."
        values = (
            self._known_value_shape.get(equal_to[0], None),
            self._known_value_shape.get(equal_to[1], None),
        )
        assert value in values, (
            f"Unexpected value={value} for name={name!r}, equal_to={equal_to}, "
            f"values={values}{self.get_debug_msg()}"
        )
        assert equal_to[0] in self._known_value_shape, (
            f"{equal_to[0]!r} should already registered, name={name!r}, "
            f"value={value!r}, equal_to={equal_to!r}{self.get_debug_msg()}"
        )
        # The logic is to get rid of one value instead of keeping
        # a mapping between equivalent values.
        new_value = self._known_value_shape[equal_to[0]]
        for n in equal_to:
            if n not in self._known_value_shape:
                self._known_value_shape[n] = new_value

    def has_type(self, name: str) -> Union[bool, int]:
        """Tells if a result has a type. This should be always true."""
        assert isinstance(name, str), f"Unexpected type {type(name)} for name."
        if name not in self._known_types:
            return False
        # If the type is undefined, then it has no type.
        return self._known_types[name]

    def get_type(self, name: str) -> int:
        """Returns the type of a result."""
        assert isinstance(name, str), f"Unexpected type {type(name)} for name."
        assert name in self._known_types, (
            f"Type is unknown for result {name!r}, "
            f"known_types={self._known_types}{self.get_debug_msg()}."
        )
        return self._known_types[name]

    def set_type(self, name: str, dtype: int, exc: bool = True) -> bool:
        """
        Sets the shape for a result. It is exists, it checks the new shape
        is equal to the existing one.

        :param name: name
        :param dtype: element type (an integer, ONNX), 0 (unknonw is a possible value)
        :param exc: raises an exception
        :return: returns True if there is no type conflict
        """
        assert (
            not name or name != self._debug_stop_type
        ), f"Requested stop, name={name!r}, dtype={dtype}{self.get_debug_msg()}"
        assert isinstance(name, str), f"Unexpected type {type(name)} for name."
        assert isinstance(dtype, int), f"Unexpected type {type(dtype)} for dtype."
        int_type = dtype
        if name in self._known_types:
            # 0 is undefined
            if self._known_types[name] != 0 and int_type != self._known_types[name]:
                if exc:
                    raise RuntimeError(
                        f"Type for name {name!r} already exists and it is different, "
                        f"known is {self._known_types[name]} != {int_type} (new) - "
                        f"(mapping={str_tensor_proto_type()}){self.get_debug_msg()}"
                    )
                if "warnings" not in self._debug_msg:
                    self._debug_msg["warnings"] = []
                self._debug_msg["warnings"].append(
                    f"Type for name {name!r} already exists and it is different, "
                    f"known is {self._known_types[name]} != {int_type} (new) - "
                )
                if self.verbose:
                    print(
                        f"Type for name {name!r} already exists and it is different, "
                        f"known is {self._known_types[name]} != {int_type} (new)"
                    )
                return False

        if self.verbose > 5:
            print(f"[ShapeBuilder-{self._hash()}.set_type] {name}:{int_type}")
        self._known_types[name] = int_type
        return True

    def has_rank(self, name: str) -> bool:
        """Tells if a result has a rank."""
        assert isinstance(name, str), f"Unexpected type {type(name)} for name."
        return name in self._known_ranks

    def get_rank(self, name: str) -> int:
        """Returns the rank of a result."""
        assert isinstance(name, str), f"Unexpected type {type(name)} for name."
        assert name in self._known_ranks, (
            f"rank is unknown for result {name!r}, has_shape={self.has_shape(name)}, "
            f"has_rank={self.has_rank(name)}, "
            f"known_ranks={self._known_ranks}{self.get_debug_msg()}"
        )
        return self._known_ranks[name]

    def set_rank(self, name: str, value: int) -> bool:
        """
        Sets the rank for a result.

        :param name: result name
        :param value: rank
        :return: True if there is no rank conflict
        """
        assert (
            not self._debug_stop_shape or name != self._debug_stop_shape
        ), f"Requested stop, name={name!r}, rank={value}"
        assert isinstance(value, int), f"Unexpected rank type {type(value)} for {name!r}"
        assert not isinstance(value, bool), f"Unexpected rank type {type(value)} for {name!r}"
        assert isinstance(name, str), f"Unexpected type {type(name)} for name."
        if name in self._known_ranks:
            assert value == self._known_ranks[name], (
                f"Inconsistent ranks for {name!r}, previous value is "
                f"{self._known_ranks[name]}, new value is {value}{self.get_debug_msg()}"
            )
            if self.verbose > 5:
                print(f"[ShapeBuilder-{self._hash()}.set_rank] (again) {name}:{value}")
            return True
        self._known_ranks[name] = value
        if self.verbose > 5:
            print(f"[ShapeBuilder-{self._hash()}.set_rank] {name}:{value}")
        return True

    def has_shape(self, name: str, full=False) -> bool:
        """
        Tells if a result has a shape.
        If *full* is True, it returns True if the shape exists and if it
        is a static shape with all dimensions > 0.
        """
        assert isinstance(name, str), f"Unexpected type {type(name)} for name."
        if name not in self._known_shapes:
            return False
        if full:
            shape = self._known_shapes[name]
            return is_static_shape(shape) and min(shape) >= 0
        return True

    def get_shape(self, name: str) -> DYNAMIC_SHAPE:
        """Returns the shape of a result."""
        assert isinstance(name, str), f"Unexpected type {type(name)} for name."
        assert name in self._known_shapes, (
            f"Shape is unknown for result {name!r}, "
            f"known_shapes={self._known_shapes}{self.get_debug_msg()}"
        )
        return self._known_shapes[name]

    def register_dynamic_objects_from_dim(self, dim: str):
        """Registers all the dynamic objects required in a dimension."""
        assert isinstance(dim, str) and " " not in dim and dim.count("(") == dim.count(")"), (
            f"type(dim)={type(dim)} must be a str and should not contain "
            f"a comma or a space dim={dim!r} and the same number of opened and closed "
            f"brackets{self.get_debug_msg()}"
        )
        for token in parse_expression_tokens(dim):
            if token not in self.dynamic_dimensions_:
                self.add_dynamic_dimension(token)

    def add_dynamic_dimension(self, name: str):
        """Adds a dynamic dimension."""
        assert (
            name not in self.dynamic_dimensions_
        ), f"Dynamic dimension {name!r}{self.get_debug_msg()}"
        self.dynamic_dimensions_[name] = {name}

    def set_shape(self, name: str, shape: DYNAMIC_SHAPE, exc: bool = False, **_kwargs):
        """
        Sets the shape for a result. It is exists, it checks the new shape
        is equal to the existing one.

        :param name: result name
        :param shape: shape
        :param exc: raise an exception if inconsistency
        """
        assert isinstance(name, str), f"Unexpected type {type(name)} for name."
        assert isinstance(shape, tuple), f"Unexpected shape type {type(shape)}"
        assert (
            not name or name != self._debug_stop_shape
        ), f"Requested stop, name={name!r}, shape={shape}{self.get_debug_msg()}"
        assert not shape or not isinstance(shape[0], tuple), f"Unexpected shape {shape}"
        for sdim in shape:
            if not isinstance(sdim, str):
                continue
            self.register_dynamic_objects_from_dim(sdim)
        if name in self._known_shapes:
            old_shape = self._known_shapes[name]
            if self._debug_dyn_dim and self._debug_dyn_dim & (set(shape) | set(old_shape)):
                print(
                    f"[ShapeBuilder.set_shape] set_shape({name!r}, {shape}), "
                    f"old_shape={old_shape}"
                )
            if shape != old_shape:
                if exc:
                    raise RuntimeError(
                        f"Name {name!r} already exists and its shape different "
                        f"{old_shape} (old) != {shape}{self.get_debug_msg()}"
                    )
                return False
            return True

        if self._debug_dyn_dim and set(shape) & self._debug_dyn_dim:
            print(f"[ShapeBuilder.set_shape] set_shape({name!r}, {shape})")
        if self.verbose > 5:
            print(f"[ShapeBuilder-{self._hash()}.set_shape] {name}:{shape}")
        self._known_shapes[name] = shape
        if not self.has_rank(name):
            self.set_rank(name, len(shape))

    def value_as_shape(self, name: str) -> bool:
        """Returns the value of a result if it is a shape."""
        if name in self._known_value_shape:
            return self._known_value_shape[name]
        if not self.has_type(name) or self.get_type(name) != onnx.TensorProto.INT64:
            return None
        if self.is_constant(name):
            # It is probably a shape because the user requested it as a shape.
            cst = self.get_constant(name, exc=False, computed_value=True)
            if cst is not None and len(cst.shape) == 1 and cst.dtype == np.int64:
                value = tuple(map(int, cst))
                self._known_value_shape[name] = value
                return value
        return None

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
        else:
            rows.append("--NOCONSTRAINTS--")
        rows.append("--SHAPE--")
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
        if self._calls:
            rows.append("--CALLS--")
            rows.extend([str(s) for s in self._calls])
        else:
            rows.append("--NOCALLS--")
        return "\n".join(rows)

    def run_node(self, node: onnx.NodeProto, exc: bool = False):
        """
        Uses shapes availables in the ShapeBuilder to infer the output shapes
        and types.
        """
        if node.op_type == "Constant" and node.domain == "":
            self.set_constant(node.output[0], node)
            self.simple_update_value_shape_with_node(node)
            if self.verbose:
                print(
                    f"[BasicShapeBuilder.run_node] {self.pretty_node(node)} - "
                    f"{self.get_type(node.output[0])}:{self.get_shape(node.output[0])}"
                )
        else:
            r = self._make_node_set_type_shape(node, exc=exc)
            self.simple_update_value_shape_with_node(node)
            if all(self.is_constant(i) for i in node.input):
                for o in node.output:
                    self.set_constant(o, node)
            if self.verbose:
                print(f"[BasicShapeBuilder.run_node] {self.pretty_node(node)}: {r}")

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
        exc: bool = False,
    ):
        """Runs inference over a model or a graph."""
        if isinstance(model, onnx.ModelProto):
            return self.run_model(
                model.graph, functions={(f.domain, f.name): f for f in model.functions}
            )
        assert isinstance(model, onnx.GraphProto), f"Unexpected type {type(model)} for model"
        graph = model
        for i in graph.initializer:
            self.set_constant(i.name, i)
        for i in graph.sparse_initializer:
            self.set_constant(i.name, i)
        for i in graph.input:
            self.run_value_info(i, True)
        for node in graph.node:
            self.run_node(node, exc=exc)
        for i in graph.output:
            self.run_value_info(i, False)

    def register_constraint_dimension(self, dim_name: str, value: Any):
        """
        Registers a constraint on a dimension.

        :param dim_name: dimension name
        :param value: value to register
        """
        if self._debug_dyn_dim and dim_name in self._debug_dyn_dim:
            print(
                f"[GraphBuilder.register_constraint_dimension] "
                f"dim_name={dim_name!r}, value={value!r}"
            )
        if dim_name not in self.constraints_:
            self.constraints_[dim_name] = set()
        if isinstance(value, set):
            self.constraints_[dim_name] |= value
        else:
            self.constraints_[dim_name].add(value)
