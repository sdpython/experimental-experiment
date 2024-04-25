import pprint
import time
import sys
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np
import onnx.helper as oh
import onnx.numpy_helper as onh
from onnx.shape_inference import infer_shapes as onnx_infer_shapes
from onnx import (
    AttributeProto,
    FunctionProto,
    GraphProto,
    ModelProto,
    NodeProto,
    TensorProto,
    TypeProto,
)
from experimental_experiment.reference import ExtendedReferenceEvaluator
from .shape_helper import (
    DYNAMIC_SHAPE,
    STATIC_SHAPE,
    all_int,
    all_int_or_str,
    is_static_dimension,
    is_static_shape,
)
from .shape_type_compute import set_shape_type_op_any, set_shape_type_custom
from ._onnx_helper import (
    choose_consistent_domain_opset,
    compatible_opsets,
    _default_OPSET_TO_IR_VERSION,
    _nice_shape,
    element_wise_binary_op_types,
    element_wise_op_cmp_types,
    unary_like_op_types,
)
from ._dtype_helper import dtype_to_tensor_dtype, onnx_dtype_to_torch_dtype
from ._helper import make_hash
from .optimization_options import OptimizationOptions
from .expression_dimension import Expression, parse_expression
from .graph_builder_opset import Opset


class GraphBuilder:
    """
    Simplifies the creation of a model.
    Important attributes:

    - `input_names: List[str]`: list of input names
    - `as_function: bool`: the model must be exported as a function or as a model
    - `optimization_options: OptimizationOptions`:
    - `nodes: List[NodeProto]`: list of nodes
    - `initializers_dict: Dict[str, Any]`: initializers
    - `inputs: List[ValueInfoTensorProto]`: inputs
    - `outputs: List[ValueInfoTensorProto]`: outputs
    - `ir_version: int`: ir version
    - `opsets: Dict[str, int]`: declared opsets
    - `input_args: List[T]`: input tensors when the class is used to convert an existing model
    - `functions: List[FunctionProto]`: list of functions to add to the model
    - `value_info: List[ValueInfoProto]`: value info of the original model
    - `dynamic_shapes: Union[Dict[str, Any], Tuple[Any]]]`: dynamic_shapes informations

    Computed attributes:

    - `_unique_names`: used to create unused result names
    - `_unique_node_names`: used to create unused node names
    - `_known_names`: set of existing results names
    - `_known_shapes: Dict[str, DYNAMIC_SHAPE]`: declared shapes
    - `_known_types: Dict[str, int]`: declared element types
    - `_known_value_shape: Dict[str, Any]`: if a result is a shape or not
      (for example the output of operator Shape)
    - `_known_ranks: Dict[str, int]`: declared ranks
    - `_known_sequences: Dict[str, Dict[str, Any]]`: known sequences
    - `constants_node_: Dict[bytes, NodeProto]`: constant node
    - `constants_alias_: Dict[str, str]`: alias for constant
    - `constants_: Dict[str, Any]`: constant values
    - `constants_computed_: Dict[str, Any]`: computed constant values
    - `dynamic_objects: Dict[str, torch.SymInt]`: list of dynamic dimension
    - `dynamic_objects_rev: Dict[str, str]`: reverse dictionary to fasten lookups
    - `_cache_shape: Dict[key,str]`: cache concatenation of shapes
    - `_values: Dict[key,str]`: cache initializer value to merge those which are equal
    - `_dynamic_alias: Dict[str,str]`: used when the user gives a different
        name to the dynamic shapes

    Debugging attributes:

    - `_raise_list: Set[str]`: the builder stop if a result falls in that list
      (debugging tool)
    """

    _op_type_element_wise_types = element_wise_binary_op_types()
    _op_type_element_wise_cmp_types = element_wise_op_cmp_types()
    _op_type_unary_like = unary_like_op_types()

    def __init__(
        self,
        target_opset_or_existing_proto: Union[
            int, Dict[str, int], ModelProto, FunctionProto
        ],
        input_names: Optional[Sequence[str]] = None,
        as_function: bool = False,
        optimization_options: Optional[OptimizationOptions] = None,
        args: Optional[List[Any]] = None,
        ir_version: Optional[int] = None,
        verbose: int = 0,
        infer_shapes: bool = False,
        raise_list: Optional[Set[str]] = None,
        dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
    ):
        import torch

        self.torch = torch
        self.optimization_options = optimization_options or OptimizationOptions(
            verbose=verbose
        )
        self.as_function = as_function
        self.input_args = args
        self.verbose = verbose
        self.ir_version = ir_version
        self._debug_msg = {}
        self.dynamic_shapes = dynamic_shapes
        self.dynamic_objects = {}
        self.dynamic_objects_rev = {}
        self.functions = []
        self.value_info = []
        self.raise_list = raise_list
        self._raise_list = raise_list or set()
        self.constants_computed_ = {}
        self._cache_shape = {}
        self._values = {}
        self._dynamic_alias = {}
        self.constants_node_ = {}
        self.constants_alias_ = {}

        self.nodes = []
        self.initializers_dict = {}
        self.inputs = []
        self.outputs = []

        self._known_shapes = {}
        self._known_types = {}
        self._known_ranks = {}
        self._known_torch_value = {}
        self._known_names = set()
        self._known_sequences = {}
        self._unique_names = set()
        self._unique_node_names = set()
        self._known_value_shape = {}
        self.constants_ = {}

        if self.dynamic_shapes:
            for _, v in self.dynamic_shapes.items():
                for __, vv in v.items():
                    if "_Dim" in str(type(vv)):
                        name = vv.__name__
                    else:
                        name = vv
                    assert isinstance(name, str), (
                        f"Unexpected type {type(v)}:{v} for dynamic "
                        f"dimension in {_!r}, name is {name!r}"
                    )
                    if not self.has_dynamic_object(name):
                        self.make_dynamic_object(name, self.torch.SymInt(name))

        if isinstance(target_opset_or_existing_proto, (int, dict)):
            # starts a model from nothing
            assert (
                not infer_shapes
            ), "infer_shapes is used if an existing model is loaded"
            self.opsets = (
                {"": target_opset_or_existing_proto}
                if isinstance(target_opset_or_existing_proto, int)
                else target_opset_or_existing_proto
            )
            self.input_names = input_names or []
            self.current_input = 0
            self._unique_names = set(self.input_names)
            self._known_names = self._unique_names.copy()

        elif isinstance(target_opset_or_existing_proto, ModelProto):
            # loads a model from nothing
            if input_names:
                raise ValueError(
                    "input_names must be empty if the input is an existing model."
                )
            self.current_input = len(self.inputs)
            self._update_structures_with_proto(
                target_opset_or_existing_proto, infer_shapes
            )
            self.constant_folding(convert_into_initializer=False)
            self._update_shape_types_with_proto(
                target_opset_or_existing_proto, infer_shapes
            )
        else:
            raise NotImplementedError(
                f"{type(target_opset_or_existing_proto)} is not supported."
            )

        self.op = Opset(self)
        self.anyop = Opset(self, allow_unknown=True)

    @property
    def output_names(self) -> List[str]:
        return [o.name for o in self.outputs]

    def empty_copy(
        self, as_function: bool = False, constant_size: int = 2**24
    ) -> "GraphBuilder":
        """
        Creates an empty copy but with the same opsets.
        """
        opt = OptimizationOptions(
            constant_size=constant_size,
            constant_fusing=False,
            remove_identity=False,
            patterns=None,
        )
        g = GraphBuilder(
            self.opsets.copy(),
            verbose=self.verbose,
            ir_version=self.ir_version,
            as_function=as_function,
            optimization_options=opt,
        )
        return g

    def make_key(self, value: Any) -> Optional[Tuple[Union[str, int], ...]]:
        """
        Builds a key identifying a value.
        Returns None if it is none possible.
        """
        if isinstance(value, TensorProto):
            return self.make_key(onh.to_array(value))
        if isinstance(value, self.torch.Tensor):
            if value.dtype == self.torch.int64:
                return self.make_key(value.detach().cpu().numpy())
            return None
        if isinstance(value, int):
            return int, value
        if isinstance(value, np.ndarray):
            if value.dtype == np.int64 and value.size < 8:
                return tuple([value.dtype, value.shape, tuple(value.ravel().tolist())])
        return None

    @classmethod
    def print_node(cls, node: NodeProto):
        return f"{node.op_type}: {node.input} -> {node.output}"

    @property
    def main_opset(self):
        "Returns the opset for the main domain (assuming it is used)."
        return self.opsets[""]

    def get_opset(self, domain: str) -> int:
        """
        Returns the opset version for a specific domain.

        :param domain: domain name
        :return: version
        """
        assert (
            domain in self.opset
        ), f"Domain {domain!r} is not registered{self.get_debug_msg()}."
        return self.opset[domain]

    def add_domain(self, domain: str, version: int = 1):
        """
        Adds a domain to the list of supported ones.
        Checks the version is the same if it exists.
        """
        if domain in self.opsets:
            assert version == self.opsets[domain], (
                f"Version mismatch for domain={domain!r}, current is "
                f"{self.opsets[domain]}, new is {version}{self.get_debug_msg()}"
            )
            return
        self.opsets[domain] = version

    def _hash(self) -> str:
        return make_hash(self)

    def _apply_slice_to_shape(
        self,
        shape: STATIC_SHAPE,
        indices: List[slice],
        axes: List[int],
        expand_axes: List[int],
    ) -> STATIC_SHAPE:
        assert isinstance(
            shape, tuple
        ), f"Unexpected type {type(shape)} for shape: {shape}"
        assert isinstance(
            indices, list
        ), f"Unexpected type {type(indices)} for index: {indices}"
        assert isinstance(axes, list), f"Unexpected type {type(axes)} for index: {axes}"
        assert len(axes) in (
            1,
            len(indices),
        ), f"Mismatch lengths {len(indices)} != {len(axes)}"

        if all(map(lambda i: isinstance(i, slice), indices)):
            new_shape = []
            for index, axis in zip(indices, axes):
                while len(new_shape) < axis:
                    assert shape[len(new_shape)] >= 0, (
                        f"Negative value in shape {shape}, indices={indices}, "
                        f"axes={axes}, expand_axes={expand_axes}"
                    )
                    new_shape.append(shape[len(new_shape)])
                assert axis < len(shape), (
                    f"axis={axis} is out of order (shape={shape}, "
                    f"indices={indices}, axes={axes}){self.get_debug_msg()}"
                )
                n = shape[axis]
                start = index.start or 0
                end = index.stop or n
                diff = end - start
                dim = diff // index.step if index.step else diff
                dim = max(dim, 0)
                assert dim >= 0, (
                    f"Negative dim={dim}, axis={axis}, shape={shape}, indices={indices}, "
                    f"axes={axes}, expand_axes={expand_axes}"
                )
                new_shape.append(dim)
        elif all_int(indices):
            assert len(axes) == 1, (
                f"Unable to guess new shape from shape={shape}, "
                f"indices={indices}, axes={axes}, expand_axes={expand_axes}"
            )
            new_shape = [len(indices), *shape[1:]]
        else:
            raise RuntimeError(
                f"Unable to guess new shape from shape={shape}, "
                f"indices={indices}, axes={axes}, expand_axes={expand_axes}"
            )
        for a in shape[len(new_shape) :]:
            assert a >= 0, (
                f"Negative value in shape {shape}, indices={indices}, "
                f"axes={axes}, expand_axes={expand_axes}"
            )
            new_shape.append(a)
        for e in expand_axes:
            new_shape.insert(e, 1)
        return tuple(new_shape)

    def _apply_reshape_to_shape(
        self, input_shape: DYNAMIC_SHAPE, new_shape: STATIC_SHAPE
    ) -> DYNAMIC_SHAPE:
        """
        Returns the shape of the output of a node Reshape.
        """
        assert isinstance(
            input_shape, tuple
        ), f"unexpected type {type(input_shape)} for input_shape."
        assert isinstance(
            new_shape, tuple
        ), f"unexpected type {type(new_shape)} for input_shape."
        assert all_int(new_shape), f"unexpected type for a dimension in {new_shape}"
        if -1 not in new_shape:
            return new_shape
        if all_int(input_shape):
            size = int(np.prod(input_shape))
            div = np.prod([i for i in new_shape if i != -1])
            if div == 0:
                return tuple((int(i) if i >= 0 else 0) for i in new_shape)
            return tuple((int(i) if i >= 0 else int(size // div)) for i in new_shape)
        if all_int_or_str(input_shape):
            if new_shape == (1, -1):
                # common case
                return (1, "*".join(map(str, input_shape)))

        if len(input_shape) == len(new_shape):
            # It is easier to handle.
            res = []
            i_1 = None
            a_int = True
            b_int = True
            for a, b in zip(input_shape, new_shape):
                if not isinstance(a, int):
                    a_int = False
                if isinstance(b, int):
                    if b >= 0:
                        res.append(b)
                    else:
                        i_1 = len(res)
                        res.append(None)
                else:
                    res.append(b)
                    b_int = False
            if i_1 is not None:
                if a_int:
                    size = int(np.prod(input_shape))
                    if b_int:
                        nz = -int(np.prod(new_shape)) // size
                        res[i_1] = nz
                    else:
                        name = "*".join([str(x) for x in res if x is not None])
                        res[i_1] = f"{name}/{size}"
                else:
                    an = "*".join(map(str, input_shape))
                    name = "*".join([str(x) for x in res if x is not None])
                    res[i_1] = f"{an}/({name})"
            return tuple(res)

        raise RuntimeError(
            f"Not implemented yet for input_shape={input_shape} and new_shape={new_shape}."
        )

    def _get_tensor_shape(self, proto: Union[NodeProto, TensorProto]) -> STATIC_SHAPE:
        if isinstance(proto, TensorProto):
            return tuple(proto.dims)
        if isinstance(proto, NodeProto):
            for att in proto.attribute:
                if att.name in ("value_float", "value_int"):
                    return tuple()
                if att.name == "value_floats":
                    return (len(att.floats),)
                if att.name == "value_ints":
                    return (len(att.ints),)
                if att.name == "value":
                    t = onh.to_array(att.t)
                    return tuple(t.shape)
        raise TypeError(
            f"Unexpected or unsupported scenario type {type(proto)}: {proto}."
        )

    def _get_tensor_type(self, proto: Union[NodeProto, TensorProto]) -> int:
        if isinstance(proto, TensorProto):
            return proto.data_type
        if isinstance(proto, NodeProto):
            for att in proto.attribute:
                if att.name == "value_float":
                    return TensorProto.FLOAT
                if att.name == "value_int":
                    return TensorProto.INT64
                if att.name == "value_floats":
                    return TensorProto.FLOAT
                if att.name == "value_ints":
                    return TensorProto.INT64
                if att.name == "value":
                    return att.t.data_type
        raise ValueError(f"Unexpected type or value {type(proto)}: {proto}.")

    def is_constant(self, name: str) -> bool:
        """Tells if a result is a constant."""
        return name in self.constants_

    def get_constant(
        self,
        name: str,
        exc: bool = True,
        computed_value: bool = False,
        as_shape: bool = False,
    ) -> Union[np.ndarray, NodeProto]:
        """
        The method returns the constant *name*. It is a tensor (numpy array)
        or a NodeProto which must be evaluated.
        If *computed_value* is True, the NodeProto is evaluated wuth the
        ReferenceEvaluator.

        :param name: constant name
        :param exc: raise an exception if anything is impossible to do
        :param computed_value: compute the value if not a constant
        :param as_shape: returns a tuple for a shape
        :return: value
        """
        if as_shape:
            res = self.get_constant(
                name, exc, computed_value=computed_value, as_shape=False
            )
            new_res = []
            for i in res:
                if isinstance(i, str):
                    new_res.append(i)
                else:
                    new_res.append(int(i))
            return tuple(new_res)

        if not self.is_constant(name):
            raise ValueError(f"Result {name!r} is not a constant{self.get_debug_msg()}")
        possible_value = self.constants_[name]
        if name in self.constants_computed_:
            return self.constants_computed_[name]

        if possible_value is not None:
            assert isinstance(
                possible_value, (np.ndarray, self.torch.Tensor, NodeProto)
            ), (
                f"Unexpected type {type(possible_value)} for a "
                f"constant{self.get_debug_msg()}"
            )
            if computed_value and isinstance(possible_value, NodeProto):
                res = self.compute_constant(name, exc=exc)[0]
                if len(res) == 1:
                    return res[0]
                index = list(possible_value.output).index(name)
                return res[index]
            return possible_value

        if name not in self.initializers_dict:
            if exc:
                raise ValueError(
                    f"Result {name!r} was never evaluated within method 'constant_folding'."
                )
            return None

        value = self.initializers_dict[name]

        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, self.torch.Tensor):
            v = value.detach().cpu().numpy()
            self.constants_computed_[name] = v
            return v
        if isinstance(value, TensorProto):
            v = onh.to_array(value)
            self.constants_computed_[name] = v
            return v
        raise TypeError(f"Unable to convert type {type(value)} into numpy array.")

    def is_sequence(self, name: str) -> bool:
        """Tells if a result is a sequence."""
        if name in self._known_sequences:
            return True
        assert self.has_name(name), f"Unknown name={name!r}{self.get_debug_msg()}"
        return False

    def get_sequence(self, name: str) -> Dict[str, Any]:
        """Returns sequence information"""
        assert (
            name in self._known_sequences
        ), f"Sequence {name!r} is not known{self.get_debug_msg()}"
        return self._known_sequences[name]

    def set_sequence(
        self,
        name: str,
        dtype: int,
        shapes: Optional[DYNAMIC_SHAPE] = None,
        ranks: Optional[Tuple[int, ...]] = None,
        unknown: bool = False,
    ):
        """
        Defines a result as a sequence.
        """
        assert (
            shapes is not None or ranks is not None or unknown
        ), f"shapes or ranks must be defines for name={name!r}{self.get_debug_msg()}"
        assert self.has_name(name), f"No result name={name!r}{self.get_debug_msg()}"
        assert isinstance(dtype, int), (
            f"Only one type is allowed in sequences but dtype={dtype!r}"
            f"{self.get_debug_msg()}"
        )
        d = dict(dtype=dtype, shapes=shapes, ranks=ranks)
        if shapes is not None and ranks is None:
            d["ranks"] = tuple(len(s) for s in shapes)
        if name not in self._known_sequences:
            self._known_sequences[name] = d
        else:
            assert self._known_sequences[name] == d, (
                f"Sequence {name!r} was already declared with a different type "
                f"or shape or rank, declared={self._known_sequences[name]}, "
                f"new={d}{self.get_debug_msg()}"
            )

    def set_name(self, name: str):
        """Adds a name to the list of known names."""
        assert (
            name != ""
        ), f"Empty name {name!r} cannot be registered{self.get_debug_msg()}"
        assert (
            name not in self._raise_list
        ), f"Name {name!r} is one of the name declared in the stop list{self.get_debug_msg()}"

        assert isinstance(name, str), f"Unexpected type {type(name)} for name."
        assert (
            name not in self._known_names
        ), f"Name {name!r} already exists{self.get_debug_msg()}"
        self._known_names.add(name)
        self._unique_names.add(name)

    def set_rank(self, name: str, value: int):
        """
        Sets the rank for a result.

        :param name: result name
        :param value: rank
        """
        assert isinstance(
            value, int
        ), f"Unexpected rank type {type(value)} for {name!r}"
        assert not isinstance(
            value, bool
        ), f"Unexpected rank type {type(value)} for {name!r}"
        assert isinstance(name, str), f"Unexpected type {type(name)} for name."
        if name in self._known_ranks:
            assert value == self._known_ranks[name], (
                f"Inconsistent ranks for {name!r}, previous value is "
                f"{self._known_ranks[name]}, new value is {value}{self.get_debug_msg()}"
            )
            if self.verbose > 5:
                print(f"[GraphBuilder-{self._hash()}.set_rank] (again) {name}:{value}")
            return
        assert (
            name not in self._known_ranks
        ), f"Name {name!r} already exists{self.get_debug_msg()}"
        self._known_ranks[name] = value
        if self.verbose > 5:
            print(f"[GraphBuilder-{self._hash()}.set_rank] {name}:{value}")

    def is_more_precise(self, shape: STATIC_SHAPE, base: STATIC_SHAPE) -> bool:
        assert len(shape) == len(
            base
        ), f"Cannot compare shapes with different ranks {shape} and {base}"
        for a, b in zip(shape, base):
            if isinstance(a, int) and isinstance(b, int):
                if a != b:
                    return False
            if isinstance(a, str) and isinstance(b, str):
                if a != b:
                    return False
        return True

    def get_is_dimension(
        self,
        name: str,
        elem_type: Optional[int] = None,
        shape: Optional[STATIC_SHAPE] = None,
    ) -> bool:
        """
        Tells if a result is a dynamic dimension or not.
        """
        if name in self.dynamic_objects:
            res = True
        elif name in self._known_torch_value:
            value = self._known_torch_value[name]
            if value[0] == "run_node":
                val1 = value[1]
                exa, val = val1
                if val is not None and len(val) == 3:
                    el_type, size = val[1:]
                    if el_type in (
                        self.torch.float32,
                        self.torch.float64,
                        self.torch.float16,
                    ):
                        return False
                    if len(size) >= 2:
                        return False
                    if el_type == self.torch.int64 and len(size) == 0:
                        # A single integer with no shape, it looks like a dimension.
                        # Let's assume it is. It is more efficient to consider it as
                        # a dimension.
                        return False
                else:
                    if elem_type is not None and elem_type in (
                        self.torch.float32,
                        self.torch.float64,
                        self.torch.float16,
                    ):
                        return False
                    if shape is not None and len(shape) >= 2:
                        return False
                    dtype = self.get_type(name)
                    if dtype in {
                        TensorProto.FLOAT16,
                        TensorProto.FLOAT,
                        TensorProto.DOUBLE,
                        TensorProto.BFLOAT16,
                    }:
                        return False
                    if self.has_shape(name):
                        shape = self.get_shape(name)
                        if dtype == TensorProto.INT64 and shape == (1,):
                            return True
                    elif self.has_rank(name):
                        if self.get_rank(name) > 1:
                            return False
                if isinstance(val1[0], tuple) and len(val1[0]) >= 1:
                    v = val1[0]
                    if (
                        isinstance(v, tuple)
                        and len(v) == 3
                        and v[0] == "example_value"
                        and len(self.dynamic_objects) == 0
                    ):
                        # No dynamic shape as input, so there shoud not be any dynamic shape as output.
                        return False
            elif value[0] == "call_module":
                if isinstance(value[1], tuple) and len(value[1]) == 2:
                    el_type, size = value[1]
                    if el_type in (
                        self.torch.float32,
                        self.torch.float64,
                        self.torch.float16,
                    ):
                        return False
                    if len(size) >= 2:
                        return False
            raise RuntimeError(
                f"Not implemented for name={name!r}, value={value!r} ({type(value)}), "
                f"elem_type={elem_type}, shape={shape}{self.get_debug_msg()}"
            )
        else:
            if elem_type in {
                TensorProto.FLOAT16,
                TensorProto.FLOAT,
                TensorProto.DOUBLE,
                TensorProto.BFLOAT16,
            }:
                return False
            raise RuntimeError(
                f"Unable to gues if {name!r}, elem_type={elem_type}, "
                f"shape={shape} is a dimension{self.get_debug_msg()}"
            )
        assert not res or (
            (
                elem_type is None
                or elem_type
                in {
                    TensorProto.INT64,
                    TensorProto.INT32,
                    TensorProto.UINT64,
                    TensorProto.UINT32,
                }
            )
            and (shape is None or (isinstance(shape, tuple) and len(shape) == 1))
        ), (
            f"Inconsistent result type for name={name!r}, is_dimension={res}, "
            f"elem_type={elem_type}, shape={shape}{self.get_debug_msg()}"
        )
        return res

    def set_shapes_types(
        self, name: Union[str, "torch.fx.Node"], where: str, value: Any  # noqa: F821
    ):
        if hasattr(name, "name"):
            name = name.name
        self._known_torch_value[name] = (where, value)

    def set_shape(
        self,
        name: str,
        shape: DYNAMIC_SHAPE,
        set_rank: bool = True,
        set_if_more_precise: bool = False,
        exc: bool = False,
    ):
        """
        Sets the shape for a result. It is exists, it checks the new shape
        is equal to the existing one.

        :param name: result name
        :param shape: shape
        :param set_rank: set the rank as well
        :param set_if_more_precise: change the shape if it is more precise
        :param exc: raise an exception if inconsistency
        """
        assert isinstance(name, str), f"Unexpected type {type(name)} for name."
        assert "torch.Size" not in str(shape), (
            f"Unexpected type {type(shape)} for a "
            f"shape={shape}{self.get_debug_msg()}"
        )
        assert isinstance(shape, tuple), f"Unexpected shape type {type(shape)}"
        shape = self.verify_shape(shape, 0, name=name)
        assert all(map(lambda t: not isinstance(t, self.torch.SymInt), shape)), (
            f"Unexpected type for a shape, shape={shape}, types={[type(_) for _ in shape]}"
            f"{self.get_debug_msg()}"
        )
        assert isinstance(shape, tuple), f"Unexpected shape type {type(shape)}"
        shape_int = [d for d in shape if isinstance(d, int)]
        assert (
            len(shape) == 0 or not shape_int or min(shape_int) >= 0
        ), f"Negative value in shape {shape} for {name!r}{self.get_debug_msg()}"
        if name in self._known_shapes:
            old_shape = self._known_shapes[name]
            if len(shape) == len(old_shape) and set_if_more_precise:
                if not self.is_more_precise(shape=shape, base=old_shape):
                    raise RuntimeError(
                        f"Name {name!r} already exists and it is not compatible "
                        f"{old_shape} != {shape}{self.get_debug_msg()}"
                    )
            elif shape != old_shape:
                if exc:
                    raise RuntimeError(
                        f"Name {name!r} already exists and its shape different "
                        f"{old_shape} (old) != {shape}{self.get_debug_msg()}"
                    )
                return
            else:
                return
        if self.verbose > 5:
            print(f"[GraphBuilder-{self._hash()}.set_shape] {name}:{shape}")
        self._known_shapes[name] = shape
        if set_rank and not self.has_rank(name):
            self.set_rank(name, len(shape))

    def set_type(self, name: str, dtype: int):
        """
        Sets the shape for a result. It is exists, it checks the new shape
        is equal to the existing one.

        :param name: name
        :param dtype: element type (an integer, ONNX)
        """
        assert isinstance(name, str), f"Unexpected type {type(name)} for name."
        if isinstance(dtype, int):
            int_type = dtype
        else:
            int_type = self._get_type(dtype)
        if name in self._known_types:
            # 0 is undefined
            if self._known_types[name] != 0 and int_type != self._known_types[name]:
                raise RuntimeError(
                    f"Type for name {name!r} already exists and it is different, "
                    f"known is {self._known_types[name]} != {int_type} (new)"
                    f"{self.get_debug_msg()}"
                )
        if self.verbose > 5:
            print(f"[GraphBuilder-{self._hash()}.set_type] {name}:{int_type}")
        self._known_types[name] = int_type

    def rank(self, name: str) -> int:
        """Shortcut to :meth:`get_rank`."""
        assert isinstance(name, str), f"Unexpected type {type(name)} for name."
        return self.get_rank(name)

    def has_name(self, name: str) -> bool:
        """Tells if a result exists."""
        assert isinstance(
            name, str
        ), f"Unexpected type {type(name)} for name (name={name!r})."
        return name in self._known_names

    def has_rank(self, name: str) -> bool:
        """Tells if a result has a rank."""
        assert isinstance(name, str), f"Unexpected type {type(name)} for name."
        return name in self._known_ranks

    def has_shape(self, name: str, full=False) -> bool:
        """Tells if a result has a shape."""
        assert isinstance(name, str), f"Unexpected type {type(name)} for name."
        if name not in self._known_shapes:
            return False
        if full:
            shape = self._known_shapes[name]
            return is_static_shape(shape) and min(shape) >= 0
        return True

    def has_type(self, name: str) -> bool:
        """Tells if a result has a type. This should be always true."""
        assert isinstance(name, str), f"Unexpected type {type(name)} for name."
        return name in self._known_types

    def get_rank(self, name: str) -> int:
        """Returns the rank of a result."""
        assert isinstance(name, str), f"Unexpected type {type(name)} for name."
        assert name in self._known_ranks, (
            f"rank is unknown for result {name!r}, "
            f"known_shapes={self._known_ranks}{self.get_debug_msg()}"
        )
        return self._known_ranks[name]

    def get_shape(self, name: str) -> int:
        """Returns the shape of a result."""
        assert isinstance(name, str), f"Unexpected type {type(name)} for name."
        assert name in self._known_shapes, (
            f"Shape is unknown for result {name!r}, "
            f"known_shapes={self._known_shapes}{self.get_debug_msg()}"
        )
        return self._known_shapes[name]

    def get_type(self, name: str) -> int:
        """Returns the type of a result."""
        assert isinstance(name, str), f"Unexpected type {type(name)} for name."
        assert name in self._known_types, (
            f"Type is unknown for result {name!r}, "
            f"known_types={self._known_types}{self.get_debug_msg()}."
        )
        return self._known_types[name]

    def value_as_shape(self, name: str) -> bool:
        """Returns the value of a result if it is a shape."""
        if name in self._known_value_shape:
            return self._known_value_shape[name]
        return None

    def set_value_shape(self, name: str, value: Any):
        """
        Sets the value for a shape result.

        :param name: name
        :param value: it cannot be empty
        """
        assert (
            name not in self._known_value_shape
        ), f"Shape value for {name!r} (value={value!r}) is already registered."
        assert value not in {
            tuple()
        }, f"Unexpected value for shape {name!r}, value={value}{self.get_debug_msg()}"
        if self.verbose > 2:
            print(f"[GraphBuilder-{self._hash()}.set_value_shape] {name}[{value}]")
        self._known_value_shape[name] = value

    def unique_name(self, prefix: str) -> str:
        if prefix in self._unique_names:
            i = 2
            sug = f"{prefix}2"
            while sug in self._unique_names:
                i += 1
                sug = f"{prefix}{i}"
            self._unique_names.add(sug)
            return sug
        self._unique_names.add(prefix)
        return prefix

    def unique_node_name(self, name: str) -> str:
        if name in self._unique_node_names:
            i = 2
            sug = f"{name}2"
            while sug in self._unique_node_names:
                i += 1
                sug = f"{name}{i}"
            self._unique_node_names.add(sug)
            return sug
        self._unique_node_names.add(name)
        return name

    def _get_type(self, elem_type: Any, exc: bool = True) -> int:
        if not isinstance(elem_type, int):
            st = str(elem_type)
            if "float32" in st:
                elem_type = TensorProto.FLOAT
            elif "float64" in st:
                elem_type = TensorProto.DOUBLE
            elif "bfloat16" in st:
                elem_type = TensorProto.BFLOAT16
            elif "float16" in st:
                elem_type = TensorProto.FLOAT16
            elif "uint64" in st:
                elem_type = TensorProto.UINT64
            elif "int64" in st:
                elem_type = TensorProto.INT64
            elif "uint32" in st:
                elem_type = TensorProto.UINT32
            elif "int32" in st:
                elem_type = TensorProto.INT32
            elif "uint16" in st:
                elem_type = TensorProto.UINT16
            elif "int16" in st:
                elem_type = TensorProto.INT16
            elif "bool" in st:
                elem_type = TensorProto.BOOL
            elif "uint8" in st:
                elem_type = TensorProto.UINT8
            elif "int8" in st:
                elem_type = TensorProto.INT8
            elif elem_type is None:
                elem_type = TensorProto.UNDEFINED
            elif exc:
                raise ValueError(f"Unable to interpret elem_type {elem_type!r}.")
        return elem_type

    def has_dynamic_object(self, name: str) -> bool:
        """Tells if a result is a dynamic object, `torch.SymInt` for torch."""
        return name in self.dynamic_objects

    def make_dynamic_object(
        self, name: str, value: Any, shape_as_input: bool = False
    ) -> str:
        """
        Creates a dynamic shapes.

        :param name: name
        :param value: value
        :param shape_as_input: adds the name to the list of the inputs
            of the onnx model
        :return: the name
        """
        assert name not in self.dynamic_objects, (
            f"Dynamic object {name!r}, value={value!r} "
            f"is already there{self.get_debug_msg()}"
        )
        assert isinstance(
            value, self.torch.SymInt
        ), f"Unexpected type {type(value)} for value{self.get_debug_msg()}"
        self.dynamic_objects[name] = value
        assert (
            name not in self._known_value_shape
        ), f"Shape value for {name!r} was already registered."
        self._known_value_shape[name] = name
        key = str(value)
        if key not in self.dynamic_objects_rev:
            self.dynamic_objects_rev[key] = []
        self.dynamic_objects_rev[str(value)].append((name, value))
        if shape_as_input:
            # torch.compile adds input for dynamic shapes
            return self.make_tensor_input(
                name, TensorProto.INT64, (1,), is_dimension=True
            )
        return name

    def make_shape_from_results(self, shape: DYNAMIC_SHAPE, name="") -> str:
        """
        Creates a shape coming from intermediate results.
        """
        assert isinstance(
            shape, (list, tuple)
        ), f"Unexpected type {type(shape)} for shape{self.get_debug_msg()}"
        if all_int(shape):
            return self.make_initializer("", np.array(shape, dtype=np.int64))

        key = []
        for d in shape:
            if isinstance(d, int):
                key.append(d)
            elif isinstance(d, (str, self.torch.SymInt)):
                value = self._torch_sym_int(d)
                key.append(value)
            else:
                raise RuntimeError(
                    f"Unexpected type {type(d)} for a dimension in {shape}{self.get_debug_msg()}"
                )

        assert all_int_or_str(key), (
            f"Unexpected key {key} type are {[type(_) for _ in key]}, "
            f"shape={shape}{self.get_debug_msg()}"
        )
        key = tuple(["Concat"] + key)
        if key in self._cache_shape:
            # The same shape was already requested.
            return self._cache_shape[key]

        conc = []
        for d in shape:
            if isinstance(d, int):
                conc.append(self.make_initializer("", np.array([d], dtype=np.int64)))
            elif isinstance(d, (str, self.torch.SymInt)):
                value = self._torch_sym_int(d)
                if value in self.dynamic_objects_rev:
                    name = self.dynamic_objects_rev[value][0]
                    assert not isinstance(name, tuple)
                else:
                    name = value
                if isinstance(name, self.torch.SymInt):
                    name = self._torch_sym_int(name)
                    assert not isinstance(name, self.torch.SymInt)
                assert not isinstance(name, self.torch.SymInt)
                assert name in self.dynamic_objects or self.has_name(
                    name
                ), f"Unknown dynamic object {d!r}-{name!r}{self.get_debug_msg()}"
                if self.has_rank(name):
                    assert (
                        self.get_rank(name) <= 1
                    ), f"Unexpected rank={self.get_rank(name)} for a shape{self.get_debug_msg()}"
                    if self.get_rank(name) == 0:
                        r = self.op.UnsqueezeAnyOpset(
                            name, np.array([0], dtype=np.int64), name=f"_mkshape_{name}"
                        )
                        self.set_type(r, self.get_type(name))
                        self.set_shape(r, (1,))
                        conc.append(r)
                    else:
                        conc.append(name)
                else:
                    conc.append(name)
            else:
                raise RuntimeError(
                    f"Unexpected type {type(d)} for a dimension in {shape}{self.get_debug_msg()}"
                )

        res = self.make_node("Concat", conc, axis=0, name=f"_mkshape_{name}")
        self._cache_shape[key] = res
        return res

    def make_initializer(
        self, name: str, value: Any, external: bool = False, msg: str = ""
    ) -> str:
        """
        Adds an initializer to the graph.
        The function detects duplicated small containers, only if they are
        integers. Other type might be used as weights. Even similar, they could
        change after training.

        :param name: name, if empty (`""`), a unique names is given, if not empty,
            it is more like a prefix, the method might change it to make it unique
        :param value: value (TensorProto)
        :param external: external initializer or not (not stored in the graph model)
        :param msg: added to the error message if something goes wrong
        :return: name of the initializer
        """
        if external:
            raise NotImplementedError("External initializers are not implemented yet.")
        if isinstance(value, int):
            value = np.array(value, dtype=np.int64)
        elif isinstance(value, float):
            value = np.array(value, dtype=np.float32)
        elif hasattr(value, "data"):
            # torch.nn.parameter.Parameter -> np.array
            pass
        elif isinstance(value, TensorProto):
            pass
        elif isinstance(value, np.ndarray):
            pass
        else:
            raise RuntimeError(
                f"Initializer name={name!r}, "
                f"unexpected type {type(value)} for value={value!r} ({msg})"
                f"{self.get_debug_msg()}"
            )

        key = self.make_key(value)
        if key and key in self._values:
            if name == "":
                return self._values[key]
            return self.make_node(
                "Identity", [self._values[key]], [name], name="make_initializer"
            )

        if isinstance(value, TensorProto):
            itype = value.data_type
            shape = tuple(value.dims)
        else:
            itype = self._get_type(value.dtype)
            shape = tuple(value.shape)
        if name == "":
            sh = "x".join(map(str, shape))
            size = np.prod(value.size()) if hasattr(value, "detach") else value.size
            sh2 = (
                "_".join(map(str, value.ravel().tolist()))
                if size <= 5 and value.dtype == np.int64
                else ""
            )
            name = self.unique_name(f"init{itype}_s{sh}_{sh2}")
        self.set_shape(name, shape)
        self.set_type(name, itype)
        self.set_name(name)
        self.initializers_dict[name] = value
        self.constants_[name] = None
        if self.verbose and (self.verbose > 1 or np.prod(value.shape) > 100):
            print(
                f"[GraphBuilder-{self._hash()}.make_initializer] {name}[{itype}:{shape}]"
            )
        if key:
            self._values[key] = name
        return name

    def is_dynamic_shape(
        self, shape: DYNAMIC_SHAPE, verify: bool = True, allow_none: bool = False
    ) -> bool:
        return all(
            map(
                lambda x: self.is_dynamic_dimension(
                    x, verify=verify, allow_none=allow_none
                ),
                shape,
            )
        )

    def is_constant_or_attribute(
        self, node: NodeProto, input_index: int, att_name: str
    ) -> bool:
        """
        Tells if an input is a constant or returns true if in an older
        opset, it was named as an attribute.
        """
        if input_index < len(node.input):
            return self.is_constant(node.input[input_index])
        return True

    def get_constant_or_attribute(
        self, node: NodeProto, input_index: int, att_name: str
    ) -> Any:
        """
        Tells if an input is a constant or returns true if in an older
        opset, it was named as an attribute.
        """
        if input_index < len(node.input):
            return self.get_constant(node.input[input_index])
        for att in node.attribute:
            if att.name == att_name:
                assert (
                    att.type == AttributeProto.INTS
                ), f"Not Implemented when att.type={att.type}{self.get_debug_msg()}"
                return np.array(list(att.ints), dtype=np.int64)
        return None

    def simple_update_value_shape_with_node(self, node) -> bool:
        if node.domain != "":
            return False
        if node.op_type not in {
            "Concat",
            "Gather",
            "Shape",
            "Add",
            "Mul",
            "Div",
            "Sub",
            "Mod",
            "Slice",
            "Abs",
            "Range",
            "Scatter",
            "Squeeze",
            "Identity",
            "Unsqueeze",
            "Greater",
            "Less",
            "GreaterOrEqual",
            "LessOrEqual",
            "Equal",
            "Not",
        }:
            return False

        if node.op_type == "Identity":
            value = self.value_as_shape(node.input[0])
            if value is not None:
                self.set_value_shape(node.output[0], value)
                return True
            return False

        if node.op_type == "Squeeze":
            if self.is_constant_or_attribute(node, 1, "axes"):
                y = self.value_as_shape(node.input[0])
                if y is None:
                    return False
                i = self.get_constant_or_attribute(node, 1, "axes")
                if isinstance(i, int):
                    ii = i
                elif (
                    isinstance(i, np.ndarray)
                    and i.dtype == np.int64
                    and i.shape in ((1,), tuple())
                ):
                    ii = int(i[0]) if i.shape == (1,) else int(i)
                else:
                    raise RuntimeError(
                        f"Not implemented when node Squeeze with inputs={node.input}, "
                        f"y={y!r}, i={i!r}{self.get_debug_msg()}"
                    )
                assert (
                    ii == 0
                ), f"A shape should only have one axis i={i}, y={y}{self.get_debug_msg()}"
                if isinstance(y, str):
                    self.set_value_shape(node.output[0], f"squeeze({y})")
                    return True
                if isinstance(y, int):
                    self.set_value_shape(node.output[0], y)
                    return True
                assert isinstance(
                    y, tuple
                ), f"Unexpected type {type(y)} for y={y} and i={i}{self.get_debug_msg()}"
                self.set_value_shape(node.output[0], y[0])
                return True
            return False

        if node.op_type == "Shape":
            if len(node.attribute) == 0:
                if self.has_shape(node.input[0]):
                    self.set_value_shape(node.output[0], self.get_shape(node.input[0]))
                else:
                    self.set_value_shape(node.output[0], node.output[0])
                return True

            start = self.get_attribute(node, "start", exc=False) or 0
            end = self.get_attribute(node, "end", exc=False)
            if end is None:
                if self.has_rank(node.input[0]):
                    end = self.get_rank(node.input[0])
            if self.has_shape(node.input[0]):
                shape = self.get_shape(node.input[0])
                assert start.i < len(shape), (
                    f"Shape mismatch, start={start.i}, shape of {node.input[0]!r} "
                    f"is {shape}{self.get_debug_msg()}"
                )
                if end is None:
                    self.set_value_shape(node.output[0], shape[start.i :])
                    return True
                assert getattr(end, "i", end) <= len(shape), (
                    f"Shape mismatch, end={getattr(end, 'i', end)}, shape of {node.input[0]!r} "
                    f"is {shape}{self.get_debug_msg()}"
                )
                self.set_value_shape(
                    node.output[0], shape[start.i : getattr(end, "i", end)]
                )
                return True

            if end is None:
                self.set_value_shape(node.output[0], f"{node.input[0]}[{start.i}:]")
                return False

            self.set_value_shape(
                node.output[0],
                f"{node.input[0]}[{start.i}:{getattr(end, 'i', end)}]",
            )
            return True

        if node.op_type == "Gather":
            if self.is_constant(node.input[1]):
                y = self.value_as_shape(node.input[0])
                if y is None:
                    return False
                i = self.get_constant(node.input[1], computed_value=True)
                if isinstance(y, str) and isinstance(i, int):
                    self.set_value_shape(node.output[0], f"{y}[{i}]")
                    return True
                if (
                    isinstance(y, str)
                    and isinstance(i, np.ndarray)
                    and i.dtype == np.int64
                    and i.shape in ((1,), tuple())
                ):
                    ii = int(i[0]) if i.shape == (1,) else int(i)
                    self.set_value_shape(node.output[0], f"{y}[{ii}]")
                    return True
                if isinstance(y, tuple) and isinstance(i, int):
                    self.set_value_shape(node.output[0], y[i])
                    return True
                if (
                    isinstance(y, tuple)
                    and isinstance(i, np.ndarray)
                    and i.dtype == np.int64
                    and i.shape in ((1,), tuple())
                ):
                    ii = int(i[0]) if i.shape == (1,) else int(i)
                    assert ii < len(y), (
                        f"Unexpected value for y={y!r}, i={i!r} in node Gather "
                        f"with inputs={node.input}{self.get_debug_msg()}"
                    )
                    self.set_value_shape(node.output[0], y[ii])
                    return True
                raise RuntimeError(
                    f"Not implemented when node Gather with inputs={node.input}, "
                    f"y={y!r}, i={i!r}{self.get_debug_msg()}"
                )
            return False

        values = [self.value_as_shape(x) for x in node.input[0]]
        if any(map(lambda x: x is None, values)):
            # it is not a shape
            return False
        if node.op_type == "Concat":
            self.set_shape_value(node.output[0], tuple(values))
            return True
        raise RuntimeError(
            f"Unable to compute a shape for node {node.op_type!r} "
            f"with inputs={node.input}{self.get_debug_msg()}"
        )

    def is_dynamic_dimension(
        self, dim: Any, verify: bool = True, allow_none: bool = False
    ) -> bool:
        if allow_none and dim is None:
            return True
        if not isinstance(dim, (int, self.torch.SymInt, str)):
            return False
        if str(dim) in self._dynamic_alias:
            dim = self._dynamic_alias[str(dim)]
        assert (
            not verify
            or is_static_dimension(dim)
            or str(dim) in self.dynamic_objects
            or str(dim) in self.dynamic_objects_rev
            or self.has_name(str(dim))
            or self.parse_dimension_expression(dim)
        ), (
            f"dim={dim!r} (type={type(dim)}) not in found in "
            f"{self.dynamic_objects}, self.dynamic_shapes={self.dynamic_shapes}, "
            f"self._dynamic_alias={self._dynamic_alias}{self.get_debug_msg()}"
        )
        return True

    def get_dynamic_dimension(self, dim: Any, keep_const: bool = True) -> Any:
        if isinstance(dim, int):
            if keep_const:
                return np.array([dim], dtype=np.int64)
            return self.make_initializer("", np.array([dim], dtype=np.int64))
        assert isinstance(
            dim, (str, self.torch.SymInt)
        ), f"Unexpected type {type(dim)} for dim={dim}{self.get_debug_msg()}"
        name = str(dim)
        try:
            value = int(name)
            dyn = False
        except ValueError:
            dyn = True
        if not dyn:
            if keep_const:
                return np.array([value], dtype=np.int64)
            return self.make_initializer("", np.array([value], dtype=np.int64))
        assert (
            name in self.dynamic_objects
        ), f"Unable to find dim={dim!r} in {self.dynamic_objects}{self.get_debug_msg()}"
        return name

    def _get_dynamic_dimension(self, name: str, dim: int) -> Optional[str]:
        if self.dynamic_shapes is None:
            return None
        if name not in self.dynamic_shapes:
            return None
        dyn = self.dynamic_shapes[name]
        if dim not in dyn:
            return None
        v = dyn[dim]
        if "_Dim" in str(type(v)):
            name = v.__name__
        else:
            name = v
        return name

    def _torch_sym_int(self, d, add: bool = False):
        assert isinstance(
            d, (self.torch.SymInt, str)
        ), f"unexpected type for d={d}, type={type(d)}"
        value = None
        try:
            dyn_val = str(d.node._expr)
            value = dyn_val
        except AttributeError:
            pass

        if value is None:
            # Is it an integer?
            try:
                val_int = int(d)
                value = val_int
            except (TypeError, ValueError):
                pass
        else:
            # maybe an expression which is a single integer
            try:
                val_int = int(value)
                value = val_int
            except (TypeError, ValueError):
                pass

        if isinstance(value, int):
            assert not isinstance(value, self.torch.SymInt)
            return value

        if value is None:
            value = str(d)

        if value in self._dynamic_alias:
            value = self._dynamic_alias[value]

        if value not in self.dynamic_objects_rev:
            # The dynamic dimension does not seem to be registered.
            # Maybe it is constant.
            try:
                val_int = int(d)
                value = val_int
                return value
            except (TypeError, ValueError):
                pass

        if value in self.dynamic_objects:
            assert not isinstance(value, self.torch.SymInt)
            return value

        if (
            value not in self.dynamic_objects_rev
            and value not in self._known_value_shape
            and add
        ):
            self.dynamic_objects[value] = value
            self.dynamic_objects_rev[value] = [value]

        assert value in self.dynamic_objects_rev or value in self._known_value_shape, (
            f"value={value!r}, unable to find dimension {d!r} ({type(d)}) "
            f"(str(d)={str(d)!r}) in {self.dynamic_objects_rev} "
            f"or {self._dynamic_alias} or {self._known_value_shape}"
            f"{dir(d)}"
            f"{self.get_debug_msg()}"
        )
        assert not isinstance(
            value, self.torch.SymInt
        ), f"Unexpected type {type(value)} for d={d!r}"
        if value in self.dynamic_objects_rev:
            new_value = self.dynamic_objects_rev[value]
            assert isinstance(new_value, list), (
                f"Unexpected type {type(new_value)} for value={value!r}, d={d}"
                f"{self.get_debug_msg()}"
            )
            assert len(new_value) == 1, (
                f"Unexpected number of items in {new_value}, value={value!r}, d={d}"
                f"{self.get_debug_msg()}"
            )
            final = new_value[0]
            assert isinstance(final, tuple) or len(final) != 2, (
                f"Unexpected type {type(final)}, final={final}, value={value}, d={d}"
                f"{self.get_debug_msg()}"
            )
            name = final[0]
            assert isinstance(name, str), (
                f"Unexpected type {type(name)}, name={final}, value={value}, d={d}"
                f"{self.get_debug_msg()}"
            )
            return name

        # Its value is in self._known_value_shape. We still return its name.
        return value

    def verify_dynamic_shape(
        self,
        shape: Any,
        name: Optional[str] = None,
        add: bool = True,
    ) -> DYNAMIC_SHAPE:
        """
        The implementation of this method should be revisited.
        """
        if shape is None:
            return None
        if is_static_shape(shape):
            return tuple(int(i) for i in shape)
        new_shape = []
        for dim, d in enumerate(shape):
            if isinstance(d, int):
                new_shape.append(d)
                continue
            if isinstance(d, (self.torch.SymInt, str)):
                dyn_name = self._get_dynamic_dimension(name, dim)
                if dyn_name is not None:
                    new_shape.append(dyn_name)
                    continue

                value = self._torch_sym_int(d, add=add)
                assert (
                    value is not None
                ), f"Unexpected type {type(d)} in shape={shape}{self.get_debug_msg()}"
                new_shape.append(value)
                continue
            assert (
                d is not None
            ), f"Unexpected type {type(d)} in shape={shape}{self.get_debug_msg()}"
        return tuple(new_shape)

    def make_tensor_input(
        self, name: str, elem_type: Any, shape: STATIC_SHAPE, is_dimension: bool
    ) -> str:
        """
        Adds a tensor input to the onnx graph.

        :param name: name
        :param elem_type: element type
        :param shape: shape
        :param is_dimension: torch is using torch.SymInt to add a dynamic input
            to the graph
        :return: input name
        """
        if self.current_input < len(self.input_names):
            # The input needs to be renamed, an identity node is added.
            input_name = self.input_names[self.current_input]
            self.make_node(
                "Identity", [input_name], [name], check=False, name="make_tensor_input"
            )
        else:
            if is_dimension:
                # The convention is to have _dim_ in the name to tell
                # it is a dimension.
                input_name = f"{name}_dim_"
                self.make_node(
                    "Identity",
                    [input_name],
                    [name],
                    check=False,
                    name="make_tensor_input",
                )
            else:
                self.input_names.append(name)
                input_name = name
                self.set_name(name)
        assert (is_dimension and "_dim_" in input_name) or (
            not is_dimension and "_dim_" not in input_name
        ), (
            f"Inconsistence for input {name!r}, input_name={input_name!r}, "
            f"elem_type={elem_type}, shape={shape!r}, is_dimension={is_dimension}"
        )

        self.current_input += 1
        elem_type = self._get_type(elem_type)
        dyn_shape = self.verify_dynamic_shape(shape, name=input_name, add=True)

        if shape is not None:
            tuple_shape = tuple(shape)
            assert len(tuple_shape) == len(
                dyn_shape
            ), f"mismatch between shape={shape}, dynamic_shape={dyn_shape}"
            for a, b in zip(tuple_shape, dyn_shape):
                if a == b:
                    continue
                sa = str(a)
                sb = str(b)
                if sa == sb:
                    continue
                self._dynamic_alias[sa] = sb

        self.inputs.append(oh.make_tensor_value_info(input_name, elem_type, dyn_shape))
        if self.verbose:
            print(
                f"[GraphBuilder-{self._hash()}.make_tensor_input] {input_name}[{elem_type}:{dyn_shape}]"
            )
        assert (
            self.as_function or elem_type
        ), f"elem_type={elem_type!r} must be specified for input {name!r}"
        if shape:
            self.set_shape(name, dyn_shape)
            if input_name != name:
                self.set_shape(input_name, dyn_shape)
        if elem_type:
            self.set_type(name, elem_type)
            if input_name != name:
                self.set_type(input_name, elem_type)
        return name

    def make_tensor_output(
        self,
        name: Union[str, List[str]],
        elem_type: Optional[int] = None,
        shape: Optional[STATIC_SHAPE] = None,
        indexed: bool = True,
        is_dimension: bool = None,
    ) -> Union[str, List[str]]:
        """
        Adds a tensor output to the onnx graph.

        :param name: name
        :param elem_type: element type
        :param shape: shape
        :param indexed: the name must be indexed?
        :param is_dimension: torch is using torch.SymInt to add a dynamic input
            to the graph
        :return: output name
        """
        assert is_dimension is not None, (
            f"is_dimension must be specified for output name={name!r}, "
            f"elem_type={elem_type}, shape={shape!r}."
        )
        if isinstance(name, list):
            assert (
                not is_dimension
            ), f"name={name!r} not compatible with is_dimension=True"
            res = []
            for n in name:
                res.append(self.make_tensor_output(n, elem_type, shape))
            return res

        assert (
            not indexed or "_" in name
        ), f"Name {name!r} is not indexed like 'output_0'{self.get_debug_msg()}"
        assert (is_dimension and "_dim_" in name) or (
            not is_dimension and "_dim_" not in name
        ), (
            f"Inconsistence for input {name!r}, "
            f"elem_type={elem_type}, shape={shape!r}, "
            f"is_dimension={is_dimension}"
        )

        elem_type = self._get_type(elem_type, False)
        if not self.as_function and elem_type == 0:
            raise RuntimeError(f"Undefined element type for {name!r}.")
        dyn_shape = self.verify_shape(shape, name=name, elem_type=elem_type)
        self.outputs.append(oh.make_tensor_value_info(name, elem_type, dyn_shape))
        if self.verbose:
            print(
                f"[GraphBuilder-{self._hash()}.make_tensor_output] {name}[{elem_type}:{dyn_shape}]"
            )
        if dyn_shape:
            self.set_shape(name, dyn_shape)
        if elem_type:
            self.set_type(name, elem_type)
        return name

    def select_outputs(self, output_names: List[str]):
        """
        Selects new outputs. The type is assumed to be unknown.
        The method only wipes out the outputs to replace them by
        others. It assumes the unused nodes are removed afterwards.

        :param output_names: new outputs
        """
        new_outputs = []
        for name in output_names:
            if self.has_type(name) and self.has_rank(name):
                if self.has_shape(name):
                    out = oh.make_tensor_value_info(
                        name, self.get_type(name), self.get_shape(name)
                    )
                    if self.verbose:
                        print(
                            f"[GraphBuilder-{self._hash()}.make_tensor_output] "
                            f"{name}[{self.get_type(name)}:R{self.get_shape(name)}]"
                        )
                else:
                    out = oh.make_tensor_value_info(
                        name, self.get_type(name), [None] * self.get_rank(name)
                    )
                    if self.verbose:
                        print(
                            f"[GraphBuilder-{self._hash()}.make_tensor_output] "
                            f"{name}[{self.get_type(name)}:R{self.get_rank(name)}]"
                        )
            else:
                out = oh.make_value_info(name, TypeProto())
                if self.verbose:
                    print(f"[GraphBuilder-{self._hash()}.make_tensor_output] {name}")
            new_outputs.append(out)

        self.outputs = new_outputs

    def verify_shape(
        self,
        shape: Optional[DYNAMIC_SHAPE],
        elem_type: int,
        name: Optional[str] = None,
    ) -> Optional[DYNAMIC_SHAPE]:
        assert isinstance(
            elem_type, int
        ), f"elem_type must be an integer not {type(elem_type)}"
        assert shape is None or isinstance(
            shape, tuple
        ), f"Shape must be a tuple not {type(shape)}"
        if shape is None:
            return None
        assert is_static_shape(shape) or self.is_dynamic_shape(
            shape, allow_none=True
        ), (
            f"Shape={shape} is not a shape (type={[type(i) for i in shape]}), "
            f"name={name!r}, elem_type={elem_type}{self.get_debug_msg()}"
        )
        new_shape = self.verify_dynamic_shape(shape, name=name)
        return new_shape

    def _get_symbol(self, i: str) -> str:
        k = 0
        if self.has_type(i):
            k += 1
        if self.has_rank(i):
            k += 2
        if self.has_shape(i):
            k += 4
        return k

    def _debug_string_inputs(
        self, inputs: List[str], outputs: List[str], align: Optional[int] = None
    ) -> str:
        """
        Meaning:

        - ``"-"``: (0) none
        - ``"T"``: (1) type
        - ``"R"``: (2) rank
        - ``"U"``: (3) rank + type
        - ``"S"``: (4) shape
        - ``"V"``: (5) shape + type
        - ``"W"``: (6) shape + rank
        - ``"#"``: (7) shape + type + rank
        """
        st = ""
        c = "-TRUSVW#"
        for i in inputs:
            st += c[self._get_symbol(i)]
        st += ":"
        for o in outputs:
            st += c[self._get_symbol(o)]
        if align and len(st) < align:
            st += " " * (align - len(st))
        return st

    def _check_op_type(
        self,
        op_type: str,
        inputs: List[str],
        outputs: List[str],
        domain: str,
        name: str,
        attributes: Optional[List[AttributeProto]] = None,
        **kwargs: Dict[str, Any],
    ):
        assert (
            not op_type.startswith("Reduce")
            or domain != ""
            or (len(inputs) == 2 and "axes" not in kwargs)
            or len(inputs) == 1
        ), (
            f"Operator {op_type!r} defines twice the axes, kwargs={kwargs}, "
            f"len(inputs)={len(inputs)}{self.get_debug_msg()}"
        )
        assert (
            op_type != "Cast"
            or domain != ""
            or ("to" in kwargs and kwargs["to"] is not None)
        ), (
            f"Operator Cast needs arguments to but kwargs={kwargs}"
            f"{self.get_debug_msg()}"
        )
        assert op_type != "Concat" or domain != "" or len(inputs) > 1, (
            f"Concatenation of zero or one input is not necessary, "
            f"len(inputs)={len(inputs)}{self.get_debug_msg()} "
        )
        if self.main_opset <= 11:
            assert op_type != "Squeeze" or domain != "" or len(inputs) == 1, (
                f"Operator Squeeze is not correclty specified for opset "
                f"{self.main_opset}, inputs={inputs}, kwargs={kwargs}, "
                f"atts={attributes}{self.get_debug_msg()}"
            )
        else:
            n_entries = len(inputs) + len(attributes or []) + len(kwargs)
            assert op_type != "Squeeze" or domain != "" or n_entries in (1, 2), (
                f"Operator Squeeze is not correclty specified for opset "
                f"{self.main_opset}, n_entries={n_entries}, "
                f"inputs={inputs}, kwargs={kwargs}, "
                f"atts={attributes}{self.get_debug_msg()}"
            )
        assert (
            op_type not in {"NegXplus1", "ReplaceZero"} or domain != ""
        ), f"Type={op_type!r} and domain {domain!r} mismatch{self.get_debug_msg()}"

    def do_not_remove(self, node: NodeProto) -> bool:
        """Tells if a node should be removed or not."""
        return node.name.startswith("_DONOTREMOVE_")

    def make_node(
        self,
        op_type: str,
        inputs: Union[str, List[str]],
        outputs: Union[int, List[str], str] = 1,
        domain: str = "",
        attributes: Optional[List[AttributeProto]] = None,
        check: Optional[bool] = None,
        name: Optional[str] = None,
        sts: Optional[Dict[str, Any]] = None,
        do_not_remove: bool = False,
        **kwargs,
    ) -> Union[str, List[str]]:
        """
        Adds a node in the graph.

        :param op_type: operator type
        :param inputs: input names
        :param outputs: output names, may be None, in that case,
            the builder chooses them for the user
        :param domain: domain
        :param attributes: list of attributes to add as AttributeProto
        :param check: do some verification
        :param name: node name
        :param sts: if not specified, tries to set the shape and the type of
            the new results aftr the node is added, it is not possible
            for every node, there is no tool which determines the output shape
            of just one node
        :param do_not_remove: prevent this node from being removed
        :param kwargs: additional attributes to add the node
        :return: output names
        """
        assert name is not None and not name.startswith("None"), (
            f"It is good practice to give every node a name so that is "
            f"easier to see where this node is created but name={name!r} "
            f"and op_type={op_type!r}."
        )
        assert (
            not kwargs or not attributes
        ), f"Only attributes or kwargs can be filled for node {op_type!r}."
        if isinstance(inputs, tuple):
            inputs = list(inputs)
        if isinstance(outputs, int):
            if outputs < 1:
                raise ValueError(f"outputs={outputs} must be > 0.")
            lower = op_type.lower()
            output_names = [
                self.unique_name(f"_onx_{lower}{i}") for i in range(outputs)
            ]
        elif isinstance(outputs, str):
            output_names = [outputs]
        else:
            output_names = outputs
        if isinstance(inputs, str):
            inputs = [inputs]

        inputs, kwargs = self._partial_rewrite_opset_version(
            op_type, inputs, kwargs, domain=domain
        )

        if self.verbose == 2:
            print(
                f"[GraphBuilder-{self._hash()}.make_node]"
                f"[{self._debug_string_inputs(inputs, output_names)}] "
                f"{op_type}:{inputs}->{outputs}"
            )

        if check is not False:
            for i in inputs:
                assert isinstance(i, str), (
                    f"Unexpected type {type(i)} in {inputs}, op_type={op_type!r}, "
                    f"name={name!r}, {self.get_debug_msg()}"
                )
                if i == "":
                    # Optional input.
                    continue
                assert self.has_name(i), (
                    f"Input {i!r} does not exist for operator {op_type!r} "
                    f"({self._hash()}){self.get_debug_msg()}"
                )
            for i in output_names:
                assert isinstance(
                    i, str
                ), f"Unexpected type {type(i)} in {output_names}{self.get_debug_msg()}"
                if i == "":
                    # Optional output.
                    continue
                assert not self.has_name(i), (
                    f"Output {i!r} already exists for operator {op_type!r} "
                    f"({self._hash()}){self.get_debug_msg()}"
                )
        if check is True:
            for i in inputs:
                assert self.has_shape(i), f"Input {i!r} has no known shape."
                assert self.has_type(i), f"Input {i!r} has no known type."

        if do_not_remove:
            name = self.unique_node_name(f"_DONOTREMOVE_{name or ''}")
        elif name:
            name = self.unique_node_name(name)

        self._check_op_type(
            op_type,
            inputs,
            outputs,
            domain=domain,
            name=name,
            attributes=attributes,
            **kwargs,
        )

        # break?
        # if op_type == "ReduceSum":
        #    raise AssertionError(f"MANUAL BREAK{self.get_debug_msg()}")

        # next
        try:
            node = oh.make_node(
                op_type, inputs, output_names, domain=domain, name=name, **kwargs
            )
        except TypeError as e:
            iti = [type(i) for i in inputs]
            ito = (
                [type(o) for o in outputs]
                if isinstance(outputs, (tuple, list))
                else outputs
            )
            raise TypeError(
                f"A node {op_type!r} cannot be created with "
                f"inputs={inputs} (types={iti}), outputs={outputs} (types={ito}), "
                f"domain={domain!r}, kwargs={kwargs}."
            ) from e

        assert len(node.output) == len(set(node.output)) or "" in node.output, (
            f"Repeated outputs for node {node.op_type}({', '.join(node.input)}) -> "
            f"{', '.join(node.output)}"
        )

        if attributes:
            node.attribute.extend(attributes)

        if node.domain == "" and node.op_type in {"Constant", "ConstantOfShape"}:

            if len(node.attribute) == 1 and node.attribute[0].name == "value":
                t = node.attribute[0].t
                size = np.prod(t.dims)
                assert size < self.optimization_options.constant_size, (
                    f"A node Constant is created with a size {size} greater than "
                    f"the limit {self.optimization_options.constant_size}"
                    f"{self.get_debug_msg()}"
                )

            # A exact constant may be already existing,
            # In that case, we just return an identity node.
            origin = self.is_exact_same_constant(node)
            if origin is not None:
                if self.verbose > 2:
                    print(
                        f"[GraphBuilder-{self._hash()}.make_node] duplicated constant detected for "
                        f"{node.op_type}:{node.input}->{node.output}"
                    )
                node = oh.make_node(
                    "Identity",
                    [origin.output[0]],
                    [node.output[0]],
                    name=self.unique_node_name(".make_node"),
                )
                self.constants_alias_[node.output[0]] = origin.output[0]
            else:
                self.add_constant_node(node)

        # constant handling, shape, type
        self._make_node_set_type_shape_constant(node, sts=sts)

        if self.verbose > 3:
            print(
                f"[GraphBuilder-{self._hash()}.make_node] "
                f"[{self._debug_string_inputs(node.input, output_names)}] "
                f"{node.op_type}:{node.input}->{node.output}"
            )

        shape_set = self.simple_update_value_shape_with_node(node)

        # add the node
        for o in node.output:
            if o == "":
                continue
            self.set_name(o)
        self.nodes.append(node)

        if not shape_set:
            # second try
            self._make_node_set_type_shape(node)

        if len(output_names) == 1:
            return output_names[0]
        return output_names

    @property
    def last_added_node(self):
        return self.nodes[-1] if self.nodes else None

    def _partial_rewrite_opset_version(
        self, op_type: str, inputs: List[str], kwargs: Dict[str, Any], domain: str
    ) -> Tuple[List[str], Dict[str, Any]]:
        if domain == "":
            opset = self.opsets[""]
            if op_type == "Unsqueeze":
                if opset < 13 and len(inputs) == 2:
                    assert isinstance(
                        inputs[1], (list, np.ndarray)
                    ), f"Unexpected type for axis={inputs[1]} and operator Unsqueeze"
                    kwargs["axes"] = (
                        [inputs[1]]
                        if isinstance(inputs[1], int)
                        else inputs[1].tolist()
                    )
                    inputs = inputs[:1]
                elif opset >= 13 and len(inputs) == 1:
                    name = self.make_initializer(
                        "", np.array(kwargs["axes"], dtype=np.int64)
                    )
                    inputs.append(name)
                    del kwargs["axes"]
        return inputs, kwargs

    def _make_node_set_type_shape_constant(
        self, node: NodeProto, sts: Optional[Dict[str, Any]]
    ):
        if node.domain != "":
            return
        if node.op_type == "Constant":
            if len(node.attribute) == 1 and node.attribute[0].name == "value":
                size = np.prod(node.attribute[0].t.dims)
            else:
                size = len(node.SerializeToString())
            assert size < self.optimization_options.constant_size, (
                f"A node Constant holds a tensor bigger than "
                f"the constant: {size} >= {self.optimization_options.constant_size}."
            )
            k = node.output[0]
            self.constants_[k] = node
            shape = self._get_tensor_shape(node)
            dtype = self._get_tensor_type(node)
            self.set_shape(k, shape)
            self.set_type(k, dtype)
            if self.verbose and (self.verbose > 3 or np.prod(shape) > 100):
                print(f"[GraphBuilder-{self._hash()}.make_node] {k}[{dtype}:{shape}]")
        elif node.op_type == "Identity":
            if self.has_shape(node.input[0]):
                self.set_shape(node.output[0], self.get_shape(node.input[0]))
            elif self.has_rank(node.input[0]):
                self.set_rank(node.output[0], self.get_rank(node.input[0]))
            if self.has_type(node.input[0]):
                self.set_type(node.output[0], self.get_type(node.input[0]))
            if self.is_constant(node.input[0]):
                self.constants_[node.output[0]] = node
        elif node.op_type == "Shape":
            self.set_type(node.output[0], TensorProto.INT64)
            if self.has_shape(node.input[0]) and len(node.attribute) == 0:
                shape = self.get_shape(node.input[0])
                self.set_shape(node.output[0], (len(shape),))
            else:
                self.set_rank(node.output[0], 1)
        elif all(map(self.is_constant, node.input)):
            for o in node.output:
                self.constants_[o] = node
            if len(node.output) == 1:
                cst, _ = self.compute_constant(node.output[0], exc=False)
                if cst is not None:
                    self.set_type(node.output[0], dtype_to_tensor_dtype(cst[0].dtype))
                    self.set_shape(node.output[0], tuple(cst[0].shape))
        elif not sts:
            if node.op_type == "GatherElements":
                if self.has_rank(node.input[0]) and self.has_rank(node.input[0]):
                    r1 = self.get_rank(node.input[0])
                    assert r1 == self.get_rank(node.input[1]), (
                        f"Rank mismatch {r1} != {self.get_rank(node.input[1])} "
                        f"(GatherElements:{node.input}){self.get_debug_msg()}"
                    )
                    self.set_rank(node.output[0], r1)

    def get_attribute(
        self, node: NodeProto, att_name: str, exc: bool = True
    ) -> Optional[AttributeProto]:
        """
        Returns an attribute for a node.
        """
        for att in node.attribute:
            if att.name == att_name:
                return att
        assert (
            not exc
        ), f"Unable to find attribute {att_name!r} for node type {node.op_type!r} in node {node}"
        return None

    def get_attributes_with_default(
        self, node: NodeProto, **default_values
    ) -> Dict[str, Any]:
        """
        Returns int or float attributes. If missing, the default value is returned.

        :param node: node
        :param default_values: default values
        """
        res = {}
        for att in node.attribute:
            if att.name in default_values:
                def_val = default_values[att.name]
                if isinstance(def_val, int):
                    res[att.name] = att.i
                elif isinstance(def_val, float):
                    res[att.name] = att.f
                elif isinstance(def_val, str):
                    res[att.name] = att.s
                else:
                    raise TypeError(
                        f"Unexpected type {type(def_val)} for attribute name {att.name!r}, "
                        f"attribute={att}"
                    )
        for k, v in default_values.items():
            if k not in res:
                res[k] = v
        return res

    def _make_node_set_type_shape(self, node: NodeProto):
        if node.domain != "":
            set_shape_type_custom(self, node)
        else:
            set_shape_type_op_any(self, node)

    def make_nodes(
        self,
        builder: "GraphBuilder",
        input_names: List[str],
        output_names: List[str],
        prefix: str = "",
    ) -> Union[str, List[str]]:
        """
        Appends all nodes and initializers from another builder.
        Handles the renaming of results.
        The content stored in 'builder' is modified inplace to avoid copying.

        :param builder: other builder
        :param input_names: input names
        :param output_names: output names
        :param prefix: prefix all name from this builder
        :return: output names
        """
        renaming = {}
        for init, value in builder.initializers_dict.items():
            name = self.unique_name(f"{prefix}{init}")
            renaming[init] = name
            if isinstance(value, TensorProto):
                value.name = name
            self.initializers_dict[name] = value

            self.constants_[name] = None
            self.set_name(name)
            self.set_shape(name, builder._known_shapes[init])
            self.set_type(name, builder._known_types[init])

        for k, v in builder.dynamic_objects.items():
            self.make_dynamic_object(k, v)

        assert len(input_names) == len(builder.inputs), (
            f"Inconsistency between input_names={input_names} "
            f"and inputs={builder.inputs}"
        )
        for name, inp in zip(input_names, builder.inputs):
            new_name = self.unique_name(f"{prefix}{inp.name}")
            renaming[inp.name] = new_name
            self.make_node("Identity", [name], [new_name], name=".make_nodes")

        for node in builder.nodes:
            assert name is not None and not name.startswith("None"), (
                f"It is good practice to give every node a name so that is "
                f"easier to see where this node is created but name={name!r} "
                f"and op_type={node.op_type!r}."
            )
            new_inputs = [renaming[i] for i in node.input]
            new_outputs = [self.unique_name(f"{prefix}{o}") for o in node.output]
            for o, no in zip(node.output, new_outputs):
                renaming[o] = no
            self.make_node(
                node.op_type,
                new_inputs,
                new_outputs,
                domain=node.domain,
                attributes=node.attribute,
                check=False,
                name=node.name,
            )
            for o, no in zip(node.output, new_outputs):
                if builder.has_shape(o):
                    shape = builder.get_shape(o)
                    if None in shape:
                        self.set_rank(no, len(shape))
                    else:
                        self.set_shape(no, shape)
                if builder.has_type(o):
                    self.set_type(no, builder.get_type(o))

        assert len(output_names) == len(builder.outputs), (
            f"Inconsistency between output_names={output_names} and "
            f"outputs={builder.outputs}, renaming={renaming}."
        )
        for name, out in zip(output_names, builder.outputs):
            self.make_node("Identity", [renaming[out.name]], [name], name=".make_nodes")

        # opsets and domains
        for o, v in builder.opsets.items():
            if o in self.opsets:
                assert self.opsets[o] == builder.opsets[o], (
                    f"Opset mismatch for domain {o!r}, "
                    f"{self.opsets[o]} != {builder.opsets[o]}."
                )
                continue
            self.opsets[o] = v

        if len(output_names) == 1:
            return output_names[0]
        return output_names

    def from_array(
        self, arr: "torch.Tensor", name: str = None  # noqa: F821
    ) -> TensorProto:
        """
        Converts a torch Tensor into a TensorProto.
        """
        import sys

        if not isinstance(arr, self.torch.Tensor):
            raise TypeError(f"Unexpected type {type(arr)}.")
        if arr.is_sparse:
            raise NotImplementedError(
                f"Sparse tensor is not supported yet but initializer {name!r} is."
            )

        arr_cont = arr.contiguous() if not arr.is_contiguous() else arr
        arr_cpu = arr_cont.cpu()
        if arr_cpu.data_ptr() == arr.data_ptr():
            copy = arr_cpu.clone().detach().requires_grad_(False)
            assert arr_cpu.data_ptr() != copy.data_ptr()
            np_arr = np.from_dlpack(copy)
        else:
            np_arr = np.from_dlpack(arr_cpu.detach())

        tensor = TensorProto()
        tensor.dims.extend(arr_cpu.shape)
        tensor.name = name
        itype = self._get_type(arr_cpu.dtype)
        assert not hasattr(TensorProto, "INT4") or itype not in {
            TensorProto.INT4,
            TensorProto.UINT4,
        }, f"Type {arr.dtype} is not supported yet for name={name!r}"
        tensor.data_type = itype

        if self.verbose and np.prod(arr_cpu.shape) > 100:
            print(
                f"[GraphBuilder-{self._hash()}.from_array] {tensor.data_type}[{arr_cpu.shape}]"
            )

        if sys.byteorder == "big":
            tensor.raw_data = np_arr.tobytes()
            np_dtype = oh.tensor_dtype_to_np_dtype(tensor.data_type)
            np.byteswap(np.frombuffer(tensor.raw_data, dtype=np_dtype), inplace=True)
        else:
            tensor.raw_data = np_arr.tobytes()

        return tensor

    def _build_initializers(self) -> List[TensorProto]:
        res = []
        for k, v in sorted(self.initializers_dict.items()):
            if isinstance(v, self.torch.Tensor):
                # no string tensor
                t = self.from_array(v, name=k)
                res.append(t)
                continue
            if isinstance(v, np.ndarray):
                if self.verbose and np.prod(v.shape) > 100:
                    print(
                        f"[GraphBuilder-{self._hash()}._build_initializers] onh.from_array:{k}:{v.dtype}[{v.shape}]"
                    )
                t = onh.from_array(v, name=k)
                res.append(t)
                continue
            if isinstance(v, TensorProto):
                res.append(v)
                continue
            raise TypeError(
                f"Unable to convert initializer {k!r} with type "
                f"{type(v)} into a TensorProto."
            )
        return res

    def get_debug_msg(self) -> str:
        """
        Returns a string providing as much information as possible
        to help the developper understand why a conversion failed.
        """

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
                return t.detach().numpy().ravel().tolist()
            if hasattr(t, "size"):
                return t.ravel().tolist()
            if hasattr(t, "dims"):
                a = onh.to_array(t)
                return a.ravel().tolist()
            raise RuntimeError(f"Values unknown for type {type(t)}-{t}.")

        rows = ["", "--DEBUG--", "--SHAPE--"]
        rows.append(f"dynamic_objects={pprint.pformat(self.dynamic_objects)[:10000]}")
        rows.append(
            f"dynamic_objects_rev={pprint.pformat(self.dynamic_objects_rev)[:10000]}"
        )
        rows.append(f"dynamic_alias={pprint.pformat(self._dynamic_alias)[:10000]}")
        rows.append(f"dynamic_shapes={pprint.pformat(self.dynamic_shapes)[:10000]}")
        rows.append(
            f"_known_value_shape={pprint.pformat(self._known_value_shape)[:10000]}"
        )
        rows.append("--TORCH-SHAPES--")
        for kk, vv in self._known_torch_value.items():
            rows.append(
                f"{kk}: {vv} --- "
                f"{self.get_type(kk) if self.has_type(kk) else ''}:"
                f"{self.get_rank(kk) if self.has_rank(kk) else ''}:"
                f"{self.get_shape(kk) if self.has_shape(kk) else ''}:"
            )
            if len(rows) > 1000:
                rows.append("...")
                rows.append(
                    f"Stopped with {len(self.initializers_dict)} "
                    f"initializers and {len(self.nodes)} nodes."
                )
                return "\n".join(rows)
        rows.append("--ONNX--")
        for k, v in self._debug_msg.items():
            rows.append(f"-- {k} --")
            rows.append(pprint.pformat(v) if isinstance(v, dict) else str(v))
            if len(rows) > 1000:
                rows.append("...")
                rows.append(
                    f"Stopped with {len(self.initializers_dict)} "
                    f"initializers and {len(self.nodes)} nodes."
                )
                return "\n".join(rows)
        rows.append("--")
        hs = self._hash()
        for io in self.inputs:
            shh = _nice_shape(io.type.tensor_type.shape)
            rows.append(
                f"[GraphBuilder-{hs}.make_tensor_input] {io.name}"
                f"[{io.type.tensor_type.elem_type}:{shh}]"
            )
        for name, init in self.initializers_dict.items():
            sval = "" if _size(init) > 5 else f":{_values(init)}"
            rows.append(
                f"[GraphBuilder-{hs}.make_initializer] "
                f"{name}[{_dtype(init)}:{_shape(init)}{sval}]"
            )
            if len(rows) > 1000:
                rows.append("...")
                rows.append(
                    f"Stopped with {len(self.initializers_dict)} "
                    f"initializers and {len(self.nodes)} nodes."
                )
                return "\n".join(rows)
        for node in self.nodes:
            if node is None:
                continue
            if node.op_type == "Cast":
                ext = f">{node.attribute[0].i}"
            else:
                ext = ""
            rows.append(
                f"[GraphBuilder-{hs}.make_node] "
                f"{_align(node.name, 15)} "
                f"[{self._debug_string_inputs(node.input, node.output, 6)}] "
                f"{node.op_type}{ext}:{node.input}->{node.output}"
            )
            if len(rows) > 1000:
                rows.append("...")
                rows.append(
                    f"Stopped with {len(self.initiliazer)} "
                    f"initializers and {len(self.nodes)} nodes."
                )
                return "\n".join(rows)
        for io in self.outputs:
            shh = _nice_shape(io.type.tensor_type.shape)
            rows.append(
                f"[GraphBuilder-{hs}.make_tensor_output] {io.name}"
                f"[{io.type.tensor_type.elem_type}:{shh}]"
            )

        rows.append(
            f"Completed with {len(self.initializers_dict)} "
            f"initializers and {len(self.nodes)} nodes."
        )
        return "\n".join(rows)

    def process(
        self,
        graph_module: "torch.f.GraphModule",  # noqa: F821
        interpreter: "Interpreter",  # noqa: F821
    ):
        self._debug_msg["process.graph_module"] = graph_module.graph

        # looks into output marked as "alias_of_input"
        # see https://pytorch.org/functorch/main/_modules/torch/_functorch/aot_autograd.html
        # in that case, gen_alias_from_base is mixing the input data and the output stride
        # places = []
        # for node in graph_module.graph.nodes:
        #     if node.op == "placeholder":
        #         places.append(node)
        # for node in places:
        #     with graph_module.graph.inserting_after(node):
        #         cloned_node = graph_module.graph.call_method("clone", args=(node.target,))
        #         node.replace_all_uses_with(cloned_node)
        # graph_module.recompile()

        for i, node in enumerate(graph_module.graph.nodes):
            self._debug_msg["process.progress"] = (
                f"node {i}/{len(graph_module.graph.nodes)} "
            )
            interpreter.run_node(node)

    def to_onnx(
        self, as_function: bool = False, optimize: bool = True
    ) -> Union[FunctionProto, ModelProto]:
        """
        Conversion to onnx. Only then the initializer are converted into
        TensorProto.

        :param as_function: converts the graph as a FunctionProto or a ModelProto
        :param optimize: disable or enable the optimization,
            the optimization are set when the class constructor is called
        :return: the proto
        """
        if len(self.nodes) == 0:
            raise RuntimeError(
                f"The onnx model is empty (no node).\n{self.get_debug_msg()}"
            )
        if optimize:
            self.optimize()
        assert len(self.nodes) > 0, (
            f"The onnx model is empty after optimization (no node)."
            f"\n{self.get_debug_msg()}"
        )
        assert not as_function, "Export as FunctionProto is not tested yet."

        opsets = [oh.make_opsetid(*o) for o in self.opsets.items()]
        if as_function:
            return oh.make_function(
                self.nodes,
                self.name,
                [i.name for i in self.inputs],
                [o.name for o in self.outputs],
                domain=self.domain,
            )

        if self.verbose:
            print(f"[GraphBuilder-{self._hash()}.to_onnx] make_model")

        # graph = oh.make_graph(
        #    self.nodes, "experiment", self.inputs, self.outputs, dense
        # )

        model = ModelProto()
        model.graph.CopyFrom(GraphProto())

        model.graph.node.extend(self.nodes)
        model.graph.name = "experiment"
        model.graph.input.extend(self.inputs)
        model.graph.output.extend(self.outputs)

        # initializer

        if sys.byteorder == "big":
            dense = self._build_initializers()
            model.graph.initializer.extend(dense)
        else:
            # Let's try to minimize the time.
            for k, v in self.initializers_dict.items():

                if isinstance(v, TensorProto):
                    if self.verbose:
                        print(
                            f"[GraphBuilder-{self._hash()}._build_initializers] "
                            f"TensorProto-{k}:{v.data_type}[{tuple(v.dims)}]"
                        )
                    model.graph.initializer.append(v)
                    continue

                if self.verbose:
                    print(
                        f"[GraphBuilder-{self._hash()}._build_initializers] "
                        f"{type(v)}-{k}:{v.dtype}[{v.shape}]"
                    )
                if isinstance(v, np.ndarray):
                    itype = dtype_to_tensor_dtype(v.dtype)
                    if itype in {
                        TensorProto.BOOL,
                        TensorProto.STRING,
                        TensorProto.UNDEFINED,
                        TensorProto.COMPLEX64,
                        TensorProto.COMPLEX128,
                        getattr(TensorProto, "UINT4", 0),
                        getattr(TensorProto, "INT4", 0),
                    }:
                        t = onh.from_array(v, name=k)
                        model.graph.initializer.append(t)
                        continue

                    from_np = True
                else:
                    assert isinstance(
                        v, self.torch.Tensor
                    ), f"tensor {k!r} has un unexpected type {type(v)}"
                    from_np = False
                    itype = dtype_to_tensor_dtype(v.dtype)

                # How to avoid a copy?
                if from_np:
                    tensor = TensorProto()
                    tensor.name = k
                    tensor.dims.extend(v.shape)
                    tensor.data_type = itype
                    tensor.raw_data = v.tobytes()
                else:
                    tensor = self.from_array(v, name=k)

                model.graph.initializer.append(tensor)

        # graph.sparse_initializer.extend(sparse_initializer)
        # graph.value_info.extend(value_info)
        # graph.doc_string = doc_string

        model.opset_import.extend(opsets)
        model.functions.extend(self.functions)

        # model = oh.make_model(graph, opset_imports=opsets, functions=self.functions)

        if self.ir_version:
            model.ir_version = self.ir_version
        elif "" in self.opsets:
            model.ir_version = _default_OPSET_TO_IR_VERSION()[self.opsets[""]]

        if len(model.graph.node) == 0:
            raise RuntimeError(
                f"The onnx model is empty after export to onnx (no node)."
                f"\n{self.get_debug_msg()}"
            )

        # restores the existing value_info
        done = (
            set(init.name for init in model.graph.initializer)
            | set(i.name for i in model.graph.input)
            | set(i.name for i in model.graph.output)
        )
        for val in self.value_info:
            if self.has_name(val.name):
                model.graph.value_info.append(val)
                done.add(val.name)

        # adding shape information
        addition = []
        for name in self._known_names:
            if name in done or not self.has_type(name) or self.get_type(name) == 0:
                continue
            if self.has_shape(name):
                addition.append(
                    oh.make_tensor_value_info(
                        name, self.get_type(name), list(self.get_shape(name))
                    )
                )
            elif self.has_rank(name):
                addition.append(
                    oh.make_tensor_value_info(
                        name, self.get_type(name), [None] * self.get_rank(name)
                    )
                )
        if addition:
            model.graph.value_info.extend(addition)
        return model

    def io_names(self):
        """
        Returns the list of inputs, output for nodes.
        """
        input_names = {i.name for i in self.inputs}
        init_names = set(self.initializers_dict)
        output_names = {i.name for i in self.outputs}
        rows = [
            "",
            f"I<-[{','.join(sorted(input_names))}]",
            f"C<-[{','.join(sorted(init_names))}]",
        ]
        for node in self.nodes:
            rows.append(
                f"N:{node.op_type}:[{','.join(sorted(node.input))}]->[{','.join(sorted(node.output))}]"
            )
        rows.append(f"O->[{','.join(sorted(output_names))}]")
        rows.append("")
        return "\n".join(rows)

    def optimize(self) -> List[Dict[str, Any]]:
        """
        Optimizes a graph.
        Returns the list of applied processed.
        """

        def _check(stats, step):
            begin = time.perf_counter()
            assert (
                len(self.nodes) > 0
            ), f"The onnx model is empty (step {step}, no node).\n{self.get_debug_msg()}"
            known = set(n.name for n in self.inputs)
            known |= set(self.initializers_dict)
            for node in self.nodes:
                assert (
                    node.domain in self.opsets
                ), f"Domain {node.domain!r} is not registered in {self.opsets}"
                for i in node.input:
                    if i == "":
                        continue
                    assert (
                        i in known
                    ), f"Unknown input {i!r}, step {step!r}  in node {node}"
                known |= set(node.output)
            for o in self.outputs:
                assert o.name in known, f"Unknown output {o.name!r}, step {step!r} "
            stats.append(
                dict(pattern=f"check_{step}", time_in=time.perf_counter() - begin)
            )

        statistics = []
        main_begin = time.perf_counter()

        _check(statistics, "A")
        if self.optimization_options.remove_identity:
            begin = time.perf_counter()
            nr, na = self.remove_identity_nodes()
            statistics.append(
                dict(
                    pattern="remove_identity_nodes",
                    removed=nr,
                    added=na,
                    time_in=time.perf_counter() - begin,
                )
            )
            _check(statistics, "B")
        if self.optimization_options.remove_unused:
            begin = time.perf_counter()
            n = self.remove_unused()
            statistics.append(
                dict(
                    pattern="remove_unused",
                    removed=n,
                    time_in=time.perf_counter() - begin,
                )
            )
            _check(statistics, "C")
        if self.optimization_options.constant_folding:
            begin = time.perf_counter()
            n = self.constant_folding()
            statistics.append(
                dict(
                    pattern="constant_folding",
                    removed=n,
                    time_in=time.perf_counter() - begin,
                )
            )
            _check(statistics, "D")
            if self.optimization_options.remove_unused:
                begin = time.perf_counter()
                n = self.remove_unused()
                statistics.append(
                    dict(
                        pattern="remove_unused",
                        removed=n,
                        time_in=time.perf_counter() - begin,
                    )
                )
                _check(statistics, "E")
        if self.optimization_options.patterns:
            assert (
                self.optimization_options.remove_unused
            ), "remove_unused must be positive for pattern optimizations"
            n = len(self.nodes)
            begin = time.perf_counter()
            res = self.optimize_with_patterns()
            statistics.extend(res)
            statistics.append(
                dict(
                    pattern="pattern_optimization",
                    removed=n - len(self.nodes),
                    time_in=time.perf_counter() - begin,
                )
            )
            _check(statistics, "F")
            begin = time.perf_counter()
            n = self.remove_unused()
            statistics.append(
                dict(
                    pattern="remove_unused",
                    removed=n,
                    time_in=time.perf_counter() - begin,
                )
            )
            _check(statistics, "G")

        if self.verbose > 1:
            duration = time.perf_counter() - main_begin
            print(
                f"[GraphBuilder] done with "
                f"{len(self.nodes)} nodes in {duration:.3f}"
            )
            msg = self._compile_statistics(statistics)
            print(msg)

        return statistics

    def _compile_statistics(self, statistics: List[Dict[str, Any]]) -> str:
        stats = {}
        for obs in statistics:
            pattern = obs["pattern"]
            if pattern not in stats:
                stats[pattern] = {
                    "time_in": 0.0,
                    "iteration": [],
                    "match_index": [],
                    "removed": 0,
                    "added": 0,
                    "instances": 0,
                }
            o = stats[pattern]
            for k, v in obs.items():
                if k == "pattern":
                    continue
                if k in {"time_in", "removed", "added", "instances"}:
                    o[k] += v
                    continue
                o[k].append(v)

        rows = []
        for k, v in sorted(stats.items()):
            line = (
                f"    STAT {k} +{v['added']} -{v['removed']} "
                f"#it={len(set(v['iteration']))} "
                f"maxmatch={max(v['match_index']) if v['match_index'] else 0} "
                f"i={v['instances']} - time={v['time_in']}"
            )
            rows.append(line)
        return "\n".join(rows)

    def optimize_with_patterns(self) -> List[Dict[str, Any]]:
        """
        Optimizes this graph with patterns.
        """
        from ..xoptim import GraphBuilderPatternOptimization

        gro = GraphBuilderPatternOptimization(
            self,
            verbose=max(self.verbose, self.optimization_options.verbose),
            patterns=self.optimization_options.patterns,
            recursive=self.optimization_options.recursive,
            verifies=self.optimization_options.verifies,
            dump_applied_patterns=self.optimization_options.dump_applied_patterns,
            processor=self.optimization_options.processor,
        )
        return gro.optimize(
            max_iter=self.optimization_options.max_iter,
            remove_identity=self.optimization_options.remove_identity,
            stop_after=self.optimization_options.stop_after,
        )

    def remove_unused(self) -> int:
        """
        Simple function to remove unused nodes.
        It does not look into subgraphs and assumes there is none.
        Everything is done in one pass.
        Returns the number of removed nodes.
        """
        start = len(self.nodes)
        # mark outputs
        marked = {o.name: set() for o in self.outputs}
        for node in reversed(self.nodes):
            used = False
            for o in node.output:
                if o in marked:
                    for i in node.input:
                        marked[o].add(i)
                        used = True
            if used or self.do_not_remove(node):
                for i in node.input:
                    marked[i] = set()

        # removed nodes
        removed = set()
        marked_set = set(marked)
        for ind, node in enumerate(self.nodes):
            if not (set(node.output) & marked_set):
                removed.add(ind)

        if self.optimization_options.verbose > 1:
            for k, v in self.initializers_dict.items():
                if k not in marked:
                    v = self.initializers_dict[k]
                    if hasattr(v, "dtype") and hasattr(v, "shape"):
                        print(
                            f"[GraphBuilder.remove_unused] remove_initializer:{k}:{v.dtype}[{v.shape}]"
                        )
                    else:
                        print(f"[GraphBuilder.remove_unused] remove_initializer:{k}]")

        self.initializers_dict = {
            k: v for k, v in self.initializers_dict.items() if k in marked
        }
        self.constants_ = {k: v for k, v in self.constants_.items() if k in marked}

        if self.optimization_options.verbose > 2:
            for i in removed:
                node = self.nodes[i]
                print(
                    f"[GraphBuilder.remove_unused_node] remove {node.op_type}-{node.name} -> {node.output}"
                )

        self.nodes = [node for i, node in enumerate(self.nodes) if i not in removed]
        return start - len(self.nodes)

    def _apply_transpose(
        self, node: NodeProto, feeds: Dict[str, "torch.Tensor"]  # noqa: F821
    ) -> "torch.Tensor":  # noqa: F821
        perm = None
        for att in node.attribute:
            if att.name == "perm":
                perm = tuple(att.ints)
                break
        assert perm, f"perm not here in node {node}"
        x = feeds[node.input[0]]
        if isinstance(x, np.ndarray):
            x = self.torch.Tensor(x)
        return [self.torch.permute(x, perm)]

    def compute_constant(
        self, name: str, exc: bool = True
    ) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
        assert self.is_constant(name), f"Name {name!r} is not a constant."
        if name is self.initializers_dict:
            print("-----", name)
            return self.initializers_dict[name], None
        v = self.constants_[name]
        assert isinstance(v, NodeProto), f"Unexpected type {type(v)} for name={name!r}"
        feeds = {i: self.get_constant(i, exc=exc, computed_value=True) for i in v.input}
        for val in feeds.values():
            if val is None:
                return None, None
        if v.op_type == "Transpose":
            # bypassing onnx.numpy_helper.from_array, too slow
            output = self._apply_transpose(v, feeds)
        else:
            ref = ExtendedReferenceEvaluator(v)
            output = ref.run(None, feeds)
        new_outputs = []
        for name, val in zip(v.output, output):
            if self.has_type(name):
                # numpy changes the expected type sometimes (like transpose(x: float36) --> float32)
                itype = self.get_type(name)
                if hasattr(val, "detach"):
                    val = val.to(onnx_dtype_to_torch_dtype(itype))
                else:
                    val = val.astype(oh.tensor_dtype_to_np_dtype(itype))
            self.constants_computed_[name] = val
            new_outputs.append(val)
        return new_outputs, feeds

    def constant_folding(self, convert_into_initializer: bool = True) -> int:
        """
        Folds all constants. Constants are marked during the creation of the graph.
        There is no need to propagate this information.

        :param convert_into_initializer: moves the constant as an initializer,
            otherwise, just evaluates it
        :return: number of removed nodes
        """
        if self.verbose > 1:
            begin_ = time.perf_counter()
            print(
                f"[GraphBuilder.constant_folding] starts with "
                f"{len(self.constants_)} constants and "
                f"{len(self.nodes)} nodes."
            )
        start = len(self.nodes)
        updates = {}
        node_to_remove = set()
        for k, v in self.constants_.items():
            if v is None:
                # this is an initiliazer
                continue
            # a node
            if all(map(self.is_constant, v.input)):
                if convert_into_initializer:
                    node_to_remove.add(tuple(v.output))
                # node evaluation
                output, feeds = self.compute_constant(k)
                for name, value in zip(v.output, output):
                    updates[name] = None
                    if convert_into_initializer:
                        self.initializers_dict[name] = value
                    else:
                        updates[name] = value
                    if self.verbose > 3:
                        print(
                            f"[GraphBuilder.constant_folding] fold_constant:"
                            f"{v.op_type}:{name}[{value.dtype}:"
                            f"{value.shape}]:from:{','.join(sorted(feeds))}"
                        )

        self.constants_.update(updates)
        new_nodes = []
        for node in self.nodes:
            if self.do_not_remove(node):
                new_nodes.append(node)
                continue
            if tuple(node.output) in node_to_remove:
                continue
            new_nodes.append(node)
        self.nodes = new_nodes
        if self.verbose > 1:
            print(
                f"[GraphBuilder.constant_folding] ends with "
                f"{len(self.constants_)} constants and "
                f"{len(self.nodes)} nodes in "
                f"{time.perf_counter() - begin_} seconds"
            )
        return start - len(self.nodes)

    def remove_identity_nodes(self) -> Tuple[int, int]:
        """
        Removes identity nodes. Returns the number of removed nodes
        and the number of added nodes.

        .. note::

            onnxruntime does not handle well when it is executing from domain
            *'org.pytorch.aten'* (ATen for example) which outputs results
            on CPU where the expected output is on CUDA. An identity node must be
            kept or inserted in that case. In that particular case, a node can be
            marked so that it does not get deleted: its name must start with
            ``'_DONOTREMOVE_'``.
        """
        if self.verbose > 1:
            begin_ = time.perf_counter()
            print(
                f"[GraphBuilder.remove_identity_nodes] "
                f"starts with {len(self.nodes)}"
            )
        # first pass: detect replacements
        new_nodes = []
        input_names = set(i.name for i in self.inputs)
        output_names = set(i.name for i in self.outputs)
        replacements = {}
        replacements_rev = {}
        removed = 0
        for node in self.nodes:
            if node.op_type != "Identity" or self.do_not_remove(node):
                new_nodes.append(node)
                continue

            if node.output[0] not in output_names:
                old_name, new_name = node.output[0], node.input[0]
            elif (
                node.input[0] not in input_names
                and node.input[0] not in output_names
                and node.input[0] not in replacements
            ):
                old_name, new_name = node.input[0], node.output[0]
            else:
                new_nodes.append(node)
                continue

            # the new name can be set for replacements as well
            if new_name in replacements:
                new_name = replacements[new_name]
                assert new_name not in replacements, (
                    f"Name {old_name!r} still in {replacements}, "
                    f"node.op_type={node.op_type!r}, "
                    f"node.input={node.input}, node.output={node.output}, "
                    f"input_names={input_names}, output_names={output_names}"
                )
            if old_name in replacements_rev:
                old_old_name = replacements_rev[old_name]
                replacements[old_old_name] = new_name
                replacements_rev[new_name] = old_old_name
            if old_name in replacements:
                replacements[replacements[old_name]] = new_name
            assert new_name not in replacements, (
                f"Name {old_name!r} still in {replacements}, "
                f"node.op_type={node.op_type!r}, "
                f"node.input={node.input}, node.output={node.output}, "
                f"input_names={input_names}, output_names={output_names}"
            )

            if old_name in replacements_rev:
                # A tricky case:
                # x -> Identity -> a -> Identity -> b -> Flatten -> output1
                # x -> Identity -> output0
                # How x should be renamed?
                assert new_name in output_names, (
                    f"replacement {old_name}->{new_name} "
                    f"is not possible because of "
                    f"[{replacements_rev[old_name]}->{old_name}] "
                    f"and {new_name!r} is not an output"
                )
                updates = {}
                for k, v in replacements.items():
                    if v == old_name:
                        updates[k] = new_name
                replacements.update(updates)

            replacements[old_name] = new_name
            replacements_rev[new_name] = old_name

            # verification
            if len(replacements) < 500:
                for k, v in replacements.items():
                    assert v not in replacements, (
                        f"replacement {k}->{v} is not possible because of "
                        f"[{v}->{replacements[v]}], old_name={old_name!r}, "
                        f"new_name={new_name!r}"
                    )
            removed += 1

        # second pass: replacements in initializer
        if self.verbose > 1:
            print(
                f"[GraphBuilder.remove_identity_nodes] found "
                f"{len(replacements)} replacements"
            )
        for k, v in replacements.items():
            if k in self.initializers_dict:
                if self.optimization_options.verbose > 2:
                    print(
                        f"[GraphBuilder.remove_identity_nodes] "
                        f"rename initializer {k!r} by {v!r}"
                    )
                self.initializers_dict[v] = self.initializers_dict[k]
                del self.initializers_dict[k]
                assert self.constants_[v]
                self.constants_[v] = self.constants_[k]
                del self.constants_[k]

        # third pass: replacements in node
        if self.verbose > 1:
            print(
                f"[GraphBuilder.remove_identity_nodes] " f"kept {len(new_nodes)} nodes"
            )
        self.nodes = []
        added = 0
        for node in new_nodes:
            repo = {o for o in node.output if o in replacements}
            repi = {o for o in node.input if o in replacements}
            if repi or repo:
                new_inputs = [replacements.get(i, i) for i in node.input]
                new_outputs = [replacements.get(i, i) for i in node.output]
                assert set(new_inputs) & set(new_outputs) in ({""}, set()), (
                    f"Node type {node.op_type}-{node.name} is incorrectly replaced "
                    f"{node.input}->{new_inputs} and {node.output}->{new_outputs}\n"
                    f"replacements are\n{pprint.pformat(replacements)}"
                )
                if self.optimization_options.verbose > 2:
                    print(
                        f"[GraphBuilder.remove_identity_nodes] "
                        f"node {node.op_type}-{node.name}:"
                        f"{node.input}->{new_inputs}:{node.output}->{new_outputs}"
                    )
                new_node = oh.make_node(
                    node.op_type,
                    new_inputs,
                    new_outputs,
                    domain=node.domain,
                    name=node.name,
                )
                added += 1
                removed += 1
                new_node.attribute.extend(node.attribute)
                self.nodes.append(new_node)
            else:
                self.nodes.append(node)

        if self.verbose > 1:
            print(
                f"[GraphBuilder.remove_identity_nodes] ends with "
                f"{len(self.nodes)} nodes in "
                f"{time.perf_counter() - begin_} seconds"
            )

        return removed, added

    def insert_and_remove_nodes(
        self,
        insert_at: Optional[int],
        new_nodes: List[NodeProto],
        removed: List[int],
        opsets: Optional[Dict[str, int]] = None,
    ) -> List[NodeProto]:
        """
        Inserts new nodes and removes others.

        :param insert_at: insert the new nodes at this position,
            if empty, the function guesses where to add them
        :param new_nodes: list of nodes to insert
        :param removed: list of nodes to removed (based on their positions)
        :param opsets: opsets used
        :return: list of removed nodes
        """
        assert insert_at is None or not removed or min(removed) <= insert_at, (
            f"The position {insert_at} must be higher than the position "
            f"of the removed nodes {removed}"
        )
        memo = []
        for i in removed:
            assert i < len(
                self.nodes
            ), f"Unable to remove node position {i}, there are {len(self.nodes)}"
            n = self.nodes[i]
            if not n:
                # already marked as removed
                continue
            assert n and not self.do_not_remove(
                n
            ), f"Node {n.name!r} marked as 'DONOTREMOVE' cannot be removed."
            memo.append(n)
            self.nodes[i] = None

        n_existing = []
        for node in new_nodes:
            for i in node.input:
                assert self.has_name(
                    i
                ), f"Input {i!r} does not exist for node {node.op_type!r}"
            for o in node.output:
                if self.has_name(o):
                    # connecting to existing input
                    n_existing.append(o)
                else:
                    self.set_name(o)

            node_domain = node.domain or ""
            if node_domain in self.opsets:
                if opsets and node_domain in opsets:
                    assert compatible_opsets(
                        node_domain,
                        node.op_type,
                        current=self.opsets[node_domain],
                        new_version=opsets[node_domain],
                    ), (
                        f"Incompatible opset for node {node_domain!r} "
                        f"from domain {node_domain!r}, "
                        f"current is {self.opsets[node_domain]}, "
                        f"new is {opsets[node_domain]}"
                    )
            else:
                if opsets and node_domain in opsets:
                    self.opsets[node_domain] = opsets[node_domain]
                else:
                    self.opsets[node_domain] = choose_consistent_domain_opset(
                        node_domain,
                        opsets=self.opsets,
                    )

        assert n_existing, "Any output of the new node is conncted to existing names."
        if insert_at is not None:
            for i, n in enumerate(new_nodes):
                assert isinstance(n, NodeProto), f"Unexpected type {type(n)} for a node"
                self.nodes.insert(insert_at + i, n)
                self._make_node_set_type_shape_constant(n, True)
                self._make_node_set_type_shape(n)
            self.nodes = [n for n in self.nodes if n is not None]
            return memo

        # Needs to insert the nodes at the right location.
        # Let's find out where the best position is.
        self.nodes = [n for n in self.nodes if n is not None]
        needed_at = {}
        first_at = {}
        for i, node in enumerate(self.nodes):
            for name in node.input:
                if name not in needed_at:
                    needed_at[name] = i
            for name in node.output:
                if name not in first_at:
                    first_at[name] = i

        # guess the position to insert the nodes at
        # the order of the new nodes is consistent but it may have to be changed
        # if it does not fit the existing order
        insert_needed_at = {}
        insert_first_at = {}
        N = len(self.nodes)
        inserted_at = []
        new_nodes_p = []
        for init, node in enumerate(new_nodes):
            if node.input:
                min_position = max(first_at.get(i, -1) for i in node.input) + 1
            else:
                # a constant node
                min_position = 0
            max_position = min(needed_at.get(o, N) for o in node.output)

            assert min_position <= max_position, (
                f"Unable to insert node {self.print_node(node)}, "
                f"min_position={min_position}, max_position={max_position}, "
                f"len(nodes)={len(self.nodes)}, previous insertions={inserted_at}, "
                f"insert_needed_at={insert_needed_at}, insert_first_at={insert_first_at}, "
                f"inserted_at={inserted_at}"
            )

            if node.input:
                local_min_position = max(insert_first_at.get(i, -1) for i in node.input)
            else:
                # a constant node
                local_min_position = 0
            local_max_position = min(insert_needed_at.get(o, N) for o in node.output)

            assert local_min_position <= local_max_position, (
                f"Unable to insert node {self.print_node(node)}, "
                f"local_min_position={local_min_position}, local_max_position={local_max_position}, "
                f"len(nodes)={len(self.nodes)}, previous insertions={inserted_at}, "
                f"insert_needed_at={insert_needed_at}, insert_first_at={insert_first_at}"
            )

            insert_position = max(min_position, local_min_position)

            new_nodes_p.append((insert_position, init, node))
            for i in node.input:
                insert_needed_at[i] = min(
                    insert_position, insert_needed_at.get(i, insert_position)
                )
            for i in node.output:
                insert_first_at[i] = min(
                    insert_position, insert_first_at.get(i, insert_position)
                )

        assert len(new_nodes) == len(new_nodes_p)
        new_nodes_p.sort()

        # do the addition
        init_nams = {}
        for p, _, n in reversed(new_nodes_p):
            assert isinstance(n, NodeProto), f"Unexpected type {type(n)} for a node"
            self.nodes.insert(p, n)
        for _, _, n in new_nodes_p:
            if n.output[0] in init_nams:
                continue
            self._make_node_set_type_shape_constant(n, True)
            self._make_node_set_type_shape(n)
        return memo

    @classmethod
    def _clean_shapes(cls, proto: Union[GraphProto, ModelProto]):
        # cleaning unresolved shapes
        if isinstance(proto, ModelProto):
            cls._clean_shapes(proto.graph)
            return
        new_shapes = []
        for sh in proto.value_info:
            if sh.type.tensor_type.elem_type == 0:
                continue
            new_shapes.append(sh)
        del proto.value_info[:]
        proto.value_info.extend(new_shapes)

    def _update_shape_types_with_proto_one_result(self, val):
        itype = val.type.tensor_type.elem_type
        self.set_type(val.name, itype)
        shape = tuple(
            d.dim_param if d.dim_param else d.dim_value
            for d in val.type.tensor_type.shape.dim
        )
        for sh in shape:
            if isinstance(sh, int):
                continue
            if not self.has_dynamic_object(sh):
                self.make_dynamic_object(sh, self.torch.SymInt(sh))
        self.set_shape(val.name, shape, exc=False)

    def _update_shape_types_with_proto(
        self, proto: ModelProto, infer_shapes: bool = False
    ):
        """
        Updates the shapes and types for an existing model.

        :param proto: model proto
        :param infer_shapes: infer shapes to fill information about type and shapes
        """
        if self.verbose > 1:
            begin_ = time.perf_counter()
            print(
                f"[GraphBuilder._update_shape_types_with_proto] starts with "
                f"{len(self.nodes)} nodes and {len(getattr(proto.graph, 'value_info', 0))} "
                f"shapes."
            )
        assert isinstance(proto, ModelProto), f"Unexpected type {type(proto)} for proto"
        if infer_shapes:
            if self.verbose > 1:
                print("[GraphBuilder._update_shape_types_with_proto] infer shapes")
            new_proto = onnx_infer_shapes(proto)
            if self.verbose > 1:
                print(
                    "[GraphBuilder._update_shape_types_with_proto] "
                    f"infer shapes done {time.perf_counter() - begin_} seconds"
                )
            self._clean_shapes(new_proto)
            if self.verbose > 1:
                print(
                    "[GraphBuilder._update_shape_types_with_proto] "
                    f"_clean_shapes after {time.perf_counter() - begin_} seconds"
                )
        else:
            new_proto = proto

        if not hasattr(new_proto.graph, "value_info"):
            if self.verbose > 1:
                print(
                    f"[GraphBuilder._update_shape_types_with_proto] ends in "
                    f"{time.perf_counter() - begin_} seconds."
                )
            return

        if self.verbose > 1:
            begin_ = time.perf_counter()
            print(
                f"[GraphBuilder._update_shape_types_with_proto] "
                f"walk through {len(proto.graph.value_info)} shapes."
            )
        for val in new_proto.graph.value_info:
            self._update_shape_types_with_proto_one_result(val)

        if self.verbose > 1:
            print(
                f"[GraphBuilder._update_shape_types_with_proto] ends in "
                f"{time.perf_counter() - begin_} seconds."
            )

    def _update_structures_with_proto(self, proto: ModelProto, bypass_shape: bool):
        """
        Updates the shapes and types for an existing model.
        """
        if self.verbose > 1:
            begin_ = time.perf_counter()
            print(
                f"[GraphBuilder._update_structures_with_proto] starts with "
                f"{len(proto.graph.node)} nodes"
            )
        self.opsets = {d.domain: d.version for d in proto.opset_import}
        if self.ir_version is None:
            self.ir_version = proto.ir_version
        self.nodes = list(proto.graph.node)
        self.initializers_dict = {i.name: i for i in proto.graph.initializer}
        self.initializers_dict.update(
            {i.name: i for i in proto.graph.sparse_initializer}
        )
        self.functions = list(proto.functions)
        self.value_info = list(proto.graph.value_info)
        self.inputs = list(proto.graph.input)
        self.outputs = list(proto.graph.output)
        self.input_names = [i.name for i in proto.graph.input]

        if hasattr(proto.graph, "value_info"):
            available_shapes = {v.name: v for v in proto.graph.value_info}
        else:
            available_shapes = {}

        for k, v in self.initializers_dict.items():
            self.constants_[k] = None
            self._unique_names.add(k)
            self.set_name(k)
            self.set_shape(k, self._get_tensor_shape(v))
            self.set_type(k, self._get_tensor_type(v))
            key = self.make_key(v)
            if key not in self._values:
                self._values[key] = k

        for i in self.inputs + self.outputs:
            self.set_name(i.name)
            self.set_type(i.name, i.type.tensor_type.elem_type)
            if i.type.tensor_type.shape.dim:
                shape = tuple(
                    d.dim_param if d.dim_param else d.dim_value
                    for d in i.type.tensor_type.shape.dim
                )
                for sh in shape:
                    if isinstance(sh, int):
                        continue
                    if not self.has_dynamic_object(sh):
                        self.make_dynamic_object(sh, self.torch.SymInt(sh))
                self.set_shape(i.name, shape)
            if (
                self.get_type(i.name) == TensorProto.INT64
                and self.has_shape(i.name)
                and self.get_shape(i.name) in ((1,), tuple)
                and "dim" in i.name
            ):
                self.set_value_shape(i, (i.name,))

        need_identity_removal = False
        new_nodes = []
        for node in self.nodes:

            self._unique_names |= set(node.output)
            shape_set = self.simple_update_value_shape_with_node(node)
            if node.name:
                self._unique_node_names.add(node.name)

            if shape_set:
                for o in node.output:
                    if not self.has_name(o):
                        self.set_name(o)
                new_nodes.append(node)
                continue

            if node.op_type == "SequenceConstruct":
                dtypes = [
                    (self.get_type(n) if self.has_type(n) else 0) for n in node.input
                ]
                ranks = [
                    (self.get_rank(n) if self.has_rank(n) else -1) for n in node.input
                ]
                unique_dtypes = set(dtypes)
                if 0 in unique_dtypes:
                    unique_dtypes = set(u for u in unique_dtypes if u != 0)
                    if len(unique_dtypes) == 0:
                        if not self.has_name(node.output[0]):
                            self.set_name(node.output[0])
                        self.set_sequence(
                            node.output[0], 0, shapes=None, ranks=None, unknown=True
                        )
                        new_nodes.append(node)
                        continue

                assert len(unique_dtypes) == 1, (
                    f"A sequence has distinct dtype: {dtypes}, "
                    f"(unique_dtypes={unique_dtypes}), node.name={node.name}, "
                    f"node.input={node.input}"
                )
                if not self.has_name(node.output[0]):
                    self.set_name(node.output[0])
                self.set_sequence(
                    node.output[0], list(unique_dtypes)[0], shapes=None, ranks=ranks
                )
                new_nodes.append(node)
                continue

            if node.op_type == "SequenceAt":
                position = self.get_constant(node.input[1], computed_value=True)
                seq = self.get_sequence(node.input[0])
                dtype = seq["dtype"]
                if not self.has_name(node.output[0]):
                    self.set_name(node.output[0])
                self.set_type(node.output[0], dtype)
                if seq["ranks"] is None:
                    new_nodes.append(node)
                    continue
                rank = seq["ranks"][int(position)]
                self.set_rank(node.output[0], rank)
                new_nodes.append(node)
                continue

            assert node.op_type not in {
                "SplitToSequence",
                "SequenceErase",
                "SequenceInsert",
                "SequenceAt",
            }, (
                f"Sequence operators are not supported yet and op_type={node.op_type!r}"
                f"(name={node.name!r})."
            )

            if node.op_type == "Constant":
                exist = self.is_exact_same_constant(node)
                if exist is not None:
                    node = oh.make_node(
                        "Identity",
                        [exist.output[0]],
                        [node.output[0]],
                        name="._update_structures_with_proto",
                    )
                    replaced = True
                    need_identity_removal = True
                else:
                    self.add_constant_node(node)
                    replaced = False

                self.constants_[node.output[0]] = node
                if not self.has_name(node.output[0]):
                    self.set_name(node.output[0])

                if replaced:
                    self.set_type(node.output[0], self.get_type(node.input[0]))
                    self.set_shape(node.output[0], self.get_shape(node.input[0]))
                else:
                    self.set_shape(node.output[0], self._get_tensor_shape(node))
                    self.set_type(node.output[0], self._get_tensor_type(node))
            elif node.op_type == "ConstantOfShape" and self.is_constant(node.input[0]):
                exist = self.is_exact_same_constant(node)
                if exist is not None:
                    node = oh.make_node(
                        "Identity",
                        [exist.output[0]],
                        [node.output[0]],
                        name="._update_structures_with_proto",
                    )
                    replaced = True
                    need_identity_removal = True
                else:
                    self.add_constant_node(node)
                    replaced = False

                self.constants_[node.output[0]] = node
                if not self.has_name(node.output[0]):
                    self.set_name(node.output[0])

                if replaced:
                    self.set_type(node.output[0], self.get_type(node.input[0]))
                    self.set_shape(node.output[0], self.get_shape(node.input[0]))
                else:
                    value = self.get_constant(node.input[0], computed_value=True)
                    shape = tuple(int(i) for i in value)
                    self.set_shape(node.output[0], shape)
                    if len(node.attribute) == 0:
                        self.set_type(node.output[0], TensorProto.FLOAT)
                    else:
                        value = node.attribute[0].t
                        self.set_type(node.output[0], value.data_type)
            else:
                for o in node.output:
                    if o == "":
                        continue
                    if not self.has_name(o):
                        self.set_name(o)
            new_nodes.append(node)
            for o in node.output:
                if o in available_shapes:
                    self._update_shape_types_with_proto_one_result(available_shapes[o])

            if not bypass_shape:
                if any(
                    map(
                        lambda x: x not in available_shapes and not self.has_type(x),
                        node.output,
                    )
                ):
                    # second try
                    self._make_node_set_type_shape(node)

                # This test should be enabled when shape inference is complete.
                # assert all(
                #     map(
                #         lambda x: x in available_shapes or self.has_type(x), node.output
                #     )
                # ), (
                #     f"One output of node {node.op_type!r} (name={node.name!r}) has no type: "
                #     f"{', '.join(o + ((':' + str(self.get_type(o))) if self.has_type(o) else ':0') for o in node.output)}"
                # )

        self.nodes = new_nodes

        if need_identity_removal:
            self.remove_identity_nodes()

        if self.verbose > 1:
            print(
                f"[GraphBuilder._update_structures_with_proto] ends with "
                f"{len(self.nodes)} nodes in "
                f"{time.perf_counter() - begin_}"
            )

    def parse_dimension_expression(self, expr: str, exc: bool = True) -> Expression:
        """
        Parses an expression involving dimension.

        :param expr: expr
        :param exc: raises an exception if it fails
        :return: an expression or None if exc is False and the parsing failed
        """
        return parse_expression(expr, exc=exc, context=self.dynamic_objects)

    def _constant_key(self, node: NodeProto) -> Optional[bytes]:
        """
        Builds a unique key for a constant.
        Returns None if the constant if too big.
        """
        if node.op_type == "ConstantOfShape":
            # We assume initializer are fused.
            name = node.input[0]
            while name in self.constants_alias_:
                name = self.constants_alias_[name]
            key = [node.op_type.encode(), name.encode()]
            for att in node.attribute:
                key.append(att.SerializeToString())
            return b"|".join(key)
        if node.op_type == "Constant":
            shape = self._get_tensor_shape(node)
            size = np.prod(shape) if shape else 1
            if size > self.optimization_options.constant_size:
                # It would be too long.
                return None
            key = [node.op_type.encode()]
            for att in node.attribute:
                key.append(att.SerializeToString())
            return b"|".join(key)

        raise RuntimeError(
            f"Unexpected node type {node.op_type!r}{self.get_debug_msg()}"
        )

    def add_constant_node(self, node: NodeProto) -> Optional[bytes]:
        """
        Adds a constant node. Any constant equivalent to this one
        will be fused.
        `self.optimization_options.constant_fusing` must be True.
        """
        if not self.optimization_options.constant_fusing:
            return None
        key = self._constant_key(node)
        assert key not in self.constants_node_, (
            f"A constant with the same key {key!r} was already added"
            f"{self.get_debug_msg()}"
        )
        self.constants_node_[key] = node
        return key

    def is_exact_same_constant(self, node: NodeProto) -> Optional[NodeProto]:
        """
        Adds a constant node. Any constant equivalent to this one
        will be fused.
        `self.optimization_options.constant_fusing` must be True.
        """
        if not self.optimization_options.constant_fusing:
            return None
        key = self._constant_key(node)
        if key in self.constants_node_:
            return self.constants_node_[key]
        return None
