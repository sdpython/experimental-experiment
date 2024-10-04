import contextlib
import pprint
import time
import os
import sys
from collections import Counter
from typing import (
    Any,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)
import numpy as np
from onnx import (
    AttributeProto,
    FunctionProto,
    GraphProto,
    ModelProto,
    NodeProto,
    TensorProto,
    TypeProto,
)
import onnx.helper as oh
import onnx.numpy_helper as onh
from onnx.external_data_helper import uses_external_data
from onnx.model_container import make_large_tensor_proto
from onnx.shape_inference import infer_shapes as onnx_infer_shapes
from experimental_experiment.reference import ExtendedReferenceEvaluator
from ._shape_helper import (
    DYNAMIC_SHAPE,
    STATIC_SHAPE,
    all_int,
    all_int_or_str,
    is_static_dimension,
    is_static_shape,
    _reshape_shape,
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
from .model_container import TorchModelContainer, proto_from_array, _get_type
from ._dtype_helper import (
    dtype_to_tensor_dtype,
    onnx_dtype_to_torch_dtype,
    torch_dtype_to_onnx_dtype,
)
from ._helper import make_hash
from .optimization_options import OptimizationOptions
from .expression_dimension import Expression, parse_expression
from .graph_builder_opset import Opset
from ._graph_builder_runtime import _GraphBuilderRuntime


@contextlib.contextmanager
def _unset_fake_temporarily() -> Generator:
    import torch

    old = torch._C._unset_dispatch_mode(torch._C._TorchDispatchModeKey.FAKE)
    try:
        yield old
    finally:
        if old is not None:
            torch._C._set_dispatch_mode(old)


class GraphBuilder(_GraphBuilderRuntime):
    """
    Simplifies the creation of a model.

    :param target_opset_or_existing_proto: a ModelProto, an integer,
        a dictionary of domain, version
    :param input_names: input names
    :param as_function: export as a function or a model
    :param optimization_options: optimizations options,
        see :class:`OptimizationOptions`
    :param args: example of inputs
    :param ir_version: ir version when exporting
    :param verbose: verbosity
    :param infer_shapes: run shape inference, if the value is `'new'`,
        existing shapes are ignored
    :param raise_list: raise an exception if a new operator belongs to that list
    :param dynamic_shapes: dynamic shapes

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
    - `input_args: List[T]`: input tensors when
      the class is used to convert an existing model
    - `functions: Dict[Tuple[str,str], FunctionProto]`:
      dictionary of functions to add to the model
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
    - `_dynamic_examples: Dict[str, Set[Union[int,float]]]`: example of dynamic dimensions
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
    - `constraints_: Dict[str, Set[Any]]`:
        if a broadcast implies a constraints on a dynamic shape,
        it is stored here

    Debugging attributes:

    - `_raise_list: Set[str]`: the builder stop if a result falls in that list
      (debugging tool)

    You can setup environment variable ``ONNXSTOP``, ``ONNXSTOPSHAPE``, ``ONNXSTOPTYPE``
    to raise an exception when the type or shape
    of a variable is set. Example: ``ONNXSTOP=attn_output python ...
    """

    class WrapSym:
        def __init__(self, sym: Union["torch.SymInt", "torch.SymFloat"]):  # noqa: F821
            self.sym = sym

        def __repr__(self) -> str:
            try:
                return f"WrapSym({self.sym!r})"
            except AttributeError:
                return "WrapSym(...)"

    # Size of a tensor kept in the onnx file and not stored as exrternal weight.
    SMALL_TENSOR = 1024

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
        infer_shapes: Union[bool, str] = False,
        raise_list: Optional[Set[str]] = None,
        dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
    ):
        import torch

        self.torch = torch
        self.maybe_disable_fake_tensor_mode = _unset_fake_temporarily
        self.optimization_options = optimization_options or OptimizationOptions(
            verbose=verbose
        )
        self.as_function = as_function
        self.input_args = args
        self.verbose = verbose
        self.ir_version = ir_version
        self._debug_msg = {}
        self.dynamic_dimensions_source = {}
        self.dynamic_shapes = dynamic_shapes
        self.dynamic_objects = {}
        self.dynamic_objects_rev = {}
        self.functions = {}
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
        self._dynamic_examples = {}
        self.constants_ = {}
        self.op = Opset(self)
        self.anyop = Opset(self, allow_unknown=True)
        self._debug_stop = os.environ.get("ONNXSTOP", "#?#")
        self._debug_stop_shape = os.environ.get("ONNXSTOPSHAPE", "#?#")
        self._debug_stop_type = os.environ.get("ONNXSTOPTYPE", "#?#")
        self.time_evaluation_constants_ = 0
        self.statistics_ = {}
        self.constraints_ = {}
        self._registered_users = {}
        self.was_inputs_renamed = input_names is not None and input_names

        assert dynamic_shapes is None or isinstance(dynamic_shapes, dict), (
            f"dynamic_shapes is expected to be empty or a dictionary "
            f"not {type(dynamic_shapes)}, dynamic_shapes={dynamic_shapes}"
        )

        if self.dynamic_shapes:
            self._register_dynamic_object_from_dynamic_shapes()

        if isinstance(target_opset_or_existing_proto, (int, dict)):
            # starts a model from nothing
            assert not infer_shapes, "infer_shapes is used if an existing model is loaded"
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
            self._update_structures_with_proto(target_opset_or_existing_proto, infer_shapes)
            self.constant_folding(convert_into_initializer=False)
            self._update_shape_types_with_proto(target_opset_or_existing_proto, infer_shapes)
        else:
            raise NotImplementedError(
                f"{type(target_opset_or_existing_proto)} is not supported."
            )

    def _register_dynamic_object_from_dynamic_shapes(self):
        assert (
            self.dynamic_shapes is not None
        ), "Call this method if self.dynamic_shapes is not None"
        for input_name, v in self.dynamic_shapes.items():
            if isinstance(v, dict):
                pos_vv = list(v.items())
            elif isinstance(v, tuple):
                pos_vv = [(f"{input_name}_{i}", v[i]) for i in range(len(v))]
            else:
                raise AssertionError(
                    f"Unexpected value for input_name={input_name!r} and "
                    f"v={v}, dynamic_shapes={self.dynamic_shapes}"
                )
            for pos, vv in pos_vv:
                if vv is None:
                    continue
                if isinstance(vv, dict):
                    # example:
                    # args_0 {0: <class '._bash_bench_model_runner.batch'>}
                    for _k, _v in vv.items():
                        if isinstance(_v, self.torch.SymInt):
                            if not self.has_dynamic_object(_v.__name__):
                                self.make_dynamic_object(
                                    _v.__name__,
                                    _v,
                                    axis=_k,
                                    input_name=pos,
                                )
                        elif isinstance(_v, self.torch.export.dynamic_shapes._Dim):
                            if not self.has_dynamic_object(_v.__name__):
                                self.make_dynamic_object(
                                    _v.__name__,
                                    self.torch.SymInt(_v.__name__),
                                    axis=_k,
                                    input_name=pos,
                                )
                        else:
                            raise AssertionError(
                                f"Unexpected type {type(_v)} in {vv} for dynamic "
                                f"dimension {pos!r}, pos_vv={pos_vv!r}, "
                                f"self.dynamic_shapes={self.dynamic_shapes}"
                            )
                elif isinstance(vv, self.torch.SymInt):
                    if not self.has_dynamic_object(vv.__name__):
                        self.make_dynamic_object(
                            vv.__name__, vv, axis=pos, input_name=input_name
                        )
                elif isinstance(vv, self.torch.export.dynamic_shapes._Dim):
                    if not self.has_dynamic_object(vv.__name__):
                        self.make_dynamic_object(
                            vv.__name__,
                            self.torch.SymInt(vv.__name__),
                            axis=pos,
                            input_name=input_name,
                        )
                else:
                    raise AssertionError(
                        f"Unexpected type {type(vv)}, vv={vv} for dynamic "
                        f"dimension {pos!r}, pos_vv={pos_vv!r}, "
                        f"self.dynamic_shapes={self.dynamic_shapes}"
                    )

    def add_stat(self, kind: str, name: str):
        """
        Increments a counter.
        """
        if kind not in self.statistics_:
            self.statistics_[kind] = {}
        stat = self.statistics_[kind]
        if name not in stat:
            stat[name] = 1
        else:
            stat[name] += 1

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
        Returns None if it is not possible.
        """
        if isinstance(value, TensorProto):
            if uses_external_data(value):
                key = None
                for info in value.external_data:
                    if info.key == "location":
                        key = info.value
                        break
                assert key is not None, (
                    f"External tensor {value.name!r} has no location, "
                    f"external_data is {value.external_data}"
                )
                return None
            return self.make_key(onh.to_array(value))
        if isinstance(value, self.torch.Tensor):
            if value.dtype == self.torch.int64 and value.numel() < 8:
                return self.make_key(value.detach().cpu().numpy())
            return None
        if isinstance(value, int):
            return int, value
        if isinstance(value, np.ndarray):
            if value.size < self.SMALL_TENSOR:
                return (value.dtype, value.shape, tuple(value.ravel().tolist()))
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
        raise TypeError(f"Unexpected or unsupported scenario type {type(proto)}: {proto}.")

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
        multiple_outputs: bool = False,
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
        :param multiple_outputs: allow multiple outputs
        :return: value
        """
        if as_shape:
            assert not multiple_outputs, "multiple outputs not allowed with as_shape=True"
            res = self.get_constant(name, exc, computed_value=computed_value, as_shape=False)
            if res is None:
                assert not exc, (
                    f"No constant for name={name!r}, exc={exc}, "
                    f"computed_value={computed_value}, as_shape={as_shape}"
                )
                return None

            assert multiple_outputs or not isinstance(
                res, tuple
            ), f"Multiple output is not allowed but type is {type(res)} for name={name!r}"
            new_res = []
            for i in res:
                if isinstance(i, str):
                    new_res.append(i)
                else:
                    new_res.append(int(i))
            return tuple(new_res)

        if not self.is_constant(name):
            if exc:
                raise ValueError(f"Result {name!r} is not a constant{self.get_debug_msg()}")
            return None
        possible_value = self.constants_[name]
        if name in self.constants_computed_:
            value = self.constants_computed_[name]
            assert value is not None, f"Constant is empty for name={name!r}"
            assert multiple_outputs or not isinstance(
                value, tuple
            ), f"Multiple output is not allowed but type is {type(value)} for name={name!r}"
            return value

        if possible_value is not None:
            assert isinstance(possible_value, (np.ndarray, self.torch.Tensor, NodeProto)), (
                f"Unexpected type {type(possible_value)} for a "
                f"constant{self.get_debug_msg()}"
            )
            if computed_value and isinstance(possible_value, NodeProto):
                res, _ = self.compute_constant(name, exc=exc)
                if res is None:
                    # The constant is too big to be computed.
                    return None

                assert multiple_outputs or not isinstance(res, tuple), (
                    f"Multiple output is not allowed but type is {type(res)} "
                    f"for name={name!r}"
                )
                assert not multiple_outputs, (
                    f"get_constants not implemented when multiple_outputs=True, "
                    f"name={name!r}"
                )
                if not isinstance(res, tuple):
                    return res

                assert isinstance(res, tuple), (
                    f"Expecting multiple outputs for name={name!r}, "
                    f"op_type={possible_value.op_type!r}"
                )
                if len(res) == 1:
                    assert multiple_outputs or not isinstance(value, tuple), (
                        f"Multiple output is not allowed but type is {type(value)} "
                        f"for name={name!r}"
                    )
                    return res[0]

                index = list(possible_value.output).index(name)
                value = res[index]
                assert value is not None, f"Constant is empty for name={name!r}"
                assert multiple_outputs or not isinstance(value, tuple), (
                    f"Multiple output is not allowed but type is {type(value)} "
                    f"for name={name!r}"
                )
                return value

            assert possible_value is not None, f"Constant is empty for name={name!r}"
            assert multiple_outputs or not isinstance(possible_value, tuple), (
                f"Multiple output is not allowed but type is {type(possible_value)} "
                f"for name={name!r}"
            )
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
            assert not multiple_outputs, f"Multiple output is not allowed for name={name!r}"
            return v

        if isinstance(value, TensorProto):
            if uses_external_data(value):
                if exc:
                    raise TypeError(
                        f"Tensor is using external data, data_type={value.data_type}, "
                        f"dims={value.dims}"
                    )
                return None
            v = onh.to_array(value)
            assert not multiple_outputs, f"Multiple output is not allowed for name={name!r}"
            self.constants_computed_[name] = v
            return v

        if isinstance(value, np.float32):
            # This should not be needed.
            return np.array(value, dtype=np.float32)

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
        assert name != "", f"Empty name {name!r} cannot be registered{self.get_debug_msg()}"
        assert name not in self._raise_list, (
            f"Name {name!r} is one of the name declared in "
            f"the stop list{self.get_debug_msg()}"
        )

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
        if name == self._debug_stop or name == self._debug_stop_shape:
            # Set ONNXSTOP or ONNXSTOPSHAPE to stop here.
            raise AssertionError(f"Requested stop, name={name!r}, rank={value}")
        assert isinstance(value, int), f"Unexpected rank type {type(value)} for {name!r}"
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
        n_outputs: Optional[int] = None,
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
                _exa, val = val1
                if val is not None and len(val) == 3:
                    el_type, size = val[1:]
                    if el_type in (
                        self.torch.float32,
                        self.torch.float64,
                        self.torch.float16,
                        self.torch.bfloat16,
                        self.torch.bool,
                    ):
                        return False
                    if len(size) >= 2:
                        return False
                    if el_type in (self.torch.int64, self.torch.int32) and len(size) == 0:
                        # A single integer with no shape, it looks like a dimension.
                        # Let's assume it is. It is more efficient to consider it as
                        # a dimension.
                        return True
                    # In another case, let's assume it is not.
                    return False
                else:
                    if elem_type is not None and elem_type in (
                        self.torch.float32,
                        self.torch.float64,
                        self.torch.float16,
                        self.torch.bfloat16,
                        self.torch.bool,
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
                        TensorProto.BOOL,
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
                        # No dynamic shape as input, so there
                        # shoud not be any dynamic shape as output.
                        return False
            elif value[0] == "call_module":
                if isinstance(value[1], tuple) and len(value[1]) == 2:
                    el_type, size = value[1]
                    if el_type in (
                        self.torch.float32,
                        self.torch.float64,
                        self.torch.float16,
                        self.torch.bfloat16,
                    ):
                        return False
                    if len(size) >= 2:
                        return False
            if n_outputs == 1:
                # We may assume a model would not output just one dimension.
                return False
            raise RuntimeError(
                f"Not implemented for name={name!r}, value={value!r} ({type(value)}), "
                f"elem_type={elem_type}, shape={shape}, n_outputs={n_outputs}"
                f"{self.get_debug_msg()}"
            )
        else:
            if elem_type in {
                TensorProto.FLOAT16,
                TensorProto.FLOAT,
                TensorProto.DOUBLE,
                TensorProto.BFLOAT16,
                TensorProto.BOOL,
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
                    # not a dimension but a result of a computation involving a dimension
                    TensorProto.FLOAT,
                }
            )
            and (shape is None or (isinstance(shape, tuple) and len(shape) == 1))
        ), (
            f"Inconsistent result type for name={name!r}, is_dimension={res}, "
            f"elem_type={elem_type}, shape={shape}{self.get_debug_msg()}"
        )
        return res

    def set_shapes_types(
        self,
        name: Union[str, "torch.fx.Node"],  # noqa: F821
        where: str,
        value: Any,
    ):
        if hasattr(name, "name"):
            name = name.name
        self._known_torch_value[name] = (where, value)

    @classmethod
    def _torch_sym_int_to_str(cls, value: "torch.SymInt") -> Union[int, str]:  #  noqa: F821
        try:
            val_int = int(value)
            return val_int
        except (TypeError, ValueError):
            pass

        if isinstance(value.node, str):
            return f"{value.node}"

        from torch.fx.experimental.sym_node import SymNode

        if isinstance(value.node, SymNode):
            # '_expr' is safer than expr
            return str(value.node._expr)
        raise AssertionError(f"Unable to convert {value!r} into string")

    def _check_two_shapes_are_compatible(
        self,
        old_shape: Tuple[Any, ...],
        shape: Tuple[Any, ...],
        register_int: bool = True,
        name: Optional[str] = None,
    ):
        """
        Raises an exception if two shapes are not compatabible.
        """
        import torch

        assert len(old_shape) == len(shape), (
            f"Rank mismatch between {old_shape} and {shape} (name={name!r}"
            f"{self.get_debug_msg()}"
        )
        for d1, d2 in zip(old_shape, shape):
            if isinstance(d1, int) and isinstance(d2, int):
                assert d1 == d2, (
                    f"Shape {name!r} already exists and one dimension "
                    f"is not compatible existing {old_shape} != {shape} "
                    f"(new) {self.get_debug_msg()}"
                )
                continue

            d1_, d2_ = d1, d2
            if isinstance(d1, torch.SymInt):
                d1 = self._torch_sym_int_to_str(d1)
            if isinstance(d2, torch.SymInt):
                d2 = self._torch_sym_int_to_str(d2)

            if isinstance(d1, (int, str)) and isinstance(d2, (int, str)):
                if d1 == d2:
                    continue
                if isinstance(d1, str) and isinstance(d2, str):
                    self.register_constraint_dimension(d1, d2)
                    self.register_constraint_dimension(d2, d1)
                continue

            raise RuntimeError(
                f"Shape {name!r} already exists "
                f"and it is not compatible "
                f"existing {old_shape} != {shape} (new) "
                f"d1={d1_!r}, d2={d2_!r}, dim types="
                f"({type(d1_)}, {type(d2_)})"
                f"{self.get_debug_msg()}"
            )

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
        if name == self._debug_stop or name == self._debug_stop_shape:
            # Set ONNXSTOP or ONNXSTOPSHAPE to stop here.
            raise AssertionError(f"Requested stop, name={name!r}, shape={shape}")
        assert isinstance(name, str), f"Unexpected type {type(name)} for name."
        assert "torch.Size" not in str(
            shape
        ), f"Unexpected type {type(shape)} for a shape={shape}{self.get_debug_msg()}"
        assert isinstance(shape, tuple), f"Unexpected shape type {type(shape)}"
        shape = self.verify_shape(shape, 0, name=name)
        assert all(not isinstance(t, self.torch.SymInt) for t in shape), (
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
                    if old_shape == (0,) or shape == (0,):
                        if "warnings" not in self._debug_msg:
                            self._debug_msg["warnings"] = []
                        self._debug_msg["warnings"].append(
                            f"Name {name!r} already exists and it is not compatible "
                            f"existing {old_shape} != {shape} (new)"
                        )
                    else:
                        self._check_two_shapes_are_compatible(
                            old_shape, shape, name=name, register_int=False
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
        for s in shape:
            if isinstance(s, str) and s not in self.dynamic_objects:
                self.add_dynamic_object(s, s)
        self._known_shapes[name] = shape
        if set_rank and not self.has_rank(name):
            self.set_rank(name, len(shape))

    def set_type_shape_or_rank(self, name: str, like: str):
        """
        Sets the type and the shape of *name* like *like*.
        """
        if self.has_type(like):
            self.set_type(name, self.get_type(like))
        if self.has_shape(like):
            self.set_shape(name, self.get_shape(like))
        elif self.has_rank(like):
            self.set_rank(name, self.get_rank(like))

    def set_type(self, name: str, dtype: int, exc: bool = True):
        """
        Sets the shape for a result. It is exists, it checks the new shape
        is equal to the existing one.

        :param name: name
        :param dtype: element type (an integer, ONNX)
        :param exc: raises an exception
        """
        if name == self._debug_stop or name == self._debug_stop_type:
            raise AssertionError(f"Requested stop, name={name!r}, dtype={dtype}")
        assert isinstance(name, str), f"Unexpected type {type(name)} for name."
        if isinstance(dtype, int):
            int_type = dtype
        else:
            int_type = _get_type(dtype)
        if name in self._known_types:
            # 0 is undefined
            if self._known_types[name] != 0 and int_type != self._known_types[name]:
                mapping = [
                    (getattr(TensorProto, att), att)
                    for att in dir(TensorProto)
                    if att.upper() == att and isinstance(getattr(TensorProto, att), int)
                ]
                mapping.sort()
                smap = ",".join(f"{k}:{v}" for k, v in mapping)
                if exc:
                    raise RuntimeError(
                        f"Type for name {name!r} already exists and it is different, "
                        f"known is {self._known_types[name]} != {int_type} (new) - "
                        f"(mapping={smap}){self.get_debug_msg()}"
                    )
                if "warnings" not in self._debug_msg:
                    self._debug_msg["warnings"] = []
                self._debug_msg["warnings"].append(
                    f"Type for name {name!r} already exists and it is different, "
                    f"known is {self._known_types[name]} != {int_type} (new) - "
                )
                if self.verbose > 5:
                    print(
                        f"Type for name {name!r} already exists and it is different, "
                        f"known is {self._known_types[name]} != {int_type} (new) - "
                    )
                return

        if self.verbose > 5:
            print(f"[GraphBuilder-{self._hash()}.set_type] {name}:{int_type}")
        self._known_types[name] = int_type

    def rank(self, name: str) -> int:
        """Shortcut to :meth:`get_rank`."""
        assert isinstance(name, str), f"Unexpected type {type(name)} for name."
        return self.get_rank(name)

    def has_name(self, name: str) -> bool:
        """Tells if a result exists."""
        assert isinstance(name, str), (
            f"Unexpected type {type(name)} for name "
            f"(name={name!r}){self.get_debug_msg()}"
        )
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
            f"rank is unknown for result {name!r}, has_shape={self.has_shape(name)}, "
            f"has_rank={self.has_rank(name)}, "
            f"known_ranks={self._known_ranks}{self.get_debug_msg()}"
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

    def get_type_known(self, name: str) -> Optional[int]:
        """Returns the type known by torch to help solve mismatches."""
        if name in self._known_torch_value:
            value = self._known_torch_value[name]
            # something like (
            #                   'run_node',
            #                   (
            #                       '',
            #                       ('val', torch.float16, torch.Size([2, 12, 2048, 2048]))
            #                   )
            #                )
            assert (
                isinstance(value, tuple)
                and len(value) == 2
                and isinstance(value[1], tuple)
                and len(value[1][1]) == 3
            ), f"Unexpected value {value} for {name!r}{self.get_debug_msg()}"
            dtype = value[1][1][1]
            itype = torch_dtype_to_onnx_dtype(dtype)
            return itype
        return None

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

    def set_value_shape(
        self, name: str, value: Any, equal_to: Optional[Tuple[str, str]] = None
    ):
        """
        Sets the value for a shape result.

        :param name: name
        :param value: it cannot be empty
        :param equal_to: if specified, the value is also
            equal to this value
        """
        assert isinstance(
            name, str
        ), f"Unexpected type {type(name)} for name={name!r}{self.get_debug_msg()}"
        assert value not in {
            tuple()
        }, f"Unexpected value for shape {name!r}, value={value}{self.get_debug_msg()}"
        if equal_to is None:
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
                print(f"[GraphBuilder-{self._hash()}.set_value_shape] {name}[{value}]")
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

    def elem_size(self, elem_type: int) -> int:
        "Returns the size in byte of the an element of this size."
        if elem_type in {TensorProto.FLOAT, TensorProto.INT32, TensorProto.UINT32}:
            return 4
        if elem_type in {TensorProto.DOUBLE, TensorProto.INT64, TensorProto.UINT64}:
            return 8
        if elem_type in {
            TensorProto.INT16,
            TensorProto.UINT16,
            TensorProto.FLOAT16,
            TensorProto.BFLOAT16,
        }:
            return 2
        if elem_type in {TensorProto.BOOL, TensorProto.UINT8, TensorProto.INT8}:
            return 1
        raise AssertionError(f"elem_size not implemented for elem_type={elem_type}.")

    def make_dynamic_object(
        self,
        name: str,
        value: Any,
        shape_as_input: bool = False,
        input_name: Optional[str] = None,
        axis: Optional[int] = None,
    ) -> str:
        """
        Creates a dynamic shapes.

        :param name: name
        :param value: value
        :param shape_as_input: adds the name to the list of the inputs
            of the onnx model
        :param input_name: the dimension comes from this input
        :param axis: the dimension comes this axis
        :return: the name
        """
        assert name not in self.dynamic_objects, (
            f"Dynamic object {name!r}, value={value!r} "
            f"is already there{self.get_debug_msg()}"
        )
        if isinstance(value, self.WrapSym):
            value = value.sym
        assert isinstance(
            value, (self.torch.SymInt, self.torch.SymFloat)
        ), f"Unexpected type {type(value)} for value{self.get_debug_msg()}"

        if input_name is not None and isinstance(value, self.torch.SymInt):
            assert axis is not None, (
                f"input_name={input_name!r} but axis is None for "
                f"dynamic shape {name!r}, value is {value!r}{self.get_debug_msg}"
            )
            assert name != input_name, (
                f"Name {name!r} cannot be defined from itself (axis={axis}), "
                f"value={value}{self.get_debug_msg()}"
            )
            source = dict(input_name=input_name, axis=axis)
            if name in self.dynamic_dimensions_source:
                self.dynamic_dimensions_source[name].append(source)
            else:
                self.dynamic_dimensions_source[name] = [source]

        self.add_dynamic_object(name, value)
        if (
            shape_as_input
            and isinstance(value, self.torch.SymInt)
            and value.node.maybe_as_int() is None
        ):
            # Then an input is a shape.
            self.add_dynamic_object(str(value), value)
        if name not in self._known_value_shape:
            self._known_value_shape[name] = name

        key = None
        if isinstance(value, (self.torch.SymInt, self.torch.SymFloat)):
            if hasattr(value, "node"):
                from torch.fx.experimental.sym_node import SymNode

                if isinstance(value.node, str):
                    key = f"{value.node}"
                elif isinstance(value.node, SymNode):
                    key = f"{value.node}"
                    key2 = value.node.maybe_as_int()
                    if key2 is not None:
                        self._add_dynamic_example(key, key2)
                else:
                    raise AssertionError(
                        f"Unexpected type {type(value.node)} for value.node={value.node}"
                    )
                    # key = str(value)
        if key is None:
            key = str(value)

        self.add_dynamic_objects_rev(key, (name, value))

        if shape_as_input:
            assert isinstance(value, (self.torch.SymInt, self.torch.SymFloat)), (
                f"shape_as_input={shape_as_input}, unexpected type "
                f"{type(value)} for value{self.get_debug_msg()}"
            )
            # torch.compile adds input for dynamic shapes
            return self.make_tensor_input(
                self._known_value_shape[name],
                (
                    TensorProto.INT64
                    if isinstance(value, self.torch.SymInt)
                    else TensorProto.FLOAT
                ),
                (1,),
                is_dimension=True,
            )
        return self._known_value_shape[name]

    def get_dimension_as_result(self, name: str) -> str:
        if self.has_name(name):
            return name
        assert (
            name in self.dynamic_dimensions_source
            and len(self.dynamic_dimensions_source[name]) > 0
        ), (
            f"Dimension {name!r} has no registered source "
            f"it cannot be created as a result{self.get_debug_msg()}"
        )
        source = self.dynamic_dimensions_source[name][0]
        axis = source["axis"]
        input_name = source["input_name"]
        shape_name = self.unique_name(f"_onx_shape_{name}")
        self.make_node("Shape", [input_name], [shape_name], name="_get_dimension_as_result")
        axis_name = self.make_initializer("", np.array([axis], dtype=np.int64))
        self.make_node(
            "Gather", [shape_name, axis_name], [name], name="_get_dimension_as_result"
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
                    f"Unexpected type {type(d)} for a dimension in "
                    f"{shape}{self.get_debug_msg()}"
                )

        assert all_int_or_str(key), (
            f"Unexpected key {key} type are {[type(_) for _ in key]}, "
            f"shape={shape}{self.get_debug_msg()}"
        )
        key = ("Concat", *key)
        if key in self._cache_shape:
            # The same shape was already requested.
            return self._cache_shape[key]

        conc = []
        for d in shape:
            if isinstance(d, int):
                conc.append(self.make_initializer("", np.array([d], dtype=np.int64)))
            elif isinstance(d, str):
                value = d
                if value in self.dynamic_objects_rev:
                    assert len(self.dynamic_objects_rev[value]) >= 1
                    name = self.dynamic_objects_rev[value][0][0]
                    assert not isinstance(name, tuple), (
                        f"Unexpected type {type(name)}, name={name!r}, value={value!r}"
                        f"{self.get_debug_msg()}"
                    )
                    name = self.get_dimension_as_result(name)
                else:
                    name = value
                assert name in self.dynamic_objects or self.has_name(
                    name
                ), f"Unknown dynamic object {d!r}  (or {name!r}){self.get_debug_msg()}"
                if self.has_rank(name):
                    assert self.get_rank(name) <= 1, (
                        f"Unexpected rank={self.get_rank(name)} "
                        "for a shape{self.get_debug_msg()}"
                    )
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

            elif isinstance(d, self.torch.SymInt):
                value = self._torch_sym_int(d)
                if value in self.dynamic_objects_rev:
                    assert len(self.dynamic_objects_rev[value]) >= 1
                    name = self.dynamic_objects_rev[value][0][0]
                    assert not isinstance(name, tuple), (
                        f"Unexpected type {type(name)}, name={name!r}, value={value!r}"
                        f"{self.get_debug_msg()}"
                    )
                    name = self.get_dimension_as_result(name)
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
                    assert self.get_rank(name) <= 1, (
                        f"Unexpected rank={self.get_rank(name)} "
                        "for a shape{self.get_debug_msg()}"
                    )
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
                    f"Unexpected type {type(d)} for a dimension in "
                    f"{shape}{self.get_debug_msg()}"
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
            assert "FakeTensor" not in str(type(value)), (
                f"FakeTensor {name!r} cannot be an initializer {type(value)}"
                f"{self.get_debug_msg()}"
            )
        elif isinstance(value, TensorProto):
            assert "FakeTensor" not in str(type(value)), (
                f"FakeTensor {name!r} cannot be an initializer {type(value)}"
                f"{self.get_debug_msg()}"
            )
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
                new_name = self._values[key]
                assert new_name in self.initializers_dict, f"Unable to find {new_name!r}"
                return new_name
            return self.make_node(
                "Identity",
                [self._values[key]],
                [name],
                name="make_initializer",
                insert_position="HEAD",
            )

        if isinstance(value, TensorProto):
            itype = value.data_type
            shape = tuple(value.dims)
        else:
            itype = _get_type(value.dtype)
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
        self.update_node_constant(name, None)
        if self.verbose and (self.verbose > 1 or np.prod(value.shape) > 100):
            print(f"[GraphBuilder-{self._hash()}.make_initializer] {name}[{itype}:{shape}]")
        if key:
            self._values[key] = name
        return name

    def is_dynamic_shape(
        self, shape: DYNAMIC_SHAPE, verify: bool = True, allow_none: bool = False
    ) -> bool:
        return all(
            self.is_dynamic_dimension(x, verify=verify, allow_none=allow_none) for x in shape
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
                self.set_value_shape(
                    node.output[0], value, equal_to=(node.input[0], node.output[0])
                )
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
                    f"Shape mismatch, end={getattr(end, 'i', end)}, "
                    f"shape of {node.input[0]!r} "
                    f"is {shape}{self.get_debug_msg()}"
                )
                self.set_value_shape(node.output[0], shape[start.i : getattr(end, "i", end)])
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
        if any(x is None for x in values):
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

    def add_dynamic_object(self, key, value):
        if isinstance(value, (self.torch.SymInt, self.torch.SymFloat)):
            self.dynamic_objects[key] = self.WrapSym(value)
        else:
            self.dynamic_objects[key] = value

    def has_dynamic_object(self, name: str) -> bool:
        """Tells if a result is a dynamic object, `torch.SymInt` for torch."""
        return name in self.dynamic_objects

    def add_dynamic_objects_rev(self, key, name_value):
        if key not in self.dynamic_objects_rev:
            self.dynamic_objects_rev[key] = []
        self.dynamic_objects_rev[key].append(name_value)

    def _add_dynamic_example(self, name: str, value: Union[int, float]):
        if name not in self._dynamic_examples:
            self._dynamic_examples[name] = set()
        self._dynamic_examples[name].add(value)

    def _torch_sym_int(self, d, add: bool = False) -> Optional[int]:
        assert isinstance(
            d, (self.torch.SymInt, str, self.torch.SymFloat)
        ), f"unexpected type for d={d}, type={type(d)}"
        value = None
        try:
            # don't use 'expr'
            dyn_val = str(d.node._expr)
            value = dyn_val
        except AttributeError:
            pass

        if isinstance(d, (str, int, float)):
            return d

        if value is None:
            # Is it an integer?
            value = d.node.maybe_as_int()
        else:
            # maybe an expression which is a single integer
            try:
                return int(value)
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
                val_int = d.node.maybe_as_int()
                if val_int is not None:
                    self._add_dynamic_example(value, val_int)
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
            self.add_dynamic_object(value, value)
            self.add_dynamic_object_rev(value, value)

        assert value in self.dynamic_objects_rev or value in self._known_value_shape, (
            f"value={value!r}, unable to find dimension {d!r} ({type(d)}) "
            f"(str(d)={str(d)!r}) in {self.dynamic_objects_rev} "
            f"or {self._dynamic_alias} or {self._known_value_shape}"
            f"{dir(d)}{self.get_debug_msg()}"
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
            assert len(new_value) >= 1, (
                f"Unexpected number of items in {new_value}, value={value!r}, d={d}"
                f"{self.get_debug_msg()}"
            )
            # We assume if len(new_value) > 1 that all names are equivalent.
            # The graph is doing the same computation multiple times.
            final = new_value[0]
            assert isinstance(final, str) or (
                isinstance(final, tuple) and len(final) == 2
            ), (
                f"Unexpected type {type(final)}, final={final}, value={value}, d={d}"
                f"new_value={new_value}, {self.get_debug_msg()}"
            )
            if isinstance(final, str):
                # A formula
                return final

            # An alias
            name = final[0]
            assert isinstance(name, str), (
                f"Unexpected type {type(name)}, name={final}, value={value}, d={d}, "
                f"new_value={new_value}, {self.get_debug_msg()}"
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

    def register_users(self, name: str, users: Iterable[str]):
        """
        Registers users. This is not used except to check the conversion
        is valid.
        """
        assert (
            name not in self._registered_users
        ), f"{name!r} is already registered{self.get_debug_msg()}"
        self._registered_users[name] = set(users)

    def make_tensor_input(
        self, name: str, elem_type: Any, shape: DYNAMIC_SHAPE, is_dimension: bool
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
            if input_name != name:
                self.make_node(
                    "Identity",
                    [input_name],
                    [name],
                    check=False,
                    name="make_tensor_input",
                )
        else:
            if is_dimension:
                # The convention is to have _dim_ in the name to tell
                # it is a dimension.
                input_name = f"{name}_dim_"
                if input_name != name:
                    self.make_node(
                        "Identity",
                        [input_name],
                        [name],
                        check=False,
                        name="make_tensor_input",
                    )
                assert self.has_name(
                    name
                ), f"Missing name={name!r}, is_dimension={is_dimension}"
                self.set_name(input_name)
            else:
                self.input_names.append(name)
                input_name = name
                self.set_name(name)

        assert (is_dimension and "_dim_" in input_name) or (
            not is_dimension and "_dim_" not in input_name
        ), (
            f"Inconsistence for input {name!r}, input_name={input_name!r}, "
            f"elem_type={elem_type}, shape={shape!r}, is_dimension={is_dimension}, "
            f"self.current_input={self.current_input}, "
            f"len(self.input_names)={len(self.input_names)}"
        )

        self.current_input += 1
        elem_type = _get_type(elem_type)
        dyn_shape = self.verify_dynamic_shape(shape, name=input_name, add=True)

        if shape is not None:
            tuple_shape = tuple(shape)
            assert len(tuple_shape) == len(
                dyn_shape
            ), f"mismatch between shape={shape}, dynamic_shape={dyn_shape}"
            for a, b in zip(tuple_shape, dyn_shape):
                if isinstance(a, int) and isinstance(b, int):
                    assert a == b, (
                        f"Unexpected shape mismatch shape={shape}, "
                        f"dyn_shape={dyn_shape}{self.get_debug_msg()}"
                    )
                    continue
                if isinstance(a, str) and isinstance(b, str):
                    assert a == b, (
                        f"Unexpected shape mismatch shape={shape}, "
                        f"dyn_shape={dyn_shape}{self.get_debug_msg()}"
                    )
                    sb = b
                    if sb not in self.dynamic_objects:
                        self.add_dynamic_object(sb, sb)
                    continue
                if isinstance(a, self.torch.SymInt) and isinstance(b, str):
                    sb = b
                    if sb not in self.dynamic_objects:
                        self.add_dynamic_object(sb, sb)
                    i = a.node.maybe_as_int()
                    if i is None:
                        sa = str(a)
                        if sa not in self.dynamic_objects:
                            self.add_dynamic_object(sa, sa)
                    continue

                raise AssertionError(
                    f"Not implemented for type(a)={type(a)}, "
                    f"type(b)={type(b)}, a={a!r}, b={b!r}{self.get_debug_msg()}"
                )

        self.inputs.append(oh.make_tensor_value_info(input_name, elem_type, dyn_shape))
        if self.verbose:
            print(
                f"[GraphBuilder-{self._hash()}.make_tensor_input] "
                f"{input_name}[{elem_type}:{dyn_shape}]"
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
        is_dimension: Optional[bool] = None,
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
            assert not is_dimension, f"name={name!r} not compatible with is_dimension=True"
            res = []
            for n in name:
                res.append(self.make_tensor_output(n, elem_type, shape))
                assert self.has_name(
                    n
                ), f"Output {n!r} among {name} not found{self.get_debug_msg()}"
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

        elem_type = _get_type(elem_type, False)
        if not self.as_function and elem_type == 0:
            raise RuntimeError(f"Undefined element type for {name!r}.")
        dyn_shape = self.verify_shape(shape, name=name, elem_type=elem_type)
        self.outputs.append(oh.make_tensor_value_info(name, elem_type, dyn_shape))
        assert self.has_name(name), f"Output {name!r} not found{self.get_debug_msg()}"
        if self.verbose:
            print(
                f"[GraphBuilder-{self._hash()}.make_tensor_output] "
                f"{name}[{elem_type}:{dyn_shape}]"
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
        assert is_static_shape(shape) or self.is_dynamic_shape(shape, allow_none=True), (
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
            or (attributes is not None and "to" in set(att.name for att in attributes))
        ), (
            f"Operator Cast needs arguments to but kwargs={kwargs}, name={name!r}"
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
        insert_position: Optional[int] = None,
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
        :param insert_position: insert the node at the end (None) or
            at the top (HEAD).
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
            output_names = [self.unique_name(f"_onx_{lower}{i}") for i in range(outputs)]
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
                    f"Input {i!r} does not exist for operator {op_type!r}, "
                    f"inputs={inputs}, outputs={outputs}, name={name!r} "
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
                [type(o) for o in outputs] if isinstance(outputs, (tuple, list)) else outputs
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
                        f"[GraphBuilder-{self._hash()}.make_node] "
                        f"duplicated constant detected for "
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
        if insert_position == "HEAD":
            self.nodes.insert(0, node)
        else:
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
                        [inputs[1]] if isinstance(inputs[1], int) else inputs[1].tolist()
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

        if all(self.is_constant(i) for i in node.input):
            for o in node.output:
                self.update_node_constant(o, node)

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
            self.update_node_constant(k, node)
            node.doc_string += ":constant-3:"
            shape = self._get_tensor_shape(node)
            dtype = self._get_tensor_type(node)
            self.set_shape(k, shape)
            self.set_type(k, dtype)
            if self.verbose and (self.verbose > 3 or np.prod(shape) > 100):
                print(f"[GraphBuilder-{self._hash()}.make_node] {k}[{dtype}:{shape}]")
        elif node.op_type == "ConstantOfShape":
            if len(node.attribute) == 1 and node.attribute[0].name == "value":
                itype = node.attribute[0].t.data_type
            else:
                itype = TensorProto.FLOAT
            self.set_type(node.output[0], itype)
            if self.is_constant(node.input[0]):
                value = self.get_constant(node.input[0], computed_value=True, as_shape=True)
                if value is not None:
                    self.set_shape(node.output[0], value)
                    node.doc_string += ":constant-9:"
        elif node.op_type == "Identity":
            if self.has_shape(node.input[0]):
                self.set_shape(node.output[0], self.get_shape(node.input[0]))
            elif self.has_rank(node.input[0]):
                self.set_rank(node.output[0], self.get_rank(node.input[0]))
            if self.has_type(node.input[0]):
                self.set_type(node.output[0], self.get_type(node.input[0]))
            if self.is_constant(node.input[0]):
                self.update_node_constant(node.output[0], node)
                node.doc_string += ":constant-4:"
        elif node.op_type == "Expand":
            if self.has_type(node.input[0]):
                self.set_type(node.output[0], self.get_type(node.input[0]))
            if self.is_constant(node.input[1]):
                cst, _ = self.compute_constant(node.input[1], exc=False, only_array=True)
                if cst is not None:
                    self.set_shape(node.output[0], tuple(int(i) for i in cst))
        elif node.op_type == "Reshape":
            self.set_type(node.output[0], self.get_type(node.input[0]))
            if self.is_constant(node.input[1]):
                cst, _ = self.compute_constant(node.input[1], exc=False, only_array=True)
                if cst is not None:
                    shape_cst = tuple(int(i) for i in cst)
                    if -1 in shape_cst:
                        if self.has_shape(node.input[0]):
                            sh = self.get_shape(node.input[0])
                            if is_static_shape(sh):
                                self.set_shape(node.output[0], _reshape_shape(sh, shape_cst))
                                node.doc_string += ":constant-7:"
                    else:
                        self.set_shape(node.output[0], shape_cst)
                        node.doc_string += ":constant-7:"
        elif node.op_type == "Shape":
            self.set_type(node.output[0], TensorProto.INT64)
            if self.has_shape(node.input[0]) and len(node.attribute) == 0:
                shape = self.get_shape(node.input[0])
                self.set_shape(node.output[0], (len(shape),))
            else:
                self.set_rank(node.output[0], 1)
            if self.is_constant(node.input[0]):
                self.update_node_constant(node.output[0], node)
                node.doc_string += ":constant-2:"
        elif node.op_type == "Size":
            self.set_type(node.output[0], TensorProto.INT64)
            self.set_shape(node.output[0], (1,))
            if self.is_constant(node.input[0]):
                self.update_node_constant(node.output[0], node)
                node.doc_string += ":constant-2s:"
        elif not sts:
            if node.op_type == "GatherElements":
                if self.has_type(node.input[0]):
                    self.set_type(node.output[0], self.get_type(node.input[0]))
                if self.has_shape(node.input[1]):
                    self.set_shape(node.output[0], self.get_shape(node.input[1]))
                elif self.has_rank(node.input[1]):
                    self.set_rank(node.output[0], self.get_rank(node.input[1]))

    def update_node_constant(self, name: str, node: NodeProto):
        assert isinstance(name, str), f"Unexpected type {type(name)} for name"
        assert node is None or isinstance(
            node, NodeProto
        ), f"Unexpected type {type(node)} for name={name!r}"
        if self.verbose and self.verbose > 2:
            print(
                f"[GraphBuilder.update_node_constant] new constant "
                f"{name!r}, node={None if node is None else node.op_type}"
            )
        self.constants_[name] = node

    def get_attribute(
        self, node: NodeProto, att_name: str, exc: bool = True
    ) -> Optional[AttributeProto]:
        """
        Returns an attribute for a node.
        """
        for att in node.attribute:
            if att.name == att_name:
                return att
        assert not exc, (
            f"Unable to find attribute {att_name!r} for node "
            f"type {node.op_type!r} in node {node}"
        )
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
            else:
                assert "FakeTensor" not in str(type(value)), (
                    f"FakeTensor {name!r} cannot be an initializer {type(value)}"
                    f"{self.get_debug_msg()}"
                )
            self.initializers_dict[name] = value

            self.update_node_constant(name, None)
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
            f"outputs={builder.outputs}, renaming={renaming}{self.get_debug_msg()}"
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

    def _build_large_initializers(self, external_threshold: int):
        new_inits = {}
        large_inits = {}
        for k, v in self.initializers_dict.items():
            itype = self.get_type(k)
            shape = self.get_shape(k)
            size = np.prod(shape) * self.elem_size(itype)
            if size < external_threshold:
                new_inits[k] = v
            else:
                location = f"#{k}"

                nt = make_large_tensor_proto(location, k, itype, shape)
                new_inits[k] = nt
                large_inits[location] = v
        return new_inits, large_inits

    def _build_initializers(
        self, large_model: bool, switch_low_high: bool, external_threshold: int
    ) -> Tuple[List[TensorProto], Dict[str, TensorProto]]:
        if self.verbose:
            begin = time.perf_counter()
            print(
                f"[GraphBuilder-{self._hash()}._build_initializers] "
                f"start with {len(self.initializers_dict)} initializers, "
                f"large_model={large_model}, external_threshold={external_threshold}"
            )

        init_dict, large_inits = (
            self._build_large_initializers(external_threshold)
            if large_model
            else (self.initializers_dict, {})
        )

        if switch_low_high:
            # Let's try to minimize the time.
            if self.verbose:
                print(
                    f"[GraphBuilder-{self._hash()}._build_initializers] "
                    f"switch low/high order"
                )
            initializer = []
            for k, v in init_dict.items():
                if isinstance(v, TensorProto):
                    if self.verbose > 1:
                        print(
                            f"[GraphBuilder-{self._hash()}._build_initializers] "
                            f"TensorProto-{k}:{v.data_type}[{tuple(v.dims)}]"
                        )
                    initializer.append(v)
                    continue

                if self.verbose > 1:
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
                        initializer.append(t)
                        continue

                    from_np = True
                elif isinstance(v, np.float32):
                    # This should not happen.
                    t = onh.from_array(np.array([v], dtype=np.float32), name=k)
                    initializer.append(t)
                    continue
                else:
                    assert isinstance(
                        v, self.torch.Tensor
                    ), f"tensor {k!r} has un unexpected type {type(v)}"
                    assert "FakeTensor" not in str(
                        type(v)
                    ), f"tensor {k!r} cannot be a FakeTensor: {type(v)}"
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
                    with self.maybe_disable_fake_tensor_mode():
                        tensor = proto_from_array(v, name=k, verbose=self.verbose)

                initializer.append(tensor)

            if self.verbose:
                begin = time.perf_counter()
                print(
                    f"[GraphBuilder-{self._hash()}._build_initializers] "
                    f"done in {time.perf_counter() - begin}s "
                    f"with {len(initializer)} initializers, "
                    f"{len(large_inits)} large initializers"
                )
            return initializer, large_inits

        assert (
            not self.large_model
        ), "_build_initializers not implemented when large_model is True"
        large_inits = {}
        res = []
        for k, v in sorted(init_dict.items()):
            if isinstance(v, self.torch.Tensor):
                # no string tensor
                t = self.from_array(v, name=k)
                res.append(t)
                continue
            if isinstance(v, np.ndarray):
                if self.verbose > 1 and np.prod(v.shape) > 100:
                    print(
                        f"[GraphBuilder-{self._hash()}._build_initializers]"
                        f"onh.from_array:{k}:{v.dtype}[{v.shape}]"
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
        if self.verbose:
            begin = time.perf_counter()
            print(
                f"[GraphBuilder-{self._hash()}._build_initializers] "
                f"done in {time.perf_counter() - begin}s "
                f"with {len(res)} initializers, "
                f"{len(large_inits)} large initializers"
            )
        return res, large_inits

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
                return t.detach().cpu().numpy().ravel().tolist()
            if hasattr(t, "size"):
                return t.ravel().tolist()
            if hasattr(t, "dims"):
                a = onh.to_array(t)
                return a.ravel().tolist()
            raise RuntimeError(f"Values unknown for type {type(t)}-{t}.")

        rows = ["", "--DEBUG--"]
        rows.append("--LOCAL FUNCTIONS--")
        for k, v in self.functions.items():
            rows.append(f"{k[0]},{k[1]}({v.input}) -> {v.output}")
        if self.constraints_:
            rows.append("--CONSTRAINTS--")
            for a, b in sorted(self.constraints_.items()):
                rows.append(f"{a} = {b}")
        rows.append("--SHAPE--")
        rows.append("dynamic_examples=")
        for i, (k, v) in enumerate(sorted(self._dynamic_examples.items())):
            try:
                rows.append(f"   {k} = {v!r}")
            except AttributeError:
                rows.append(f"   {k} = ERR: {type(v)!r}:{getattr(v, 'node', 'node=?')!r}")
            if i >= 10000:
                break
        rows.append("dynamic_objects=")
        for i, (k, v) in enumerate(sorted(self.dynamic_objects.items())):
            try:
                rows.append(f"   {k} = {v!r}")
            except AttributeError:
                rows.append(f"   {k} = ERR: {type(v)!r}:{getattr(v, 'node', 'node=?')!r}")
            if i >= 10000:
                break

        rows.append("dynamic_objects_rev=")
        for i, (k, v) in enumerate(sorted(self.dynamic_objects_rev.items())):
            if isinstance(v, (list, tuple)):
                rows.append(f"   {k!r} = {type(v)}")
                for vv in v:
                    if isinstance(vv, tuple):
                        rows.append("     tuple")
                        for vvv in vv:
                            try:
                                rows.append(f"       {vvv!r}")
                            except AttributeError:
                                rows.append(
                                    f"       ERR**: {type(vvv)!r}:"
                                    f"{getattr(vvv, 'node', 'node=?')!r}"
                                )
                    else:
                        try:
                            rows.append(f"       {vv!r}")
                        except AttributeError:
                            rows.append(
                                f"       ERR*: {type(vv)!r}:"
                                f"{getattr(vv, 'node', 'node=?')!r}"
                            )
            else:
                try:
                    rows.append(f"   {k} = {v!r}")
                except AttributeError:
                    rows.append(
                        f"   {k} = ERR-: {type(v)!r}:{getattr(v, 'node', 'node=?')!r}"
                    )
            if i >= 10000:
                break

        rows.append(
            f"dynamic_dimensions_source="
            f"{pprint.pformat(self.dynamic_dimensions_source)[:10000]}"
        )
        rows.append(f"dynamic_alias={pprint.pformat(self._dynamic_alias)[:10000]}")
        rows.append(f"dynamic_shapes={pprint.pformat(self.dynamic_shapes)[:10000]}")
        rows.append(f"_known_value_shape={pprint.pformat(self._known_value_shape)[:10000]}")
        rows.append(f"_known_types={pprint.pformat(self._known_types)[:10000]}")
        rows.append(f"_known_shapes={pprint.pformat(self._known_shapes)[:10000]}")
        rows.append(
            f"_known_constants={pprint.pformat(list(sorted(self.constants_))[:10000])}"
        )
        reminaing_ranks = {
            k: v for k, v in self._known_ranks.items() if k not in self._known_shapes
        }
        rows.append(f"_known_ranks={pprint.pformat(reminaing_ranks )[:10000]}")

        rows.append("--TORCH-USERS--")
        for k, v in sorted(self._registered_users.items()):
            rows.append(f"{k} -> {v}")

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
                    f"Stopped with {len(self.initializers_dict)} "
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
            f"[GraphBuilder-{hs}] Message completed, there are "
            f"{len(self.initializers_dict)} initializers, "
            f"{len(self.nodes)} nodes, {len(self.inputs)} inputs, "
            f"{len(self.inputs)} outputs."
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
                f"node {i}/{len(graph_module.graph.nodes)} target={node.target}"
            )
            interpreter.run_node(node)

    def to_onnx(
        self,
        as_function: bool = False,
        optimize: bool = True,
        large_model: bool = False,
        external_threshold: int = 1024,
        return_optimize_report: bool = False,
        function_name: str = "",
        function_domain: str = "",
        inline: bool = False,
    ) -> Union[FunctionProto, ModelProto, TorchModelContainer]:
        """
        Conversion to onnx. Only then the initializer are converted into
        TensorProto.

        :param as_function: converts the graph as a FunctionProto or a ModelProto
        :param optimize: disable or enable the optimization,
            the optimization are set when the class constructor is called
        :param large_model: if True returns a :class:`onnx.model_container.ModelContainer`,
            it lets the user to decide later if the weights should be part of the model
            or saved as external weights
        :param external_threshold: if large_model is True, every tensor above this limit
            is stored as external
        :param return_optimize_report: return statistics about the optimization as well
        :param function_name: only used if as_function is True
        :param function_domain: only used if as_function is True
        :param inline: inline local functions, this is done before
            any optimization takes place
        :return: the proto
        """
        if len(self.nodes) == 0:
            raise RuntimeError(f"The onnx model is empty (no node).\n{self.get_debug_msg()}")

        if inline:
            stats = self.inline_functions(verbose=self.verbose)
        else:
            stats = None

        if optimize:
            statso = self.optimize()
            if stats:
                stats.extend(statso)
            else:
                stats = statso

        assert len(self.nodes) > 0, (
            f"The onnx model is empty after optimization (no node)."
            f"\n{self.get_debug_msg()}"
        )

        opsets = [oh.make_opsetid(*o) for o in self.opsets.items()]
        if as_function:
            assert function_name, "Function name cannot be empty."
            assert function_domain, "Function domain cannot be empty."
            assert not self.initializers_dict, (
                f"function_name={function_name!r}, initializers "
                f"are not supported when exporting a local function "
                f"{self.get_debug_msg()}"
            )
            assert not self.functions, (
                f"function_name={function_name!r}, local functions "
                f"[{', '.join(f.name for f in self.functions.values())}] "
                f"are not supported yet when exporting a local function "
                f"{self.get_debug_msg()}"
            )
            return oh.make_function(
                function_domain,
                function_name,
                [i.name for i in self.inputs],
                [o.name for o in self.outputs],
                self.nodes,
                [_ for _ in opsets if _.domain != function_domain],
            )

        if self.ir_version:
            ir_version = self.ir_version
        elif "" in self.opsets:
            ir_version = _default_OPSET_TO_IR_VERSION()[self.opsets[""]]

        if self.verbose:
            print(f"[GraphBuilder-{self._hash()}.to_onnx] make_model")
            print(
                f"[GraphBuilder-{self._hash()}.time_evaluation_constants_] "
                f"{self.time_evaluation_constants_}"
            )

        # building the model
        model = ModelProto()
        model.graph.CopyFrom(GraphProto())

        model.graph.node.extend(self.nodes)
        model.graph.name = "experiment"
        model.graph.input.extend(self.inputs)
        model.graph.output.extend(self.outputs)

        # initializer

        initializers, large_initializers = self._build_initializers(
            switch_low_high=sys.byteorder != "big",
            large_model=large_model,
            external_threshold=external_threshold,
        )
        model.graph.initializer.extend(initializers)

        model.opset_import.extend(opsets)
        model.functions.extend(self.functions.values())
        model.ir_version = ir_version
        self._add_shape_information(model)

        if large_model:
            lm = TorchModelContainer()
            lm.model_proto = model
            if large_initializers:
                lm.set_large_initializers(large_initializers)
                lm.check_large_initializers()
            return (lm, stats) if return_optimize_report else lm

        return (model, stats) if return_optimize_report else model

    def _add_shape_information(self, model: ModelProto):
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

    def _check(self, stats: List, step: str):
        begin = time.perf_counter()
        assert (
            len(self.nodes) > 0
        ), f"The onnx model is empty (step {step}, no node){self.get_debug_msg()}"
        known = set(n.name for n in self.inputs)
        known |= set(self.initializers_dict)
        for node in self.nodes:
            assert (
                node.domain in self.opsets
            ), f"Domain {node.domain!r} is not registered in {self.opsets}"
            for i in node.input:
                assert not i or self.has_name(i), (
                    f"Name {i!r} not registered, node type is "
                    f"{node.op_type!r}, node name is {node.name!r}, "
                    f"input are {node.input}{self.get_debug_msg()}"
                )
                if i == "":
                    continue
                assert i in known, f"Unknown input {i!r}, step {step!r} in node {node}"
            known |= set(node.output)
        for o in self.outputs:
            assert o.name in known, f"Unknown output {o.name!r}, step {step!r}"
        stats.append(dict(pattern=f"check_{step}", time_in=time.perf_counter() - begin))

    def optimize(self) -> List[Dict[str, Any]]:
        """
        Optimizes a graph.
        Returns the list of applied processes.
        """
        self._clean_values_cache()

        statistics = []
        main_begin = time.perf_counter()

        if self.verbose or self.optimization_options.verbose:
            print(f"[GraphBuilder.optimize] start with {len(self.nodes)} nodes")
            print(f"[GraphBuilder.optimize] options={self.optimization_options!r}")

        self._check(statistics, "A")
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
            self._check(statistics, "B")
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
            self._check(statistics, "C")

        if self.optimization_options.constant_folding:
            # First constant removal
            begin = time.perf_counter()
            n = self.constant_folding()
            statistics.append(
                dict(
                    pattern="constant_folding",
                    removed=n,
                    time_in=time.perf_counter() - begin,
                    iteration=0,
                )
            )
            self._check(statistics, "Da")
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
                self._check(statistics, "Ea")

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
            self._check(statistics, "F")
            begin = time.perf_counter()
            n = self.remove_unused()
            statistics.append(
                dict(
                    pattern="remove_unused",
                    removed=n,
                    time_in=time.perf_counter() - begin,
                )
            )
            self._check(statistics, "G")

        if self.optimization_options.constant_folding:
            # Second constant removal
            begin = time.perf_counter()
            n = self.constant_folding()
            statistics.append(
                dict(
                    pattern="constant_folding",
                    removed=n,
                    time_in=time.perf_counter() - begin,
                    iteration=1,
                )
            )
            self._check(statistics, "Db")
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
                self._check(statistics, "Eb")

        if self.optimization_options.order:
            res = self.optimize_order()
            statistics.extend(res)
            self._check(statistics, "order")

        if self.verbose or self.optimization_options.verbose:
            duration = time.perf_counter() - main_begin
            print(
                f"[GraphBuilder.optimize] done with "
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
                if k in {"changed", "scale"}:
                    if k not in o:
                        o[k] = 0
                    o[k] += v
                    continue
                if k in {"iter"}:
                    if k not in o:
                        o[k] = 0
                    o[k] = max(o[k], v)
                    continue
                if k in {"algo"} and k not in o:
                    o[k] = []
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

        # adding statistics on node type
        rows.append(self._compile_model_statistics(False))
        rows.append(self._compile_model_statistics(True))
        return "\n".join(rows)

    def _compile_model_statistics(self, detailed: bool):
        rows = [
            f"--MODEL: {len(self.nodes)} nodes, {len(self.inputs)} inputs, "
            f"{len(self.outputs)} outputs, "
            f"{len(self.initializers_dict)} initializers--"
            f"{'DETAILED--' if detailed else ''}"
        ]
        if detailed:

            def _shape(name):
                if self.has_shape(name):
                    s = self.get_shape(name)
                    if len(s) == 0:
                        return "1"
                    return "x".join(map(str, s))
                if self.has_rank(name):
                    r = self.get_rank(name)
                    if r == 0:
                        return "1"
                    return "x".join(["?"] * r)
                return "?"

            def _key(name):
                if not name:
                    return ""
                if isinstance(name, str):
                    tt = self.get_type(name) if self.has_type(name) else "?"
                    return f"{tt}t[{_shape(name)}]"
                if name.op_type == "Transpose":
                    perm = ";".join(map(str, self.get_attribute(name, "perm").ints))
                    return f"{_key(name.input[0])}-perm={perm}"
                return ", ".join(map(_key, name.input))

            cc = Counter([_key(i) for i in self.input_names])
            for k, v in sorted(cc.items()):
                rows.append(f"     INPUT: {v:3d} x {k}")
            cc = Counter([_key(i) for i in self.output_names])
            for k, v in sorted(cc.items()):
                rows.append(f"    OUTPUT: {v:3d} x {k}")
            cc = Counter([_key(i) for i in self.initializers_dict])
            for k, v in sorted(cc.items()):
                rows.append(f"      INIT: {v:3d} x {k}")
            op_types = [(n.domain, n.op_type, _key(n)) for n in self.nodes]
            cc = Counter(op_types)
            for k, v in sorted(cc.items()):
                if k[0] == "":
                    rows.append(f"      NODE: {v:3d} x {k[1]} -SIG- {k[2]}")
                else:
                    rows.append(f"      NODE: {v:3d} x {k[0]}.{k[1]} -SIG- {k[2]}")
        else:
            cc = Counter([self.get_type(i) for i in self.input_names])
            for k, v in sorted(cc.items()):
                rows.append(f"     INPUT: {v:3d} x {k}t")
            cc = Counter([self.get_type(i) for i in self.output_names])
            for k, v in sorted(cc.items()):
                rows.append(f"    OUTPUT: {v:3d} x {k}t")
            cc = Counter([self.get_type(i) for i in self.initializers_dict])
            for k, v in sorted(cc.items()):
                rows.append(f"      INIT: {v:3d} x {k}t")
            op_types = [(n.domain, n.op_type) for n in self.nodes]
            cc = Counter(op_types)
            for k, v in sorted(cc.items()):
                if k[0] == "":
                    rows.append(f"      NODE: {v:3d} x {k[1]}")
                else:
                    rows.append(f"      NODE: {v:3d} x {k[0]}.{k[1]}")
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

    def optimize_order(self):
        from ..xoptim.order_optim import OrderOptimization

        opt = OrderOptimization(
            self,
            algorithm=self.optimization_options.order,
            verbose=max(self.verbose, self.optimization_options.verbose),
        )
        return opt.optimize()

    @classmethod
    def _get_hidden_inputs(self, graph: GraphProto) -> Set[str]:
        """
        Returns the hidden inputs (inputs coming from an upper context)
        used by a subgraph.
        """
        hidden = set()
        memo = set(i.name for i in graph.initializer)
        memo |= set(i.name for i in graph.sparse_initializer)
        for node in graph.node:
            for i in node.input:
                if i not in memo:
                    hidden.add(i)
            for att in node.attribute:
                if att.type == AttributeProto.GRAPH and att.g:
                    hid = self._get_hidden_inputs(att.g)
                    less = set(h for h in hid if h not in memo)
                    hidden |= less
            memo |= set(node.output)
        return hidden

    @classmethod
    def _enumerate_inputs_with_subgraph(cls, node: NodeProto) -> Iterator[str]:
        """
        Enumerates all inputs from a node including all the hidden inputs
        from subgraphs.
        """
        yield from node.input
        if node.op_type in {"Loop", "Scan", "If", "SequenceMap"}:
            for att in node.attribute:
                if att.type == AttributeProto.GRAPH:
                    hidden_inputs = cls._get_hidden_inputs(att.g)
                    yield from hidden_inputs

    def remove_unused(self) -> int:
        """
        Simple function to remove unused nodes.
        It does not look into subgraphs and assumes there is none.
        Everything is done in one pass.
        Returns the number of removed nodes.
        """
        self._clean_values_cache()
        start = len(self.nodes)
        # mark outputs
        marked = {o.name: set() for o in self.outputs}
        for node in reversed(self.nodes):
            used = False
            node_inputs = list(self._enumerate_inputs_with_subgraph(node))
            for o in node.output:
                if o in marked:
                    for i in node_inputs:
                        marked[o].add(i)
                        used = True
            if used or self.do_not_remove(node):
                for i in node_inputs:
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
                            f"[GraphBuilder.remove_unused] "
                            f"remove_initializer:{k}:{v.dtype}[{v.shape}]"
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
                    f"[GraphBuilder.remove_unused_node] remove "
                    f"{node.op_type}-{node.name} -> {node.output}"
                )

        self.nodes = [node for i, node in enumerate(self.nodes) if i not in removed]
        return start - len(self.nodes)

    def compute_constant(
        self, name: str, exc: bool = True, only_array: bool = False
    ) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
        """
        Computes a constant.

        :param name: constant name
        :param exc: raises an exception if any failure
        :param only_array: do not return TensorProto
        :return: constant
        """
        assert self.is_constant(name), f"Name {name!r} is not a constant"
        if name in self.initializers_dict:
            value = self.initializers_dict[name]
            assert not isinstance(
                value, tuple
            ), f"Unexpected type {type(value)} for name={name!r}"
            if only_array and isinstance(value, TensorProto):
                # Should reuse memory buffer here.
                v = onh.to_array(value)
                self.initializers_dict[name] = v
                return v, None
            return value, None

        v = self.constants_[name]
        # It should not be None but a node as it is not an initializer.
        assert isinstance(v, NodeProto), f"Unexpected type {type(v)} for name={name!r}"
        feeds = {i: self.get_constant(i, exc=exc, computed_value=True) for i in v.input}
        for kval, val in feeds.items():
            assert "FakeTensor" not in str(type(val)), (
                f"FakeTensor {kval!r} cannot be an initializer {type(val)}, "
                f"v.op_type={v.op_type!r}"
                f"{self.get_debug_msg()}"
            )
            if val is None:
                return None, None

        with self.maybe_disable_fake_tensor_mode():
            if v.op_type == "Identity":
                # much faster this way
                output = [feeds[v.input[0]]]
            elif v.op_type in {"Mul", "Add", "Sub", "Div"}:
                # bypassing onnx.numpy_helper.from_array, too slow
                output = self._apply_binary_op(v, feeds)
            elif v.op_type in {"Sqrt"}:
                # bypassing onnx.numpy_helper.from_array, too slow
                output = self._apply_unary_function(v, feeds)
            elif hasattr(self, f"_apply_{v.op_type.lower()}"):
                output = getattr(self, f"_apply_{v.op_type.lower()}")(v, feeds)
            elif all(isinstance(v, np.ndarray) for v in feeds.values()):
                # Let's avoid big computation on CPU.
                max_dim = 0
                for _v in feeds.values():
                    max_dim = max(max_dim, np.prod(_v.shape))
                if max_dim >= 2**22:
                    if self.verbose > 1:
                        print(
                            f"[GraphBuilder.compute_constant] stop computing a "
                            f"constant as it may be too big, shapes are "
                            f"{[_.shape for _ in feeds.values()]}"
                        )
                    return None, None

                begin = time.perf_counter()
                ref = ExtendedReferenceEvaluator(v)
                try:
                    output = ref.run(None, feeds)
                except (ValueError, TypeError) as e:
                    sf = ", ".join(f"{k}:{v.dtype}:{v.shape}" for k, v in feeds.items())
                    if "warnings" not in self._debug_msg:
                        self._debug_msg["warnings"] = []
                    sv = str(v).replace("\n", " ")
                    self._debug_msg["warnings"].append(
                        f"Issue with v={sv}, feeds={sf}, e={e}"
                    )
                    self.time_evaluation_constants_ += time.perf_counter() - begin
                    return None, None

                self.time_evaluation_constants_ += time.perf_counter() - begin
            else:
                return None, None

            cst = None
            for n, val in zip(v.output, output):
                assert not isinstance(val, tuple), f"Unexpected type {type(val)} for n={n!r}"
                assert "FakeTensor" not in str(type(val)), (
                    f"FakeTensor detected {type(val)} in constant {name!r}, "
                    f"v.op_type={v.op_type!r}{self.get_debug_msg()}"
                )
                if self.has_type(n):
                    # numpy changes the expected type sometimes
                    # (like transpose(x: float36) --> float32)
                    itype = self.get_type(n)
                    if hasattr(val, "detach"):
                        val = val.to(onnx_dtype_to_torch_dtype(itype))
                    else:
                        val = val.astype(oh.tensor_dtype_to_np_dtype(itype))
                self.constants_computed_[n] = val
                if name == n:
                    cst = val

        assert cst is not None, f"Constant {name!r} was not found in {v.output}"
        return cst, feeds

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
            if self.verbose >= 10:
                for name in self._known_names:
                    print(
                        f"[GraphBuilder.constant_folding] "
                        f"cst:: {1 if self.is_constant(name) else '.'} :: {name}"
                    )
        start = len(self.nodes)
        updates = {}
        node_to_remove = set()
        for k, v in self.constants_.items():
            if v is None:
                # this is an initiliazer
                if self.verbose > 4:
                    print(f"[GraphBuilder.constant_folding] initializer: {k}")
                continue
            assert isinstance(v, NodeProto), f"Unexpected type {type(v)} for k={k!r}"
            if self.verbose > 4:
                print(f"[GraphBuilder.constant_folding] from: {v.op_type}({k})")
            # a node
            if all(map(self.is_constant, v.input)):
                # node evaluation
                output, feeds = self.compute_constant(k, exc=False)
                if output is None:
                    # Evaluation failed.
                    continue
                if convert_into_initializer:
                    node_to_remove.add(tuple(v.output))
                if not isinstance(output, tuple):
                    output = (output,)
                for name, value in zip(v.output, output):
                    updates[name] = None
                    if convert_into_initializer:
                        assert "FakeTensor" not in str(type(value)), (
                            f"FakeTensor {name!r} cannot be an initializer {type(value)}, "
                            f"v.op_type={v.op_type!r} (input types: "
                            f"{[type(i) for i in feeds.values()]})"
                            f"{self.get_debug_msg()}"
                        )
                        self.initializers_dict[name] = value
                    else:
                        updates[name] = v
                    if self.verbose > 3:
                        print(
                            f"[GraphBuilder.constant_folding] fold_constant:"
                            f"{v.op_type}:{name}[{value.dtype}:"
                            f"{value.shape}]:from:{','.join(sorted(feeds))}"
                        )

        for k, v in updates.items():
            self.update_node_constant(k, v)

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

    def _clean_values_cache(self):
        """
        Cleans the cache. The cache is used to avoid the creation of new constants
        while creating a graph. It should be removed the graph is modified.
        """
        if self._values:
            self._values.clear()

    @classmethod
    def _rename_inputs_in_node(
        cls, node: NodeProto, replacements: Dict[str, str], to_rename: Set[str]
    ) -> NodeProto:
        set_inputs = set(cls._enumerate_inputs_with_subgraph(node))
        if set_inputs & to_rename:
            new_inputs = [replacements.get(i, i) for i in node.input]
            new_outputs = [replacements.get(i, i) for i in node.output]
            assert set(new_inputs) & set(new_outputs) in ({""}, set()), (
                f"Node type {node.op_type}-{node.name} is incorrectly replaced "
                f"{node.input}->{new_inputs} and {node.output}->{new_outputs}\n"
                f"replacements are\n{pprint.pformat(replacements)}"
            )
            new_node = oh.make_node(
                node.op_type,
                new_inputs,
                new_outputs,
                domain=node.domain,
                name=node.name,
            )

            if node.op_type in {"Loop", "Scan", "If", "SequenceMap"}:
                # Hidden inputs must be taken care of.
                node_attributes = []
                for att in node.attribute:
                    node_attributes.append(
                        att
                        if att.type != AttributeProto.GRAPH
                        else oh.make_attribute(
                            att.name,
                            cls._rename_inputs_in_subgraph(att.g, replacements),
                        )
                    )
            else:
                node_attributes = node.attribute
            new_node.attribute.extend(node_attributes)
            return new_node
        return node

    @classmethod
    def _rename_inputs_in_subgraph(
        cls, graph: GraphProto, replacements: Dict[str, str]
    ) -> GraphProto:
        """
        Renames inputs.
        """
        # graph inputs and outputs should node be changed, initializer as well
        to_rename = set(replacements)
        nodes = []
        for node in graph.node:
            nodes.append(cls._rename_inputs_in_node(node, replacements, to_rename))
        return oh.make_graph(
            nodes,
            graph.name,
            graph.input,
            graph.output,
            graph.initializer,
            sparse_initializer=graph.sparse_initializer,
        )

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
        self._clean_values_cache()
        # make_initializer
        if self.verbose > 1:
            begin_ = time.perf_counter()
            print(f"[GraphBuilder.remove_identity_nodes] starts with {len(self.nodes)}")
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
                assert "FakeTensor" not in str(type(self.initializers_dict[k])), (
                    f"FakeTensor {k!r} cannot be an initializer "
                    f"{type(self.initializers_dict[k])}{self.get_debug_msg()}"
                )
                self.initializers_dict[v] = self.initializers_dict[k]
                del self.initializers_dict[k]
                assert self.constants_[v]
                self.update_node_constant(v, self.constants_[k])
                del self.constants_[k]

        # third pass: replacements in node
        if self.verbose > 1:
            print(f"[GraphBuilder.remove_identity_nodes] kept {len(new_nodes)} nodes")
        self.nodes = []
        added = 0
        for node in new_nodes:
            repo = {o for o in node.output if o in replacements}
            repi = {
                o for o in self._enumerate_inputs_with_subgraph(node) if o in replacements
            }
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

                if node.op_type in {"Loop", "Scan", "If", "SequenceMap"}:
                    # Hidden inputs must be taken care of.
                    node_attributes = []
                    for att in node.attribute:
                        node_attributes.append(
                            att
                            if att.type != AttributeProto.GRAPH
                            else oh.make_attribute(
                                att.name,
                                self._rename_inputs_in_subgraph(att.g, replacements),
                            )
                        )
                else:
                    node_attributes = node.attribute
                new_node.attribute.extend(node_attributes)
                self.nodes.append(new_node)
                if all(self.is_constant(i) for i in new_node.input):
                    for o in new_node.output:
                        self.update_node_constant(o, new_node)
            else:
                self.nodes.append(node)

        if self.verbose > 1:
            print(
                f"[GraphBuilder.remove_identity_nodes] ends with "
                f"{len(self.nodes)} nodes in "
                f"{time.perf_counter() - begin_} seconds"
            )

        # fourth pass: simplify the graph.
        identity_outputs = {}
        for node in self.nodes:
            if node.op_type != "Identity" or node.domain != "":
                continue
            anc = node.input[0]
            while anc in identity_outputs:
                anc = identity_outputs[anc]
            identity_outputs[node.output[0]] = anc

        for node in self.nodes:
            new_inputs = []
            rename = False
            for i in node.input:
                if i in identity_outputs:
                    new_inputs.append(identity_outputs[i])
                    rename = True
                else:
                    new_inputs.append(i)
            if rename:
                del node.input[:]
                node.input.extend(new_inputs)

        # results
        return removed, added

    def _position_msg(
        self, nodes: List[NodeProto], around: Optional[Tuple[int, int]] = None
    ) -> str:
        "Buids an error message."
        pos = {}
        posi = {}
        for i, n in enumerate(self.nodes):
            if n is None:
                continue
            pos[id(n)] = i
            for o in n.output:
                pos[o] = i
            for o in n.input:
                if o not in posi:
                    posi[o] = i

        rows = []
        for node in nodes:
            if node is None:
                continue
            rows.append(
                f"{node.op_type}({', '.join(node.input)}) -> "
                f"[{', '.join(node.output)}]  -- {node.name}"
            )
            for i in node.input:
                rows.append(f"  -> pos({i}) = {pos.get(i, ' -')} -> {posi.get(i, ' -')}")
            for i in node.output:
                rows.append(f"  <- pos({i}) = {pos.get(i, ' -')} -> {posi.get(i, ' -')}")
        if around is None:
            return "\n".join(rows)

        rows.append("---")
        for i in range(max(0, around[0] - 3), min(len(self.nodes), around[1] + 3)):
            n = self.nodes[i]
            if n is None:
                continue
            rows.append(
                f"P{i}: {n.op_type}({', '.join(n.input)}) -> "
                f"[{', '.join(n.output)}]                   -- {n.name}"
            )
        return "\n".join(rows)

    def _needed_at_first_at(self):
        "Needed by insert_and_remove_nodes."
        needed_at = {}
        first_at = {}
        for i, node in enumerate(self.nodes):
            for name in node.input:
                if name not in needed_at:
                    needed_at[name] = i
            for name in node.output:
                if name not in first_at:
                    first_at[name] = i
        return needed_at, first_at

    def _move_node_position(self, pos: int) -> int:
        """Tries to move a node at position pos closed to the beginning."""
        the_node = self.nodes[pos]
        first_at = {}
        for i, node in enumerate(self.nodes):
            if i > pos:
                break
            for name in node.output:
                if name not in first_at:
                    first_at[name] = i
        can_be = max(first_at.get(i, 0) for i in the_node.input) + 1
        if can_be >= pos:
            return None
        self.nodes[can_be + 1 : pos + 1] = self.nodes[can_be:pos]
        self.nodes[can_be] = the_node
        return can_be

    def insert_and_remove_nodes(
        self,
        insert_at: Optional[int],
        new_nodes: List[NodeProto],
        removed: List[int],
        opsets: Optional[Dict[str, int]] = None,
        debug: Optional[Any] = None,
    ) -> List[NodeProto]:
        """
        Inserts new nodes and removes others.

        :param insert_at: insert the new nodes at this position,
            if empty, the function guesses where to add them
        :param new_nodes: list of nodes to insert
        :param removed: list of nodes to removed (based on their positions)
        :param opsets: opsets used
        :param debug: anything added to exception messages
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
            # We need to remove the constant.
            for o in n.output:
                if o in self.constants_:
                    del self.constants_[o]
            self.nodes[i] = None

        n_existing = []
        for node in new_nodes:
            for i in self._enumerate_inputs_with_subgraph(node):
                assert self.has_name(i), (
                    f"Input {i!r} does not exist for node {node.op_type!r}, "
                    f"debug={debug}{self.get_debug_msg()}"
                )
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

        assert n_existing, "Any output of the new node is connected to existing names."
        if insert_at is not None:
            for i, n in enumerate(new_nodes):
                assert isinstance(n, NodeProto), f"Unexpected type {type(n)} for a node"
                self.nodes.insert(insert_at + i, n)
                self._make_node_set_type_shape_constant(n, True)
                self._make_node_set_type_shape(n)
            self.nodes = [n for n in self.nodes if n is not None]
            return memo

        self.nodes = [n for n in self.nodes if n is not None]

        # Needs to insert the nodes at the right location.
        # Let's find out where the best position is.
        needed_at, first_at = self._needed_at_first_at()

        # First loop to check positions are ok otherwise move a node or two.
        N = len(self.nodes)
        inode = 0
        while inode < len(new_nodes):
            node = new_nodes[inode]
            if node.input:
                min_position = max(first_at.get(i, -1) for i in node.input) + 1
            else:
                # a constant node
                min_position = 0
            max_position = min(needed_at.get(o, N) for o in node.output)

            if min_position <= max_position:
                inode += 1
                continue

            # trouble, let's assume one move is ok.
            mini = max((first_at.get(i, -1), i) for i in node.input)
            pos, name = mini
            assert (
                name in self.nodes[pos].output
            ), f"Name {name!r} should be at node position {pos}"
            new_position = self._move_node_position(pos)
            assert new_position, (
                f"Node at position {pos} cannot be moved.\n----\n"
                f"{self._position_msg([node])}"
                f"\n-------\n{self._position_msg(new_nodes)}"
            )
            needed_at, first_at = self._needed_at_first_at()

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
                f"inserted_at={inserted_at}\n{self._position_msg([node])}"
                f"\n-------\n{self._position_msg(new_nodes)}"
            )

            if node.input:
                local_min_position = max(insert_first_at.get(i, -1) for i in node.input)
            else:
                # a constant node
                local_min_position = 0
            local_max_position = min(insert_needed_at.get(o, N) for o in node.output)

            assert local_min_position <= local_max_position, (
                f"Unable to insert node {self.print_node(node)}, "
                f"local_min_position={local_min_position}, "
                f"local_max_position={local_max_position}, "
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
        if itype > 0:
            self.set_type(val.name, itype)
        shape = tuple(
            d.dim_param if d.dim_param else d.dim_value
            for d in val.type.tensor_type.shape.dim
        )
        if all(isinstance(s, int) for s in shape) and -1 in shape:
            # Some converters uses -1 to specify a dynamic dimension.
            # We replace the value with a string
            new_shape = []
            for index, s in enumerate(shape):
                if s >= 0:
                    new_shape.append(s)
                    continue
                dyn_name = f"{val.name}_{index}"
                new_shape.append(dyn_name)
            shape = tuple(new_shape)
        for i, sh in enumerate(shape):
            if isinstance(sh, int):
                continue
            if not self.has_dynamic_object(sh):
                self.make_dynamic_object(
                    sh, self.torch.SymInt(sh), input_name=val.name, axis=i
                )
        self.set_shape(val.name, shape, exc=False)

    def _update_shape_types_with_proto(
        self, proto: ModelProto, infer_shapes: Union[bool, str] = False
    ):
        """
        Updates the shapes and types for an existing model.

        :param proto: model proto
        :param infer_shapes: infer shapes to fill information about type and shapes
            run shape inference, if the value is `'new'`,
            existing shapes are ignored
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
            if infer_shapes == "new":
                del proto.graph.value_info[:]
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
        self.initializers_dict.update({i.name: i for i in proto.graph.sparse_initializer})
        self.functions = {}
        for f in proto.functions:
            self.functions[f.domain, f.name] = f
        self.value_info = list(proto.graph.value_info)
        self.inputs = list(proto.graph.input)
        self.outputs = list(proto.graph.output)
        self.input_names = [i.name for i in proto.graph.input]

        if hasattr(proto.graph, "value_info"):
            available_shapes = {v.name: v for v in proto.graph.value_info}
        else:
            available_shapes = {}

        for k, v in self.initializers_dict.items():
            self.update_node_constant(k, None)
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
                if all(isinstance(s, int) for s in shape) and -1 in shape:
                    # Some converters uses -1 to specify a dynamic dimension.
                    # We replace the value with a string
                    new_shape = []
                    for index, s in enumerate(shape):
                        if s >= 0:
                            new_shape.append(s)
                            continue
                        dyn_name = f"{i.name}_{index}"
                        new_shape.append(dyn_name)
                    shape = tuple(new_shape)
                for axis, sh in enumerate(shape):
                    if isinstance(sh, int):
                        continue
                    if not self.has_dynamic_object(sh):
                        self.make_dynamic_object(
                            sh, self.torch.SymInt(sh), input_name=i.name, axis=axis
                        )
                self.set_shape(i.name, shape)
            if (
                self.get_type(i.name) == TensorProto.INT64
                and self.has_shape(i.name)
                and self.get_shape(i.name) in ((1,), tuple)
                and "dim" in i.name
            ):
                self.set_value_shape(i.name, (i.name,))

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
                dtypes = [(self.get_type(n) if self.has_type(n) else 0) for n in node.input]
                ranks = [(self.get_rank(n) if self.has_rank(n) else -1) for n in node.input]
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
                    node.output[0],
                    next(_ for _ in unique_dtypes),
                    shapes=None,
                    ranks=ranks,
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

                self.update_node_constant(node.output[0], node)
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

                self.update_node_constant(node.output[0], node)
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

                self._make_node_set_type_shape_constant(node, True)
                self._make_node_set_type_shape(node)

            new_nodes.append(node)
            for o in node.output:
                if o in available_shapes:
                    self._update_shape_types_with_proto_one_result(available_shapes[o])

            if not bypass_shape:
                if any(
                    (x not in available_shapes and not self.has_type(x)) for x in node.output
                ):
                    # second try
                    self._make_node_set_type_shape(node)

                # This test should be enabled when shape inference is complete.
                # assert all(
                #     map(
                #         lambda x: x in available_shapes or self.has_type(x), node.output
                #     )
                # ), (
                #     f"One output of node {node.op_type!r} "
                #     f"(name={node.name!r}) has no type: "
                #     f"{', '.join(o + ((':' + str(self.get_type(o))) "
                #     f"if self.has_type(o) else ':0') for o in node.output)}"
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
        try:
            return parse_expression(expr, exc=exc, context=self.dynamic_objects)
        except AssertionError as e:
            raise AssertionError(
                f"Unable to parse an expression expr={expr!r} "
                f"due to {e}{self.get_debug_msg()}"
            ) from e

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

        raise RuntimeError(f"Unexpected node type {node.op_type!r}{self.get_debug_msg()}")

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

    def make_local_function(self, name: str, builder: "GraphBuilder", domain: str = ""):
        """
        Adds a local function to exiting graph.

        :param name: local function name
        :param builder: builder
        :domain: domain name

        Method :meth:`GraphBuilder.inline_functions` is called on
        the builder. It modifies the builder inplace.
        """
        assert not self.has_local_function(
            name=name, domain=domain
        ), f"Function {name!r}, domain={domain!r} already exists"
        builder.inline_functions(verbose=max(0, self.verbose - 1))
        builder.move_initializers_to_constant(verbose=max(0, self.verbose - 1))
        onx = builder.to_onnx(
            as_function=True, function_name=name, function_domain=domain, optimize=False
        )
        assert isinstance(
            onx, FunctionProto
        ), f"Unexpected type {type(onx)}, name={name!r}, domain={domain!r}"
        assert all(node.op_type != name or node.domain != domain for node in onx.node), (
            f"Recursivity is not allowed in function {name!r}, domain={domain!r}"
            f"\n------ONNX----\n{onx}"
            f"{self.get_debug_msg()}"
        )
        self.functions[domain, name] = onx
        if domain not in self.opsets:
            self.opsets[domain] = 1

    def has_local_function(self, name: str, domain: str = ""):
        """
        Checks if a local function exists.
        """
        return (domain, name) in self.functions

    def get_local_function_outputs(self, name: str, domain: str = ""):
        """
        Returns the outputs of a local functions.
        """
        return self.functions[domain, name].output

    def register_constraint_dimension(self, dim_name: str, value: Any):
        """
        Registers a constraint on a dimension.

        :param dim_name: dimension name
        :param value: value to register
        """
        if dim_name not in self.constraints_:
            self.constraints_[dim_name] = set()
        self.constraints_[dim_name].add(value)

    def _to_torch_tensor(self, a: Any) -> "torch.Tensor":  # noqa: F821
        """
        Torch does not convert numpy dtype very well.
        """
        if isinstance(a, self.torch.Tensor):
            return a
        if isinstance(a, np.ndarray):
            if len(a.shape) == 0:
                # Then torch may consider this as a the creation of empty array.
                tt = self.torch.Tensor(a.reshape((1,)))
                tt = tt[0]
            else:
                tt = self.torch.Tensor(a)
            ttype = onnx_dtype_to_torch_dtype(dtype_to_tensor_dtype(a.dtype))
            res = tt.to(ttype)
            assert a.shape == tuple(res.shape), (
                f"Unexpected shape {res.shape}, expecting shape={a.shape}, "
                f"dtype={res.dtype}, expected dtype={a.dtype}"
            )
            return res
        raise AssertionError(
            f"Unsupported type {type(a)}, unable to convert to a torch.Tensor."
        )

    def inline_functions(self, verbose: int = 0) -> int:
        """
        Inlines local functions.
        Returns the number of inlined nodes.
        """
        if not self.functions:
            # Nothing to do
            return

        begin0 = time.perf_counter()

        # Checks opsets
        for v in self.functions.values():
            for op in v.opset_import:
                if op.domain in self.opsets:
                    assert op.version == self.opsets[op.domain], (
                        f"Opset version mismatch for domain {op.domain!r}, "
                        f"existing version is {self.opsets[op.domain]}, "
                        f"version for function {v.name!r} is {op.version}"
                    )
                else:
                    self.opsets[op.domain] = op.version

        if verbose:
            print(f"[inline_functions] begin graph {id(self)}")
        stats = []
        self._check(stats, step="before inline")
        begin = time.perf_counter()
        inlined = self._inline_functions_iteration(verbose=verbose)
        stats.append(
            dict(
                pattern="inline",
                time_inline=time.perf_counter() - begin,
                iteration=0,
                inlined=inlined,
            )
        )
        self._check(stats, step="after inline iteration 0")
        it = 0
        while inlined:
            it += 1
            begin = time.perf_counter()
            inlined = self._inline_functions_iteration(verbose=verbose)
            stats.append(
                dict(
                    pattern="inline",
                    time_inline=time.perf_counter() - begin,
                    iteration=0,
                    inlined=inlined,
                )
            )
            self._check(stats, step=f"after inline iteration {it}")

        # We can remove the local functions now.
        self.functions = {}
        if verbose:
            print(
                f"[inline_functions] done graph {id(self)} in {time.perf_counter()-begin0}"
            )
        return stats

    def local_functions_found(self, g: GraphProto) -> bool:
        for node in g.node:
            if node.domain == "":
                continue
            key = node.domain, node.op_type
            if key in self.functions:
                return True
            for att in node.attribute:
                assert att.type != AttributeProto.GRAPHS, (
                    f"node.op_type={node.op_type!r}, node.name={node.name!r}, "
                    f"att.name={att.name!r}, not implemented with att.type={att.type} "
                    f"(AttributeProto.GRAPHS)"
                )
                if att.type == AttributeProto.GRAPH:
                    if self.has_local_function(att.g):
                        return True
        return False

    def _inline_functions_iteration(self, verbose: int = 0) -> int:
        """
        Inlines local functions. Returns the number of replacements.
        """
        n_replacements = 0
        replacements = []
        for pos, node in enumerate(self.nodes):
            for att in node.attribute:
                assert att.type != AttributeProto.GRAPHS, (
                    f"node.op_type={node.op_type!r}, node.name={node.name!r}, "
                    f"att.name={att.name!r}, not implemented with att.type={att.type} "
                    f"(AttributeProto.GRAPHS)"
                )
                if self.local_functions_found(att.g):
                    # A function was detected in a subgraphs.
                    if verbose:
                        print(
                            f"[_inline_functions_iterations] replace local "
                            f"functions in node {node.op_type!r}, name={node.name!r}"
                        )
                    n_replacements += self._inline_functions_subgraph(att.g, verbose)

            key = node.domain, node.op_type
            if key not in self.functions:
                continue

            n_replacements += 1
            if verbose:
                print(
                    f"[_inline_functions_iterations] inline function "
                    f"{self.functions[key].name!r} domain {self.functions[key].domain!r}"
                )
            new_nodes = self._convert_function(node.input, node.output, self.functions[key])
            if verbose:
                print(
                    f"[_inline_functions_iterations] {len(new_nodes)} new nodes "
                    f"for {self.functions[key].name!r}, {self.functions[key].domain!r}"
                )
            replacements.append((pos, node, new_nodes))

        if n_replacements == 0:
            # No replacements to do.
            return n_replacements

        stat = []
        for pos, node, new_nodes in reversed(replacements):
            self.insert_and_remove_nodes(pos, new_nodes, removed=[pos])
            self._check(stat, step=f"after inlining function {node.op_type!r}")

        return n_replacements

    def _inline_functions_subgraph(self, g: GraphProto, verbose: int = 0) -> int:
        """
        Inlines local functions in subgraph (inplace).
        Returns the number of replacements.
        """
        stat = []
        self._check(stat, step="before inline")
        inlined = self._inline_functions_subgraph_iteration(g, verbose=verbose)
        self._check(stat, step="after inline iteration 0")
        total = inlined
        it = 0
        while inlined:
            it += 1
            inlined = self._inline_functions_subgraph_iteration(g, verbose=verbose)
            total += inlined
            it += 1
            self._check(stat, step=f"after inline iteration {it}")
        return total

    def _inline_functions_subgraph_iteration(self, g: GraphProto, verbose: int = 0) -> int:
        new_nodes = []
        print(f"[_inline_functions_subgraph_iteration] begin with {id(g)}")
        n_replacements = 0
        for node in g.node:
            for att in node.attribute:
                assert att.type != AttributeProto.GRAPHS, (
                    f"node.op_type={node.op_type!r}, node.name={node.name!r}, "
                    f"att.name={att.name!r}, not implemented with att.type={att.type} "
                    f"(AttributeProto.GRAPHS)"
                )
                if self.local_functions_found(att.g):
                    # A function was detected in a subgraphs.
                    if verbose:
                        print(
                            f"[_inline_functions_subgraph_iteration] replace local "
                            f"functions in node {node.op_type!r}, name={node.name!r}"
                        )
                    n_replacements += self._inline_functions_subgraph(att.g, verbose)

            key = node.domain, node.op_type
            if key not in self.functions:
                new_nodes.append(node)
                continue

            n_replacements += 1
            if verbose:
                print(
                    f"[_inline_functions_subgraph_iteration] inline function "
                    f"{self.functions[key].name!r} domain {self.functions[key].domain!r}"
                )
            functions_nodes = self._rename_results(
                self._convert_function(node.input, node.output, self.functions[key]),
                replacements={n: n for n in (set(node.input) | set(node.output))},
            )
            new_nodes.extend(functions_nodes)
            if verbose:
                print(
                    f"[_inline_functions_subgraph_iteration] {len(new_nodes)} new nodes "
                    f"for {self.functions[key].name!r}, {self.functions[key].domain!r}"
                )

        if n_replacements > 0:
            del g.node[:]
            g.node.extend(new_nodes)
        print(
            f"[_inline_functions_subgraph_iteration] done with "
            f"{id(g)} and {n_replacements} replacements"
        )
        return n_replacements

    def _rename_results(
        self, nodes: List[NodeProto], replacements: Dict[str, str]
    ) -> List[NodeProto]:
        new_nodes = []
        for node in nodes:
            assert all(i in replacements for i in node.input), (
                f"An input from {node.input} is inkonwn input in node "
                f"{node.op_type!r}, name={node.name!r}"
            )
            new_outputs = []
            for o in node.output:
                if o in replacements:
                    assert (
                        replacements[o] == o
                    ), f"o={o!r} must be an output in {sorted(replacements)!r}"
                    new_outputs.append(o)
                    continue
                new_name = self.unique_name(o)
                replacements[o] = new_name
                new_outputs.append(new_name)

            new_node = oh.make_node(
                node.op_type,
                [replacements[i] for i in node.input],
                new_outputs,
                name=self.unique_node_name(node.name),
                domain=node.domain,
            )
            new_atts = []
            for att in node.attribute:
                assert att.type != AttributeProto.GRAPHS, (
                    f"node.op_type={node.op_type!r}, node.name={node.name!r}, "
                    f"att.name={att.name!r}, not implemented with att.type={att.type} "
                    f"(AttributeProto.GRAPHS)"
                )
                new_atts.append(
                    oh.make_attribute(
                        att.name,
                        self._rename_results_in_subgraph(
                            att.g, replacements=replacements.copy()
                        ),
                    )
                    if att.type == AttributeProto.GRAPH
                    else att
                )
            new_node.attribute.extend(new_atts)
            new_nodes.append(new_node)
        return new_nodes

    def _rename_results_in_subgraph(
        self, g: GraphProto, replacements: Dict[str, str]
    ) -> GraphProto:
        set_rep = set(replacements)
        new_nodes = []
        do = False
        for node in g:
            diff = bool(set(node.input) & set_rep)
            new_inputs = [replacements.get(i, i) for i in node.input]
            new_atts = []
            for att in node.attribute:
                assert att.type != AttributeProto.GRAPHS, (
                    f"node.op_type={node.op_type!r}, node.name={node.name!r}, "
                    f"att.name={att.name!r}, not implemented with att.type={att.type} "
                    f"(AttributeProto.GRAPHS)"
                )
                new_g = self._rename_results_in_subgraph(
                    att.g, replacements=replacements.copy()
                )
                if id(new_g) != id(g):
                    diff = True
                    new_atts.append(
                        oh.make_attribute(att.name, new_g)
                        if att.type == AttributeProto.GRAPH
                        else att
                    )
                else:
                    new_atts.append(att)
            if diff:
                do = True
                new_node = oh.make_node(
                    node.op_type, new_inputs, node.output, domain=node.domain, name=node.name
                )
                new_node.attribute.extend(new_atts)
                new_nodes.append(new_node)
            else:
                new_nodes.append(node)
        if not do:
            return g
        g2 = oh.make_graph(
            new_nodes, g.name, g.input, g.output, g.initializer, g.sparse_initializer
        )
        return g2

    def _convert_function(
        self, inputs: List[str], outputs: List[str], proto: FunctionProto
    ) -> List[NodeProto]:
        """
        Converts a function into a list of nodes.

        :param inputs: inputs in the calling nodes
        :param outputs: outputs in the calling nodes
        :param proto: function proto
        :return: list of nodes
        """
        renamed = dict(zip(proto.input, inputs))
        renamed.update(dict(zip(proto.output, outputs)))
        new_nodes = []
        for node in proto.node:
            new_inputs = []
            for name in node.input:
                assert name in renamed, f"Unable to find {name!r} in renamed={renamed}"
                new_inputs.append(renamed[name])
            new_outputs = []
            for name in node.output:
                if name in renamed:
                    new_outputs.append(renamed[name])
                else:
                    new_name = self.unique_name(name)
                    renamed[name] = new_name
                    new_outputs.append(new_name)

            new_node = oh.make_node(
                node.op_type,
                new_inputs,
                new_outputs,
                name=self.unique_node_name(node.name),
                domain=node.domain,
            )
            new_node.attribute.extend(node.attribute)
            new_nodes.append(new_node)

        return new_nodes

    def move_initializers_to_constant(self, verbose: int = 0):
        """
        Moves initializers as constant nodes.
        """
        if not self.initializers_dict:
            return

        initializers, _ = self._build_initializers(
            switch_low_high=sys.byteorder != "big",
            large_model=False,
            external_threshold=False,
        )

        cst_nodes = []
        for proto in initializers:
            if self.verbose:
                print(
                    f"[move_initializers_to_constant] convert "
                    f"{proto.name!r} into a node 'Constant'"
                )
            cst = oh.make_node("Constant", [], [proto.name], value=proto)
            cst_nodes.append(cst)
            self.constants_node_[proto.name] = cst

        self.initializers_dict = {}
        self.nodes = [*cst_nodes, *self.nodes]
