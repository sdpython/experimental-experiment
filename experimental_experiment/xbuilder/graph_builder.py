import pprint
import time
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np
import onnx.helper as oh
import onnx.numpy_helper as onh
from onnx.shape_inference import infer_shapes
from onnx import AttributeProto, FunctionProto, ModelProto, NodeProto, TensorProto
from experimental_experiment.reference import ExtendedReferenceEvaluator
from .shape_helper import (
    DYNAMIC_SHAPE,
    STATIC_SHAPE,
    all_int,
    all_int_or_str,
    is_static_dimension,
    is_static_shape,
)
from .shape_type_compute import (
    set_type_shape_binary_op,
    set_type_shape_matmul,
    set_type_shape_gemm,
)
from ._onnx_helper import (
    choose_consistent_domain_opset,
    compatible_opsets,
    _default_OPSET_TO_IR_VERSION,
    _nice_shape,
    element_wise_op_types,
    element_wise_op_cmp_types,
)
from ._dtype_helper import dtype_to_tensor_dtype
from ._helper import make_hash
from .optimization_options import OptimizationOptions


class Opset:
    # defined for opset >= 18
    # name: number of expected outputs
    _implemented = {
        "Abs": 1,
        "Add": 1,
        "And": 1,
        "ArgMax": 1,
        "ArgMin": 1,
        "Cast": 1,
        "CastLike": 1,
        "Concat": 1,
        "Constant": 1,
        "ConstantOfShape": 1,
        "Div": 1,
        "Dropout": 2,
        "Elu": 1,
        "Equal": 1,
        "Exp": 1,
        "Expand": 1,
        "Flatten": 1,
        "Gather": 1,
        "GatherElements": 1,
        "GatherND": 1,
        "Gemm": 1,
        "Greater": 1,
        "GreaterOrEqual": 1,
        "Identity": 1,
        "MatMul": 1,
        "MaxPool": 2,
        "Mul": 1,
        "Less": 1,
        "LessOrEqual": 1,
        "Log": 1,
        "LogSoftmax": 1,
        "Neg": 1,
        "Not": 1,
        "Or": 1,
        "Pow": 1,
        "Range": 1,
        "Reciprocal": 1,
        "ReduceMax": 1,
        "ReduceMean": 1,
        "ReduceMin": 1,
        "ReduceSum": 1,
        "Relu": 1,
        "Reshape": 1,
        "ScatterElements": 1,
        "ScatterND": 1,
        "Shape": 1,
        "Sigmoid": 1,
        "Slice": 1,
        "Softmax": 1,
        "Sqrt": 1,
        "Squeeze": 1,
        "Sub": 1,
        "Tile": 1,
        "Transpose": 1,
        "Unsqueeze": 1,
        "Where": 1,
    }

    def __init__(self, builder: "GraphBuilder", opset: int):
        self.opset = opset
        self.builder = builder

    def __getattr__(self, name):
        if name in self._implemented:
            return partial(self.make_node, name)
        try:
            return super().__getattr__(name)
        except AttributeError as e:
            raise AttributeError(
                f"Unable to access attribute {name!r}, "
                f"you can still use this operator with method 'make_node'."
            ) from e

    def make_node(
        self,
        op_type: str,
        *inputs: Optional[Union[str, List[str]]],
        outputs: Optional[Union[int, List[str], str]] = None,
        domain: str = "",
        name: Optional[str] = None,
        **kwargs,
    ):
        if outputs is None:
            outputs = self._implemented[op_type]
        if inputs is None:
            inputs = []
        new_inputs = []
        for i in inputs:
            assert not isinstance(
                i, (list, tuple)
            ), f"Wrong inputs for operator {op_type!r}: {inputs!r}"
            if isinstance(i, str):
                new_inputs.append(i)
            elif hasattr(i, "name"):
                # torch.fx.Node
                new_inputs.append(i.name)
            else:
                cst_name = self.builder.make_initializer(
                    "", i, msg=f"input {i} of op_type={op_type!r}"
                )
                new_inputs.append(cst_name)

        return self.builder.make_node(
            op_type, new_inputs, outputs=outputs, domain=domain, name=name, **kwargs
        )

    @staticmethod
    def _iaxes(op_type, axes) -> int:
        if isinstance(axes, np.ndarray):
            iaxes = axes.tolist()
        elif isinstance(axes, int):
            iaxes = [axes]
        else:
            raise RuntimeError(
                f"Unable to call {op_type} on a dynamic input axis={axes}"
            )
        return iaxes

    def ReduceMaxAnyOpset(self, *args, **kwargs):
        if len(args) == 1:
            return self.ReduceMax(*args, **kwargs)
        assert len(args) == 2, f"ReduceMaxAnyOpset expects 2 arguments not {len(args)}"
        if self.builder.main_opset >= 18:
            return self.ReduceMax(*args, **kwargs)
        return self.ReduceMax(args[0], axes=self._iaxes("ReduceMax", args[1]), **kwargs)

    def ReduceMeanAnyOpset(self, *args, **kwargs):
        if len(args) == 1:
            return self.ReduceMean(*args, **kwargs)
        assert len(args) == 2, f"ReduceMeanAnyOpset expects 2 arguments not {len(args)}"
        if self.builder.main_opset >= 18:
            return self.ReduceMean(*args, **kwargs)
        return self.ReduceMean(
            args[0], axes=self._iaxes("ReduceMean", args[1]), **kwargs
        )

    def UnsqueezeAnyOpset(self, *args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0:
            return self.Unsqueeze(*args)
        assert len(args) == 2, f"UnsqueezeAnyOpset expects 2 arguments not {len(args)}"
        if self.builder.main_opset >= 13:
            return self.Unsqueeze(*args, **kwargs)
        return self.Unsqueeze(args[0], axes=self._iaxes("Unsqueeze", args[1]), **kwargs)

    def ReduceSumAnyOpset(self, *args, **kwargs):
        if len(args) == 1:
            return self.ReduceSum(*args, **kwargs)
        assert len(args) == 2, f"ReduceSumAnyOpset expects 2 arguments not {len(args)}"
        if self.builder.main_opset >= 13:
            return self.ReduceSum(*args, **kwargs)
        return self.ReduceSum(args[0], axes=self._iaxes("ReduceSum", args[1]), **kwargs)


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
    - `_known_ranks: Dict[str, int]`: declared ranks
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

    _op_element_wise_types = element_wise_op_types()
    _op_element_wise_cmp_types = element_wise_op_cmp_types()

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
        self._raise_list = raise_list or set()
        self.constants_computed_ = {}
        self._cache_shape = {}
        self._values = {}
        self._dynamic_alias = {}

        self.nodes = []
        self.initializers_dict = {}
        self.inputs = []
        self.outputs = []

        self._known_shapes = {}
        self._known_types = {}
        self._known_ranks = {}
        self._known_torch_value = {}
        self._known_names = set()
        self._unique_names = set()
        self._unique_node_names = set()
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
            self._update_structures_with_proto(target_opset_or_existing_proto)
            self.constant_folding(convert_into_initializer=False)
            if infer_shapes:
                self._update_shape_types_with_proto(target_opset_or_existing_proto)
        else:
            raise NotImplementedError(
                f"{type(target_opset_or_existing_proto)} is not supported."
            )

        self.op = Opset(self, self.opsets[""])

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
            v = value.detach().numpy()
            self.constants_computed_[name] = v
            return v
        if isinstance(value, TensorProto):
            v = onh.to_array(value)
            self.constants_computed_[name] = v
            return v
        raise TypeError(f"Unable to convert type {type(value)} into numpy array.")

    def set_name(self, name: str):
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
        for_onnx: bool = False,
        exc: bool = False,
    ):
        assert isinstance(name, str), f"Unexpected type {type(name)} for name."
        assert "torch.Size" not in str(shape), (
            f"Unexpected type {type(shape)} for a "
            f"shape={shape}{self.get_debug_msg()}"
        )
        assert isinstance(shape, tuple), f"Unexpected shape type {type(shape)}"
        shape = self.verify_shape(shape, 0, name=name, for_onnx=for_onnx)
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
        assert isinstance(name, str), f"Unexpected type {type(name)} for name."
        if isinstance(dtype, int):
            int_type = dtype
        else:
            int_type = self._get_type(dtype)
        if name in self._known_types:
            if int_type != self._known_types[name]:
                raise RuntimeError(
                    f"Name {name!r} already exists and it is different "
                    f"{self._known_types[name]} != {int_type}."
                )
        if self.verbose > 5:
            print(f"[GraphBuilder-{self._hash()}.set_type] {name}:{int_type}")
        self._known_types[name] = int_type

    def rank(self, name: str) -> int:
        assert isinstance(name, str), f"Unexpected type {type(name)} for name."
        return self.get_rank(name)

    def has_name(self, name: str) -> bool:
        assert isinstance(name, str), f"Unexpected type {type(name)} for name."
        return name in self._known_names

    def has_rank(self, name: str) -> bool:
        assert isinstance(name, str), f"Unexpected type {type(name)} for name."
        return name in self._known_ranks

    def has_shape(self, name: str, full=False) -> bool:
        assert isinstance(name, str), f"Unexpected type {type(name)} for name."
        if name not in self._known_shapes:
            return False
        if full:
            shape = self._known_shapes[name]
            return is_static_shape(shape) and min(shape) >= 0
        return True

    def has_type(self, name: str) -> bool:
        assert isinstance(name, str), f"Unexpected type {type(name)} for name."
        return name in self._known_types

    def get_rank(self, name: str) -> int:
        assert isinstance(name, str), f"Unexpected type {type(name)} for name."
        assert name in self._known_ranks, (
            f"Rank is unknown for result {name!r}, "
            f"known_shapes={self._known_ranks}{self.get_debug_msg()}"
        )
        return self._known_ranks[name]

    def get_shape(self, name: str) -> int:
        assert isinstance(name, str), f"Unexpected type {type(name)} for name."
        assert name in self._known_shapes, (
            f"Shape is unknown for result {name!r}, "
            f"known_shapes={self._known_shapes}{self.get_debug_msg()}"
        )
        return self._known_shapes[name]

    def get_type(self, name: str) -> int:
        assert isinstance(name, str), f"Unexpected type {type(name)} for name."
        assert name in self._known_types, (
            f"Type is unknown for result {name!r}, "
            f"known_types={self._known_types}{self.get_debug_msg()}."
        )
        return self._known_types[name]

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
        return name in self.dynamic_objects

    def make_dynamic_object(
        self, name: str, value: Any, shape_as_input: bool = False
    ) -> str:
        assert name not in self.dynamic_objects, (
            f"Dynamic object {name!r}, value={value!r} "
            f"is already there{self.get_debug_msg()}"
        )
        assert isinstance(
            value, self.torch.SymInt
        ), f"Unexpected type {type(value)} for value{self.get_debug_msg()}"
        self.dynamic_objects[name] = value
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
                        r = self.op.Unsqueeze(
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
        if external:
            raise NotImplementedError("External initializers are not implemented yet.")
        if isinstance(value, int):
            value = np.array(value, dtype=np.int64)
        elif isinstance(value, float):
            value = np.array(value, dtype=np.float32)
        elif hasattr(value, "data"):
            # torch.nn.parameter.Parameter -> np.array
            pass
        elif isinstance(value, np.ndarray):
            pass
        else:
            raise RuntimeError(
                f"Initializer name={name!r}, "
                f"unexpected type {type(value)} for value={value!r} ({msg})."
            )

        key = self.make_key(value)
        if key and key in self._values:
            if name == "":
                return self._values[key]
            return self.make_node("Identity", [self._values[key]], [name])

        itype = self._get_type(value.dtype)
        if name == "":
            sh = "x".join(map(str, value.shape))
            sh2 = (
                "_".join(map(str, value.ravel().tolist()))
                if value.size <= 5 and value.dtype == np.int64
                else ""
            )
            name = self.unique_name(f"init{itype}_s{sh}_{sh2}")
        self.set_shape(name, tuple(value.shape))
        self.set_type(name, itype)
        self.set_name(name)
        self.initializers_dict[name] = value
        self.constants_[name] = None
        if self.verbose and (self.verbose > 1 or np.prod(value.shape) > 100):
            print(
                f"[GraphBuilder-{self._hash()}.make_initializer] {name}[{value.dtype}:{value.shape}]"
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

    def _torch_sym_int(self, d):
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
        assert value in self.dynamic_objects_rev, (
            f"value={value!r}, unable to find dimension {d!r} ({type(d)}) "
            f"(str(d)={str(d)!r}) in {self.dynamic_objects_rev} "
            f"or {self._dynamic_alias} "
            f"{dir(d)}"
            f"{self.get_debug_msg()}"
        )
        assert not isinstance(value, self.torch.SymInt)
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

    def verify_dynamic_shape(
        self, shape: Any, for_onnx: bool = True, name: Optional[str] = None
    ) -> DYNAMIC_SHAPE:
        """
        The implementation of this method should be revisited.
        """
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

                value = self._torch_sym_int(d)
                new_shape.append(value if for_onnx else d)
                continue
            if for_onnx and d is None:
                new_shape.append(None)
                continue
            raise RuntimeError(
                f"Unexpected type {type(d)} in shape={shape} (for_onnx={for_onnx}"
                f"{self.get_debug_msg()}"
            )
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
            self.make_node("Identity", [input_name], [name], check=False)
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
        dyn_shape = self.verify_dynamic_shape(shape, name=input_name)

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
        dyn_shape = self.verify_shape(
            shape, name=name, elem_type=elem_type, for_onnx=True
        )
        self.outputs.append(oh.make_tensor_value_info(name, elem_type, dyn_shape))
        if self.verbose:
            print(
                f"[GraphBuilder-{self._hash()}.make_tensor_output] {name}[{elem_type}:{dyn_shape}]"
            )
        if dyn_shape:
            self.set_shape(name, dyn_shape, for_onnx=True)
        if elem_type:
            self.set_type(name, elem_type)
        return name

    def verify_shape(
        self,
        shape: Optional[DYNAMIC_SHAPE],
        elem_type: int,
        name: Optional[str] = None,
        for_onnx: bool = False,
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
        new_shape = self.verify_dynamic_shape(shape, for_onnx=for_onnx, name=name)
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

    def make_node(
        self,
        op_type: str,
        inputs: Union[str, List[str]],
        outputs: Union[int, List[str], str] = 1,
        domain: str = "",
        attributes: Optional[List[AttributeProto]] = None,
        check: Optional[bool] = None,
        name: Optional[str] = None,
        set_type_shape: bool = False,
        **kwargs,
    ) -> Union[str, List[str]]:
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
                assert self.has_name(i), (
                    f"Input {i!r} does not exist for operator {op_type!r} "
                    f"({self._hash()}){self.get_debug_msg()}"
                )
            for i in output_names:
                assert not self.has_name(i), (
                    f"Output {i!r} already exists for operator {op_type!r} "
                    f"({self._hash()}){self.get_debug_msg()}"
                )
        if check is True:
            for i in inputs:
                assert self.has_shape(i), f"Input {i!r} has no known shape."
                assert self.has_type(i), f"Input {i!r} has no known type."

        if name:
            name = self.unique_node_name(name)

        self._check_op_type(
            op_type, inputs, outputs, domain=domain, name=name, **kwargs
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
        if attributes:
            node.attribute.extend(attributes)

        # constant handling, shape, type
        self._make_node_set_type_shape_constant(node, set_type_shape=set_type_shape)

        if self.verbose > 3:
            print(
                f"[GraphBuilder-{self._hash()}.make_node] "
                f"[{self._debug_string_inputs(node.input, output_names)}] "
                f"{node.op_type}:{node.input}->{node.output}"
            )

        # add the node
        for o in node.output:
            self.set_name(o)
        self.nodes.append(node)
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

    def _make_node_set_type_shape_constant(self, node: NodeProto, set_type_shape: bool):
        if node.domain != "":
            return
        if node.op_type == "Constant":
            size = len(node.SerializeToString())
            if size >= self.optimization_options.constant_size:
                raise ValueError(
                    f"A node Constant holds a tensor bigger than "
                    f"the constant: {size} >= {self.constant_size}."
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
        elif set_type_shape:
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
    ) -> AttributeProto:
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

    def _make_node_set_type_shape(self, node: NodeProto):
        if node.domain != "":
            return
        if node.op_type == "Reshape":
            k = node.output[0]
            self.set_type(k, self.get_type(node.input[0]))
            shape_set = False
            if self.is_constant(node.input[1]):
                cst = tuple(
                    self.get_constant(node.input[1], computed_value=True, as_shape=True)
                )
                if all_int(cst):
                    if -1 not in cst:
                        self.set_shape(k, cst)
                        shape_set = True
                    elif all_int(cst) and self.has_shape(node.input[0]):
                        sh = self.get_shape(node.input[0])
                        new_shape = self._apply_reshape_to_shape(sh, cst)
                        if new_shape is not None:
                            self.set_shape(k, new_shape)
                            shape_set = True
            if not shape_set:
                if self.has_shape(node.input[1]):
                    rk = self.get_shape(node.input[1])
                    self.set_rank(k, rk[0])
        elif node.op_type in self._op_element_wise_cmp_types:
            set_type_shape_binary_op(self, node.output[0], *node.input, cmp_op=True)
        elif node.op_type in self._op_element_wise_types:
            set_type_shape_binary_op(self, node.output[0], *node.input)
        elif node.op_type == "MatMul":
            set_type_shape_matmul(self, node.output[0], *node.input)
        elif node.op_type == "Gemm":
            set_type_shape_gemm(
                self,
                node.output[0],
                *node.input[:2],
                transA=self.get_attribute(node, "transA").i,
                transB=self.get_attribute(node, "transB").i,
            )

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

        assert len(input_names) == len(builder.inputs), (
            f"Inconsistency between input_names={input_names} "
            f"and inputs={builder.inputs}"
        )
        for name, inp in zip(input_names, builder.inputs):
            new_name = self.unique_name(f"{prefix}{inp.name}")
            renaming[inp.name] = new_name
            self.make_node("Identity", [name], [new_name])

        for node in builder.nodes:
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
            self.make_node("Identity", [renaming[out.name]], [name])

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
        tensor.data_type = self._get_type(arr_cpu.dtype)

        if self.verbose and np.prod(arr_cpu.shape) > 100:
            print(
                f"[GraphBuilder-{self._hash()}.from_array] {tensor.data_type}[{arr_cpu.shape}]"
            )

        raw = np_arr.tobytes()
        tensor.raw_data = raw

        if sys.byteorder == "big":
            np_dtype = oh.tensor_dtype_to_np_dtype(tensor.data_type)
            np.byteswap(np.frombuffer(tensor.raw_data, dtype=np_dtype), inplace=True)
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
        def _align(s, length):
            if len(s) < length:
                s += " " * (length - len(s))
            return s

        if not self._debug_msg:
            rows = [""]
            for n in self.nodes:
                if n is None:
                    continue
                rows.append(
                    f"{_align(n.op_type, 20)}: {','.join(n.input)} -> "
                    f"{','.join(n.output)} --- {n.name}"
                )
            return "\n".join(rows)

        def _size(t):
            if hasattr(t, "numel"):
                return t.numel()
            if hasattr(t, "size"):
                return t.size
            raise RuntimeError(f"Size unknown for type {t}.")

        def _values(t):
            if hasattr(t, "detach"):
                return t.detach().numpy().ravel().tolist()
            if hasattr(t, "size"):
                return t.ravel().tolist()
            raise RuntimeError(f"Values unknown for type {t}.")

        rows = ["", "--DEBUG--", "--SHAPE--"]
        rows.append(f"dynamic_objects={pprint.pformat(self.dynamic_objects)}")
        rows.append(f"dynamic_objects_rev={pprint.pformat(self.dynamic_objects_rev)}")
        rows.append(f"dynamic_alias={pprint.pformat(self._dynamic_alias)}")
        rows.append(f"dynamic_shapes={pprint.pformat(self.dynamic_shapes)}")
        rows.append("--TORCH-SHAPES--")
        for kk, vv in self._known_torch_value.items():
            rows.append(
                f"{kk}: {vv} --- "
                f"{self.get_type(kk) if self.has_type(kk) else ''}:"
                f"{self.get_rank(kk) if self.has_rank(kk) else ''}:"
                f"{self.get_shape(kk) if self.has_shape(kk) else ''}:"
            )
        rows.append("--ONNX--")
        for k, v in self._debug_msg.items():
            rows.append(f"-- {k} --")
            if isinstance(v, dict):
                rows.append(pprint.pformat(v))
            else:
                rows.append(str(v))
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
                f"[GraphBuilder-{hs}.make_initializer] {name}[{init.dtype}:{init.shape}{sval}]"
            )
        for node in self.nodes:
            if node is None:
                continue
            rows.append(
                f"[GraphBuilder-{hs}.make_node] "
                f"{_align(node.name, 15)} "
                f"[{self._debug_string_inputs(node.input, node.output, 6)}] "
                f"{node.op_type}:{node.input}->{node.output}"
            )
        for io in self.outputs:
            shh = _nice_shape(io.type.tensor_type.shape)
            rows.append(
                f"[GraphBuilder-{hs}.make_tensor_output] {io.name}"
                f"[{io.type.tensor_type.elem_type}:{shh}]"
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

        dense = self._build_initializers()
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
            print(f"[GraphBuilder-{self._hash()}.to_onnx] onh.make_graph")
        graph = oh.make_graph(
            self.nodes, "experiment", self.inputs, self.outputs, dense
        )
        if self.verbose:
            print(f"[GraphBuilder-{self._hash()}.to_onnx] onh.make_model")
        model = oh.make_model(graph, opset_imports=opsets, functions=self.functions)
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
        for val in self.value_info:
            if self.has_name(val.name):
                model.graph.value_info.append(val)
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
                for i in node.input:
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

        _check(statistics, "A")
        if self.optimization_options.remove_identity:
            begin = time.perf_counter()
            n = self.remove_identity_nodes()
            statistics.append(
                dict(
                    pattern="remove_identity_nodes",
                    removed=n,
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

        return statistics

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
        )
        return gro.optimize(
            max_iter=self.optimization_options.max_iter,
            remove_identity=self.optimization_options.remove_identity,
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
            if used:
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
        assert len(perm) == 2, f"perm={perm} is not supported with torch"
        x = feeds[node.input[0]]
        if isinstance(x, np.ndarray):
            x = self.torch.Tensor(x)
        return [self.torch.transpose(x, *perm)]

    def compute_constant(
        self, name: str, exc: bool = True
    ) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
        assert self.is_constant(name), f"Name {name!r} is not a constant."
        if name is self.initializers_dict:
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
        for name, val in zip(v.output, output):
            self.constants_computed_[name] = val
        return output, feeds

    def constant_folding(self, convert_into_initializer: bool = True) -> int:
        """
        Folds all constants. Constants are marked during the creation of the graph.
        There is no need to propagate this information.

        :param convert_into_initializer: moves the constant as an initializer,
            otherwise, just evaluates it
        :return: number of removed nodes
        """
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
                    if self.verbose:
                        print(
                            f"[GraphBuilder.constant_folding] fold_constant:"
                            f"{v.op_type}:{name}[{value.dtype}:"
                            f"{value.shape}]:from:{','.join(sorted(feeds))}"
                        )

        self.constants_.update(updates)
        new_nodes = []
        for node in self.nodes:
            if tuple(node.output) in node_to_remove:
                continue
            new_nodes.append(node)
        self.nodes = new_nodes
        return start - len(self.nodes)

    def remove_identity_nodes(self) -> int:
        """
        Removes identity nodes. Returns the number of removed nodes.
        """
        # first pass: detect replacements
        start = len(self.nodes)
        new_nodes = []
        input_names = set(i.name for i in self.inputs)
        output_names = set(i.name for i in self.outputs)
        replacements = {}
        replacements_rev = {}
        for node in self.nodes:
            if node.op_type != "Identity":
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
                    f"Name {old_name!r} still in {replacements}, node.op_type={node.op_type!r}, "
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
                f"Name {old_name!r} still in {replacements}, node.op_type={node.op_type!r}, "
                f"node.input={node.input}, node.output={node.output}, "
                f"input_names={input_names}, output_names={output_names}"
            )
            replacements[old_name] = new_name
            replacements_rev[new_name] = old_name

            # verification
            for k, v in replacements.items():
                assert v not in replacements, (
                    f"replacement {k}->{v} is not possible because of "
                    f"{v}->{replacements[v]}, old_name={old_name!r}, new_name={new_name!r}"
                )

        # second pass: replacements in initializer
        for k, v in replacements.items():
            if k in self.initializers_dict:
                if self.optimization_options.verbose > 2:
                    print(
                        f"[GraphBuilder.remove_identity_nodes] rename initializer {k!r} by {v!r}"
                    )
                self.initializers_dict[v] = self.initializers_dict[k]
                del self.initializers_dict[k]
                assert self.constants_[v]
                self.constants_[v] = self.constants_[k]
                del self.constants_[k]

        # third pass: replacements in node
        self.nodes = []
        for node in new_nodes:
            repo = {o for o in node.output if o in replacements}
            repi = {o for o in node.input if o in replacements}
            if repi or repo:
                new_inputs = [replacements.get(i, i) for i in node.input]
                new_outputs = [replacements.get(i, i) for i in node.output]
                assert not (set(new_inputs) & set(new_outputs)), (
                    f"Node type {node.op_type}-{node.name} is incorrectly replaced "
                    f"{node.input}->{new_inputs} and {node.output}->{new_outputs}\n"
                    f"replacements are\n{pprint.pformat(replacements)}"
                )
                if self.optimization_options.verbose > 2:
                    print(
                        f"[GraphBuilder.remove_identity_nodes] node {node.op_type}-{node.name}:"
                        f"{node.input}->{new_inputs}:{node.output}->{new_outputs}"
                    )
                new_node = oh.make_node(
                    node.op_type,
                    new_inputs,
                    new_outputs,
                    domain=node.domain,
                    name=node.name,
                )
                new_node.attribute.extend(node.attribute)
                self.nodes.append(new_node)
            else:
                self.nodes.append(node)

        return start - len(self.nodes)

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
            memo.append(self.nodes[i])
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
        N = len(self.nodes)
        last_position = 0
        new_nodes_p = []
        for node in new_nodes:
            min0_position = max(first_at.get(i, -1) for i in node.input)
            max_position = min(needed_at.get(o, N) for o in node.output)
            min_position = max(min0_position + 1, last_position)
            assert min_position <= max_position, (
                f"Unable to insert node {self.print_node(node)}, "
                f"min_position={min_position}, true_min_position={min0_position}, "
                f"max_position={max_position}, len(nodes)={len(self.nodes)}"
            )
            new_nodes_p.append((min_position, node))
            last_position = min_position
        assert len(new_nodes) == len(new_nodes_p)

        # do the addition
        for i, (p, n) in enumerate(new_nodes_p):
            assert isinstance(n, NodeProto), f"Unexpected type {type(n)} for a node"
            self.nodes.insert(p + i, n)
            self._make_node_set_type_shape_constant(n, True)
            self._make_node_set_type_shape(n)
        return memo

    def _update_shape_types_with_proto(self, proto: ModelProto):
        """
        Updates the shapes and types for an existing model.
        """
        assert isinstance(proto, ModelProto), f"Unexpected type {type(proto)} for proto"
        new_proto = infer_shapes(proto)

        for val in new_proto.graph.value_info:
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

    def _update_structures_with_proto(self, proto: ModelProto):
        """
        Updates the shapes and types for an existing model.
        """
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

        for node in self.nodes:
            self._unique_names |= set(node.output)
            if node.name:
                self._unique_node_names.add(node.name)
            if node.op_type == "Constant":
                self.constants_[node.output[0]] = node
                if not self.has_name(node.output[0]):
                    self.set_name(node.output[0])
                self.set_shape(node.output[0], self._get_tensor_shape(node))
                self.set_type(node.output[0], self._get_tensor_type(node))
            elif node.op_type == "ConstantOfShape" and self.is_constant(node.input[0]):
                self.constants_[node.output[0]] = node
                if not self.has_name(node.output[0]):
                    self.set_name(node.output[0])
                self.set_shape(node.output[0], self.get_shape(node.input[0]))
                if len(node.attribute) == 0:
                    self.set_type(node.output[0], TensorProto.FLOAT)
                else:
                    value = node.attribute[0].t
                    self.set_type(node.output[0], value.data_type)
            else:
                for o in node.output:
                    if not self.has_name(o):
                        self.set_name(o)
