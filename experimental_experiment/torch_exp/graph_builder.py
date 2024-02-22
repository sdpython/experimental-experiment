import pprint
import textwrap
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import onnx.helper as oh
import onnx.numpy_helper as onh
from onnx.shape_inference import infer_shapes
from onnx import AttributeProto, FunctionProto, ModelProto, NodeProto, TensorProto
from onnx_array_api.reference import ExtendedReferenceEvaluator
from .annotations import (
    DYNAMIC_SHAPE,
    STATIC_SHAPE,
    all_int,
    is_static_dimension,
    is_static_shape,
)
from ._aten_helper import dtype_to_tensor_dtype, _nice_shape
from ._helper import make_hash
from .graph_builder_optim import PatternOptimization, GraphBuilderPatternOptimization
from .optimization_patterns import get_default_patterns, get_pattern


def _default_OPSET_TO_IR_VERSION():
    return {
        1: 3,
        2: 3,
        3: 3,
        4: 3,
        5: 3,
        6: 3,
        7: 3,
        8: 4,
        9: 4,
        10: 5,
        11: 6,
        12: 7,
        13: 7,
        14: 7,
        15: 8,
        16: 8,
        17: 8,
        18: 8,
        19: 9,
        20: 9,
        21: 10,
    }


class OptimizationOptions:
    """
    Defines all the optimization to apply.

    :param remove_unused: remove all unused nodes, this must be true if
        pattern optimization is enabled
    :param constant_folding: folds constant as much as possible
    :param constant_size: all node Constant above this threshold should be
        defined as initializer
    :param remove_identity: remove identity nodes
    :param patterns: list of pattern optimization to apply to the graph,
        it looks a a specific subsequence of nodes in a graph
        and do some replacements,
        `'default'` means a default list of optimization patterns are applied
    :param max_iter: maximum number of iteration when doing pattern optimizations,
        -1 to let it undefined
    :param recursive: optimizes subgraphs and functions as well
    :param verbose: verbosity level (for pattern optimization)
    """

    def __init__(
        self,
        remove_unused: bool = True,
        constant_folding: bool = False,
        constant_size: int = 1024,
        remove_identity: bool = True,
        patterns: Union[str, List["PatternOptimization"]] = "default",
        max_iter: int = -1,
        recursive: bool = False,
        verbose: int = 0,
    ):
        self.remove_unused = remove_unused
        self.constant_folding = constant_folding
        self.remove_identity = remove_identity
        self.constant_size = constant_size
        if isinstance(patterns, str):
            assert patterns == "default", f"Unexpected value {patterns!r} for patterns"
            self.patterns = get_default_patterns()
        else:
            assert patterns is None or isinstance(
                patterns, list
            ), f"Unexpected type {type(patterns)} for patterns"
            self.patterns = (
                None if patterns is None else [get_pattern(p) for p in patterns]
            )
        self.max_iter = -1
        self.verbose = verbose
        self.recursive = recursive

    def __repr__(self):
        pats = "None" if self.patterns is None else [str(p) for p in self.patterns]
        code = (
            f"{self.__class__.__name__}(remove_unused={self.remove_unused}, "
            f"constant_folding={self.constant_folding}, "
            f"constant_size={self.constant_size}, verbose={self.verbose}, "
            f"max_iter={self.max_iter}, recursive={self.recursive}, patterns={pats})"
        )
        return "\n".join(
            textwrap.wrap(code, width=80, tabsize=4, subsequent_indent="    ")
        )


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
        "Equal": 1,
        "Exp": 1,
        "Expand": 1,
        "Flatten": 1,
        "Gather": 1,
        "GatherElements": 1,
        "GatherND": 1,
        "Gemm": 1,
        "Greater": 1,
        "Identity": 1,
        "MatMul": 1,
        "MaxPool": 2,
        "Mul": 1,
        "Less": 1,
        "Log": 1,
        "LogSoftmax": 1,
        "Neg": 1,
        "Or": 1,
        "Pow": 1,
        "Range": 1,
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

    def ReduceMeanAnyOpset(self, *args, **kwargs):
        if len(args) == 1:
            return self.ReduceMean(*args, **kwargs)
        assert len(args) == 2, f"ReduceMeanAnyOpset expects 2 arguments not {len(args)}"
        if self.builder.main_opset >= 18:
            return self.ReduceMean(*args, **kwargs)
        return self.ReduceMean(
            args[0], axes=self._iaxes("ReduceMean", args[1]), **kwargs
        )


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

    - `_unique_names`: used to create unused result names
    - `_unique_node_names`: used to create unused node names
    - `_known_names`: set of existing results names
    - `_known_shapes: Dict[str, DYNAMIC_SHAPE]`: declared shapes
    - `_known_types: Dict[str, int]`: declared element types
    - `_known_ranks: Dict[str, int]`: declared ranks
    - `constants_: Dict[str, Any]`: constant values
    - `dynamic_objects: Dict[str, torch.SymInt]`: list of dynamic dimension
    - `dynamic_objects_rev: Dict[str, str]`: reverse dictionary to fasten lookups
    """

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
        self.dynamic_objects = {}
        self.dynamic_objects_rev = {}
        self.functions = []
        self.value_info = []

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
            self.nodes = []
            self.initializers_dict = {}
            self.inputs = []
            self.outputs = []
            self.input_names = input_names or []
            self._unique_names = set(self.input_names)
            self._unique_node_names = set()
            self.current_input = 0
            self._known_shapes = {}
            self._known_types = {}
            self._known_ranks = {}
            self.constants_ = {}
            self._known_names = self._unique_names.copy()

        elif isinstance(target_opset_or_existing_proto, ModelProto):
            # loads a model from nothing
            if input_names:
                raise ValueError(
                    "input_names must be empty if the input is an existing model."
                )
            proto = target_opset_or_existing_proto
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
            self.current_input = len(self.inputs)
            # This should be improve.
            self._known_shapes = {}
            self._known_types = {}
            self._known_ranks = {}
            self.constants_ = {}
            self._unique_node_names = set()
            self._unique_names = set()
            self._known_names = set()

            for k, v in self.initializers_dict.items():
                self.constants_[k] = None
                self._unique_names.add(k)
                self.set_name(k)
                self.set_shape(k, self._get_tensor_shape(v))
                self.set_type(k, self._get_tensor_type(v))
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
                else:
                    for o in node.output:
                        if not self.has_name(o):
                            self.set_name(o)
            if infer_shapes:
                self._update_shape_types_with_proto(target_opset_or_existing_proto)
        else:
            raise NotImplementedError(
                f"{type(target_opset_or_existing_proto)} is not supported."
            )

        self.op = Opset(self, self.opsets[""])

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

    def get_constant(self, name: str, exc: bool = True) -> np.ndarray:
        if not self.is_constant(name):
            raise ValueError(f"Result {name!r} is not a constant.")
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
            return value.detach().numpy()
        if isinstance(value, TensorProto):
            return onh.to_array(value)
        raise TypeError(f"Unable to convert type {type(value)} into numpy array.")

    def set_name(self, name: str):
        assert isinstance(name, str), f"Unexpected type {type(name)} for name."
        assert (
            name not in self._known_names
        ), f"Name {name!r} already exists{self.get_debug_msg()}"
        self._known_names.add(name)
        self._unique_names.add(name)

    def set_rank(self, name: str, value: int):
        assert isinstance(name, str), f"Unexpected type {type(name)} for name."
        assert (
            name not in self._known_ranks
        ), f"Name {name!r} already exists{self.get_debug_msg()}"
        self._known_ranks[name] = value

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

    def _prepare_inputs(self, schema: Optional[Any], *inputs: List[Any]) -> List[str]:
        input_names = []
        for i in inputs:
            self.make_input(i.name, i.dtype, i.shape)
            input_names.append(i.name)
        return input_names

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
                elem_type = TensorProto.INT16
            elif "uint16" in st:
                elem_type = TensorProto.UINT16
            elif "int16" in st:
                elem_type = TensorProto.INT32
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
            return self.make_tensor_input(name, TensorProto.INT64, (1,))
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
        conc = []
        for d in shape:
            if isinstance(d, int):
                conc.append(self.make_initializer("", np.array([d], dtype=np.int64)))
            elif isinstance(d, (str, self.torch.SymInt)):
                name = str(d)
                assert name in self.dynamic_objects or self.has_name(
                    name
                ), f"Unknonw dynamic object {d}-{name!r}{self.get_debug_msg()}"
                conc.append(name)
            else:
                raise RuntimeError(
                    f"Unexpected type {type(d)} for a dimension in {shape}{self.get_debug_msg()}"
                )
        return self.make_node("Concat", conc, axis=0, name=name)

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
        assert (
            not verify
            or is_static_dimension(dim)
            or str(dim) in self.dynamic_objects
            or str(dim) in self.dynamic_objects_rev
            or self.has_name(str(dim))
        ), f"dim={dim!r} (type={type(dim)}) not in found in {self.dynamic_objects}{self.get_debug_msg()}"
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

    def verify_dynamic_shape(self, shape: Any, for_onnx: bool = True) -> DYNAMIC_SHAPE:
        if is_static_shape(shape):
            return tuple(int(i) for i in shape)
        new_shape = []
        for d in shape:
            if isinstance(d, int):
                new_shape.append(d)
                continue
            if isinstance(d, (self.torch.SymInt, str)):
                try:
                    val_int = int(d)
                    new_shape.append(val_int)
                    continue
                except TypeError:
                    pass
                assert str(d) in self.dynamic_objects_rev, (
                    f"Unable to find dimension {d!r} ({type(d)}) "
                    f"in {self.dynamic_objects_rev}"
                    f"{dir(d)}"
                    f"{self.get_debug_msg()}"
                )
                new_shape.append(str(d) if for_onnx else d)
                continue
            if for_onnx and d is None:
                new_shape.append(None)
                continue
            raise RuntimeError(
                f"Unexpected type {type(d)} in shape={shape} (for_onnx={for_onnx}"
                f"{self.get_debug_msg()}"
            )
        return tuple(new_shape)

    def make_tensor_input(self, name: str, elem_type: Any, shape: STATIC_SHAPE) -> str:
        if self.current_input < len(self.input_names):
            # The input needs to be renamed, an identity node is added.
            input_name = self.input_names[self.current_input]
            self.make_node("Identity", [input_name], [name], check=False)
        else:
            self.input_names.append(name)
            input_name = name
            self.set_name(name)
        self.current_input += 1
        elem_type = self._get_type(elem_type)
        dyn_shape = self.verify_dynamic_shape(shape)
        self.inputs.append(oh.make_tensor_value_info(input_name, elem_type, dyn_shape))
        if self.verbose:
            print(
                f"[GraphBuilder-{self._hash()}.make_tensor_input] {name}[{elem_type}:{shape}]"
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
    ) -> Union[str, List[str]]:
        if isinstance(name, list):
            res = []
            for n in name:
                res.append(self.make_tensor_output(n, elem_type, shape))
            return res

        assert (
            not indexed or "_" in name
        ), f"Name {name!r} is not indexed like 'output_0'{self.get_debug_msg()}"
        elem_type = self._get_type(elem_type, False)
        if not self.as_function and elem_type == 0:
            raise RuntimeError(f"Undefined element type for {name!r}.")
        shape = self.verify_shape(shape, name=name, elem_type=elem_type, for_onnx=True)
        self.outputs.append(oh.make_tensor_value_info(name, elem_type, shape))
        if self.verbose:
            print(
                f"[GraphBuilder-{self._hash()}.make_tensor_output] {name}[{elem_type}:{shape}]"
            )
        if shape:
            self.set_shape(name, shape, for_onnx=True)
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
            f"Shape={shape} is not a shape, "
            f"name={name!r}, elem_type={elem_type}{self.get_debug_msg()}"
        )
        return self.verify_dynamic_shape(shape, for_onnx=for_onnx)

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
        if op_type == "Concat":
            for i in inputs:
                if self.has_rank(i) and self.get_rank(i) == 0:
                    raise RuntimeError(
                        f"Input {i} for node Concat has no rank{self.get_debug_msg()}"
                    )
        if op_type.startswith("Reduce"):
            assert (
                len(inputs) == 1 or "axes" not in kwargs
            ), f"Operator {op_type} defines twice the axes{self.get_debug_msg()}"

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

        assert len(input_names) == len(
            builder.inputs
        ), f"Inconsistency between input_names={input_names} and inputs={builder.inputs}."
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
        if not self._debug_msg:
            return ""

        def _align(s, length):
            if len(s) < length:
                s += " " * (length - len(s))
            return s

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
        rows.append("--ONNX--")
        for k, v in self._debug_msg.items():
            rows.append(f"-- {k}")
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
            rows.append(
                f"[GraphBuilder-{hs}.make_node] "
                f"{_align(node.name, 15)} [{self._debug_string_inputs(node.input, node.output, 6)}] "
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

    def optimize(self):
        def _check(step):
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

        _check("A")
        if self.optimization_options.remove_identity:
            self.remove_identity_nodes()
            _check("B")
        if self.optimization_options.remove_unused:
            self.remove_unused()
            _check("C")
        if self.optimization_options.constant_folding:
            self.constant_folding()
            _check("D")
            if self.optimization_options.remove_unused:
                self.remove_unused()
                _check("E")
        if self.optimization_options.patterns:
            assert (
                self.optimization_options.remove_unused
            ), "remove_unused must be positive for pattern optimizations"
            self.optimize_with_patterns()
            _check("F")
            self.remove_unused()
            _check("G")

    def optimize_with_patterns(self):
        """
        Optimizes this graph with patterns.
        """
        gro = GraphBuilderPatternOptimization(
            self,
            verbose=self.optimization_options.verbose,
            patterns=self.optimization_options.patterns,
            recursive=self.optimization_options.recursive,
        )
        gro.optimize(
            max_iter=self.optimization_options.max_iter,
            remove_identity=self.optimization_options.remove_identity,
        )

    def remove_unused(self):
        """
        Simple function to remove unused nodes.
        It does not look into subgraphs and assumes there is none.
        Everything is done in one pass.
        """
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
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        assert self.is_constant(name), f"Name {name!r} is not a constant."
        if name is self.initializers_dict:
            return self.initializers_dict[name]
        v = self.constants_[name]
        assert isinstance(v, NodeProto), f"Unexpected type {type(v)} for name={name!r}"
        feeds = {i: self.get_constant(i, exc=exc) for i in v.input}
        for val in feeds.values():
            if val is None:
                return None, None
        if v.op_type == "Transpose":
            # bypassing onnx.numpy_helper.from_array, too slow
            output = self._apply_transpose(v, feeds)
        else:
            ref = ExtendedReferenceEvaluator(v)
            output = ref.run(None, feeds)
        return output, feeds

    def constant_folding(self):
        """
        Folds all constants. Constants are marked during the creation of the graph.
        There is no need to propagate this information.
        """

        updates = {}
        node_to_remove = set()
        for k, v in self.constants_.items():
            if v is None:
                # this is an initiliazer
                continue
            # a node
            if all(map(self.is_constant, v.input)):
                node_to_remove.add(tuple(v.output))
                # node evaluation
                output, feeds = self.compute_constant(k)
                for name, value in zip(v.output, output):
                    updates[name] = None
                    self.initializers_dict[name] = value
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

    def remove_identity_nodes(self):
        """
        Removes identity nodes.
        """
        # first pass: detect replacements
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

    def insert_and_remove_nodes(
        self, insert_at: int, new_nodes: List[NodeProto], removed: List[int]
    ) -> List[NodeProto]:
        """
        Inserts new nodes and removes others.

        :param insert_at: insert the new nodes at this position
        :param new_nodes: list of nodes to insert
        :param removed: list of nodes to removed (based on their positions)
        :return: list of removed nodes
        """
        assert not removed or min(removed) <= insert_at, (
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
        assert n_existing, "Any output of the new node is conncted to existing names."
        for i, n in enumerate(new_nodes):
            assert isinstance(n, NodeProto), f"Unexpected type {type(n)} for a node"
            self.nodes.insert(insert_at + i, n)
        self.nodes = [n for n in self.nodes if n is not None]
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
