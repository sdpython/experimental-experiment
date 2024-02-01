from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import onnx.helper as oh
import onnx.numpy_helper as onh
from onnx import AttributeProto, FunctionProto, ModelProto, NodeProto, TensorProto
from onnx.reference import ReferenceEvaluator
from ._aten_helper import dtype_to_tensor_dtype, _nice_shape
from ._helper import make_hash


class Opset:
    # defined for opset >= 18
    # name: number of expected outputs
    _implemented = {
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
        "Shape": 1,
        "Sigmoid": 1,
        "Slice": 1,
        "Softmax": 1,
        "Squeeze": 1,
        "Sub": 1,
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

    def ReduceSumAnyOpset(self, *args, **kwargs):
        if len(args) == 1:
            return self.ReduceSum(*args, **kwargs)
        assert len(args) == 2, f"ReduceSumAnyOpset expects 2 arguments not {len(args)}"
        if self.builder.main_opset >= 13:
            return self.ReduceSum(*args, **kwargs)
        axes = args[1]
        if isinstance(axes, np.ndarray):
            iaxes = axes.tolist()
        elif isinstance(axes, int):
            iaxes = axes
        else:
            raise RuntimeError(
                f"Unable to call ReduceSum on a dynamic input axis={axes}"
            )
        return self.ReduceSum(args[0], axes=iaxes, **kwargs)


class OptimizationOptions:
    def __init__(
        self,
        remove_unused: bool = False,
        constant_folding: bool = True,
        constant_size: int = 1024,
    ):
        self.remove_unused = remove_unused
        self.constant_folding = constant_folding
        self.constant_size = constant_size

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(remove_unused={self.remove_unused}, "
            f"constant_folding={self.constant_folding}, "
            f"constant_size={self.constant_size})"
        )


class GraphBuilder:
    def _hash(self) -> str:
        return make_hash(self)

    def _apply_slice_to_shape(
        self,
        shape: Tuple[int],
        indices: List[slice],
        axes: List[int],
        expand_axes: List[int],
    ) -> Tuple[int]:
        assert isinstance(
            shape, tuple
        ), f"Unexpected type {type(shape)} for shape: {shape}"
        assert isinstance(
            indices, list
        ), f"Unexpected type {type(indices)} for index: {indices}"
        assert isinstance(axes, list), f"Unexpected type {type(axes)} for index: {axes}"
        assert len(indices) == len(
            axes
        ), f"Mismatch lengths {len(indices)} != {len(axes)}"
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
        for a in shape[len(new_shape) :]:
            assert a >= 0, (
                f"Negative value in shape {shape}, indices={indices}, "
                f"axes={axes}, expand_axes={expand_axes}"
            )
            new_shape.append(a)
        for e in expand_axes:
            new_shape.insert(e, 1)
        return tuple(new_shape)

    def __init__(
        self,
        target_opset_or_existing_proto: Union[
            int, Dict[str, int], ModelProto, FunctionProto
        ],
        input_names: Optional[Sequence[str]] = None,
        as_function: bool = False,
        optimization_options: Optional[OptimizationOptions] = None,
        args: Optional[List[Any]] = None,
        verbose: int = 0,
    ):
        self.optimization_options = optimization_options or OptimizationOptions()
        self.as_function = as_function
        self.input_args = args
        self.verbose = verbose
        self._debug_msg = {}

        if isinstance(target_opset_or_existing_proto, (int, dict)):
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
            if input_names:
                raise ValueError(
                    "input_names must be empty if the input is an existing model."
                )
            proto = target_opset_or_existing_proto
            self.opsets = {d.domain: d.version for d in proto.opset_import}
            self.nodes = list(proto.graph.node)
            self.initializers_dict = {i.name: i for i in proto.graph.initializer}
            self.initializers_dict.update(
                {i.name: i for i in proto.graph.sparse_initializer}
            )
            self.inputs = list(proto.graph.input)
            self.outputs = list(proto.graph.output)
            self.input_names = [i.name for i in proto.graph.input]
            self.current_input = len(self.inputs)
            # This should be improve.
            self._known_shapes = {}
            self._known_types = {}
            self._known_ranks = {}
            self._known_names = set()
            self.constants_ = {}
            self._unique_node_names = set()
            self._unique_names = set(self.input_names)

            for k, v in self.initializers_dict.items():
                self.constants_[k] = None
                self._unique_names.add(k)
                self.set_name(k)
                self.set_shape(k, self._get_tensor_shape(v))
                self.set_type(k, self._get_tensor_type(v))
            for node in self.nodes:
                self._unique_names |= set(node.output)
                if node.name:
                    self._unique_node_names.add(node.name)
                if node.op_type == "Constant":
                    self.constants_[node.output[0]] = node
                    self.set_name(node.output[0])
                    self.set_shape(node.output[0], self._get_tensor_shape(node))
                    self.set_type(node.output[0], self._get_tensor_type(node))
        else:
            raise NotImplementedError(
                f"{type(target_opset_or_existing_proto)} is not supported."
            )

        self.op = Opset(self, self.opsets[""])

    @property
    def main_opset(self):
        "Returns the opset for the main domain (assuming it is used)."
        return self.opsets[""]

    def _get_tensor_shape(
        self, proto: Union[NodeProto, TensorProto]
    ) -> Tuple[int, ...]:
        if isinstance(proto, TensorProto):
            return tuple(proto.dims)
        if isinstance(proto, NodeProto):
            for att in proto.attribute:
                if att.name == "value_float":
                    return tuple()
                if att.name == "value_int":
                    return tuple(att.i)
                if att.name == "value_floats":
                    return tuple(att.floats)
                if att.name == "value_ints":
                    return (len(att.ints),)
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

        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().numpy()
        raise TypeError(f"Unable to convert type {type(value)} into numpy array.")

    def set_name(self, name: str):
        assert isinstance(name, str), f"Unexpected type {type(name)} for name."
        assert name not in self._known_names, f"Name {name!r} already exists."
        self._known_names.add(name)

    def set_rank(self, name: str, value: int):
        assert isinstance(name, str), f"Unexpected type {type(name)} for name."
        assert name not in self._known_ranks, f"Name {name!r} already exists."
        self._known_ranks[name] = value

    def set_shape(self, name: str, shape: Tuple[int, ...], set_rank: bool = True):
        assert isinstance(name, str), f"Unexpected type {type(name)} for name."
        self._check_shape(shape, 0, name=name)
        assert (
            len(shape) == 0 or None in shape or min(shape) >= 0
        ), f"Negative value in shape {shape} for {name!r}{self.get_debug_msg()}"
        if name in self._known_shapes:
            if shape != self._known_shapes[name]:
                raise RuntimeError(
                    f"Name {name!r} already exists and it is different "
                    f"{self._known_shapes[name]} != {shape}{self.get_debug_msg()}"
                )
            return
        assert isinstance(shape, tuple), f"Unexpected shape type {type(shape)}."
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

    def has_shape(self, name: str) -> bool:
        assert isinstance(name, str), f"Unexpected type {type(name)} for name."
        return name in self._known_shapes

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
        if name == "":
            sh = "".join(map(str, value.shape))
            sh2 = (
                "x".join(map(str, value.ravel().tolist()))
                if value.size <= 5 and value.dtype == np.int64
                else ""
            )
            name = self.unique_name(f"init{sh}_{sh2}")
        self.set_shape(name, value.shape)
        self.set_type(name, self._get_type(value.dtype))
        self.set_name(name)
        self.initializers_dict[name] = value
        self.constants_[name] = None
        if self.verbose and (self.verbose > 1 or np.prod(value.shape) > 100):
            print(
                f"[GraphBuilder-{self._hash()}.make_initializer] {name}[{value.dtype}:{value.shape}]"
            )
        return name

    def make_tensor_input(
        self, name: str, elem_type: Any, shape: Tuple[int, ...]
    ) -> str:
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
        self.inputs.append(oh.make_tensor_value_info(input_name, elem_type, shape))
        if self.verbose:
            print(
                f"[GraphBuilder-{self._hash()}.make_tensor_input] {name}[{elem_type}:{shape}]"
            )
        if shape:
            self.set_shape(name, shape)
        if elem_type:
            self.set_type(name, elem_type)
        return name

    def make_tensor_output(
        self,
        name: Union[str, List[str]],
        elem_type: Optional[int] = None,
        shape: Optional[Tuple[int, ...]] = None,
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
        self._check_shape(shape, name=name, elem_type=elem_type)
        self.outputs.append(oh.make_tensor_value_info(name, elem_type, shape))
        if self.verbose:
            print(
                f"[GraphBuilder-{self._hash()}.make_tensor_output] {name}[{elem_type}:{shape}]"
            )
        if shape:
            self.set_shape(name, shape)
        if elem_type:
            self.set_type(name, elem_type)
        return name

    def _check_shape(self, shape: Any, elem_type: int, name: Optional[str] = None):
        assert isinstance(
            elem_type, int
        ), f"elem_type must be an integer not {type(elem_type)}"
        assert shape is None or isinstance(
            shape, tuple
        ), f"Shape must be a tuple not {type(shape)}"
        if shape is None:
            return None
        for s in shape:
            assert s is None or isinstance(s, (str, int)), (
                f"One element of shape={shape} has type {type(s)}, "
                f"name={name!r}, elem_type={elem_type}{self.get_debug_msg()}"
            )

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

    def make_node(
        self,
        op_type: str,
        inputs: Union[str, List[str]],
        outputs: Union[int, List[str], str] = 1,
        domain: str = "",
        attributes: Optional[List[AttributeProto]] = None,
        check: Optional[bool] = None,
        name: Optional[str] = None,
        set_shape_type: bool = False,
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
        self._make_node_set_shape_type_constant(node, set_shape_type=set_shape_type)

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

    def _make_node_set_shape_type_constant(self, node: NodeProto, set_shape_type: bool):
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
                self.set_shape(node.output[0], self._known_shapes[node.input[0]])
            if self.has_type(node.input[0]):
                self.set_type(node.output[0], self._known_types[node.input[0]])
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
                    self.set_shape(node.output[0], cst[0].shape)
        elif set_shape_type:
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
            self.set_shape(new_name, builder.get_shape(inp.name))
            self.set_type(new_name, builder.get_type(inp.name))

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
                    self.set_shape(no, builder.get_shape(o))
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
        import torch

        if not isinstance(arr, torch.Tensor):
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
        import torch

        res = []
        for k, v in sorted(self.initializers_dict.items()):
            if isinstance(v, torch.Tensor):
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

        rows = ["", "--DEBUG--"]
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
        if len(self.nodes) == 0:
            raise RuntimeError(
                f"The onnx model is empty after optimization (no node)."
                f"\n{self.get_debug_msg()}"
            )
        if as_function:
            raise NotImplementedError("Export as FunctionProto is not implemented yet.")
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
        model = oh.make_model(graph, opset_imports=opsets)
        if len(model.graph.node) == 0:
            raise RuntimeError(
                f"The onnx model is empty after export to onnx (no node)."
                f"\n{self.get_debug_msg()}"
            )
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

        _check("A")
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
                _check("D")

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

        if self.verbose:
            for k, v in self.initializers_dict.items():
                if k not in marked:
                    v = self.initializers_dict[k]
                    print(
                        f"[GraphBuilder.remove_unused] remove_initializer:{k}:{v.dtype}[{v.shape}]"
                    )
        self.initializers_dict = {
            k: v for k, v in self.initializers_dict.items() if k in marked
        }
        self.constants_ = {k: v for k, v in self.constants_.items() if k in marked}
        self.nodes = [node for i, node in enumerate(self.nodes) if i not in removed]

    def _apply_transpose(
        self, node: NodeProto, feeds: Dict[str, "torch.Tensor"]  # noqa: F821
    ) -> "torch.Tensor":  # noqa: F821
        import torch

        perm = None
        for att in node.attribute:
            if att.name == "perm":
                perm = tuple(att.ints)
                break
        assert perm, f"perm not here in node {node}"
        assert len(perm) == 2, f"perm={perm} is not supported with torch"
        x = feeds[node.input[0]]
        if isinstance(x, np.ndarray):
            x = torch.Tensor(x)
        return [torch.transpose(x, *perm)]

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
            ref = ReferenceEvaluator(v)
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
