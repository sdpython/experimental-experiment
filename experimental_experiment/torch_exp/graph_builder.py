from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import onnx.helper as oh
import onnx.numpy_helper as onh
from onnx import AttributeProto, FunctionProto, ModelProto, NodeProto, TensorProto
from onnx.reference import ReferenceEvaluator
from ._helper import make_hash


class Opset:
    # defined for opset >= 18
    # name: number of expected outputs
    _implemented = {
        "Add": 1,
        "And": 1,
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
        "Gemm": 1,
        "Identity": 1,
        "MatMul": 1,
        "MaxPool": 2,
        "Mul": 1,
        "Less": 1,
        "Log": 1,
        "Or": 1,
        "Range": 1,
        "ReduceMin": 1,
        "Relu": 1,
        "Reshape": 2,
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
            if isinstance(i, str):
                new_inputs.append(i)
            elif hasattr(i, "name"):
                # torch.fx.Node
                new_inputs.append(i.name)
            else:
                cst_name = self.builder.unique_name("cst")
                self.builder.make_initializer(cst_name, i)
                new_inputs.append(cst_name)

        return self.builder.make_node(
            op_type, new_inputs, outputs=outputs, domain=domain, name=name, **kwargs
        )


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
            new_shape.append(dim)
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

    def get_constant(self, name: str) -> np.ndarray:
        if not self.is_constant(name):
            raise ValueError(f"Result {name!r} is not a constant.")
        if name not in self.initializers_dict:
            raise ValueError(
                f"Result {name!r} was never evaluated within method 'constant_folding'."
            )
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
        if name in self._known_shapes:
            if shape != self._known_shapes[name]:
                raise RuntimeError(
                    f"Name {name!r} already exists and it is different "
                    f"{self._known_shapes[name]} != {shape}"
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
            elif "int64" in st:
                elem_type = TensorProto.INT64
            elif "bool" in st:
                elem_type = TensorProto.BOOL
            elif elem_type is None:
                elem_type = TensorProto.UNDEFINED
            elif exc:
                raise ValueError(f"Unable to interpret elem_type {elem_type!r}.")
        return elem_type

    def make_initializer(self, name: str, value: Any, external: bool = False) -> str:
        if external:
            raise NotImplementedError("External initializers are not implemented yet.")
        if name == "":
            name = self.unique_name("cst")
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
            raise RuntimeError(f"Unexpected type {type(value)} for value={value!r}.")
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
    ) -> Union[str, List[str]]:
        if isinstance(name, list):
            res = []
            for n in name:
                res.append(self.make_tensor_output(n, elem_type, shape))
            return res

        elem_type = self._get_type(elem_type, False)
        if not self.as_function and elem_type == 0:
            raise RuntimeError(f"Undefined element type for {name!r}.")
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

    def _debug_string_inputs(self, inputs, align=None):
        st = ""
        c = "-TRUSVW#"
        for i in inputs:
            k = 0
            if self.has_type(i):
                k += 1
            if self.has_rank(i):
                k += 2
            if self.has_shape(i):
                k += 4
            st += c[k]
        if align and len(st) < align:
            st += " " * (align - len(st))
        return st

    def make_node(
        self,
        op_type: str,
        inputs: Union[str, List[str]],
        outputs: Union[int, List[str], str] = 1,
        domain: str = "",
        attributes: Optional[List[AttributeProto]] = None,
        check: Optional[bool] = None,
        name: Optional[str] = None,
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
                f"[{self._debug_string_inputs(inputs)}] "
                f"{op_type}:{inputs}->{outputs}"
            )

        if check is not False:
            for i in inputs:
                assert self.has_name(
                    i
                ), f"Input {i!r} does not exist for operator {op_type!r} ({self._hash()})."
            for i in output_names:
                assert not self.has_name(
                    i
                ), f"Output {i!r} already exists for operator {op_type!r} ({self._hash()})."
        if check is True:
            for i in inputs:
                assert self.has_shape(i), f"Input {i!r} has no known shape."
                assert self.has_type(i), f"Input {i!r} has no known type."

        if name:
            name = self.unique_node_name(name)

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
            if node.input[0] in self._known_shapes:
                self.set_shape(node.output[0], self._known_shapes[node.input[0]])
            if node.input[0] in self._known_types:
                self.set_type(node.output[0], self._known_types[node.input[0]])
            if self.is_constant(node.input[0]):
                self.constants_[node.output[0]] = node
        else:
            if all(map(self.is_constant, node.input)):
                for o in node.output:
                    self.constants_[o] = node
        if self.verbose > 3:
            print(
                f"[GraphBuilder-{self._hash()}.make_node] "
                f"[{self._debug_string_inputs(node.input)}] "
                f"{node.op_type}:{node.input}->{node.output}"
            )

        # add the node
        for o in node.output:
            self.set_name(o)
        self.nodes.append(node)
        if len(output_names) == 1:
            return output_names[0]
        return output_names

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

        rows = ["", "--DEBUG--"]
        for k, v in self._debug_msg.items():
            rows.append(f"-- {k}")
            rows.append(str(v))
        rows.append("--")
        hs = self._hash()
        for io in self.inputs:
            shh = str(io.type.tensor_type.shape).replace("\n", "")
            rows.append(
                f"[GraphBuilder-{hs}.make_tensor_input] {io.name}"
                f"[{io.type.tensor_type.elem_type}:{shh}]"
            )
        for name, init in self.initializers_dict.items():
            rows.append(
                f"[GraphBuilder-{hs}.make_initializer] {name}[{init.dtype}:{init.shape}]"
            )
        for node in self.nodes:
            rows.append(
                f"[GraphBuilder-{hs}.make_node] "
                f"{_align(node.name, 15)} [{self._debug_string_inputs(node.input, 4)}] "
                f"{node.op_type}:{node.input}->{node.output}"
            )
        for io in self.outputs:
            shh = str(io.type.tensor_type.shape).replace("\n", "")
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
        for i, node in enumerate(graph_module.graph.nodes):
            self._debug_msg[
                "process.progress"
            ] = f"node {i}/{len(graph_module.graph.nodes)} "
            interpreter.run_node(node)

    def to_onnx(
        self, as_function: bool = False, optimize: bool = True
    ) -> Union[FunctionProto, ModelProto]:
        if optimize:
            self.optimize()
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
        self.remove_identity_nodes()
        if self.optimization_options.remove_unused:
            self.remove_unused()
        if self.optimization_options.constant_folding:
            self.constant_folding()
            if self.optimization_options.remove_unused:
                self.remove_unused()

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
        return [torch.transpose(feeds[node.input[0]], *perm)]

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
                if v.op_type == "Transpose":
                    # bypassing onnx.numpy_helper.from_array, too slow
                    feeds = {i: self.initializers_dict[i] for i in v.input}
                    output = self._apply_transpose(v, feeds)
                else:
                    ref = ReferenceEvaluator(v)
                    feeds = {i: self.get_constant(i) for i in v.input}
                    output = ref.run(None, feeds)
                for name, value in zip(v.output, output):
                    updates[name] = None
                    self.initializers_dict[name] = value
                    if self.verbose:
                        print(
                            f"[GraphBuilder.constant_folding] fold_constant:{v.op_type}:{name}[{value.dtype}:"
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
        for node in self.nodes:
            if node.op_type != "Identity":
                new_nodes.append(node)
                continue

            if node.output[0] not in output_names:
                old_name, new_name = node.output[0], node.input[0]
            elif node.input[0] not in input_names and node.input[0] not in replacements:
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
            if old_name in replacements:
                replacements[replacements[old_name]] = new_name
            replacements[old_name] = new_name

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
                assert "output_1" not in new_inputs
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
