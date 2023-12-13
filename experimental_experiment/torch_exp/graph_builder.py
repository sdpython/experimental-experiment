from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import onnx.helper as oh
import onnx.numpy_helper as onh
from onnx import FunctionProto, ModelProto, NodeProto, TensorProto
from onnx.reference import ReferenceEvaluator


class Opset:
    # defined for opset >= 18
    # name: number of expected outputs
    _implemented = {
        "Add": 1,
        "And": 1,
        "Cast": 1,
        "Constant": 1,
        "Div": 1,
        "Exp": 1,
        "Expand": 1,
        "Gemm": 1,
        "Identity": 1,
        "MatMul": 1,
        "MaxPool": 2,
        "Mul": 1,
        "Log": 1,
        "Or": 1,
        "Relu": 1,
        "Reshape": 2,
        "Slice": 1,
        "Squeeze": 1,
        "Sub": 1,
        "Transpose": 1,
        "Unsqueeze": 1,
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
            raise AttributeError(f"Unable to access attribute {name!r}.") from e

    def make_node(
        self,
        op_type: str,
        *inputs: Optional[Union[str, List[str]]],
        outputs: Optional[Union[int, List[str], str]] = None,
        domain: str = "",
        **kwargs,
    ):
        if outputs is None:
            outputs = self._implemented[op_type]
        if inputs is None:
            inputs = []
        return self.builder.make_node(
            op_type, inputs, outputs=outputs, domain=domain, **kwargs
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
    def __init__(
        self,
        target_opset_or_existing_proto: Union[
            int, Dict[str, int], ModelProto, FunctionProto
        ],
        input_names: Optional[Sequence[str]] = None,
        as_function: bool = False,
        optimization_options: Optional[OptimizationOptions] = None,
    ):
        self.optimization_options = optimization_options or OptimizationOptions()
        self.as_function = as_function

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
            self._unique_names = set()
            self.input_names = input_names or []
            self.current_input = 0
            self._known_shapes = {}
            self._known_types = {}
            self.constants_ = {}
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
            self.constants_ = {}
            for k, v in self.initializers_dict.items():
                self.constants_[k] = None
                self._known_shapes = self._get_tensor_shape(v)
                self._known_types = self._get_tensor_type(v)
            for node in self.nodes:
                if node.op_type == "Constant":
                    self.constants_[node.output[0]] = node
                    self._known_shapes = self._get_tensor_shape(node)
                    self._known_types = self._get_tensor_type(node)
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
        raise TypeError(f"Unexpected type {type(proto)}: {proto}.")

    def _get_tensor_type(self, proto: Union[NodeProto, TensorProto]) -> int:
        if isinstance(proto, TensorProto):
            return proto.data_type
        if isinstance(proto, NodeProto):
            for att in proto.attribute:
                if att.name == "value_float":
                    return TensorProto.FLOAT
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

    def set_shape(self, name: str, shape: Tuple[int, ...]):
        if not isinstance(name, str):
            raise TypeError(f"Unexpected type {type(name)} for name.")
        if name in self._known_shapes:
            raise RuntimeError(f"Name {name!r} already exists.")
        if not isinstance(shape, tuple):
            raise TypeError(f"Unexpected shape type {type(shape)}.")
        self._known_shapes[name] = shape

    def set_type(self, name: str, dtype: int):
        if not isinstance(name, str):
            raise TypeError(f"Unexpected type {type(name)} for name.")
        if name in self._known_types:
            raise RuntimeError(f"Name {name!r} already exists.")
        if not isinstance(dtype, int):
            raise TypeError(f"Unexpected dtype type {type(dtype)}.")
        self._known_types[name] = dtype

    def rank(self, name: str) -> int:
        if name not in self._known_shapes:
            raise ValueError(f"Shape is unknown for result {name!r}.")
        return len(self._known_shapes[name])

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
            elif "int64" in st:
                elem_type = TensorProto.INT64
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
        self.set_shape(name, value.shape)
        self.set_type(name, self._get_type(value.dtype))
        self.initializers_dict[name] = value
        self.constants_[name] = None
        return name

    def make_tensor_input(
        self, name: str, elem_type: Any, shape: Tuple[int, ...]
    ) -> str:
        if self.current_input < len(self.input_names):
            # The input needs to be renamed, an identity node is added.
            input_name = self.input_names[self.current_input]
            self.make_node("Identity", [input_name], [name])
        else:
            self.input_names.append(name)
            input_name = name
        self.current_input += 1
        elem_type = self._get_type(elem_type)
        self.inputs.append(oh.make_tensor_value_info(input_name, elem_type, shape))
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
        if elem_type == 0:
            raise RuntimeError(f"Undefined element type for {name!r}.")
        self.outputs.append(oh.make_tensor_value_info(name, elem_type, shape))
        if shape:
            self.set_shape(name, shape)
        if elem_type:
            self.set_type(name, elem_type)
        return name

    def make_node(
        self,
        op_type: str,
        inputs: Union[str, List[str]],
        outputs: Union[int, List[str], str] = 1,
        domain: str = "",
        **kwargs,
    ) -> Union[str, List[str]]:
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

        # next
        try:
            node = oh.make_node(op_type, inputs, output_names, domain=domain, **kwargs)
        except TypeError as e:
            iti = [type(i) for i in inputs]
            ito = [type(o) for o in outputs]
            raise TypeError(
                f"A node {op_type!r} cannot be created with "
                f"inputs={inputs} (types={iti}), oututs={outputs} (types={ito}), "
                f"domain={domain!r}, kwargs={kwargs}."
            ) from e

        # constant handling
        if node.op_type == "Constant":
            size = len(node.SerializeToString())
            if size >= self.optimization_options.constant_size:
                raise ValueError(
                    f"A node Constant holds a tensor bigger than "
                    f"the constant: {size} >= {self.constant_size}."
                )
            self.constants_[node.output[0]] = node
        else:
            if all(map(self.is_constant, node.input)):
                for o in node.output:
                    self.constants_[o] = node

        # add the node
        self.nodes.append(node)
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
            np_arr = np.from_dlpack(arr_cpu)

        tensor = TensorProto()
        tensor.dims.extend(arr_cpu.shape)
        tensor.name = name
        tensor.data_type = self._get_type(arr_cpu.dtype)

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
                t = onh.from_array(v, name=k)
                res.append(t)
                continue
            raise TypeError(
                f"Unable to convert initializer {k!r} with type "
                f"{type(v)} into a TensorProto."
            )
        return res

    def process(
        self,
        graph_module: "torch.f.GraphModule",  # noqa: F821
        interpreter: "Interpreter",  # noqa: F821
    ):
        for node in graph_module.graph.nodes:
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

        graph = oh.make_graph(
            self.nodes, "experiment", self.inputs, self.outputs, dense
        )
        model = oh.make_model(graph, opset_imports=opsets)
        return model

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

        self.initializers_dict = {
            k: v for k, v in self.initializers_dict.items() if k in marked
        }
        self.constants_ = {k: v for k, v in self.constants_.items() if k in marked}
        self.nodes = [node for i, node in enumerate(self.nodes) if i not in removed]

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
            if all(map(self.is_constant, v.output)):
                node_to_remove.add(tuple(v.output))
                # node evaluation
                ref = ReferenceEvaluator(v)
                feeds = {i: self.get_constant(i) for i in v.input}
                output = ref.run(None, feeds)
                for name, value in zip(v.output, output):
                    updates[name] = None
                    self.initializers_dict[name] = value

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
        # f<irst pass: detect replacements
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
            elif node.input[0] not in input_names:
                old_name, new_name = node.input[0], node.output[0]
            else:
                new_nodes.append(node)
                continue

            # the new name can be set for replacements as well
            assert old_name not in replacements
            if new_name in replacements:
                new_name = replacements[new_name]
                assert new_name not in replacements
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
                new_node = oh.make_node(
                    node.op_type,
                    [replacements.get(i, i) for i in node.input],
                    [replacements.get(i, i) for i in node.output],
                    domain=node.domain,
                    name=node.name,
                )
                new_node.attribute.extend(node.attribute)
                self.nodes.append(new_node)
            else:
                self.nodes.append(node)
