from collections import OrderedDict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import onnx
from ...onnx_function import OnnxFunction
from ...evaluator import Evaluator
from ...tensor import Tensor
from ...values import ParamSchema, _EmptyDefault

torch_dtype_Type = "torch.dtype"
torch_Graph_Type = "torch.Graph"
torch_Node_Type = "torch.Node"
torch_Size_Type = "torch.Size"
torch_Tensor_Type = "torch.Tensor"
torch_Value_Type = "torch.Value"


class TorchScriptTensor(Tensor):
    def __init__(self, value: torch_Value_Type, opset=None):  # noqa: F821
        super().__init__(None, opset=opset)
        self._torch_value = value
        self._name: Optional[str] = None
        self._torch_dtype = None
        self._shape = None
        if value is not None:
            try:
                self._torch_dtype = value.dtype
            except AttributeError as e:
                try:
                    self._torch_dtype = value.type
                except AttributeError:
                    raise AttributeError(
                        f"Unable to get dtype from type {type(value)} among {dir(value)}."
                    ) from e

    def __repr__(self):
        return f"TorchScriptTensor('{self._torch_value!r}')"

    @property
    def name(self) -> str:
        if self._name is not None:
            return self._name
        return self._torch_value.debugName()

    @name.setter
    def name(self, name: str):
        self._name = name
        self._torch_value.setDebugName(name)

    @property
    def shape(self) -> Tuple[int | str | None, ...] | None:
        if self._shape is not None:
            return self._shape

        value_type = self._torch_value.type()
        if value_type is None:
            return None
        import torch

        if not issubclass(value_type, torch.TensorType):
            raise TypeError(f"Unexpected type {value_type}.")
        if isinstance(value_type, torch.OptionalType):
            shape = value_type.getElementType().varyingSizes()
        else:
            shape = value_type.varyingSizes()
        if shape is None:
            return None
        return tuple(shape)

    @shape.setter
    def shape(self, shape: Union[torch_Size_Type, Tuple[int | str | None, ...]]):
        import torch

        torch_sym_types = (torch.SymInt, torch.SymFloat, torch.SymBool)
        self._shape = tuple(
            str(dim.node) if isinstance(dim, torch_sym_types) else dim for dim in shape
        )
        jit_shape = tuple(dim if isinstance(dim, int) else None for dim in shape)
        self._torch_value.setType(self._torch_value.type().with_sizes(list(jit_shape)))

    @property
    def dtype(self) -> Optional[torch_dtype_Type]:
        if self._torch_dtype is not None:
            return self._torch_dtype
        from torch.onnx import _type_utils

        torch_dtype = _type_utils.JitScalarType.from_value(
            self._torch_value, default=_type_utils.JitScalarType.UNDEFINED
        )
        if torch_dtype == _type_utils.JitScalarType.UNDEFINED:
            return None
        self._torch_dtype = torch_dtype.dtype()
        return self._torch_dtype

    @dtype.setter
    def dtype(self, dtype: torch_dtype_Type):
        if hasattr(self._torch_value, "dtype"):
            raise NotImplementedError("Unable to set type.")
        self._torch_dtype = dtype
        self._torch_value.setType(self._torch_value.type().with_dtype(dtype))

    def get_evaluator(self) -> Evaluator:
        from ...evaluator import default

        obj = default()
        if not isinstance(obj, TorchScriptTracingEvaluator):
            raise RuntimeError(
                f"Unexpected type {type(obj)} for the default evaluator."
            )
        return obj

    def symbolic_value(self) -> torch_Value_Type:
        """The symbolic Value in torch.Graph."""
        return self._torch_value


class TorchScriptGraph:
    def __init__(
        self,
        parent_torch_script_graph: Optional["TorchScriptGraph"] = None,
        domain_name: Optional[str] = None,
        opsets=None,
    ):
        import torch

        self._torch_graph = torch.Graph()
        self._function_store: Dict[Tuple[str, str], OnnxFunction] = {}
        self._initializers: Dict[str, torch.Tensor] = {}
        self._initializers_inputs: Dict[str, TorchScriptTensor] = {}
        self._initializers_inputs_from_parent: Dict[str, TorchScriptTensor] = {}
        self._sub_torch_script_graphs: Dict[str, TorchScriptGraph] = {}
        self._parent_torch_script_graph = parent_torch_script_graph
        self._domain_name: Optional[str] = domain_name
        self._value_to_tensor: Dict[torch.Value, TorchScriptTensor] = {}
        if opsets is None:
            from ...onnx_opset import Opset18

            self._opsets = {"": Opset18}
        else:
            self._opsets = opsets

        if self._domain_name is None and self._parent_torch_script_graph is not None:
            raise RuntimeError(
                "Domain name is not set. It is required because this "
                "'TorchScriptGraph' instance "
                "is a subgraph that represents an ONNX local function."
            )

    @property
    def torch_graph(self):
        return self._torch_graph

    @property
    def initializers(self) -> Mapping[str, torch_Tensor_Type]:
        return self._initializers

    # NOTE: This setter is used in torch converter when we activate fake mode,
    #       we need to filter out the initializers that has fake tensor. This
    #       is because we don't want to introduce fake tensor in onnxscript.
    @initializers.setter
    def initializers(self, initializers: Dict[str, torch_Tensor_Type]):
        self._initializers = initializers

    @property
    def initializers_inputs(self) -> Mapping[str, TorchScriptTensor]:
        return self._initializers_inputs

    @property
    def initializers_inputs_from_parent(self) -> Mapping[str, TorchScriptTensor]:
        return self._initializers_inputs_from_parent

    @property
    def num_outputs(self) -> int:
        return len(list(self._torch_graph.outputs()))

    @property
    def domain_name(self) -> Optional[str]:
        return self._domain_name

    def _wrap_torch_value_to_tensor(
        self,
        value: Union[torch_Value_Type, Mapping[str, Any], Sequence[Any]],
        *,
        shape: Optional[
            Union[torch_Size_Type, Tuple[Union[int, str, None], ...]]
        ] = None,
        dtype: Optional[torch_dtype_Type] = None,
    ) -> Any:
        import torch

        if isinstance(value, torch.Value):
            tensor = TorchScriptTensor(value, opset=self._opsets[""])
            if shape is not None:
                tensor.shape = shape
            if dtype is not None:
                tensor.dtype = dtype
            return tensor
        if isinstance(value, dict):
            return {k: self._wrap_torch_value_to_tensor(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._wrap_torch_value_to_tensor(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self._wrap_torch_value_to_tensor(v) for v in value)
        return value

    def add_input(
        self,
        input_name: Optional[str],
        shape: Optional[
            Union[torch_Size_Type, Tuple[Union[int, str, None], ...]]
        ] = None,
        dtype: Optional[torch_dtype_Type] = None,
    ) -> TorchScriptTensor:
        import torch

        if input_name is None:
            # This input argument is None, which is mapped
            # to a NULL value in TorchScript type system.
            torch_value = self._create_op_call_in_torch_graph(
                self._torch_graph, "prim::Constant", inputs=(), attributes={}
            )[0]
            torch_value.setType(torch.OptionalType.ofTensor())
        else:
            torch_value = self._torch_graph.addInput(input_name)
            torch_value.setType(torch_value.type().with_dtype(dtype))
            # TODO(titaiwang): This approach loses the information that "same SymInts
            # indicates same shape", for example, [symint0, symint0, symint1]
            # would all be [None, None, None]
            torch_value.setType(
                torch_value.type().with_sizes(
                    [dim if isinstance(dim, int) else None for dim in shape]
                )
            )
        tensor_value = self._wrap_torch_value_to_tensor(
            torch_value, shape=shape, dtype=dtype
        )
        if isinstance(tensor_value, TorchScriptTensor):
            # NOTE: Only track value that maps to tensor.
            # Value that maps to Sequence/Dict of tensors is not tracked.
            self._value_to_tensor[torch_value] = tensor_value
        return tensor_value

    def add_initializer(self, name: str, value: torch_Tensor_Type) -> TorchScriptTensor:
        if name in self._initializers_inputs:
            if name in self._initializers and self._initializers[name] is not value:
                raise ValueError(
                    f"Initializer {name!r} exists already with a different value."
                )
            return self._initializers_inputs[name]

        import torch

        if (
            self != self._parent_torch_script_graph
            and self._parent_torch_script_graph is not None
        ):
            self._initializers_inputs_from_parent[
                name
            ] = self._parent_torch_script_graph.add_initializer(name, value)
        else:
            self._initializers[name] = value

        torch_value = self._torch_graph.addInput(name)
        torch_value.setType(torch.TensorType.create_from_tensor(value))
        tensor_value = self._wrap_torch_value_to_tensor(
            torch_value, shape=value.shape, dtype=value.dtype
        )
        if isinstance(tensor_value, TorchScriptTensor):
            self._value_to_tensor[torch_value] = tensor_value
        self._initializers_inputs[name] = tensor_value
        return tensor_value

    def _unwrap_tensor_to_torch_value(
        self,
        value: Union[Any, Mapping[str, Any], Sequence[Any]],
    ) -> Union[Any, Dict[str, Any], List[Any], Tuple[Any, ...]]:
        if isinstance(value, TorchScriptTensor):
            return value.symbolic_value()
        if isinstance(value, dict):
            return {k: self._unwrap_tensor_to_torch_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._unwrap_tensor_to_torch_value(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self._unwrap_tensor_to_torch_value(v) for v in value)

        return value

    def _unwrap_tensors_to_torch_values(self, tensors):
        if isinstance(tensors, Sequence):
            return [self._unwrap_tensor_to_torch_value(output) for output in tensors]
        return self._unwrap_tensor_to_torch_value(tensors)

    def register_outputs(
        self, outputs: Union[TorchScriptTensor, Tuple[TorchScriptTensor, ...]]
    ):
        import torch

        unwrapped_outputs = self._unwrap_tensors_to_torch_values(outputs)
        if isinstance(unwrapped_outputs, torch.Value):
            self._torch_graph.registerOutput(unwrapped_outputs)
            return
        for ts_output in unwrapped_outputs:
            assert isinstance(
                ts_output, torch.Value
            ), f"ts_output must be a torch.Value, not {type(ts_output)}"
            self._torch_graph.registerOutput(ts_output)

    def _add_constant_to_graph(self, constant) -> torch_Value_Type:
        import torch

        if constant is None:
            value = self._create_op_call_in_torch_graph(
                self._torch_graph, "prim::Constant", inputs=(), attributes={}
            )[0]
            value.setType(torch.OptionalType.ofTensor())
            value.setDebugName(self._rename_intermediate_value(value.debugName()))
            return value

        if isinstance(constant, (bool, float, int, tuple, list)):
            # Be sure to put bool before int, because bool is a subclass of int
            # constant_tensor = torch.tensor(constant, dtype=torch.bool)
            raise NotImplementedError(
                f"Unable to create a constant without ambiguity "
                f"for type={type(constant)}={constant}."
            )
        value = self._create_op_call_in_torch_graph(
            self._torch_graph,
            "onnx::Constant",
            inputs=(),
            #  attributes=dict(value=constant_tensor),
        )[0]
        value.setDebugName(self._rename_intermediate_value(value.debugName()))
        return value

    def fetch_function_proto_dict(
        self, opset_version: int
    ) -> Mapping[Tuple[str, str], onnx.FunctionProto]:
        function_proto_dict: Dict[Tuple[str, str], onnx.FunctionProto] = {}
        # Fetch local function protos. E.g., local functions representing module calls.
        for (
            sub_graph_name,
            sub_torch_script_graph,
        ) in self._sub_torch_script_graphs.items():
            function_proto_dict.update(
                sub_torch_script_graph.fetch_function_proto_dict(opset_version)
            )
            domain = sub_torch_script_graph.domain_name
            assert domain is not None
            name_domain = (
                sub_graph_name,
                domain,
            )
            assert (
                name_domain not in function_proto_dict
            ), f"Sub graph name already exists. {name_domain}"
            function_proto_dict[name_domain] = sub_torch_script_graph.to_function_proto(
                opset_version, sub_graph_name
            )
        # Fetch torchlib function protos.
        for name_domain, function in self._function_store.items():
            function_proto_dict[name_domain] = function.to_function_proto()
        return function_proto_dict

    def to_model_proto(self, opset_version: int) -> onnx.ModelProto:
        import torch

        # function_proto_dict: Mapping[
        #    Tuple[str, str], onnx.FunctionProto
        # ] = self.fetch_function_proto_dict(opset_version)
        unique_custom_domains: Dict[str, int] = {}

        export_kwargs = dict(
            initializers=self.initializers,
            onnx_opset_version=opset_version,
            dynamic_axes={},
            defer_weight_export=False,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
            strip_doc_string=False,
            keep_initializers_as_inputs=True,
            custom_opsets={},
            add_node_names=True,
            node_attr_to_name={},
        )

        (
            proto,
            _,
            _,
            _,
        ) = self._torch_graph._export_onnx(**export_kwargs)
        onnx_model = onnx.load_from_string(proto)

        unique_custom_domains = {d.domain: d.version for d in onnx_model.opset_import}
        onnx_model.opset_import.extend(
            [
                onnx.helper.make_opsetid(domain, version)
                for domain, version in unique_custom_domains.items()
            ]
        )

        onnx.checker.check_model(onnx_model)
        return onnx_model

    def add_function_call(
        self,
        onnx_function: OnnxFunction,
        onnx_inputs: Sequence[Any],
        onnx_attributes: Mapping[str, Any],
    ) -> Union[TorchScriptTensor, Tuple[TorchScriptTensor, ...]]:
        identifier = (onnx_function.name, onnx_function.domain)
        self._function_store[identifier] = onnx_function

        result = self._add_torchscript_op_call(
            f"{onnx_function.domain}::{onnx_function.name}",
            onnx_inputs,
            onnx_attributes,
            n_outputs=len(onnx_function.output_names),
        )

        return result

    def _add_attribute_to_torchscript_node(
        self,
        node: torch_Node_Type,
        key: str,
        value: Union[
            float, int, str, bytes, Sequence[float], Sequence[int], torch_Tensor_Type
        ],
    ):
        import torch

        if isinstance(value, float):
            return node.f_(key, value)
        if isinstance(value, int):
            return node.i_(key, value)
        if isinstance(value, (str, bytes)):
            return node.s_(key, value)
        if isinstance(value, torch.Tensor):
            return node.t_(key, value)
        if isinstance(value, Sequence):
            if not value:
                return node.is_(key, list(value))
            if isinstance(value[0], float):
                return node.fs_(key, list(value))
            if isinstance(value[0], int):
                return node.is_(key, list(value))
            raise TypeError(
                f"Unsupported sequence type '{type(value)}' for attribute '{key}'"
            )
        raise TypeError(
            f"Unsupported attribute type '{type(value)}' for attribute '{key}'"
        )

    def _create_op_call_in_torch_graph(
        self,
        graph: torch_Graph_Type,
        opname: str,
        *,
        inputs: Sequence[torch_Value_Type],
        attributes: Mapping[str, Any],
        n_outputs: int = 1,
    ) -> Tuple[torch_Value_Type, ...]:
        attributes = {k: v for k, v in attributes.items() if v is not None}

        node = graph.create(opname, inputs, n_outputs)
        node = graph.insertNode(node)
        node_ouputs = tuple(node.outputs())

        assert len(node_ouputs) == n_outputs
        # Add all attributes
        for key, value in sorted(attributes.items()):
            self._add_attribute_to_torchscript_node(node, key, value)

        return node_ouputs

    def add_module_call(
        self,
        name: str,
        sub_torch_script_graph: "TorchScriptGraph",
        onnx_inputs: Sequence[Any],
    ) -> Union[TorchScriptTensor, Tuple[TorchScriptTensor, ...]]:
        self._sub_torch_script_graphs[name] = sub_torch_script_graph
        domain_name = sub_torch_script_graph.domain_name
        assert domain_name is not None
        return self._add_torchscript_op_call(
            f"{domain_name}::{name}",
            onnx_inputs=(
                *onnx_inputs,
                *sub_torch_script_graph.initializers_inputs_from_parent.values(),
            ),
            onnx_attributes={},
            n_outputs=sub_torch_script_graph.num_outputs,
        )

    def _rename_intermediate_value(self, name: str) -> str:
        if name.isdigit():
            return f"_val_{name}"
        return name

    def _add_torchscript_op_call(
        self,
        name: str,
        onnx_inputs: Sequence[Any],
        onnx_attributes: Mapping[str, Any],
        n_outputs: int,
    ) -> Union[TorchScriptTensor, Tuple[TorchScriptTensor, ...]]:
        import torch

        unwrapped_inputs = self._unwrap_tensors_to_torch_values(onnx_inputs)
        graph_inputs = []
        assert isinstance(unwrapped_inputs, Sequence)
        for input in unwrapped_inputs:
            # NOTE(titaiwang): input could be empty list
            if (
                isinstance(input, Sequence)
                and input
                and all(isinstance(elem, torch.Value) for elem in input)
            ):
                # If all elements in the Sequence are torch.Values we know it
                # should be a Sequence input in ONNX.
                input_sequence = self._create_op_call_in_torch_graph(
                    self._torch_graph,
                    "onnx::SequenceConstruct",
                    inputs=input,
                    attributes={},
                )[0]
                graph_inputs.append(input_sequence)
            elif not isinstance(input, torch.Value):
                graph_inputs.append(self._add_constant_to_graph(input))
            else:
                graph_inputs.append(input)
        for key, value in onnx_attributes.items():
            assert not isinstance(
                value, TorchScriptTensor
            ), f"ONNX attribute must not be a TorchScriptTensor, got {key}: {value}."
        result = self._create_op_call_in_torch_graph(
            self._torch_graph,
            name,
            inputs=graph_inputs,
            attributes=onnx_attributes,
            n_outputs=n_outputs,
        )
        assert result, "Expected at least one output from ONNX op call."
        # NOTE: TorchScriptTensor is created here, however neither dtype nor shape is
        # set. It is expected that exporter will modify the tensor being returned and
        # set these info.
        if len(result) == 1:
            tensor = TorchScriptTensor(result[0], opset=self._opsets[""])
            tensor.name = self._rename_intermediate_value(tensor.name)
            self._value_to_tensor[result[0]] = tensor
            return tensor
        tensors = tuple(TorchScriptTensor(v) for v in result)
        self._value_to_tensor.update(dict(zip(result, tensors)))
        for tensor in tensors:
            tensor.name = self._rename_intermediate_value(tensor.name)
        return tensors


class TorchScriptTracingEvaluator(Evaluator):
    def __init__(self, graph: TorchScriptGraph):
        self._graph: TorchScriptGraph = graph

    @property
    def graph(self) -> TorchScriptGraph:
        return self._graph

    def eval(self, schema, inputs, attributes):
        return self._graph.add_op_call(schema, inputs, attributes)

    def separate_input_attributes_from_arguments(
        self,
        param_schemas: Sequence[ParamSchema],
        args,
        kwargs,
        fill_defaults: bool = True,
        allow_extra_kwargs: bool = False,
    ) -> tuple[list[Any], OrderedDict[str, Any]]:
        all_param_names = {param.name for param in param_schemas}
        extra_kwargs = set(kwargs).difference(all_param_names)
        if extra_kwargs and not allow_extra_kwargs:
            raise TypeError(f"Unexpected keyword arguments '{extra_kwargs}'")

        onnx_inputs = []
        onnx_attributes = OrderedDict()

        for i, param in enumerate(param_schemas):
            if param.is_variadic_input:
                # Exhaust all remaining args
                onnx_inputs.extend(args[i:])
                args = []
                continue
            if i < len(args):
                if param.is_input:
                    onnx_inputs.append(args[i])
                else:
                    onnx_attributes[param.name] = args[i]
            elif param.name in kwargs:
                if param.is_input:
                    onnx_inputs.append(kwargs[param.name])
                else:
                    onnx_attributes[param.name] = kwargs[param.name]
            elif param.is_attribute and param.default is not _EmptyDefault:
                # User did not provide the attribute
                if fill_defaults:
                    onnx_attributes[param.name] = param.default
            elif param.required:
                raise TypeError(f"Required input '{param}' was not provided")

        return onnx_inputs, onnx_attributes

    def eval_function(
        self,
        function: OnnxFunction,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
    ):
        # args/kwargs are TorchScriptTensor/python built-in based
        param_schemas = function.param_schemas()
        (
            inputs,
            attributes,
        ) = self.separate_input_attributes_from_arguments(
            param_schemas, args, kwargs, fill_defaults=True, allow_extra_kwargs=True
        )
        # name_to_schema = {param.name: param for param in param_schemas}
        return self._graph.add_function_call(function, inputs, attributes)
