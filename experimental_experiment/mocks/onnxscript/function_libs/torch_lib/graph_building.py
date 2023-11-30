from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union
import onnx
from ...onnx_function import OnnxFunction
from ...evaluator import Evaluator
from ...tensor import Tensor

torch_dtype_Type = "torch.dtype"
torch_Size_Type = "torch.Size"
torch_Tensor_Type = "torch.Tensor"
torch_Value_Type = "torch.Value"


class TorchScriptTensor(Tensor):
    def __init__(self, value: torch_Value_Type, opset=None):  # noqa: F821
        super().__init__(None, opset=opset)
        self._torch_value = value

    def __repr__(self):
        return f"TorchScriptTensor('{self._torch_value!r}')"

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
    def dtype(self, dtype: "torch_dtype_Type"):
        self._torch_dtype = dtype
        self._torch_value.setType(self._torch_value.type().with_dtype(dtype))


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
            torch_value.setType(torch_value.type().with_dtype(dtype))  # type: ignore[arg-type]
            # TODO(titaiwang): This approach loses the information that "same SymInts
            # indicates same shape", for example, [symint0, symint0, symint1]
            # would all be [None, None, None]
            torch_value.setType(
                torch_value.type().with_sizes(
                    [dim if isinstance(dim, int) else None for dim in shape]  # type: ignore[union-attr]
                )
            )
        tensor_value = self._wrap_torch_value_to_tensor(
            torch_value, shape=shape, dtype=dtype
        )
        if isinstance(tensor_value, TorchScriptTensor):
            # NOTE: Only track value that maps to tensor.
            # Value that maps to Sequence/Dict of tensors is not tracked.
            self._value_to_tensor[torch_value] = tensor_value
        return tensor_value  # type: ignore[return-value]

    def add_initializer(self, name: str, value: torch_Tensor_Type) -> TorchScriptTensor:
        if name in self._initializers_inputs:
            if name in self._initializers and self._initializers[name] is not value:
                raise ValueError(
                    f"Initializer {name!r} exists already with a different value."
                )
            return self._initializers_inputs[name]  # type: ignore[return-value]

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
        self._initializers_inputs[name] = tensor_value  # type: ignore[assignment]
        return tensor_value  # type: ignore[return-value]

    def register_outputs(
        self, outputs: Union[TorchScriptTensor, Tuple[TorchScriptTensor, ...]]
    ):
        import torch

        unwrapped_outputs = _unwrap_tensors_to_torch_values(outputs)
        if isinstance(unwrapped_outputs, torch.Value):
            self._torch_graph.registerOutput(unwrapped_outputs)
            return
        for ts_output in unwrapped_outputs:
            assert isinstance(
                ts_output, torch.Value
            ), f"ts_output must be a torch.Value, not {type(ts_output)}"
            self._torch_graph.registerOutput(ts_output)
        return

    def _add_constant_to_graph(self, constant) -> torch_Value_Type:
        import torch

        if constant is None:
            value = _create_op_call_in_torch_graph(
                self._torch_graph, "prim::Constant", inputs=(), attributes={}
            )[0]
            value.setType(torch.OptionalType.ofTensor())
            value.setDebugName(_rename_intermediate_value(value.debugName()))
            return value

        if isinstance(constant, (bool, float, int, tuple, list)):
            # Be sure to put bool before int, because bool is a subclass of int
            # constant_tensor = torch.tensor(constant, dtype=torch.bool)
            raise NotImplementedError(
                f"Unable to create a constant without ambiguity "
                f"for type={type(constant)}={constant}."
            )
        value = _create_op_call_in_torch_graph(
            self._torch_graph,
            "onnx::Constant",
            inputs=(),
            attributes=dict(value=constant_tensor),
        )[0]
        value.setDebugName(_rename_intermediate_value(value.debugName()))
        return value

    def to_model_proto(self, opset_version: int) -> onnx.ModelProto:
        import torch

        function_proto_dict: Mapping[
            Tuple[str, str], onnx.FunctionProto
        ] = self.fetch_function_proto_dict(opset_version)
        unique_custom_domains: Dict[str, int] = {}

        export_kwargs = dict(
            initializers=self.initializers,
            onnx_opset_version=opset_version,
            dynamic_axes={},
            defer_weight_export=False,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
            strip_doc_string=False,
            keep_initializers_as_inputs=_flags.EXPERIMENTAL_INITIALIZERS_AS_INPUTS,
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


class TorchScriptTracingEvaluator(Evaluator):
    def __init__(self, graph: TorchScriptGraph):
        self._graph: TorchScriptGraph = graph

    @property
    def graph(self) -> TorchScriptGraph:
        return self._graph

    def eval(self, schema, inputs, attributes):
        if _flags.EXPERIMENTAL_PREFER_TRACING:
            if schema.name == "CastLike":
                assert len(inputs) == 2
                # Skip CastLike if the input and output types are the same
                src_input = inputs[0]
                target_input = inputs[1]
                dtypes_available = (
                    isinstance(src_input, TorchScriptTensor)
                    and isinstance(target_input, TorchScriptTensor)
                    and src_input.dtype is not None
                    and target_input.dtype is not None
                )
                if dtypes_available:
                    if src_input.dtype == target_input.dtype:
                        # Same type. No cast needed
                        return src_input
                    else:
                        # Create a Cast node
                        return self._graph.add_op_call(
                            onnx.defs.get_schema("Cast"),
                            (src_input,),
                            {"to": target_input.onnx_dtype},
                        )
        return self._graph.add_op_call(schema, inputs, attributes)

    def eval_function(
        self,
        function: OnnxFunction,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
    ):
        if _flags.EXPERIMENTAL_PREFER_TRACING:
            # Special cases for handling IsScalar and Rank
            if function.name == "IsScalar":
                if len(args) != 1:
                    raise TypeError(
                        f"Expected 1 positional argument for function '{function}', got {len(args)}."
                    )
                if isinstance(args[0], TorchScriptTensor):
                    if args[0].rank is not None:
                        return args[0].rank == 0
                    else:
                        # Fall to call add_function_call
                        pass
                else:
                    # Python constants are scalars
                    return True
            if function.name == "Rank":
                if len(args) != 1:
                    raise TypeError(
                        f"Expected 1 positional argument for function '{function}', got {len(args)}."
                    )
                if isinstance(args[0], TorchScriptTensor):
                    if args[0].rank is not None:
                        return args[0].rank
                    else:
                        # Fall to call add_function_call
                        pass
                else:
                    # Python constants are scalars
                    return 0
            elif function.experimental_traceable:
                # Trace the function call instead of adding the function as a node
                return function.function(*args, **kwargs)

        # args/kwargs are TorchScriptTensor/python built-in based
        param_schemas = function.param_schemas()
        (
            inputs,
            attributes,
        ) = param_manipulation.separate_input_attributes_from_arguments(
            param_schemas, args, kwargs, fill_defaults=True, allow_extra_kwargs=True
        )
        name_to_schema = {param.name: param for param in param_schemas}
        for name, value in attributes.items():
            param = name_to_schema[name]
        return self._graph.add_function_call(function, inputs, attributes)
