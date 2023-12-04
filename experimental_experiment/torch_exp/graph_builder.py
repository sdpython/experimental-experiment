from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import onnx.helper as oh
import onnx.numpy_helper as onh
from onnx import FunctionProto, ModelProto, TensorProto


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


class GraphBuilder:
    def __init__(
        self,
        target_opset: Union[int, Dict[str, int]],
        input_names: Optional[Sequence[str]] = None,
    ):
        self.opsets = (
            {"": target_opset} if isinstance(target_opset, int) else target_opset
        )
        self.nodes = []
        self.initializers = []
        self.inputs = []
        self.outputs = []
        self._unique_names = set()
        self.input_names = input_names or []
        self.current_input = 0
        self.op = Opset(self, self.opsets[""])
        self._known_shapes = {}
        self._known_types = {}
        self._checked_added = set()

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
        if hasattr(value, "numpy"):
            value = value.numpy()
        if name == "":
            name = self.unique_name("cst")
        tensor = onh.from_array(value, name=name)
        self.set_shape(name, value.shape)
        self.set_type(name, oh.np_dtype_to_tensor_dtype(value.dtype))
        self.initializers.append(tensor)
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
        if "ceil_mode" in kwargs and not isinstance(kwargs["ceil_mode"], int):
            raise RuntimeError(
                f"Wrong value for ceil_mode operator is {op_type}, kwargs={kwargs}."
            )
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
        # basic checking, to be removed later
        key = (op_type, tuple(output_names))
        if key in self._checked_added:
            raise RuntimeError(f"This node was already added: {key}.")
        self._checked_added.add(key)
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
        self.nodes.append(node)
        if len(output_names) == 1:
            return output_names[0]
        return output_names

    def to_onnx(self, as_function: bool = False) -> Union[FunctionProto, ModelProto]:
        dense = [i for i in self.initializers if isinstance(i, TensorProto)]
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
