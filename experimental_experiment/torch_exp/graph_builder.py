from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import onnx.helper as oh
import onnx.numpy_helper as onh
from onnx import FunctionProto, ModelProto, TensorProto


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

    def unique_name(self, prefix: str) -> str:
        if prefix in self._unique_names:
            i = 2
            sug = f"{prefix}2"
            while sug not in self._unique_names:
                i += 1
                sug = f"{prefix}{i}"
            return sug
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
        tensor = onh.from_array(value, name=name)
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
        return name

    def make_node(
        self,
        op_type: str,
        inputs: Union[str, List[str]],
        outputs: Union[int, List[str], str] = 1,
        domain: str = "",
        **kwargs,
    ) -> Union[str, List[str]]:
        if isinstance(outputs, int):
            if outputs < 1:
                raise ValueError(f"outputs={outputs} must be > 0.")
            lower = op_type.lower()
            output_names = [self.unique_name(f"{lower}{i}") for i in range(outputs)]
        elif isinstance(outputs, str):
            output_names = [outputs]
        else:
            output_names = outputs
        if isinstance(inputs, str):
            inputs = [inputs]
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
