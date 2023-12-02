from typing import Any, List, Optional, Tuple, Union
import onnx.helper as oh
from onnx import ModelProto, TensorProto
from .values import Opset


class GraphBuilder:
    def __init__(self, opsets: List[Opset]):
        self.opsets = {o.domain: o for o in opsets}
        self.nodes = []
        self.initializers = []
        self.inputs = []
        self.outputs = []
        self.unique_names = set()

    @property
    def op(self) -> Opset:
        return self.opset("")

    def opset(self, name: str) -> Opset:
        return self.opsets[name]

    def unique_name(self, prefix: str) -> str:
        if prefix in self.unique_names:
            i = 2
            sug = f"{prefix}2"
            while sug not in self.unique_names:
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

    def make_input(self, name: str, elem_type: Any, shape: Tuple[int, ...]) -> str:
        elem_type = self._get_type(elem_type)
        self.inputs.append(oh.make_tensor_value_info(name, elem_type, shape))
        return name

    def make_output(
        self,
        name: Union[str, List[str]],
        elem_type: Optional[int] = None,
        shape: Optional[Tuple[int, ...]] = None,
    ) -> Union[str, List[str]]:
        if isinstance(name, list):
            res = []
            for n in name:
                res.append(self.make_output(n, elem_type, shape))
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
        node = oh.make_node(op_type, inputs, output_names, domain=domain, **kwargs)
        self.nodes.append(node)
        if len(output_names) == 1:
            return output_names[0]
        return output_names

    def to_onnx(self) -> ModelProto:
        dense = [
            self._fix_name_tensor(i)
            for i in self.initializers
            if isinstance(i, TensorProto)
        ]
        graph = oh.make_graph(
            self.nodes, "experiment", self.inputs, self.outputs, dense
        )
        opsets = [oh.make_opsetid(o.domain, o.version) for o in self.opsets.values()]
        model = oh.make_model(graph, opset_imports=opsets)
        return model
