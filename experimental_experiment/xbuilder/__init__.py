from onnx import TensorProto
from .optimization_options import OptimizationOptions
from .graph_builder import GraphBuilder, FunctionOptions
from .virtual_tensor import VirtualTensor


def str_tensor_proto_type() -> str:
    mapping = [
        (getattr(TensorProto, att), att)
        for att in dir(TensorProto)
        if att.upper() == att and isinstance(getattr(TensorProto, att), int)
    ]
    mapping.sort()
    return ", ".join(f"{k}:{v}" for k, v in mapping)
