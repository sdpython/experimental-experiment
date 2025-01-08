from onnx import TensorProto


def string_to_elem_type(name: str) -> int:
    """
    Converts a string into an element type.
    INT64 -> TensorProto.INT64
    """
    assert hasattr(TensorProto, name), f"Unable to interpret type {name!r}"
    return getattr(TensorProto, name)
