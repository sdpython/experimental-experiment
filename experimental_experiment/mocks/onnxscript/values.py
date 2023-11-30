import dataclasses
from typing import Any


class Opset:
    domain: str
    version: int
    cache = {}

    def __new__(cls, domain: str, version: int):
        key = (cls, domain, version)
        existing = cls.cache.get(key)
        if existing:
            return existing
        instance = super().__new__(cls)
        instance.domain = domain
        instance.version = version
        instance.function_defs = {}
        cls.cache[key] = instance
        return instance

    def __init__(self, domain: str, version: int):
        pass


_EmptyDefault = object()


@dataclasses.dataclass
class TypeConstraint:
    """Represents a type constraint for an ONNX op.

    Attributes:
        name: The name of the type constraint.
        allowed_types: The allowed types for the type constraint.
    """

    name: str
    allowed_types: list[str]
    description: str = ""

    def as_tuple(self) -> tuple[str, list[str], str]:
        """Returns the type constraint as a tuple."""
        return (self.name, self.allowed_types, self.description)


@dataclasses.dataclass(frozen=True)
class ParamSchema:
    """A schema for a parameter of an Op or a OnnxFunction.

    Attributes:
        name: The name of the parameter.
        type: The type of the parameter.
        default: The default value of the parameter.
        required: Whether the input or attribute is required.
            For example, `Slice` has two optional inputs `axes` and `steps`.
            `SoftmaxCrossEntropyLoss` has an optional attribute `ignore_index`.
        is_input: Whether the parameter is an ONNX input.
        is_variadic_input: Whether the parameter, which has to be an INPUT, is variadic.
    """

    name: str
    type: Any = None
    default: Any = _EmptyDefault
    required: bool = True
    is_input: bool = True
    is_variadic_input: bool = False

    def __str__(self) -> str:
        """Return a string representation of the parameter.

        E.g. "x: Input[INT64]" or "axis: Attribute[int] = 0"
        """
        param_kind = "Input" if self.is_input else "Attribute"
        text = f"{self.name}: {param_kind}[{self.type}]"
        if self.default is not _EmptyDefault:
            text += f" = {self.default}"
        return text

    @property
    def is_attribute(self) -> bool:
        """Returns True if the parameter is an ONNX attribute."""
        return not self.is_input
