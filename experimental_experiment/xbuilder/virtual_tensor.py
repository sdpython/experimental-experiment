from typing import Any, Tuple, Union


class VirtualTensor:
    """
    Defines a the type and shape for a tensor without its content.
    """

    def __init__(self, name: str, dtype: Any, shape: Tuple[Union[int, str], ...]):
        self.name = name
        self.dtype = dtype
        self.shape = shape

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name!r}, dtype={self.dtype}, shape={self.shape})"
