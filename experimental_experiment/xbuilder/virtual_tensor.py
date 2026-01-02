from typing import Any, Optional, Tuple, Union


class VirtualTensor:
    """Defines a the type and shape for a tensor without its content."""

    def __init__(
        self,
        name: str,
        dtype: Any,
        shape: Tuple[Union[int, str], ...],
        device: Optional[int] = None,
    ):
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.device = device
        assert device is None or isinstance(device, int), f"Unexpected type for device={device!r}"

    def __repr__(self) -> str:
        "Usual"
        if self.device is None:
            return (
                f"{self.__class__.__name__}({self.name!r}, dtype={self.dtype}, "
                f"shape={self.shape})"
            )
        return (
            f"{self.__class__.__name__}({self.name!r}, dtype={self.dtype}, "
            f"shape={self.shape}, device={self.device})"
        )

    def get_device(self) -> Optional[int]:
        "Returns device."
        return self.device
