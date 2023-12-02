from typing import Any, Generator


class OverloadedFunction:
    """Overloaded function.

    Attributes:
        name: Name of the op. E.g. "aten::add".
        overloads: Overloads function.
        privates: Private functions not exposed to users.
        complex: Support complex functions.
    """

    def __init__(self, name: str):
        self.name = name
        self.overloads: list[Any] = []
        self.privates: list[Any] = []
        self.complex: list[Any] = []


class Registry:
    """Registry for aten functions."""

    def __init__(self):
        self._registry: dict[str, OverloadedFunction] = {}

    def register(
        self, func: Any, name: str, *, private: bool = False, complex: bool = False
    ) -> None:
        """Register a function."""
        over = OverloadedFunction(name)
        if private:
            self._registry.setdefault(name, over).privates.append(func)
        elif complex:
            self._registry.setdefault(name, over).complex.append(func)
        else:
            self._registry.setdefault(name, over).overloads.append(func)

    def __getitem__(self, name: str) -> OverloadedFunction:
        return self._registry[name]

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __iter__(self) -> Generator[str, None, None]:
        return iter(self._registry)

    def __repr__(self) -> str:
        return repr(self._registry)

    def items(self) -> Generator[tuple[str, OverloadedFunction], None, None]:
        yield from self._registry.items()

    def values(self) -> Generator[OverloadedFunction, None, None]:
        yield from self._registry.values()


default_registry = Registry()
