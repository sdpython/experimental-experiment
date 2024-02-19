from typing import Any, Sequence, Tuple, Union

STATIC_SHAPE = Tuple[int, ...]
DYNAMIC_SHAPE = Tuple[Union[int, "torch.SymInt", str], ...]  # noqa: F821


def all_int(seq: Sequence[Any]) -> bool:
    return all(map(lambda i: isinstance(i, int), seq))


def all_float(seq: Sequence[Any]) -> bool:
    return all(map(lambda i: isinstance(i, float), seq))


def all_int_or_float(seq: Sequence[Any]) -> bool:
    return all(map(lambda i: isinstance(i, (int, float)), seq))


def is_static_shape(shape: DYNAMIC_SHAPE) -> bool:
    return all(map(is_static_dimension, shape))


def is_static_dimension(d: int) -> bool:
    if isinstance(d, int):
        return True
    import torch

    if isinstance(d, torch.SymInt):
        try:
            int(str(d))
            return True
        except ValueError:
            return False
    return False
