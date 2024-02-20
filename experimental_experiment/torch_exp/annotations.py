from typing import Any, Sequence, Tuple, Union

STATIC_SHAPE = Tuple[int, ...]
DYNAMIC_SHAPE = Tuple[Union[int, "torch.SymInt", str], ...]  # noqa: F821


def all_int(seq: Sequence[Any]) -> bool:
    return all(map(lambda i: isinstance(i, int), seq))


def all_float(seq: Sequence[Any]) -> bool:
    return all(map(lambda i: isinstance(i, float), seq))


def all_int_or_float(seq: Sequence[Any]) -> bool:
    return all(map(lambda i: isinstance(i, (int, float)), seq))


def all_int_or_str(seq: Sequence[Any]) -> bool:
    return all(map(lambda i: isinstance(i, (int, str)), seq))


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


def compatible_shapes(sh1: DYNAMIC_SHAPE, sh2: DYNAMIC_SHAPE) -> bool:
    """
    Checks that two shapes are compatible. If both static, they must be equal.
    If dynamic, the variable part must be compatible meaning they could be equal.

    :param sh1: first shape
    :param sh2: seconde shape
    :return: compatibility

    .. runpython::
        :showcode:

        from experimental_experiment.torch_exp.annotations import compatible_shapes

        print(compatible_shapes((1, 2), (1, 2)))  # True
        print(compatible_shapes((1, 2), (1, "D2")))  # True
        print(compatible_shapes(("D2", 2), (1, "D2")))  # False
        print(compatible_shapes(("D2", 2), (2, "D2")))  # True
    """
    assert isinstance(sh1, tuple), f"type(sh1)={type(sh1)} is unexpected"
    assert isinstance(sh2, tuple), f"type(sh2)={type(sh2)} is unexpected"
    if len(sh1) != len(sh2):
        # incompatible rank
        return False
    assert all_int_or_str(
        sh1
    ), f"Shape sh1={sh1} has unexpected dimension type: {[type(i) for i in sh1]}"
    assert all_int_or_str(
        sh2
    ), f"Shape sh2={sh2} has unexpected dimension type: {[type(i) for i in sh2]}"
    constraints = {}
    for a, b in zip(sh1, sh2):
        if a == b:
            continue
        if type(a) == type(b):
            # The same name should be used for the same value.
            if a != b:
                return False
            continue
        name, value = (b, a) if isinstance(a, int) else (a, b)
        if name not in constraints:
            constraints[name] = value
            continue
        if constraints[name] != value:
            return False
    return True


def compatible_dimensions(*dims: Sequence[Union[int, str]]) -> bool:
    """
    Evaluates the fact all the dimensions can be equal or not.

    :param dims: dimensions
    :return: compatibility

    .. runpython::
        :showcode:

        from experimental_experiment.torch_exp.annotations import compatible_dimensions

        print(compatible_dimensions(1, 1))  # True
        print(compatible_dimensions(1, 2))  # False
        print(compatible_dimensions(1, "D"))  # True
        print(compatible_dimensions(1, "D", "DD"))  # False
    """
    assert all_int_or_str(
        dims
    ), f"unexpected types in {dims} ({[type(i) for i in dims]})"
    unique = set(dims)
    ints = [i for i in unique if isinstance(i, int)]
    if len(ints) > 1:
        return False
    strs = [i for i in unique if isinstance(i, str)]
    if len(strs) > 1:
        return False
    return True
