import copy
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from ..helpers import string_type


def extract_names_from_schema(schema: str) -> List[str]:
    """
    Extracts name from a C++ schema produced by ``infer_schema``.
    Example: ``(Tensor x, Tensor y) -> Tensor`` returns ["x", "y"]
    """
    pattern = r"\w+\??\s+(\w+)"
    matches = re.findall(pattern, schema)
    return matches


def serialize_one(
    obj: Any, name: Union[str, int], schema: str
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Serializes one object into a tensor or a list of tensors.
    *name* and *schema* are just better error messages.
    """
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, (tuple, list)):
        assert all(
            isinstance(t, torch.Tensor) for t in obj
        ), f"Unexpected type in {string_type(obj)}. It should be all tensors."
        return obj
    if obj.__class__.__name__ in {"DynamicCache", "patched_DynamicCache"}:
        return [*obj.key_cache, *obj.value_cache]
    if obj is None:
        return None
    raise NotImplementedError(
        f"Unable to serialize type {type(obj)}, "
        f"class_name={obj.__class__.__name__!r}, "
        f"types={string_type(obj, with_shape=True)}, "
        f"name={name!r} from schema={schema!r}"
    )


def serialize_args(
    args: Tuple[Any],
    kwargs: Optional[Dict[str, Any]],
    schema: str,
    args_names: Optional[List[str]] = None,
) -> Tuple[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
    """
    Serializes args and kwargs before calling a custom ops.

    :param args: unnamed arguments
    :param kwargs: named arguments
    :param schema: schema function
    :param args_names: ordered argument names, it must be specified
        if *kwargs* is specified
    """
    if isinstance(args, torch.Tensor):
        new_args = args
        n_args = 1
        is_tensor = True
    else:
        new_args = []
        for i, a in enumerate(args):
            r = serialize_one(a, name=i, schema=schema)
            if r is None or isinstance(r, torch.Tensor):
                new_args.append(r)
            else:
                new_args.extend(r)
        new_args = tuple(new_args)
        n_args = len(new_args)
        is_tensor = False

    if not kwargs:
        return new_args

    assert args_names is not None or schema, (
        f"Not implemented when args_names={args_names}, "
        f"args={string_type(args, with_shape=True)}, "
        f"kwargs={string_type(kwargs, with_shape=True)}, "
        f"schema={schema!r}"
    )
    if args_names is None:
        args_names = extract_names_from_schema(schema)
    new_args = [new_args] if is_tensor else list(new_args)
    args_names = args_names[n_args:]
    assert args_names, (
        f"kwargs={string_type(kwargs, with_shape=True)} is specified "
        f"then args_names should be as well, schema={schema!r}, "
        f"args={string_type(args, with_shape=True)}"
    )
    # handling arguments
    for name in args_names:
        if name not in kwargs:
            new_args.append(None)
            continue
        v = kwargs[name]
        r = serialize_one(a, name=name, schema=schema)
        if r is None or isinstance(r, torch.Tensor):
            new_args.append(r)
        else:
            new_args.extend(r)

    # remaining arguments (**kwargs, *args)
    set_names = set(args_names)
    for k, v in kwargs.items():
        if k in set_names:
            continue
        if v is None:
            new_args.append(None)
        r = serialize_one(a, name=name, schema=schema)
        if r is None or isinstance(r, torch.Tensor):
            new_args.append(r)
        else:
            new_args.extend(r)
    return tuple(new_args), {}


def type_as_str_with_info(obj: Any) -> str:
    """Returns a string with information about how to deserialize."""
    if isinstance(obj, torch.Tensor):
        return "Tensor"
    if obj.__class__.__name__ in {"DynamicCache", "patched_DynamicCache"}:
        return f"{obj.__class__.__name__}__{len(obj.key_cache)}_{len(obj.value_cache)}"
    if obj is None:
        return "None"
    raise NotImplementedError(
        f"Unable to produce serialize info for type {type(obj)}, "
        f"class_name={obj.__class__.__name__!r}."
    )


def deserialize_args(
    res: List[torch.Tensor], expected_types: List[str], clone: bool = False
) -> Tuple[Any, ...]:
    """
    Deserizalizes output results coming from the custom op and restores
    the python classes attached to it.

    :param res: args to deserialize
    :param expected_types: information on how to deserialize
    :param clone: clone tensors before returning them
    :return: new args
    """
    assert isinstance(res, (list, tuple, torch.Tensor)), f"unexpected type for res {type(res)}"
    if isinstance(res, torch.Tensor):
        assert expected_types == [
            "Tensor"
        ], f"Mismatch information, expected_types={expected_types!r}"
        return res
    assert all(
        t is None or isinstance(t, (list, torch.Tensor)) for t in res
    ), f"unexpected element type in res: {string_type(res)}"
    des = []
    pos_res = 0
    for tt in expected_types:
        if tt in (None, "None"):
            des.append(None)
            pos_res += 1
            continue
        if tt == "Tensor":
            des.append(res[pos_res].clone() if clone else res[pos_res])
            pos_res += 1
            continue
        if tt.startswith(("DynamicCache__", "patched_DynamicCache__")):
            info = tt.split("__")[-1]
            n1, n2 = tuple(map(int, info.split("_")))
            assert n1 == n2, f"Unexpected sizes for n1={n1} and n2={n2} for a DynamicCache"
            if isinstance(res[pos_res], torch.Tensor):
                # All flattened.
                key_cache = res[pos_res : pos_res + n1]
                value_cache = res[pos_res + n1 : pos_res + n1 + n2]
                pos_res += n1 + n2
            else:
                value = res[pos_res]
                assert isinstance(value, list) and all(
                    isinstance(t, torch.Tensor) for t in value
                ), (
                    f"Unexpected type at position {pos_res}: "
                    f"{string_type(value, with_shape=True)}, "
                    f"deserialized into {tt}"
                )
                assert len(value) % 2 == 0 and len(value) == n1 + n2, (
                    f"Number of tensors at position {pos_res} "
                    f"in {string_type(value, with_shape=True)} "
                    f"should be even. Unable to deserialize into {tt}, "
                    f"n1={n1}, n2={n2}, len(res[pos_res])={len(value)}"
                )
                key_cache = value[:n1]
                value_cache = value[n1:]
                pos_res += 1

            if tt.startswith("DynamicCache__"):
                import transformers

                cache = transformers.cache_utils.DynamicCache()
            elif tt.startswith("patched_DynamicCache__"):
                from .patches.patch_transformers import patched_DynamicCache

                cache = patched_DynamicCache()
            elif tt is None:
                cache = None
            else:
                raise NotImplementedError(f"Unable to handle type info(2) {tt!r}")
            if clone:
                cache.key_cache = [t.clone() for t in key_cache]
                cache.value_cache = [t.clone() for t in value_cache]
            else:
                cache.key_cache = key_cache
                cache.value_cache = value_cache
            des.append(cache)
            continue

        raise NotImplementedError(f"Unable to handle type info {tt!r}")
    assert pos_res == len(res), (
        f"Deserialization went wrong, pos_res={pos_res}, len(res)={len(res)}, "
        f"expected_types={expected_types}, "
        f"input types={string_type(res)}"
    )
    return des


def deserialize_args_kwargs(
    args: List[torch.Tensor],
    kwargs: Dict[str, Any],
    expected_types: Tuple[List[str], List[str]],
    clone: bool = False,
) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    """
    Deserializes a list of tensor or list of tensors into args and kwargs.
    *kwargs* should be empty since this type is allowed as a serialized type.

    :param args: arguments
    :param kwargs: named arguments, they should be empty
    :param expected_types: needed to understand how to deserialize
    :param clone: clone every tensor
    :return: new args, new named args
    """
    assert not kwargs, (
        f"inputs coming from C++ functions should not have "
        f"named arguments but kwargs={string_type(kwargs, with_shape=True)}."
    )
    assert (
        isinstance(expected_types, tuple)
        and len(expected_types) == 2
        and not expected_types[1]
    ), (
        f"Unexpected value for expected_types={expected_types}, "
        f"args={string_type(args, with_shape=True)}, "
        f"kwargs={string_type(kwargs, with_shape=True)}, "
    )
    new_args = deserialize_args(args, expected_types[0], clone=clone)
    return new_args, {}


def make_copy(obj: Any) -> Any:
    """Makes a copy of the objects."""
    if isinstance(obj, np.ndarray):
        return obj.copy()
    if isinstance(obj, tuple):
        return tuple(make_copy(_) for _ in obj)
    if isinstance(obj, list):
        return [make_copy(_) for _ in obj]
    if isinstance(obj, dict):
        return {k: make_copy(v) for k, v in obj.items()}
    if hasattr(obj, "clone"):
        return obj.clone()
    if obj.__class__.__name__ in ("DynamicCache", "patched_DynamicCache"):
        cache = obj.__class__()
        if hasattr(obj, "_seen_tokens"):
            cache._seen_tokens = obj._seen_tokens
        cache.key_cache = make_copy(obj.key_cache)
        cache.value_cache = make_copy(obj.value_cache)
        return cache
    try:
        return copy.deepcopy(obj)
    except RuntimeError as e:
        raise RuntimeError(
            f"deepcopy did not work on type {type(obj)}: {string_type(obj)}"
        ) from e
