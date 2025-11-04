import base64
import io
import itertools
import pickle
import re
import zlib
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from onnx_diagnostic.helpers.cache_helper import make_dynamic_cache, CacheKeyValue
from ..helpers import string_type


def extract_names_from_schema(schema: str) -> List[str]:
    """
    Extracts name from a C++ schema produced by ``infer_schema``.
    Example: ``(Tensor x, Tensor y) -> Tensor`` returns ["x", "y"].
    """
    pattern = r"\w+\??\s+(\w+)"
    matches = re.findall(pattern, schema)
    return matches


def choose_kwargs_for_dynamic_shapes(
    ds_args: Tuple[Dict[int, Any], ...],
    ds_kwargs: Dict[str, Dict[int, Any]],
    forward_names: List[str],
) -> Dict[str, Dict[int, Any]]:
    """
    Chooses a dictionary to express dynamic shapes when the module uses both
    unnamed arguments and named ones.
    """
    assert ds_args and ds_kwargs and forward_names, (
        f"No need to choose as ds_args={ds_args} or ds_kwargs={ds_kwargs} "
        f"or forward_names={forward_names} is None."
    )
    if len(forward_names) >= len(ds_args):
        # No *args.
        ds = ds_kwargs
        for k, v in zip(forward_names, ds_args):
            ds[k] = v
        return ds
    raise NotImplementedError(
        f"forward_positioned_parameter_names={forward_names}, "
        f"ds_args={ds_args}, ds_kwargs={ds_kwargs}"
    )


def serialize_one(
    obj: Any, name: Union[str, int], schema: str
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Serializes one object into a tensor or a list of tensors.
    *name* and *schema* are just better error messages.
    """
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, (tuple, list)) and all(isinstance(t, torch.Tensor) for t in obj):
        return obj
    if isinstance(obj, dict) and all(isinstance(t, torch.Tensor) for t in obj.values()):
        sorted_items = sorted(obj.items())
        return [_[1] for _ in sorted_items]
    if obj.__class__.__name__ == "DynamicCache":
        return list(itertools.chain.from_iterable(zip(obj.key_cache, obj.value_cache)))
    if obj is None:
        return None
    if isinstance(obj, (bool, int, float)):
        # This cannot be traced.
        return obj
    # Let's flatten the structure using pytorch.
    flat_list, _spec = torch.utils._pytree.tree_flatten(obj)
    return flat_list


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
    elif isinstance(args, dict) and all(isinstance(t, torch.Tensor) for t in args.values()):
        sorted_items = sorted(args.items())
        new_args = [_[1] for _ in sorted_items]
        n_args = len(new_args)
        is_tensor = False
    elif isinstance(args, (list, tuple)) and all(isinstance(i, torch.Tensor) for i in args):
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
    else:
        # Let's use pytorch serialization.
        new_args, _spec = torch.utils._pytree.tree_flatten(args)
        n_args = len(new_args)
        is_tensor = False

    if not kwargs:
        if kwargs is None:
            return new_args if is_tensor else tuple(new_args)
        return new_args, {}

    assert args_names is not None or schema, (
        f"Not implemented when args_names={args_names}, "
        f"args={string_type(args, with_shape=True, limit=20)}, "
        f"kwargs={string_type(kwargs, with_shape=True, limit=20)}, "
        f"schema={schema!r}"
    )
    if args_names is None:
        args_names = extract_names_from_schema(schema)

    new_args = [new_args] if is_tensor else list(new_args)
    args_names = args_names[n_args:]
    assert args_names, (
        f"kwargs={string_type(kwargs, with_shape=True, limit=20)} is specified "
        f"then args_names should be as well, schema={schema!r}, "
        f"args={string_type(args, with_shape=True, limit=20)}"
    )
    # handling arguments
    for name in args_names:
        if name not in kwargs:
            new_args.append(None)
            continue
        v = kwargs[name]
        r = serialize_one(v, name=name, schema=schema)
        if r is None or isinstance(r, (torch.Tensor, int, bool, float)):
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
        r = serialize_one(v, name=name, schema=schema)
        if r is None or isinstance(r, torch.Tensor):
            new_args.append(r)
        else:
            new_args.extend(r)
    return tuple(new_args), {}


def type_as_str_with_info(obj: Any) -> str:
    """Returns a string with information about how to deserialize."""
    if isinstance(obj, torch.Tensor):
        return "Tensor"
    if isinstance(obj, list) and all(isinstance(t, torch.Tensor) for t in obj):
        return f"list__{len(obj)}"
    if isinstance(obj, dict) and all(isinstance(t, torch.Tensor) for t in obj.values()):
        sorted_keys = "__".join(sorted(obj))
        return f"dict__{len(obj)}_{sorted_keys}"
    if obj.__class__.__name__ == "DynamicCache":
        kv = CacheKeyValue(obj)
        return f"{obj.__class__.__name__}__{len(kv.key_cache)+len(kv.value_cache)}"
    if obj is None:
        return "None"
    if isinstance(obj, bool):
        return "bool"
    if isinstance(obj, float):
        return "float"
    if isinstance(obj, int):
        return "int"
    # Let's use pytorch serialization.
    flat_list, spec = torch.utils._pytree.tree_flatten(obj)
    return tree_spec_as_name(spec, len(flat_list))


def deserialize_args(
    res: List[torch.Tensor],
    expected_types: List[str],
    clone: bool = False,
    return_n_args: bool = False,
) -> Tuple[Any, ...]:
    """
    Deserizalizes output results coming from the custom op and restores
    the python classes attached to it.

    :param res: args to deserialize
    :param expected_types: information on how to deserialize
    :param clone: clone tensors before returning them
    :param return_n_args: if True, the function returns the number of deserialized arguments,
        if False, it assumes this number if equal to the number of expected types
    :return: new args
    """
    assert isinstance(res, (list, tuple, torch.Tensor)), f"unexpected type for res {type(res)}"
    if isinstance(res, torch.Tensor):
        assert expected_types == [
            "Tensor"
        ], f"Mismatch information, expected_types={expected_types!r}"
        return res
    assert all(
        t is None or isinstance(t, (list, torch.Tensor, bool, int, float)) for t in res
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
        if tt in ("bool", "int", "float"):
            des.append(res[pos_res])
            pos_res += 1
            continue
        if tt.startswith("dict__"):
            nl = tt[6:].split("_", maxsplit=1)
            n = int(nl[0])
            keys = nl[1].split("__")
            assert len(keys) == n, f"Unable to parse {tt!r}, expecting {n} keys but got {keys}"
            value = res[pos_res]
            if isinstance(res[pos_res], torch.Tensor):
                values = res[pos_res : pos_res + n]
                des.append(dict(zip(keys, values)))
                pos_res += n
            else:
                des.append(dict(zip(keys, value)))
                pos_res += 1
            continue

        if tt.startswith("list__"):
            n = int(tt[6:])
            value = res[pos_res]
            if isinstance(res[pos_res], torch.Tensor):
                values = res[pos_res : pos_res + n]
                des.append(values)
                pos_res += n
            else:
                des.append(value)
                pos_res += 1
            continue

        if tt.startswith("DynamicCache__"):
            info = tt.split("__")[-1]
            n = int(info)
            assert n % 2 == 0, f"Unexpected sizes for n={n} for a DynamicCache"
            assert isinstance(
                res[pos_res], torch.Tensor
            ), f"Unexpected type for a tensor in a DynamicCache {type(res[pos_res])}"
            key_cache = res[pos_res : pos_res + n : 2]
            value_cache = res[pos_res + 1 : pos_res + n : 2]
            pos_res += n

            assert tt.startswith("DynamicCache__"), f"Unable to handle type info(2) {tt!r}"
            if clone:
                cache = make_dynamic_cache(
                    list(zip([t.clone() for t in key_cache], [t.clone() for t in value_cache]))
                )
            else:
                cache = make_dynamic_cache(list(zip(key_cache, value_cache)))
            des.append(cache)
            continue

        if tt.startswith("___"):
            n_args, spec = tree_spec_from_name(tt)
            obj = torch.utils._pytree.tree_unflatten(res[pos_res : pos_res + n_args], spec)
            pos_res += n_args
            des.append(obj)
            continue

        raise NotImplementedError(
            f"Unable to handle type info {tt!r}, "
            f"expected_types={expected_types!r}, "
            f"res={string_type(res, with_shape=True)}, "
            f"des={string_type(des, with_shape=True)}"
        )
    if return_n_args:
        return des, pos_res
    assert pos_res == len(res) or (
        pos_res == len(res) - 1 and isinstance(res[-1], list) and len(res[-1]) == 0
    ), (
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
    ordered_names: Optional[List[str]] = None,
    fill_kwargs: bool = False,
) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    """
    Deserializes a list of tensor or list of tensors into args and kwargs.
    *kwargs* should be empty since this type is allowed as a serialized type.

    :param args: arguments
    :param kwargs: named arguments, they should be empty
    :param expected_types: needed to understand how to deserialize
    :param clone: clone every tensor
    :param ordered_names: ordered needed to restore ``**kwargs``
    :param fill_kwargs: if True, the last parameter is ``**kwargs``
        and it should be empty
    :return: new args, new named args
    """
    assert not kwargs, (
        f"inputs coming from C++ functions should not have "
        f"named arguments but kwargs={string_type(kwargs, with_shape=True)}."
    )
    assert (
        isinstance(expected_types, tuple)
        and not kwargs
        and len(expected_types) == 2
        and (not expected_types[1] or ordered_names)
    ), (
        f"Unexpected value for expected_types={expected_types}, "
        f"args={string_type(args, with_shape=True)}, "
        f"kwargs={string_type(kwargs, with_shape=True)}, "
        f"ordered_names={ordered_names}"
    )
    if expected_types[1]:
        new_args, n_args = deserialize_args(
            args, expected_types[0], clone=clone, return_n_args=True
        )
        left_args = args[n_args:]
        left_names = ordered_names[n_args:]
        new_kwargs = {}
        pos_res = 0
        for name in left_names:
            if name not in expected_types[1]:
                # no **kwargs
                continue
            if expected_types[1][name] == "Tensor":
                new_kwargs[name] = left_args[pos_res]
                pos_res += 1
                continue
            if expected_types[1][name] in ("bool", "int", "float", "None"):
                new_kwargs[name] = left_args[pos_res]
                pos_res += 1
                continue
            assert isinstance(expected_types[1][name], str), (
                f"Unexpected type {type(expected_types[1][name])} for name={name!r} in "
                f"expected_types={expected_types}"
            )
            a, n = deserialize_args(
                left_args[pos_res:], [expected_types[1][name]], clone=clone, return_n_args=True
            )
            assert len(a) == 1, (
                f"Unexpected length, a={string_type(a, limit=20)}, "
                f"expected_types[1][name]={expected_types[1][name]!r}"
            )
            pos_res += n
            new_kwargs[name] = a[0]
        assert pos_res + n_args + (1 if fill_kwargs else 0) == len(args), (
            f"Deserialization went wrong, pos_res={pos_res + n_args}, "
            f"n_args={n_args}, len(args)={len(args)}, "
            f"\nfill_kwargs={fill_kwargs}, "
            f"\nexpected_types={expected_types}, "
            f"\nargs={string_type(args, limit=20)}, "
            f"\nnew_args={string_type(new_args, limit=20)}, "
            f"\nnew_kwargs={string_type(new_kwargs)}, "
            f"\nordered_names={ordered_names}"
        )
        return new_args, new_kwargs

    new_args = deserialize_args(args, expected_types[0], clone=clone)
    return new_args, {}


def tree_spec_as_name(tree_spec: torch.utils._pytree.TreeSpec, n_elements: int) -> str:
    """
    Returns a string containing all the information needed to code
    an instance of TreeSpec in a string.

    :param tree_spec: instance of TreeSpec to convert into a name
    :param n_elements: number of elements it serializes
    :return: string
    """
    buffer = io.BytesIO()
    pickle.dump((n_elements, tree_spec), buffer)
    data = buffer.getvalue()
    compressed_data = zlib.compress(data)
    chars = base64.b32encode(compressed_data).decode("utf-8")
    chars = chars.replace("=", "_")
    return f"___{chars}"


def tree_spec_from_name(name: str) -> Tuple[int, torch.utils._pytree.TreeSpec]:
    """
    Restores the instance of TreeSpec converted into a name by
    function :func:`tree_spec_as_name`.

    :param name: name of the TreeSpec
    :return: instance of TreeSpec, number of elements it contains
    """
    assert len(name) > 3 and name[:3] == "___", f"Incorrect name {name!r}"
    name = name[3:].replace("_", "=")
    buffer = base64.b32decode(name)
    unzip = zlib.decompress(buffer)
    n_elements, treespec = pickle.load(io.BytesIO(unzip))
    return n_elements, treespec
