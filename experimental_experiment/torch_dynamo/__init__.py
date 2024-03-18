from typing import Any, Dict, List, Optional, Union
import numpy as np
from onnx import ModelProto
from .debug_backend import onnx_debug_backend
from .fast_backend import onnx_custom_backend


def get_decomposition_table():
    """
    Returns the decomposition table needed to translate backward
    graph into onnx. It should used as follows:

    ::

        import torch
        from torch._dynamo.backends.common import aot_autograd
        from experimental_experiment.torch_dynamo import get_decomposition_table

        aot_compiler = aot_autograd(fw_compiler=backend_debug, decompositions=get_decomposition_table())

        compiled_model = torch.compile(
            model,
            backend=aot_compiler,
            dynamic=dynamic,
            fullgraph=fullgraph,
        )

    The value is:

    .. runpython::
        :showcode:

        import pprint
        from experimental_experiment.torch_dynamo import get_decomposition_table

        pprint.pprint(get_decomposition_table())
    """
    import torch

    new_table = {}
    for k, v in torch._decomp.decomposition_table.items():
        if k.name() in {
            "aten::embedding_dense_backward",
            "aten::rrelu_with_noise",
            "aten::native_layer_norm_backward",
        }:
            new_table[k] = v
    return new_table


def filter_decomposition_table(existing_table: Optional[Dict] = None) -> Dict:
    """
    Returns the decomposition table when some conversions because
    their translation in ONNX is less efficient.

    :param existing_table: dictionary of decompositions, by default,
        it is ``torch._decomp.decomposition_table``.
    :return: new table

    ::

        import torch
        from torch._dynamo.backends.common import aot_autograd
        from experimental_experiment.torch_dynamo import filter_decomposition_table

        aot_compiler = aot_autograd(fw_compiler=backend_debug, decompositions=filter_decomposition_table())

        compiled_model = torch.compile(
            model,
            backend=aot_compiler,
            dynamic=dynamic,
            fullgraph=fullgraph,
        )

    The value is:

    .. runpython::
        :showcode:

        import pprint
        from experimental_experiment.torch_dynamo import filter_decomposition_table

        pprint.pprint(filter_decomposition_table())
    """
    if existing_table is None:
        import torch

        existing_table = torch._decomp.decomposition_table.items()

    new_table = {}
    for k, v in existing_table:
        if k.name() in {
            "aten::slice_backward",
            "aten::select_backward.out",
            "aten::slice.Tensor",
        }:
            continue
        new_table[k] = v
    return new_table


def _single_print(v):
    if v is None:
        return "None"
    if isinstance(v, (int, bool, str, float)):
        return str(v)
    if isinstance(v, np.ndarray):
        return f"array:{v.dtype}:{v.shape}:{v.mean()}"
    if hasattr(v, "numpy"):
        return _single_print(v.detach().cpu().numpy())
    if isinstance(v, ModelProto):
        s = str(v).replace("\n", "").replace(" ", "")
        return "ModelProto:" + s[:20] + "..." + s[-20:]
    if "GraphModule" in str(type(v)):
        s = str(v).replace("\n", "")
        return "GraphModule:" + s[:20] + "..." + s[-20:]
    if "GraphBuilder" in str(type(v)):
        s = str(v).replace("\n", "")
        return "GraphBuilder:" + s[:20] + "..." + s[-20:]
    if "ExtendedReferenceEvaluator" in str(type(v)):
        s = str(v).replace("\n", "")
        return "ExtendedReferenceEvaluator:" + s[:20] + "..." + s[-20:]
    raise TypeError(f"Unexpected type {type(v)}.")


def pprint_storage(
    storage: Any, indent: int = 0, as_list: bool = False
) -> Union[List[str], str]:
    """
    Pretty print for the storage.

    :param storage: any object
    :param indent: indentation
    :param as_list: return list or string
    :return: list or string
    """
    sind = "  " * indent
    if isinstance(storage, (np.ndarray, int, float, str, bool, type(None))):
        rows = [sind + _single_print(storage)]
    elif isinstance(storage, dict):
        if len(storage) <= 10 and all(
            map(
                lambda v: isinstance(v, (int, float, str, bool, type(None))),
                storage.values(),
            )
        ):
            rows = [sind + str(storage)]
        else:
            rows = [sind + "{"]
            for k, v in storage.items():
                r = pprint_storage(v, indent=indent + 1, as_list=True)
                if len(r) > 1:
                    r[0] = f"{r[0][:-1]}{k!r}: {r[0][-1]}"
                    r[-1] += ","
                    rows.extend(r)
                else:
                    rows.append(f"  {sind}{k!r}: {r[0].lstrip(' ')},")
            rows.append(sind + "}")
    elif isinstance(storage, list):
        if len(storage) <= 10 and all(
            map(lambda v: isinstance(v, (int, float, str, bool, type(None))), storage)
        ):
            rows = [sind + str(storage)]
        else:
            rows = [sind + "["]
            for v in storage:
                r = pprint_storage(v, indent=indent + 1, as_list=True)
                if len(r) > 1:
                    r[-1] += ","
                    rows.extend(r)
                else:
                    rows.append(f"  {sind}{r[0].lstrip(' ')},")
            rows.append(sind + "]")
    elif isinstance(storage, tuple):
        if len(storage) <= 10 and all(
            map(lambda v: isinstance(v, (int, float, str, bool, type(None))), storage)
        ):
            rows = [sind + str(storage)]
        else:
            rows = [sind + "("]
            for v in storage:
                r = pprint_storage(v, indent=indent + 1, as_list=True)
                if len(r) > 1:
                    r[-1] += ","
                    rows.extend(r)
                else:
                    rows.append(f"  {sind}{r[0].lstrip(' ')},")
            rows.append(sind + ")")
    elif hasattr(storage, "numpy"):
        rows = [sind + _single_print(storage)]
    elif storage is None:
        rows = [sind + _single_print(storage)]
    elif (
        "GraphModuleImpl" in str(type(storage))
        or "ModelProto" in str(type(storage))
        or "GraphBuilder" in str(type(storage))
        or "ExtendedReferenceEvaluator" in str(type(storage))
    ):
        rows = [sind + _single_print(storage)]
    else:
        raise RuntimeError(f"Unexpected type {type(storage)}")
    if as_list:
        return rows
    return "\n".join(rows)
