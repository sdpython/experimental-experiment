from typing import Dict, Optional
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
