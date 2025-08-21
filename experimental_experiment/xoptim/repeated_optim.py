from collections import Counter
from typing import Dict, List, Sequence, Tuple, Union
import onnx


def node_type_frequency(
    onx: Union[Sequence[onnx.NodeProto], onnx.ModelProto, onnx.GraphProto, onnx.FunctionProto],
    min_freq: int = 2,
) -> Tuple[Dict[Tuple[str, str], int], Dict[Tuple[str, str], int], int, List[Tuple[str, str]]]:
    """
    Computes the frequency of every node type in a list.

    :param onx: any object containing a sequence of NodeProto
    :param min_freq: do not consider any frequency below that threshold
    :return: 4 results,
        the frequencies of the node types,
        the frequencies of the frequencies,
        the most frequent frequency (the estimation of the number of layers),
        all types having the exact same frequency as the previous one

    .. note::
        This function assumes at least one type of node is present only once in every layer.
    """
    if isinstance(onx, onnx.ModelProto):
        return node_type_frequency(onx.graph)
    h = Counter((node.domain, node.op_type) for node in onx.node)
    freq = {k: v for k, v in h.items() if v >= min_freq}
    freq_freq = Counter(freq.values())
    freqs = dict(freq_freq)
    for k, v in freq_freq.items():
        for i in range(2, k):
            if k % i == 0 and i in freq_freq:
                freqs[i] += k // i * v
    ret = max((v, k) for k, v in freqs.items())
    types = [k for k, v in freq.items() if v == ret[1]]
    return freq, freqs, ret[1], types
