from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple, Union
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
        return node_type_frequency(onx.graph, min_freq=min_freq)
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


class _GraphPattern:
    def __init__(self, cursor: int):
        self.cursor = cursor
        self.subgraph = set()

    def add_cursor(self):
        assert (
            self.cursor not in self.subgraph
        ), f"Cursor {self.cursor} already added in {self.subgraph}"
        self.subgraph.add(self.cursor)


class _GraphPatterns:
    def __init__(self, nodes: List[onnx.NodeProto], cursor: Sequence[int]):
        self.nodes = nodes
        self.pats = [_GraphPattern(c) for c in cursor]

    def validate_cursor(self):
        # op_types
        nodes = [self.nodes[p.cursor] for p in self.pats]
        rec = set((n.op_type, len(n.input), len(n.output), len(n.attribute)) for n in nodes)
        if len(rec) != 1:
            return False
        n_atts = rec.pop()[-1]
        if n_atts == 0:
            return True

        # Needs to check attributes
        base = nodes[0].attribute.SerializeToString()
        for n in nodes[1:]:
            get = n.attribute.SerializeToString()
            if get != base:
                return False
        return True

    def add_cursor(self):
        for p in self.pats:
            p.add_cursor()


def find_largest_repeated_pattern(
    onx: Union[Sequence[onnx.NodeProto], onnx.ModelProto, onnx.GraphProto, onnx.FunctionProto],
    min_freq: int = 2,
) -> Optional[List[onnx.NodeProto]]:
    """
    Finds the largest repeated pattern in a graph.

    :param onx: any object containing a sequence of NodeProto
    :param min_freq: do not consider any frequency below that threshold
    :return: list of node in the pattern
    """
    if isinstance(onx, onnx.ModelProto):
        return find_largest_repeated_pattern(onx.graph, min_freq=min_freq)
    _freq, _freqs, npats, types = node_type_frequency(onx, min_freq)
    if not types:
        return None

    # initialization
    nodes = list(onx.node)
    cursor = []
    for i, n in enumerate(nodes):
        if n.op_type == types[0]:
            cursor.append(i)
    patterns = _GraphPatterns(cursor)

    valid = patterns.validate_cursor()
    if valid:
        patterns.add_cursor()

    # explores backward
    # _step_backward(nodes, )
    return [nodes[i] for i in sorted(patterns[0].subgraph)]
