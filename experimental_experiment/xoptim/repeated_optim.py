import hashlib
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


def _serialize_attribute(attribute: Sequence[onnx.AttributeProto]) -> bytes:
    return b"/".join(a.SerializeToString() for a in attribute)


class _GraphPattern:
    def __init__(self, first_node: int):
        self.cursor = first_node
        self.first_node = first_node
        self.subgraph = set()

    def add_cursor(self):
        assert self.cursor >= 0, f"Cannot add a negative cursor ({self.cursor})"
        assert (
            self.cursor not in self.subgraph
        ), f"Cursor {self.cursor} already added in {self.subgraph}"
        self.subgraph.add(self.cursor)


class _GraphIterator:
    def __init__(self, graph: "_GraphPatterns", node_index: int):
        self.graph = graph
        self.node_index = node_index
        self.io_index = None
        self.io_kind = None
        self.io_name = None
        self.o_suc = None
        self.o_suc_index = None

    def __str__(self) -> str:
        if self.node_index is None:
            return "it()"
        indices = [
            self.node_index,
            self.io_index,
            "N" if self.io_kind is None else ("I" if self.io_kind else "O"),
            "." if self.o_suc is None else self.o_suc,
            self.io_name,
        ]
        s = ", ".join(map(str, indices))
        return f"it({s})"

    def next(self):
        # assumes node.output is never empty
        node = self.graph.nodes[self.node_index]
        if self.io_name is None:
            self.io_index = 0
            self.o_suc = 0
            self.io_kind = bool(node.input)
            self.io_name = node.input[0] if self.io_kind else node.output[0]
            self.o_suc_index = (
                None
                if self.io_kind or not self.graph.successors[self.io_name]
                else self.graph.successors[self.io_name][self.o_suc]
            )
        else:
            if self.io_kind:
                self.io_index += 1
                self.o_suc = 0
                if self.io_index >= len(node.input):
                    self.io_kind = False
                    self.io_index = 0
                self.io_name = (
                    node.input[self.io_index] if self.io_kind else node.output[self.io_index]
                )
                self.o_suc_index = (
                    None
                    if not self.graph.successors[self.io_name]
                    else self.graph.successors[self.io_name][self.o_suc]
                )
            else:
                self.io_name = node.output[self.io_index]
                if self.io_name not in self.graph.successors:
                    self.io_name = None
                    return False
                self.o_suc += 1
                if self.o_suc < len(self.graph.successors[self.io_name]):
                    self.o_suc_index = self.graph.successors[self.io_name][self.o_suc]
                    return True
                self.o_suc = 0
                self.io_index += 1
                if self.io_index >= len(node.output):
                    self.io_name = None
                    self.o_suc_index = None
                    return False
                self.io_name = node.output[self.io_index]
                self.o_suc_index = (
                    None
                    if not self.graph.successors[self.io_name]
                    else self.graph.successors[self.io_name][self.o_suc]
                )
        return True

    def get_name(self, node_index: int) -> str:
        node = self.graph.nodes[node_index]
        if self.io_kind is None:
            return None
        if self.io_kind:
            name = node.input[self.io_index]
        else:
            name = node.output[self.io_index]
        assert (
            node_index != self.node_index or name == self.io_name
        ), f"Inconsistency with node_index={node_index}, name={name!r}, self={self!r}"
        return name

    def get_node_index(self, node_index: int) -> int:
        node = self.graph.nodes[node_index]
        if self.io_kind is None:
            return None
        if self.io_kind:
            name = node.input[self.io_index]
            index = self.graph.predecessor.get(name, -1)
        else:
            name = node.output[self.io_index]
            suc = self.graph.successors.get(name, [])
            if not suc:
                return -1
            # It is tricky here because the order of the successors
            # is not necessarily the same.
            if self.o_suc == 0 and len(suc) == 1:
                # Only one possible.
                index = suc[self.o_suc]
            else:
                assert self.o_suc_index is not None, (
                    f"Unable to guess the forward node, node_index={node_index}, "
                    f"self={self}, mapped={self.graph.mapped}"
                )
                expected_sig = self.graph.signatures[self.o_suc_index]
                sigs = {self.graph.signatures[s]: s for s in suc}
                assert len(sigs) == len(suc), (
                    f"Unable to distinguish between successors signatares: {sigs}, "
                    f"node_index={node_index}, self={self}"
                )
                if expected_sig not in sigs:
                    # Cannot find the expected successor
                    return -1
                return sigs[expected_sig]

        assert node_index != self.node_index or name == self.io_name, (
            f"Inconsistency with node_index={node_index}, "
            f"self.io_index={self.io_index!r}, name={name!r}, self={self!r}"
        )
        return index


class _GraphPatterns:
    def __init__(self, nodes: List[onnx.NodeProto], cursor: Sequence[int]):
        self.nodes = nodes
        self.pats = [_GraphPattern(c) for c in cursor]
        self.current: List[_GraphIterator] = []
        self.build_edges()
        self.processed_indices = set()
        self.mapped: Dict[int : List[int]] = {}

    def make_sig(self, node: onnx.NodeProto) -> str:
        hash = (
            f"H{hashlib.sha256(_serialize_attribute(node.attribute)).hexdigest()[:20]}"
            if node.attribute
            else ""
        )
        sigi = []
        for i in node.input:
            p = self.predecessor.get(i, -1)
            sigi.append(self.nodes[p].op_type if p >= 0 else "")
        sig = (
            f"{node.domain}/{node.op_type}/{len(node.input)}-{len(node.output)}"
            f"{hash}//{'/'.join(sigi)}"
        )
        return sig

    def build_edges(self):
        self.successors: Dict[str, Dict[str, int]] = {}
        self.predecessor: Dict[str, int] = {}
        self.signatures: Dict[int, str] = {}
        for node_index, node in enumerate(self.nodes):
            for i in node.output:
                self.predecessor[i] = node_index
            for i in node.input:
                if i not in self.successors:
                    self.successors[i] = []
                self.successors[i].append(node_index)
            sig = self.make_sig(node)
            self.signatures[node_index] = sig

    def validate_cursor(self):
        # op_types
        if any(p.cursor < 0 for p in self.pats):
            return False
        # already processed
        if any(p.cursor in self.processed_indices for p in self.pats):
            return False
        nodes = [self.nodes[p.cursor] for p in self.pats]
        rec = set((n.op_type, len(n.input), len(n.output), len(n.attribute)) for n in nodes)
        if len(rec) != 1:
            return False
        n_atts = rec.pop()[-1]
        if n_atts == 0:
            return True

        # Needs to check attributes
        base = _serialize_attribute(nodes[0].attribute)
        for n in nodes[1:]:
            get = _serialize_attribute(n.attribute)
            if get != base:
                return False
        return True

    def add_cursor(self):
        bug = set()
        for pi, p in enumerate(self.pats):
            assert p.cursor not in bug, (
                f"Every cursor pi={pi}, should be different but "
                f"{[p.cursor for p in self.pats]}"
            )
            bug.add(p.cursor)
            p.add_cursor()
            if p.cursor not in self.mapped:
                self.mapped[p.cursor] = set()
        for p in self.pats:
            for pp in self.pats:
                self.mapped[p.cursor].add(pp.cursor)

    def apply_path(self, node_index: int) -> int:
        if not self.current:
            return node_index
        for p in self.current:
            node_index = p.get_node_index(node_index)
        return node_index

    def set_cursor(self):
        bug = set()
        for pi, p in enumerate(self.pats):
            p.cursor = self.apply_path(p.first_node)
            assert p.cursor is not None, (
                f"Wonrg cursor for p.first_node={p.first_node} and "
                f"path={'/'.join(map(str,self.current))}"
            )
            if p.cursor >= 0:
                assert p.cursor not in bug, (
                    f"Every cursor pi={pi}, should be different but "
                    f"{[p.cursor for p in self.pats]}, "
                    f"first_nodes={[p.first_node for p in self.pats]}"
                )
                bug.add(p.cursor)

    def next_valid(self):
        i = self.pats[0].cursor
        self.current.append(_GraphIterator(self, i))
        return self.next_not_valid()

    def next_not_valid(self):
        has_next = self.current[-1].next()
        while not has_next:
            self.current.pop()
            if not self.current:
                return False
            has_next = self.current[-1].next()
        self.set_cursor()
        return True

    def add_processed_cursor(self):
        if any(p.cursor == -1 for p in self.pats):
            return
        for p in self.pats:
            if p.cursor != -1:
                self.processed_indices.add(p.cursor)

    def process(self) -> Optional[Tuple[List[int], List[onnx.NodeProto]]]:
        valid = self.validate_cursor()
        n_iter = 0
        while True and n_iter < len(self.nodes):
            if valid:
                self.add_cursor()
                self.add_processed_cursor()
                is_next = self.next_valid()
            else:
                self.add_processed_cursor()
                is_next = self.next_not_valid()
            if not is_next:
                valid = True
                break
            valid = self.validate_cursor()
            n_iter += 1

        if self.pats[0].subgraph:
            indices = sorted(self.pats[0].subgraph)
            return indices, [self.nodes[i] for i in indices]
        return None


def find_largest_repeated_pattern(
    onx: Union[Sequence[onnx.NodeProto], onnx.ModelProto, onnx.GraphProto, onnx.FunctionProto],
    min_freq: int = 2,
) -> Optional[Tuple[List[int], List[onnx.NodeProto]]]:
    """
    Finds the largest repeated pattern in a graph.

    :param onx: any object containing a sequence of NodeProto
    :param min_freq: do not consider any frequency below that threshold
    :return: list of node indices in the pattern, list of nodes in the pattern
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
        if (n.domain, n.op_type) == types[0]:
            cursor.append(i)
    patterns = _GraphPatterns(nodes, cursor)
    return patterns.process()
