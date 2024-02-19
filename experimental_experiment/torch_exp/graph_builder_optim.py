from typing import Iterator, List, Optional
from ._onnx_helper import enumerate_subgraphs


class PatternOptimization:
    """
    Defines an optimization pattern.
    """

    def __init__(self):
        pass


class GraphBuilderPatternOptimization:
    """
    Implements optimization after the conversion is done.
    """

    _default_patterns = []

    def __init__(
        self,
        builder: "GraphBuilder",  # noqa: F821
        patterns: Optional[List[PatternOptimization]] = None,
        verbose: int = 0,
    ):
        self.builder = builder
        self.patterns = patterns or self._default_patterns
        self.verbose = verbose
        self._build()

    def iter_nodes(self) -> Iterator:
        for node in self.builder.nodes:
            yield node

    def _build(self):
        """
        Builds successor and predecessor.
        """
        self.nodes_ = {}
        for node in self.iter_nodes():
            self.nodes_[id(node)] = node

        self.predecessors_ = {}
        self.successors_ = {}
        self.used_ = {}
        for k, v in self.nodes_.items():
            for o in v.output:
                self.predecessors_[o] = k
            for i in v.input:
                if i not in self.successors_:
                    self.successors_[i] = []
                self.successors_[i].append(k)

            for sub in enumerate_subgraphs(v):
                g = sub[-1]
                sub_knowns = set()
                for n in g.input:
                    sub_knowns.add(n.name)
                for n in g.initializer:
                    sub_knowns.add(n.name)
                for n in g.sparse_initializer:
                    sub_knowns.add(n.name)
                for n in g.node:
                    for i in n.input:
                        if i not in sub_knowns:
                            # an input coming from the parent
                            self.used_.add(i)
                    for i in n.output:
                        sub_knowns.add(i)

    def optimize(self, max_iter=-1):
        """
        Optimizes the based on the given list of patterns.

        :param max_iter: maximum number of iterations
        """
