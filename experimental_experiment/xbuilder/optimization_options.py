import textwrap
from typing import List, Union


class OptimizationOptions:
    """
    Defines all the optimization to apply.

    :param remove_unused: remove all unused nodes, this must be true if
        pattern optimization is enabled
    :param constant_folding: folds constant as much as possible
    :param constant_size: all node Constant above this threshold should be
        defined as initializer
    :param remove_identity: remove identity nodes
    :param patterns: list of pattern optimization to apply to the graph,
        it looks a a specific subsequence of nodes in a graph
        and do some replacements,
        `'default'` means a default list of optimization patterns are applied
    :param max_iter: maximum number of iteration when doing pattern optimizations,
        -1 to let it undefined
    :param recursive: optimizes subgraphs and functions as well
    :param verbose: verbosity level (for pattern optimization)
    """

    def __init__(
        self,
        remove_unused: bool = True,
        constant_folding: bool = False,
        constant_size: int = 1024,
        remove_identity: bool = True,
        patterns: Union[str, List["PatternOptimization"]] = "default",  # noqa: F821
        max_iter: int = -1,
        recursive: bool = False,
        verbose: int = 0,
    ):
        self.remove_unused = remove_unused
        self.constant_folding = constant_folding
        self.remove_identity = remove_identity
        self.constant_size = constant_size
        if isinstance(patterns, str):
            from ..xoptim.patterns import get_pattern_list

            self.patterns = get_pattern_list(patterns)
        else:
            assert patterns is None or isinstance(
                patterns, list
            ), f"Unexpected type {type(patterns)} for patterns"
            from ..xoptim.patterns import get_pattern

            self.patterns = (
                None if patterns is None else [get_pattern(p) for p in patterns]
            )
        self.max_iter = -1
        self.verbose = verbose
        self.recursive = recursive

    def __repr__(self):
        pats = "None" if self.patterns is None else [str(p) for p in self.patterns]
        code = (
            f"{self.__class__.__name__}(remove_unused={self.remove_unused}, "
            f"constant_folding={self.constant_folding}, "
            f"constant_size={self.constant_size}, verbose={self.verbose}, "
            f"max_iter={self.max_iter}, recursive={self.recursive}, patterns={pats})"
        )
        return "\n".join(
            textwrap.wrap(code, width=80, tabsize=4, subsequent_indent="    ")
        )
