import textwrap
from typing import List, Optional, Union


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
    :param constant_fusing: similar node Constant and ConstantOfShape are used,
        this options avoids creating new nodes when they are the same
    :param max_iter: maximum number of iteration when doing pattern optimizations,
        -1 to let it undefined
    :param recursive: optimizes subgraphs and functions as well
    :param stop_after: for investigation, stop_after this number of applies patterns,
        -1 to never stop
    :param verbose: verbosity level (for pattern optimization)
    :param verifiers: run verifications to ensure the model is
        correct everytime it is modifies, it is mostly to find bugs,
        it is very slow
    :param dump_applied_patterns: dump applied patterns in a folder,
        the users can check every pattern dumped as a :epkg:`FunctionProto`
    """

    def __init__(
        self,
        remove_unused: bool = True,
        constant_folding: bool = False,
        constant_size: int = 1024,
        constant_fusing: bool = True,
        remove_identity: bool = True,
        patterns: Union[str, List["PatternOptimization"]] = "default",  # noqa: F821
        max_iter: int = -1,
        recursive: bool = False,
        stop_after: int = -1,
        verbose: int = 0,
        verifies: bool = False,
        dump_applied_patterns: Optional[str] = None,
    ):
        self.remove_unused = remove_unused
        self.constant_folding = constant_folding
        self.remove_identity = remove_identity
        self.constant_size = constant_size
        self.constant_fusing = constant_fusing
        self.stop_after = stop_after
        if isinstance(patterns, str):
            from ..xoptim import get_pattern_list

            self.patterns = get_pattern_list(patterns, verbose=verbose)
        else:
            assert patterns is None or isinstance(
                patterns, list
            ), f"Unexpected type {type(patterns)} for patterns"
            from ..xoptim import get_pattern

            self.patterns = (
                None
                if patterns is None
                else [get_pattern(p, verbose=verbose) for p in patterns]
            )
        self.max_iter = -1
        self.verbose = verbose
        self.recursive = recursive
        self.verifies = verifies
        self.dump_applied_patterns = dump_applied_patterns

    def __repr__(self):
        pats = "None" if self.patterns is None else [str(p) for p in self.patterns]
        add = []
        for att in ["verifies", "stop_after", "dump_applied_patterns"]:
            val = getattr(self, att)
            if val in (-1, None, False):
                continue
            add.append(f", {att}={val!r}")
        opts = "".join(add)
        code = (
            f"{self.__class__.__name__}(remove_unused={self.remove_unused}, "
            f"constant_folding={self.constant_folding}, "
            f"constant_size={self.constant_size}, "
            f"constant_fusing={self.constant_fusing}, "
            f"verbose={self.verbose}, "
            f"max_iter={self.max_iter}, recursive={self.recursive}, "
            f"patterns={pats}{opts})"
        )
        return "\n".join(
            textwrap.wrap(code, width=80, tabsize=4, subsequent_indent="    ")
        )
