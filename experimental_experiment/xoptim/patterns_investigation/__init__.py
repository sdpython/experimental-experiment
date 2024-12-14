from typing import List


def get_investigation_patterns(
    verbose: int = 0,
) -> List["PatternOptimization"]:  # noqa: F821
    """
    Returns a default list of patterns for investigations.
    They do nothing but prints information if verbose > 0.

    .. runpython::
        :showcode:

        import pprint
        from experimental_experiment.xoptim.patterns_investigation import (
            get_investigation_patterns
        )
        pprint.pprint(get_investigation_patterns())
    """
    from .element_wise import BinaryInvestigation
    from .llm_patterns import FunctionPackedMatMulPattern

    return [
        BinaryInvestigation(verbose=verbose),
        FunctionPackedMatMulPattern(verbose=verbose),
    ]
