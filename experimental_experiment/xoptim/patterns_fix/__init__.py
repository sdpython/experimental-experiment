from typing import List


def get_fix_patterns(
    verbose: int = 0,
) -> List["PatternOptimization"]:  # noqa: F821
    """
    Returns a default list of optimization patters for experimentation.
    It is equal to the following list.

    .. runpython::
        :showcode:

        import pprint
        from experimental_experiment.xoptim.patterns_fix import get_fix_patterns
        pprint.pprint(get_fix_patterns())
    """
    from .add_reduction_scatter_nd import AddReductionScatterND

    return [
        AddReductionScatterND(verbose=verbose),
    ]
