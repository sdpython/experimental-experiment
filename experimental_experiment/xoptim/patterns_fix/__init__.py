from typing import List


def get_fix_patterns(
    verbose: int = 0,
) -> List["PatternOptimization"]:  # noqa: F821
    """
    Returns a default list of optimization patterns for experimentation.
    It is equal to the following list.

    .. runpython::
        :showcode:
        :rst:

        from experimental_experiment.xoptim.patterns_api import pattern_table_doc
        from experimental_experiment.xoptim.patterns_fix import get_fix_patterns

        print(pattern_table_doc(get_fix_patterns(), as_rst=True))
    """
    from .add_reduction_scatter_nd import AddReductionScatterND

    return [
        AddReductionScatterND(verbose=verbose),
    ]
