from typing import List


def get_experimental_patterns(
    verbose: int = 0,
) -> List["PatternOptimization"]:  # noqa: F821
    """
    Returns a default list of optimization patterns for experimentation.
    It is equal to the following list.

    .. runpython::
        :showcode:

        import pprint
        from experimental_experiment.xoptim.patterns_exp import get_experimental_patterns
        pprint.pprint(get_experimental_patterns())
    """
    from .binary_operators import AddAddMulMulPattern, AddMulPattern, MulSigmoidPattern
    from .constant_of_shape_scatter_nd import ConstantOfShapeScatterNDPattern
    from .simple_rotary import SimpleRotaryPattern
    from .where_replace import ReplaceZeroPattern

    return [
        AddAddMulMulPattern(verbose=verbose),
        AddMulPattern(verbose=verbose),
        ConstantOfShapeScatterNDPattern(verbose=verbose),
        MulSigmoidPattern(verbose=verbose),
        ReplaceZeroPattern(verbose=verbose),
        SimpleRotaryPattern(verbose=verbose),
    ]
