from typing import List


def get_experimental_patterns(
    verbose: int = 0,
) -> List["PatternOptimization"]:  # noqa: F821
    """
    Returns a default list of optimization patterns for experimentation.
    It is equal to the following list.

    .. runpython::
        :showcode:
        :rst:

        from experimental_experiment.xoptim.patterns_api import pattern_table_doc
        from experimental_experiment.xoptim.patterns_exp import (
            get_experimental_patterns,
        )

        print(pattern_table_doc(get_experimental_patterns(), as_rst=True))
    """
    from .binary_operators import (
        AddAddMulMulPattern,
        AddAddMulMulBroadcastPattern,
        AddMulPattern,
        AddMulBroadcastPattern,
        AddMulSharedInputPattern,
        AddMulSharedInputBroadcastPattern,
        AddMulTransposePattern,
        MulSigmoidPattern,
        NegXplus1Pattern,
        SubMulPattern,
        SubMulBroadcastPattern,
    )
    from .constant_of_shape_scatter_nd import (
        ConstantOfShapeScatterNDPattern,
        MaskedShapeScatterNDPattern,
    )
    from .constants import TriMatrixPattern
    from .simple_rotary import SimpleRotaryPattern
    from .unary_operators import TransposeCastPattern
    from .where_replace import ReplaceZeroPattern

    return [
        AddAddMulMulPattern(verbose=verbose),
        AddAddMulMulBroadcastPattern(verbose=verbose),
        AddMulPattern(verbose=verbose),
        AddMulBroadcastPattern(verbose=verbose),
        AddMulSharedInputPattern(verbose=verbose),
        AddMulSharedInputBroadcastPattern(verbose=verbose),
        AddMulTransposePattern(verbose=verbose),
        ConstantOfShapeScatterNDPattern(verbose=verbose),
        MaskedShapeScatterNDPattern(verbose=verbose),
        MulSigmoidPattern(verbose=verbose),
        NegXplus1Pattern(verbose=verbose),
        ReplaceZeroPattern(verbose=verbose),
        SimpleRotaryPattern(verbose=verbose),
        SubMulPattern(verbose=verbose),
        SubMulBroadcastPattern(verbose=verbose),
        TransposeCastPattern(verbose=verbose),
        TriMatrixPattern(verbose=verbose),
    ]
