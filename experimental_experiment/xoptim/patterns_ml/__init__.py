from typing import List


def get_ml_patterns(
    verbose: int = 0,
) -> List["PatternOptimization"]:  # noqa: F821
    """
    Returns a default list of optimization patterns for ai.onnx.ml.
    It is equal to the following list.

    .. runpython::
        :showcode:
        :rst:

        from experimental_experiment.xoptim.patterns_api import pattern_table_doc
        from experimental_experiment.xoptim.patterns_ml import (
            get_ml_patterns,
        )

        print(pattern_table_doc(get_ml_patterns(), as_rst=True))
    """
    from .tree_ensemble import (
        TreeEnsembleRegressorConcatPattern,
        TreeEnsembleRegressorMulPattern,
    )

    return [
        TreeEnsembleRegressorConcatPattern(verbose=verbose),
        TreeEnsembleRegressorMulPattern(verbose=verbose),
    ]
