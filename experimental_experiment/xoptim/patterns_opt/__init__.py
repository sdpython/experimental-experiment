from typing import List

# API
from .patterns_api import MatchResult, PatternOptimization  # noqa: F401

# ort patterns
from .ort_constant_of_shape_scatter_nd import ConstantOfShapeScatterNDPattern


def get_onnxruntime_patterns() -> List[PatternOptimization]:
    """
    Returns a default list of optimization patters for onnxruntime.
    It is equal to the following list.

    .. runpython::
        :showcode:

        import pprint
        from experimental_experiment.torch_exp.optimization_patterns import get_onnxruntime_patterns
        pprint.pprint(get_onnxruntime_patterns())
    """
    return [
        ConstantOfShapeScatterNDPattern(),
    ]
