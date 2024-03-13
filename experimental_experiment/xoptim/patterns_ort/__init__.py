from typing import List


def get_onnxruntime_patterns(
    verbose: int = 0,
) -> List["PatternOptimization"]:  # noqa: F821
    """
    Returns a default list of optimization patters for onnxruntime.
    It is equal to the following list.

    .. runpython::
        :showcode:

        import pprint
        from experimental_experiment.xoptim.patterns_ort import get_onnxruntime_patterns
        pprint.pprint(get_onnxruntime_patterns())
    """
    from .constant_of_shape_scatter_nd import ConstantOfShapeScatterNDPattern
    from .fused_matmul import FusedMatMulPattern

    return [
        ConstantOfShapeScatterNDPattern(verbose=verbose),
        FusedMatMulPattern(verbose=verbose),
    ]
