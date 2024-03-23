from typing import List


def get_onnxruntime_patterns(
    verbose: int = 0,
) -> List["PatternOptimization"]:  # noqa: F821
    """
    Returns a default list of optimization patterns for onnxruntime.
    It is equal to the following list.

    .. runpython::
        :showcode:

        import pprint
        from experimental_experiment.xoptim.patterns_ort import get_onnxruntime_patterns
        pprint.pprint(get_onnxruntime_patterns())
    """
    from .activation_grad import SoftmaxGradPattern
    from .fused_matmul import FusedMatMulPattern
    from .simplified_layer_normalization import SimplifiedLayerNormalizationPattern

    return [
        FusedMatMulPattern(verbose=verbose),
        SimplifiedLayerNormalizationPattern(verbose=verbose),
        SoftmaxGradPattern(verbose=verbose),
    ]
