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
    from .activation import (
        BiasGeluPattern,
        BiasSoftmaxPattern,
        FastGeluPattern,
        GeluOrtPattern,
        GeluErfPattern,
    )
    from .activation_grad import SoftmaxGradPattern
    from .fused_conv import FusedConvPattern
    from .fused_matmul import (
        FusedMatMulDivPattern,
        FusedMatMulPattern,
        FusedMatMulx2Pattern,
        FusedMatMulTransposePattern,
    )

    # from .gather_grad import GatherGradPattern
    from .simplified_layer_normalization import SimplifiedLayerNormalizationPattern

    return [
        BiasGeluPattern(verbose=verbose),
        BiasSoftmaxPattern(verbose=verbose),
        GeluOrtPattern(verbose=verbose),
        GeluErfPattern(verbose=verbose),
        FusedConvPattern(verbose=verbose),
        FastGeluPattern(verbose=verbose),
        FusedMatMulPattern(verbose=verbose),
        FusedMatMulx2Pattern(verbose=verbose),
        FusedMatMulDivPattern(verbose=verbose),
        FusedMatMulTransposePattern(verbose=verbose),
        # GatherGradPattern(verbose=verbose),
        SimplifiedLayerNormalizationPattern(verbose=verbose),
        SoftmaxGradPattern(verbose=verbose),
    ]
