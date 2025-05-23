from typing import List


def get_onnxruntime_patterns(
    verbose: int = 0,
) -> List["PatternOptimization"]:  # noqa: F821
    """
    Returns a default list of optimization patterns for onnxruntime.
    It is equal to the following list.

    .. runpython::
        :showcode:
        :rst:

        from experimental_experiment.xoptim.patterns_api import pattern_table_doc
        from experimental_experiment.xoptim.patterns_ort import get_onnxruntime_patterns

        print(pattern_table_doc(get_onnxruntime_patterns(), as_rst=True))
    """
    from .activation import (
        BiasGeluPattern,
        BiasSoftmaxPattern,
        FastGeluPattern,
        GeluOrtPattern,
        GeluErfPattern,
        QuickGeluPattern,
    )
    from .activation_grad import SoftmaxGradPattern
    from .attention_patterns import AttentionPattern
    from .batch_normalization import OrtBatchNormalizationTrainingPattern
    from .fused_conv import FusedConvPattern
    from .fused_matmul import (
        FusedMatMulDivPattern,
        FusedMatMulPattern,
        FusedMatMulx2Pattern,
        FusedMatMulTransposePattern,
        ReshapeGemmPattern,
        TransposeFusedMatMulBPattern,
    )

    # from .llm_optim import RotaryEmbeddingPattern

    # from .gather_grad import GatherGradPattern
    from .simplified_layer_normalization import (
        SimplifiedLayerNormalizationPattern,
        SkipLayerNormalizationPattern,
        SkipSimplifiedLayerNormalizationPattern,
    )

    return [
        AttentionPattern(verbose=verbose),
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
        OrtBatchNormalizationTrainingPattern(verbose=verbose),
        QuickGeluPattern(verbose=verbose),
        # RotaryEmbeddingPattern(verbose=verbose),
        # GatherGradPattern(verbose=verbose),
        ReshapeGemmPattern(verbose=verbose),
        SimplifiedLayerNormalizationPattern(verbose=verbose),
        SkipLayerNormalizationPattern(verbose=verbose),
        SkipSimplifiedLayerNormalizationPattern(verbose=verbose),
        SoftmaxGradPattern(verbose=verbose),
        TransposeFusedMatMulBPattern(verbose=verbose),
    ]
