from typing import List


def get_onnxruntime_patterns(
    verbose: int = 0,
) -> List["PatternOptimization"]:  # noqa: F821
    """
    Returns a default list of optimization patterns for :epkg:`onnxruntime`.
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
    from .llm_optim import ContribRotaryEmbeddingPattern, ContribRotaryEmbedding3DPattern

    # from .gather_grad import GatherGradPattern
    from .simplified_layer_normalization import (
        SimplifiedLayerNormalizationPattern,
        SimplifiedLayerNormalizationMulPattern,
        SkipLayerNormalizationPattern,
        SkipSimplifiedLayerNormalizationPattern,
        SkipSimplifiedLayerNormalizationMulPattern,
    )

    return [
        BiasGeluPattern(verbose=verbose),
        BiasSoftmaxPattern(verbose=verbose),
        ContribRotaryEmbeddingPattern(verbose=verbose),
        ContribRotaryEmbedding3DPattern(verbose=verbose),
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
        # GatherGradPattern(verbose=verbose),
        ReshapeGemmPattern(verbose=verbose),
        SimplifiedLayerNormalizationPattern(verbose=verbose),
        SimplifiedLayerNormalizationMulPattern(verbose=verbose),
        SkipLayerNormalizationPattern(verbose=verbose),
        SkipSimplifiedLayerNormalizationPattern(verbose=verbose),
        SkipSimplifiedLayerNormalizationMulPattern(verbose=verbose),
        SoftmaxGradPattern(verbose=verbose),
        TransposeFusedMatMulBPattern(verbose=verbose),
    ]
