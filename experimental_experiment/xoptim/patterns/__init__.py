import inspect
from typing import List, Optional
from onnx import NodeProto
from ..patterns_api import PatternOptimization, MatchResult

# onnx patterns
from .onnx_any import IdentityPattern, SameChildrenPattern
from .onnx_cast import (
    CastPattern,
    CastCastBinaryPattern,
    CastOpCastPattern,
    ComputationCastOpCastPattern,
)
from .onnx_clip import ClipClipPattern
from .onnx_concat import ConcatEmptyPattern, ConcatGatherPattern
from .onnx_constants import ConstantToInitializerPattern  # noqa: F401
from .onnx_conv import ConvBiasNullPattern
from .onnx_dropout import DropoutPattern
from .onnx_equal import UnsqueezeEqualPattern
from .onnx_expand import ExpandPattern, ExpandBroadcastPattern, ExpandSwapPattern
from .onnx_functions import GeluPattern, LeakyReluPattern, SoftmaxCrossEntropyLossCastPattern
from .onnx_layer_normalization import (
    BatchNormalizationPattern,
    BatchNormalizationTrainingPattern,
    CastLayerNormalizationCastPattern,
    LayerNormalizationPattern,
    LayerNormalizationScalePattern,
)
from .onnx_mul import (
    MulMulMulScalarPattern,
    SwitchOrderBinaryPattern,
)
from .onnx_matmul import (
    GemmTransposePattern,
    MatMulAddPattern,
    MatMulReshape2Of3Pattern,
    MulMulMatMulPattern,
    ReshapeMatMulReshapePattern,
    SwitchReshapeActivationPattern,
    TransposeMatMulPattern,
    TransposeReshapeMatMulPattern,
)
from .onnx_reduce import ReduceSumNormalizePattern
from .onnx_reshape import (
    ConcatReshapePattern,
    ReshapePattern,
    ReduceReshapePattern,
    Reshape2Of3Pattern,
    ReshapeReshapeBinaryPattern,
    ReshapeReshapePattern,
    ShapeBasedEditDistanceReshapePattern,
    ShapeBasedReshapeIsSqueezePattern,
    StaticConcatReshapePattern,
)
from .onnx_rotary import RotaryConcatPartPattern, RotaryEmbeddingPattern
from .onnx_sequence import SequenceConstructAtPattern
from .onnx_slice import SliceSlicePattern
from .onnx_split import SlicesSplitPattern, SplitConcatPattern
from .onnx_sub import Sub1MulPattern
from .onnx_transpose import (
    TransposeTransposePattern,
    TransposeReshapeTransposePattern,
    TransposeEqualReshapePattern,
)
from .onnx_unsqueeze import (
    SqueezeUnsqueezePattern,
    UnsqueezeUnsqueezePattern,
    SqueezeAddPattern,
)


class AlmostDoNothingPattern(PatternOptimization):
    """Checks that a Expand is really needed."""

    n_count = 0

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if self.n_count >= 0:
            return self.none()
        if node.op_type != "Pow" or node.domain != "":
            return self.none()
        if node.name is not None and "AlmostDoNothing" in node.name:
            return self.none(node, inspect.currentframe().f_lineno)
        self.n_count += 1
        return MatchResult(self, [node], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node: NodeProto,
    ) -> List[NodeProto]:
        return [
            g.make_node(
                node.op_type,
                node.input,
                node.output,
                name=f"AlmostDoNothing--{node.name}",
            )
        ]


def get_default_patterns(verbose: int = 0) -> List[PatternOptimization]:
    """
    Returns a default list of optimization patterns.
    It is equal to the following list.

    .. runpython::
        :showcode:
        :rst:

        from experimental_experiment.xoptim.patterns_api import pattern_table_doc
        from experimental_experiment.xoptim.patterns import get_default_patterns

        print(pattern_table_doc(get_default_patterns(), as_rst=True))
    """
    return [
        # AlmostDoNothingPattern(verbose=verbose),
        BatchNormalizationPattern(verbose=verbose),
        BatchNormalizationTrainingPattern(verbose=verbose),
        CastLayerNormalizationCastPattern(verbose=verbose),
        CastPattern(verbose=verbose),
        CastCastBinaryPattern(verbose=verbose),
        CastOpCastPattern(verbose=verbose),
        ClipClipPattern(verbose=verbose),
        ComputationCastOpCastPattern(verbose=verbose),
        ConcatEmptyPattern(verbose=verbose),
        ConcatGatherPattern(verbose=verbose),
        ConcatReshapePattern(verbose=verbose),
        ConvBiasNullPattern(verbose=verbose),
        DropoutPattern(verbose=verbose),
        ExpandPattern(verbose=verbose),
        ExpandBroadcastPattern(verbose=verbose),
        ExpandSwapPattern(verbose=verbose),
        GeluPattern(verbose=verbose),
        IdentityPattern(verbose=verbose),
        LayerNormalizationPattern(verbose=verbose),
        LayerNormalizationScalePattern(verbose=verbose),
        LeakyReluPattern(verbose=verbose),
        MulMulMulScalarPattern(verbose=verbose),
        ReduceReshapePattern(verbose=verbose),
        ReduceSumNormalizePattern(verbose=verbose),
        ReshapePattern(verbose=verbose),
        ReshapeMatMulReshapePattern(verbose=verbose),
        Reshape2Of3Pattern(verbose=verbose),
        ReshapeReshapeBinaryPattern(verbose=verbose),
        MatMulAddPattern(verbose=verbose),
        GemmTransposePattern(verbose=verbose),
        MatMulReshape2Of3Pattern(verbose=verbose),
        MulMulMatMulPattern(verbose=verbose),
        ShapeBasedEditDistanceReshapePattern(verbose=verbose),
        ShapeBasedReshapeIsSqueezePattern(verbose=verbose),
        ReshapeReshapePattern(verbose=verbose),
        RotaryConcatPartPattern(verbose=verbose),
        RotaryEmbeddingPattern(verbose=verbose),
        SameChildrenPattern(verbose=verbose),
        SequenceConstructAtPattern(verbose=verbose),
        SliceSlicePattern(verbose=verbose),
        SlicesSplitPattern(verbose=verbose),
        SoftmaxCrossEntropyLossCastPattern(verbose=verbose),
        SplitConcatPattern(verbose=verbose),
        SqueezeAddPattern(verbose=verbose),
        SqueezeUnsqueezePattern(verbose=verbose),
        StaticConcatReshapePattern(verbose=verbose),
        Sub1MulPattern(verbose=verbose),
        SwitchOrderBinaryPattern(verbose=verbose),
        SwitchReshapeActivationPattern(verbose=verbose),
        TransposeEqualReshapePattern(verbose=verbose),
        TransposeMatMulPattern(verbose=verbose),
        TransposeReshapeMatMulPattern(verbose=verbose),
        TransposeReshapeTransposePattern(verbose=verbose),
        TransposeTransposePattern(verbose=verbose),
        UnsqueezeEqualPattern(verbose=verbose),
        UnsqueezeUnsqueezePattern(verbose=verbose),
    ]
