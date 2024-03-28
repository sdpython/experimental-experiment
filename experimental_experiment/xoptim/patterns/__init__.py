from typing import List

from ..patterns_api import PatternOptimization

# onnx patterns
from .onnx_cast import CastPattern, CastCastBinaryPattern
from .onnx_expand import ExpandPattern, ExpandBroadcastPattern, ExpandSwapPattern
from .onnx_mul import MulMulMulScalarPattern, SwitchOrderBinaryPattern
from .onnx_matmul import (
    MatMulReshape2Of3Pattern,
    ReshapeMatMulReshapePattern,
    TransposeMatMulPattern,
    TransposeReshapeMatMulPattern,
)
from .onnx_reshape import (
    ReduceReshapePattern,
    Reshape2Of3Pattern,
    ReshapeReshapeBinaryPattern,
    ReshapeReshapePattern,
)
from .onnx_rotary import RotaryConcatPartPattern
from .onnx_split import SlicesSplitPattern
from .onnx_sub import Sub1MulPattern
from .onnx_transpose import TransposeTransposePattern
from .onnx_unsqueeze import UnsqueezeUnsqueezePattern


def get_default_patterns(verbose: int = 0) -> List[PatternOptimization]:
    """
    Returns a default list of optimization patterns.
    It is equal to the following list.

    .. runpython::
        :showcode:

        import pprint
        from experimental_experiment.xoptim.patterns import get_default_patterns
        pprint.pprint(get_default_patterns())
    """
    return [
        CastPattern(verbose=verbose),
        CastCastBinaryPattern(verbose=verbose),
        ExpandPattern(verbose=verbose),
        ExpandBroadcastPattern(verbose=verbose),
        ExpandSwapPattern(verbose=verbose),
        MulMulMulScalarPattern(verbose=verbose),
        ReduceReshapePattern(verbose=verbose),
        ReshapeMatMulReshapePattern(verbose=verbose),
        Reshape2Of3Pattern(verbose=verbose),
        ReshapeReshapeBinaryPattern(verbose=verbose),
        MatMulReshape2Of3Pattern(verbose=verbose),
        ReshapeReshapePattern(verbose=verbose),
        RotaryConcatPartPattern(verbose=verbose),
        SlicesSplitPattern(verbose=verbose),
        Sub1MulPattern(verbose=verbose),
        SwitchOrderBinaryPattern(verbose=verbose),
        TransposeMatMulPattern(verbose=verbose),
        TransposeReshapeMatMulPattern(verbose=verbose),
        TransposeTransposePattern(verbose=verbose),
        UnsqueezeUnsqueezePattern(verbose=verbose),
    ]
