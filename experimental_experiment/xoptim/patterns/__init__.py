from typing import List, Optional, Union

# API
from .patterns_api import MatchResult, PatternOptimization  # noqa: F401

# onnx patterns
from .onnx_cast import CastPattern
from .onnx_expand import ExpandPattern, ExpandBroadcastPattern
from .onnx_mul import MulMulMulScalarPattern
from .onnx_matmul import (
    ReshapeMatMulReshapePattern,
    TransposeMatMulPattern,
    MatMulReshape2Of3Pattern,
)
from .onnx_reshape import (
    ReduceReshapePattern,
    Reshape2Of3Pattern,
    ReshapeReshapePattern,
)
from .onnx_rotary import RotaryConcatPartPattern
from .onnx_sub import Sub1MulPattern
from .onnx_transpose import TransposeTransposePattern
from .onnx_unsqueeze import UnsqueezeUnsqueezePattern


def get_default_patterns() -> List[PatternOptimization]:
    """
    Returns a default list of optimization patters.
    It is equal to the following list.

    .. runpython::
        :showcode:

        import pprint
        from experimental_experiment.xoptim.patterns import get_default_patterns
        pprint.pprint(get_default_patterns())
    """
    return [
        CastPattern(),
        ExpandPattern(),
        ExpandBroadcastPattern(),
        MulMulMulScalarPattern(),
        ReduceReshapePattern(),
        ReshapeMatMulReshapePattern(),
        Reshape2Of3Pattern(),
        MatMulReshape2Of3Pattern(),
        ReshapeReshapePattern(),
        RotaryConcatPartPattern(),
        Sub1MulPattern(),
        TransposeMatMulPattern(),
        TransposeTransposePattern(),
        UnsqueezeUnsqueezePattern(),
    ]


def get_pattern(
    obj: Union[PatternOptimization, str], as_list: bool = False
) -> PatternOptimization:
    """
    Returns an optimization pattern based on its name.
    """
    if isinstance(obj, PatternOptimization):
        return [obj] if as_list else obj

    from ..patterns_ort import get_onnxruntime_patterns

    if isinstance(obj, str):
        _pattern = dict(
            default=get_default_patterns, onnxruntime=get_onnxruntime_patterns
        )
        if obj in _pattern:
            assert as_list, f"Returns a list for obj={obj!r}, as_list must be True."
            return _pattern[obj]()

    mapping = {
        v.__class__.__name__.replace("Pattern", ""): v for v in get_default_patterns()
    }
    mapping.update(
        {
            v.__class__.__name__.replace("Pattern", ""): v
            for v in get_onnxruntime_patterns()
        }
    )
    if isinstance(obj, list):
        assert as_list, f"obj={obj!r} is already a list"
        return [mapping[s] for s in obj]
    if obj in mapping:
        return [mapping[obj]] if as_list else mapping[obj]
    raise RuntimeError(f"Unable to find pattern for {obj!r}.")


def get_pattern_list(
    positive_list: Optional[Union[str, List[Union[str, type]]]] = "default",
    negative_list: Optional[Union[str, List[Union[str, type]]]] = None,
):
    """
    Builds a list of patterns based on two lists, negative and positive.

    .. runpython::
        :showcode:

        import pprint
        from experimental_experiment.xoptim.patterns import get_pattern_list
        pprint.pprint(get_pattern_list("default", ["Cast"]))
    """
    if positive_list is None:
        return []
    if isinstance(positive_list, str):
        pos_list = get_pattern(positive_list, as_list=True)
    else:
        pos_list = []
        for t in positive_list:
            pos_list.extend(get_pattern(t, as_list=True))

    if negative_list is None:
        return pos_list

    if isinstance(positive_list, str):
        neg_list = get_pattern(negative_list, as_list=True)
    else:
        neg_list = []
        for t in negative_list:
            neg_list.extend(get_pattern(t, as_list=True))

    res = []
    for p in pos_list:
        if p in neg_list:
            continue
        res.append(p)
    return res
