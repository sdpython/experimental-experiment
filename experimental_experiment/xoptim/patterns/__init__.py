from typing import List, Optional, Union

# API
from .patterns_api import MatchResult, PatternOptimization

# onnx patterns
from .onnx_cast import CastPattern
from .onnx_expand import ExpandPattern, ExpandBroadcastPattern
from .onnx_mul import MulMulMulPattern
from .onnx_matmul import ReshapeMatMulReshapePattern, TransposeMatMulPattern
from .onnx_reshape import ReshapeReshapePattern
from .onnx_rotary import RotaryConcatPartPattern
from .onnx_sub import Sub1MulPattern
from .onnx_transpose import TransposeTransposePattern
from .onnx_unsqueeze import UnsqueezeUnsqueezePattern

# ort patterns
from .ort_constant_of_shape_scatter_nd import ConstantOfShapeScatterNDPattern




def get_default_patterns() -> List[PatternOptimization]:
    """
    Returns a default list of optimization patters.
    It is equal to the following list.

    .. runpython::
        :showcode:

        import pprint
        from experimental_experiment.torch_exp.optimization_patterns import get_default_patterns
        pprint.pprint(get_default_patterns())
    """
    return [
        CastPattern(),
        ExpandPattern(),
        ExpandBroadcastPattern(),
        MulMulMulPattern(),
        ReshapeMatMulReshapePattern(),
        ReshapeReshapePattern(),
        RotaryConcatPartPattern(),
        Sub1MulPattern(),
        TransposeMatMulPattern(),
        TransposeTransposePattern(),
        UnsqueezeUnsqueezePattern(),
    ]


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
    from .patterns import ConstantOfShapeScatterNDPattern

    return [
        ConstantOfShapeScatterNDPattern(),
    ]


def get_pattern(
    obj: Union[PatternOptimization, str], as_list: bool = False
) -> PatternOptimization:
    """
    Returns an optimization pattern based on its name.
    """
    if isinstance(obj, PatternOptimization):
        return [obj] if as_list else obj

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
        from experimental_experiment.torch_exp.optimization_patterns import get_pattern_list
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
