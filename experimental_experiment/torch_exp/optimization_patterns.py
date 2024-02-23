from typing import List, Optional, Union
from .optimization_patterns_api import PatternOptimization
from ._optimization_onnx_patterns import (
    CastPattern,
    ExpandPattern,
    ReshapeMatMulReshapePattern,
    ReshapeReshapePattern,
    RotaryConcatPartPattern,
    TransposeMatMulPattern,
    TransposeTransposePattern,
    UnsqueezeUnsqueezePattern,
)


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
        ReshapeMatMulReshapePattern(),
        ReshapeReshapePattern(),
        RotaryConcatPartPattern(),
        TransposeMatMulPattern(),
        TransposeTransposePattern(),
        UnsqueezeUnsqueezePattern(),
    ]


def get_pattern(obj: Union[PatternOptimization, str]) -> PatternOptimization:
    """
    Returns an optimization pattern based on its name.
    """
    if isinstance(obj, PatternOptimization):
        return obj

    mapping = {
        v.__class__.__name__.replace("Pattern", ""): v for v in get_default_patterns()
    }
    if obj in mapping:
        return mapping[obj]
    raise RuntimeError(f"Unable to find pattern for {obj!r}.")


def get_pattern_list(
    positive_list: Optional[Union[str, List[Union[str, type]]]] = "default",
    negative_list: Optional[Union[str, List[Union[str, type]]]] = None,
):
    """
    Builds a list of patterns based on two lists, negative and positive.

    .. runpython::
        :showcode:

        from experimental_experiment.torch_exp.optimization_patterns import get_pattern_list
        print(get_pattern_list("default", ["Cast"]))
    """
    if positive_list is None:
        return []
    if isinstance(positive_list, str):
        assert positive_list == "default", f"List {positive_list!r} is not defined."
        positive_list = get_default_patterns()
    else:
        positive_list = [get_pattern(t) for t in positive_list]

    if negative_list is None:
        return positive_list
    if isinstance(negative_list, str):
        assert negative_list == "default", f"List {negative_list!r} is not defined."
        negative_list = get_default_patterns()
    else:
        negative_list = [get_pattern(t) for t in negative_list]

    disabled = [get_pattern(t) for t in negative_list]
    res = []
    for p in positive_list:
        if p in disabled:
            continue
        res.append(p)
    return res
