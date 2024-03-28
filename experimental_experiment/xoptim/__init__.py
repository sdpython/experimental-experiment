import pprint
from typing import List, Optional, Union
from .graph_builder_optim import GraphBuilderPatternOptimization
from .patterns_api import MatchResult, PatternOptimization, EasyPatternOptimization


def get_pattern(
    obj: Union[PatternOptimization, str], as_list: bool = False, verbose: int = 0
) -> PatternOptimization:
    """
    Returns an optimization pattern based on its name.
    """
    if isinstance(obj, PatternOptimization):
        return [obj] if as_list else obj

    from .patterns import get_default_patterns
    from .patterns_ort import get_onnxruntime_patterns
    from .patterns_exp import get_experimental_patterns
    from .patterns_fix import get_fix_patterns
    from .patterns_investigation import get_investigation_patterns

    _pattern = dict(
        default=get_default_patterns,
        onnxruntime=get_onnxruntime_patterns,
        experimental=get_experimental_patterns,
        fix=get_fix_patterns,
        investigation=get_investigation_patterns,
    )

    if isinstance(obj, str):
        sep = "," if "," in obj else ("+" if "+" in obj else None)
        if sep:
            assert as_list, f"Returns a list for obj={obj!r}, as_list must be True."
            objs = obj.split(sep)
            res = []
            for o in objs:
                res.extend(get_pattern(o, as_list=True, verbose=verbose))
            return res

        if obj in _pattern:
            assert as_list, f"Returns a list for obj={obj!r}, as_list must be True."
            return _pattern[obj](verbose=verbose)

    mapping = {
        v.__class__.__name__.replace("Pattern", ""): v
        for v in get_default_patterns(verbose=verbose)
    }
    for fct in _pattern.values():
        mapping.update(
            {
                v.__class__.__name__.replace("Pattern", ""): v
                for v in fct(verbose=verbose)
            }
        )
    if isinstance(obj, list):
        assert as_list, f"obj={obj!r} is already a list"
        res = []
        for s in obj:
            if isinstance(s, str) and s in mapping:
                res.append(mapping[s])
            else:
                res.extend(get_pattern(s, as_list=True, verbose=verbose))
        return res
    if obj in mapping:
        return [mapping[obj]] if as_list else mapping[obj]
    raise RuntimeError(
        f"Unable to find pattern for {obj!r} among {pprint.pformat(mapping)}."
    )


def get_pattern_list(
    positive_list: Optional[Union[str, List[Union[str, type]]]] = "default",
    negative_list: Optional[Union[str, List[Union[str, type]]]] = None,
    verbose: int = 0,
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
    pos_list = get_pattern(positive_list, as_list=True, verbose=verbose)
    if negative_list is None:
        return pos_list

    neg_list = get_pattern(negative_list, as_list=True, verbose=verbose)

    res = []
    for p in pos_list:
        if p in neg_list:
            continue
        res.append(p)
    return res
