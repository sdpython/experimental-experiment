import pprint
from typing import Any, Dict, Iterator


def flatten_outputs(output: Any) -> Iterator[Any]:
    """
    Flattens output results.
    """
    if isinstance(output, (list, tuple)):
        for item in output:
            yield from flatten_outputs(item)
    elif isinstance(output, dict):
        yield from flatten_outputs(list(output.values()))
    elif hasattr(output, "to_tuple"):
        yield from flatten_outputs(output.to_tuple())
    else:
        yield output


def assert_found(kwargs: Dict[str, Any], config: Dict[str, Any]):
    """
    Checks a parameter is available.
    """
    for k in kwargs:
        assert (
            k in config or k == "_attn_implementation"
        ), f"Parameter {k!r} is not mentioned in the configuration {pprint.pformat(config)}"
