import pprint
from typing import Any, Dict, Iterator


def flatten_outputs(output: Any) -> Iterator[Any]:
    """Flattens output results."""
    if isinstance(output, (list, tuple)):
        for item in output:
            yield from flatten_outputs(item)
    elif isinstance(output, dict):
        yield from flatten_outputs(list(output.values()))
    elif hasattr(output, "to_tuple"):
        yield from flatten_outputs(output.to_tuple())
    elif hasattr(output, "shape"):
        yield output
    elif output.__class__.__name__ == "MambaCache":
        if isinstance(output.conv_states, list):
            yield from flatten_outputs(output.conv_states)
            yield from flatten_outputs(output.ssm_states)
        else:
            yield output.conv_states
            yield output.ssm_states
    elif output.__class__.__name__ == "DynamicCache":
        yield output.key_cache
        yield output.value_cache
    else:
        raise TypeError(f"Unable to flatten type {type(output)}")


def assert_found(kwargs: Dict[str, Any], config: Dict[str, Any]):
    """Checks a parameter is available."""
    for k in kwargs:
        assert (
            k in config or k == "_attn_implementation"
        ), f"Parameter {k!r} is not mentioned in the configuration {pprint.pformat(config)}"
