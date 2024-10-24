from typing import Any, Iterator


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
