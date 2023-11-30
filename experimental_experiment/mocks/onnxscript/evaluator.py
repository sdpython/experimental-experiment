import contextlib
from typing import Any, Mapping, Sequence
import onnx
from .onnx_function import OnnxFunction


class Evaluator:
    """Protocol for evaluating ONNX ops."""

    def eval(
        self,
        schema: onnx.defs.OpSchema,
        inputs: Sequence[Any],
        attributes: Mapping[str, Any],
    ):
        """Evaluates an ONNX op.

        Args:
            schema: The OpSchema of the operator to evaluate.
            inputs: The ONNX inputs to the op.
            attributes: The ONNX attributes to the op.
        """
        raise NotImplementedError(
            f"Method 'eval' is not overloaded in class {self.__class__.__name__!r}."
        )

    def eval_function(
        self,
        function: OnnxFunction,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
    ):
        """Evaluates an OnnxFunction.

        Args:
            function: The OnnxFunction to evaluate.
            args: The positional arguments to the function.
            kwargs: The keyword arguments to the function.
        """
        raise NotImplementedError(
            f"Method 'eval_function' is not overloaded in class {self.__class__.__name__!r}."
        )


class DefaultEvaluator(Evaluator):
    def __call__(self, *args, **kwargs):
        """Implements an eager-mode execution."""
        return default().eval_function(self, args, kwargs)


_default_evaluator: Evaluator = DefaultEvaluator()


def default() -> Evaluator:
    """Returns the default Evaluator default."""
    return _default_evaluator


def set_default(new_default: Evaluator) -> None:
    """Sets the current Evaluator default."""
    global _default_evaluator
    _default_evaluator = new_default


@contextlib.contextmanager
def default_as(temp_default: Evaluator):
    """Context manager that temporarily switches the default evaluator."""
    old_default = _default_evaluator
    set_default(temp_default)
    try:
        yield
    finally:
        set_default(old_default)
