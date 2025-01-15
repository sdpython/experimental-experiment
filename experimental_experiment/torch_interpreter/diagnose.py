import contextlib
import copy
from typing import Any, Dict, List, Tuple

from ..helpers import string_type


def make_copy(obj: Any) -> Any:
    """
    Makes a copy of the objects.
    """
    return copy.deepcopy(obj)


class ModelDiagnoseOutput:
    """
    Contains inputs and outputs, diagnose results when tracing
    intermediate results. An instance of this class is produced
    by :func:`infer_shape_type_from_execution`.
    """

    def __init__(self, name: str, model: "torch.nn.Module", level: int = 0):  # noqa: F821
        self.name = name
        self.model = model
        self.level = level
        self.forward = model.forward
        self.inputs = []
        self.outputs = []
        self.children: List[ModelDiagnoseOutput] = []

    def pretty_text(
        self, with_dynamic_shape=False, with_shape=True, with_min_max=True, with_device=True
    ) -> str:
        """
        Renders the outputs.

        :param with_dynamic_shape: show dynamic shapes
        :param with_shape: see :func:`experimental_experiment.helpers.string_type`.
        :param with_min_max: see :func:`experimental_experiment.helpers.string_type`.
        :param with_device: see :func:`experimental_experiment.helpers.string_type`.
        :return: text
        """
        assert len(self.inputs) == len(self.outputs), (
            f"Number if inputs / outputs mismatch {len(self.inputs)} != "
            f"{len(self.outputs)}"
        )
        kws = dict(with_shape=with_shape, with_min_max=with_min_max, with_device=with_device)
        indent = "    " * self.level
        rows = [f"{indent}>>> {self.name}: {self.model.__class__.__name__}"]
        if with_dynamic_shape:
            ds = self.guess_dynamic_shapes()
            rows.append(f"{indent}DS={ds}")
        for i in self.inputs:
            rows.append(f"{indent}  > {string_type(i, **kws)}")
        for child in self.children:
            t = child.pretty_text(with_dynamic_shape=with_dynamic_shape, **kws)
            rows.extend((indent + s) for s in t.split("\n"))
        for i in self.outputs:
            rows.append(f"{indent}  < {string_type(i, **kws)}")
        rows.append(f"{indent}<<<")
        return "\n".join(rows)

    @property
    def dot_name(self):
        "Returns a kind of indented name."
        return f"{'..' * self.level}{self.name} - {self.model.__class__.__name__}"

    def add_inputs(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]):
        """Stores used inputs. Makes a copy."""
        self.inputs.append(make_copy((args, kwargs)))

    def add_outputs(self, args: Tuple[Any, ...]):
        """Stores returned outputs. Makes a copy."""
        if not isinstance(args, tuple):
            args = (args,)
        self.outputs.append(make_copy(args))

    def add_child(self, diag: "ModelDiagnoseOutput"):
        """Adds a submodule."""
        self.children.append(diag)

    def guess_dynamic_dimensions(self, *shapes) -> Any:
        """Infers the dynamic dimension from multiple shapes."""
        if len(shapes) == 1:
            return {}

        import torch

        dynamic = torch.export.Dim.DYNAMIC
        set_length = set(len(s) for s in shapes)
        assert len(set_length) == 1, (
            f"Shapes can be different but not ranks possible shapes={set_length} "
            f"shapes={shapes} for module {self.name!r}, "
            f"class={self.model.__class__.__name__!r}"
        )
        rk = set_length.pop()
        res = {}
        for i in range(rk):
            if len(set(s[i] for s in shapes)) > 1:
                res[i] = dynamic
        return res

    def guess_dynamic_shapes(self) -> Any:
        """
        Guesses the dynamic shapes for that module from two execution.
        If there is only one execution, then that would be static dimensions.
        """
        if len(self.inputs) == 1:
            # No dynamic shapes.
            args = tuple({} for _ in self.inputs[0])
            kwargs = {k: {} for k, v in self.inputs[1]}
            return args, kwargs

        # Otherwise.
        s1 = set(len(i[0]) for i in self.inputs)
        assert len(s1) == 1, f"Different numbers of unnamed arguments {s1}"
        s2 = set(tuple(sorted(set(i[1]))) for i in self.inputs)
        assert len(s1) == 1, f"Different named arguments {s2}"
        args = []
        kwargs = {}
        for i in range(s1.pop()):
            shapes = [_[0][i].shape for _ in self.inputs]
            args.append(self.guess_dynamic_dimensions(*shapes))
        for name in s2.pop():
            shapes = [_[1][name].shape for _ in self.inputs]
            kwargs[name] = self.guess_dynamic_dimensions(*shapes)
        return tuple(args), kwargs


def _rewrite_forward(diag: ModelDiagnoseOutput, *args, **kwargs):
    diag.add_inputs(args, kwargs)
    res = diag.forward(*args, **kwargs)
    diag.add_outputs(res)
    return res


def _trace_forward_execution(
    model: "torch.nn.Module",  # noqa: F821
    name: str = "__main__",
    level: int = 0,
    verbose: int = 0,
):
    diag = ModelDiagnoseOutput(name, model, level=level)
    if verbose:
        print(f"[_trace_forward_execution] {diag.dot_name}")
    model.forward = lambda *args, _diag=diag, **kwargs: _rewrite_forward(
        _diag, *args, **kwargs
    )
    for name, mod in model.named_children():
        d = _trace_forward_execution(mod, name, verbose=verbose, level=level + 1)
        diag.add_child(d)
    return diag


def _untrace_forward_execution(diag: ModelDiagnoseOutput, verbose: int = 0):
    if verbose:
        print(f"[_untrace_forward_execution] {diag.dot_name}")
    diag.model.forward = diag.forward
    for child in diag.children:
        _untrace_forward_execution(child, verbose=verbose)


@contextlib.contextmanager
def trace_forward_execution(
    model: "torch.nn.Module", verbose: int = 0  # noqa: F821
) -> ModelDiagnoseOutput:
    """
    Replaces all forward to store the inputs and outputs of the module
    and every submodules.
    """
    diag = _trace_forward_execution(model, verbose=verbose)
    try:
        yield diag
    finally:
        _untrace_forward_execution(diag, verbose=verbose)


def infer_shape_type_from_execution(
    model: "torch.nn.Module",  # noqa: F821
    inputs: List[Tuple[Tuple[Any, ...], Dict[str, Any]]],
    verbose: int = 0,
) -> ModelDiagnoseOutput:
    """
    Runs a model, traces the intermediate output and infers dynamic shapes
    based on it.

    :param model: model
    :param inputs: list of input sets with different shapes
        (at least for the dynamic dimensions)
    :param verbose: verbosity
    :return: see :class:`ModelDiagnoseOutput`
    """
    with trace_forward_execution(model, verbose=verbose) as tracer:
        for i in inputs:
            if verbose:
                print(
                    f"[infer_shape_type_from_execution] run with "
                    f"{string_type(i, with_shape=True)}"
                )
            assert (
                isinstance(i, tuple)
                and len(i) == 2
                and isinstance(i[0], tuple)
                and isinstance(i[1], dict)
            ), (
                f"Unexpected types as inputs, it should (args, kwargs) but got "
                f"{string_type(i)}"
            )
            args, kwargs = i
            model(*args, **kwargs)
        if verbose:
            print(
                f"[trace_forward_execution] traced execution of model "
                f"{model.__class__.__name__}"
            )
            print(tracer.pretty_text())
        return tracer
