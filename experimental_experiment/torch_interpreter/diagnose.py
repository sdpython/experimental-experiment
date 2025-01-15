import contextlib
import copy
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import torch
from ..helpers import string_type


def make_copy(obj: Any) -> Any:
    """
    Makes a copy of the objects.
    """
    if isinstance(obj, np.ndarray):
        return obj.copy()
    if isinstance(obj, tuple):
        return tuple(make_copy(_) for _ in obj)
    if isinstance(obj, list):
        return [make_copy(_) for _ in obj]
    if isinstance(obj, dict):
        return {k: make_copy(v) for k, v in obj.items()}
    if hasattr(obj, "clone"):
        return obj.clone()
    if obj.__class__.__name__ == "DynamicCache":
        import transformers

        assert isinstance(
            obj, transformers.cache_utils.DynamicCache
        ), f"Unexpected type {string_type(obj)}"
        cache = obj.__class__()
        cache._seen_tokens = obj.seen_tokens
        cache.key_cache = make_copy(obj.key_cache)
        cache.value_cache = make_copy(obj.value_cache)
        return cache
    try:
        return copy.deepcopy(obj)
    except RuntimeError as e:
        raise RuntimeError(
            f"deepcopy did not work on type {type(obj)}: {string_type(obj)}"
        ) from e


class ModelDiagnoseOutput:
    """
    Contains inputs and outputs, diagnose results when tracing
    intermediate results. An instance of this class is produced
    by :func:`infer_shape_type_from_execution`.
    """

    def __init__(self, name: str, model: torch.nn.Module, level: int = 0):
        self.name = name
        self.model = model
        self.level = level
        self.forward = model.forward
        self.inputs = []
        self.outputs = []
        self.children: List[ModelDiagnoseOutput] = []
        assert not isinstance(model, torch.nn.ModuleList), "ModuleList should not be traced."

    def pretty_text(
        self,
        with_dynamic_shape: bool = False,
        with_shape: bool = True,
        with_min_max: bool = True,
        with_device: bool = True,
        with_inputs: bool = True,
    ) -> str:
        """
        Renders the outputs.

        :param with_dynamic_shape: show dynamic shapes
        :param with_shape: see :func:`experimental_experiment.helpers.string_type`.
        :param with_min_max: see :func:`experimental_experiment.helpers.string_type`.
        :param with_device: see :func:`experimental_experiment.helpers.string_type`.
        :param with_inputs: show inputs and outputs shapes
        :return: text
        """
        assert len(self.inputs) == len(self.outputs), (
            f"Number if inputs / outputs mismatch {len(self.inputs)} != "
            f"{len(self.outputs)}"
        )
        kws = dict(with_shape=with_shape, with_min_max=with_min_max, with_device=with_device)
        indent = "    " * self.level
        if not self.children and not with_inputs and not any(kws.values()):
            return (
                (
                    f"{indent}>>> {self.name}: {self.model.__class__.__name__}: "
                    f"DS={self.guess_dynamic_shapes()} <<<"
                )
                if with_dynamic_shape
                else f"{indent}>>> {self.name}: {self.model.__class__.__name__} <<<"
            )
        rows = [f"{indent}>>> {self.name}: {self.model.__class__.__name__}"]
        if with_dynamic_shape:
            ds = self.guess_dynamic_shapes()
            rows.append(f"{indent}  DS={ds}")
        if with_inputs:
            for i in self.inputs:
                rows.append(f"{indent}  > {string_type(i, **kws)}")
        for child in self.children:
            t = child.pretty_text(
                with_dynamic_shape=with_dynamic_shape, with_inputs=with_inputs, **kws
            )
            rows.extend((indent + s) for s in t.split("\n"))
        if with_inputs:
            for i in self.outputs:
                rows.append(f"{indent}  < {string_type(i, **kws)}")
        rows.append(f"{indent}<<<")
        return "\n".join(rows)

    @property
    def dot_name(self):
        "Returns a kind of indented name."
        return f"{'..' * self.level}{self.name} - {self.model.__class__.__name__}"

    @property
    def module_name_type(self):
        "Returns name and module type."
        return f"type({self.name})={self.model.__class__.__name__}"

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

    def guess_dynamic_dimensions(self, *tensors) -> Any:
        """Infers the dynamic dimension from multiple shapes."""
        if len(tensors) == 1:
            return {}
        shapes = [t.shape for t in tensors]
        set_length = set(len(s) for s in shapes)
        assert len(set_length) == 1, (
            f"Shapes can be different but not ranks possible shapes={set_length} "
            f"shapes={shapes} for module {self.name!r}, "
            f"class={self.model.__class__.__name__!r}"
        )
        dynamic = torch.export.Dim.DYNAMIC
        rk = set_length.pop()
        res = {}
        for i in range(rk):
            if len(set(s[i] for s in shapes)) > 1:
                res[i] = dynamic
        return res

    def guess_dynamic_shape_object(self, *objs: Any, msg: Optional[Callable] = None) -> Any:
        """
        Guesses the dynamic shapes for one argument.
        """
        assert (
            len(objs) > 1
        ), f"Unable to infer shapes with only one object {string_type(objs)}"
        set_types = set(type(o) for o in objs)
        assert (
            len(set_types) == 1
        ), f"Unexpected variety of input type {set_types}{msg() if msg else ''})"
        obj = objs[0]
        if obj is None:
            return None
        if isinstance(obj, (bool, int, float, str)):
            return None
        if isinstance(obj, (torch.Tensor, np.ndarray)):
            return self.guess_dynamic_dimensions(*objs)

        if isinstance(obj, tuple):
            kl = set(len(o) for o in objs)
            assert (
                len(kl) == 1
            ), f"Unexpected variety of tuple lengths {kl}{msg() if msg else ''}"
            shapes = []
            for i in range(kl.pop()):
                shapes.append(self.guess_dynamic_shape_object(*[o[i] for o in objs]))
            return tuple(shapes)

        if isinstance(obj, list):
            kl = set(len(o) for o in objs)
            assert (
                len(kl) == 1
            ), f"Unexpected variety of list lengths {kl}{msg() if msg else ''}"
            shapes = []
            for i in range(kl.pop()):
                shapes.append(self.guess_dynamic_shape_object(*[o[i] for o in objs]))
            return shapes

        if obj.__class__.__name__ == "DynamicCache":
            import transformers

            assert isinstance(
                obj, transformers.cache_utils.DynamicCache
            ), f"Unexpected type {string_type(obj)} in {self.module_name_type}"
            kc = set(len(o.key_cache) for o in objs)
            assert (
                len(kc) == 1
            ), f"All attribute 'key_cache' should have the same length but found {kc}"
            vc = set(len(o.value_cache) for o in objs)
            assert (
                len(vc) == 1
            ), f"All attribute 'value_cache' should have the same length but found {vc}"
            key_cache = []
            for i in range(kc.pop()):
                key_cache.append(
                    self.guess_dynamic_dimensions(*[o.key_cache[i] for o in objs])
                )
            value_cache = []
            for i in range(vc.pop()):
                value_cache.append(
                    self.guess_dynamic_dimensions(*[o.value_cache[i] for o in objs])
                )
            return [key_cache, value_cache]

        raise NotImplementedError(
            f"Unable to build dynamic shapes for type {set_types.pop()}: "
            f"{string_type(objs)}{msg() if msg else ''} in {self.module_name_type}"
        )

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
            objs = [_[0][i] for _ in self.inputs]
            args.append(
                self.guess_dynamic_shape_object(*objs, msg=lambda i=i: f" failing input {i}")
            )
        for name in s2.pop():
            objs = [_[1][name] for _ in self.inputs]
            kwargs[name] = self.guess_dynamic_shape_object(
                *objs, msg=lambda name=name: f" failing input {name!r}"
            )
        return tuple(args), kwargs


def _rewrite_forward(
    *args, _diag: Optional[ModelDiagnoseOutput] = None, verbose: int = 0, **kwargs
):
    assert _diag is not None, "_diag cannot be None"
    if verbose:
        indent = "  " * _diag.level
        if not args:
            print(
                f"[{_diag.name}:{_diag.model.__class__.__name__}] "
                f"{indent}> **{string_type(kwargs)}"
            )
        elif kwargs:
            print(
                f"[{_diag.name}:{_diag.model.__class__.__name__}] "
                f"{indent}> *{string_type(args)}, **{string_type(kwargs)}"
            )
        else:
            if len(args) == 1 and isinstance(args[0], torch.Tensor):
                print(
                    f"[{_diag.name}:{_diag.model.__class__.__name__}] "
                    f"{indent}> {string_type(args[0])}"
                )
            else:
                print(
                    f"[{_diag.name}:{_diag.model.__class__.__name__}] "
                    f"{indent}> *{string_type(args)}"
                )
    _diag.add_inputs(args, kwargs)
    res = _diag.forward(*args, **kwargs)
    _diag.add_outputs(res)
    if verbose:
        if isinstance(res, torch.Tensor):
            print(
                f"[{_diag.name}:{_diag.model.__class__.__name__}] "
                f"{indent}< {string_type(res)}"
            )
        else:
            print(
                f"[{_diag.name}:{_diag.model.__class__.__name__}] "
                f"{indent}< *{string_type(res)}"
            )
    return res


def _trace_forward_execution(
    model: torch.nn.Module,
    name: str = "__main__",
    level: int = 0,
    verbose: int = 0,
):
    diag = ModelDiagnoseOutput(name, model, level=level)
    if verbose:
        print(f"[_trace_forward_execution] {diag.dot_name}")
    model.forward = lambda *args, _diag=diag, verbose=verbose, **kwargs: _rewrite_forward(
        *args, _diag=_diag, verbose=verbose, **kwargs
    )
    for name, mod in model.named_children():
        if isinstance(mod, torch.nn.ModuleList):
            for i, m in enumerate(mod):
                d = _trace_forward_execution(
                    m, f"{name}[{i}]", verbose=verbose + 1, level=level + 1
                )
                diag.add_child(d)
        else:
            d = _trace_forward_execution(mod, name, verbose=verbose + 1, level=level + 1)
            diag.add_child(d)
    return diag


def _untrace_forward_execution(diag: ModelDiagnoseOutput, verbose: int = 0):
    if verbose:
        print(f"[_untrace_forward_execution] {diag.dot_name}")
    diag.model.forward = diag.forward
    for child in diag.children:
        _untrace_forward_execution(child, verbose=verbose)


@contextlib.contextmanager
def trace_forward_execution(model: torch.nn.Module, verbose: int = 0) -> ModelDiagnoseOutput:
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
    model: torch.nn.Module,
    inputs: List[Tuple[Tuple[Any, ...], Dict[str, Any]]],
    verbose: int = 0,
) -> ModelDiagnoseOutput:
    """
    Runs a model, traces the intermediate output and infers dynamic shapes
    based on it.

    :param model: model
    :param inputs: list of input sets ``[(args, kwargs), (args, kwargs), ...]``
        with different shapes (at least for the dynamic dimensions)
    :param verbose: verbosity
    :return: see :class:`ModelDiagnoseOutput`
    """
    with trace_forward_execution(model, verbose=verbose) as tracer:
        for i in inputs:
            if isinstance(i, dict):
                i = (tuple(), i)
            elif isinstance(i, tuple) and (
                len(i) != 2 or not isinstance(i[0], tuple) or not isinstance(i[1], dict)
            ):
                i = (i, {})
            if verbose:
                print(
                    f"[infer_shape_type_from_execution] run with "
                    f"{string_type(dict(args=i[0], kwargs=i[1]), with_shape=True)}"
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
