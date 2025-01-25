import contextlib
import copy
import enum
import inspect
import os
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
import numpy as np
import torch
from ..helpers import string_type, max_diff


def serialize_one(
    obj: Any, name: Union[str, int], schema: str
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Serializes one object into a tensor or a list of tensors.
    *name* and *schema* are just better error messages.
    """
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, (tuple, list)):
        assert all(
            isinstance(t, torch.Tensor) for t in obj
        ), f"Unexpected type in {string_type(obj)}. It should be all tensors."
        return obj
    if obj.__class__.__name__ in {"DynamicCache", "patched_DynamicCache"}:
        return [*obj.key_cache, *obj.value_cache]
    if obj is None:
        return None
    raise NotImplementedError(
        f"Unable to serialize type {type(obj)}, "
        f"class_name={obj.__class__.__name__!r}, "
        f"types={string_type(obj, with_shape=True)}, "
        f"name={name!r} from schema={schema!r}"
    )


def serialize_args(
    args: Tuple[Any], kwargs: Optional[Dict[str, Any]], schema: str
) -> Tuple[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
    """Serializes args and kwargs before calling a custom ops."""
    if isinstance(args, torch.Tensor):
        new_args = args
    else:
        new_args = []
        for i, a in enumerate(args):
            r = serialize_one(a, name=i, schema=schema)
            if r is None or isinstance(r, torch.Tensor):
                new_args.append(r)
            else:
                new_args.extend(r)
        new_args = tuple(new_args)
    assert not kwargs, (
        f"Not implemented with args={string_type(args, with_shape=True)}, "
        f"kwargs={string_type(kwargs, with_shape=True)}"
    )
    if kwargs is None:
        return new_args
    return new_args, {}


def type_as_str_with_info(obj: Any) -> str:
    """Returns a string with information about how to deserialize."""
    if isinstance(obj, torch.Tensor):
        return "Tensor"
    if obj.__class__.__name__ in {"DynamicCache", "patched_DynamicCache"}:
        return f"{obj.__class__.__name__}__{len(obj.key_cache)}_{len(obj.value_cache)}"
    if obj is None:
        return "None"
    raise NotImplementedError(
        f"Unable to produce serialize info for type {type(obj)}, "
        f"class_name={obj.__class__.__name__!r}."
    )


def deserialize_args(
    res: List[torch.Tensor], expected_types: List[str], clone: bool = False
) -> Tuple[Any, ...]:
    """
    Deserizalizes output results coming from the custom op and restores
    the python classes attached to it.

    :param res: args to deserialize
    :param expected_types: information on how to deserialize
    :param clone: clone tensors before returning them
    :return: new args
    """
    assert isinstance(res, (list, tuple, torch.Tensor)), f"unexpected type for res {type(res)}"
    if isinstance(res, torch.Tensor):
        assert expected_types == [
            "Tensor"
        ], f"Mismatch information, expected_types={expected_types!r}"
        return res
    assert all(
        isinstance(t, (list, torch.Tensor)) for t in res
    ), f"unexpected element type in res: {string_type(res)}"
    des = []
    pos_res = 0
    for tt in expected_types:
        if tt == "Tensor":
            des.append(res[pos_res].clone() if clone else res[pos_res])
            pos_res += 1
            continue
        if tt.startswith(("DynamicCache__", "patched_DynamicCache__")):
            info = tt.split("__")[-1]
            n1, n2 = tuple(map(int, info.split("_")))
            assert n1 == n2, f"Unexpected sizes for n1={n1} and n2={n2} for a DynamicCache"
            if isinstance(res[pos_res], torch.Tensor):
                # All flattened.
                key_cache = res[pos_res : pos_res + n1]
                value_cache = res[pos_res + n1 : pos_res + n1 + n2]
                pos_res += n1 + n2
            else:
                value = res[pos_res]
                assert isinstance(value, list) and all(
                    isinstance(t, torch.Tensor) for t in value
                ), (
                    f"Unexpected type at position {pos_res}: "
                    f"{string_type(value, with_shape=True)}, "
                    f"deserialized into {tt}"
                )
                assert len(value) % 2 == 0 and len(value) == n1 + n2, (
                    f"Number of tensors at position {pos_res} "
                    f"in {string_type(value, with_shape=True)} "
                    f"should be even. Unable to deserialize into {tt}, "
                    f"n1={n1}, n2={n2}, len(res[pos_res])={len(value)}"
                )
                key_cache = value[:n1]
                value_cache = value[n1:]
                pos_res += 1

            if tt.startswith("DynamicCache__"):
                import transformers

                cache = transformers.cache_utils.DynamicCache()
            elif tt.startswith("patched_DynamicCache__"):
                from .patches.patch_transformers import patched_DynamicCache

                cache = patched_DynamicCache()
            else:
                raise NotImplementedError(f"Unable to handle type info {tt!r}")
            if clone:
                cache.key_cache = [t.clone() for t in key_cache]
                cache.value_cache = [t.clone() for t in value_cache]
            else:
                cache.key_cache = key_cache
                cache.value_cache = value_cache
            des.append(cache)
            continue

        raise NotImplementedError(f"Unable to handle type info {tt!r}")
    assert pos_res == len(res), (
        f"Deserialization went wrong, pos_res={pos_res}, len(res)={len(res)}, "
        f"expected_types={expected_types}, "
        f"input types={string_type(res)}"
    )
    return des


def deserialize_args_kwargs(
    args: List[torch.Tensor],
    kwargs: Dict[str, Any],
    expected_types: Tuple[List[str], List[str]],
    clone: bool = False,
) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    """
    Deserializes a list of tensor or list of tensors into args and kwargs.
    *kwargs* should be empty since this type is allowed as a serialized type.

    :param args: arguments
    :param kwargs: named arguments, they should be empty
    :param expected_types: needed to understand how to deserialize
    :param clone: clone every tensor
    :return: new args, new named args
    """
    assert not kwargs, (
        f"inputs coming from C++ functions should not have "
        f"named arguments but kwargs={string_type(kwargs, with_shape=True)}."
    )
    assert (
        isinstance(expected_types, tuple)
        and len(expected_types) == 2
        and not expected_types[1]
    ), (
        f"Unexpected value for expected_types={expected_types}, "
        f"args={string_type(args, with_shape=True)}, "
        f"kwargs={string_type(kwargs, with_shape=True)}, "
    )
    new_args = deserialize_args(args, expected_types[0], clone=clone)
    return new_args, {}


class CustomOpStrategy(enum.IntEnum):
    """
    Defines when to switch to CustomOp to see if the module successfully
    exports with none of its children.
    """

    NONE = 0
    ONLY_IF_FAILING = 1
    ALWAYS = 2


def make_copy(obj: Any) -> Any:
    """Makes a copy of the objects."""
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
    if obj.__class__.__name__ in ("DynamicCache", "patched_DynamicCache"):
        cache = obj.__class__()
        if hasattr(obj, "_seen_tokens"):
            cache._seen_tokens = obj._seen_tokens
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
    by :func:`trace_execution_piece_by_piece`.
    Example :ref:`l-plot-exporter-recipes-custom-phi35` tells you
    more about how to use this class.
    """

    def __init__(
        self,
        parent: Optional["ModelDiagnoseOutput"],
        name: str,
        model: torch.nn.Module,
        level: int = 0,
    ):
        self.parent = parent
        self.name = name
        self.model = model
        self.level = level
        self.forward = model.forward
        self.inputs = []
        self.outputs = []
        self.children: List[ModelDiagnoseOutput] = []
        sig = inspect.signature(self.forward)
        self.forward_parameter_names = set(
            p.name
            for p in sig.parameters.values()
            if p.kind not in {p.VAR_POSITIONAL, p.VAR_KEYWORD}
        )
        self.forward_ordered_parameter_names = list(sig.parameters)
        names = [p.name for p in sig.parameters.values() if p.kind == p.VAR_POSITIONAL]
        self.forward_args = names[0] if names else None
        names = [p.name for p in sig.parameters.values() if p.kind == p.VAR_KEYWORD]
        self.forward_kwargs = names[0] if names else None
        self.forward_custom_op_schema = None
        self.forward_need_serialization = False
        assert not isinstance(model, torch.nn.ModuleList), "ModuleList should not be traced."
        self._debug_noquiet_name = os.environ.get("DIAGNAME", "")
        self.device = "cpu"

    def __iter__(self) -> Iterator:
        """Iterates on all the nodes in the graph."""
        yield self
        yield from self.children

    def get_debug_msg(self) -> str:
        """Returns information about this instances to help debugging."""
        rows = [
            "",
            f"name={self.name!r}",
            f"cls={self.model.__class__.__name__}",
            f"level={self.level}",
            f"forward_ordered_parameter_names={self.forward_ordered_parameter_names}",
            f"forward_args={self.forward_args}",
            f"forward_kwargs={self.forward_kwargs}",
            f"device={self.device}",
            f"n_children={len(self.children)}",
        ]
        for i, inp in enumerate(self.inputs):
            rows.append(f"inputs[{i}]={string_type(inp, with_shape=True)}")
        for i, inp in enumerate(self.outputs):
            rows.append(f"outputs[{i}]={string_type(inp, with_shape=True)}")
        for att in ["exporter_status", "forward_custom_op_schema"]:
            if not hasattr(self, att):
                continue
            rows.append(f"{att}={getattr(self, att)}")
        return "\n".join(rows)

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
            rows.extend(t.split("\n"))
        if with_inputs:
            for i in self.outputs:
                rows.append(f"{indent}  < {string_type(i, **kws)}")
        rows.append(f"{indent}<<<")
        return "\n".join(rows)

    @property
    def full_name(self):
        "Returns a name and class name."
        return f"{self.name}:{self.model.__class__.__name__}"

    @property
    def custom_op_name(self):
        "Returns a name and class name."
        if self.parent is None:
            return f"C_{self.model.__class__.__name__}"
        return f"{self.parent.custom_op_name}_{self.name}"

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
        for k in kwargs:
            assert self.forward_kwargs or k in self.forward_parameter_names, (
                f"Unexpected parameter {k!r} (not found in {self.forward_parameter_names}), "
                f"name={self.name!r}, model={self.model.__class__.__name__}, "
                f"module={self.model.__class__.__module__}, model={self.model}"
            )
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

        if obj.__class__.__name__ in ("DynamicCache", "patched_DynamicCache"):
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

    def _move_to_kwargs(self, args, kwargs, dynamic_shapes):
        """
        Uses the signatures to move unnamed arguments (args) to named arguments (kwargs)
        with the corresponding dynamic shapes.
        *kwargs*, *dynamic_shapes* are modified inplace.
        """
        sig = inspect.signature(self.forward)
        arg_dyn, kw_dyn = dynamic_shapes
        for i, p in enumerate(sig.parameters):
            if i >= len(arg_dyn):
                break
            kwargs[p] = args[i]
            kw_dyn[p] = arg_dyn[i]
        if self.forward_kwargs:
            kdw = {}
            for k, v in kw_dyn.items():
                if k not in self.forward_parameter_names:
                    kdw[k] = v
            if kdw:
                for k in kdw:
                    del kw_dyn[k]
                kw_dyn[self.forward_kwargs] = kdw
            # Let's reorder as it seems to matter later
            # in the shape inference algorithm.
            _kwargs = kwargs
            kwargs = {}
            _kw_dyn = kw_dyn
            kw_dyn = {}
            for name in self.forward_ordered_parameter_names:
                if name in _kwargs:
                    kwargs[name] = _kwargs[name]
                if name in _kw_dyn:
                    kw_dyn[name] = _kw_dyn[name]
            for k in _kwargs:
                if k not in kwargs:
                    # Then it is part of **kwargs.
                    kwargs[k] = _kwargs[k]
            assert len(kw_dyn) == len(_kw_dyn), (
                f"{self.full_name}: unexpected mismatch between _kw_dyn={set(_kw_dyn)} "
                f"and kw_dyn={set(kw_dyn)}, "
                f"forward_ordered_parameter_names={self.forward_ordered_parameter_names}"
            )
            assert len(kwargs) == len(_kwargs), (
                f"{self.full_name}: unexpected mismatch between _kwargs={set(_kwargs)} "
                f"and kwargs={set(kwargs)}, "
                f"forward_ordered_parameter_names={self.forward_ordered_parameter_names}"
            )
        return tuple(), kwargs, (tuple(), kw_dyn)

    def _try_export_no_bypass_export(
        self,
        export_inputs,
        exporter_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        quiet: bool = True,
        use_dynamic_shapes: Optional[bool] = None,
    ):
        if quiet:
            quiet = self._debug_noquiet_name != self.name
            debug = not quiet
        else:
            debug = False

        args, kwargs = export_inputs
        dynamic_shapes = self.guess_dynamic_shapes() if use_dynamic_shapes else None
        if dynamic_shapes and (self.forward_kwargs or self.forward_args):
            # The export should change dynamic shapes to have only named arguments.
            if debug or verbose > 1:
                sds = str(dynamic_shapes).replace("<_DimHint.DYNAMIC: 3>", "DYN")
                print(f"[try_export-FX] {self.dot_name}: change dynamic_shapes={sds}")
                if debug or verbose > 2:
                    print(
                        f"[try_export-FX] {self.dot_name}: "
                        f"args={string_type(args, with_shape=True)}"
                    )
                    print(
                        f"[try_export-FX] {self.dot_name}: "
                        f"kwargs={string_type(kwargs, with_shape=True)}"
                    )
            args, kwargs, dynamic_shapes = self._move_to_kwargs(args, kwargs, dynamic_shapes)
        ds = dynamic_shapes[0] or dynamic_shapes[1]
        if debug or verbose > 1:
            sds = str(dynamic_shapes).replace("<_DimHint.DYNAMIC: 3>", "DYN")
            print(f"[try_export-FX] {self.dot_name}: dynamic_shapes={sds}")
            if debug or verbose > 2:
                print(
                    f"[try_export-FX] {self.dot_name}: ds="
                    f"{str(ds).replace('<_DimHint.DYNAMIC: 3>', 'DYN')}"
                )
                print(
                    f"[try_export-FX] {self.dot_name}: "
                    f"args={string_type(args, with_shape=True)}"
                )
                print(
                    f"[try_export-FX] {self.dot_name}: "
                    f"kwargs={string_type(kwargs, with_shape=True)}"
                )
            if debug and len(self.inputs) > 1:
                print(
                    f"[try_export-FX-DEBUG] {self.dot_name}: inputs[0]="
                    f"{string_type(self.inputs[0], with_shape=True)}"
                )
                print(
                    f"[try_export-FX-DEBUG] {self.dot_name}: inputs[1]="
                    f"{string_type(self.inputs[1], with_shape=True)}"
                )
                print(
                    f"[try_export-FX-DEBUG] {self.dot_name}: outputs[0]="
                    f"{string_type(self.outputs[0], with_shape=True)}"
                )
                print(
                    f"[try_export-FX-DEBUG] {self.dot_name}: outputs[1]="
                    f"{string_type(self.outputs[1], with_shape=True)}"
                )

        if quiet:
            try:
                ep = torch.export.export(
                    self.model,
                    args,
                    kwargs=kwargs,
                    dynamic_shapes=ds,
                    **(exporter_kwargs or {}),
                )
                self.exporter_status = "OK"
            except Exception as e:
                if "'NoneType' object is not iterable" in str(e):
                    raise
                self.last_error = e
                se = str(e).split("\n")[0].replace("<_DimHint.DYNAMIC: 3>", "DYN")
                self.exporter_status = f"FAIL-EXPORT: {se}"
                if verbose:
                    print(f"[try_export-FX] {self.dot_name} --- {self.exporter_status}")
                return None, None
        else:
            ep = torch.export.export(
                self.model,
                args,
                kwargs=kwargs,
                dynamic_shapes=ds,
                **(exporter_kwargs or {}),
            )
            self.exporter_status = "OK"
        if verbose > 1:
            print(f"[try_export-FX] {self.dot_name}: done")
        mod = ep.module()

        def _call_model_(*args, _mod=mod, **kwargs):
            if verbose > 2:
                print(
                    f"[try-export-FX] call module {mod!r} with "
                    f"args={string_type(args, with_shape=True)} and "
                    f"kwargs={string_type(kwargs, with_shape=True)}"
                )
            res = mod(*args, **kwargs)
            if verbose > 2:
                print(
                    f"[try-export-FX] after called {mod!r} "
                    f"res={string_type(args, with_shape=True)}"
                )
            return res

        return ep, _call_model_

    def _do_replace_by_custom_op(
        self, replace_by_custom_op: Union[bool, CustomOpStrategy, Dict[str, CustomOpStrategy]]
    ) -> CustomOpStrategy:
        """
        Tells if a module must be replaced by a custom op.
        """
        if isinstance(replace_by_custom_op, bool):
            return (
                CustomOpStrategy.ONLY_IF_FAILING
                if replace_by_custom_op
                else CustomOpStrategy.NONE
            )
        if isinstance(replace_by_custom_op, CustomOpStrategy):
            return replace_by_custom_op
        if isinstance(replace_by_custom_op, set):
            if self.model.__class__.__name__ in replace_by_custom_op:
                return replace_by_custom_op[self.model.__class__.__name__]
            if self.name.__class__.__name__ in replace_by_custom_op:
                return replace_by_custom_op[self.name.__class__.__name__]
        return False

    def _annotation_from_type(self, obj) -> str:
        if isinstance(obj, torch.Tensor):
            return "Tensor"
        if obj.__class__.__name__ in ("DynamicCache", "patched_DynamicClass"):
            # It is safer to serialize everything, it is aligned with ONNX,
            # and the use of list brought the following error:
            # ::
            #   RuntimeError: C_Model (with implementation in
            #   <module 'torch._library.custom_ops' from
            #   'torch/_library/custom_ops.py'>):
            #   The output of this custom operator (1) must not also be an input to this
            #   custom operator and (2) may not alias any inputs to this custom operator
            #   or other returns. The most common way to trigger this error is if we have
            #   y = custom_op(x) and y and x are the same Tensor. Please instead return a
            #   clone of the offending output tensor(s) (e.g. return x.clone()) or
            #   refactor the custom operator to not return y.
            return ["Tensor" for i in range(len(obj.key_cache) + len(obj.value_cache))]
        if obj is None:
            # Let's assume it is a tensor. It should not matter anyway.
            # Unless it becomes None in another call.
            return "Tensor?"
        raise NotImplementedError(
            f"Annotation for type {string_type(obj)} is not implemented{self.get_debug_msg()}"
        )

    def _annotated_input(self, name):
        args, kwargs = self.inputs[0]
        if name in kwargs:
            o = kwargs[name]
            annotated = self._annotation_from_type(o)
        else:
            index = self.forward_ordered_parameter_names.index(name)
            o = args[index]
            annotated = self._annotation_from_type(o)
        if isinstance(annotated, str):
            return f"{annotated} {name}"
        assert isinstance(
            annotated, list
        ), f"unexpected type {type(annotated)} for name={name!r}"
        return ", ".join(
            [f"{t} {name}_n{len(annotated)}_{i}" for i, t in enumerate(annotated)]
        )

    def _annotated_output(self):
        outputs = []
        for o in self.outputs[0]:
            annotated = self._annotation_from_type(o)
            if isinstance(annotated, str):
                outputs.append(annotated)
                continue
            assert isinstance(
                annotated, list
            ), f"unexpected type {type(annotated)} for name={o!r}"
            outputs.extend(annotated)
        unique = set(outputs)
        assert unique == {
            "Tensor"
        }, f"{self.full_name}: no other tyoe than Tensor is supported, types={unique}"
        return "Tensor" if len(outputs) == 1 else "Tensor[]"

    def build_c_schema(self) -> str:
        """Returns a schema for the C function."""
        # schema_str = return f"({', '.join(params)}) -> {ret}"
        args = []
        for p in self.forward_ordered_parameter_names:
            if p == self.forward_args:
                args.append(f"*{p}")
            elif p == self.forward_kwargs:
                args.append(f"**{p}")
            else:
                args.append(self._annotated_input(p))
        outputs = self._annotated_output()
        schema_str = f"({', '.join(args)}) -> {outputs}"
        return schema_str

    def _register(
        self, fct: Callable, fct_shape: Callable, namespace: str, fname: str, verbose: int = 0
    ):
        schema_str = self.build_c_schema()
        if verbose > 1:
            print(f"[try_export._register] {self.dot_name} schema_str={schema_str!r}")

        # registration
        custom_def = torch.library.CustomOpDef(namespace, fname, schema_str, fct)
        custom_def.register_kernel(self.device)(fct)
        custom_def._abstract_fn = fct_shape
        return custom_def, schema_str

    def build_shape_mapping_indices(
        self,
    ) -> List[Tuple[Union[int, Tuple[int, ...]], torch.dtype]]:
        """
        Builds a mapping output and input shapes so that a function
        returns dynamic shapes can automatically inferred.
        """
        # The main idea. Knowning everything is going to be serialized,
        # inputs and outputs are serialized, we try to match the output
        # shapes with the inputs one.
        flattened_inputs = [serialize_args(*i, schema=None) for i in self.inputs]
        shaped_mapped = [{} for i in flattened_inputs]
        for row in range(len(shaped_mapped)):
            inp_args, inp_kwargs = flattened_inputs[row]
            assert not inp_kwargs, f"Not implemented yet with kwargs={string_type(inp_kwargs)}"
            for i, inp in enumerate(inp_args):
                if inp.shape not in shaped_mapped[row]:
                    shaped_mapped[row][inp.shape] = []
                shaped_mapped[row][inp.shape].append(i)

        flattened_outputs = [serialize_args(i, None, schema=None) for i in self.outputs]
        n_outputs = len(flattened_outputs[0])

        indices_map = [None for _ in range(n_outputs)]

        def _msg_(i):
            return (
                f"{self.full_name}: inconsistencies for output {i}, "
                f"\nflattened_inputs={string_type(flattened_inputs, with_shape=True)}, "
                f"\nflattened_outputs={string_type(flattened_outputs, with_shape=True)}, "
                f"\nshaped_mapped={shaped_mapped}, "
                f"\nindices_map={indices_map}"
            )

        for i in range(n_outputs):
            for row in range(len(shaped_mapped)):
                shape = flattened_outputs[row][i].shape
                if shape in shaped_mapped[row]:
                    obtained = min(shaped_mapped[row][shape])
                    if indices_map[i] is None:
                        indices_map[i] = obtained
                    else:
                        assert obtained == indices_map[i], _msg_(i)

        # When a shape is not mapped, it is a constant.
        for i, mapped in enumerate(indices_map):
            if mapped is not None:
                continue
            shapes = set(flattened_outputs[row][i] for row in len(self.outputs))
            assert len(shapes) == 1, _msg_(i)
            indices_map[i] = shapes.pop()

        return tuple((i, f.dtype) for i, f in zip(indices_map, flattened_outputs[0]))

    def _get_symbolic_function_for_forward_shape(self) -> Callable:
        """
        Returns a function computed the output shape assuming it can be inferred
        from inputs and outputs.
        """
        if all(isinstance(t, torch.Tensor) for t in self.outputs[0]):
            inp_args, inp_kwargs = self.inputs[0]
            out = self.outputs[0]
            input_shape = inp_args[0].shape
            unique_output_shape = set(t.shape for t in out)
            if (
                not inp_kwargs
                and len(unique_output_shape) == 1
                and unique_output_shape.pop() == input_shape
            ):
                n_outputs = len(out)

                if n_outputs == 1:

                    def _symbolic_forward_tensor_1_tensor_like_first_input(*args, **kwargs):
                        # TODO: change this
                        return torch.empty_like(args[0])

                    return _symbolic_forward_tensor_1_tensor_like_first_input

                def _symbolic_forward_tensor_n_tensor_like_first_input(
                    *args, _n_outputs=n_outputs, **kwargs
                ):
                    return tuple(torch.empty_like(args[0]) for t in range(_n_outputs))

                return _symbolic_forward_tensor_n_tensor_like_first_input

        indices_map = self.build_shape_mapping_indices()
        if indices_map is not None:

            def _symbolic_forward_tensor_mapped_io_shapes(
                *args, _indices_map=indices_map, **kwargs
            ):
                outputs = []
                for ii, dtype in _indices_map:
                    out = (
                        torch.empty(ii)
                        if isinstance(ii, tuple)
                        else torch.empty_like(args[ii])
                    )
                    outputs.append(out if out.dtype == dtype else out.to(dtype))
                return tuple(outputs)

            return _symbolic_forward_tensor_mapped_io_shapes

        raise NotImplementedError(
            f"{self.full_name}: unable to create function producing the symbolic shapes, "
            f"input_types={string_type(self.inputs[0], with_shape=True)}, "
            f"output_types={string_type(self.outputs[0], with_shape=True)}"
        )

    def _put_custom_op_inplace_tensor(self, verbose: int = 0):
        """
        Replaces the submodule by a custom operator.
        It rewrites the forward method to call a function.
        Only tensors are supported so that there is no serialization
        or deserialization to support.
        """

        def _rewrite_forward_tensor_(*args, _diag=self, **kwargs):
            if verbose > 1:
                print(
                    f"[_rewrite_forward_tensor_] {_diag.full_name}: IN: "
                    f"args={string_type(args, with_shape=True)}, "
                    f"kwargs={string_type(kwargs, with_shape=True)}"
                )
            res = _diag.forward(*args, **kwargs)
            if verbose > 1:
                print(
                    f"[_rewrite_forward_tensor_] {_diag.full_name}: "
                    f"OUT: args={string_type(res, with_shape=True)}"
                )
            return res

        name_fct = self.custom_op_name
        if verbose > 1:
            print(
                f"[try_export.put_custom_op_inplace_tensor] {self.dot_name}: "
                f"registers 'diag_lib.{name_fct}"
            )

        cusdef, schema_str = self._register(
            _rewrite_forward_tensor_,
            self._get_symbolic_function_for_forward_shape(),
            "diag_lib",
            name_fct,
            verbose=verbose,
        )
        assert cusdef is not None, f"{self.full_name}: registration of a custom op has failed."
        if verbose > 1:
            print(
                f"[try_export.put_custom_op_inplace_tensor] {self.dot_name}: "
                f"schema_str={schema_str}"
            )
        # We stored to avoid the registration twice.
        self.forward_custom_op_schema = cusdef
        expected_output_type = [type_as_str_with_info(o) for o in self.outputs[0]]
        self.forward_expected_output_type = expected_output_type
        if verbose > 1:
            print(
                f"[try_export.put_custom_op_inplace_tensor] {self.dot_name}: "
                f"expected_output_type={expected_output_type}"
            )

        # Apparently, we need a function with the exact same signature.
        def _replaced_forward_tensor_(*args, **kwargs):
            fct = getattr(torch.ops.diag_lib, name_fct)
            if verbose > 1:
                print(
                    f"[_replaced_forward_tensor_] {name_fct}-IN: "
                    f"args={string_type(args, with_shape=True)}, "
                    f"kwargs={string_type(kwargs, with_shape=True)}, "
                    f"schema_str={schema_str}"
                )
                sfct = str(fct).replace("\n", " ")
                print(f"[_replaced_forward_tensor_] {name_fct}-CALL: {sfct}")
            res = fct(*args, **kwargs)
            if verbose > 1:
                print(
                    f"[_replaced_forward_tensor_] {name_fct}-OUT: "
                    f"des={string_type(res, with_shape=True)}"
                )
            return res

        _replaced_forward_tensor_.__signature__ = inspect.Signature.from_callable(self.forward)
        self.forward_calling_custom_op = _replaced_forward_tensor_
        self.model.forward = _replaced_forward_tensor_
        return cusdef

    def _put_custom_op_inplace_any(self, verbose: int = 0):
        """
        Replaces the submodule by a custom operator.
        It rewrites the forward method to call a function.
        Any type among the supported list if support (an exception will
        be raised otherwise). C++ does not support custom types so
        serialization and deserialization are needed.
        """

        def _rewrite_forward_(*args, _diag=self, **kwargs):
            if verbose > 1:
                print(
                    f"[_rewrite_forward_] {_diag.full_name}-SERIALIZE_IN: "
                    f"args={string_type(args, with_shape=True)}, "
                    f"kwargs={string_type(kwargs, with_shape=True)}"
                )
            # We need to deserialize back before calling forward.
            new_args, new_kwargs = deserialize_args_kwargs(
                args, kwargs, _diag.forward_expected_input_type, clone=True
            )
            if verbose > 1:
                print(
                    f"[_rewrite_forward_] {_diag.full_name}-IN: "
                    f"args={string_type(new_args, with_shape=True)}, "
                    f"kwargs={string_type(new_kwargs, with_shape=True)}"
                )
                sfct = str(_diag.forward).replace("\n", " ")
                print(f"[_rewrite_forward_] {_diag.full_name}-CALL: {sfct}")
            res = _diag.forward(*new_args, **new_kwargs)
            if verbose > 1:
                print(
                    f"[_rewrite_forward_] {_diag.full_name}-OUT: "
                    f"args={string_type(res, with_shape=True)}"
                )
            # And we need to serialize before before returning the output.
            serialized_res = serialize_args(res, None, _diag.forward_custom_op_schema)
            if verbose > 1:
                print(
                    f"[_rewrite_forward_] {_diag.full_name}-SERIALIZE-OUT: "
                    f"args={string_type(serialized_res, with_shape=True)}"
                )
            return serialized_res

        name_fct = self.custom_op_name
        if verbose > 1:
            print(
                f"[try_export.put_custom_op_inplace] {self.dot_name}: "
                f"registers 'diag_lib.{name_fct}"
            )

        cusdef, schema_str = self._register(
            _rewrite_forward_,
            self._get_symbolic_function_for_forward_shape(),
            "diag_lib",
            name_fct,
            verbose=verbose,
        )
        assert cusdef is not None, f"{self.full_name}: registration of a custom op has failed."
        if verbose > 1:
            print(
                f"[try_export.put_custom_op_inplace] {self.dot_name}: "
                f"schema_str={schema_str}"
            )
        # We stored to avoid the registration twice.
        self.forward_custom_op_schema = cusdef
        expected_output_type = [type_as_str_with_info(o) for o in self.outputs[0]]
        self.forward_expected_output_type = expected_output_type
        self.forward_expected_input_type = (
            [type_as_str_with_info(o) for o in self.inputs[0][0]],
            {k: type_as_str_with_info(v) for k, v in self.inputs[0][1].items()},
        )
        if verbose > 1:
            print(
                f"[try_export.put_custom_op_inplace] {self.dot_name}: "
                f"expected_output_type={expected_output_type}"
            )

        # Apparently, we need a function with the exact same signature.
        def _replaced_forward_(*args, **kwargs):
            fct = getattr(torch.ops.diag_lib, name_fct)
            if verbose > 1:
                print(
                    f"[_replaced_forward_] {name_fct}-IN: "
                    f"args={string_type(args)}, kwargs={string_type(kwargs)}, "
                    f"schema_str={schema_str}"
                )
            args, kwargs = serialize_args(args, kwargs, schema=schema_str)
            if verbose > 1:
                print(
                    f"[_replaced_forward_] {name_fct}-SERIALIZED_IN: "
                    f"args={string_type(args, with_shape=True)}, "
                    f"kwargs={string_type(kwargs, with_shape=True)}"
                )
                print(f"[_replaced_forward_] {name_fct}-CALL: {fct}")
            res = fct(*args, **kwargs)
            if verbose > 1:
                print(
                    f"[_replaced_forward_] {name_fct}-SERIALIZED_OUT: "
                    f"res={string_type(res, with_shape=True)}, "
                    f"expected_output_type={expected_output_type}"
                )
            des = deserialize_args(res, expected_output_type)
            if verbose > 1:
                print(
                    f"[_replaced_forward_] {name_fct}-OUT: "
                    f"des={string_type(des, with_shape=True)}"
                )
            if isinstance(des, torch.Tensor):
                return des
            return tuple(des) if len(des) > 1 else des[0]

        _replaced_forward_.__signature__ = inspect.Signature.from_callable(self.forward)
        self.forward_calling_custom_op = _replaced_forward_
        self.model.forward = _replaced_forward_
        return cusdef

    def put_custom_op_inplace(self, verbose: int = 0):
        """
        Replaces the submodule by a custom operator.
        It rewrites the forward method to call a function
        """
        if self.forward_custom_op_schema is not None:
            # Registration was already done.
            self.model.forward = self.forward_calling_custom_op
            return self.forward_custom_op_schema

        if all(isinstance(t, torch.Tensor) for t in self.inputs[0]) and all(
            isinstance(t, torch.Tensor) for t in self.outputs[0]
        ):
            self.forward_custom_op_serialize = False
            return self._put_custom_op_inplace_only_tensor(verbose=verbose)
        self.forward_custom_op_serialize = True
        self.forward_need_serialization = self.forward_custom_op_serialize
        return self._put_custom_op_inplace_any(verbose=verbose)

    def remove_custom_op_inplace(self, verbose: int = 0):
        """
        Just replaces the forward, hoping the registration does not have to
        be removed.
        """
        self.forward_need_serialization = False
        self.model.forward = self.forward
        if verbose > 1:
            print(
                f"[try_export.remove_custom_op_inplace] {self.dot_name}: unregisters "
                f"'diag_lib.{self.custom_op_name}"
            )

    def _try_export_no_bypass(
        self,
        modificator: Optional[Callable] = None,
        exporter: str = "fx",
        exporter_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        quiet: bool = True,
        discrepancies: bool = True,
        use_dynamic_shapes: Optional[bool] = None,
        replace_by_custom_op: bool = False,
        atol: float = 1e-2,
        rtol: float = 1e-1,
    ) -> Any:
        """
        Tries to export the module of submodule held by this class.
        Stores intermediate results of the export in attributes prefixed by ``forward_``.
        """
        export_inputs = modificator(self.inputs[0]) if modificator else self.inputs[0]
        export_inputs = make_copy(export_inputs)
        if use_dynamic_shapes is None:
            use_dynamic_shapes = len(self.inputs) > 1
        assert (
            not use_dynamic_shapes or len(self.inputs) > 1
        ), "Unable to use dynamic_shapes, only one set of inputs is available."
        cusdef = self.put_custom_op_inplace(verbose=verbose) if replace_by_custom_op else None

        self.status = "START"
        if exporter == "fx":
            exported, fct = self._try_export_no_bypass_export(
                export_inputs,
                exporter_kwargs=exporter_kwargs,
                verbose=verbose,
                quiet=quiet,
                use_dynamic_shapes=use_dynamic_shapes,
            )
            setattr(self, exporter, exported)
        else:
            raise NotImplementedError(f"Export not implemented yet for exporter={exporter!r}")

        if not exported:
            if cusdef is not None:
                self.remove_custom_op_inplace(verbose=verbose)
            return None
        if discrepancies:
            has_disc = False
            self.exporter_outputs = []
            self.exporter_discs = []
            for i, (inp, out) in enumerate(zip(self.inputs, self.outputs)):
                copy_inp = make_copy(modificator(inp) if modificator else inp)
                args, kwargs = copy_inp
                if verbose > 1:
                    print(f"[try_export-{exporter.upper()}] {self.dot_name}: CALL {fct}")
                    if verbose > 2:
                        print(
                            f"[try_export-{exporter.upper()}] {self.dot_name}: "
                            f"args={string_type(args, with_shape=True)}"
                        )
                        print(
                            f"[try_export-{exporter.upper()}] {self.dot_name}: "
                            f"kwargs={string_type(kwargs, with_shape=True)}"
                        )
                    if verbose >= 10:
                        print(f"[try_export-{exporter.upper()}] {self.dot_name}: GRAPH")
                        print(exported.graph)
                if quiet:
                    try:
                        got = fct(*args, **kwargs)
                    except Exception as e:
                        self.last_error = e
                        se = str(e).split("\n")[0]
                        self.exporter_status = f"FAIL-EVAL: {se}"
                        break
                else:
                    got = fct(*args, **kwargs)
                self.exporter_outputs.append(got)
                diff = max_diff(out, got)
                if verbose > 1:
                    print(f"[try_export-{exporter.upper()}] {self.dot_name}: diff[{i}]={diff}")
                self.exporter_discs.append(diff)
                if diff["abs"] > atol or diff["rel"] > rtol:
                    self.exporter_status = "DISC: abs"
                    has_disc = True
                    if not quiet:
                        raise AssertionError(
                            f"{self.full_name}: discrepancies were observed, "
                            f"diff={diff}, expected="
                            f"{string_type(out, with_shape=True, with_min_max=True)}, "
                            f"got={string_type(got, with_shape=True, with_min_max=True)}."
                        )
                    break
            if not has_disc:
                self.exporter_status = "OK"

        if cusdef is not None:
            self.remove_custom_op_inplace(verbose=verbose)

        if verbose:
            print(
                f"[try_export-{exporter.upper()}] {self.dot_name} --- {self.exporter_status}"
            )
        return exported if self.exporter_status == "OK" else None

    def _try_export(
        self,
        exporter: str = "fx",
        exporter_kwargs: Optional[Dict[str, Any]] = None,
        bypass_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        quiet: bool = True,
        discrepancies: bool = True,
        use_dynamic_shapes: Optional[bool] = None,
        replace_by_custom_op: bool = False,
        atol: float = 1e-2,
        rtol: float = 1e-1,
    ) -> Any:
        """
        Tries to export the module of submodule held by this class.
        Stores intermediate results of the export in attributes prefixed by ``forward_``.
        """
        if bypass_kwargs:
            from .onnx_export_errors import bypass_export_some_errors

            with bypass_export_some_errors(
                verbose=max(verbose - 1, 0), **bypass_kwargs
            ) as modificator:
                return self._try_export_no_bypass(
                    modificator,
                    exporter,
                    exporter_kwargs=exporter_kwargs,
                    quiet=quiet,
                    verbose=verbose,
                    use_dynamic_shapes=use_dynamic_shapes,
                    discrepancies=discrepancies,
                    replace_by_custom_op=replace_by_custom_op,
                    atol=atol,
                    rtol=rtol,
                )
        return self._try_export_no_bypass(
            None,
            exporter,
            exporter_kwargs=exporter_kwargs,
            quiet=quiet,
            verbose=verbose,
            use_dynamic_shapes=use_dynamic_shapes,
            discrepancies=discrepancies,
            replace_by_custom_op=replace_by_custom_op,
            atol=atol,
            rtol=rtol,
        )

    def try_export(
        self,
        exporter: str = "fx",
        exporter_kwargs: Optional[Dict[str, Any]] = None,
        bypass_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        quiet: bool = True,
        discrepancies: bool = True,
        use_dynamic_shapes: Optional[bool] = None,
        replace_by_custom_op: Union[
            bool, CustomOpStrategy, Dict[str, CustomOpStrategy]
        ] = CustomOpStrategy.NONE,
        atol: float = 1e-2,
        rtol: float = 1e-1,
    ) -> Any:
        """
        Tries to export a model. If not possible,
        tries every child until it is possible.
        The function stores the export and other results in the class itself,
        in attributes prefixed by ``forward_``.

        :param exporter: export way, 'fx' for :func:`torch.export.export`,
            `'onnx_dynamo`' to call :func:`torch.onnx.export` ``(..., dynamo=True)``,
            `'torch_script'` to call :func:`torch.onnx.export` ``(..., dynamo=False)``,
            `'to_onnx'` to call :func:`experimental_experiment.torch_interpreter.to_onnx`.
        :param exporter_kwargs: argument for the export function
        :param bypass_kwargs: argument for function :func:`bypass_export_some_errors
            <experimental_experiment.torch_interpreter.onnx_export_errors.bypass_export_some_errors>`
        :param verbose: verbosity, to see what the function is doing
        :param discrepancies: run the exported model to measure the discrepancies
        :param quiet: do not catch the first exception
        :param use_dynamic_shapes: use dynamic shapes
        :param replace_by_custom_op: before exporting,
            it replaces submodules by custom ops,
            it can be a boolean to replace all or a selected classes (name or type), or names
        :param atol: absolute tolerance
        :param rtol: relative tolerance
        :return: result of the export function

        See :ref:`l-plot-exporter-recipes-custom-phi35` for an example.
        Environment variable ``DIAGNAME=<name>`` can be set to increase the verbosity
        on a particular op and avoid catching the exception if any.
        """
        allowed = {"fx", "onnx_dynamo", "torch_script", "to_onnx"}
        assert exporter in allowed, (
            f"{self.full_name}: unexpected value for exporter={exporter!r} "
            f"not in {allowed}"
        )
        custom_op_strat = self._do_replace_by_custom_op(replace_by_custom_op)

        exported = self._try_export(
            exporter=exporter,
            exporter_kwargs=exporter_kwargs,
            bypass_kwargs=bypass_kwargs,
            verbose=verbose,
            quiet=quiet,
            discrepancies=discrepancies,
            use_dynamic_shapes=use_dynamic_shapes,
            replace_by_custom_op=custom_op_strat == CustomOpStrategy.ALWAYS,
            atol=atol,
            rtol=rtol,
        )

        if exported is None and replace_by_custom_op == CustomOpStrategy.ONLY_IF_FAILING:
            # The conversion of the model with its submodule failed.
            # We try to export the module without its children.
            if verbose:
                print(
                    f"[try_export-{exporter.upper()}] {self.dot_name} "
                    f"children replace by custom ops"
                )

            for child in self.children:
                child.put_custom_op_inplace(verbose=verbose)

            # We export again.
            exported = self._try_export(
                exporter=exporter,
                exporter_kwargs=exporter_kwargs,
                bypass_kwargs=bypass_kwargs,
                verbose=verbose,
                quiet=quiet,
                discrepancies=discrepancies,
                use_dynamic_shapes=use_dynamic_shapes,
                replace_by_custom_op=False,
                atol=atol,
                rtol=rtol,
            )

            # We restore the initial state.
            for child in self.children:
                child.remove_custom_op_inplace(verbose=verbose)

            if exported is not None:
                self.exporter_status = "OK with children as custom ops"

            if verbose:
                print(
                    f"[try_export-{exporter.upper()}] {self.dot_name} "
                    f"--- {self.exporter_status}"
                )

        if exported is not None and custom_op_strat == CustomOpStrategy.NONE:
            # We don't want to return if custom ops were applied,
            # we need to look into every of them.
            return exported

        # If a custom op was used to bypass the export or
        # if the export failed, we look into the children.
        for child in self.children:
            child.try_export(
                exporter=exporter,
                exporter_kwargs=exporter_kwargs,
                bypass_kwargs=bypass_kwargs,
                verbose=verbose,
                quiet=quiet,
                discrepancies=discrepancies,
                use_dynamic_shapes=use_dynamic_shapes,
                replace_by_custom_op=replace_by_custom_op,
                atol=atol,
                rtol=rtol,
            )
        return exported

    def get_export_status(self) -> str:
        """
        Returns a report status on the conversion.
        """

        def iter_status(here):
            rows = [here]
            for child in here.children:
                rows.extend(iter_status(child))
            return rows

        rows = iter_status(self)
        to_display = [
            (
                f"{'..' * r.level}{r.name}",
                r.model.__class__.__name__,
                getattr(r, "exporter_status", "OK as part of its owner"),
            )
            for r in rows
        ]
        mc1 = max(len(c[0]) for c in to_display) + 3
        mc2 = max(len(c[1]) for c in to_display) + 3
        srows = []
        for t in to_display:
            c = t[0]
            s = " " * (mc1 - len(t[0]))
            c2 = t[1]
            s2 = " " * (mc2 - len(t[1]))
            srows.append(f"{c}{s}{c2}{s2}{t[2]}")
        return "\n".join(srows)


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
    parent: ModelDiagnoseOutput,
    model: torch.nn.Module,
    name: str = "__main__",
    level: int = 0,
    verbose: int = 0,
):
    diag = ModelDiagnoseOutput(parent, name, model, level=level)
    if verbose:
        print(f"[_trace_forward_execution] {diag.dot_name}")
    model.forward = lambda *args, _diag=diag, verbose=verbose, **kwargs: _rewrite_forward(
        *args, _diag=_diag, verbose=verbose, **kwargs
    )
    for name, mod in model.named_children():
        if isinstance(mod, torch.nn.ModuleList):
            for i, m in enumerate(mod):
                d = _trace_forward_execution(
                    diag, m, f"{name}[{i}]", verbose=max(verbose - 1, 0), level=level + 1
                )
                diag.add_child(d)
        else:
            d = _trace_forward_execution(
                diag, mod, name, verbose=max(verbose - 1, 0), level=level + 1
            )
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
    See :ref:`l-plot-exporter-recipes-custom-phi35` for an example.
    """
    diag = _trace_forward_execution(None, model, verbose=verbose)
    try:
        yield diag
    finally:
        _untrace_forward_execution(diag, verbose=verbose)


def trace_execution_piece_by_piece(
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

    See :ref:`l-plot-exporter-recipes-custom-phi35` for an example.
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
                    f"[trace_execution_piece_by_piece] run with "
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