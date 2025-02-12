import ast
import contextlib
import enum
import inspect
import os
import pickle
import pprint
import sys
import textwrap
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union
import numpy as np
import onnx
import torch
from ..helpers import string_type, string_sig, max_diff, string_diff
from ..xbuilder import OptimizationOptions
from . import to_onnx, FunctionOptions, Dispatcher, ExportOptions
from .piece_by_piece_serialize import (
    choose_kwargs_for_dynamic_shapes,
    deserialize_args,
    deserialize_args_kwargs,
    make_copy,
    serialize_args,
    type_as_str_with_info,
)


class CustomOpStrategy(enum.IntEnum):
    """
    Defines when to switch to CustomOp to see if the module successfully
    exports with none of its children.

    * ``NONE``: tries to export the module
    * ``ONLY_IF_FAILING``: look into submodule only if it fails
    * ``ALWAYS``: always export as a custom op
    * ``LOCAL``: export all submodules as a custom op and tries the
        conversion of the module itself after it was done
    """

    NONE = 0
    ONLY_IF_FAILING = 1
    ALWAYS = 2
    LOCAL = 3


class StatusExportCode(enum.IntEnum):
    """
    Defines the the exporter status.

    * `FAIL`: exporter has failed
    * `OK`: export succeeds with all the submodule included
    * `CHILDC`: export succeeds with submodule replaced by custom ops
    * `CUSTOM`: export succeds with this module replaced by a custom ops
    * `DISC`: fails due to discrepancy

    This options can be combined.
    """

    NONE = 0
    OK = 1
    FAIL = 2
    CHILDC = 4
    CUSTOM = 8
    DISC = 16

    # combinations
    OK_CHILDC = 5
    OK_CUSTOM = 9
    FAIL_CHILDC = 6
    FAIL_CUSTOM = 10
    FAIL_DISC = 18

    def __not__(self) -> int:
        "Compose.."
        return StatusExportCode(~self.value)

    def __or__(self, a: "StatusExportCode") -> "StatusExportCode":
        "Compose.."
        return StatusExportCode(int(self) | int(a))

    def __ior__(self, a: "StatusExportCode"):
        "Compose.."
        raise NotImplementedError("use |")

    def __and__(self, a: "StatusExportCode") -> "StatusExportCode":
        "Compose.."
        return StatusExportCode(int(self) | int(a))

    def __iand__(self, a: "StatusExportCode"):
        "Compose.."
        raise NotImplementedError("use &")

    def remove(self, a: "StatusExportCode") -> "StatusExportCode":
        "Compose.."
        return StatusExportCode(int(self) - int(a))


class StatusExport:
    """
    Defines the the exporter status.

    :param status: status exporter
    :param step: step it fails
    :param reason: details about the failure
    :param exported: whatever is exporter
    """

    def __init__(
        self,
        status: StatusExportCode,
        step: str = "",
        reason: str = "",
        exported: Optional[Any] = None,
    ):
        assert isinstance(status, StatusExportCode)
        self.status = status
        self.step = step
        self.reason = reason
        self.exported = exported

    @property
    def short_reason(self) -> str:
        "Shortened reason"
        r = self.reason.split("\n", maxsplit=1)[0]
        if len(r) < 30:
            return r
        return f"{r[:30]}..."

    def is_ok(self) -> bool:
        "Returns True if the export succeeds."
        return bool(int(self.status) & int(StatusExportCode.OK))

    def __repr__(self) -> str:
        "usual"
        return string_sig(self)

    def pretty_text(self) -> str:
        "pretty text"
        if self.is_ok():
            return f"{self.status.name} -- {self.exported.__class__.__name__}"
        reason = self.reason.split("\n", maxsplit=1)[0]
        if len(reason) > 70:
            reason = reason[:70] + "..."
        return f"{self.status.name} -- step={self.step}, reason={reason!r}"


class ModelDiagnoseOutput:
    """
    Contains inputs and outputs, traced results when tracing
    intermediate results. An instance of this class is produced
    by :func:`trace_execution_piece_by_piece`.
    Example :ref:`l-plot-exporter-recipes-custom-phi35` tells you
    more about how to use this class.

    * ``parent``: parent owning this instance
    * ``name``: module name
    * ``model``: module
    * ``level``: depth level
    * ``device``: device
    * ``inputs``: stored inputs like the following ``(args, kwargs)``
    * ``outputs``: stored outputs
    * ``signature``: signature of the module or function

    The default method spied on is ``forward`` but it can be changed.
    After the tracing:

    * ``inputs``: traced inputs
    * ``outputs``: traced outputs

    Attribute added to store the export results:

    * ``forward``: forward method of the module
    * ``forward_parameter_names``
    * ``forward_ordered_parameter_names``
    * ``forward_args``
    * ``forward_kwargs``
    * ``forward_custom_op_schema``
    * ``forward_need_serialization``

    Results from the last status:

    * ``exporter``: exporter name
    * ``last_error``: last error
    * ``exporter_status``: last exporter status
    * ``setattr(self, exporter, exported)``: whatever is exported

    Debugging options::

        self._debug_noquiet_name = os.environ.get("DIAGNAME", "")
        self._debug_print_status = os.environ.get("DIAGPRINTSTATUS", "")
        self._debug_print_export = os.environ.get("DIAGPRINTEXPORT", "")

    The class can be improved:

    * It cannot infer how to produce in all cases outputs with expected dynamic
      dimensions based on inputs ones
    * Custom ops are not working well yet with forward method using ``**kwargs``
      ``*args`` in their signature. It is better to keep them empty.
    """

    def __init__(
        self,
        parent: Optional["ModelDiagnoseOutput"],
        name: str,
        model: torch.nn.Module,
        level: int = 0,
        method_name: str = "forward",
    ):
        assert isinstance(model, torch.nn.Module) or inspect.ismodule(
            model
        ), f"unexpected type for model={type(model)}"
        self.parent = parent
        self.name = name
        self.model = model
        self.level = level
        self.method_name = method_name
        self.forward = getattr(model, method_name)
        self.inputs = []
        self.outputs = []
        self.children: List[ModelDiagnoseOutput] = []
        self.signature = inspect.signature(self.forward)
        self.forward_parameter_names = set(
            p.name
            for p in self.signature.parameters.values()
            if p.kind not in {p.VAR_POSITIONAL, p.VAR_KEYWORD}
        )
        self.forward_ordered_parameter_names = list(self.signature.parameters)
        self.forward_positioned_parameter_names = [
            p.name
            for p in self.signature.parameters.values()
            if p.kind in (p.VAR_POSITIONAL, p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        names = [
            p.name for p in self.signature.parameters.values() if p.kind == p.VAR_POSITIONAL
        ]
        self.forward_args = names[0] if names else None
        names = [p.name for p in self.signature.parameters.values() if p.kind == p.VAR_KEYWORD]
        self.forward_kwargs = names[0] if names else None
        self.forward_custom_op_schema = None
        self.forward_need_serialization = False
        self.forward_fill_kwargs = bool(self.forward_kwargs)
        assert not isinstance(
            model, (torch.nn.ModuleList, torch.nn.ModuleDict)
        ), f"ModuleList or ModuleDict should not be traced: {type(model)}"
        self.device = "cpu"

        self._debug_noquiet_name = os.environ.get("DIAGNAME", "")
        self._debug_print_status = os.environ.get("DIAGPRINTSTATUS", "")
        self._debug_print_export = os.environ.get("DIAGPRINTEXPORT", "")

    def __iter__(self) -> Iterator:
        """Iterates on all the nodes in the graph."""
        yield self
        for ch in self.children:
            yield from ch

    def get_debug_msg(self) -> str:
        """Returns information about this instances to help debugging."""
        import sys

        if inspect.ismodule(self.model):
            module_name = self.model.__name__
            module_file = self.model.__file__
        else:
            module_name = self.model.__class__.__module__
            module_file = sys.modules[self.model.__class__.__module__].__file__
        rows = [
            "",
            f"name={self.name!r}",
            f"cls={self.true_model_name!r}",
            f"module={module_name}",
            f"module_file=\n    {module_file}",
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
                    f"{indent}>>> {self.name}: {self.true_model_name}: "
                    f"DS={self.guess_dynamic_shapes()} <<<"
                )
                if with_dynamic_shape
                else f"{indent}>>> {self.name}: {self.true_model_name} <<<"
            )
        rows = [f"{indent}>>> {self.name}: {self.true_model_name}"]
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

    def __str__(self) -> str:
        "usual"
        return f"{self.__class__.__name__}(~{self.full_name}~)"

    @property
    def true_model_name(self):
        "Returns class name or module name."
        return (
            self.model.__class__.__name__
            if isinstance(self.model, torch.nn.Module)
            else self.model.__name__
        )

    @property
    def full_name(self):
        "Returns a name and class name."
        if self.method_name == "forward":
            return f"{self.name}:{self.true_model_name}"
        return f"{self.name}:{self.true_model_name}.{self.method_name}"

    @property
    def custom_op_name(self):
        "Returns a name and class name."

        def _clean_(s):
            return s.replace("[", "_").replace("]", "_")

        if self.method_name == "forward":
            if self.parent is None:
                return _clean_(f"C_{self.true_model_name}")
            return _clean_(f"{self.parent.custom_op_name}_{self.name}")
        if self.parent is None:
            return _clean_(f"C_{self.true_model_name}_{self.method_name}")
        return _clean_(f"{self.parent.custom_op_name}_{self.name}_{self.method_name}")

    @property
    def dot_name(self):
        "Returns a kind of indented name."
        prefix = "M:" if isinstance(self.model, torch.nn.Module) else "f:"
        if self.method_name == "forward":
            return f"{'..' * self.level} {prefix}{self.name}-{self.true_model_name}"
        return (
            f"{'..' * self.level} {prefix}{self.name}-"
            f"{self.true_model_name}.{self.method_name}"
        )

    @property
    def module_name_type(self):
        "Returns name and module type."
        if self.method_name == "forward":
            return f"type({self.name})={self.true_model_name}"
        return f"type({self.name})={self.true_model_name}.{self.method_name}"

    def add_inputs(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]):
        """Stores used inputs. Makes a copy."""
        for k in kwargs:
            assert self.forward_kwargs or k in self.forward_parameter_names, (
                f"Unexpected parameter {k!r} (not found in {self.forward_parameter_names}), "
                f"name={self.name!r}, model={self.true_model_name}, "
                f"module={self.model.__class__.__module__}, model={self.model}"
            )
            if self.forward_kwargs and k not in self.forward_ordered_parameter_names:
                self.forward_fill_kwargs = False

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
            f"class={self.true_model_name!r}"
        )
        dynamic = torch.export.Dim.DYNAMIC
        rk = set_length.pop()
        res = {}
        for i in range(rk):
            if len(set(s[i] for s in shapes)) > 1:
                res[i] = dynamic
        return res

    def guess_dynamic_shape_object(self, *objs: Any, msg: Optional[Callable] = None) -> Any:
        """Guesses the dynamic shapes for one argument."""
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
        assert (
            len(s1) == 1
        ), f"Different numbers of unnamed arguments {s1} for {self.full_name}"
        s2 = set(tuple(sorted(set(i[1]))) for i in self.inputs)
        assert len(s2) == 1, f"Different named arguments {s2} for {self.full_name}"
        args = []
        kwargs = {}
        for i in range(s1.pop()):
            objs = [_[0][i] for _ in self.inputs]
            args.append(
                self.guess_dynamic_shape_object(*objs, msg=lambda i=i: f" failing input {i}")
            )
        names = s2.pop()
        for name in names:
            assert name not in {"_diag", "verbose"}, (
                f"{self.full_name}: unexpected parameter {name!r}, names={names}"
                f"\ninputs[0]={string_type(self.inputs[0], with_shape=True)}"
                f"\ninputs[1]={string_type(self.inputs[1], with_shape=True)}"
            )

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
        return args, kwargs, (tuple(), kw_dyn)

    def _do_replace_by_custom_op(
        self, replace_by_custom_op: Union[bool, CustomOpStrategy, Dict[str, CustomOpStrategy]]
    ) -> CustomOpStrategy:
        """Tells if a module must be replaced by a custom op."""
        if isinstance(replace_by_custom_op, bool):
            return (
                CustomOpStrategy.ONLY_IF_FAILING
                if replace_by_custom_op
                else CustomOpStrategy.NONE
            )
        if isinstance(replace_by_custom_op, CustomOpStrategy):
            return replace_by_custom_op
        if isinstance(replace_by_custom_op, set):
            if self.true_model_name in replace_by_custom_op:
                return replace_by_custom_op[self.true_model_name]
            if self.name.__class__.__name__ in replace_by_custom_op:
                return replace_by_custom_op[self.name.__class__.__name__]
        return False

    def _annotation_from_type(self, obj) -> str:
        if isinstance(obj, torch.Tensor):
            return "Tensor"
        if isinstance(obj, (tuple, list)) and all(isinstance(t, torch.Tensor) for t in obj):
            return ["Tensor" for _t in obj]
        if isinstance(obj, dict) and all(isinstance(t, torch.Tensor) for t in obj.values()):
            return ["Tensor" for _t in obj]
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
        if isinstance(obj, bool):
            return "bool"
        if isinstance(obj, float):
            return "float"
        if isinstance(obj, int):
            return "int"

        # Let's use torch to flatten the list.
        flat, _spec = torch.utils._pytree.tree_flatten(obj)
        return ["Tensor" for _ in flat]

    def _annotated_input(self, name):
        args, kwargs = self.inputs[0]
        if name in kwargs:
            o = kwargs[name]
            annotated = self._annotation_from_type(o)
        else:
            index = self.forward_ordered_parameter_names.index(name)
            assert index < len(args), (
                f"{self.full_name}: unexpected index={index} for name={name!r}, "
                f"forward_ordered_parameter_names={self.forward_ordered_parameter_names}, "
                f"args={string_type(args, with_shape=True)}"
            )
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

    def build_c_schema(self, verbose: int = 0) -> str:
        """Returns a schema for the C function."""
        # schema_str = return f"({', '.join(params)}) -> {ret}"
        args = []
        for p in self.forward_ordered_parameter_names:
            if p in (self.forward_args, self.forward_kwargs):
                args.append(f"Tensor[] {p}")
            else:
                args.append(self._annotated_input(p))
        return f"({', '.join(args)}) -> {self._annotated_output()}"

    def _register(
        self, fct: Callable, fct_shape: Callable, namespace: str, fname: str, verbose: int = 0
    ):
        schema_str = self.build_c_schema(verbose=verbose)
        if self._debug_print_export:
            print(f"-- REGISTER: {self.custom_op_name} - {self.full_name}")
            print(f"   schema_str={schema_str}")
            print(f"   inputs={string_type(self.inputs[0], limit=20)}")
        assert (
            "**" not in schema_str
        ), f"{self.full_name}: '**' is not support in {schema_str!r}"
        if verbose > 2:
            print(f"[try_export._register] {self.dot_name} schema_str={schema_str!r}")

        # registration
        assert "." not in fname, f"Not dot allowed in fname={fname!r}"
        custom_def = torch.library.CustomOpDef(namespace, fname, schema_str, fct)
        custom_def.register_kernel(self.device)(fct)
        custom_def._abstract_fn = fct_shape
        return custom_def, schema_str

    def build_shape_mapping_indices(
        self, shape_functions: Optional[Dict[str, Callable]] = None, verbose: int = 0
    ) -> List[Tuple[Union[int, Tuple[int, ...]], torch.dtype, Optional[Callable]]]:
        """
        Builds a mapping output and input shapes so that a function
        returns dynamic shapes can automatically inferred.

        The main idea: knowning everything is going to be serialized,
        inputs and outputs are serialized, we try to match the output
        shapes with the inputs one.

        It returns for every output:

        * a list if indices of input to consider
        * an element type
        * if the output shape is not one of the input, it adds a function
          which can automatically create it
        """
        flattened_inputs = [
            serialize_args(*i, schema=None, args_names=self.forward_ordered_parameter_names)
            for i in self.inputs
        ]
        shaped_mapped = [{} for i in flattened_inputs]
        for row in range(len(shaped_mapped)):
            inp_args, inp_kwargs = flattened_inputs[row]
            assert not inp_kwargs, f"Not implemented yet with kwargs={string_type(inp_kwargs)}"
            for i, inp in enumerate(inp_args):
                if not hasattr(inp, "shape"):
                    continue
                key = tuple(map(int, inp.shape))
                if inp.shape not in shaped_mapped[row]:
                    shaped_mapped[row][key] = []
                shaped_mapped[row][key].append(i)

        flattened_outputs = [serialize_args(i, None, schema=None) for i in self.outputs]
        n_outputs = len(flattened_outputs[0])

        indices_map = [None for _ in range(n_outputs)]

        def _msg_(i):
            return (
                f"Module {self.full_name}: shape inconsistencies for output i={i}, "
                f"the new shape cannot be automatically determined."
                f"\nflattened_inputs={string_type(flattened_inputs, with_shape=True)}, "
                f"\nflattened_outputs={string_type(flattened_outputs, with_shape=True)}, "
                f"\nshaped_mapped={shaped_mapped}, "
                f"\nindices_map={indices_map}"
            )

        shape_fct = [None for i in range(n_outputs)]
        for i in range(n_outputs):
            for row in range(len(shaped_mapped)):
                assert hasattr(flattened_outputs[row][i], "shape"), (
                    f"Not implemented for type {string_type(flattened_outputs[row][i])} "
                    f"{_msg_(i)}"
                )

                possible_values = set(
                    tuple(map(int, trow[i].shape)) for trow in flattened_outputs
                )
                if len(possible_values) == 1:
                    # It is a constant shape.
                    continue

                shape = flattened_outputs[row][i].shape
                if shape in shaped_mapped[row]:
                    obtained = min(shaped_mapped[row][shape])
                    if indices_map[i] is None:
                        indices_map[i] = obtained
                    else:
                        assert obtained == indices_map[i], _msg_(i)
                    continue

                # The output shape is not found in the inputs. We try to expand it.
                sh_fct = self.determine_shape_fct(
                    i,
                    flattened_inputs,
                    flattened_outputs,
                    verbose=verbose,
                    shape_functions=shape_functions,
                )
                assert (
                    sh_fct
                ), f"{_msg_(i)}\nNo function could be found to produce shape {shape}."
                shape_fct[i] = sh_fct

        # When a shape is not mapped, it is a constant.
        for i, mapped in enumerate(indices_map):
            if mapped is not None or shape_fct[i] is not None:
                continue
            assert isinstance(
                flattened_outputs[0][i], torch.Tensor
            ), f"Unexpected type {string_type(flattened_outputs[0][i])}, {_msg_(i)}"
            shapes = set(
                tuple(map(int, flattened_outputs[row][i].shape))
                for row in range(len(self.outputs))
            )
            assert len(shapes) == 1, f"{_msg_(i)}\nfound shapes for output {i} are {shapes}"
            indices_map[i] = shapes.pop()

        res = tuple(
            (i, f.dtype, fh) for i, f, fh in zip(indices_map, flattened_outputs[0], shape_fct)
        )
        if verbose > 3:
            print(f"[try_export.build_shape_mapping_indices] {self.full_name}")
            print(
                f"  --- flattened_inputs="
                f"{string_type(flattened_inputs[0], with_shape=True, limit=20)}"
            )
            print(
                f"  -- flattened_outputs={string_type(flattened_outputs[0], with_shape=True)}"
            )
            for i in res:
                print(f"  -- {i}")
        return res

    @classmethod
    def _find_sub_shape(cls, sub_shape: Tuple[int, ...], shape: Tuple[int, ...]) -> int:
        """
        Finds a subshape into another one. Example::

            _find_sub_shape((1,2), (16,1,2,3)) -> 1
        """
        if len(sub_shape) > len(shape):
            return -1
        for k in range(len(shape) - len(sub_shape) + 1):
            if shape[k : k + len(sub_shape)] == sub_shape:
                return k
        return -1

    def determine_shape_fct(
        self,
        output_index: int,
        flattened_inputs: List[Tuple[Tuple[Any, ...], Dict[str, Any]]],
        flattened_outputs: List[Tuple[Any, ...]],
        verbose: int = 0,
        shape_functions: Optional[Dict[str, Callable]] = None,
    ) -> Callable:
        """Determines a function producing an output shape based in this inputs."""
        assert (
            isinstance(flattened_inputs, list)
            and isinstance(flattened_outputs, list)
            and len(flattened_inputs) == len(flattened_outputs)
        ), (
            f"Expected the same number of observations in two lists, "
            f"{len(flattened_inputs)} != {len(flattened_outputs)}."
        )

        if shape_functions and (
            self.true_model_name in shape_functions or self.custom_op_name in shape_functions
        ):
            shape_functions_class = (
                shape_functions[self.true_model_name]
                if self.true_model_name in shape_functions
                else shape_functions[self.custom_op_name]
            )
            if output_index in shape_functions_class:
                fct = shape_functions_class[output_index]
                if verbose > 4:
                    print(
                        f"[try_export.determine_shape_fct] {self.full_name}, "
                        f"output_index={output_index} found custom shape function "
                        f"{fct}"
                    )

                for fin, fout in zip(flattened_inputs, flattened_outputs):
                    shape_o = tuple(map(int, fct(*fin[0], **fin[1]).shape))
                    expected_shape_o = tuple(map(int, fout[output_index].shape))
                    assert shape_o == expected_shape_o, (
                        f"Custom shape function {fct} for output_index={output_index} "
                        f"for class {self.true_model_name} is failing. "
                        f"It returns {shape_o} when the expected shape is {expected_shape_o}, "
                        f"inputs={string_type(fin, with_shape=True)}, "
                        f"outputs={string_type(fout, with_shape=True)}."
                    )

                return fct

        def list_indices(flat):
            yield from enumerate(flat[0])
            yield from flat[1].items()

        def get_input(flat, j):
            if isinstance(j, int):
                return flat[0][j]
            return flat[1][j]

        shape = tuple(map(int, flattened_outputs[0][output_index].shape))
        for j, value in list_indices(flattened_inputs[0]):
            if not hasattr(value, "shape"):
                continue
            shape_j = tuple(map(int, value.shape))
            found_j = j
            found_k = self._find_sub_shape(shape_j, shape)
            if found_k >= 0:
                # The output shape includes an input shape.
                for fin, fout in zip(flattened_inputs, flattened_outputs):
                    shape_o = tuple(map(int, fout[output_index].shape))
                    shape_i = tuple(map(int, get_input(fin, found_j).shape))
                    k2 = self._find_sub_shape(shape_i, shape_o)
                    if k2 != found_k:
                        found_k = -1
                        break
            if found_k == -1:
                continue

            # We determine
            shape_j = tuple(map(int, get_input(flattened_inputs[0], found_j).shape))
            unsqueeze_dims = tuple(
                [k for k in range(len(shape)) if k < found_k or k >= found_k + len(shape_j)]
            )
            expand_shape = tuple(
                (shape[k] if k < found_k or k >= found_k + len(shape_j) else -1)
                for k in range(len(shape))
            )

            # functions
            def _modifiy_(
                *args,
                _unsqueeze_dims=unsqueeze_dims,
                _expand_shape=expand_shape,
                _input_index=found_j,
                **kwargs,
            ):
                t = kwargs[_input_index] if _input_index in kwargs else args[_input_index]
                for dim in _unsqueeze_dims:
                    t = t.unsqueeze(dim)
                return t.expand(_expand_shape)

            # Verification
            for fin, fout in zip(flattened_inputs, flattened_outputs):
                shape_o = tuple(map(int, _modifiy_(*fin[0], **fin[1]).shape))
                expected_shape_o = tuple(map(int, fout[output_index].shape))
                if shape_o != expected_shape_o:
                    # It does not work.
                    found_j = -1
                    break

            if found_j >= 0:
                if verbose > 4:
                    print(
                        f"[try_export.determine_shape_fct] {self.full_name}, "
                        f"output_index={output_index} unsqueeze_dims={unsqueeze_dims} "
                        f"expand_shape={expand_shape}, input_index={found_j}"
                    )
                return _modifiy_

        raise NotImplementedError(
            f"{self.full_name} (custom_op_name is {self.custom_op_name!r}): "
            f"unable to produce a function able to compute the output shape "
            f"for output {output_index} (shape={shape}), a custom shape function "
            f"should be added.\n"
            f"flattened_inputs={string_type(flattened_inputs, with_shape=True, limit=20)}\n"
            f"flattened_outputs={string_type(flattened_outputs, with_shape=True, limit=20)}"
        )

    def _get_symbolic_function_for_forward_shape(
        self, shape_functions: Optional[Dict[str, Callable]] = None, verbose: int = 0
    ) -> Callable:
        """
        Returns a function computed the output shape assuming it can be inferred
        from inputs and outputs.
        """
        if all(isinstance(t, torch.Tensor) for t in self.outputs[0]) and isinstance(
            self.inputs[0][0][0], torch.Tensor
        ):
            inp_args, inp_kwargs = self.inputs[0]
            out = self.outputs[0]
            input_shape = inp_args[0].shape
            unique_output_shape = set(t.shape for t in out)
            unique_output_dtype = set(t.dtype for t in out)
            if (
                not inp_kwargs
                and len(unique_output_shape) == 1
                and len(unique_output_dtype) == 1
                and unique_output_shape.pop() == input_shape
                and unique_output_dtype.pop() == inp_args[0].dtype
            ):
                n_outputs = len(out)

                if n_outputs == 1:
                    if verbose > 2:
                        print(
                            f"[try_export._get_symbolic_function_for_forward_shape] "
                            f"{self.full_name}: 1 tensor like 1st input inp_args="
                            f"{string_type(inp_args, with_shape=True)}"
                        )

                    def _symbolic_forward_tensor_1_tensor_like_first_input(*args, **kwargs):
                        # TODO: change this
                        return torch.empty_like(args[0])

                    return _symbolic_forward_tensor_1_tensor_like_first_input

                def _symbolic_forward_tensor_n_tensor_like_first_input(
                    *args, _n_outputs=n_outputs, **kwargs
                ):
                    return tuple(torch.empty_like(args[0]) for t in range(_n_outputs))

                if verbose > 2:
                    print(
                        f"[try_export._get_symbolic_function_for_forward_shape] "
                        f"{self.full_name}: n tensors like 1st input inp_args="
                        f"{string_type(inp_args, with_shape=True)}"
                    )
                return _symbolic_forward_tensor_n_tensor_like_first_input

        indices_map = self.build_shape_mapping_indices(
            verbose=verbose, shape_functions=shape_functions
        )
        if indices_map is not None:
            if verbose > 2:
                print(
                    f"[try_export._get_symbolic_function_for_forward_shape] "
                    f"-- {self.full_name}, len(indices_map)={len(indices_map)}"
                )

            def _symbolic_forward_tensor_mapped_io_shapes(
                *args, _indices_map=indices_map, **kwargs
            ):
                outputs = []
                for ii, dtype, shape_fct in _indices_map:
                    if shape_fct is None:
                        out = (
                            torch.empty(ii)
                            if isinstance(ii, tuple)
                            else torch.empty_like(args[ii])
                        )
                    else:
                        out = shape_fct(*args, **kwargs)
                    outputs.append(out if out.dtype == dtype else out.to(dtype))

                if len(outputs) == 1 and isinstance(self.outputs[0][0], torch.Tensor):
                    return outputs[0]
                return tuple(outputs)

            return _symbolic_forward_tensor_mapped_io_shapes

        raise NotImplementedError(
            f"{self.full_name}: unable to create function producing the symbolic shapes, "
            f"input_types={string_type(self.inputs[0], with_shape=True)}, "
            f"output_types={string_type(self.outputs[0], with_shape=True)}"
        )

    def _put_custom_op_inplace_tensor(
        self, shape_functions: Optional[Dict[str, Callable]] = None, verbose: int = 0
    ):
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

        if verbose > 2:
            # Let's check the rewrite function is working.
            print(
                f"[try_export._rewrite_forward_tensor_] {self.full_name}: "
                f"check _rewrite_forward_tensor_ is working"
            )
            got = _rewrite_forward_tensor_(*self.inputs[0][0], **self.inputs[0][1])
            diff = max_diff(self.outputs[0], got)
            print(
                f"[try_export._rewrite_forward_tensor_] {self.full_name}: "
                f"done checking with diff={diff}"
            )

        name_fct = self.custom_op_name
        if verbose > 2:
            print(
                f"[try_export.put_custom_op_inplace_tensor] {self.dot_name}: "
                f"registers diag_lib.{name_fct}"
            )

        shape_fct = self._get_symbolic_function_for_forward_shape(verbose=verbose)
        cusdef, schema_str = self._register(
            _rewrite_forward_tensor_,
            shape_fct,
            "diag_lib",
            name_fct,
            verbose=verbose,
        )
        assert cusdef is not None, f"{self.full_name}: registration of a custom op has failed."
        if verbose > 2:
            print(
                f"[try_export.put_custom_op_inplace_tensor] {self.dot_name}: "
                f"schema_str={schema_str!r}"
            )
        # We stored to avoid the registration twice.
        self.forward_custom_op_schema = cusdef
        expected_output_type = [type_as_str_with_info(o) for o in self.outputs[0]]
        self.forward_expected_output_type = expected_output_type
        if verbose > 2:
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
                    f"schema_str={schema_str!r}"
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
        # self.model.forward = _replaced_forward_tensor_
        setattr(self.model, self.method_name, _replaced_forward_tensor_)
        return cusdef

    def _put_custom_op_inplace_any(
        self, shape_functions: Optional[Dict[str, Callable]] = None, verbose: int = 0
    ):
        """
        Replaces the submodule by a custom operator.
        It rewrites the forward method to call a function.
        Any type among the supported list if support (an exception will
        be raised otherwise). C++ does not support custom types so
        serialization and deserialization are needed.
        """

        def _rewrite_forward_(*args, _diag=self, **kwargs):
            if verbose > 2:
                print(
                    f"[_rewrite_forward_] {_diag.full_name}-SERIALIZE_IN: "
                    f"args={string_type(args, with_shape=True, limit=20)}, "
                    f"kwargs={string_type(kwargs, with_shape=True, limit=20)}, "
                    f"_diag.forward_expected_input_type={_diag.forward_expected_input_type}, "
                    f"_diag.forward_fill_kwargs={_diag.forward_fill_kwargs}"
                )
            # We need to deserialize back before calling forward.
            new_args, new_kwargs = deserialize_args_kwargs(
                args,
                kwargs,
                _diag.forward_expected_input_type,
                clone=True,
                ordered_names=_diag.forward_ordered_parameter_names,
                fill_kwargs=_diag.forward_fill_kwargs,
            )
            if verbose > 2:
                print(
                    f"[_rewrite_forward_] {_diag.full_name}-IN: "
                    f"args={string_type(new_args, with_shape=True)}, "
                    f"kwargs={string_type(new_kwargs, with_shape=True)}"
                )
                sfct = str(_diag.forward).replace("\n", " ")
                print(f"[_rewrite_forward_] {_diag.full_name}-CALL: {sfct}")
            res = _diag.forward(*new_args, **new_kwargs)
            if verbose > 2:
                print(
                    f"[_rewrite_forward_] {_diag.full_name}-OUT: "
                    f"res={string_type(res, with_shape=True)}"
                )
                if verbose > 3:
                    print(f"[_rewrite_forward_] schema={_diag.forward_custom_op_schema}")
            # And we need to serialize before before returning the output.
            serialized_res = serialize_args(res, None, _diag.forward_custom_op_schema)
            if verbose > 2:
                print(
                    f"[_rewrite_forward_] {_diag.full_name}-SERIALIZE-OUT: "
                    f"args={string_type(serialized_res, with_shape=True)}"
                )
            return serialized_res

        name_fct = self.custom_op_name
        if verbose > 2:
            print(
                f"[try_export.put_custom_op_inplace] {self.dot_name}: "
                f"registers 'diag_lib.{name_fct}"
            )

        shape_fct = self._get_symbolic_function_for_forward_shape(
            verbose=verbose, shape_functions=shape_functions
        )
        cusdef, schema_str = self._register(
            _rewrite_forward_,
            shape_fct,
            "diag_lib",
            name_fct,
            verbose=verbose,
        )
        assert cusdef is not None, f"{self.full_name}: registration of a custom op has failed."
        if verbose > 2:
            print(
                f"[try_export.put_custom_op_inplace] {self.dot_name}: "
                f"schema_str={schema_str!r}"
            )
        # We stored to avoid the registration twice.
        self.forward_custom_op_schema = cusdef
        expected_output_type = [type_as_str_with_info(o) for o in self.outputs[0]]
        self.forward_expected_output_type = expected_output_type
        self.forward_expected_input_type = (
            [type_as_str_with_info(o) for o in self.inputs[0][0]],
            {k: type_as_str_with_info(v) for k, v in self.inputs[0][1].items()},
        )

        if verbose > 2:
            print(
                f"[try_export.put_custom_op_inplace] {self.dot_name}: "
                f"expected_output_type={expected_output_type}"
            )
            # Let's check the rewrite function is working.
            print(
                f"[try_export._rewrite_forward_] {self.full_name}: "
                f"check _rewrite_forward_ is working with "
                f"{string_type(self.inputs[0], with_shape=True)}"
            )
            # The schema is misleading as everything is serialized into a flat list
            # of tensors. It is better to use *forward_ordered_parameter_names*.
            a, kw = serialize_args(
                *self.inputs[0], schema=None, args_names=self.forward_ordered_parameter_names
            )
            print(
                f"[try_export._rewrite_forward_] {self.full_name}: serialized into "
                f"{string_type(a, with_shape=True, limit=20)} and "
                f"{string_type(kw, with_shape=True, limit=20)}, "
                f"schema_str={schema_str!r}"
            )
            got = _rewrite_forward_(*a, **kw)
            print(
                f"[try_export._rewrite_forward_] {self.full_name}: "
                f"check shape_fct={shape_fct} is working with "
                f"{string_type(self.inputs[0], with_shape=True)}"
            )
            got_shape = shape_fct(*a, **kw)
            print(
                f"[try_export._rewrite_forward_] {self.full_name}: "
                f"check done, forward returned "
                f"{string_type(got, with_shape=True)}, shape_fct returned "
                f"{string_type(got_shape, with_shape=True)}, outputs[0]="
                f"{string_type(self.outputs[0], with_shape=True)}"
            )

        # Apparently, we need a function with the exact same signature.
        def _replaced_forward_(*args, **kwargs):
            fct = getattr(torch.ops.diag_lib, name_fct)
            if verbose > 2:
                print(
                    f"[_replaced_forward_] {name_fct}-IN: "
                    f"args={string_type(args)}, kwargs={string_type(kwargs)}, "
                    f"schema_str={schema_str!r}, "
                    f"self.forward_fill_kwargs={self.forward_fill_kwargs}, "
                    f"self.forward_ordered_parameter_names={self.forward_ordered_parameter_names}"
                )
            # , args_names=self.forward_ordered_parameter_names
            _args, _kwargs = args, kwargs
            has_kwargs = bool(kwargs)
            args, kwargs = serialize_args(
                args, kwargs, schema=None, args_names=self.forward_ordered_parameter_names
            )
            if has_kwargs and self.forward_fill_kwargs:
                assert args and isinstance(args[-1], list) and len(args[-1]) == 0, (
                    f"Unexpected value for args[-1], "
                    f"schema_str={schema_str!r}, "
                    f"self.forward_fill_kwargs={self.forward_fill_kwargs}, "
                    f"self.forward_ordered_parameter_names={self.forward_ordered_parameter_names}"
                    f"\nargs={string_type(_args, limit=20)},\nkwargs={string_type(_kwargs)}"
                    f"\n-- after serialization --"
                    f"\nargs={string_type(args, limit=20)},\nkwargs={string_type(kwargs)}"
                )
                args = args[:-1]

            if self.forward_fill_kwargs:
                args = (*args, [])
            if verbose > 2:
                print(
                    f"[_replaced_forward_] {name_fct}-SERIALIZED_IN: "
                    f"args={string_type(args, with_shape=True)}, "
                    f"kwargs={string_type(kwargs, with_shape=True)}"
                )
                print(f"[_replaced_forward_] {name_fct}-CALL: {fct}")
            if self._debug_print_export:
                print(f"-- CALL custom op {self.custom_op_name} - {self.full_name}")
                print(f"   args={string_type(args, limit=20)}")
                print(f"   kwargs={string_type(kwargs, limit=20)}")
                print(f"   schema_str={schema_str}")
            res = fct(*args, **kwargs)
            if verbose > 2:
                print(
                    f"[_replaced_forward_] {name_fct}-SERIALIZED_OUT: "
                    f"res={string_type(res, with_shape=True)}, "
                    f"expected_output_type={expected_output_type}"
                )
            des = deserialize_args(res, expected_output_type)
            if verbose > 2:
                print(
                    f"[_replaced_forward_] {name_fct}-OUT: "
                    f"des={string_type(des, with_shape=True)}"
                )
            if isinstance(des, torch.Tensor):
                return des
            return tuple(des) if len(des) > 1 else des[0]

        _replaced_forward_.__signature__ = inspect.Signature.from_callable(self.forward)
        self.forward_calling_custom_op = _replaced_forward_
        # self.model.forward = _replaced_forward_
        setattr(self.model, self.method_name, _replaced_forward_)
        return cusdef

    def put_custom_op_inplace(
        self, shape_functions: Optional[Dict[str, Callable]] = None, verbose: int = 0
    ):
        """
        Replaces the submodule by a custom operator.
        It rewrites the forward method to call a function
        """
        if verbose > 2:
            print(f"[try_export.put_custom_op_inplace] {self.dot_name}")
        if self.forward_custom_op_schema is not None:
            # Registration was already done.
            # self.model.forward = self.forward_calling_custom_op
            setattr(self.model, self.method_name, self.forward_calling_custom_op)
            return self.forward_custom_op_schema

        if all(isinstance(t, torch.Tensor) for t in self.inputs[0]) and all(
            isinstance(t, torch.Tensor) for t in self.outputs[0]
        ):
            self.forward_custom_op_serialize = False
            return self._put_custom_op_inplace_only_tensor(
                verbose=verbose, shape_functions=shape_functions
            )
        self.forward_custom_op_serialize = True
        self.forward_need_serialization = self.forward_custom_op_serialize
        return self._put_custom_op_inplace_any(
            verbose=verbose, shape_functions=shape_functions
        )

    def remove_custom_op_inplace(self, verbose: int = 0):
        """
        Just replaces the forward, hoping the registration does not have to
        be removed.
        """
        self.forward_need_serialization = False
        # self.model.forward = self.forward
        setattr(self.model, self.method_name, self.forward)
        if verbose > 2:
            print(
                f"[try_export.remove_custom_op_inplace] {self.dot_name}: unregisters "
                f"'diag_lib.{self.custom_op_name}"
            )

    def is_customized(self):
        """Tells of this module was bypassed."""
        return (
            hasattr(self, "forward_calling_custom_op")
            and getattr(self.model, self.method_name) == self.forward_calling_custom_op
        )

    def _post_process_dynamic_shapes(self, args, kwargs, dynamic_shapes, debug, verbose):
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
        ds = (
            choose_kwargs_for_dynamic_shapes(
                *dynamic_shapes, self.forward_positioned_parameter_names
            )
            if dynamic_shapes[0] and dynamic_shapes[1]
            else (dynamic_shapes[0] or dynamic_shapes[1])
        )
        return args, kwargs, ds

    def _try_export_no_bypass_export(
        self,
        export_inputs,
        exporter_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        quiet: bool = True,
        use_dynamic_shapes: Optional[bool] = None,
        shape_functions: Optional[Dict[str, Callable]] = None,
    ) -> Tuple[Optional[Any], Optional[Callable]]:
        if quiet:
            quiet = self._debug_noquiet_name != self.name
            debug = not quiet
        else:
            debug = False

        args, kwargs = export_inputs
        dynamic_shapes = self.guess_dynamic_shapes() if use_dynamic_shapes else None
        args, kwargs, ds = self._post_process_dynamic_shapes(
            args, kwargs, dynamic_shapes, debug, verbose
        )
        if debug or verbose > 1:
            sds = str(ds).replace("<_DimHint.DYNAMIC: 3>", "DYN")
            print(f"[try_export-FX] {self.dot_name}: convert with dynamic_shapes={sds}")
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

        if debug or self._debug_print_export:
            print("-- DEBUG")
            print(f"[try_export-FX] {self.dot_name}")
            print(f"inputs={string_type(self.inputs[0], with_shape=True, with_min_max=True)}")
            print(f"  args={string_type(args, with_shape=True, with_min_max=True)}")
            print(f"kwargs={string_type(kwargs, with_shape=True, with_min_max=True)}")
            if dynamic_shapes:
                print(f"   ds0={dynamic_shapes}")
            if ds:
                print(f"    ds={ds}")
            if exporter_kwargs:
                print(f"    exporter_kwargs={exporter_kwargs}")

        if inspect.ismodule(self.model):
            # The function needs to be wrapped into a module before being exported.
            class LocalModuleCallingAFunction(torch.nn.Module):
                def __init__(self, local_f):
                    super().__init__()
                    self.local_f = local_f

                def forward(self, *args, **kwargs):
                    return self.local_f(*args, **kwargs)

            if not kwargs and isinstance(ds, tuple):
                ds = (ds,)
            else:
                raise NotImplementedError(
                    f"{self.full_name}: exporting function {self.forward} "
                    f"with:\nargs={string_type(args)},"
                    f"\nforward_positioned_parameter_names={self.forward_positioned_parameter_names}"
                    f"\nkwargs={string_type(kwargs)},\nds={ds}"
                    f"\ndynamic_shapes={dynamic_shapes}"
                    f"\ninputs[0]={string_type(self.inputs[0])}"
                )
            model_to_export = LocalModuleCallingAFunction(self.forward)
        else:
            model_to_export = self.model

        if quiet:
            try:
                ep = torch.export.export(
                    model_to_export,
                    args,
                    kwargs=kwargs,
                    dynamic_shapes=ds,
                    **(exporter_kwargs or {}),
                )
                self.exporter_status = StatusExport(StatusExportCode.OK, exported=ep)
                if self._debug_print_status:
                    print(
                        f"-- torch.export.export name={self.full_name} -- "
                        f"{'CUSTOM -- ' if self.is_customized() else ''}"
                        f"{self.exporter_status.status.name} "
                        f"-- _try_export_no_bypass_export-1"
                    )
            except Exception as e:
                if "'NoneType' object is not iterable" in str(e) or "stop" in str(e):
                    raise
                self.last_error = e
                se = str(e).split("\n")[0].replace("<_DimHint.DYNAMIC: 3>", "DYN")
                self.exporter_status = StatusExport(
                    StatusExportCode.FAIL, reason=se, step="EXPORT"
                )
                if self._debug_print_status:
                    print(
                        f"-- torch.export.export name={self.full_name} -- "
                        f"{'CUSTOM -- ' if self.is_customized() else ''}"
                        f"{self.exporter_status.status.name} -- {se} "
                        f"-- _try_export_no_bypass_export-2"
                    )
                if verbose:
                    if self.exporter_status.is_ok():
                        print(
                            f"[try_export-FX] {self.dot_name} --- "
                            f"{self.exporter_status.status.name}"
                        )
                    else:
                        print(
                            f"[try_export-FX] {self.dot_name} --- "
                            f"{self.exporter_status.status.name}, "
                            f"step={self.exporter_status.step}, "
                            f"reason={self.exporter_status.reason}"
                        )
                return self.exporter_status, None
        else:
            ep = torch.export.export(
                model_to_export,
                args,
                kwargs=kwargs,
                dynamic_shapes=ds,
                **(exporter_kwargs or {}),
            )
            self.exporter_status = StatusExport(StatusExportCode.OK, exported=ep)
            if self._debug_print_status:
                print(
                    f"-- torch.export.export name={self.full_name} -- "
                    f"{'CUSTOM -- ' if self.is_customized() else ''}"
                    f"{self.exporter_status.status.name} "
                    f"-- _try_export_no_bypass_export-3"
                )
        if verbose > 1:
            print(
                f"[try_export-FX] {self.dot_name}: done, "
                f"{'CUSTOMIZED -- ' if self.is_customized() else ''}"
                f"status={self.exporter_status.status.name}"
            )
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

        return self.exporter_status, _call_model_

    def _try_export_no_bypass(
        self,
        modificator: Optional[Callable] = None,
        exporter: str = "fx",
        exporter_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        quiet: bool = True,
        discrepancies: bool = True,
        use_dynamic_shapes: Optional[bool] = None,
        replace_by_custom_op: CustomOpStrategy = CustomOpStrategy.NONE,
        atol: float = 1e-2,
        rtol: float = 1e-1,
        shape_functions: Optional[Dict[str, Callable]] = None,
    ) -> Tuple[StatusExport, Optional[Callable]]:
        """
        Tries to export the module of submodule held by this class.
        Stores intermediate results of the export in attributes prefixed by ``forward_``.
        """
        assert isinstance(
            replace_by_custom_op, bool
        ), f"Not implemented yet for replace_by_custom_op={replace_by_custom_op!r}"
        export_inputs = modificator(self.inputs[0]) if modificator else self.inputs[0]
        export_inputs = make_copy(export_inputs)
        if use_dynamic_shapes is None:
            use_dynamic_shapes = len(self.inputs) > 1
        assert (
            not use_dynamic_shapes or len(self.inputs) > 1
        ), "Unable to use dynamic_shapes, only one set of inputs is available."
        cusdef = (
            self.put_custom_op_inplace(verbose=verbose, shape_functions=shape_functions)
            if replace_by_custom_op
            else None
        )

        if exporter == "fx":
            exported, fct = self._try_export_no_bypass_export(
                export_inputs,
                exporter_kwargs=exporter_kwargs,
                verbose=verbose,
                quiet=quiet,
                use_dynamic_shapes=use_dynamic_shapes,
            )
            if self._debug_print_status:
                print(
                    f"-- torch.export.export name={self.full_name} -- "
                    f"{'CUSTOM -- ' if self.is_customized() else ''}"
                    f"{exported.status.name} "
                    f"-- _try_export_no_bypass-4"
                )
            assert exported.is_ok() or exported.reason, (
                f"Confused status: exported.status={exported.status.name!r}, "
                f"exported.reason={exported.reason!r}"
            )
            setattr(self, exporter, exported.exported)
        else:
            raise NotImplementedError(f"Export not implemented yet for exporter={exporter!r}")
        if not exported.is_ok():
            if cusdef is not None:
                self.remove_custom_op_inplace(verbose=verbose)
                exported.status = exported.status | StatusExportCode.CUSTOM
            return exported, fct

        assert fct is not None, (
            f"exported.status={exported.status!r}, but fct is None, "
            f"exported.is_ok()={exported.is_ok()}"
        )

        if discrepancies:
            has_disc = False
            self.exporter_outputs = []
            self.exporter_discs = []
            for i, (inp, out) in enumerate(zip(self.inputs, self.outputs)):
                copy_inp = make_copy(modificator(inp) if modificator else inp)
                args, kwargs = copy_inp
                if verbose > 2:
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
                        print(exported.exported)
                        print(exported.exported.graph)
                if quiet:
                    try:
                        got = fct(*args, **kwargs)
                    except Exception as e:
                        self.last_error = e
                        se = str(e).split("\n")[0]
                        self.exporter_status.status = (
                            self.exporter_status.status.remove(StatusExportCode.OK)
                            | StatusExportCode.FAIL
                        )
                        self.exporter_status.status.step = "EVAL"
                        self.exporter_status.reason = se
                        if "Ran into a kwarg keyword misma" in se:
                            raise
                        if self._debug_print_status:
                            print(
                                f"-- torch.export.export name={self.full_name} -- "
                                f"{'CUSTOM -- ' if self.is_customized() else ''}"
                                f"{exported.status.name}/{self.exporter_status.status.name}"
                                f"/{exported.step} -- _try_export_no_bypass-12"
                            )
                            print(self.exporter_status.reason)
                            print(self.exporter_status.exported)
                        break
                else:
                    got = fct(*args, **kwargs)

                self.exporter_outputs.append(got)
                diff = max_diff(out, got)
                if verbose > 1:
                    print(f"[try_export-{exporter.upper()}] {self.dot_name}: diff[{i}]={diff}")
                self.exporter_discs.append(diff)
                if diff["abs"] > atol or diff["rel"] > rtol:
                    has_disc = True
                    if not quiet:
                        raise AssertionError(
                            f"{self.full_name}: discrepancies were observed, "
                            f"diff={diff}, \nargs="
                            f"{string_type(args, with_shape=True, with_min_max=True)}, "
                            f"\nkwargs="
                            f"{string_type(kwargs, with_shape=True, with_min_max=True)}, "
                            f"\nexpected="
                            f"{string_type(out, with_shape=True, with_min_max=True)}, "
                            f"\ngot={string_type(got, with_shape=True, with_min_max=True)}."
                        )
                    break

            if has_disc:
                self.exporter_status.status = (
                    exported.status.remove(StatusExportCode.OK)
                    | StatusExportCode.FAIL
                    | StatusExportCode.DISC
                )
                self.exporter_status.step = "DISC"
                self.exporter_status.reason = f"discrepancies: {diff}"
                if self._debug_print_status:
                    print(
                        f"-- torch.export.export name={self.full_name} -- "
                        f"{'CUSTOM -- ' if self.is_customized() else ''}"
                        f"{exported.status.name}/{self.exporter_status.status.name}"
                        f"/{exported.step} -- _try_export_no_bypass-13"
                    )

        assert exported.is_ok() or exported.reason, (
            f"Confused status: exported.status={exported.status.name!r}, "
            f"exported.reason={exported.reason!r}"
        )
        if cusdef is not None:
            self.remove_custom_op_inplace(verbose=verbose)
        return exported, fct

    def _try_export(
        self,
        exporter: str = "fx",
        exporter_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        quiet: bool = True,
        discrepancies: bool = True,
        use_dynamic_shapes: Optional[bool] = None,
        replace_by_custom_op: CustomOpStrategy = CustomOpStrategy.NONE,
        atol: float = 1e-2,
        rtol: float = 1e-1,
        shape_functions: Optional[Dict[str, Callable]] = None,
    ) -> Tuple[StatusExport, Optional[Any]]:
        """
        Tries to export the module of submodule held by this class.
        Stores intermediate results of the export in attributes prefixed by ``forward_``.
        """
        assert isinstance(
            replace_by_custom_op, bool
        ), f"Not implemented yet for replace_by_custom_op={replace_by_custom_op!r}"

        exported, fct = self._try_export_no_bypass(
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
            shape_functions=shape_functions,
        )
        if self._debug_print_status:
            print(
                f"-- torch.export.export name={self.full_name} -- "
                f"{'CUSTOM -- ' if self.is_customized() else ''}"
                f"{exported.status.name}/{self.exporter_status.status.name}"
                f"/{exported.step} -- _try_export-5"
            )
        return exported, fct

    def verifies(self, verbose: int = 0):
        """Does some verifications. Raises an exception if it fails."""
        assert (
            isinstance(self.inputs, list)
            and all(isinstance(i, tuple) for i in self.inputs)
            and all(len(i) == 2 for i in self.inputs)
            and all(isinstance(i[0], tuple) and isinstance(i[1], dict) for i in self.inputs)
        ), (
            f"{self.full_name}: unexpected type for self.inputs: "
            f"{string_type(self.inputs)}{self.get_debug_msg()}"
        )
        assert isinstance(self.outputs, list) and all(
            isinstance(i, tuple) for i in self.outputs
        ), (
            f"{self.full_name}: unexpected type for self.outputs: "
            f"{string_type(self.outputs)}{self.get_debug_msg()}"
        )
        assert len(self.inputs) == len(self.outputs), (
            f"{self.full_name}: input and outputs mismatch: "
            f"{len(self.inputs)} != {len(self.outputs)}{self.get_debug_msg()}"
        )
        assert (
            len(self.inputs) > 0
        ), f"{self.full_name}: expecting at least one input{self.get_debug_msg()}"
        for child in self.children:
            child.verifies()

    def try_export(
        self,
        exporter: str = "fx",
        exporter_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        quiet: bool = True,
        discrepancies: bool = True,
        use_dynamic_shapes: Optional[bool] = None,
        replace_by_custom_op: Union[
            bool, CustomOpStrategy, Dict[str, CustomOpStrategy]
        ] = CustomOpStrategy.NONE,
        atol: float = 1e-2,
        rtol: float = 1e-1,
        shape_functions: Optional[Dict[str, Callable]] = None,
    ) -> StatusExport:
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
        :param verbose: verbosity, to see what the function is doing
        :param discrepancies: run the exported model to measure the discrepancies
        :param quiet: do not catch the first exception
        :param use_dynamic_shapes: use dynamic shapes
        :param replace_by_custom_op: before exporting,
            it replaces submodules by custom ops,
            it can be a boolean to replace all or a selected classes (name or type), or names
        :param atol: absolute tolerance
        :param rtol: relative tolerance
        :param shape_functions: dictionary of functions to compute the shape of the output,
            the signature should be the following
            ``fct(_output_index:i, *args, **kwargs) -> Optional[Any]``.
            If it returns None, the shape is automacally computed.
            The key of the dictionary is a class name, the class of the submodule
            to handle with this function.
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
        self.verifies()
        custom_op_strat = self._do_replace_by_custom_op(replace_by_custom_op)
        if verbose and self.level == 0:
            print()

        if custom_op_strat in (
            CustomOpStrategy.ONLY_IF_FAILING,
            CustomOpStrategy.ALWAYS,
            CustomOpStrategy.NONE,
        ) or (custom_op_strat == CustomOpStrategy.LOCAL and not self.children):
            if verbose > 1:
                print(
                    f"[try-export-{exporter.upper()}] {'.' * self.level * 2} START "
                    f"{custom_op_strat.name}, no custom op for {self.full_name}"
                )
            exported, _fct = self._try_export(
                exporter=exporter,
                exporter_kwargs=exporter_kwargs,
                verbose=verbose,
                quiet=quiet,
                discrepancies=discrepancies,
                use_dynamic_shapes=use_dynamic_shapes,
                replace_by_custom_op=custom_op_strat == CustomOpStrategy.ALWAYS,
                atol=atol,
                rtol=rtol,
                shape_functions=shape_functions,
            )
            if self._debug_print_status:
                print(
                    f"-- torch.export.export name={self.full_name} -- "
                    f"{'CUSTOM -- ' if self.is_customized() else ''}"
                    f"{exported.status.name}/{self.exporter_status.status.name}"
                    f"-- try_export-8"
                )
            assert exported.is_ok() or exported.reason, (
                f"Confused status: exported.status={exported.status.name!r}, "
                f"exported.reason={exported.reason!r}"
            )
            if verbose > 1:
                print(
                    f"[try-export-{exporter.upper()}] {'.' * self.level * 2} -DONE "
                    f"{custom_op_strat.name} for {self.full_name} no custom op, "
                    f"name={self.name!r}, exported={exported.exported.__class__.__name__}, "
                    f"status={self.exporter_status.status.name}"
                )
        else:
            exported = None

        if (
            (exported is None or not exported.is_ok())
            and custom_op_strat
            in (
                CustomOpStrategy.ONLY_IF_FAILING,
                CustomOpStrategy.LOCAL,
                CustomOpStrategy.ALWAYS,
            )
            and self.children
        ):
            # The conversion of the model with its submodule failed.
            # We try to export the module without its children.
            if verbose > 1:
                print(
                    f"[try-export-{exporter.upper()}] {'.' * self.level * 2} START "
                    f"{custom_op_strat.name}, name={self.full_name!r} "
                    f"--- children replaced by custom ops "
                    f"[{', '.join(ch.full_name for ch in self.children)}]"
                )
                if verbose >= 10:
                    print(
                        f"[try-export-{exporter.upper()}] {'.' * self.level * 2} "
                        f"       args={string_type(self.inputs[0][0], with_shape=True)}"
                    )
                    print(
                        f"[try-export-{exporter.upper()}] {'.' * self.level * 2} "
                        f"     kwargs={string_type(self.inputs[0][1], with_shape=True)}"
                    )
                    print(
                        f"[try_export-{exporter.upper()}] {'.' * self.level * 2} "
                        f"    outputs={string_type(self.outputs[0], with_shape=True)}"
                    )
            elif verbose:
                print(
                    f"[try_export-{exporter.upper()}] {self.dot_name} "
                    f"--- children replaced by custom ops "
                    f"[{', '.join(ch.full_name for ch in self.children)}]"
                )

            for child in self.children:
                if verbose >= 10:
                    print(
                        f"[try_export-{exporter.upper()}]     "
                        f"child {child.full_name} as custom op"
                    )
                    print(
                        f"[try_export-{exporter.upper()}]           "
                        f"args={string_type(child.inputs[0][0], with_shape=True)}"
                    )
                    print(
                        f"[try_export-{exporter.upper()}]         "
                        f"kwargs={string_type(child.inputs[0][1], with_shape=True)}"
                    )
                    print(
                        f"[try_export-{exporter.upper()}]        "
                        f"outputs={string_type(child.outputs[0], with_shape=True)}"
                    )
                child.put_custom_op_inplace(verbose=verbose, shape_functions=shape_functions)

            # We export again.
            exported, _fct = self._try_export(
                exporter=exporter,
                exporter_kwargs=exporter_kwargs,
                verbose=verbose,
                quiet=quiet,
                discrepancies=discrepancies,
                use_dynamic_shapes=use_dynamic_shapes,
                replace_by_custom_op=False,
                atol=atol,
                rtol=rtol,
                shape_functions=shape_functions,
            )
            if self._debug_print_status:
                print(
                    f"-- torch.export.export name={self.full_name} -- "
                    f"{'CUSTOM -- ' if self.is_customized() else ''}"
                    f"{exported.status.name} "
                    f"-- try_export-7"
                )
            assert exported.is_ok() or exported.reason, (
                f"Confused status: exported.status={exported.status.name!r}, "
                f"exported.reason={exported.reason!r}"
            )

            # We restore the initial state.
            for child in self.children:
                child.remove_custom_op_inplace(verbose=verbose)

            exported.status = exported.status | StatusExportCode.CHILDC

            if verbose > 1:
                print(
                    f"[try-export-{exporter.upper()}] {'.' * self.level * 2} -DONE "
                    f"{custom_op_strat.name}, name={self.full_name!r}, "
                    f"exported={exported.__class__.__name__}, "
                    f"status={self.exporter_status.status.name}"
                )
            elif verbose:
                print(
                    f"[try_export-{exporter.upper()}] {self.dot_name} "
                    f"--- {self.exporter_status.status.name}: "
                    f"{self.exporter_status.short_reason}"
                )

        assert exported is not None, (
            f"Something went wrong, custom_op_strat={custom_op_strat}, "
            f"#children={len(self.children)}"
        )
        if not self.children or (
            custom_op_strat == CustomOpStrategy.NONE and exported.is_ok()
        ):
            # We don't want to return if custom ops were applied,
            # we need to look into every of them.
            if verbose > 1:
                print(
                    f"[try-export-{exporter.upper()}] {'.' * self.level * 2} -DONE "
                    f"{custom_op_strat.name}, name={self.name!r}, "
                    f"exported={exported.exported.__class__.__name__} "
                    f"status={self.exporter_status.status.name}"
                )
            elif verbose:
                print(
                    f"[try_export-{exporter.upper()}] {self.dot_name} "
                    f"--- {self.exporter_status.status.name}: "
                    f"{self.exporter_status.short_reason}"
                )
            return exported

        # If a custom op was used to bypass the export or
        # if the export failed, we look into the children.
        for child in self.children:
            child.try_export(
                exporter=exporter,
                exporter_kwargs=exporter_kwargs,
                verbose=verbose,
                quiet=quiet,
                discrepancies=discrepancies,
                use_dynamic_shapes=use_dynamic_shapes,
                replace_by_custom_op=replace_by_custom_op,
                atol=atol,
                rtol=rtol,
                shape_functions=shape_functions,
            )
        if self._debug_print_status:
            print(
                f"-- torch.export.export name={self.full_name} -- "
                f"{'CUSTOM -- ' if self.is_customized() else ''}"
                f"{exported.status.name}/{self.exporter_status.status.name} "
                f"-- try_export-10"
            )
        if verbose > 1:
            print(
                f"[try-export-{exporter.upper()}] {'.' * self.level * 2} =DONE "
                f"{custom_op_strat.name}, exported={exported.exported.__class__.__name__}, "
                f"status={self.exporter_status.status}"
            )
        return exported

    def get_export_report(self, exported_program: bool = False, fx: bool = False) -> str:
        """
        Returns a report status on the conversion.

        :param exported_program: adds the exported program if available
        :param fx: display the graph instead of the exported program
        :return: string
        """

        def _suffix(r):
            return "::func" if inspect.ismodule(r.model) else ""

        def iter_status(here):
            rows = [here]
            for child in here.children:
                rows.extend(iter_status(child))
            return rows

        rows = iter_status(self)
        to_display = [
            (
                f"{'..' * r.level}{r.name}{_suffix(r)}",
                (
                    r.model.__class__.__name__
                    if isinstance(r.model, torch.nn.Module)
                    else f"mod::{r.model.__name__}"
                ),
                (r.exporter_status.pretty_text() if hasattr(r, "exporter_status") else "<OK>"),
                r,
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

            diag = t[-1]
            if exported_program:
                if hasattr(diag, "fx"):
                    eps = str(diag.fx)
                    indent = " " * (mc1 + mc2)
                    prefix = f"{indent}ep: "
                    srows.append(textwrap.indent(eps, prefix))
                else:
                    indent = " " * (mc1 + mc2)
                    srows.append(f"{indent}ep: -")
            if fx:
                if hasattr(diag, "fx") and diag.fx is not None:
                    eps = str(diag.fx.graph)
                    indent = " " * (mc1 + mc2)
                    prefix = f"{indent}fx: "
                    srows.append(textwrap.indent(eps, prefix))
                else:
                    indent = " " * (mc1 + mc2)
                    srows.append(f"{indent}fx: -")

        return "\n".join(srows)

    def draft_export_local(
        self,
        use_dynamic_shapes: Optional[bool] = None,
        exporter_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        shape_functions: Optional[Dict[str, Callable]] = None,
    ):
        """
        Draft Exports with every submodule converted into a submodule.
        This can be used to understand where the conversion is failing.

        :param exporter_kwargs: argument for the export function
        :param verbose: verbosity, to see what the function is doing
        :param use_dynamic_shapes: use dynamic shapes
        :param shape_functions: dictionary of functions to compute the shape of the output,
            the signature should be the following
            ``fct(_output_index:i, *args, **kwargs) -> Optional[Any]``.
            If it returns None, the shape is automacally computed.
            The key of the dictionary is a class name, the class of the submodule
            to handle with this function.
        :return: result of the export function
        """
        if use_dynamic_shapes is None:
            use_dynamic_shapes = len(self.inputs) > 1
        args, kwargs = self.inputs[0]
        dynamic_shapes = self.guess_dynamic_shapes() if use_dynamic_shapes else None
        args, kwargs, ds = self._post_process_dynamic_shapes(
            args, kwargs, dynamic_shapes, False, verbose
        )

        import torch.export._draft_export

        for child in self.children:
            if verbose >= 10:
                print(f"[try_export-FX]     child {child.full_name} as custom op")
                print(
                    f"[try_export-FX]           "
                    f"args={string_type(child.inputs[0][0], with_shape=True)}"
                )
                print(
                    f"[try_export-FX]         "
                    f"kwargs={string_type(child.inputs[0][1], with_shape=True)}"
                )
                print(
                    f"[try_export-FX]        "
                    f"outputs={string_type(child.outputs[0], with_shape=True)}"
                )
            child.put_custom_op_inplace(verbose=verbose, shape_functions=shape_functions)
        ep_report = torch.export._draft_export.draft_export(
            self.model,
            args,
            kwargs=kwargs,
            dynamic_shapes=ds,
            **(exporter_kwargs or {}),
        )
        assert ep_report is not None, f"No result for {self.full_name}"
        # We restore the initial state.
        for child in self.children:
            child.remove_custom_op_inplace(verbose=verbose)
        return ep_report

    def export_local(
        self,
        use_dynamic_shapes: Optional[bool] = None,
        exporter_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        shape_functions: Optional[Dict[str, Callable]] = None,
    ):
        """
        Exports with every submodule converted into a submodule.

        :param exporter_kwargs: argument for the export function
        :param verbose: verbosity, to see what the function is doing
        :param use_dynamic_shapes: use dynamic shapes
        :param shape_functions: dictionary of functions to compute the shape of the output,
            the signature should be the following
            ``fct(_output_index:i, *args, **kwargs) -> Optional[Any]``.
            If it returns None, the shape is automacally computed.
            The key of the dictionary is a class name, the class of the submodule
            to handle with this function.
        :return: result of the export function
        """
        if use_dynamic_shapes is None:
            use_dynamic_shapes = len(self.inputs) > 1
        args, kwargs = self.inputs[0]
        dynamic_shapes = self.guess_dynamic_shapes() if use_dynamic_shapes else None
        args, kwargs, ds = self._post_process_dynamic_shapes(
            args, kwargs, dynamic_shapes, False, verbose
        )

        for child in self.children:
            if verbose >= 10:
                print(f"[try_export-FX]     child {child.full_name} as custom op")
                print(
                    f"[try_export-FX]           "
                    f"args={string_type(child.inputs[0][0], with_shape=True)}"
                )
                print(
                    f"[try_export-FX]         "
                    f"kwargs={string_type(child.inputs[0][1], with_shape=True)}"
                )
                print(
                    f"[try_export-FX]        "
                    f"outputs={string_type(child.outputs[0], with_shape=True)}"
                )
            child.put_custom_op_inplace(verbose=verbose, shape_functions=shape_functions)
        ep_report = torch.export.export(
            self.model,
            args,
            kwargs=kwargs,
            dynamic_shapes=ds,
            **(exporter_kwargs or {}),
        )
        assert ep_report is not None, f"No result for {self.full_name}"
        # We restore the initial state.
        for child in self.children:
            child.remove_custom_op_inplace(verbose=verbose)
        return ep_report

    def to_onnx_local(
        self,
        target_opset: Optional[Union[int, Dict[str, int]]] = None,
        as_function: bool = False,
        options: Optional[OptimizationOptions] = None,
        optimize: bool = True,
        filename: Optional[str] = None,
        inline: bool = False,
        input_names: Optional[Sequence[str]] = None,
        output_names: Optional[List[str]] = None,
        large_model: bool = False,
        verbose: int = 0,
        return_builder: bool = False,
        raise_list: Optional[Set[str]] = None,
        external_threshold: int = 1024,
        return_optimize_report: bool = False,
        function_options: Optional[FunctionOptions] = None,
        dispatcher: Optional["Dispatcher"] = None,  # noqa: F821
        output_dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
        export_options: Optional[Union[str, ExportOptions]] = None,
        check_conversion_cls: Optional[Union[Dict[str, Any], type]] = None,
    ):
        """
        Exports into ONNX with submodule as local functions.

        :param input_names: input names
        :param target_opset: targeted opset or targeted opsets as a dictionary
        :param as_function: export as a ModelProto or a FunctionProto
        :param options: optimization options
        :param verbose: verbosity level
        :param return_builder: returns the builder as well
        :param raise_list: the builder stops any time a name falls into that list,
            this is a debbuging tool
        :param optimize: optimize the model before exporting into onnx
        :param large_model: if True returns a :class:`onnx.model_container.ModelContainer`,
            it lets the user to decide later if the weights should be part of the model
            or saved as external weights
        :param external_threshold: if large_model is True, every tensor above this limit
            is stored as external
        :param return_optimize_report: returns statistics on the optimization as well
        :param filename: if specified, stores the model into that file
        :param inline: inline the model before converting to onnx, this is done before
                any optimization takes place
        :param export_options: to apply differents options before to get the exported program
        :param function_options: to specify what to do
            with the initializers in local functions,
            add them as constants or inputs
        :param dispatcher: see :class:`experimental_experiment.torch_interpreter.Dispatcher`
        :param output_names: to rename the output names
        :param output_dynamic_shapes: same as *dynamic_shapes* but for the output
        :param export_options: to apply differents options before to get the exported program
        :param check_conversion_cls: a runtime with the same API than
            :class:`onnx.reference.ReferenceEvaluator` than can be used to check that the
            onnx models produce the same outputs,
            it can be also a dictionary to specify atol, rtol to be used after it runs
        :return: onnx model
        """
        # First export the submodule.
        assert hasattr(self, "exporter_status"), (
            f"{self.full_name}: run 'try_export' to get an "
            f"exported program for this instance, existing attributes {dir(self)}"
        )
        assert self.exporter_status.status in (
            StatusExportCode.OK,
            StatusExportCode.OK_CHILDC,
        ), (
            f"{self.full_name}: exporter failed, "
            f"status={self.exporter_status.status!r}, "
            f"a custom onnx converter must be provided for "
            f"'diag_lib::{self.custom_op_name}', "
            f"args={string_type(self.inputs[0][0], with_shape=True)}, "
            f"kwargs={string_type(self.inputs[0][1], with_shape=True)}, outputs="
            f"{string_type(self.outputs[0], with_shape=True)}"
        )
        all_stats = {}
        dispatched = {}
        local_functions = []
        if verbose:
            print(f"[to_onnx_local] {self.dot_name} - to_onnx_local ")

        if self.children:
            stats = []
            for child in self.children:
                if dispatcher and dispatcher.find_function(
                    f"diag_lib::{child.custom_op_name}"
                ):
                    # No need to export.
                    if verbose:
                        print(
                            f"[to_onnx_local] {self.dot_name} - "
                            f"skip child {child.custom_op_name!r}"
                        )
                    continue
                if verbose:
                    print(
                        f"[to_onnx_local] {self.dot_name} - "
                        f"export child {child.custom_op_name!r}"
                    )
                res = child.to_onnx_local(
                    target_opset=target_opset,
                    as_function=False,
                    options=options,
                    optimize=optimize,
                    filename=None,
                    inline=inline,
                    input_names=None,
                    output_names=None,
                    large_model=False,
                    verbose=verbose,
                    return_builder=True,
                    raise_list=raise_list,
                    return_optimize_report=return_optimize_report,
                    function_options=function_options,
                    dispatcher=dispatcher,
                    export_options=export_options,
                    check_conversion_cls=check_conversion_cls,
                )
                if return_optimize_report:
                    _, builder, st = res
                    stats.append(st)
                else:
                    _, builder = res

                local_functions.append(builder)
                dispatched[f"diag_lib::{child.custom_op_name}"] = builder
            all_stats["children"] = stats

        if verbose:
            print(f"[to_onnx_local] {self.dot_name} - export starts {self.custom_op_name}")
        if dispatched:
            new_dispatcher = Dispatcher(dispatched)
            if dispatcher:
                new_dispatcher.merge(dispatcher)
        else:
            new_dispatcher = dispatcher

        res = to_onnx(
            self.exporter_status.exported,
            as_function=False,
            options=options,
            optimize=optimize,
            filename=filename,
            inline=inline,
            input_names=input_names,
            large_model=large_model,
            verbose=max(verbose - 2, 0),
            return_builder=return_builder,
            raise_list=raise_list,
            return_optimize_report=return_optimize_report,
            function_options=function_options,
            dispatcher=new_dispatcher,
            output_names=output_names,
            output_dynamic_shapes=output_dynamic_shapes,
            export_options=export_options,
        )
        if verbose:
            print(f"[to_onnx_local] {self.dot_name} - export done")

        if check_conversion_cls:
            if verbose:
                print(f"[to_onnx_local] {self.dot_name} - run validation")
            onx = res[0] if isinstance(res, tuple) else res
            diff = self.compute_onnx_discrepancies(onx, check_conversion_cls, verbose=verbose)
            if verbose:
                print(f"[to_onnx_local] {self.dot_name} - done")
                for d in diff:
                    print(f"[to_onnx_local] {self.dot_name} - discrepancies: {string_diff(d)}")
            self.onnx_discrepancies = diff

        if return_optimize_report:
            res[-1].update(all_stats)
        return res

    def compute_onnx_discrepancies(
        self,
        onx: Union[onnx.FunctionProto, onnx.ModelProto],
        check_conversion_cls: Union[Dict[str, Any], type],
        verbose: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Computes the discrepancies by using the intermediate inputs and outputs.

        :param onx: proto
        :param check_conversion_cls: class to use to compute the discrepancies,
            it should follow the same API as :class:`onnx.reference.ReferenceEvaluator`,
            it can be also a dictionary to specify atol, rtol to be used after it runs
        :param verbose: verbosity
        :return: discrepancies for each set of inputs
        """
        flattened_inputs = [
            serialize_args(*i, schema=None, args_names=self.forward_ordered_parameter_names)
            for i in self.inputs
        ]
        flattened_outputs = [serialize_args(i, None, schema=None) for i in self.outputs]
        input_names = (
            list(onx.input)
            if isinstance(onx, onnx.FunctionProto)
            else [i.name for i in onx.graph.input]
        )
        cls, atol, rtol = (
            (
                check_conversion_cls["cls"],
                check_conversion_cls["atol"],
                check_conversion_cls["rtol"],
            )
            if isinstance(check_conversion_cls, dict)
            else (check_conversion_cls, None, None)
        )
        if verbose:
            print(
                f"[onnx_run_disc] {self.dot_name} run with "
                f"cls={cls.__name__} on {onx.__class__.__name__}"
            )

        sess = cls(onx)
        diffs = []
        for i in range(len(flattened_inputs)):
            inp, out = flattened_inputs[i], flattened_outputs[i]
            if verbose:
                print(
                    f"[onnx_run_disc] {self.dot_name} run with "
                    f"{string_type(self.inputs[i], with_shape=True)}"
                )
                print(
                    f"[onnx_run_disc] {self.dot_name} flattened into "
                    f"{string_type(inp, with_shape=True, with_min_max=True)}"
                )
                print(
                    f"[onnx_run_disc] {self.dot_name} expecting "
                    f"{string_type(out, with_shape=True, with_min_max=True)}"
                )
            assert not inp[1], (
                f"Flattened inputs {string_type(flattened_inputs[i], with_shape=True)}, "
                f"are wrong, original inputs  "
                f"{string_type(self.inputs[i], with_shape=True)}"
            )
            feeds = dict(
                zip(
                    input_names,
                    [
                        t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t
                        for t in inp[0]
                    ],
                )
            )
            got = sess.run(None, feeds)
            if verbose:
                if len(got) == 1:
                    print(
                        f"[onnx_run_disc] {self.dot_name} computing "
                        f"{string_type(got[0], with_shape=True, with_min_max=True)}"
                    )
                else:
                    print(
                        f"[onnx_run_disc] {self.dot_name} computing "
                        f"{string_type(tuple(got), with_shape=True, with_min_max=True)}"
                    )
            diff = max_diff(out, got)
            if not (
                atol is None or rtol is None or (diff["abs"] <= atol and diff["rel"] <= rtol)
            ):
                if verbose:
                    name = self.full_name.replace(":", "_")
                    name = f"{name}.onnx"
                    ep_txt_name = f"{name}.ep.txt"
                    ep_name = f"{name}.ep"
                    pkl_name = f"{name}.pkl"
                    with open(name, "wb") as f:
                        f.write(onx.SerializeToString())
                    with open(ep_txt_name, "w") as f:
                        f.write(str(self.exporter_status.exported))
                        if hasattr(self.exporter_status.exported, "graph"):
                            f.write("\n----------------------\n")
                            f.write(str(self.exporter_status.exported.graph))
                    torch.export.save(self.exporter_status.exported, ep_name)
                    with open(pkl_name, "wb") as f:
                        pickle.dump(
                            dict(feeds=feeds, inputs=self.inputs[i], outputs=self.outputs[i]),
                            f,
                        )
                    print(
                        f"[onnx_run_disc] {self.dot_name} saving failing model into {name!r}"
                    )
                    print(
                        f"[onnx_run_disc] {self.dot_name} saving "
                        f"exported program into {ep_name!r}"
                    )

                    # exported program can be saved but not necessarily restored without
                    # the custom ops registered so run some verification here.
                    print(f"[onnx_run_disc] {self.dot_name} run_aligned")
                    from experimental_experiment.torch_interpreter.investigate_helper import (
                        run_aligned,
                    )
                    from experimental_experiment.reference import ExtendedReferenceEvaluator

                    diffs = []
                    for di in run_aligned(
                        self.exporter_status.exported,
                        onx,
                        self.inputs[i][0],
                        kwargs=self.inputs[i][1],
                        check_conversion_cls=dict(
                            cls=ExtendedReferenceEvaluator, atol=1e-5, rtol=1e-5
                        ),
                        verbose=verbose,
                    ):
                        diffs.append(di)

                    raise ValueError(
                        f"{self.full_name}: discrepancies detected: "
                        f"{string_diff(diff)}, model saved into {name!r} and {ep_name!r}"
                        f"\n-- intermediate differences --\n{pprint.pformat(diffs)}"
                    )
                raise ValueError(
                    f"{self.full_name}: discrepancies detected: {string_diff(diff)}"
                )
            diffs.append(diff)
            if verbose:
                print(f"[onnx_run_disc] {self.dot_name} diff={string_diff(diff)}")
        if verbose:
            print(f"[onnx_run_disc] {self.dot_name} validation done")
        return diffs


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


def _flatten_module_containes(
    mod: Union[torch.nn.ModuleList, torch.nn.ModuleDict],
) -> Iterator[Tuple[Tuple[Union[str, int], ...], torch.nn.Module]]:
    if isinstance(mod, torch.nn.ModuleList):
        for i, m in enumerate(mod):
            for ind, mn in _flatten_module_containes(m):
                yield (i, *ind), mn
    elif isinstance(mod, torch.nn.ModuleDict):
        for i, m in mod.items():
            for ind, mn in _flatten_module_containes(m):
                yield (i, *ind), mn
    else:
        yield tuple(), mod


def _trace_forward_execution(
    parent: ModelDiagnoseOutput,
    model: Union["module", torch.nn.Module],  # noqa: F821
    name: str = "__main__",
    level: int = 0,
    verbose: int = 0,
    traced_method: Optional[Dict[Union[type[torch.nn.Module], str], str]] = None,
    trace_functions: bool = False,
    black_list_functions: Optional[Set[str]] = None,
):
    if traced_method is None:
        traced_method = {}

    if not isinstance(model, torch.nn.Module):
        # a function is traced.

        method_name = name
        diag = ModelDiagnoseOutput(parent, name, model, level=level, method_name=name)
        if verbose:
            print(
                f"[_trace_forward_execution] -trace-function- "
                f"{diag.dot_name!r} imported in {model.__name__!r}"
            )
        rewritten_method = lambda *args, _diag=diag, verbose=verbose, **kwargs: (  # noqa: E731
            _rewrite_forward(*args, _diag=_diag, verbose=verbose, **kwargs)
        )
        setattr(model, method_name, rewritten_method)

    else:
        # a module is traced

        if model.__class__ in traced_method:
            method_name = traced_method[model.__class__]
        elif model.__class__.__name__ in traced_method:
            method_name = traced_method[model.__class__.__name__]
        else:
            method_name = "forward"

        if method_name is None:
            # This is not traced.
            return None

        diag = ModelDiagnoseOutput(parent, name, model, level=level, method_name=method_name)
        if verbose:
            print(f"[_trace_forward_execution] -trace- {diag.dot_name}.{diag.method_name}")
        rewritten_method = lambda *args, _diag=diag, verbose=verbose, **kwargs: (  # noqa: E731
            _rewrite_forward(*args, _diag=_diag, verbose=verbose, **kwargs)
        )
        setattr(model, method_name, rewritten_method)
        for name, mod in model.named_children():
            if isinstance(mod, (torch.nn.ModuleList, torch.nn.ModuleDict)):
                mod_items = _flatten_module_containes(mod)
                for i, m in mod_items:
                    d = _trace_forward_execution(
                        diag,
                        m,
                        name if i == tuple() else f"{name}[{','.join(map(repr,i))}]",
                        verbose=verbose,
                        level=level + 1,
                        traced_method=traced_method,
                        trace_functions=trace_functions,
                        black_list_functions=black_list_functions,
                    )
                    if d is None:
                        continue
                    diag.add_child(d)
            else:
                d = _trace_forward_execution(
                    diag,
                    mod,
                    name,
                    verbose=verbose,
                    level=level + 1,
                    traced_method=traced_method,
                    trace_functions=trace_functions,
                    black_list_functions=black_list_functions,
                )
                if d is None:
                    continue
                diag.add_child(d)

    if trace_functions:
        if black_list_functions is None:
            # Usual functions to skip.
            black_list_functions = {
                "replace_return_docstrings",
                "add_start_docstrings_to_model_forward",
            }
        lines = inspect.getsource(diag.forward)
        parsed = ast.parse(textwrap.dedent(lines))
        names = set()
        for node in ast.walk(parsed):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name):
                # This is a method.
                continue
            name = node.func.id
            if name in names or name in black_list_functions:
                continue
            mod = model if inspect.ismodule(model) else sys.modules[model.__class__.__module__]
            called_function = getattr(mod, name, None)
            if called_function is None or not inspect.isfunction(called_function):
                continue
            names.add(name)

            if hasattr(mod, name):
                d = _trace_forward_execution(
                    diag,
                    mod,
                    name,
                    verbose=verbose,
                    level=level + 1,
                    traced_method=traced_method,
                    trace_functions=trace_functions,
                    black_list_functions=black_list_functions,
                )
                if d is None:
                    continue
                diag.add_child(d)
            else:
                if verbose:
                    print(
                        f"[_trace_forward_execution] -cannot-trace- function={name!r}, "
                        f"not found in {mod.__name__!r}"
                    )
    return diag


def _untrace_forward_execution(diag: ModelDiagnoseOutput, verbose: int = 0):
    if verbose:
        print(f"[_untrace_forward_execution] {diag.dot_name}")
    # diag.model.forward = diag.forward
    setattr(diag.model, diag.method_name, diag.forward)
    for child in diag.children:
        _untrace_forward_execution(child, verbose=verbose)


@contextlib.contextmanager
def trace_forward_execution(
    model: torch.nn.Module,
    verbose: int = 0,
    traced_method: Optional[Dict[Union[type[torch.nn.Module], str], str]] = None,
    trace_functions: bool = False,
    black_list_functions: Optional[Set[str]] = None,
) -> ModelDiagnoseOutput:
    """
    Replaces all forward to store the inputs and outputs of the module
    and every submodules.
    See :ref:`l-plot-exporter-recipes-custom-phi35` for an example.
    """
    diag = _trace_forward_execution(
        None,
        model,
        verbose=verbose,
        traced_method=traced_method,
        trace_functions=trace_functions,
        black_list_functions=black_list_functions,
    )
    assert (
        diag is not None
    ), f"Model type {type(model)} cannot be traced, see traced_method={traced_method!r}"
    try:
        yield diag
    finally:
        _untrace_forward_execution(diag, verbose=verbose)


def trace_execution_piece_by_piece(
    model: torch.nn.Module,
    inputs: List[Tuple[Tuple[Any, ...], Dict[str, Any]]],
    verbose: int = 0,
    traced_method: Optional[Dict[Union[type[torch.nn.Module], str], str]] = None,
    trace_functions: bool = False,
    black_list_functions: Optional[Set[str]] = None,
) -> ModelDiagnoseOutput:
    """
    Runs a model, traces the intermediate output and infers dynamic shapes
    based on it.

    :param model: model
    :param inputs: list of input sets ``[(args, kwargs), (args, kwargs), ...]``
        with different shapes (at least for the dynamic dimensions)
    :param verbose: verbosity
    :param traced_method: by default the class traced method ``forward`` but another
        one can be traced, if the traced method is empty, then it is not traced at all
    :param trace_functions: traces not only submodules but function called by
        the traced method or function inside this method or function
    :param black_list_functions: if trace_functions is true, this option
        can be used to avoid tracing some functions
    :return: see :class:`ModelDiagnoseOutput`

    See :ref:`l-plot-exporter-recipes-custom-phi35` for an example.
    """
    if traced_method is None:
        traced_method = {}
    with trace_forward_execution(
        model,
        verbose=verbose,
        traced_method=traced_method,
        trace_functions=trace_functions,
        black_list_functions=black_list_functions,
    ) as tracer:
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
            if model.__class__ in traced_method:
                method = getattr(model, traced_method[model.__class__])
                method(*args, **kwargs)
            else:
                model(*args, **kwargs)
        if verbose:
            print(
                f"[trace_forward_execution] traced execution of model "
                f"{model.__class__.__name__}"
            )
            print(tracer.pretty_text())
        return tracer
