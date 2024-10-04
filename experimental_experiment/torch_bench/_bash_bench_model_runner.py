import contextlib
import collections
import inspect
import os
import pprint
import shutil
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import onnx
import torch
from .export_model_helper import compute_weight_size


@contextlib.contextmanager
def bypass_export_some_errors():
    """
    Tries to bypass some functions torch.export.export does not
    support such as ``torch.jit.isinstance``.
    """
    import torch.jit

    f = torch.jit.isinstance
    torch.jit.isinstance = isinstance

    try:
        yield
    finally:
        torch.jit.isinstance = f


class MakeConfig:
    """Creates a dictionary where keys are attributes."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _rand_int_tensor(
    device: str, low: int, high: int, shape: Tuple[int, ...]
) -> torch.Tensor:
    """Creates a random input integer tensor.

    :param device: device
    :param low: lower value
    :param high: high value
    :param shape: shape
    :return: tensor
    """
    return torch.randint(
        low,
        high,
        shape,
        device=device,
        dtype=torch.int64,
        requires_grad=False,
    )


def download_retry_decorator(retry: int = 5) -> Callable:  # type: ignore[arg-type]
    """
    Decorator function for applying retry logic to a download function.

    The wrapped function will be called up to 5 times
    and raises an exception if the function fails each time.
    After each unsuccessful attempt, there is a delay before
    the next attempt, which is increased linearly with the number of tries.

    :param retry: number of times to retry

    Usage:

    ::

        @download_retry_decorator(retry=5)
        def download_function(model_name: str):
            # download logic goes here
            # ...
    """

    def decorator(download_fn):
        def wrapper(*args, **kwargs) -> Any:
            tries = 0
            total_allowed_tries = retry
            while tries <= total_allowed_tries:
                try:
                    model = download_fn(*args, **kwargs)
                    return model
                except RuntimeError as e:
                    if "Unknown model" in str(e):
                        raise
                    tries += 1
                    if tries <= total_allowed_tries:
                        wait = tries * 30
                        time.sleep(wait)
                    else:
                        raise RuntimeError(  # noqa: B904
                            f"Failed to load model {args!r} "
                            f"with following error(s): {e!r}."
                        )

        return wrapper

    return decorator


def get_dynamo_stats() -> Dict[str, float]:
    """Returns statistics on memory as a dictionary."""
    return collections.Counter(
        {
            "calls_captured": torch._dynamo.utils.counters["stats"]["calls_captured"],
            "unique_graphs": torch._dynamo.utils.counters["stats"]["unique_graphs"],
            "graph_breaks": sum(torch._dynamo.utils.counters["graph_break"].values()),
            # NB: The plus removes zero counts
            "unique_graph_breaks": len(+torch._dynamo.utils.counters["graph_break"]),
            "autograd_captures": torch._dynamo.utils.counters["compiled_autograd"][
                "captures"
            ],
            "autograd_compiles": torch._dynamo.utils.counters["compiled_autograd"][
                "compiles"
            ],
            "cudagraph_skips": torch._dynamo.utils.counters["inductor"]["cudagraph_skips"],
        }
    )


def get_peak_memory():
    """Retuns the memory peak."""
    return torch.cuda.max_memory_allocated() / 10**9


class WrappedModelBase(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def parameters(self):
        yield from self.model.parameters()


class WrappedModelToTuple(WrappedModelBase):
    def __init__(self, model):
        super().__init__(model)

    def __call__(self, *args, **kwargs):
        res = self.model(*args, **kwargs)
        return res.to_tuple()

    def forward(self, *args, **kwargs):
        res = self.model.forward(*args, **kwargs)
        return res.to_tuple()


class ModelRunner:
    """
    Wrappers around a model.
    Makes it easier to load, run inference.

    :param model: torch model
    :param inputs: example of inputs
    :param device: device
    :param dtype: if the model needs to be converted
    :param warmup: number of iteration to warmup the model
    :param repeat: number of iteration to repeat the model
    :param suite: model suite
    :param wrap_kind: to wrap the model and tuple as much as possible,
        None is default behavior,
        'nowrap' to explicit avoid wrapping
    :param nvtx: enable nvtx events
    """

    _patched = None

    @classmethod
    def allowed_configuration(
        cls, exporter: str, optimization: Optional[str] = None
    ) -> bool:
        """Defines the allowed configurations."""
        if not optimization or optimization == "none":
            # always possible
            return True
        if exporter in {"custom"}:
            return True
        if exporter in {"torch_script", "dynamo_export"}:
            return optimization in {"default"}
        if exporter in {"onnx_dynamo", "onnx_dynamo-fallback", "onnx_dynamo-detailed"}:
            return optimization in {"default", "ir"}
        return False

    @classmethod
    def isinstance_namedtuple(cls, x):
        return isinstance(x, tuple) and getattr(x, "_fields", None) is not None

    @classmethod
    def _to_type_or_device(cls, o, dtype_or_device):
        if dtype_or_device is None or o is None or isinstance(o, (str, bool, int, float)):
            return o
        if isinstance(o, list):
            return [cls._to_type_or_device(v, dtype_or_device) for v in o]
        if isinstance(o, tuple):
            return tuple(cls._to_type_or_device(v, dtype_or_device) for v in o)
        if hasattr(o, "dtype"):
            if isinstance(dtype_or_device, str) or o.dtype in {
                torch.float32,
                torch.float64,
                torch.float16,
                torch.bfloat16,
            }:
                return o.to(dtype_or_device)
            return o

        if cls.isinstance_namedtuple(o):
            new_vals = {}
            for k in o._fields:
                new_vals[k] = cls._to_type_or_device(getattr(o, k), dtype_or_device)
            return o.__class__(**new_vals)

        if o.__class__.__name__.endswith("KeyedJaggedTensor"):
            ext = dict(
                weights=o.weights_or_none(),
                values=o.values(),
                offsets=o.offsets_or_none(),
                keys=o.keys(),
                # index_per_key=o.index_per_key(),
                length_per_key=o.length_per_key_or_none(),
                lengths=o.lengths_or_none(),
                # stride_per_key=o.stride_per_key(),
                stride_per_key_per_rank=o.stride_per_key_per_rank(),
                # variable_stride_per_key=o.variable_stride_per_key(),
                offset_per_key=o.offset_per_key_or_none(),
                inverse_indices=o.inverse_indices_or_none(),
            )
            ext = {k: cls._to_type_or_device(v, dtype_or_device) for k, v in ext.items()}
            return o.__class__(**ext)

        if isinstance(o, dict):
            res = {}
            for k, v in o.items():
                res[k] = cls._to_type_or_device(v, dtype_or_device)
            return res
        try:
            return o.to(dtype_or_device)
        except (AttributeError, AssertionError) as e:
            raise AssertionError(
                f"Unable to convert class {type(o)} to {dtype_or_device} "
                f"(namedtuple={cls.isinstance_namedtuple(o)}), o={o})"
            ) from e

    def __init__(
        self,
        model: Any,
        inputs: Any,
        device: str,
        dtype: torch.dtype,
        warmup: int,
        repeat: int,
        suite: str,
        autocast: bool = False,
        wrap_kind: Optional[None] = None,
        nvtx: bool = False,
    ):
        if dtype is None:
            cvt = lambda o: self._to_type_or_device(o, device)  # noqa: E731
        else:
            cvt = lambda o: self._to_type_or_device(  # noqa: E731
                self._to_type_or_device(o, dtype), device
            )

        if isinstance(inputs, dict):
            inputs = {k: cvt(v) for k, v in inputs.items()}
        elif isinstance(inputs, (list, tuple)):
            inputs = tuple(cvt(v) for v in inputs)
        else:
            raise AssertionError - (f"Input type is {type(inputs)}")

        if isinstance(inputs, dict):
            # torch.export.export does not allow that.
            sig = inspect.signature(model.forward)
            added = 0
            new_inputs = []
            new_names = []
            use_default = []
            for n in sig.parameters:
                if n in inputs:
                    new_inputs.append(inputs[n])
                    added += 1
                    use_default.append(False)
                else:
                    if sig.parameters[n].default is inspect._empty:
                        # probably one optional input
                        continue
                    new_inputs.append(sig.parameters[n].default)
                    use_default.append(True)
                new_names.append(n)
            assert added == len(inputs), (
                f"Unexpected input name in {sorted(inputs)} and "
                f"parameters={list(sig.parameters)}"
            )
            inputs = tuple(new_inputs)
            self.raw_input_names = new_names
            self.raw_use_defaults = use_default
        else:
            self.raw_input_names = [f"input{i}" for i in range(len(inputs))]
            self.raw_use_defaults = [i is None for i in inputs]

        config = getattr(model, "config", {})
        to_tuple = not (hasattr(config, "to_tuple") and not config.to_tuple)
        assert (
            "AlexNet" not in model.__class__.__name__
            and "Mixer" not in model.__class__.__name__
        ) or not to_tuple, (
            f"Model {type(model)} does not need to call "
            f"to_tuple, has config={hasattr(model, 'config')}."
        )

        model_cvt = cvt(model)
        del model
        if wrap_kind == "nowrap":
            self.model = model_cvt
        else:
            assert wrap_kind is None, f"Not implemented for wrap_kind={wrap_kind!r}"
            if to_tuple:
                self.model = WrappedModelToTuple(model_cvt)
            else:
                self.model = WrappedModelBase(model_cvt)

        assert (
            not isinstance(inputs, tuple)
            or not isinstance(inputs[0], torch.Tensor)
            or "cuda" not in device
            or inputs[0].get_device() >= 0
        ), (
            f"device={device!r} but input device is {inputs[0].get_device()} "
            f"(check {cvt(inputs[0]).get_device()})"
        )
        self.device = device
        self.dtype = dtype
        self.inputs = inputs
        self.repeat = repeat
        self.warmup = warmup
        self.suite = suite
        self.autocast = autocast
        self.nvtx = nvtx
        assert self.autocast is not None
        self.std_to_dump = []

    @property
    def input_names(self):
        return self.raw_input_names

    def get_dynamic_shapes(
        self,
        dynamic: bool = False,
        wrapped: bool = False,
        input_names: Optional[List[str]] = None,
    ) -> Optional[Union[Dict[str, Any], Tuple[Any], List[Any]]]:
        """
        Returns dynamic shapes specifying the first dimension as dynamic.

        :param dynamic: make it dynamic or not
        :param wrapped: the model is wrapped into a class defining forward method
            as ``forward(self, *args, **kargs)``
        :param input_names: to overwrite the input names,
            (not used)
        """
        if not dynamic:
            return None
        assert (
            input_names is None
        ), f"This method is not implemented if input_names={input_names!r}"
        assert input_names is None or len(input_names) == len(self.inputs), (
            f"Unexpected number of input_names {len(input_names)} != "
            f"{len(self.inputs)} (expected), input_names={input_names!r}"
        )
        dim = torch.export.Dim("batch", min=1, max=1024)
        res = []
        for x in self.inputs:
            if x is None:
                res.append(None)
                continue
            res.append({0: dim} if len(x.shape) > 1 else None)

        final = tuple(res)
        if wrapped:
            return (final,)
        return final

    def dump_std(self, filename: str):
        """Dumps some information in the given filename."""
        if self.std_to_dump:
            with open(filename, "w", encoding="utf-8") as f:
                f.write("\n".join(map(str, self.std_to_dump)))

    def run(self) -> Any:
        if self.autocast:
            if self.nvtx:
                torch.cuda.nvtx.range_push("ModelRunner.Eager.AutoCast")
            with torch.autocast(device_type=self.device, dtype=self.dtype):
                res = self.model(*self.inputs)
            if self.nvtx:
                torch.cuda.nvtx.range_pop()
            return res
        if self.nvtx:
            torch.cuda.nvtx.range_push("ModelRunner.Eager")
        res = self.model(*self.inputs)
        if self.nvtx:
            torch.cuda.nvtx.range_pop()
        return res

    def run_dynamic(self, wrapped: bool = False) -> Any:
        dynamic_inputs = self.make_dynamic_inputs(wrapped=wrapped)
        if self.autocast:
            if self.nvtx:
                torch.cuda.nvtx.range_push("ModelRunner.Eager.AutoCast.dynamic")
            with torch.autocast(device_type=self.device, dtype=self.dtype):
                res = self.model(*dynamic_inputs)
            if self.nvtx:
                torch.cuda.nvtx.range_pop()
            return res
        if self.nvtx:
            torch.cuda.nvtx.range_push("ModelRunner.Eager.Dynamic")
        res = self.model(*dynamic_inputs)
        if self.nvtx:
            torch.cuda.nvtx.range_pop()
        return res

    def compute_weight_size(self) -> int:
        """Returns the weight size."""
        return compute_weight_size(self.model)

    def parameters_dtype(self) -> str:
        """Returns the unique dtypes of all parameters."""
        return ",".join(
            sorted({str(p.dtype).replace("torch.", "") for p in self.model.parameters()})
        )

    def export_as(
        self,
        exporter: str,
        name: str,
        dynamic: bool,
        fake_tensor: bool,
        no_grad: bool,
        optimization: str,
        verbose: int,
        target_opset: int,
        decomposition_table: Optional[str],
    ) -> Tuple[onnx.ModelProto, Optional[Dict[str, Any]]]:
        """
        Converts a model into onnx.

        :param exporter: exporter
        :param name: filename
        :param dynamic: use dynamic shape
        :param fake_tensor: use fake_tensor
        :param no_grad: use no_grad
        :param optimization: defines the optimizations
        :param verbose: verbosity
        :param target_opset: target opset
        :param decomposition_table: decomposition table to apply
        :return: the model proto with or without weights, statistics
        """
        assert not fake_tensor, "fake_tensor not implemented."

        if name == "1001Fail":
            raise RuntimeError(f"Model {name!r} is meant to fail for unit test purpose.")

        if exporter == "custom":
            return self._to_onnx_custom(
                name,
                dynamic=dynamic,
                fake_tensor=fake_tensor,
                no_grad=no_grad,
                optimization=optimization,
                verbose=verbose,
                target_opset=target_opset,
                decomposition_table=decomposition_table,
            )

        if exporter in ("cort", "cortgrad"):
            return self._to_cort(
                name,
                dynamic=dynamic,
                fake_tensor=fake_tensor,
                no_grad=no_grad,
                optimization=optimization,
                verbose=verbose,
                target_opset=target_opset,
                autograd=exporter == "cortgrad",
                decomposition_table=decomposition_table,
            )

        if exporter == "torch_script":
            return self._to_onnx_script(
                name,
                dynamic=dynamic,
                fake_tensor=fake_tensor,
                no_grad=no_grad,
                optimization=optimization,
                verbose=verbose,
                target_opset=target_opset,
            )

        if exporter == "onnx_dynamo":
            return self._to_onnx_dynamo(
                name,
                dynamic=dynamic,
                fake_tensor=fake_tensor,
                no_grad=no_grad,
                optimization=optimization,
                verbose=verbose,
                target_opset=target_opset,
            )
        if exporter == "onnx_dynamo-detailed":
            return self._to_onnx_dynamo(
                name,
                dynamic=dynamic,
                fake_tensor=fake_tensor,
                no_grad=no_grad,
                optimization=optimization,
                verbose=verbose,
                target_opset=target_opset,
                detailed=True,
            )
        if exporter == "onnx_dynamo-fallback":
            return self._to_onnx_dynamo(
                name,
                dynamic=dynamic,
                fake_tensor=fake_tensor,
                no_grad=no_grad,
                optimization=optimization,
                verbose=verbose,
                target_opset=target_opset,
                fallback=True,
            )
        if exporter == "dynamo_export":
            return self._to_onnx_dynamo2(
                name,
                dynamic=dynamic,
                fake_tensor=fake_tensor,
                no_grad=no_grad,
                optimization=optimization,
                verbose=verbose,
                target_opset=target_opset,
            )
        if exporter == "eager":
            return self._to_eager(
                name,
                dynamic=dynamic,
                fake_tensor=fake_tensor,
                no_grad=no_grad,
                optimization=optimization,
                verbose=verbose,
                target_opset=target_opset,
            )
        if exporter == "compile":
            return self._to_compile(
                name,
                dynamic=dynamic,
                fake_tensor=fake_tensor,
                no_grad=no_grad,
                optimization=optimization,
                verbose=verbose,
                target_opset=target_opset,
            )
        if exporter == "export":
            return self._to_export(
                name,
                dynamic=dynamic,
                fake_tensor=fake_tensor,
                no_grad=no_grad,
                optimization=optimization,
                verbose=verbose,
                target_opset=target_opset,
            )
        if exporter == "inductor":
            return self._to_inductor(
                name,
                dynamic=dynamic,
                fake_tensor=fake_tensor,
                no_grad=no_grad,
                optimization=optimization,
                verbose=verbose,
                target_opset=target_opset,
                fullgraph=True,
            )
        if exporter == "inductor-partial":
            return self._to_inductor(
                name,
                dynamic=dynamic,
                fake_tensor=fake_tensor,
                no_grad=no_grad,
                optimization=optimization,
                verbose=verbose,
                target_opset=target_opset,
                fullgraph=False,
            )
        if exporter == "dort":
            return self._to_dort(
                name,
                dynamic=dynamic,
                fake_tensor=fake_tensor,
                no_grad=no_grad,
                optimization=optimization,
                verbose=verbose,
                target_opset=target_opset,
            )
        if exporter == "ORTModule":
            return self._to_ortmodule(
                name,
                dynamic=dynamic,
                fake_tensor=fake_tensor,
                no_grad=no_grad,
                optimization=optimization,
                verbose=verbose,
                target_opset=target_opset,
            )
        raise AssertionError(f"Exporter {exporter!r} is not implemented.")

    def _to_onnx_custom(
        self,
        name: str,
        dynamic: bool,
        fake_tensor: bool,
        no_grad: bool,
        optimization: str,
        verbose: int,
        target_opset: int,
        decomposition_table: Optional[str],
    ):
        assert not fake_tensor, "fake_tensor not implemented."
        assert no_grad, "no_grad false not implemented yet"
        from ..torch_interpreter import to_onnx
        from ..xbuilder import OptimizationOptions

        if optimization and optimization != "none":
            # cuda = any(m.is_cuda for m in self.model.parameters())
            options = OptimizationOptions(
                constant_folding=True,
                patterns=optimization,
                verbose=10 if verbose >= 100 else (1 if verbose > 1 else 0),
                processor="CUDA" if self.device.startswith("cuda") else "CPU",
            )
        else:
            options = None

        if self.autocast:
            with torch.autocast(device_type=self.device, dtype=self.dtype), torch.no_grad():
                onx, builder, stats = to_onnx(
                    self.model,
                    self.make_export_inputs(dynamic, wrapped=True),
                    optimize=bool(optimization),
                    large_model=True,
                    verbose=10 if verbose >= 100 else (1 if verbose > 1 else 0),
                    target_opset=target_opset,
                    return_optimize_report=True,
                    options=options,
                    return_builder=True,
                    dynamic_shapes=self.get_dynamic_shapes(dynamic, wrapped=True),
                    decomposition_table=decomposition_table,
                )
        else:
            with torch.no_grad():
                onx, builder, stats = to_onnx(
                    self.model,
                    self.make_export_inputs(dynamic, wrapped=True),
                    optimize=bool(optimization),
                    large_model=True,
                    verbose=10 if verbose >= 100 else (1 if verbose > 1 else 0),
                    target_opset=target_opset,
                    return_optimize_report=True,
                    options=options,
                    return_builder=True,
                    dynamic_shapes=self.get_dynamic_shapes(dynamic, wrapped=True),
                    decomposition_table=decomposition_table,
                )
        begin = time.perf_counter()
        self.std_to_dump.append(pprint.pformat(stats))
        self.std_to_dump.append("----------------------------")
        self.std_to_dump.append(builder.get_debug_msg())
        stats["time_export_debuginfo"] = time.perf_counter() - begin
        begin = time.perf_counter()
        onx.save(name, all_tensors_to_one_file=True)
        stats["time_export_save"] = time.perf_counter() - begin
        for k, v in onx._stats.items():
            if v > 0:
                stats[k] = v
        return onx.model_proto, stats

    def _to_cort(
        self,
        name: str,
        dynamic: bool,
        fake_tensor: bool,
        no_grad: bool,
        optimization: str,
        verbose: int,
        target_opset: int,
        autograd: bool = False,
        decomposition_table: Optional[str] = None,
    ):
        assert not fake_tensor, "fake_tensor not implemented."
        assert not dynamic, "dynamic true not implemented yet"
        assert no_grad, "no_grad true not implemented yet"
        from ..xbuilder import OptimizationOptions
        from ..torch_dynamo import onnx_custom_backend, get_decomposition_table

        if optimization:
            # cuda = any(m.is_cuda for m in self.model.parameters())
            options = OptimizationOptions(
                constant_folding=True,
                patterns=optimization,
                verbose=10 if verbose >= 100 else (1 if verbose > 1 else 0),
                processor="CUDA" if self.device.startswith("cuda") else "CPU",
            )
        else:
            options = None

        cbff = lambda *args, **kwargs: onnx_custom_backend(  # noqa: E731
            *args,
            target_opset=target_opset,
            verbose=verbose,
            options=options,
            optimize=bool(optimization),
            decomposition_table=decomposition_table,
            **kwargs,
        )

        if autograd:
            from torch._dynamo.backends.common import aot_autograd

            assert decomposition_table is None or decomposition_table in (
                "none",
                "default",
            ), (
                f"No other option than 'default' for decomposition_table="
                f"{decomposition_table!r} is supported"
            )

            cbf = aot_autograd(fw_compiler=cbff, decompositions=get_decomposition_table())

            if self.autocast:
                with torch.autocast(device_type=self.device, dtype=self.dtype):
                    res = torch.compile(self.model, backend=cbf, fullgraph=True)
            else:
                res = torch.compile(self.model, backend=cbf, fullgraph=True)
        else:
            if self.autocast:
                with torch.autocast(
                    device_type=self.device, dtype=self.dtype
                ), torch.no_grad():
                    res = torch.compile(self.model, backend=cbff, fullgraph=True)
            else:
                with torch.no_grad():
                    res = torch.compile(self.model, backend=cbff, fullgraph=True)
        return res, None

    def _optimize_rewrite(
        self, name: str, optimization: str
    ) -> Tuple[onnx.ModelProto, Dict[str, Any]]:
        stats = {}
        begin = time.perf_counter()
        shutil.copy(name, name + ".raw.onnx")
        model_proto = onnx.load(name, load_external_data=True)
        rule_sets = []

        opts = optimization.split("+")
        for opt in opts:
            if opt in ("", "-", "none"):
                continue
            if opt == "default":
                # from onnx.inliner import inline_local_functions
                from onnxscript.optimizer import optimize
                from onnxscript.rewriter import rewrite

                first_model_proto = model_proto
                model_proto = optimize(
                    model_proto,
                    num_iterations=2,
                    onnx_shape_inference=False,
                )
                model_proto = rewrite(model_proto)
                # On MegatronBertForQuestionAnswering, this step hurts the latency by 10%.
                # model_proto = inline_local_functions(model_proto)
                del first_model_proto.graph.node[:]
                del first_model_proto.functions[:]
                first_model_proto.graph.node.extend(model_proto.graph.node)
                first_model_proto.functions.extend(model_proto.functions)
                continue

            if opt == "llm":
                from onnxscript.rewriter.llama_rule_sets import llama_p0_rule_set

                rule_sets.append(llama_p0_rule_set)
                continue

            raise AssertionError(f"Unexpected value for optimization={optimization!r}.")

        if rule_sets:
            from onnxscript import ir

            begin_pat = time.perf_counter()
            ir_model = ir.serde.deserialize_model(model_proto)
            for rs in rule_sets:
                rs().apply_to_model(ir_model)
            model_proto = ir.serde.serialize_model(ir_model)
            stats["time_export_optimization_pattern"] = time.perf_counter() - begin_pat

        onnx.save(model_proto, name, save_as_external_data=True)
        model_proto = onnx.load(name, load_external_data=False)
        stats["time_export_optimization"] = time.perf_counter() - begin
        return model_proto, stats

    def _to_onnx_script(
        self,
        name: str,
        dynamic: bool,
        fake_tensor: bool,
        no_grad: bool,
        optimization: str,
        verbose: int,
        target_opset: int,
    ):
        assert not fake_tensor, "fake_tensor not implemented."
        assert no_grad, "no_grad false not implemented yet"

        if (
            isinstance(self.inputs, tuple)
            and len(self.inputs) == 1
            and isinstance(self.inputs[0], list)
            and len(self.inputs[0]) == 1
            and isinstance(self.inputs[0][0], dict)
            and len(set(self.inputs[0][0]) & {"file_name", "image"}) == 2
        ):
            # detectron2 take inputs such as
            # ([{'file_name': ..., 'height': ..., 'image': torch.Tensor(...)}])
            inputs = (self.inputs[0][0]["image"],)
        else:
            inputs = self.inputs

        dynamic_shapes_for_export = self.get_dynamic_shapes(dynamic, wrapped=True)
        inputs = self.make_export_inputs(dynamic, wrapped=True, inputs=inputs)
        kwargs_export = {}
        if dynamic_shapes_for_export is not None:
            # torch_script only supports a dictionary
            assert isinstance(
                dynamic_shapes_for_export, tuple
            ), f"dynamic_axes not supported when it is {dynamic_shapes_for_export}"
            input_names = []
            dynamic_axes = {}
            for i, dyn in enumerate(dynamic_shapes_for_export):
                if dyn is None:
                    continue
                assert isinstance(dyn, tuple), (
                    f"Model is wrapped, unexpected type {type(dyn)} for input {i}, "
                    f"dynamic_shapes_for_export={dynamic_shapes_for_export}"
                )
                for inp in dyn:
                    if inp is None:
                        continue
                    dname = f"input{len(input_names)}"
                    assert isinstance(inp, dict), (
                        f"Unexpected for input {dname!r}, {type(inp)}, i={i}, dyn={dyn}, "
                        f"dynamic_shapes_for_export={dynamic_shapes_for_export}"
                    )
                    daxes = {}
                    for k, v in inp.items():
                        daxes[k] = v.__name__
                    dynamic_axes[dname] = daxes
                    input_names.append(dname)

            if verbose:
                print(f"[ModelRunner._to_onnx_script] dynamic_axes={dynamic_axes}")
                print(f"[ModelRunner._to_onnx_script] input_names={input_names}")
                print(f"[ModelRunner._to_onnx_script] n_inputs={len(inputs)}")
            kwargs_export["dynamic_axes"] = dynamic_axes
            kwargs_export["input_names"] = input_names

        if self.autocast:
            with torch.autocast(device_type=self.device, dtype=self.dtype), torch.no_grad():
                torch.onnx.export(
                    self.model,
                    inputs,
                    name,
                    do_constant_folding=False,
                    opset_version=target_opset,
                    verbose=max(verbose - 1, 0),
                    **kwargs_export,
                )
        else:
            with torch.no_grad():
                torch.onnx.export(
                    self.model,
                    inputs,
                    name,
                    do_constant_folding=False,
                    opset_version=target_opset,
                    verbose=max(verbose - 1, 0),
                    **kwargs_export,
                )

        if verbose:
            print(f"[ModelRunner._to_onnx_script] done saved into {name}")

        if optimization and optimization != "none":
            return self._optimize_rewrite(name, optimization)
        return onnx.load(name, load_external_data=False), None

    def _to_onnx_dynamo(
        self,
        name: str,
        dynamic: bool,
        fake_tensor: bool,
        no_grad: bool,
        optimization: str,
        verbose: int,
        target_opset: int,
        detailed: bool = False,
        fallback: bool = False,
    ):
        assert not fake_tensor, "fake_tensor not implemented."
        assert no_grad, "no_grad false not implemented yet"

        if optimization:
            opts = optimization.split("+")
            if "ir" in opts:
                os.environ["TORCH_ONNX_ENABLE_OPTIMIZATION"] = "1"
                opts.pop(opts.index("ir"))
                optimization = "+".join(opts)
            else:
                os.environ["TORCH_ONNX_ENABLE_OPTIMIZATION"] = "0"
        else:
            os.environ["TORCH_ONNX_ENABLE_OPTIMIZATION"] = "0"

        additional_kwargs = {}
        if detailed:
            additional_kwargs.update(
                dict(
                    profile=True,
                    report=True,
                    verify=True,
                    dump_exported_program=True,
                    artifacts_dir=os.path.dirname(name),
                )
            )
        if fallback:
            additional_kwargs.update(dict(fallback=True))

        if self.autocast:
            with torch.autocast(
                device_type=self.device, dtype=self.dtype
            ), torch.no_grad(), bypass_export_some_errors():
                onnx_program = torch.onnx.export(
                    self.model,
                    self.make_export_inputs(dynamic, wrapped=True),
                    name,
                    opset_version=target_opset,
                    dynamo=True,
                    external_data=True,
                    dynamic_shapes=self.get_dynamic_shapes(dynamic, wrapped=True),
                    **additional_kwargs,
                )
        else:
            with torch.no_grad(), bypass_export_some_errors():
                onnx_program = torch.onnx.export(
                    self.model,
                    self.make_export_inputs(dynamic, wrapped=True),
                    name,
                    opset_version=target_opset,
                    dynamo=True,
                    external_data=True,
                    dynamic_shapes=self.get_dynamic_shapes(dynamic, wrapped=True),
                    **additional_kwargs,
                )

        if optimization:
            return self._optimize_rewrite(name, optimization)
        return onnx_program.model_proto, None

    def _to_onnx_dynamo2(
        self,
        name: str,
        dynamic: bool,
        fake_tensor: bool,
        no_grad: bool,
        optimization: str,
        verbose: int,
        target_opset: int,
    ):
        assert not fake_tensor, "fake_tensor not implemented."
        assert no_grad, "no_grad false not implemented yet"
        stats = {}

        if self.autocast:
            with torch.autocast(device_type=self.device, dtype=self.dtype), torch.no_grad():
                exported = torch.onnx.dynamo_export(
                    self.model,
                    *self.inputs,
                    export_options=torch.onnx.ExportOptions(
                        dynamic_shapes=dynamic,
                        # registry=torch.onnx.OnnxRegistry()
                    ),
                )
        else:
            with torch.no_grad():
                exported = torch.onnx.dynamo_export(
                    self.model,
                    *self.inputs,
                    export_options=torch.onnx.ExportOptions(
                        dynamic_shapes=dynamic,
                        # registry=torch.onnx.OnnxRegistry()
                    ),
                )

        begin = time.perf_counter()
        exported.save(name)
        stats["time_export_save"] = time.perf_counter() - begin

        onx = onnx.load(name, load_external_data=True)
        onnx.save(onx, name, save_as_external_data=True)

        if optimization:
            return self._optimize_rewrite(name, optimization)
        return onnx.load(name, load_external_data=False), stats

    def _to_export(
        self,
        name: str,
        dynamic: bool,
        fake_tensor: bool,
        no_grad: bool,
        optimization: str,
        verbose: int,
        target_opset: int,
    ):
        assert not fake_tensor, "fake_tensor not implemented."
        assert no_grad, "no_grad false not implemented yet"
        assert (
            not optimization or optimization == "none"
        ), f"optimization {optimization!r} not compatible with export"
        from torch.export import export

        with torch.no_grad():
            res = export(
                self.model,
                self.make_export_inputs(dynamic, wrapped=True),
                dynamic_shapes=self.get_dynamic_shapes(dynamic, wrapped=True),
            )
        return res.module(), None

    def _to_eager(
        self,
        name: str,
        dynamic: bool,
        fake_tensor: bool,
        no_grad: bool,
        optimization: str,
        verbose: int,
        target_opset: int,
    ):
        assert not fake_tensor, "fake_tensor not implemented."
        assert no_grad, "no_grad false not implemented yet"
        assert (
            not optimization or optimization == "none"
        ), f"optimization {optimization!r} not compatible with eager"

        return self.model, None

    def _to_ortmodule(
        self,
        name: str,
        dynamic: bool,
        fake_tensor: bool,
        no_grad: bool,
        optimization: str,
        verbose: int,
        target_opset: int,
    ):
        assert not fake_tensor, "fake_tensor not implemented."
        assert not dynamic, "dynamic true not implemented yet"
        assert no_grad, "no_grad false not implemented yet"
        assert (
            not optimization or optimization == "none"
        ), f"optimization {optimization!r} not compatible with eager"
        from onnxruntime.training.ortmodule import ORTModule

        return ORTModule(self.model), None

    def _to_compile(
        self,
        name: str,
        dynamic: bool,
        fake_tensor: bool,
        no_grad: bool,
        optimization: str,
        verbose: int,
        target_opset: int,
    ):
        assert not fake_tensor, "fake_tensor not implemented."
        assert no_grad, "no_grad true not implemented yet"
        assert (
            not optimization or optimization == "none"
        ), f"optimization {optimization!r} not compatible with compile"

        def custom_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
            if verbose:
                print("[compile: fx_graph]")
                print(gm)
                self.std_to_dump.append(str(gm))

            return gm.forward

        if self.autocast:
            with torch.autocast(device_type=self.device, dtype=self.dtype), torch.no_grad():
                res = torch.compile(
                    self.model,
                    fullgraph=True,
                    backend=custom_backend,
                )
        else:
            with torch.no_grad():
                res = torch.compile(self.model, fullgraph=True, backend=custom_backend)
        return res, None

    def _to_inductor(
        self,
        name: str,
        dynamic: bool,
        fake_tensor: bool,
        no_grad: bool,
        optimization: str,
        verbose: int,
        target_opset: int,
        fullgraph: bool,
    ):
        assert not fake_tensor, "fake_tensor not implemented."
        assert no_grad, "no_grad true not implemented yet"
        assert (
            not optimization or optimization == "none"
        ), f"optimization {optimization!r} not compatible with inductor"

        if self.autocast:
            with torch.autocast(device_type=self.device, dtype=self.dtype), torch.no_grad():
                res = torch.compile(self.model, backend="inductor", fullgraph=fullgraph)
        else:
            with torch.no_grad():
                res = torch.compile(self.model, backend="inductor", fullgraph=fullgraph)
        return res, None

    def _to_dort(
        self,
        name: str,
        dynamic: bool,
        fake_tensor: bool,
        no_grad: bool,
        optimization: str,
        verbose: int,
        target_opset: int,
    ):
        assert not fake_tensor, "fake_tensor not implemented."
        assert not dynamic, "dynamic true not implemented yet"
        assert no_grad, "no_grad true not implemented yet"
        assert (
            not optimization or optimization == "none"
        ), f"optimization {optimization!r} not compatible with dort"

        if self.autocast:
            with torch.autocast(device_type=self.device, dtype=self.dtype), torch.no_grad():
                res = torch.compile(
                    self.model, backend="onnxrt", fullgraph=True, dynamic=dynamic
                )
        else:
            with torch.no_grad():
                res = torch.compile(
                    self.model, backend="onnxrt", fullgraph=True, dynamic=dynamic
                )
        return res, None

    def make_export_inputs(
        self, dynamic: bool = False, wrapped: bool = False, inputs: Optional[Any] = None
    ) -> Any:
        """
        Creates the new inputs for the benchmarks.
        :func:`torch.export.export` fails when a dimension is dynamic
        and the value for this dimension is 1. This function
        expands the input on that dimension to make it 2
        if it is 1. These inputs should only be used at export time.

        :param dynamic: dynamic, yes or no?
        :param wrapped: wrapped model
        :param inputs: existing inputs or None to use `self.inputs`
        :return: new inputs
        """
        if not dynamic:
            # easy case
            return self.inputs if inputs is None else inputs

        if inputs is None:
            inputs = self.inputs
        assert isinstance(
            inputs, tuple
        ), f"Not implemented for type(self.inputs)={type(inputs)}"
        dynamic_shapes = self.get_dynamic_shapes(dynamic, wrapped=wrapped)
        if wrapped:
            assert isinstance(dynamic_shapes, tuple), (
                f"Unexpected type {type(dynamic_shapes)} for dynamic_shapes, "
                f"wrapped={wrapped}, input type is {type(self.inputs)}"
            )
            assert len(dynamic_shapes) == 1, (
                f"Unexpected number of dynamic_shapes {len(dynamic_shapes)}, "
                f"input type is {type(self.inputs)}"
            )
            dynamic_shapes = dynamic_shapes[0]

        dyn_inputs = []
        dyn_values = {}
        for i in range(len(self.inputs)):
            inp = inputs[i]
            if inp is None:
                dyn_inputs.append(None)
                continue
            if i >= len(dynamic_shapes):
                dyn_inputs.append(inp)
                continue
            new_shape = []
            dyn_shape = dynamic_shapes[i]
            assert isinstance(dyn_shape, dict), (
                f"Unexpected type for input {i}, dyn_shape={dyn_shape}, "
                f"shape of input[{i}]={inp.shape}, "
                f"dynamic_shapes={dynamic_shapes}, "
            )
            for j in range(len(inp.shape)):
                if inp.shape[j] != 1 or j not in dyn_shape:
                    new_shape.append(inp.shape[j])
                    continue
                name = dyn_shape[j]
                if name in dyn_values:
                    new_shape.append(dyn_values[name])
                    continue
                d = inp.shape[j]
                d += 1
                dyn_values[name] = d
                new_shape.append(d)

            new_shape = tuple(new_shape)
            if new_shape == inp.shape:
                dyn_inputs.append(inp)
                continue
            dyn_inputs.append(inp.expand(new_shape))
        return tuple(dyn_inputs)

    def get_input_shapes(
        self,
        dynamic: bool = False,
        wrapped: bool = False,
        export: bool = False,
        inputs: Optional[Any] = None,
    ) -> Any:
        """
        Returns the input shapes.

        :param dynamic: dynamic, yes or no?
        :param wrapped: wrapped model
        :param inputs: existing inputs or None to use `self.inputs`
        :param export: returns the shapes for the inputs used for export
        :return: new inputs
        """
        if inputs is None:
            return self.get_input_shapes(
                dynamic=dynamic, export=export, wrapped=wrapped, inputs=self.inputs
            )

        assert isinstance(
            inputs, tuple
        ), f"Not implemented for type(self.inputs)={type(inputs)}"
        dynamic_shapes = self.get_dynamic_shapes(True, wrapped=wrapped)
        if wrapped:
            assert isinstance(dynamic_shapes, tuple), (
                f"Unexpected type {type(dynamic_shapes)} for dynamic_shapes, "
                f"wrapped={wrapped}, input type is {type(self.inputs)}"
            )
            assert len(dynamic_shapes) == 1, (
                f"Unexpected number of dynamic_shapes {len(dynamic_shapes)}, "
                f"input type is {type(self.inputs)}"
            )
            dynamic_shapes = dynamic_shapes[0]
        dyn_input_shapes = []
        dyn_values = {}
        for i in range(len(self.inputs)):
            inp = self.inputs[i]
            if inp is None:
                dyn_input_shapes.append(None)
                continue
            if i >= len(dynamic_shapes):
                dyn_input_shapes.append(inp.shape)
                continue
            new_shape = []
            dyn_shape = dynamic_shapes[i]
            assert isinstance(dyn_shape, dict), (
                f"Unexpected type for input {i}, wrapped={wrapped}, "
                f"dyn_shape{dyn_shape}, shape of input[{i}]={inp.shape}, "
                f"dynamic_shapes={dynamic_shapes}, "
            )
            for j in range(len(inp.shape)):
                if not export or j not in dyn_shape or inp.shape[j] != 1:
                    new_shape.append(inp.shape[j])
                    continue
                name = dyn_shape[j]
                if name in dyn_values:
                    new_shape.append(dyn_values[name])
                    continue
                d = inp.shape[j]
                d += 1
                dyn_values[name] = d
                new_shape.append(d)

            new_shape = tuple(new_shape)
            dyn_input_shapes.append(new_shape)
        return tuple(dyn_input_shapes)

    def make_dynamic_inputs(self, wrapped: bool = False):
        """
        Creates dynamic inputs based on the static ones by changing the dynamic
        according to the definition of the dynamic_shapes.
        """
        assert isinstance(
            self.inputs, tuple
        ), f"Not implemented for type(self.inputs)={type(self.inputs)}"
        dynamic_shapes = self.get_dynamic_shapes(True, wrapped=wrapped)
        if wrapped:
            assert isinstance(dynamic_shapes, tuple), (
                f"Unexpected type {type(dynamic_shapes)} for dynamic_shapes, "
                f"wrapped={wrapped}, input type is {type(self.inputs)}"
            )
            assert len(dynamic_shapes) == 1, (
                f"Unexpected number of dynamic_shapes {len(dynamic_shapes)}, "
                f"input type is {type(self.inputs)}"
            )
            dynamic_shapes = dynamic_shapes[0]
        dyn_inputs = []
        dyn_values = {}
        for i in range(len(self.inputs)):
            inp = self.inputs[i]
            if i >= len(dynamic_shapes):
                dyn_inputs.append(inp)
                continue
            new_shape = []
            dyn_shape = dynamic_shapes[i]
            if inp is None:
                dyn_inputs.append(None)
                continue
            assert isinstance(dyn_shape, dict), (
                f"Unexpected type for input {i}, dyn_shape{dyn_shape}, "
                f"shape of input[{i}]={inp.shape}, "
                f"dynamic_shapes={dynamic_shapes}, "
            )
            for j in range(len(inp.shape)):
                if j not in dyn_shape:
                    new_shape.append(inp.shape[j])
                    continue
                name = dyn_shape[j]
                if name in dyn_values:
                    new_shape.append(dyn_values[name])
                    continue
                d = inp.shape[j]
                d += 1
                dyn_values[name] = d
                new_shape.append(d)

            new_shape = tuple(new_shape)
            zeros = torch.zeros(new_shape, dtype=inp.dtype, device=inp.device)
            slices = tuple(slice(0, s) for s in inp.shape)
            zeros[slices] = inp[slices]
            dyn_inputs.append(zeros)
        return tuple(dyn_inputs)

    def make_feeds(
        self, exporter: str, filename: Optional[str] = None, dynamic: bool = False
    ):
        """Creates feed inputs."""
        if exporter in {
            "eager",
            "export",
            "compile",
            "inductor",
            "inductor-partial",
            "dort",
            "cort",
            "cortgrad",
        }:
            return self.inputs

        use_inputs = self.inputs if not dynamic else self.make_dynamic_inputs(wrapped=True)

        # for onnx
        onx = onnx.load(filename, load_external_data=False)
        initializer_names = {i.name for i in onx.graph.initializer}
        names = [_.name for _ in onx.graph.input if _.name not in initializer_names]
        if isinstance(use_inputs, dict):
            assert set(names) == set(
                self.inputs
            ), f"Input names mismatch, got {set(use_inputs)}, expecting {set(names)}."
            return self.inputs
        inputs = [i for i, d in zip(use_inputs, self.raw_use_defaults) if not d]
        if len(names) > len(inputs) and any(isinstance(i, list) for i in inputs):
            # We need to flatten the inputs.
            new_inputs = []
            for i in inputs:
                if isinstance(i, torch.Tensor):
                    new_inputs.append(i)
                    continue
                if isinstance(i, list):
                    for u in i:
                        if isinstance(u, torch.Tensor):
                            new_inputs.append(u)
                            continue
                        raise AssertionError(
                            f"Unable to process input type {type(u)} in input list"
                        )
                    continue
                raise AssertionError(f"Unable to process input type {type(i)}")
        else:
            new_inputs = inputs
        assert len(names) == len(new_inputs), (
            f"Mismatch number of inputs, {len(inputs)} ({len(new_inputs)}) "
            f"inputs, there are {len(new_inputs)} flattened inputs.\n----\n"
            f"names={names}\n----\ninput types={[type(i) for i in inputs]}\n----\n"
            f"named parameters={sorted(p[0] for p in self.model.named_parameters())}"
            f"\n----\nnamed buffers={sorted(p[0] for p in self.model.named_buffers())}"
            f"\n----\nself.raw_input_names={self.raw_input_names}\n----\n"
            f"self.raw_use_defaults={self.raw_use_defaults}\n----\n"
            f"initializer_names={sorted(initializer_names)}\n----\n"
        )
        return dict(zip(names, new_inputs))
