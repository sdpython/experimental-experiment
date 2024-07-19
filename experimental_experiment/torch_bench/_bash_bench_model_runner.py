import collections
import inspect
import time
import shutil
import os
from typing import Any, Callable, Optional, Tuple, Dict, List
import numpy as np
import onnx
import torch


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
                        raise RuntimeError(
                            f"Failed to load model {args!r} "
                            f"with following error(s): {e!r}."
                        )

        return wrapper

    return decorator


def get_dynamo_stats() -> Dict[str, float]:
    """
    Returns statistics on memory as a dictionary.
    """
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
            "cudagraph_skips": torch._dynamo.utils.counters["inductor"][
                "cudagraph_skips"
            ],
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
        for k in self.model.parameters():
            yield k


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
    """

    _patched = None

    @classmethod
    def isinstance_namedtuple(cls, x):
        return isinstance(x, tuple) and getattr(x, "_fields", None) is not None

    @classmethod
    def _to_type(cls, o, dtype):
        if dtype is None or o is None or isinstance(o, (str, bool, int, float)):
            return o
        if isinstance(o, list):
            return [cls._to_type(v, dtype) for v in o]
        if hasattr(o, "dtype"):
            if o.dtype in {torch.float32, torch.float64, torch.float16, torch.bfloat16}:
                return o.to(dtype)
            return o
        if cls.isinstance_namedtuple(o):
            new_vals = {}
            for k in o._fields:
                new_vals[k] = cls._to_type(getattr(o, k), dtype)
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
            ext = {k: cls._to_type(v, dtype) for k, v in ext.items()}
            return o.__class__(**ext)
        try:
            return o.to(dtype)
        except (AttributeError, AssertionError) as e:
            raise AssertionError(
                f"Unable to convert class {type(o)} to {dtype} "
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
    ):
        if dtype is None:
            cvt = lambda o: self._to_type(o, device)  # noqa: E731
        else:
            cvt = lambda o: self._to_type(self._to_type(o, dtype), device)  # noqa: E731

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
                f"Unexpected input name in {list(sorted(inputs))} and "
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
        if to_tuple:
            self.model = WrappedModelToTuple(model_cvt)
        else:
            self.model = WrappedModelBase(model_cvt)
        self.device = device
        self.dtype = dtype
        self.inputs = inputs
        self.repeat = repeat
        self.warmup = warmup
        self.suite = suite

    def run(self) -> Any:
        return self.model(*self.inputs)

    def parameters_size(self) -> int:
        """Returns the size of all parameters (do not take into account dtype)."""
        res = 0
        for p in self.model.parameters():
            res += np.prod(list(p.shape))
        return res

    def parameters_dtype(self) -> str:
        """Returns the unique dtypes of all parameters."""
        return ",".join(
            sorted(
                set(str(p.dtype).replace("torch.", "") for p in self.model.parameters())
            )
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
    ):
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
        """
        assert not fake_tensor, "fake_tensor not implemented."

        if ModelRunner._patched == "torch-onnx":
            try:
                import torch_onnx

                torch_onnx.unpatch_torch()
                ModelRunner._patched = "unpatched"
            except ImportError:
                pass

        assert ModelRunner._patched in (None, "unpatched") or (
            ModelRunner._patched == "torch-onnx" and exporter == "torch-onnx"
        ), f"Unable to continue as ModelRunner is patched with {ModelRunner._patched!r}."

        if exporter == "custom":
            return self._to_onnx_custom(
                name,
                dynamic=dynamic,
                fake_tensor=fake_tensor,
                no_grad=no_grad,
                optimization=optimization,
                verbose=verbose,
                target_opset=target_opset,
            )
        if exporter == "script":
            return self._to_onnx_script(
                name,
                dynamic=dynamic,
                fake_tensor=fake_tensor,
                no_grad=no_grad,
                optimization=optimization,
                verbose=verbose,
                target_opset=target_opset,
            )

        if exporter == "torch-onnx":
            assert ModelRunner._patched in (
                None,
                "torch-onnx",
            ), f"A new patch should not be applied on {ModelRunner._patched!r}."
            import torch_onnx

            if ModelRunner._patched is None:
                torch_onnx.patch_torch(error_report=True)
                ModelRunner._patched = "torch-onnx"
            onx, stats = self._to_onnx_script(
                name,
                dynamic=dynamic,
                fake_tensor=fake_tensor,
                no_grad=no_grad,
                optimization=optimization,
                verbose=verbose,
                target_opset=target_opset,
            )
            return onx, stats

        if exporter == "dynamo":
            return self._to_onnx_dynamo(
                name,
                dynamic=dynamic,
                fake_tensor=fake_tensor,
                no_grad=no_grad,
                optimization=optimization,
                verbose=verbose,
                target_opset=target_opset,
            )
        if exporter == "dynamo2":
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
    ):
        assert not fake_tensor, "fake_tensor not implemented."
        assert not dynamic, "dynamic true not implemented yet"
        assert no_grad, "no_grad false not implemented yet"
        from ..torch_interpreter import to_onnx
        from ..xbuilder import OptimizationOptions

        if optimization:
            # cuda = any(m.is_cuda for m in self.model.parameters())
            options = OptimizationOptions(
                patterns=optimization,
                verbose=10 if verbose >= 100 else (1 if verbose > 1 else 0),
                processor="CUDA" if self.device == "cuda" else "CPU",
            )
        else:
            options = None

        with torch.no_grad():
            onx, stats = to_onnx(
                self.model,
                self.inputs,
                optimize=bool(optimization),
                large_model=True,
                verbose=10 if verbose >= 100 else (1 if verbose > 1 else 0),
                target_opset=target_opset,
                return_optimize_report=True,
                options=options,
            )
        onx.save(name)
        return onx.model_proto, stats

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
        assert not dynamic, "dynamic true not implemented yet"
        assert no_grad, "no_grad false not implemented yet"
        assert (
            not optimization
        ), f"optimization {optimization!r} not compatible with script"

        with torch.no_grad():
            torch.onnx.export(
                self.model,
                self.inputs,
                name,
                do_constant_folding=False,
                opset_version=target_opset,
            )
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
    ):
        assert not fake_tensor, "fake_tensor not implemented."
        assert not dynamic, "dynamic true not implemented yet"
        assert no_grad, "no_grad false not implemented yet"
        assert (
            not optimization
        ), f"optimization {optimization!r} not compatible with dynamo"

        with torch.no_grad():
            torch.onnx.export(
                self.model,
                self.inputs,
                name,
                do_constant_folding=False,
                opset_version=target_opset,
                dynamo=True,
            )
        sarif = "report_dynamo_export.sarif"
        if os.path.exists(sarif):
            folder = os.path.split(name)[0]
            with open(sarif, "r", encoding="utf-8") as f:
                self.error_report = f.read()
            os.remove(sarif)
            with open(os.path.join(folder, sarif), "w", encoding="utf-8") as f:
                f.write(self.error_report)
        return onnx.load(name, load_external_data=False), None

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
        assert not dynamic, "dynamic true not implemented yet"
        assert no_grad, "no_grad false not implemented yet"

        with torch.no_grad():
            exported = torch.onnx.dynamo_export(
                self.model,
                *self.inputs,
                export_options=torch.onnx.ExportOptions(
                    dynamic_shapes=dynamic,
                    # registry=torch.onnx.OnnxRegistry()
                ),
            )

        exported.save(name)
        onx = onnx.load(name, load_external_data=True)
        onnx.save(onx, name, save_as_external_data=True)

        if not optimization:
            return onnx.load(name, load_external_data=False), None

        opts = optimization.split("+")
        shutil.copy(name, name + ".raw.onnx")
        model_proto = onnx.load(name, load_external_data=True)

        for opt in opts:
            if opt == "default":
                from onnxscript.optimizer import optimize
                from onnxscript.rewriter import rewrite
                from onnx.inliner import inline_local_functions

                first_model_proto = model_proto
                model_proto = optimize(
                    model_proto,
                    num_iterations=2,
                    onnx_shape_inference=False,
                )
                model_proto = rewrite(model_proto)
                model_proto = inline_local_functions(model_proto)
                del first_model_proto.graph.node[:]
                del first_model_proto.functions[:]
                first_model_proto.graph.node.extend(model_proto.graph.node)
                first_model_proto.functions.extend(model_proto.functions)
                continue

            if opt == "llm":
                from onnxscript import ir
                from onnxscript.rewriter.llama_rule_sets import llama_p0_rule_set

                ir_model = ir.serde.deserialize_model(model_proto)
                rule_set = llama_p0_rule_set()
                rule_set.apply_to_model(ir_model)
                model_proto = ir.serde.serialize_model(ir_model)
                continue

            raise AssertionError(f"Unexpected value for optimization={optimization!r}.")

        onnx.save(model_proto, name, save_as_external_data=True)
        model_proto = onnx.load(name, load_external_data=False)
        return model_proto, None

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
        assert not dynamic, "dynamic true not implemented yet"
        assert no_grad, "no_grad false not implemented yet"
        assert (
            not optimization
        ), f"optimization {optimization!r} not compatible with export"
        from torch.export import export

        with torch.no_grad():
            res = export(self.model, self.inputs)
        return res, None

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
        assert not dynamic, "dynamic true not implemented yet"
        assert no_grad, "no_grad false not implemented yet"
        assert (
            not optimization
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
            not optimization
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
        assert not dynamic, "dynamic true not implemented yet"
        assert no_grad, "no_grad true not implemented yet"
        assert (
            not optimization
        ), f"optimization {optimization!r} not compatible with compile"

        def custom_backend(
            gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]
        ):
            return gm.forward

        with torch.no_grad():
            res = torch.compile(
                self.model, fullgraph=True, backend=lambda gm, inputs: gm.forward
            )
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
    ):
        assert not fake_tensor, "fake_tensor not implemented."
        assert not dynamic, "dynamic true not implemented yet"
        assert no_grad, "no_grad true not implemented yet"
        assert (
            not optimization
        ), f"optimization {optimization!r} not compatible with inductor"

        with torch.no_grad():
            res = torch.compile(self.model, backend="inductor", fullgraph=True)
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
            not optimization
        ), f"optimization {optimization!r} not compatible with dort"

        with torch.no_grad():
            res = torch.compile(self.model, backend="onnxrt", fullgraph=True)
        return res, None

    def make_feeds(self, exporter: str, filename: Optional[str] = None):
        """Creates feed inputs."""
        if exporter in {"eager", "export", "compile", "inductor", "dort"}:
            return self.inputs
        onx = onnx.load(filename, load_external_data=False)
        initializer_names = set(i.name for i in onx.graph.initializer)
        names = [_.name for _ in onx.graph.input if _.name not in initializer_names]
        if isinstance(self.inputs, dict):
            assert set(names) == set(self.inputs), (
                f"Input names mismatch, "
                f"got {set(self.inputs)}, expecting {set(names)}."
            )
            return self.inputs
        inputs = [i for i, d in zip(self.inputs, self.raw_use_defaults) if not d]
        assert len(names) == len(inputs), (
            f"Mismatch number of outputs, {len(inputs)} inputs for {names}.\n"
            f"self.raw_input_names={self.raw_input_names},\n"
            f"self.raw_use_defaults={self.raw_use_defaults},\n"
            f"initializer_names={initializer_names}"
        )
        return dict(zip(names, inputs))
