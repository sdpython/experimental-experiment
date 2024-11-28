import contextlib
import io
import itertools
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import onnx


def discover():
    """
    Discovers all model cases used to evaluate an exporter.

    .. runpython::
        :showcode:

        import pprint
        from experimental_experiment.torch_interpret.eval import discover

        pprint.pprint(discover())
    """
    from . import model_cases

    res = {}
    for m in model_cases.__dict__.values():
        if m is None or isinstance(m, str):
            continue
        if not hasattr(m, "forward"):
            continue
        assert m.__name__ not in res, f"Case {m.__name__!r} is duplicated."
        assert hasattr(m, "_inputs"), f"Attribute '_inputs' is missing from class {m}"
        assert hasattr(m, "_dynamic"), f"Attribute '_inputs' is missing from class {m}"
        res[m.__name__] = m
    return res


def evaluation(
    exporters: Tuple[str] = (
        "export-strict",
        "export-nostrict",
        "export-tracing",
        "custom-strict",
        "custom-nostrict",
        "custom-tracing",
    ),
    dynamic: Tuple[bool] = (False, True),
    cases: Optional[Dict[str, type]] = None,
    verbose: int = 0,
    quiet: bool = True,
) -> List[Dict[str, Any]]:
    """
    Evaluates exporter for a list of cases.

    :param exporters: exporters to evaluate
    :param dynamic: evaluate static shape and dynamic shapes
    :param cases: model cases to evaluate
    :param verbose: verbosity
    :param quiet: catch exception
    :return: results, list of dictionaries
    """
    if cases is None:
        cases = discover()
    elif isinstance(cases, str):
        cases = (cases,)
    if isinstance(exporters, str):
        exporters = (exporters,)
    if isinstance(cases, (list, tuple)):
        all_cases = discover()
        new_cases = []
        for c in cases:
            if "*" in c or "?" in c:
                # regex
                reg = re.compile(c)
                new_cases.extend(k for k in all_cases if reg.match(k))
            else:
                new_cases.append(c)
        cases = {k: v for k, v in all_cases.items() if k in set(new_cases)}
    if isinstance(dynamic, (bool, int)):
        dynamic = (dynamic,)
    sorted_cases = sorted(cases.items())
    loop = list(itertools.product(sorted_cases, dynamic, exporters))
    if verbose:
        try:
            import tqdm

            loop = tqdm.tqdm(loop)
        except ImportError:

            def _loop():
                for _ in loop:
                    print(f"[evaluation] {_}")
                    yield _

    assert len(loop) > 0, f"No case to test for cases={cases!r}."
    obs = []
    for case, dyn, exporter in loop:
        name, cls_model = case
        res = run_exporter(exporter, cls_model, dyn, quiet=quiet, verbose=max(0, verbose - 1))
        res.update(dict(name=name, dynamic=int(dyn), exporter=exporter))
        obs.append(res)
    return obs


def flatten_inputs(x: Any) -> List["torch.Tensor"]:  # noqa: F821
    """
    Flatten inputs.
    """
    if x is None:
        return x
    import torch

    if isinstance(x, (list, tuple)):
        res = []
        for i in x:
            if i is None or isinstance(
                i,
                (
                    torch.Tensor,
                    torch.SymInt,
                    torch.SymFloat,
                    int,
                    float,
                ),
            ):
                res.append(i)
            else:
                res.extend(flatten_inputs(i))
        return tuple(res) if isinstance(x, tuple) else res
    raise AssertionError(f"Unexpected type {type(x)} for x")


def _to_numpy(x):
    if hasattr(x, "numpy"):
        return x.numpy()
    if isinstance(x, int):
        # onnxruntime does not like scalar
        return np.array([x], dtype=np.int64)
    if isinstance(x, float):
        # onnxruntime does not like scalar
        return np.array([x], dtype=np.float64)
    raise TypeError(f"Unable to convert type {type(x)}, x={x} into numpy")


def _make_feeds(names, args):
    if len(names) == len(args):
        return {k: _to_numpy(v) for k, v in zip(names, args)}
    if len(names) > len(args):
        flats = flatten_inputs(args)
        return {k: _to_numpy(v) for k, v in zip(names, flats)}
    from ...helpers import string_type

    raise RuntimeError(f"Unable to handle names={names!r} and args={string_type(args)}")


def run_exporter(
    exporter: str,
    cls_model: type,
    dynamic: bool = False,
    quiet: bool = False,
    verbose: int = 0,
) -> Dict[str, Any]:
    """
    Runs an exporter and returns whether it fails or not.

    :param exporter: exporter
    :param cls_model: model class to create
    :param inputs: list of inputs to try
    :param dynamic: use dynamic shape or not
    :param quiet: raise exception or not
    :param verbose: verbosity
    :return: results
    """
    from ...bench_run import max_diff
    from ...helpers import string_type, pretty_onnx

    assert hasattr(
        cls_model, "_inputs"
    ), f"Attribute '_inputs' is missing from class {cls_model}"

    model = cls_model()
    inputs = cls_model._inputs
    if isinstance(inputs, tuple):
        inputs = [inputs]
    if dynamic:
        assert hasattr(
            cls_model, "_dynamic"
        ), f"Attribute '_inputs' is missing from class {cls_model}"
        dynamic_shapes = cls_model._dynamic
    else:
        dynamic_shapes = None

    if verbose > 0:
        print(
            f"[run_exporter] exporter={exporter}, model={cls_model.__name__}, "
            f"dynamic={dynamic}, inputs={string_type(inputs,with_shape=True)}"
        )

    builder = None
    onx = None

    if exporter.startswith("export-"):
        if exporter == "export-strict":
            import torch

            try:
                exported = torch.export.export(
                    model, inputs[0], dynamic_shapes=dynamic_shapes, strict=True
                )
            except Exception as e:
                if not quiet:
                    raise
                return dict(error=str(e), success=0, error_step="export")
            if verbose >= 9:
                print(exported.graph)
            mod = exported.module()
        elif exporter == "export-nostrict":
            import torch

            try:
                exported = torch.export.export(
                    model, inputs[0], dynamic_shapes=dynamic_shapes, strict=False
                )
            except Exception as e:
                if not quiet:
                    raise
                return dict(error=str(e), success=0, error_step="export")
            if verbose >= 9:
                print(exported.graph)
            mod = exported.module()
        elif exporter == "export-tracing":
            import torch

            try:
                tracer_class = torch.fx.Tracer
                graph = tracer_class().trace(model)
                mod = torch.fx.GraphModule(model, graph)
            except Exception as e:
                if not quiet:
                    raise
                return dict(error=str(e), success=0, error_step="export")
            if verbose >= 9:
                print(graph)
        else:
            raise AssertionError(f"Unexpected exporter={exporter!r}")
    else:
        import onnxruntime

        if exporter == "custom-strict":
            from experimental_experiment.torch_interpreter import to_onnx, ExportOptions

            try:
                onx, builder = to_onnx(
                    model,
                    inputs[0],
                    dynamic_shapes=dynamic_shapes,
                    export_options=ExportOptions(strict=True),
                    return_builder=True,
                )
            except Exception as e:
                if not quiet:
                    raise RuntimeError(
                        f"Unable to convert model={model.__class__.__name__}, "
                        f"input={string_type(inputs[0], with_shape=True)}, "
                        f"dynamic_shapes={dynamic_shapes}, "
                        f"exporter={exporter!r}"
                    ) from e
                return dict(error=str(e), success=0, error_step="export")
        elif exporter == "custom-strict-dec":
            from experimental_experiment.torch_interpreter import to_onnx, ExportOptions

            try:
                onx, builder = to_onnx(
                    model,
                    inputs[0],
                    dynamic_shapes=dynamic_shapes,
                    export_options=ExportOptions(strict=True, decomposition_table="default"),
                    return_builder=True,
                )
            except Exception as e:
                if not quiet:
                    raise RuntimeError(
                        f"Unable to convert model={model.__class__.__name__}, "
                        f"input={string_type(inputs[0], with_shape=True)}, "
                        f"dynamic_shapes={dynamic_shapes}, "
                        f"exporter={exporter!r}"
                    ) from e
                return dict(error=str(e), success=0, error_step="export")
        elif exporter == "custom-nostrict":
            from experimental_experiment.torch_interpreter import to_onnx, ExportOptions

            try:
                onx, builder = to_onnx(
                    model,
                    inputs[0],
                    dynamic_shapes=dynamic_shapes,
                    export_options=ExportOptions(strict=False),
                    return_builder=True,
                )
            except Exception as e:
                if not quiet:
                    raise RuntimeError(
                        f"Unable to convert model={model.__class__.__name__}, "
                        f"input={string_type(inputs[0], with_shape=True)}, "
                        f"dynamic_shapes={dynamic_shapes}, "
                        f"exporter={exporter!r}"
                    ) from e
                return dict(error=str(e), success=0, error_step="export")
        elif exporter == "custom-nostrict-dec":
            from experimental_experiment.torch_interpreter import to_onnx, ExportOptions

            try:
                onx, builder = to_onnx(
                    model,
                    inputs[0],
                    dynamic_shapes=dynamic_shapes,
                    export_options=ExportOptions(strict=False, decomposition_table="default"),
                    return_builder=True,
                )
            except Exception as e:
                if not quiet:
                    raise RuntimeError(
                        f"Unable to convert model={model.__class__.__name__}, "
                        f"input={string_type(inputs[0], with_shape=True)}, "
                        f"dynamic_shapes={dynamic_shapes}, "
                        f"exporter={exporter!r}"
                    ) from e
                return dict(error=str(e), success=0, error_step="export")
        elif exporter == "custom-tracing":
            from experimental_experiment.torch_interpreter import to_onnx, ExportOptions

            try:
                onx, builder = to_onnx(
                    model,
                    inputs[0],
                    dynamic_shapes=dynamic_shapes,
                    export_options=ExportOptions(tracing=True),
                    return_builder=True,
                )
            except Exception as e:
                if not quiet:
                    raise RuntimeError(
                        f"Unable to convert model={model.__class__.__name__}, "
                        f"input={string_type(inputs[0], with_shape=True)}, "
                        f"dynamic_shapes={dynamic_shapes}, "
                        f"exporter={exporter!r}"
                    ) from e
                return dict(error=str(e), success=0, error_step="export")
        elif exporter == "script":
            import torch

            f = f"evaluation-{model.__class__.__name__}-{dynamic}-{exporter}.onnx"
            dynamic_axes = {}
            input_names = []
            if dynamic_shapes:
                for k, v in dynamic_shapes.items():
                    dynamic_axes[k] = {_k: _v.__name__ for _k, _v in v.items()}
                    input_names.append(k)
            while len(input_names) < len(inputs[0]):
                input_names.append(f"args_{len(input_names)}")
            if verbose >= 5:
                print(
                    f"[run_exporter] dynamic_axes={dynamic_axes}, "
                    f"dynamic_shapes={dynamic_shapes}, input_names={input_names}"
                )
            try:
                if verbose >= 2:
                    torch.onnx.export(
                        model,
                        inputs[0],
                        f,
                        dynamic_axes=dynamic_axes,
                        input_names=input_names if dynamic_axes else None,
                        dynamo=False,
                    )
                    onx = onnx.load(f)
                else:
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                        io.StringIO()
                    ):
                        torch.onnx.export(
                            model,
                            inputs[0],
                            f,
                            dynamic_axes=dynamic_axes,
                            input_names=input_names if dynamic_axes else None,
                            dynamo=False,
                        )
                        onx = onnx.load(f)
            except Exception as e:
                if not quiet:
                    raise RuntimeError(
                        f"Unable to convert model={model.__class__.__name__}, "
                        f"input={string_type(inputs[0], with_shape=True)}, "
                        f"dynamic_shapes={dynamic_shapes}, "
                        f"exporter={exporter!r}"
                    ) from e
                return dict(error=str(e), success=0, error_step="export")
        elif exporter == "dynamo":
            import torch

            try:
                if verbose >= 2:
                    onx = torch.onnx.export(
                        model,
                        inputs[0],
                        dynamic_shapes=dynamic_shapes,
                        dynamo=True,
                    ).model_proto
                else:
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                        io.StringIO()
                    ):
                        onx = torch.onnx.export(
                            model,
                            inputs[0],
                            dynamic_shapes=dynamic_shapes,
                            dynamo=True,
                        ).model_proto
            except Exception as e:
                if not quiet:
                    raise RuntimeError(
                        f"Unable to convert model={model.__class__.__name__}, "
                        f"input={string_type(inputs[0], with_shape=True)}, "
                        f"dynamic_shapes={dynamic_shapes}, "
                        f"exporter={exporter!r}"
                    ) from e
                return dict(error=str(e), success=0, error_step="export")
        elif exporter == "dynamo-ir":
            import torch

            try:
                if verbose >= 2:
                    ep = torch.onnx.export(
                        model,
                        inputs[0],
                        dynamic_shapes=dynamic_shapes,
                        dynamo=True,
                    )
                    ep.optimize()
                    onx = ep.model_proto
                else:
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                        io.StringIO()
                    ):
                        ep = torch.onnx.export(
                            model,
                            inputs[0],
                            dynamic_shapes=dynamic_shapes,
                            dynamo=True,
                        )
                        ep.optimize()
                        onx = ep.model_proto
            except Exception as e:
                if not quiet:
                    raise RuntimeError(
                        f"Unable to convert model={model.__class__.__name__}, "
                        f"input={string_type(inputs[0], with_shape=True)}, "
                        f"dynamic_shapes={dynamic_shapes}, "
                        f"exporter={exporter!r}"
                    ) from e
                return dict(error=str(e), success=0, error_step="export")
        else:
            raise AssertionError(f"Unexpected exporter={exporter!r}")

        if verbose >= 9:
            print("[run_exporter] onnx model")
            print(builder.pretty_text() if builder is not None else pretty_onnx(onx))
        if verbose >= 2:
            onnx.save(onx, f"evaluation-{model.__class__.__name__}-{dynamic}-{exporter}.onnx")

        names = [i.name for i in onx.graph.input]
        flats = flatten_inputs(inputs[0]) if len(names) > len(inputs[0]) else inputs[0]

        assert quiet or len(names) == len(flats), (
            f"Input mismatch, inputs[0]={string_type(inputs[0])} "
            f"inputs but names={names!r}, "
            f"model={cls_model.__name__}, export={exporter!r}"
        )
        if len(names) != len(flats):
            return dict(
                error=f"Input mismatch, inputs[0]={string_type(inputs[0])} "
                f"but names={names!r}, model={cls_model.__name__}, export={exporter!r}",
                success=0,
                error_step="inputs",
            )
        try:
            sess = onnxruntime.InferenceSession(
                onx.SerializeToString(), providers=["CPUExecutionProvider"]
            )
        except Exception as e:
            if not quiet:
                raise
            return dict(error=str(e), success=0, error_step="ort-init")

        mod = lambda *args, names=names: sess.run(None, _make_feeds(names, args))  # noqa: E731

    expected = model(*inputs[0])
    try:
        got = mod(*inputs[0])
    except Exception as e:
        if not quiet:
            raise RuntimeError(
                f"onnxruntime failed, feeds=\n{string_type(inputs[0], with_shape=True)} "
                f"\nmodel=\n{pretty_onnx(onx)}"
            ) from e
        return dict(error=str(e), success=0, error_step="run.0")

    try:
        disc = max_diff(expected, got)
    except Exception as e:
        if not quiet:
            raise
        return dict(error=str(e), success=0, error_step="discrepancy")

    if verbose >= 5 and np.isinf(disc["abs"]):
        print("[run_exporter] comparison issues with")
        print(f"--   inputs={string_type(inputs[0], with_shape=True)}")
        print(f"-- exported={string_type(expected, with_shape=True)}")
        print(f"--      got={string_type(got, with_shape=True)}")
    elif verbose >= 9:
        print("[run_exporter] inputs and outputs")
        print(f"--   inputs={string_type(inputs[0], with_shape=True, with_min_max=True)}")
        print(f"-- exported={string_type(expected, with_shape=True, with_min_max=True)}")
        print(f"--      got={string_type(got, with_shape=True, with_min_max=True)}")
    del disc["n"]
    del disc["sum"]
    disc.update(dict(success=1 if disc["abs"] < 0.1 else 0))
    if disc["abs"] >= 0.1:
        disc["error"] = "DIFF"
        disc["error_step"] = "DIFF"
    else:
        disc["success"] = 1

    if dynamic and onx is not None:
        ds = []
        for i in onx.graph.input:
            if i.type.tensor_type:
                for di, dim in enumerate(i.type.tensor_type.shape.dim):
                    if dim.dim_param:
                        ds.append((i.name, di, dim.dim_param))
        if verbose >= 2:
            print(f"[run_exporter] dynamic dimension={ds}")
        if not ds:
            return dict(error="no dynamic shape", success=0, error_step="dynamic")

    if dynamic and len(inputs) > 1:
        for index, i in enumerate(inputs):
            expected = model(*i)
            try:
                got = mod(*i)
            except Exception as e:
                if not quiet:
                    raise RuntimeError(
                        f"onnxruntime failed,\n-- feeds=\n{string_type(i, with_shape=True)} "
                        f"\n-- model=\n{pretty_onnx(onx)}"
                    ) from e
                return dict(error=str(e), success=0, error_step=f"run.{index}")

            try:
                d = max_diff(expected, got)
            except Exception as e:
                if not quiet:
                    raise
                return dict(error=str(e), success=0, error_step=f"discrepancy.{index}")

            if verbose >= 5 and np.isinf(d["abs"]):
                print(f"[run_exporter] comparison issues iteration {index}")
                print(f"--   inputs={string_type(i, with_shape=True)}")
                print(f"-- exported={string_type(expected, with_shape=True)}")
                print(f"--      got={string_type(got, with_shape=True)}")
            elif verbose >= 9:
                print(f"[run_exporter] inputs and outputs iteration {index}")
                print(f"--   inputs={string_type(i, with_shape=True, with_min_max=True)}")
                print(
                    f"-- exported={string_type(expected, with_shape=True, with_min_max=True)}"
                )
                print(f"--      got={string_type(got, with_shape=True, with_min_max=True)}")
            del d["n"]
            del d["sum"]
            if d["abs"] >= 0.1:
                d["error"] = f"diff.{index}"
                d["error_step"] = f"diff.{index}"
                disc.update(d)

    return disc
