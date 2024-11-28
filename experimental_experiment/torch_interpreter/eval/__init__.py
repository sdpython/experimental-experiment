import contextlib
import io
import itertools
from typing import Any, Dict, List, Optional, Tuple


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
    if isinstance(cases, list):
        cases = {k: v for k, v in discover().items() if k in set(cases)}
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

    obs = []
    for case, dyn, exporter in loop:
        name, cls_model = case
        res = run_exporter(exporter, cls_model, dyn, quiet=quiet)
        res.update(dict(name=name, dynamic=int(dyn), exporter=exporter))
        obs.append(res)
    return obs


def run_exporter(
    exporter: str, cls_model: type, dynamic: bool = False, quiet: bool = False
) -> Dict[str, Any]:
    """
    Runs an exporter and returns whether it fails or not.

    :param exporter: exporter
    :param cls_model: model class to create
    :param inputs: list of inputs to try
    :param dynamic: use dynamic shape or not
    :param quiet: raise exception or not
    :return: results
    """
    from ...bench_run import max_diff

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
        mod = exported.module()

    else:
        import onnxruntime

        if exporter == "custom-strict":
            from experimental_experiment.torch_interpreter import to_onnx, ExportOptions

            try:
                onx = to_onnx(
                    model,
                    inputs[0],
                    dynamic_shapes=dynamic_shapes,
                    export_options=ExportOptions(strict=True),
                )
            except Exception as e:
                if not quiet:
                    raise
                return dict(error=str(e), success=0, error_step="export")
        elif exporter == "custom-strict-dec":
            from experimental_experiment.torch_interpreter import to_onnx, ExportOptions

            try:
                onx = to_onnx(
                    model,
                    inputs[0],
                    dynamic_shapes=dynamic_shapes,
                    export_options=ExportOptions(strict=True, decomposition_table="default"),
                )
            except Exception as e:
                if not quiet:
                    raise
                return dict(error=str(e), success=0, error_step="export")
        elif exporter == "custom-nostrict":
            from experimental_experiment.torch_interpreter import to_onnx, ExportOptions

            try:
                onx = to_onnx(
                    model,
                    inputs[0],
                    dynamic_shapes=dynamic_shapes,
                    export_options=ExportOptions(strict=False),
                )
            except Exception as e:
                if not quiet:
                    raise
                return dict(error=str(e), success=0, error_step="export")
        elif exporter == "custom-nostrict-dec":
            from experimental_experiment.torch_interpreter import to_onnx, ExportOptions

            try:
                onx = to_onnx(
                    model,
                    inputs[0],
                    dynamic_shapes=dynamic_shapes,
                    export_options=ExportOptions(strict=False, decomposition_table="default"),
                )
            except Exception as e:
                if not quiet:
                    raise
                return dict(error=str(e), success=0, error_step="export")
        elif exporter == "custom-tracing":
            from experimental_experiment.torch_interpreter import to_onnx, ExportOptions

            try:
                onx = to_onnx(
                    model,
                    inputs[0],
                    dynamic_shapes=dynamic_shapes,
                    export_options=ExportOptions(tracing=True),
                )
            except Exception as e:
                if not quiet:
                    raise
                return dict(error=str(e), success=0, error_step="export")
        elif exporter == "dynamo":
            import torch

            try:
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
                    raise
                return dict(error=str(e), success=0, error_step="export")
        elif exporter == "dynamo-ir":
            import torch

            try:
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
                    raise
                return dict(error=str(e), success=0, error_step="export")
        else:
            raise AssertionError(f"Unexpected exporter={exporter!r}")

        names = [i.name for i in onx.graph.input]
        assert len(names) == len(
            inputs[0]
        ), f"Input mismatch, {len(inputs[0])} inputs but names={names!r}"
        try:
            sess = onnxruntime.InferenceSession(
                onx.SerializeToString(), providers=["CPUExecutionProvider"]
            )
        except Exception as e:
            if not quiet:
                raise
            return dict(error=str(e), success=0, error_step="ort-init")

        mod = lambda *args: sess.run(  # noqa: E731
            None, {k: v.numpy() for k, v in zip(names, args)}
        )

    expected = model(*inputs[0])
    try:
        got = mod(*inputs[0])
    except Exception as e:
        if not quiet:
            raise
        return dict(error=str(e), success=0, error_step="run")

    disc = max_diff(expected, got)
    del disc["n"]
    del disc["sum"]
    disc.update(dict(success=1 if disc["abs"] < 0.1 else 0))
    if disc["abs"] >= 0.1:
        disc["error"] = "DISCREPANCY"
        disc["error_step"] = "DISCREPANCY"
    else:
        disc["success"] = 1
    return disc
