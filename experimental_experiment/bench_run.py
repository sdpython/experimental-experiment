import itertools
import multiprocessing
import os
import platform
import re
import subprocess
import sys
from argparse import Namespace
from typing import Any, Dict, List, Tuple, Union


class BenchmarkError(RuntimeError):
    pass


def get_machine() -> Dict[str, Union[str, int, float, Tuple[int, int]]]:
    """
    Returns the machine specification.
    """
    config: Dict[str, Union[str, int, float, Tuple[int, int]]] = dict(
        machine=str(platform.machine()),
        processor=str(platform.processor()),
        version=str(sys.version),
        cpu=int(multiprocessing.cpu_count()),
        executable=str(sys.executable),
    )
    try:
        import torch.cuda
    except ImportError:
        return config

    config["has_cuda"] = bool(torch.cuda.is_available())
    if config["has_cuda"]:
        config["capability"] = torch.cuda.get_device_capability(0)
        config["device_name"] = str(torch.cuda.get_device_name(0))
    return config


def _cmd_line(
    script_name: str, **kwargs: Dict[str, Union[str, int, float]]
) -> List[str]:
    args = [sys.executable, "-m", script_name]
    for k, v in kwargs.items():
        args.append(f"--{k}")
        args.append(str(v))
    return args


def _extract_metrics(text: str) -> Dict[str, str]:
    reg = re.compile(":(.*?),(.*.?);")
    res = reg.findall(text)
    if len(res) == 0:
        return {}
    kw = dict(res)
    new_kw = {}
    for k, w in kw.items():
        assert isinstance(k, str) and isinstance(
            w, str
        ), f"Unexpected type for k={k!r}, types={type(k)}, {type(w)})."
        assert "\n" not in w, f"Unexpected multi-line value for k={k!r}, value is\n{w}"
        assert len(w) < 100, f"Unexpected long value for k={k!r}, value is\n{w}"
        try:
            wi = int(w)
            new_kw[k] = wi
            continue
        except ValueError:
            pass
        try:
            wf = float(w)
            new_kw[k] = wf
            continue
        except ValueError:
            pass
        new_kw[k] = w
    return new_kw


def _make_prefix(script_name: str, index: int) -> str:
    name = os.path.splitext(script_name)[0]
    return f"{name}_dort_c{index}_"


def run_benchmark(
    script_name: str,
    configs: List[Dict[str, Union[str, int, float]]],
    verbose: int = 0,
    stop_if_exception: bool = True,
    dump: bool = False,
) -> List[Dict[str, Union[str, int, float, Tuple[int, int]]]]:
    """
    Runs a script multiple times and extract information from the output
    following the pattern ``:<metric>,<value>;``.

    :param script_name: python script to run
    :param configs: list of execution to do
    :param stop_if_exception: stop if one experiment failed, otherwise continue
    :param verbose: use tqdm to follow the progress
    :param dump: dump onnx file
    :return: values
    """
    assert configs, f"No configuration was given (script_name={script_name!r})"
    if verbose:
        from tqdm import tqdm

        loop = tqdm(configs)
    else:
        loop = configs

    data: List[Dict[str, Union[str, int, float, Tuple[int, int]]]] = []
    for i, config in enumerate(loop):
        cmd = _cmd_line(script_name, **config)

        if dump:
            os.environ["ONNXRT_DUMP_PATH"] = _make_prefix(script_name, i)
        else:
            os.environ["ONNXRT_DUMP_PATH"] = ""
        if verbose > 3:
            print(
                f"[run_benchmark] cmd={cmd if isinstance(cmd, str) else ' '.join(cmd)}"
            )
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        res = p.communicate()
        out, err = res
        sout = out.decode("utf-8", errors="ignore")
        serr = err.decode("utf-8", errors="ignore")

        if "ONNXRuntimeError" in serr or "ONNXRuntimeError" in sout:
            if stop_if_exception:
                raise RuntimeError(
                    f"Unable to continue with config {config} due to the "
                    f"following error\n{serr}"
                    f"\n----OUTPUT--\n{sout}"
                )

        metrics = _extract_metrics(sout)
        if len(metrics) == 0:
            if stop_if_exception:
                raise BenchmarkError(
                    f"Unable (2) to continue with config {config}, no metric was "
                    f"collected.\n--ERROR--\n{serr}\n--OUTPUT--\n{sout}"
                )
            else:
                metrics = {}
        metrics.update(config)
        metrics["ERROR"] = serr
        metrics["OUTPUT"] = sout
        metrics["CMD"] = f"[{' '.join(cmd)}]"
        data.append(metrics)
        if verbose > 5:
            print("--------------- ERROR")
            print(serr)
        if verbose >= 10:
            print("--------------- OUTPUT")
            print(sout)

    return data


def multi_run(kwargs: Namespace) -> bool:
    """
    Checks if multiple values were sent for one argument.
    """
    return any(isinstance(v, str) and "," in v for v in kwargs.__dict__.values())


def make_configs(kwargs: Namespace) -> List[Dict[str, Any]]:
    """
    Creates all the configurations based on the command line arguments.
    """
    args = []
    for k, v in kwargs.__dict__.items():
        if isinstance(v, str):
            args.append([(k, s) for s in v.split(",")])
        else:
            args.append([(k, v)])
    configs = list(itertools.product(*args))
    return [dict(c) for c in configs]


def make_dataframe_from_benchmark_data(data: List[Dict], detailed: bool = True) -> Any:
    """
    Creates a dataframe from the received data.

    :param data: list of dictionaries for every run
    :param detailed: remove multi line and long values
    :return: dataframe
    """
    import pandas

    if detailed:
        return pandas.DataFrame(data)

    new_data = []
    for d in data:
        g = {}
        for k, v in d.items():
            if not isinstance(v, str):
                g[k] = v
                continue
            if "\n" in v or len(v) > 100:
                continue
            g[k] = v
        new_data.append(g)
    return pandas.DataFrame(new_data)


def measure_discrepancies(
    expected: List[Tuple["torch.Tensor", ...]],  # noqa: F821
    outputs: List[Tuple["torch.Tensor", ...]],  # noqa: F821
) -> Tuple[float, float]:
    """
    Computes the discrepancies.

    :param expected: list of outputs coming from a torch model
    :param outputs: list of outputs coming from an onnx model
    :return: max absole errors, max relative errors
    """

    def _flatten(outputs):
        flat = []
        for tensor in outputs:
            if isinstance(tensor, tuple):
                flat.extend(_flatten(tensor))
            else:
                flat.append(tensor)
        return tuple(flat)

    abs_errs = []
    rel_errs = []
    for torch_outputs_mixed_types, onnx_outputs in zip(expected, outputs):
        torch_outputs = _flatten(torch_outputs_mixed_types)
        assert len(torch_outputs) == len(
            onnx_outputs
        ), f"Length mismatch {len(torch_outputs)} != {len(onnx_outputs)}"
        for torch_tensor, onnx_tensor in zip(torch_outputs, onnx_outputs):
            assert (
                torch_tensor.dtype == onnx_tensor.dtype
            ), f"Type mismatch {torch_tensor.dtype} != {onnx_tensor.dtype}"
            assert (
                torch_tensor.shape == onnx_tensor.shape
            ), f"Type mismatch {torch_tensor.shape} != {onnx_tensor.shape}"
            diff = torch_tensor - onnx_tensor
            abs_err = float(diff.abs().max())
            rel_err = float((diff.abs() / torch_tensor).max())
            abs_errs.append(abs_err)
            rel_errs.append(rel_err)
    return max(abs_errs), max(rel_errs)
