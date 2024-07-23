import itertools
import multiprocessing
import os
import platform
import re
import subprocess
import sys
import time
import warnings
from argparse import Namespace
from datetime import datetime
from typing import Any, Callable, Dict, List, Set, Tuple, Union, Optional


ILLEGAL_CHARACTERS_RE = re.compile(r"([\000-\010]|[\013-\014]|[\016-\037])")


class BenchmarkError(RuntimeError):
    pass


def _clean_string(s: str) -> str:
    if next(ILLEGAL_CHARACTERS_RE.finditer(s), None):
        ns = ILLEGAL_CHARACTERS_RE.sub("", s)
        return ns
    return s


def get_machine() -> Dict[str, Union[str, int, float, Tuple[int, int]]]:
    """
    Returns the machine specifications.
    """
    config: Dict[str, Union[str, int, float, Tuple[int, int]]] = dict(
        machine=str(platform.machine()),
        processor=str(platform.processor()),
        version=str(sys.version).split()[0],
        cpu=int(multiprocessing.cpu_count()),
        executable=str(sys.executable),
    )
    try:
        import torch.cuda
    except ImportError:
        return config

    config["has_cuda"] = bool(torch.cuda.is_available())
    if config["has_cuda"]:
        config["capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
        config["device_name"] = str(torch.cuda.get_device_name(0))
    return config


def _cmd_line(
    script_name: str, **kwargs: Dict[str, Union[str, int, float]]
) -> List[str]:
    args = [sys.executable, "-m", script_name]
    for k, v in kwargs.items():
        if v is None:
            continue
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
        if not (
            "err" in k.lower()
            or k
            in {
                "onnx_output_names",
                "onnx_input_names",
                "filename",
                "time_latency_t_detail",
                "time_latency_t_qu",
                "time_latency_eager_t_detail",
                "time_latency_eager_t_qu",
            }
            or len(w) < 500
        ):
            warnings.warn(
                f"Unexpected long value for model={kw.get('model_name', '?')}, k={k!r}, value has length {len(w)} is\n{w}"
            )
            continue
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


def _cmd_string(s: str) -> str:
    if s == "":
        return '""'
    return s.replace('"', '\\"')


def run_benchmark(
    script_name: str,
    configs: List[Dict[str, Union[str, int, float]]],
    verbose: int = 0,
    stop_if_exception: bool = True,
    dump: bool = False,
    temp_output_data: Optional[str] = None,
    dump_std: Optional[str] = None,
    start: int = 0,
    summary: Optional[Callable] = None,
) -> List[Dict[str, Union[str, int, float, Tuple[int, int]]]]:
    """
    Runs a script multiple times and extract information from the output
    following the pattern ``:<metric>,<value>;``.

    :param script_name: python script to run
    :param configs: list of execution to do
    :param stop_if_exception: stop if one experiment failed, otherwise continue
    :param verbose: use tqdm to follow the progress
    :param dump: dump onnx file, sets variable ONNXRT_DUMP_PATH
    :param temp_output_data: to save the data after every run to avoid losing data
    :param dump_std: dumps stdout and stderr in this folder
    :param start: start at this iteration
    :param summary: function to call on the temporary data and the final data
    :return: values
    """
    assert (
        temp_output_data is None or "temp" in temp_output_data
    ), f"Unexpected value for {temp_output_data!r}"
    assert configs, f"No configuration was given (script_name={script_name!r})"
    if verbose:
        from tqdm import tqdm

        loop = tqdm(configs)
    else:
        loop = configs

    data: List[Dict[str, Union[str, int, float, Tuple[int, int]]]] = []
    for iter_loop, config in enumerate(loop):
        if iter_loop < start:
            continue
        cmd = _cmd_line(script_name, **config)
        begin = time.perf_counter()

        if dump:
            os.environ["ONNXRT_DUMP_PATH"] = _make_prefix(script_name, iter_loop)
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

        if dump_std:
            if not os.path.exists(dump_std):
                os.makedirs(dump_std)
            root = os.path.split(script_name)[-1]
            filename = os.path.join(dump_std, f"{root}.{iter_loop}")
            if out.strip(b"\n \r\t"):
                with open(f"{filename}.stdout", "w") as f:
                    f.write(sout)
            if err.strip(b"\n \r\t"):
                with open(f"{filename}.stderr", "w") as f:
                    f.write(serr)

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
        metrics["DATE"] = f"{datetime.now():%Y-%m-%d}"
        metrics["ITER"] = iter_loop
        metrics["TIME_ITER"] = time.perf_counter() - begin
        metrics["ERROR"] = _clean_string(serr)
        metrics["OUTPUT"] = _clean_string(sout)
        metrics["CMD"] = f"[{' '.join(map(_cmd_string, cmd))}]"
        data.append(metrics)
        if verbose > 5:
            print(f"--------------- ITER={iter_loop} in {metrics['TIME_ITER']}")
            print("--------------- ERROR")
            print(serr)
        if verbose >= 10:
            print("--------------- OUTPUT")
            print(sout)
        if temp_output_data:
            df = make_dataframe_from_benchmark_data(data, detailed=False)
            if verbose > 2:
                print(f"Prints out the results into file {temp_output_data!r}")
            df.to_csv(temp_output_data, index=False)
            try:
                df.to_excel(temp_output_data + ".xlsx", index=False)
            except Exception:
                continue
            if summary:
                fn = f"{temp_output_data}.summary-partial.xlsx"
                if verbose > 2:
                    print(f"Prints out the results into file {fn!r}")
                summary(df, excel_output=fn, exc=False)

    return data


def multi_run(kwargs: Namespace) -> bool:
    """
    Checks if multiple values were sent for one argument.
    """
    return any(isinstance(v, str) and "," in v for v in kwargs.__dict__.values())


def make_configs(
    kwargs: Namespace,
    drop: Optional[Set[str]] = None,
    replace: Optional[Dict[str, str]] = None,
    last: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Creates all the configurations based on the command line arguments.
    """
    args = []
    slast = set(last) if last else set()
    for k, v in kwargs.__dict__.items():
        if drop and k in drop or k in slast:
            continue
        if replace and k in replace:
            v = replace[k]
        if isinstance(v, str):
            args.append([(k, s) for s in v.split(",")])
        else:
            args.append([(k, v)])
    if last:
        for k in last:
            if k not in kwargs.__dict__:
                continue
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
            v = v.replace("\n", " -- ").replace(",", "_")
            if len(v) > 300:
                v = v[:300]
            g[k] = v
        new_data.append(g)
    df = pandas.DataFrame(new_data)
    sorted_columns = list(sorted(df.columns))
    if "_index" in sorted_columns:
        set_cols = set(df.columns)
        addition = {"_index", "CMD", "OUTPUT", "ERROR"} & set_cols
        new_columns = []
        if "_index" in addition:
            new_columns.append("_index")
            new_columns.extend([i for i in sorted_columns if i not in addition])
            for c in ["ERROR", "OUTPUT", "CMD"]:
                if c in addition:
                    new_columns.append(c)
        sorted_columns = new_columns

    return df[sorted_columns].copy()


def measure_discrepancies(
    expected: List[Tuple["torch.Tensor", ...]],  # noqa: F821
    outputs: List[Tuple["torch.Tensor", ...]],  # noqa: F821
) -> Tuple[float, float]:
    """
    Computes the discrepancies.

    :param expected: list of outputs coming from a torch model
    :param outputs: list of outputs coming from an onnx model
    :return: max absolute errors, max relative errors
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
