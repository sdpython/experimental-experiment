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
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union


_DEFAULT_STRING_LIMIT = 2000


class BenchmarkError(RuntimeError):
    pass


def _clean_string(s: str) -> str:
    cleaned = [c for c in s if 32 <= ord(c) < 127 and c not in {","}]
    return "".join(cleaned)


def get_processor_name():
    """Returns the processor name."""
    if platform.system() in ("Windows", "Darwin"):
        return platform.processor()
    if platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        all_info = subprocess.check_output(command, shell=True).decode().strip()
        for line in all_info.split("\n"):
            if "model name" in line:
                return re.sub(".*model name.*:", "", line, count=1, flags=0).strip()
    # fails
    # if platform.system() == "Darwin":
    #     os.environ["PATH"] = os.environ["PATH"] + os.pathsep + "/usr/sbin"
    #     command = "sysctl -n machdep.cpu.brand_string"
    #     return subprocess.check_output(command).strip()

    raise AssertionError("get_process_name not implemented on this platform.")


def get_machine(
    capability_as_str: bool = True,
) -> Dict[str, Union[str, int, float, Tuple[int, int]]]:
    """Returns the machine specifications."""
    arch = platform.architecture()
    config: Dict[str, Union[str, int, float, Tuple[int, int]]] = dict(
        machine=str(platform.machine()),
        architecture=(
            "/".join(str(_) for _ in arch) if isinstance(arch, (list, tuple)) else str(arch)
        ),
        processor=str(platform.processor()),
        version=str(sys.version).split()[0],
        cpu=int(multiprocessing.cpu_count()),
        executable=str(sys.executable),
        processor_name=get_processor_name(),
        system=str(platform.system()),
    )
    try:
        import torch.cuda
    except ImportError:
        return config

    config["has_cuda"] = bool(torch.cuda.device_count() > 0)
    if config["has_cuda"]:
        config["capability"] = (
            ".".join(map(str, torch.cuda.get_device_capability(0)))
            if capability_as_str
            else torch.cuda.get_device_capability(0)
        )
        config["device_name"] = str(torch.cuda.get_device_name(0))
    return config


def _cmd_line(script_name: str, **kwargs: Dict[str, Union[str, int, float]]) -> List[str]:
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
                "time_latency_t_qu_10t",
                "time_latency_eager_t_detail",
                "time_latency_eager_t_qu",
                "time_latency_eager_t_qu_10t",
            }
            or len(w) < 500
        ):
            warnings.warn(
                f"Unexpected long value for model={kw.get('model_name', '?')}, "
                f"k={k!r}, value has length {len(w)} is\n{w}",
                stacklevel=2,
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
    timeout: int = 600,
    missing: Optional[Dict[str, Union[str, Callable]]] = None,
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
    :param timeout: timeout for the subprocesses
    :param missing: populate with this missing value if not found
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
        if hasattr(loop, "set_description"):
            for c in ["name", "model"]:
                if c not in config:
                    continue
                loop.set_description(f"[{config[c]}]")
                break
        cmd = _cmd_line(script_name, **config)
        begin = time.perf_counter()

        if dump:
            os.environ["ONNXRT_DUMP_PATH"] = _make_prefix(script_name, iter_loop)
        else:
            os.environ["ONNXRT_DUMP_PATH"] = ""
        if verbose > 3:
            print(f"[run_benchmark] cmd={cmd if isinstance(cmd, str) else ' '.join(cmd)}")
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        timeout_error = ""
        try:
            res = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as e:
            # see https://docs.python.org/3/library/subprocess.html#subprocess.Popen.communicate
            timeout_error = str(e)
            if verbose:
                print(f"[run_benchmark] timeout {e} for cmd={cmd}")
            p.terminate()
            try:
                # Use communicate with a timeout to prevent hanging
                res = p.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                # Force kill if terminate doesn't work
                if verbose:
                    print(f"[run_benchmark] force killing cmd={cmd}")
                p.kill()
                res = p.communicate()
        out, err = res
        sout = out.decode("utf-8", errors="ignore")
        serr = err.decode("utf-8", errors="ignore")

        if dump_std:
            if dump_std and not os.path.exists(dump_std):
                os.makedirs(dump_std)
            root = os.path.split(script_name)[-1].split(".")[-1]
            filename = os.path.join(dump_std, f"{root}.{iter_loop}")
            filename_out = f"{filename}.stdout"
            filename_err = f"{filename}.stderr"
            if out.strip(b"\n \r\t"):
                with open(filename_out, "w") as f:
                    f.write(sout)
            if err.strip(b"\n \r\t"):
                with open(filename_err, "w") as f:
                    f.write(serr)
        else:
            filename_out, filename_err = None, None

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
        if filename_out and os.path.exists(filename_out):
            if "model_name" in metrics:
                new_name = f"{filename_out}.{_clean_string(metrics['model_name'])}"
                os.rename(filename_out, new_name)
                filename_out = new_name
            metrics["file.stdout"] = filename_out
        if filename_err and os.path.exists(filename_err):
            if "model_name" in metrics:
                new_name = f"{filename_err}.{_clean_string(metrics['model_name'])}"
                os.rename(filename_err, new_name)
                filename_err = new_name
            metrics["file.stderr"] = filename_err
        metrics["DATE"] = f"{datetime.now():%Y-%m-%d}"
        metrics["ITER"] = iter_loop
        metrics["TIME_ITER"] = time.perf_counter() - begin
        metrics["ERROR"] = _clean_string(serr)
        metrics["ERR_stdout"] = _clean_string(sout)
        if metrics["ERROR"]:
            metrics["ERR_std"] = metrics["ERROR"]
            if "CUDA out of memory" in metrics["ERROR"]:
                metrics["ERR_CUDA_OOM"] = 1
            if "Cannot access gated repo for url" in metrics["ERROR"]:
                metrics["ERR_ACCESS"] = 1
        if timeout_error:
            metrics["ERR_timeout"] = _clean_string(timeout_error)
        metrics["OUTPUT"] = _clean_string(sout)
        for k, v in config.items():
            metrics[f"config_{k}"] = str(v).replace("\n", " ")
        if missing:
            update_missing = {}
            for k, v in missing.items():
                if k not in metrics:
                    if isinstance(v, str):
                        update_missing[k] = v
                        continue
                    if callable(v):
                        update_missing.update(v(missing, config))
                        continue
                    raise AssertionError(
                        f"Unable to interpret {type(v)} for k={k!r}, config={config!r}"
                    )
            if update_missing:
                metrics.update(update_missing)
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
            fold, _ = os.path.split(temp_output_data)
            # fold could be empty string
            if fold and not os.path.exists(fold):
                os.makedirs(fold)
            df.to_csv(temp_output_data, index=False, errors="ignore")
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
    """Checks if multiple values were sent for one argument."""
    return any(isinstance(v, str) and "," in v for v in kwargs.__dict__.values())


def make_configs(
    kwargs: Union[Namespace, Dict[str, Any]],
    drop: Optional[Set[str]] = None,
    replace: Optional[Dict[str, str]] = None,
    last: Optional[List[str]] = None,
    filter_function: Optional[Callable[Dict[str, Any], bool]] = None,
) -> List[Dict[str, Any]]:
    """
    Creates all the configurations based on the command line arguments.

    :param kwargs: parameters the command line,
        every value having a comma means multiple values,
        it multiplies the number of configurations to try by the number of comma
        separated values
    :param drop: keys to drop in kwargs if specified
    :param replace: values to replace for a particular key
    :param last: to change the order of the loop created the configuration,
        if ``last == ["part"]`` and ``kwargs[part] == "0,1"``,
        then configuration where ``part==0`` is always followed by a configuration
        having ``part==1``
    :param filter_function: function taking a configuration and returning True
        if it is must be kept
    :return: list of configurations
    """
    kwargs_ = kwargs if isinstance(kwargs, dict) else kwargs.__dict__
    args = []
    slast = set(last) if last else set()
    for k, v in kwargs_.items():
        if (drop and k in drop) or k in slast:
            continue
        if replace and k in replace:
            v = replace[k]
        if isinstance(v, str):
            args.append([(k, s) for s in v.split(",")])
        else:
            args.append([(k, v)])
    if last:
        for k in last:
            if k not in kwargs_:
                continue
            v = kwargs[k]
            if isinstance(v, str):
                args.append([(k, s) for s in v.split(",")])
            else:
                args.append([(k, v)])

    configs = list(itertools.product(*args))
    confs = [dict(c) for c in configs]
    if filter_function:
        confs = [c for c in confs if filter_function(c)]
    return confs


def make_dataframe_from_benchmark_data(
    data: List[Dict], detailed: bool = True, string_limit: int = _DEFAULT_STRING_LIMIT
) -> Any:
    """
    Creates a dataframe from the received data.

    :param data: list of dictionaries for every run
    :param detailed: remove multi line and long values
    :param string_limit: truncate the strings
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
            if len(v) > string_limit:
                v = v[:string_limit] + "..."
            g[k] = v
        new_data.append(g)
    df = pandas.DataFrame(new_data)
    sorted_columns = sorted(df.columns)
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
