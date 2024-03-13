import glob
import os
import re
import sys
import unittest
import warnings
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from timeit import Timer
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy
from numpy.testing import assert_allclose


def is_azure() -> bool:
    "Tells if the job is running on Azure DevOps."
    return os.environ.get("AZURE_HTTP_USER_AGENT", "undefined") != "undefined"


def is_windows() -> bool:
    return sys.platform == "win32"


def is_apple() -> bool:
    return sys.platform == "darwin"


def skipif_ci_windows(msg) -> Callable:
    """
    Skips a unit test if it runs on :epkg:`azure pipeline` on :epkg:`Windows`.
    """
    if is_windows() and is_azure():
        msg = f"Test does not work on azure pipeline (Windows). {msg}"
        return unittest.skip(msg)
    return lambda x: x


def skipif_ci_apple(msg) -> Callable:
    """
    Skips a unit test if it runs on :epkg:`azure pipeline` on :epkg:`Windows`.
    """
    if is_apple() and is_azure():
        msg = f"Test does not work on azure pipeline (Apple). {msg}"
        return unittest.skip(msg)
    return lambda x: x


def with_path_append(path_to_add: Union[str, List[str]]) -> Callable:
    """
    Adds a path to sys.path to check.
    """

    def wraps(f, path_to_add=path_to_add):
        def wrapped(self, path_to_add=path_to_add):
            cpy = sys.path.copy()
            if path_to_add is not None:
                if isinstance(path_to_add, str):
                    path_to_add = [path_to_add]
                sys.path.extend(path_to_add)
            f(self)
            sys.path = cpy

        return wrapped

    return wraps


def unit_test_going():
    """
    Enables a flag telling the script is running while testing it.
    Avois unit tests to be very long.
    """
    going = int(os.environ.get("UNITTEST_GOING", 0))
    return going == 1


def ignore_warnings(warns: List[Warning]) -> Callable:
    """
    Catches warnings.

    :param warns:   warnings to ignore
    """

    def wrapper(fct):
        if warns is None:
            raise AssertionError(f"warns cannot be None for '{fct}'.")

        def call_f(self):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", warns)
                return fct(self)

        return call_f

    return wrapper


def measure_time(
    stmt: Union[str, Callable],
    context: Optional[Dict[str, Any]] = None,
    repeat: int = 10,
    number: int = 50,
    warmup: int = 1,
    div_by_number: bool = True,
    max_time: Optional[float] = None,
) -> Dict[str, Union[str, int, float]]:
    """
    Measures a statement and returns the results as a dictionary.

    :param stmt: string or callable
    :param context: variable to know in a dictionary
    :param repeat: average over *repeat* experiment
    :param number: number of executions in one row
    :param warmup: number of iteration to do before starting the
        real measurement
    :param div_by_number: divide by the number of executions
    :param max_time: execute the statement until the total goes
        beyond this time (approximatively), *repeat* is ignored,
        *div_by_number* must be set to True
    :return: dictionary

    .. runpython::
        :showcode:

        from pprint import pprint
        from math import cos
        from experimental_experiment.ext_test_case import measure_time

        res = measure_time(lambda: cos(0.5))
        pprint.pprint(res)

    See `Timer.repeat <https://docs.python.org/3/library/
    timeit.html?timeit.Timer.repeat>`_
    for a better understanding of parameter *repeat* and *number*.
    The function returns a duration corresponding to
    *number* times the execution of the main statement.
    """
    if not callable(stmt) and not isinstance(stmt, str):
        raise TypeError(
            f"stmt is not callable or a string but is of type {type(stmt)!r}."
        )
    if context is None:
        context = {}

    if isinstance(stmt, str):
        tim = Timer(stmt, globals=context)
    else:
        tim = Timer(stmt)

    if warmup > 0:
        warmup_time = tim.timeit(warmup)
    else:
        warmup_time = 0

    if max_time is not None:
        if not div_by_number:
            raise ValueError(
                "div_by_number must be set to True of max_time is defined."
            )
        i = 1
        total_time = 0.0
        results = []
        while True:
            for j in (1, 2):
                number = i * j
                time_taken = tim.timeit(number)
                results.append((number, time_taken))
                total_time += time_taken
                if total_time >= max_time:
                    break
            if total_time >= max_time:
                break
            ratio = (max_time - total_time) / total_time
            ratio = max(ratio, 1)
            i = int(i * ratio)

        res = numpy.array(results)
        tw = res[:, 0].sum()
        ttime = res[:, 1].sum()
        mean = ttime / tw
        ave = res[:, 1] / res[:, 0]
        dev = (((ave - mean) ** 2 * res[:, 0]).sum() / tw) ** 0.5
        mes = dict(
            average=mean,
            deviation=dev,
            min_exec=numpy.min(ave),
            max_exec=numpy.max(ave),
            repeat=1,
            number=tw,
            ttime=ttime,
        )
    else:
        res = numpy.array(tim.repeat(repeat=repeat, number=number))
        if div_by_number:
            res /= number

        mean = numpy.mean(res)
        dev = numpy.mean(res**2)
        dev = (dev - mean**2) ** 0.5
        mes = dict(
            average=mean,
            deviation=dev,
            min_exec=numpy.min(res),
            max_exec=numpy.max(res),
            repeat=repeat,
            number=number,
            ttime=res.sum(),
        )

    if "values" in context:
        if hasattr(context["values"], "shape"):
            mes["size"] = context["values"].shape[0]
        else:
            mes["size"] = len(context["values"])
    else:
        mes["context_size"] = sys.getsizeof(context)
    mes["warmup_time"] = warmup_time
    return mes


class ExtTestCase(unittest.TestCase):
    _warns: List[Tuple[str, int, Warning]] = []

    def assertExists(self, name):
        if not os.path.exists(name):
            raise AssertionError(f"File or folder {name!r} does not exists.")

    def assertGreaterOrEqual(self, a, b, msg=None):
        if a < b:
            return AssertionError(
                f"{a} < {b}, a not greater or equal than b\n{msg or ''}"
            )

    def assertEqualArrays(
        self,
        expected: Sequence[numpy.ndarray],
        value: Sequence[numpy.ndarray],
        atol: float = 0,
        rtol: float = 0,
        msg: Optional[str] = None,
    ):
        self.assertEqual(len(expected), len(value))
        for a, b in zip(expected, value):
            self.assertEqualArray(a, b, atol=atol, rtol=rtol)

    def assertEqualArray(
        self,
        expected: numpy.ndarray,
        value: numpy.ndarray,
        atol: float = 0,
        rtol: float = 0,
        msg: Optional[str] = None,
    ):
        if hasattr(expected, "detach"):
            expected = expected.detach().cpu().numpy()
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        self.assertEqual(expected.dtype, value.dtype)
        self.assertEqual(expected.shape, value.shape)

        try:
            assert_allclose(expected, value, atol=atol, rtol=rtol)
        except AssertionError as e:
            if msg:
                raise AssertionError(msg) from e
            raise

    def assertAlmostEqual(
        self,
        expected: numpy.ndarray,
        value: numpy.ndarray,
        atol: float = 0,
        rtol: float = 0,
    ):
        if not isinstance(expected, numpy.ndarray):
            expected = numpy.array(expected)
        if not isinstance(value, numpy.ndarray):
            value = numpy.array(value).astype(expected.dtype)
        self.assertEqualArray(expected, value, atol=atol, rtol=rtol)

    def assertRaise(self, fct: Callable, exc_type: type[Exception]):
        try:
            fct()
        except exc_type as e:
            if not isinstance(e, exc_type):
                raise AssertionError(f"Unexpected exception {type(e)!r}.")
            return
        raise AssertionError("No exception was raised.")

    def assertEmpty(self, value: Any):
        if value is None:
            return
        if not value:
            return
        raise AssertionError(f"value is not empty: {value!r}.")

    def assertNotEmpty(self, value: Any):
        if value is None:
            raise AssertionError(f"value is empty: {value!r}.")
        if isinstance(value, (list, dict, tuple, set)):
            if not value:
                raise AssertionError(f"value is empty: {value!r}.")

    def assertStartsWith(self, prefix: str, full: str):
        if not full.startswith(prefix):
            raise AssertionError(f"prefix={prefix!r} does not start string  {full!r}.")

    @classmethod
    def tearDownClass(cls):
        for name, line, w in cls._warns:
            warnings.warn(f"\n{name}:{line}: {type(w)}\n  {str(w)}")

    def capture(self, fct: Callable):
        """
        Runs a function and capture standard output and error.

        :param fct: function to run
        :return: result of *fct*, output, error
        """
        sout = StringIO()
        serr = StringIO()
        with redirect_stdout(sout):
            with redirect_stderr(serr):
                try:
                    res = fct()
                except Exception as e:
                    raise AssertionError(
                        f"function {fct} failed, stdout="
                        f"\n{sout.getvalue()}\n---\nstderr=\n{serr.getvalue()}"
                    ) from e
        return res, sout.getvalue(), serr.getvalue()

    def tryCall(
        self, fct: Callable, msg: Optional[str] = None, none_if: Optional[str] = None
    ) -> Optional[Any]:
        """
        Calls the function, catch any error.

        :param fct: function to call
        :param msg: error message to display if failing
        :param none_if: returns None if this substring is found in the error message
        :return: output of *fct*
        """
        try:
            return fct()
        except Exception as e:
            if none_if is not None and none_if in str(e):
                return None
            if msg is None:
                raise e
            raise AssertionError(msg) from e


def get_figure(ax):
    """
    Returns the figure of a matplotlib figure.
    """
    if hasattr(ax, "get_figure"):
        return ax.get_figure()
    if len(ax.shape) == 0:
        return ax.get_figure()
    if len(ax.shape) == 1:
        return ax[0].get_figure()
    if len(ax.shape) == 2:
        return ax[0, 0].get_figure()
    raise RuntimeError(f"Unexpected shape {ax.shape} for axis.")


def dump_dort_onnx(fn):
    prefix = fn.__name__
    folder = "tests_dump"
    if not os.path.exists(folder):
        os.mkdir(folder)

    def wrapped(self):
        value = os.environ.get("ONNXRT_DUMP_PATH", None)
        os.environ["ONNXRT_DUMP_PATH"] = os.path.join(folder, f"{prefix}_")
        res = fn(self)
        os.environ["ONNXRT_DUMP_PATH"] = value or ""
        return res

    return wrapped


def requires_torch(version: str, msg: str) -> Callable:
    """
    Skips a unit test if torch is not recent enough.
    """
    import packaging.version as pv
    import torch

    if pv.Version(".".join(torch.__version__.split(".")[:2])) < pv.Version(version):
        msg = f"torch version {torch.__version__} < {version}: {msg}"
        return unittest.skip(msg)
    return lambda x: x


def statistics_on_file(filename: str) -> Dict[str, Union[int, float, str]]:
    """
    Computes statistics on a file.

    .. runpython::
        :showcode:

        import pprint
        from experimental_experiment.ext_test_case import statistics_on_file, __file__

        pprint.pprint(statistics_on_file(__file__))
    """
    assert os.path.exists(filename), f"File {filename!r} does not exists."

    ext = os.path.splitext(filename)[-1]
    if ext not in {".py", ".rst", ".md", ".txt"}:
        size = os.stat(filename).st_size
        return {"size": size}
    alpha = set("abcdefghijklmnopqrstuvwxyz0123456789")
    with open(filename, "r", encoding="utf-8") as f:
        n_line = 0
        n_ch = 0
        for line in f.readlines():
            s = line.strip("\n\r\t ")
            if s:
                n_ch += len(s.replace(" ", ""))
                ch = set(s.lower()) & alpha
                if ch:
                    # It avoid counting line with only a bracket, a comma.
                    n_line += 1

    stat = dict(lines=n_line, chars=n_ch, ext=ext)
    if ext != ".py":
        return stat
    # add statistics on python syntax?
    return stat


def statistics_on_folder(
    folder: Union[str, List[str]],
    pattern: str = ".*[.]((py|rst))$",
    aggregation: int = 0,
) -> List[Dict[str, Union[int, float, str]]]:
    """
    Computes statistics on files in a folder.

    :param folder: folder or folders to investigate
    :param pattern: file pattern
    :param aggregation: show the first subfolders
    :return: list of dictionaries

    .. runpython::
        :showcode:

        import os
        import pprint
        from experimental_experiment.ext_test_case import statistics_on_folder, __file__

        pprint.pprint(statistics_on_folder(os.path.dirname(__file__)))
    """
    if isinstance(folder, list):
        rows = []
        for fo in folder:
            last = fo.replace("\\", "/").split("/")[-1]
            r = statistics_on_folder(
                fo, pattern=pattern, aggregation=max(aggregation - 1, 0)
            )
            if aggregation == 0:
                rows.extend(r)
                continue
            for line in r:
                line["dir"] = os.path.join(last, line["dir"])
            rows.extend(r)
        return rows

    rows = []
    reg = re.compile(pattern)
    for name in glob.glob("**/*", root_dir=folder, recursive=True):
        if not reg.match(name):
            continue
        if os.path.isdir(os.path.join(folder, name)):
            continue
        n = name.replace("\\", "/")
        spl = n.split("/")
        level = len(spl)
        stat = statistics_on_file(os.path.join(folder, name))
        stat["name"] = name
        if aggregation <= 0:
            rows.append(stat)
            continue
        spl = os.path.dirname(name).replace("\\", "/").split("/")
        level = "/".join(spl[:aggregation])
        stat["dir"] = level
        rows.append(stat)
    return rows
