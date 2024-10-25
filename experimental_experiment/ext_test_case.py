"""
The module contains the main class ``ExtTestCase`` which adds
specific functionalities to this project.
"""

import glob
import importlib
import logging
import os
import re
import sys
import unittest
import warnings
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from timeit import Timer
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy
from numpy.testing import assert_allclose

BOOLEAN_VALUES = (1, "1", True, "True", "true", "TRUE")


def is_azure() -> bool:
    """Tells if the job is running on Azure DevOps."""
    return os.environ.get("AZURE_HTTP_USER_AGENT", "undefined") != "undefined"


def is_windows() -> bool:
    return sys.platform == "win32"


def is_apple() -> bool:
    return sys.platform == "darwin"


def is_linux() -> bool:
    return sys.platform == "linux"


def skipif_transformers(version_to_skip: Union[str, Set[str]], msg: str) -> Callable:
    """Skips a unit test if transformers has a specific version."""
    if isinstance(version_to_skip, str):
        version_to_skip = {version_to_skip}
    import transformers

    if transformers.__version__ in version_to_skip:
        msg = f"Unstable test. {msg}"
        return unittest.skip(msg)
    return lambda x: x


def skipif_not_onnxrt(msg) -> Callable:
    """Skips a unit test if it runs on :epkg:`azure pipeline` on :epkg:`Windows`."""
    UNITTEST_ONNXRT = os.environ.get("UNITTEST_ONNXRT", "0")
    value = int(UNITTEST_ONNXRT)
    if not value:
        msg = f"Set UNITTEST_ONNXRT=1 to run the unittest. {msg}"
        return unittest.skip(msg)
    return lambda x: x


def skipif_ci_windows(msg) -> Callable:
    """Skips a unit test if it runs on :epkg:`azure pipeline` on :epkg:`Windows`."""
    if is_windows() and is_azure():
        msg = f"Test does not work on azure pipeline (Windows). {msg}"
        return unittest.skip(msg)
    return lambda x: x


def skipif_ci_linux(msg) -> Callable:
    """Skips a unit test if it runs on :epkg:`azure pipeline` on :epkg:`Linux`."""
    if is_linux() and is_azure():
        msg = f"Takes too long (Linux). {msg}"
        return unittest.skip(msg)
    return lambda x: x


def skipif_ci_apple(msg) -> Callable:
    """Skips a unit test if it runs on :epkg:`azure pipeline` on :epkg:`Windows`."""
    if is_apple() and is_azure():
        msg = f"Test does not work on azure pipeline (Apple). {msg}"
        return unittest.skip(msg)
    return lambda x: x


def with_path_append(path_to_add: Union[str, List[str]]) -> Callable:
    """Adds a path to sys.path to check."""

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
    if not isinstance(warns, (tuple, list)):
        warns = (warns,)
    new_list = []
    for w in warns:
        if w == "TracerWarning":
            from torch.jit import TracerWarning

            new_list.append(TracerWarning)
        else:
            new_list.append(w)
    warns = tuple(new_list)

    def wrapper(fct):
        if warns is None:
            raise AssertionError(f"warns cannot be None for '{fct}'.")

        def call_f(self):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", warns)
                return fct(self)

        return call_f

    return wrapper


def hide_stdout(f: Optional[Callable] = None) -> Callable:
    """
    Catches warnings.

    :param f: the function is called with the stdout as an argument
    """

    def wrapper(fct):
        def call_f(self):
            st = StringIO()
            with redirect_stdout(st), warnings.catch_warnings():
                warnings.simplefilter("ignore", (UserWarning, DeprecationWarning))
                try:
                    return fct(self)
                except AssertionError as e:
                    if "torch is not recent enough, file" in str(e):
                        raise unittest.SkipTest(str(e))  # noqa: B904
                    raise
            if f is not None:
                f(st.getvalue())

        return call_f

    return wrapper


def long_test(msg: str = "") -> Callable:
    """
    Catches warnings.

    :param f: the function is called with the stdout as an argument
    """

    """Skips a unit test if it runs on :epkg:`azure pipeline` on :epkg:`Windows`."""
    if os.environ.get("LONGTEST", "0") in ("0", 0, False, "False", "false"):
        msg = f"Skipped (set LONGTEST=1 to run it. {msg}"
        return unittest.skip(msg)
    return lambda x: x


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
        pprint(res)

    See `Timer.repeat <https://docs.python.org/3/library/
    timeit.html?timeit.Timer.repeat>`_
    for a better understanding of parameter *repeat* and *number*.
    The function returns a duration corresponding to
    *number* times the execution of the main statement.
    """
    if not callable(stmt) and not isinstance(stmt, str):
        raise TypeError(f"stmt is not callable or a string but is of type {type(stmt)!r}.")
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
            raise ValueError("div_by_number must be set to True of max_time is defined.")
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
    """
    Inherits from :class:`unittest.TestCase` and adds specific comprison
    functions and other helpers.
    """

    _warns: List[Tuple[str, int, Warning]] = []

    @classmethod
    def setUpClass(cls):
        logger = logging.getLogger("onnxscript.optimizer.constant_folding")
        logger.setLevel(logging.ERROR)
        unittest.TestCase.setUpClass()

    def print_model(self, model: "ModelProto"):  # noqa: F821
        from onnx_array_api.plotting.text_plot import onnx_simple_text_plot

        print(onnx_simple_text_plot(model))

    def get_dump_file(self, name: str, folder: Optional[str] = None) -> str:
        """Returns a filename to dump a model."""
        if folder is None:
            folder = "dump_test"
        if folder and not os.path.exists(folder):
            os.mkdir(folder)
        return os.path.join(folder, name)

    def dump_onnx(
        self,
        name: str,
        proto: Any,
        folder: Optional[str] = None,
    ) -> str:
        """Dumps an onnx file."""
        fullname = self.get_dump_file(name, folder=folder)
        with open(fullname, "wb") as f:
            f.write(proto.SerializeToString())
        return fullname

    def assertExists(self, name):
        """Checks the existing of a file."""
        if not os.path.exists(name):
            raise AssertionError(f"File or folder {name!r} does not exists.")

    def assertGreaterOrEqual(self, a, b, msg=None):
        """In the name"""
        if a < b:
            return AssertionError(f"{a} < {b}, a not greater or equal than b\n{msg or ''}")

    def assertInOr(self, tofind: Tuple[str, ...], text: str):
        for tof in tofind:
            if tof in text:
                return
        raise AssertionError(
            f"Unable to find one string in the list {tofind!r} in\n--\n{text}"
        )

    def assertIn(self, tofind: str, text: str):
        if tofind in text:
            return
        raise AssertionError(f"Unable to find the list of strings {tofind!r} in\n--\n{text}")

    def assertEqualArrays(
        self,
        expected: Sequence[numpy.ndarray],
        value: Sequence[numpy.ndarray],
        atol: float = 0,
        rtol: float = 0,
        msg: Optional[str] = None,
    ):
        """In the name"""
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
        """In the name"""
        if hasattr(expected, "detach") and hasattr(value, "detach"):
            if msg:
                try:
                    self.assertEqual(expected.dtype, value.dtype)
                except AssertionError as e:
                    raise AssertionError(msg) from e
                try:
                    self.assertEqual(expected.shape, value.shape)
                except AssertionError as e:
                    raise AssertionError(msg) from e
            else:
                self.assertEqual(expected.dtype, value.dtype)
                self.assertEqual(expected.shape, value.shape)

            import torch

            try:
                torch.testing.assert_close(value, expected, atol=atol, rtol=rtol)
            except AssertionError as e:
                if msg:
                    raise AssertionError(msg) from e
                raise
            return

        if hasattr(expected, "detach"):
            expected = expected.detach().cpu().numpy()
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        if msg:
            try:
                self.assertEqual(expected.dtype, value.dtype)
            except AssertionError as e:
                raise AssertionError(msg) from e
            try:
                self.assertEqual(expected.shape, value.shape)
            except AssertionError as e:
                raise AssertionError(msg) from e
        else:
            self.assertEqual(expected.dtype, value.dtype)
            self.assertEqual(expected.shape, value.shape)

        try:
            assert_allclose(desired=expected, actual=value, atol=atol, rtol=rtol)
        except AssertionError as e:
            if msg:
                raise AssertionError(msg) from e
            raise

    def assertEqual(self, expected: Any, value: Any, msg: str = ""):
        "Overwrites the error message to get a more explicit message about what is what."
        if msg:
            super().assertEqual(expected, value, msg)
        else:
            try:
                super().assertEqual(expected, value)
            except AssertionError as e:
                raise AssertionError(  # noqa: B904
                    f"expected is {expected!r}, value is {value!r}\n{e}"
                )

    def assertAlmostEqual(
        self,
        expected: numpy.ndarray,
        value: numpy.ndarray,
        atol: float = 0,
        rtol: float = 0,
    ):
        """In the name"""
        if not isinstance(expected, numpy.ndarray):
            expected = numpy.array(expected)
        if not isinstance(value, numpy.ndarray):
            value = numpy.array(value).astype(expected.dtype)
        self.assertEqualArray(expected, value, atol=atol, rtol=rtol)

    def assertRaise(self, fct: Callable, exc_type: type[Exception]):
        """In the name"""
        try:
            fct()
        except exc_type as e:
            if not isinstance(e, exc_type):
                raise AssertionError(f"Unexpected exception {type(e)!r}.")  # noqa: B904
            return
        raise AssertionError("No exception was raised.")  # noqa: B904

    def assertEmpty(self, value: Any):
        """In the name"""
        if value is None:
            return
        if not value:
            return
        raise AssertionError(f"value is not empty: {value!r}.")

    def assertNotEmpty(self, value: Any):
        """In the name"""
        if value is None:
            raise AssertionError(f"value is empty: {value!r}.")
        if isinstance(value, (list, dict, tuple, set)):
            if not value:
                raise AssertionError(f"value is empty: {value!r}.")

    def assertStartsWith(self, prefix: str, full: str):
        """In the name"""
        if not full.startswith(prefix):
            raise AssertionError(f"prefix={prefix!r} does not start string  {full!r}.")

    @classmethod
    def tearDownClass(cls):
        for name, line, w in cls._warns:
            warnings.warn(f"\n{name}:{line}: {type(w)}\n  {w!s}", stacklevel=2)

    def capture(self, fct: Callable):
        """
        Runs a function and capture standard output and error.

        :param fct: function to run
        :return: result of *fct*, output, error
        """
        sout = StringIO()
        serr = StringIO()
        with redirect_stdout(sout), redirect_stderr(serr):
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
                raise
            raise AssertionError(msg) from e


def get_figure(ax):
    """Returns the figure of a matplotlib figure."""
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
    """Context manager to dump onnx model created by dort."""
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


def has_cuda() -> bool:
    """Returns ``torch.cuda.is_available()``."""
    import torch

    return torch.cuda.is_available()


def requires_cuda(msg: str = "", version: str = "", memory: int = 0):
    """
    Skips a test if cuda is not available.

    :param msg: to overwrite the message
    :param version: minimum version
    :param memory: minimun number of Gb to run the test
    """
    import torch

    if not torch.cuda.is_available():
        msg = msg or "only runs on CUDA but torch does not have it"
        return unittest.skip(msg or "cuda not installed")
    if version:
        import packaging.versions as pv

        if pv.Version(torch.version.cuda) < pv.Version(version):
            msg = msg or f"CUDA older than {version}"
        return unittest.skip(msg or f"cuda not recent enough {torch.version.cuda} < {version}")

    if memory:
        m = torch.cuda.get_device_properties(0).total_memory / 2**30
        if m < memory:
            msg = msg or f"available memory is not enough {m} < {memory} (Gb)"
            return unittest.skip(msg)

    return lambda x: x


def requires_zoo(msg: str = "") -> Callable:
    """Skips a unit test if environment variable ZOO is not equal to 1."""
    var = os.environ.get("ZOO", "0") in BOOLEAN_VALUES

    if not var:
        msg = f"ZOO not set up or != 1. {msg}"
        return unittest.skip(msg or "zoo not installed")
    return lambda x: x


def has_executorch(version: str = "", msg: str = "") -> Callable:
    """Tells if :epkg:`ExecuTorch` is installed."""
    if not version:
        return importlib.util.find_spec("executorch")

    import packaging.version as pv
    import executorch

    return pv.Version(".".join(executorch.__version__.split(".")[:2])) < pv.Version(version)


def requires_sklearn(version: str, msg: str = "") -> Callable:
    """Skips a unit test if :epkg:`scikit-learn` is not recent enough."""
    import packaging.version as pv
    import sklearn

    if pv.Version(".".join(sklearn.__version__.split(".")[:2])) < pv.Version(version):
        msg = f"scikit-learn version {sklearn.__version__} < {version}: {msg}"
        return unittest.skip(msg)
    return lambda x: x


def requires_torch(version: str, msg: str = "") -> Callable:
    """Skips a unit test if :epkg:`pytorch` is not recent enough."""
    import packaging.version as pv
    import torch

    if pv.Version(".".join(torch.__version__.split(".")[:2])) < pv.Version(version):
        msg = f"torch version {torch.__version__} < {version}: {msg}"
        return unittest.skip(msg)
    return lambda x: x


def requires_executorch(version: str, msg: str = "") -> Callable:
    """Skips a unit test if :epkg:`executorch` is not recent enough."""
    if not has_executorch():
        msg = f"executorch is not installed: {msg}"
        return unittest.skip(msg)

    import packaging.version as pv
    import executorch

    if hasattr(executorch, "__version__") and pv.Version(
        ".".join(executorch.__version__.split(".")[:2])
    ) < pv.Version(version):
        msg = f"torch version {executorch.__version__} < {version}: {msg}"
        return unittest.skip(msg)
    return lambda x: x


def requires_monai(version: str = "", msg: str = "") -> Callable:
    """Skips a unit test if :epkg:`monai` is not recent enough."""
    import packaging.version as pv

    try:
        import monai
    except ImportError:
        return unittest.skip(msg or "monai not installed")

    if version and pv.Version(".".join(monai.__version__.split(".")[:2])) < pv.Version(
        version
    ):
        return unittest.skip(f"monai version {monai.__version__} < {version}: {msg}")
    return lambda x: x


def requires_vocos(version: str = "", msg: str = "") -> Callable:
    """Skips a unit test if :epkg:`vocos` is not recent enough."""
    import packaging.version as pv

    try:
        import vocos
    except ImportError:
        return unittest.skip(msg or "vocos not installed")

    if version and pv.Version(".".join(vocos.__version__.split(".")[:2])) < pv.Version(
        version
    ):
        msg = f"vocos version {vocos.__version__} < {version}: {msg}"
        return unittest.skip(msg)
    return lambda x: x


def requires_pyinstrument(version: str = "", msg: str = "") -> Callable:
    """Skips a unit test if :epkg:`pyinstrument` is not recent enough."""
    import packaging.version as pv

    try:
        import pyinstrument
    except ImportError:
        return unittest.skip(msg or "pyinstrument is not installed")

    if version and pv.Version(".".join(pyinstrument.__version__.split(".")[:2])) < pv.Version(
        version
    ):
        msg = f"torch version {pyinstrument.__version__} < {version}: {msg}"
        return unittest.skip(msg)
    return lambda x: x


def requires_numpy(version: str, msg: str = "") -> Callable:
    """Skips a unit test if :epkg:`numpy` is not recent enough."""
    import packaging.version as pv
    import numpy

    if pv.Version(".".join(numpy.__version__.split(".")[:2])) < pv.Version(version):
        msg = f"numpy version {numpy.__version__} < {version}: {msg}"
        return unittest.skip(msg)
    return lambda x: x


def requires_transformers(
    version: str, msg: str = "", or_older_than: Optional[str] = None
) -> Callable:
    """Skips a unit test if :epkg:`transformers` is not recent enough."""
    import packaging.version as pv

    try:
        import transformers
    except ImportError:
        msg = f"diffusers not installed {msg}"
        return unittest.skip(msg)

    v = pv.Version(".".join(transformers.__version__.split(".")[:2]))
    if v < pv.Version(version):
        msg = f"transformers version {transformers.__version__} < {version} {msg}"
        return unittest.skip(msg)
    if or_older_than and v > pv.Version(or_older_than):
        msg = (
            f"transformers version {or_older_than} < "
            f"{transformers.__version__} < {version} {msg}"
        )
        return unittest.skip(msg)
    return lambda x: x


def require_diffusers(
    version: str, msg: str = "", or_older_than: Optional[str] = None
) -> Callable:
    """Skips a unit test if :epkg:`transformers` is not recent enough."""
    import packaging.version as pv

    try:
        import diffusers
    except ImportError:
        msg = f"diffusers not installed {msg}"
        return unittest.skip(msg)

    v = pv.Version(".".join(diffusers.__version__.split(".")[:2]))
    if v < pv.Version(version):
        msg = f"diffusers version {diffusers.__version__} < {version} {msg}"
        return unittest.skip(msg)
    if or_older_than and v > pv.Version(or_older_than):
        msg = (
            f"diffusers version {or_older_than} < "
            f"{diffusers.__version__} < {version} {msg}"
        )
        return unittest.skip(msg)
    return lambda x: x


def requires_onnxscript(version: str, msg: str = "") -> Callable:
    """Skips a unit test if :epkg:`onnxscript` is not recent enough."""
    import packaging.version as pv
    import onnxscript

    if not hasattr(onnxscript, "__version__"):
        # development version
        return lambda x: x

    if pv.Version(".".join(onnxscript.__version__.split(".")[:2])) < pv.Version(version):
        msg = f"onnxscript version {onnxscript.__version__} < {version}: {msg}"
        return unittest.skip(msg)
    return lambda x: x


def requires_onnxruntime(version: str, msg: str = "") -> Callable:
    """Skips a unit test if :epkg:`onnxruntime` is not recent enough."""
    import packaging.version as pv
    import onnxruntime

    if pv.Version(".".join(onnxruntime.__version__.split(".")[:2])) < pv.Version(version):
        msg = f"onnxruntime version {onnxruntime.__version__} < {version}: {msg}"
        return unittest.skip(msg)
    return lambda x: x


def has_onnxruntime_training(push_back_batch: bool = False):
    """Tells if onnxruntime_training is installed."""
    try:
        from onnxruntime import training
    except ImportError:
        # onnxruntime not training
        training = None
    if training is None:
        return False

    if push_back_batch:
        try:
            from onnxruntime.capi.onnxruntime_pybind11_state import OrtValueVector
        except ImportError:
            return False

        if not hasattr(OrtValueVector, "push_back_batch"):
            return False
    return True


def requires_onnxruntime_training(push_back_batch: bool = False, msg: str = "") -> Callable:
    """Skips a unit test if :epkg:`onnxruntime` is not onnxruntime_training."""
    try:
        from onnxruntime import training
    except ImportError:
        # onnxruntime not training
        training = None
    if training is None:
        msg = msg or "onnxruntime_training is not installed"
        return unittest.skip(msg)

    if push_back_batch:
        try:
            from onnxruntime.capi.onnxruntime_pybind11_state import OrtValueVector
        except ImportError:
            msg = msg or "OrtValue has no method push_back_batch"
            return unittest.skip(msg)

        if not hasattr(OrtValueVector, "push_back_batch"):
            msg = msg or "OrtValue has no method push_back_batch"
            return unittest.skip(msg)
    return lambda x: x


def requires_onnx(version: str, msg: str = "") -> Callable:
    """Skips a unit test if :epkg:`onnx` is not recent enough."""
    import packaging.version as pv
    import onnx

    if pv.Version(".".join(onnx.__version__.split(".")[:2])) < pv.Version(version):
        msg = f"onnx version {onnx.__version__} < {version}: {msg}"
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
            r = statistics_on_folder(fo, pattern=pattern, aggregation=max(aggregation - 1, 0))
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
