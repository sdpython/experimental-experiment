import unittest
import os
import sys
import packaging.version as pv
import importlib.util
import subprocess
import time
from experimental_experiment import __file__ as experimental_experiment_file
from experimental_experiment.ext_test_case import (
    ExtTestCase,
    is_windows,
    is_apple,
    has_onnxruntime_training,
    has_executorch,
)

VERBOSE = 0
ROOT = os.path.realpath(os.path.abspath(os.path.join(experimental_experiment_file, "..", "..")))


def import_source(module_file_path, module_name):
    if not os.path.exists(module_file_path):
        raise FileNotFoundError(module_file_path)
    module_spec = importlib.util.spec_from_file_location(module_name, module_file_path)
    if module_spec is None:
        raise FileNotFoundError(
            "Unable to find '{}' in '{}'.".format(module_name, module_file_path)
        )
    module = importlib.util.module_from_spec(module_spec)
    return module_spec.loader.exec_module(module)


class TestDocumentationExamples(ExtTestCase):
    def run_test(self, fold: str, name: str, verbose=0) -> int:
        ppath = os.environ.get("PYTHONPATH", "")
        if not ppath:
            os.environ["PYTHONPATH"] = ROOT
        elif ROOT not in ppath:
            sep = ";" if is_windows() else ":"
            os.environ["PYTHONPATH"] = ppath + sep + ROOT
        perf = time.perf_counter()
        try:
            mod = import_source(fold, os.path.splitext(name)[0])
            assert mod is not None
        except FileNotFoundError:
            # try another way
            cmds = [sys.executable, "-u", os.path.join(fold, name)]
            p = subprocess.Popen(cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            res = p.communicate()
            _out, err = res
            st = err.decode("ascii", errors="ignore")
            if st and "Traceback" in st:
                if '"dot" not found in path.' in st:
                    # dot not installed, this part
                    # is tested in onnx framework
                    if verbose:
                        print(f"failed: {name!r} due to missing dot.")
                    return 0
                raise AssertionError(  # noqa: B904
                    "Example '{}' (cmd: {} - exec_prefix='{}') "
                    "failed due to\n{}"
                    "".format(name, cmds, sys.exec_prefix, st)
                )
        dt = time.perf_counter() - perf
        if verbose:
            print(f"{dt:.3f}: run {name!r}")
        return 1

    @classmethod
    def add_test_methods(cls):
        import torch

        this = os.path.abspath(os.path.dirname(__file__))
        fold = os.path.normpath(os.path.join(this, "..", "..", "_doc", "examples"))
        found = os.listdir(fold)
        for name in found:
            if not name.endswith(".py") or not name.startswith("plot_"):
                continue
            reason = None

            # Windows, Apple

            if not reason and (
                name.startswith("plot_exporter_")
                or name
                in {
                    "plot_llama_diff_dort_301.py",
                    "plot_llama_diff_export_301.py",
                    "plot_profile_existing_onnx_101.py",
                    "plot_torch_export_201.py",
                    "plot_torch_export_with_dyamic_cache_201",
                }
            ):
                if sys.platform == "win32":
                    reason = "does not work on windows"

            if not reason and name in {"plot_llama_bench_102.py"}:
                if sys.platform in {"darwin"}:
                    reason = "apple not supported"

            # too long

            if not reason and name in {
                "plot_llama_bench_102.py",
                "plot_torch_custom_backend_101.py",
                "plot_custom_backend_llama_102.py",
            }:
                if sys.platform in {"win32"}:
                    # dynamo not supported on windows
                    reason = "dynamo not supported on windows"
                if sys.platform in {"darwin"}:
                    # dynamo not supported on windows
                    reason = "onnxruntime-training not available"
                if sys.platform == "linux" and name in {
                    "plot_llama_bench_102.py",
                    "plot_custom_backend_llama_102.py",
                }:
                    reason = "too long"

            if not reason and is_apple():
                reason = "last for ever"

            if not reason and name in {
                "plot_torch_aot_201.py",
                "plot_torch_dort_201.py",
            }:
                # too long
                reason = "not working yet or too long"

            # missing

            if not reason and is_apple() and name in {"plot_convolutation_matmul_102.py"}:
                reason = "dot is missing"

            if not reason and name in {
                "plot_convolutation_matmul_102.py",
                "plot_optimize_101.py",
                "plot_torch_linreg_101.py",
                "plot_rewrite_101.py",
            }:
                if sys.platform in {"win32", "darwin"}:
                    # dynamo not supported on windows
                    reason = "graphviz not installed"

            if not reason and name in {
                "plot_torch_custom_backend_101.py",
                "lot_llama_bench_102.py",
                "plot_custom_backend_llama_102.py",
            }:
                if not has_onnxruntime_training(True):
                    reason = "OrtValueVector.push_back_batch is missing (onnxruntime)"

            if not reason and name in {"plot_convolutation_matmul_102.py"}:
                if not has_onnxruntime_training(True):
                    reason = "OrtModuleGraphBuilder is missing (onnxruntime)"

            if not reason and name in {"plot_executorch_102.py"}:
                if not has_executorch():
                    reason = "executorch is not installed"

            # version

            if pv.Version(torch.__version__) < pv.Version("2.5"):
                reason = "too long, pytorch < 2.5"

            if not reason and name in {
                "plot_torch_export_201.py",
                "plot_llama_diff_export_301.py",
            }:
                from torch import __version__ as tv
                from onnx_array_api import __version__ as toaa

                if pv.Version(".".join(tv.split(".")[:2])) < pv.Version("2.5"):
                    reason = "requires torch 2.5"
                if pv.Version(".".join(toaa.split(".")[:2])) < pv.Version("0.3"):
                    reason = "requires onnx-array-api 0.3"

            if not reason and (
                name.startswith("plot_exporter_")
                or name
                in {
                    "plot_torch_export_101.py",
                    "plot_torch_export_compile_102.py",
                    "plot_llama_diff_export_301.py",
                }
            ):
                if pv.Version(torch.__version__) < pv.Version("2.6"):
                    reason = "requires torch 2.6"

            if name in {"plot_torch_linreg_101.py"}:
                if pv.Version(torch.__version__) < pv.Version("2.6"):
                    reason = "requires torch 2.6"

            if not reason and (
                name.startswith("plot_exporter_recipes_oe_")
                or name == "plot_torch_export_with_dyamic_cache_201.py"
            ):
                if pv.Version(torch.__version__) < pv.Version("2.7"):
                    reason = "requires torch 2.7"

            if not reason and name in {"plot_torch_sklearn_201.py", "plot_model_to_python.py"}:
                import onnx_array_api

                if pv.Version(onnx_array_api.__version__) < pv.Version("0.3.1"):
                    reason = "requires onnx_array_api>=0.3.1"

            if not reason and name in {"plot_torch_sklearn_201.py"}:
                if pv.Version(torch.__version__) < pv.Version("2.9"):
                    reason = "requires torch>=2.9"

            if not reason and name in {"plot_torch_export_201.py"}:
                try:
                    import onnx_array_api

                    if pv.Version(onnx_array_api.__version__) < pv.Version("0.3.2"):
                        reason = "requires onnx-array-api>=3.2"
                except ImportError:
                    reason = "missing onnx-array-pi"

            if (
                not reason
                and not has_onnxruntime_training()
                and name in {"plot_llama_diff_dort_301.py"}
            ):
                reason = "onnxruntime-training is missing"

            if reason:

                @unittest.skip(reason)
                def _test_(self, name=name):
                    res = self.run_test(fold, name, verbose=VERBOSE)
                    self.assertTrue(res)

            else:

                def _test_(self, name=name):
                    res = self.run_test(fold, name, verbose=VERBOSE)
                    self.assertTrue(res)

            short_name = os.path.split(os.path.splitext(name)[0])[-1]
            setattr(cls, f"test_{short_name}", _test_)


TestDocumentationExamples.add_test_methods()

if __name__ == "__main__":
    unittest.main(verbosity=2)
