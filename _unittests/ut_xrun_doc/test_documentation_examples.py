import unittest
import os
import sys
import packaging.version as pv
import importlib.util
import subprocess
import time
from experimental_experiment import __file__ as experimental_experiment_file
from experimental_experiment.ext_test_case import ExtTestCase, is_windows, is_apple

VERBOSE = 0
ROOT = os.path.realpath(
    os.path.abspath(os.path.join(experimental_experiment_file, "..", ".."))
)


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
            out, err = res
            st = err.decode("ascii", errors="ignore")
            if st and "Traceback" in st:
                if '"dot" not found in path.' in st:
                    # dot not installed, this part
                    # is tested in onnx framework
                    if verbose:
                        print(f"failed: {name!r} due to missing dot.")
                    return 0
                raise AssertionError(
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
        this = os.path.abspath(os.path.dirname(__file__))
        fold = os.path.normpath(os.path.join(this, "..", "..", "_doc", "examples"))
        found = os.listdir(fold)
        for name in found:
            if not name.endswith(".py") or not name.startswith("plot_"):
                continue
            reason = None

            if name in {"plot_torch_export_201.py"}:
                if sys.platform in {"win32"}:
                    # dynamo not supported on windows
                    reason = "windows not supported"

            if name in {"plot_llama_bench_102.py", "plot_torch_custom_backend_101.py"}:
                if sys.platform in {"win32", "darwin"}:
                    # dynamo not supported on windows
                    reason = "onnxruntime-training not available"

            if not reason and name in {
                "plot_convolutation_matmul_102.py",
                "plot_optimize_101.py",
                "plot_torch_linreg_101.py",
            }:
                if sys.platform in {"win32", "darwin"}:
                    # dynamo not supported on windows
                    reason = "graphviz not installed"

            if name in {"plot_llama_diff_dort_301.py", "plot_llama_diff_export_301.py"}:
                import torch

                if pv.Version(".".join(torch.__version__.split(".")[:2])) < pv.Version(
                    "2.3"
                ):
                    reason = "requires torch 2.3"

            if name in {"plot_llama_bench_102.py"}:
                if sys.platform in {"darwin"}:
                    reason = "apple not supported"

            if not reason and name in {
                # "plot_convolutation_matmul.py",
                # "plot_profile_existing_onnx.py",
                # "test_plot_torch_dort.py",
                "plot_torch_aot_201.py",
                "plot_torch_dort_201.py",
                # "plot_torch_export.py",
            }:
                # too long
                reason = "not working yet or too long"

            if (
                not reason
                and is_apple()
                and name in {"plot_convolutation_matmul_102.py"}
            ):
                reason = "dot is missing"

            if not reason and is_apple():
                reason = "last for ever"

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
