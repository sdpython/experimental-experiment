import sys


def print_import_time():
    """
    Prints out the import time for the modules involved.
    """
    import time

    def _sys_in(mod):
        if mod in sys.modules:
            print(f"{mod!r} already imported")

    ####################

    _sys_in("onnx")
    begin = time.perf_counter()
    import onnx

    print(f"time to import onnx --- {time.perf_counter() - begin}")

    ####################

    _sys_in("onnx_array_api")
    begin = time.perf_counter()
    import onnx_array_api

    print(f"time to import onnx_array_api --- {time.perf_counter() - begin}")

    ####################

    _sys_in("torch")
    begin = time.perf_counter()
    import torch

    print(f"time to import torch --- {time.perf_counter() - begin}")

    ####################

    _sys_in("torch.export")
    begin = time.perf_counter()
    import torch.export

    print(f"time to import torch.export --- {time.perf_counter() - begin}")

    ####################

    _sys_in("onnxscript")
    begin = time.perf_counter()
    import onnxscript

    print(f"time to import onnxscript --- {time.perf_counter() - begin}")

    ####################

    _sys_in("onnxruntime")
    begin = time.perf_counter()
    import onnxruntime

    print(f"time to import onnxruntime --- {time.perf_counter() - begin}")

    ####################

    _sys_in("torch.onnx")
    begin = time.perf_counter()
    import torch.onnx

    print(f"time to import torch.onnx --- {time.perf_counter() - begin}")

    ####################

    _sys_in("torch._dynamo")
    begin = time.perf_counter()
    import torch._dynamo

    print(f"time to import torch._dynamo --- {time.perf_counter() - begin}")

    #

    _sys_in("time to import experimental_experiment.torch_interpreter")
    begin = time.perf_counter()
    import experimental_experiment.torch_interpreter

    print(
        f"time to import experimental_experiment.torch_interpreter "
        f"--- {time.perf_counter() - begin}"
    )

    #

    _sys_in("experimental_experiment.torch_interpreter.aten_functions")
    begin = time.perf_counter()
    import experimental_experiment.torch_interpreter.aten_functions

    print(
        f"time to import experimental_experiment.torch_interpreter."
        f"aten_functions --- {time.perf_counter() - begin}"
    )
