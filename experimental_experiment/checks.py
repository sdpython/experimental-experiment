import sys


def print_import_time():
    """
    Prints out the import time for the modules involved.
    """
    import time

    ####################

    assert "onnx" not in sys.modules
    begin = time.perf_counter()
    import onnx

    print(f"time to import onnx --- {time.perf_counter() - begin}")

    ####################

    assert "onnx_array_api" not in sys.modules
    begin = time.perf_counter()
    import onnx_array_api

    print(f"time to import onnx_array_api --- {time.perf_counter() - begin}")

    ####################

    assert "torch" not in sys.modules
    begin = time.perf_counter()
    import torch

    print(f"time to import torch --- {time.perf_counter() - begin}")

    ####################

    if "torch.export" in sys.modules:
        print("torch.export was imported.")
    else:
        begin = time.perf_counter()
        import torch.export

        print(f"time to import torch.export --- {time.perf_counter() - begin}")

    ####################

    assert "onnxscript" not in sys.modules
    begin = time.perf_counter()
    import onnxscript

    print(f"time to import onnxscript --- {time.perf_counter() - begin}")

    ####################

    assert "onnxruntime" not in sys.modules
    begin = time.perf_counter()
    import onnxruntime

    print(f"time to import onnxruntime --- {time.perf_counter() - begin}")

    ####################

    if "torch.onnx" in sys.modules:
        print("torch.onnx was imported.")
    else:
        begin = time.perf_counter()
        import torch.onnx

        print(f"time to import torch.onnx --- {time.perf_counter() - begin}")

    ####################

    assert "torch._dynamo" not in sys.modules
    begin = time.perf_counter()
    import torch._dynamo

    print(f"time to import torch._dynamo --- {time.perf_counter() - begin}")

    #

    assert "experimental_experiment.torch_interpreter" not in sys.modules
    begin = time.perf_counter()
    import experimental_experiment.torch_interpreter

    print(
        f"time to import experimental_experiment.torch_interpreter --- {time.perf_counter() - begin}"
    )

    #

    assert "experimental_experiment.torch_interpreter.aten_functions" not in sys.modules
    begin = time.perf_counter()
    import experimental_experiment.torch_interpreter.aten_functions

    print(
        f"time to import experimental_experiment.torch_interpreter.aten_functions --- {time.perf_counter() - begin}"
    )
