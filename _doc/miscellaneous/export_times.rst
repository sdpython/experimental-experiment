============
Export Times
============

fx_mode
=======

symbolic
++++++++

.. runpython::
    :showcode:

    import time
    import warnings
    import numpy as np
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaModel
    import onnx
    import onnxruntime
    import torch
    import torch._dynamo
    import torch.export
    import onnxscript
    import torch.onnx
    import experimental_experiment
    import experimental_experiment.torch_interpreter
    import experimental_experiment.torch_interpreter.aten_functions
    from experimental_experiment.torch_models.llama_helper import get_llama_model

    begin = time.perf_counter()
    print("creating model")
    model, example_args_collection = get_llama_model(
        input_dims=[(2, 1024)],
        hidden_size=4096,
        num_hidden_layers=2,
        vocab_size=32000,
        intermediate_size=11008,
        max_position_embeddings=2048,
        num_attention_heads=32,
        _attn_implementation="eager",
    )

    torch._dynamo.reset()
    begin = time.perf_counter()
    torch._dynamo.export(model, tracing_mode="symbolic")(*example_args_collection[0])
    print(f"time to export symbolic --- {time.perf_counter() - begin}")

fake
++++

.. runpython::
    :showcode:

    import time
    import warnings
    import numpy as np
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaModel
    import onnx
    import onnxruntime
    import torch
    import torch._dynamo
    import torch.export
    import onnxscript
    import torch.onnx
    import experimental_experiment
    import experimental_experiment.torch_interpreter
    import experimental_experiment.torch_interpreter.aten_functions
    from experimental_experiment.torch_models.llama_helper import get_llama_model

    begin = time.perf_counter()
    print("creating model")
    model, example_args_collection = get_llama_model(
        input_dims=[(2, 1024)],
        hidden_size=4096,
        num_hidden_layers=2,
        vocab_size=32000,
        intermediate_size=11008,
        max_position_embeddings=2048,
        num_attention_heads=32,
        _attn_implementation="eager",
    )

    torch._dynamo.reset()
    begin = time.perf_counter()
    torch._dynamo.export(model, tracing_mode="fake")(*example_args_collection[0])
    print(f"time to export fake --- {time.perf_counter() - begin}")


Custom Exporter
===============

With a very simple model:

.. runpython::
    :showcode:
    :process:

    import time
    from experimental_experiment.checks import print_import_time

    print_import_time()

    import torch
    import experimental_experiment.torch_interpreter

    class Neuron(torch.nn.Module):
        def __init__(self, n_dims: int, n_targets: int):
            super(Neuron, self).__init__()
            self.linear = torch.nn.Linear(n_dims, n_targets)

        def forward(self, x):
            return torch.sigmoid(self.linear(x))


    model = Neuron(3, 1)
    x = torch.rand(5, 3)

    begin = time.perf_counter()
    onx = experimental_experiment.torch_interpreter.to_onnx(model, (x,))
    print(f"time to export 1x --- {time.perf_counter() - begin}")

    begin = time.perf_counter()
    onx = experimental_experiment.torch_interpreter.to_onnx(model, (x,))
    print(f"time to export 2x --- {time.perf_counter() - begin}")

With a bigger model:

.. runpython::
    :showcode:
    :process:

    import time
    import warnings
    import numpy as np
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaModel
    import onnx
    import onnxruntime
    import torch
    import torch._dynamo
    import torch.export
    import onnxscript
    import torch.onnx
    import experimental_experiment
    import experimental_experiment.torch_interpreter
    import experimental_experiment.torch_interpreter.aten_functions
    from experimental_experiment.torch_models.llama_helper import get_llama_model

    model, example_args_collection = get_llama_model(
        input_dims=[(2, 1024)],
        hidden_size=4096,
        num_hidden_layers=1,
        vocab_size=32000,
        intermediate_size=11008,
        max_position_embeddings=2048,
        num_attention_heads=32,
        _attn_implementation="eager",
    )

    begin = time.perf_counter()
    onx = experimental_experiment.torch_interpreter.to_onnx(model, example_args_collection[0])
    print(f"time to export 1x --- {time.perf_counter() - begin}")

    begin = time.perf_counter()
    onx = experimental_experiment.torch_interpreter.to_onnx(model, example_args_collection[0])
    print(f"time to export 2x --- {time.perf_counter() - begin}")


Dynamo Exporter
===============

.. runpython::
    :showcode:
    :process:

    import time
    import warnings

    from experimental_experiment.checks import print_import_time

    print_import_time()

    import torch
    import experimental_experiment.torch_interpreter

    class Neuron(torch.nn.Module):
        def __init__(self, n_dims: int, n_targets: int):
            super(Neuron, self).__init__()
            self.linear = torch.nn.Linear(n_dims, n_targets)

        def forward(self, x):
            return torch.sigmoid(self.linear(x))


    model = Neuron(3, 1)
    x = torch.rand(5, 3)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        begin = time.perf_counter()
        onx = torch.onnx.dynamo_export(model, x)
        print(f"time to export 1x --- {time.perf_counter() - begin}")

        begin = time.perf_counter()
        onx = torch.onnx.dynamo_export(model, x)
        print(f"time to export 2x --- {time.perf_counter() - begin}")

With a bigger model:

.. runpython::
    :showcode:
    :process:

    import time
    import warnings
    import numpy as np
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaModel
    import onnx
    import onnxruntime
    import torch
    import torch._dynamo
    import torch.export
    import onnxscript
    import torch.onnx
    import experimental_experiment
    import experimental_experiment.torch_interpreter
    import experimental_experiment.torch_interpreter.aten_functions
    from experimental_experiment.torch_models.llama_helper import get_llama_model

    model, example_args_collection = get_llama_model(
        input_dims=[(2, 1024)],
        hidden_size=4096,
        num_hidden_layers=1,
        vocab_size=32000,
        intermediate_size=11008,
        max_position_embeddings=2048,
        num_attention_heads=32,
        _attn_implementation="eager",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        begin = time.perf_counter()
        onx = torch.onnx.dynamo_export(model, *example_args_collection[0])
        print(f"time to export 1x --- {time.perf_counter() - begin}")

        begin = time.perf_counter()
        onx = torch.onnx.dynamo_export(model, *example_args_collection[0])
        print(f"time to export 2x --- {time.perf_counter() - begin}")
