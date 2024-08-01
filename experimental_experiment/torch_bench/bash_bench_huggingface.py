"""
Benchmark exporters
===================

Benchmarks many models from the `HuggingFace <https://huggingface.co/models>`_.
Available exporters:

* eager: identity
* export: :func:`torch.export.export`
* compile: :func:`torch.compile`
* custom: :func:`experimental_experiment.torch_interpreter.to_onnx`
* torch_script: :func:`torch.onnx.export`
* onnx_dynamo: :func:`torch.onnx.export` with ``dynamo=True``
* dynamo_export: :func:`torch.onnx.dynamo_export`

::

    python -m experimental_experiment.torch_bench.bash_bench_huggingface --help
    
    
::

    python -m experimental_experiment.torch_bench.bash_bench_huggingface --model ""
    
::

    python -m experimental_experiment.torch_bench.bash_bench_huggingface --model 101Dummy --exporter eager
    
::

    python -m experimental_experiment.torch_bench.bash_bench_huggingface --model 101Dummy,101Dummy16 --verbose=1
    
"""

from experimental_experiment.torch_bench._bash_bench_cmd import bash_bench_main


def main(args=None):
    bash_bench_main("bash_bench_huggingface", __doc__, args)


if __name__ == "__main__":
    main()
