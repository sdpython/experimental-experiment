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

::

    python -m experimental_experiment.torch_bench.bash_bench_huggingface_big --help


::

    python -m experimental_experiment.torch_bench.bash_bench_huggingface_big --model ""

::

    python -m experimental_experiment.torch_bench.bash_bench_huggingface_big \\
           --model codellama --exporter eager

"""

from experimental_experiment.torch_bench._bash_bench_cmd import bash_bench_main


def main(args=None):
    """
    Main function for command line
    ``python -m experimental_experiment.torch_bench.bash_bench_huggingface_big``.
    """
    bash_bench_main("bash_bench_huggingface_big", __doc__, args)


if __name__ == "__main__":
    main()
