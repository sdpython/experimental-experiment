"""
Benchmark exporters
===================

Benchmarks many models.
Available exporters:

* eager: identity
* export: :func:`torch.export.export`
* compile: :func:`torch.compile`
* custom: :func:`experimental_experiment.torch_interpreter.to_onnx`
* torch_script: :func:`torch.onnx.export`
* onnx_dynamo: :func:`torch.onnx.export` with ``dynamo=True``
* dynamo_export: :func:`torch.onnx.dynamo_export`

::

    python -m experimental_experiment.torch_bench.bash_bench_explicit --help

::

    python -m experimental_experiment.torch_bench.bash_bench_explicit --model ""
"""

from experimental_experiment.torch_bench._bash_bench_cmd import bash_bench_main


def main(args=None):
    """
    Main function for command line
    ``python -m experimental_experiment.torch_bench.bash_bench_explicit``.
    """
    bash_bench_main("bash_bench_explicit", __doc__, args)


if __name__ == "__main__":
    main()
