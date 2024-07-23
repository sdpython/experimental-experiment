"""
Benchmark exporters
===================

Benchmarks many models from the `HuggingFace <https://huggingface.co/models>`_.
Available exporters:

* eager: identity
* export: :func:`torch.export.export`
* compile: :func:`torch.compile`
* custom: :func:`experimental_experiment.torch_interpreter.to_onnx`
* script: :func:`torch.onnx.export`
* dynamo: :func:`torch.onnx.export` with ``dynamo=True``
* dynamo: :func:`torch.onnx.dynamo_export`

::

    python -m experimental_experiment.torch_bench.bash_bench_torchbench --help
    
    
::

    python -m experimental_experiment.torch_bench.bash_bench_torchbench --model ""
    
::

    python -m experimental_experiment.torch_bench.bash_bench_torchbench --model 101Dummy --exporter eager
    
::

    python -m experimental_experiment.torch_bench.bash_bench_torchbench --model 101Dummy,101Dummy16 --verbose=1

Extra dependencies:

* https://github.com/pytorch/benchmark
* fbgemm_gpu_nightly
* iopath
* opencv-python
* pyre-extensions
* torchrec
"""

from experimental_experiment.torch_bench._bash_bench_cmd import bash_bench_main


def main(args=None):
    bash_bench_main("bash_bench_torchbench", __doc__, args)


if __name__ == "__main__":
    main()
