"""
Benchmark exporters
===================

Benchmarks many models from the `HuggingFace <https://huggingface.co/models>`_.

::

    python -m experimental_experiment.torch_bench.bash_bench_huggingface --help
    
    
::

    python -m experimental_experiment.torch_bench.bash_bench_huggingface --model ""
    
::

    python -m experimental_experiment.torch_bench.bash_bench_huggingface --model dummy --exporter eager
    
::

    python -m experimental_experiment.torch_bench.bash_bench_huggingface --model dummy,dummy16 --verbose=1
    
"""

from experimental_experiment.torch_bench._bash_bench_common_cmd import bash_bench_main


def main(args=None):
    bash_bench_main("bash_bench_huggingface", __doc__, args)


if __name__ == "__main__":
    main()
