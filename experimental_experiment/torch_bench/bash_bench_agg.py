"""
Benchmark Aggregator
====================

Calls :func:`experimental_experiment.torch_bench._bash_bench_benchmark_runner.merge_benchmark_reports`.

::

    python -m experimental_experiment.torch_bench.bash_bench_agg summary.xlsx a.csv b.csv
"""

from argparse import ArgumentParser


def main(args=None):
    parser = ArgumentParser(
        "experimental_experiment.torch_bench.bash_bench_agg", description=__doc__
    )
    parser.add_argument("output", help="output excel file")
    parser.add_argument("inputs", nargs="+", help="input csv files, at least 1")
    res = parser.parse_args(args=args)

    from experimental_experiment.torch_bench._bash_bench_benchmark_runner_agg import (
        merge_benchmark_reports,
    )

    merge_benchmark_reports(res.inputs, excel_output=res.output)


if __name__ == "__main__":
    main()
