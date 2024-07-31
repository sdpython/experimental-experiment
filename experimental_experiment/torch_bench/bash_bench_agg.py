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
    parser.add_argument(
        "--filter_in", default="", help="adds a filter to filter in data"
    )
    parser.add_argument(
        "--filter_out", default="", help="adds a filter to filter out data"
    )
    res = parser.parse_args(args=args)

    from experimental_experiment.torch_bench._bash_bench_benchmark_runner_agg import (
        merge_benchmark_reports,
    )

    merge_benchmark_reports(
        res.inputs,
        excel_output=res.output,
        filter_in=res.filter_in,
        filter_out=res.filter_out,
    )


if __name__ == "__main__":
    main()
