"""
Benchmark Aggregator
====================

Calls :func:`merge_benchmark_reports
<experimental_experiment.torch_bench._bash_bench_benchmark_runner.merge_benchmark_reports>`.

::

    python -m experimental_experiment.torch_bench.bash_bench_agg summary.xlsx a.csv b.csv
"""

import inspect
from argparse import ArgumentParser


def main(args=None):
    parser = ArgumentParser(
        "experimental_experiment.torch_bench.bash_bench_agg", description=__doc__
    )
    parser.add_argument("output", help="output excel file")
    parser.add_argument("inputs", nargs="+", help="input csv files, at least 1")
    parser.add_argument(
        "--filter_in",
        default="",
        help="adds a filter to filter in data, syntax is "
        '``"<column1>:<value1>;<value2>/<column2>:<value3>"`` ...',
    )
    parser.add_argument(
        "--filter_out",
        default="",
        help="adds a filter to filter out data, syntax is "
        '``"<column1>:<value1>;<value2>/<column2>:<value3>"`` ...',
    )
    parser.add_argument(
        "--skip_keys",
        default="",
        help="skip the differences on those columns, example: "
        "``--skip_keys=version,version_onnxscript,version_torch``",
    )
    parser.add_argument(
        "--save_raw",
        default="",
        help="save the concatanated cleaned raw data in a csv file",
    )
    parser.add_argument(
        "--baseline",
        default="",
        help="a csv file containing the baseline the new figures "
        "needs to be compared to",
    )
    parser.add_argument(
        "--quiet",
        default=0,
        help="avoid raising an exception if it fails",
    )
    parser.add_argument("--verbose", default=0, help="verbosity level")
    res = parser.parse_args(args=args)

    from experimental_experiment.torch_bench._bash_bench_benchmark_runner_agg import (
        merge_benchmark_reports,
    )

    kwargs = {}
    if res.skip_keys:
        sig = inspect.signature(merge_benchmark_reports)
        keys = None
        for p in sig.parameters:
            if p == "keys":
                keys = sig.parameters[p].default
        assert (
            keys is not None
        ), f"Unable to extract the default values for keys in {sig}"
        skip = set(res.skip_keys.split(","))
        kwargs["keys"] = tuple(c for c in keys if c not in skip)

    merge_benchmark_reports(
        res.inputs,
        excel_output=res.output,
        filter_in=res.filter_in,
        filter_out=res.filter_out,
        verbose=int(res.verbose),
        output_clean_raw_data=res.save_raw,
        baseline=res.baseline,
        exc=args.quiet not in (1, "1", True, "True"),
        **kwargs,
    )


if __name__ == "__main__":
    main()
