"""
Benchmark Aggregator
====================

Calls :func:`merge_benchmark_reports
<experimental_experiment.torch_bench._bash_bench_benchmark_runner.merge_benchmark_reports>`.

::

    python -m experimental_experiment.torch_bench.bash_bench_agg summary.xlsx a.csv b.csv
"""

import inspect
import os
import zipfile
from argparse import ArgumentParser, BooleanOptionalAction


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
        help="a csv file containing the baseline the new figures needs to be compared to",
    )
    parser.add_argument(
        "--quiet",
        default=0,
        help="avoid raising an exception if it fails",
    )
    parser.add_argument(
        "--export_simple",
        default="",
        help="if not empty, export main figures into a csv file",
    )
    parser.add_argument(
        "--export_correlations",
        default="",
        help="if not empty, gives insights on model running for two exporters",
    )
    parser.add_argument(
        "--broken",
        default=0,
        help="if true, creates a secondary file per exporter with all broken models",
    )
    parser.add_argument(
        "--disc",
        default=1e9,
        help="if < 10, creates a secondary file per exporter with all "
        "models and wrong discrepancy",
    )
    parser.add_argument(
        "--slow",
        default=1e9,
        help="if < 10, creates a secondary file per exporter with all "
        "models whose speedup is lower than this",
    )
    parser.add_argument(
        "--fast",
        default=1e9,
        help="if < 10, creates a secondary file per exporter with all "
        "models whose speedup is higher than this",
    )
    parser.add_argument(
        "--slow_script",
        default=1e9,
        help="if < 10, creates a secondary file per exporter with all "
        "models whose speedup is lower than torch_script",
    )
    parser.add_argument(
        "--fast_script",
        default=1e9,
        help="if < 10, creates a secondary file per exporter with all "
        "models whose speedup is higher than torch_script",
    )
    parser.add_argument(
        "--list",
        default=False,
        action=BooleanOptionalAction,
        help="List of files involved in the aggregation",
    )
    parser.add_argument(
        "--recent",
        default=False,
        action=BooleanOptionalAction,
        help="If the same experiment was run twice, keeps only the latest one",
    )
    parser.add_argument(
        "--exclude",
        default="",
        help="excludes files from the aggregation given by their number "
        "(use --list to determine them)",
    )
    parser.add_argument("--verbose", default=0, help="verbosity level")
    res = parser.parse_args(args=args)

    from experimental_experiment.torch_bench import BOOLEAN_VALUES
    from experimental_experiment.torch_bench._bash_bench_benchmark_runner_agg import (
        merge_benchmark_reports,
        enumerate_csv_files,
        open_dataframe,
    )

    kwargs = {}
    if res.skip_keys:
        sig = inspect.signature(merge_benchmark_reports)
        keys = None
        for p in sig.parameters:
            if p == "keys":
                keys = sig.parameters[p].default
        assert keys is not None, f"Unable to extract the default values for keys in {sig}"
        skip = set(res.skip_keys.split(","))
        kwargs["keys"] = tuple(c for c in keys if c not in skip)

    if res.list:
        print(f"Looking into {res.inputs!r}...")
        for i, filename in enumerate(sorted(enumerate_csv_files(res.inputs))):
            df = open_dataframe(filename)
            exporters = sorted(set(df.exporter))
            opt_patterns = sorted(set(df.opt_patterns))
            dynamic = sorted(set(df.dynamic))
            suites = sorted(set(df.suite))
            if len(suites) == 1 and len(exporters) == 1 and len(dynamic) == 1:
                prefix = (
                    f"{1 if dynamic[0] in ('True', '1', 1, True) else 0}:"
                    f"{exporters[0]}:{suites[0]}:{','.join(opt_patterns)} - "
                )
            else:
                prefix = ""

            if isinstance(filename, str):
                print(
                    f"{filename} - {os.stat(filename).st_mltime} - "
                    f"{os.stat(filename).st_size} bytes"
                )
            elif isinstance(filename, tuple) and not filename[-1]:
                assert len(filename) == 4, f"Unexpected value {filename!r}"
                print(
                    f"{i} - {prefix}{filename[0]} - {filename[1]} - "
                    f"{os.stat(filename[2]).st_size} bytes (fullname={filename[2]})"
                )
            elif isinstance(filename, tuple):
                assert len(filename) == 4, f"Unexpected value {filename!r}"
                zf = zipfile.ZipFile(filename[-1], "r")
                info = zf.getinfo(filename[2])
                zf.close()
                print(
                    f"{i} - {prefix}{filename[0]} - {filename[1]} - "
                    f"{info.file_size} bytes in {filename[-1]}"
                )
            if not prefix:
                print(
                    f"        suites={','.join(suites)}, "
                    f"dynamic={','.join(map(str, dynamic))}, "
                    f"exporters={','.join(exporters)}, "
                    f"opt_patterns={','.join(opt_patterns)}"
                )
        return

    merge_benchmark_reports(
        res.inputs,
        excel_output=res.output,
        filter_in=res.filter_in,
        filter_out=res.filter_out,
        verbose=int(res.verbose),
        output_clean_raw_data=res.save_raw,
        baseline=res.baseline,
        exc=res.quiet not in BOOLEAN_VALUES,
        export_simple=res.export_simple,
        export_correlations=res.export_correlations,
        broken=res.broken in BOOLEAN_VALUES,
        disc=float(res.disc),
        slow=float(res.slow),
        fast=float(res.fast),
        slow_script=float(res.slow_script),
        fast_script=float(res.fast_script),
        exclude=(
            [int(i) for i in res.exclude.strip().split(",")] if res.exclude.strip() else None
        ),
        keep_more_recent=res.recent,
        **kwargs,
    )


if __name__ == "__main__":
    main()
