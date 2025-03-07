import datetime
import glob
import os
import pprint
import warnings
import zipfile
from collections import Counter
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas

from . import BOOLEAN_VALUES
from ._bash_bench_benchmark_runner_agg_helper import (
    SELECTED_FEATURES,
    _apply_excel_style,
    _compute_correlations,
    _create_aggregation_figures,
    _filter_data,
    _fix_report_piv,
    _format_excel_cells,
    _process_formulas,
    _reorder_columns_level,
    _reorder_index_level,
    _reverse_column_names_order,
    _select_metrics,
    _select_model_metrics,
)


def enumerate_csv_files(
    data: Union[
        pandas.DataFrame, List[Union[str, Tuple[str, str]]], str, Tuple[str, str, str, str]
    ],
    verbose: int = 0,
) -> Iterator[Union[pandas.DataFrame, str, Tuple[str, str, str, str]]]:
    """
    Enumerates files considered for the aggregation.
    Only csv files are considered.
    If a zip file is given, the function digs into the zip files and
    loops over csv candidates.

    :param data: dataframe with the raw data or a file or list of files

    data can contains:
    * a dataframe
    * a string for a filename, zip or csv
    * a list of string
    * a tuple
    """
    if not isinstance(data, list):
        data = [data]
    for filename in data:
        if isinstance(filename, pandas.DataFrame):
            yield filename
            continue

        if isinstance(filename, tuple):
            # A file in a zipfile
            yield filename

        if not os.path.exists(filename):
            found = glob.glob(filename)
            if verbose and not found:
                print(f"[enumerate_csv_files] unable to find file in {filename!r}")
            for f in found:
                yield from enumerate_csv_files(f, verbose=verbose)
            continue

        ext = os.path.splitext(filename)[-1]
        if ext == ".csv":
            # We check the first line is ok.
            with open(filename, "r", encoding="utf-8") as f:
                line = f.readline()
                if "~help" in line or (",CMD" not in line and ",DATE" not in line):
                    continue
                dt = datetime.datetime.fromtimestamp(os.stat(filename).st_mtime)
                du = dt.strftime("%Y-%m-%d %H:%M:%S")
                yield (os.path.split(filename)[-1], du, filename, "")
            continue

        if ext == ".zip":
            zf = zipfile.ZipFile(filename, "r")
            for info in zf.infolist():
                name = info.filename
                ext = os.path.splitext(name)[-1]
                if ext != ".csv":
                    continue
                with zf.open(name) as f:
                    line = f.readline()
                if b"~help" in line or (b",CMD" not in line and b",DATE" not in line):
                    continue
                yield (
                    os.path.split(name)[-1],
                    "%04d-%02d-%02d %02d:%02d:%02d" % info.date_time,
                    name,
                    filename,
                )
            zf.close()
            continue

        raise AssertionError(f"Unexpected {filename!r}, cannot read it.")


def open_dataframe(
    data: Union[str, Tuple[str, str, str, str], pandas.DataFrame],
) -> pandas.DataFrame:
    """
    Opens a filename.

    :param data: a dataframe, a filename, a tuple indicating the file is coming
        from a zip file
    :return: a dataframe
    """
    if isinstance(data, pandas.DataFrame):
        return data
    if isinstance(data, str):
        df = pandas.read_csv(data)
        df["RAWFILENAME"] = data
        return df
    if isinstance(data, tuple):
        if not data[-1]:
            df = pandas.read_csv(data[2])
            df["RAWFILENAME"] = data[2]
            return df
        zf = zipfile.ZipFile(data[-1])
        with zf.open(data[2]) as f:
            df = pandas.read_csv(f)
            df["RAWFILENAME"] = f"{data[-1]}/{data[2]}"
        zf.close()
        return df

    raise ValueError(f"Unexpected value for data: {data!r}")


def merge_benchmark_reports(
    data: Union[pandas.DataFrame, List[str], str],
    model=("suite", "model_name"),
    keys=(
        "architecture",
        "exporter",
        "opt_patterns",
        "rtopt",
        "device",
        "device_name",
        "dtype",
        "dynamic",
        "flag_fake_tensor",
        "flag_no_grad",
        "flag_training",
        "machine",
        "processor",
        "processor_name",
        "version_python",
        "version_onnx",
        "version_onnxruntime",
        "version_onnxscript",
        "version_tag",
        "version_torch",
        "version_transformers",
        "version_monai",
        "version_timm",
        "strategy",
    ),
    column_keys=(
        "stat",
        "exporter",
        "opt_patterns",
        "dynamic",
        "rtopt",
    ),
    report_on=(
        "speedup",
        "speedup_increase",
        "speedup_med",
        "discrepancies_*",
        "TIME_ITER",
        "time_*",
        "ERR_*",
        "onnx_*",
        "op_*",
        "memory_*",
        "mem_*",
        "config_*",
        "torch_*",
    ),
    formulas=(
        "export",
        "memory_peak",
        "buckets",
        "status",
        "memory_delta",
        "control_flow",
        "pass_rate",
        "accuracy_rate",
        "accuracy_dynamic_rate",
        "date",
        "correction",
        "error",
    ),
    timestamp_column: str = "timestamp",
    excel_output: Optional[str] = None,
    exc: bool = True,
    filter_in: Optional[str] = None,
    filter_out: Optional[str] = None,
    verbose: int = 0,
    output_clean_raw_data: Optional[str] = None,
    baseline: Optional[pandas.DataFrame] = None,
    export_simple: Optional[str] = None,
    export_correlations: Optional[str] = None,
    broken: bool = False,
    disc: Optional[float] = None,
    slow: Optional[float] = None,
    fast: Optional[float] = None,
    slow_script: Optional[float] = None,
    fast_script: Optional[float] = None,
    exclude: Optional[List[int]] = None,
    keep_more_recent: bool = False,
) -> Dict[str, pandas.DataFrame]:
    """
    Merges multiple files produced by bash_benchmark...

    ::

        _index,DATE,ERR_export,ITER,TIME_ITER,capability,cpu,date_start,device,device_name,...
        101Dummy-custom,2024-07-08,,0,7.119158490095288,7.0,40,2024-07-08,cuda,...
        101Dummy-script,2024-07-08,,1,6.705480073112994,7.0,40,2024-07-08,cuda,...
        101Dummy16-custom,2024-07-08,,2,6.970448340754956,7.0,40,2024-07-08,cuda,...

    :param data: dataframe with the raw data or a file or list of files
    :param model: columns defining a unique model
    :param keys: colimns definined a unique experiment
    :param report_on: report on those metrics, ``<prefix>*`` means all
        columns starting with this prefix
    :param formulas: add computed metrics
    :param timestamp_column: a day, used to tell the user this was run on this day
    :param excel_output: output the computed dataframe into a excel document
    :param exc: raise exception by default
    :param filter_in: filter in some data to make the report smaller (see below)
    :param filter_out: filter out some data to make the report smaller (see below)
    :param verbose: verbosity
    :param output_clean_raw_data: output the concatenated raw data so that it can
        be used later to make a comparison
    :param baseline: to compute difference
    :param export_simple: if not None, export simple in this file.
    :param export_correlations: if not None, export correlations between exporters
    :param broken: produce a document for the broken models per exporter
    :param slow: produce a document for the slow models per exporter
    :param fast: produce a document for the fast models per exporter
    :param slow_script: produce a document for the slow models
        per exporter compare to torch_script
    :param fast_script: produce a document for the fast models
        per exporter compare to torch_script
    :param exclude: exclude a list of files in the list
    :param keep_more_recent: in case of duplicates, keep the most recent value
    :return: dictionary of dataframes

    Every key with a unique value is removed.
    Every column with a unique value is displayed on main.
    List of knowns columns::

        DATE
        ERR_export
        ERR_warmup
        ITER
        TIME_ITER
        capability
        cpu
        date_start
        device
        device_name
        discrepancies_abs
        discrepancies_rel
        dtype
        dump_folder
        dynamic
        executable
        exporter
        ...

    Argument `filter_in` or `filter_out` follows the syntax
    ``<column1>:<fmt1>/<column2>:<fmt2>``.

    The format is the following:

    * a value or a set of values separated by ``;``
    """
    if baseline:
        assert not exclude, f"exclude={exclude} not compatiable with baseline={baseline!r}"
        base_dfs = merge_benchmark_reports(
            baseline,
            model=model,
            keys=keys,
            column_keys=column_keys,
            timestamp_column=timestamp_column,
            report_on=report_on,
            formulas=formulas,
            exc=exc,
            filter_in=filter_in,
            filter_out=filter_out,
            verbose=max(verbose - 1, 0),
        )
    else:
        base_dfs = None

    dfs = []
    for i, dname in enumerate(sorted(enumerate_csv_files(data))):
        if exclude and i in exclude:
            if verbose:
                print(f"[merge_benchmark_reports] - {' '.join(dname)}")
            continue
        if verbose:
            print(f"[merge_benchmark_reports] + {' '.join(dname)}")
        df = open_dataframe(dname)
        if df.columns[0] == "#order":
            # probably a status report
            continue
        updates = {}
        for c in df.columns:
            values = set(df[c])
            if values & {True, False, np.nan} == values:
                updates[c] = (
                    df[c].apply(lambda x: x if np.isnan(x) else (1 if x else 0)).astype(float)
                )
            if (
                df[c].dtype not in {float, np.float64, np.float32}
                and "_qu" not in c
                and c.startswith(("time_", "discrepancies_", "memory"))
            ):
                try:
                    val = df[c].astype(float)
                except (ValueError, TypeError) as e:
                    raise AssertionError(
                        f"Unable to convert to float column {c!r} from file "
                        f"{dname!r}, values\n---\n"
                        f"{pprint.pformat(list(enumerate(zip(df['_index'], df[c]))))}"
                    ) from e
                updates[c] = val

        if updates:
            for k, v in updates.items():
                df[k] = v
        dfs.append(df)

    assert len(dfs) > 0, f"No dataframe found in {data!r}"
    df = pandas.concat(dfs, axis=0) if len(dfs) > 1 else dfs[0]

    if verbose:
        print(f"[merge_benchmark_reports] shape={df.shape}")

    if "STEP" in df.columns:
        # Experiment is run in two step. We remove the export rows
        # if it was successful as the metrics are reported in the row with the speedup.
        if verbose:
            print("[merge_benchmark_reports] remove exporter rows (column STEP)")

        if any(df["STEP"].isna()):
            vals = set(df["STEP"][~df["STEP"].isna()])
            if vals == {"last"}:
                if "ERR_export" in df.columns:
                    df = df[(df["STEP"] == "last") | ~df["ERR_export"].isna()]
                else:
                    df = df[(df["STEP"] == "last")]
            elif "ERR_export" in df.columns:
                df = df[
                    (df["STEP"].isna()) | (df["STEP"] != "export") | ~df["ERR_export"].isna()
                ]
            else:
                df = df[(df["STEP"].isna()) | (df["STEP"] != "export")]
        elif "ERR_export" in df.columns:
            df = df[(df["STEP"] != "export") | ~df["ERR_export"].isna()]
        else:
            df = df[df["STEP"] != "export"]

        if verbose:
            print(f"[merge_benchmark_reports] done, new shape={df.shape}")

    if isinstance(model, str):
        model = [model]
    elif isinstance(model, tuple):
        model = list(model)
    assert isinstance(model, list), f"Unexpected type {type(model)} for model={model}"

    # Let's rename version into version_python
    if "version" in df.columns:
        df = df.copy()
        df["version_python"] = df["version"]
        df = df.drop("version", axis=1)

    if verbose:
        print(f"[merge_benchmark_reports] model={model!r}")

    # checks all columns defining a model are available
    for m in model:
        if m not in df.columns:
            df = df.copy()
            df[m] = ""

    if filter_in or filter_out:
        if verbose:
            print("[merge_benchmark_reports] filtering data")

        df = _filter_data(df, filter_in=filter_in, filter_out=filter_out)

        if verbose:
            print(f"[merge_benchmark_reports] done, new shape={df.shape}")
        if df.shape[0] == 0:
            return {}

    if verbose:
        print("[merge_benchmark_reports] remove empty lines")
    # let's remove the empty line
    df = df[~df[model].isna().max(axis=1)].copy()

    if verbose:
        print(f"[merge_benchmark_reports] new shape={df.shape}")

    # replace nan values by numerical values for some columns
    for k, v in {"rtopt": 1, "dynamic": 0}.items():
        if k not in df.columns:
            df[k] = v
        else:
            df[k] = df[k].apply(lambda x: x in BOOLEAN_VALUES).astype(float).fillna(v)

    for c in ("rtopt", "dynamic"):
        assert c in df.columns and df[c].dtype in {
            np.float64,
            np.int64,
            np.int32,
            np.float32,
            np.dtype("float64"),
            np.dtype("float32"),
            np.dtype("int32"),
            np.dtype("int64"),
        }, (
            f"Column {c!r} is missing or with a wrong type, "
            f"keys={keys!r}, df.columns={sorted(df.columns)}, "
            f"df.dtypes={sorted(zip(df.columns,df.dtypes))}"
        )

    # replace nan values
    # groupby do not like nan values
    set_columns = set(df.columns)
    for c in ["opt_patterns", "ERR_export", "ERR_warmup"]:
        if c in set_columns:
            df[c] = df[c].fillna("-")

    # unique values
    unique = {}
    for c in df.columns:
        u = df[c].dropna().unique()
        if len(u) == 1:
            unique[c] = u.tolist()[0]
    if "exporter" in unique:
        del unique["exporter"]

    # replace nan values in key columns
    # groupby do not like nan values
    for c in keys:
        if c in set_columns:
            if verbose and c in df.columns:
                print(
                    f"[merge_benchmark_reports] KEY {len(set(df[c].dropna()))} "
                    f"unique values for {c!r} : {set(df[c].dropna())}"
                )
            if c.startswith("flag"):
                df[c] = df[c].astype(bool).fillna(False)
            elif c in {"dynamic"}:
                df[c] = df[c].fillna(0)
            elif c in {"rtopt"}:
                df[c] = df[c].fillna(1)
            elif c.startswith("version"):
                df[c] = df[c].fillna("")
            else:
                df[c] = df[c].fillna("-")

    #######################
    # preprocessing is done
    #######################

    # uniques keys
    report_on_star = [r for r in report_on if "*" in r]
    report_on = [r for r in report_on if "*" not in r]
    keys = [k for k in keys if k in set_columns]
    report_on = [k for k in report_on if k in set_columns]
    for col in sorted(set_columns):
        for star in report_on_star:
            assert star[-1] == "*", f"Unsupported position for * in {star!r}"
            s = star[:-1]
            if col.startswith(s):
                report_on.append(col)
                break

    if verbose:
        print(f"[merge_benchmark_reports] report_on {len(report_on)} metrics")

    main = [dict(column="dates", value=", ".join(sorted(df["DATE"].unique().tolist())))]
    for k, v in unique.items():
        main.append(dict(column=k, value=v))
    new_keys = [k for k in keys if k not in unique]

    # new_keys should not be empty.
    if not new_keys:
        for m in column_keys:
            if m == "stat":
                continue
            if m in df.columns:
                new_keys.append(m)
                break
    assert new_keys, f"new_keys is empty, column_keys={column_keys}"

    # remove duplicated rows
    full_index = [
        s
        for s in df.columns
        if s in {*column_keys, *keys, *model, "date", "date_start", timestamp_column}
    ]
    if verbose:
        print(f"[merge_benchmark_reports] remove duplicated rows, starts with {df.shape}")
        print(f"[merge_benchmark_reports] index is {full_index}")
    dupli = ~df.duplicated(full_index, keep="last")
    df = df[dupli].copy()
    if verbose:
        print("[merge_benchmark_reports] continue with {df.shape}")

    if keep_more_recent:
        unique = [*model, *new_keys]
        dates = [c for c in df.columns if c in ["date", "date_start", timestamp_column]]
        if verbose:
            # Keeps the most recent result.
            print(
                f"[merge_benchmark_reports] keep the most recent, "
                f"starts with {df.shape}, with columns={unique} "
                f"and dates={dates}"
            )
        g = df[[*unique, *dates]].copy()
        for c in g.columns:
            if g[c].dtype in (str, object):
                g[c] = g[c].fillna("")
        dfg = g.groupby(unique, as_index=False).max()
        df = pandas.merge(df, dfg, on=[*unique, *dates], how="inner")
        if verbose:
            # Keeps the most recent result.
            print(f"[merge_benchmark_reports] continue with {df.shape}")

    if verbose:
        print(f"[merge_benchmark_reports] done, shape={df.shape}")
        if output_clean_raw_data:
            print(
                f"[merge_benchmark_reports] save clean raw data in {output_clean_raw_data!r}"
            )
            df.to_csv(output_clean_raw_data)
        elif verbose > 1 and excel_output:
            nn = f"{excel_output}.raw.csv"
            print(f"[merge_benchmark_reports] save raw data in {nn!r}")
            df.to_csv(nn)

    # formulas
    if formulas:
        df, report_on_add = _process_formulas(
            df,
            formulas,
            column_keys=column_keys,
            new_keys=new_keys,
            model=model,
            set_columns=set_columns,
            verbose=verbose,
        )
        report_on.extend(report_on_add)

    if verbose:
        print(f"[merge_benchmark_reports] done, shape={df.shape}")

    main = pandas.DataFrame(main)

    def _applies_subset(suffix, filter_fct):
        assert (
            excel_output
        ), f"excel_output={excel_output} must be specified if broken={broken!r}"
        unique_exporters = set(df["exporter"].dropna())
        excel, ext = os.path.splitext(excel_output)
        for exporter in unique_exporters:
            subdf = df[filter_fct(exporter)]
            if subdf.shape[0] == 0:
                continue
            if verbose:
                print(
                    f"[merge_benchmark_reports] build {suffix} model "
                    f"for exporter {exporter!r} shape={subdf.shape}"
                )
            _build_aggregated_document(
                base_dfs=None,
                df=subdf.copy(),
                main=main,
                model=model,
                column_keys=column_keys,
                new_keys=new_keys,
                report_on=report_on,
                verbose=max(verbose - 1, 0),
                excel_output=f"{excel}.{suffix}.{exporter}{ext}",
                exc=exc,
                apply_style=False,
            )

    if broken and "exporter" in df.columns and "time_latency" in df.columns:
        # we select models for which speedup is not specified
        subdf = df[df["time_latency_eager"].isna()]
        if subdf.shape[0] > 0:
            if verbose:
                print(
                    f"[merge_benchmark_reports] build broken model for "
                    f"eager shape={subdf.shape}"
                )
            excel, ext = os.path.splitext(excel_output)
            _build_aggregated_document(
                base_dfs=None,
                df=subdf.copy(),
                main=main,
                model=model,
                column_keys=column_keys,
                new_keys=new_keys,
                report_on=report_on,
                verbose=max(verbose - 1, 0),
                excel_output=f"{excel}.broken.eager{ext}",
                exc=exc,
                apply_style=False,
            )
        _applies_subset(
            "broken",
            lambda exporter: (df["exporter"] == exporter)
            & df["time_latency"].isna()
            & (~df["time_latency_eager"].isna()),
        )

    if (
        disc is not None
        and disc < 10
        and "exporter" in df.columns
        and "discrepancies_abs" in df.columns
    ):
        # we select models for which speedup is not specified
        _applies_subset(
            "disc",
            lambda exporter: (df["exporter"] == exporter)
            & (~df["discrepancies_abs"].isna())
            & (df["discrepancies_abs"] > disc),
        )

    if (
        slow is not None
        and slow < 10
        and "exporter" in df.columns
        and "time_latency" in df.columns
        and "time_latency_eager" in df.columns
    ):
        # we select models for which speedup is not specified
        _applies_subset(
            "slow",
            lambda exporter: (df["exporter"] == exporter)
            & (~df["time_latency"].isna())
            & (~df["time_latency_eager"].isna())
            & (df["time_latency"] * slow > df["time_latency_eager"]),
        )

    if (
        fast is not None
        and fast < 10
        and "exporter" in df.columns
        and "time_latency" in df.columns
        and "time_latency_eager" in df.columns
    ):
        # we select models for which speedup is not specified
        _applies_subset(
            "fast",
            lambda exporter: (df["exporter"] == exporter)
            & (~df["time_latency"].isna())
            & (~df["time_latency_eager"].isna())
            & (df["time_latency"] * fast < df["time_latency_eager"]),
        )

    if (
        slow_script is not None
        and slow_script < 10
        and "exporter" in df.columns
        and "speedup_script" in df.columns
    ):
        # we select models for which speedup is not specified
        _applies_subset(
            "slow_script",
            lambda exporter: df["exporter"].isin({exporter, "torch_script"})
            & (~df["speedup"].isna())
            & (~df["speedup_script"].isna())
            & (df["speedup"] < df["speedup_script"] * slow_script),
        )

    if (
        fast_script is not None
        and fast_script < 10
        and "exporter" in df.columns
        and "speedup_script" in df.columns
    ):
        # we select models for which speedup is not specified
        _applies_subset(
            "fast_script",
            lambda exporter: df["exporter"].isin({exporter, "torch_script"})
            & (~df["speedup"].isna())
            & (~df["speedup_script"].isna())
            & (df["speedup"] > df["speedup_script"] * fast_script),
        )

    return _build_aggregated_document(
        base_dfs=base_dfs,
        df=df,
        main=main,
        model=model,
        column_keys=column_keys,
        new_keys=new_keys,
        report_on=report_on,
        verbose=verbose,
        excel_output=excel_output,
        export_simple=export_simple,
        export_correlations=export_correlations,
        exc=exc,
    )


def _build_aggregated_document(
    base_dfs: Dict[str, pandas.DataFrame],
    df: pandas.DataFrame,
    main: pandas.DataFrame,
    model: Tuple[str, ...],
    column_keys: Tuple[str, ...],
    new_keys: List[str],
    report_on: List[str],
    excel_output: str,
    export_simple: Optional[str] = None,
    export_correlations: Optional[str] = None,
    verbose: int = 0,
    exc: bool = True,
    apply_style: bool = True,
) -> Dict[str, pandas.DataFrame]:
    # values
    res = {"0raw": df, "0main": main}
    for c in report_on:
        keep = [*model, *new_keys, c]
        dfc = df[keep]
        dfc = dfc[~dfc[model].isna().min(axis=1)]
        if new_keys:
            try:
                pivot = dfc.pivot(index=model, columns=new_keys, values=c)
            except ValueError as e:
                cc = [*model, *new_keys]
                dg = dfc.copy()
                dg["__C__"] = 1
                try:
                    under = dg.groupby(cc).count()[["__C__"]]
                except ValueError as e:
                    raise AssertionError(
                        f"Unable to deal with index={model}, columns={new_keys}, "
                        f"values={c}, dfc={dfc.head()}"
                    ) from e
                under = under[under["__C__"] > 1]
                if "RAWFILENAME" in df.columns:
                    filenames = df[[*cc, "RAWFILENAME"]].set_index(cc)
                    under_with_file = under.join(filenames, how="left")
                    text = "\n".join(
                        map(
                            str,
                            under_with_file.reset_index(drop=False).head().values.tolist(),
                        )
                    )
                else:
                    under_with_file = ""
                raise ValueError(
                    f"Ambiguities for columns {cc}, model={model}, "
                    f"new_keys={new_keys}\n{under}\n-----\n"
                    f"{text}"
                ) from e
        else:
            pivot = dfc.set_index(model)
        res[c] = pivot

    if verbose:
        print(f"[merge_benchmark_reports] {len(res)} metrics")

    # let's remove empty variables
    if verbose:
        print("[merge_benchmark_reports] remove empty variables")

    for v in res.values():
        drop = []
        for c in v.columns:
            if all(v[c].isna()) or set(v[c]) == {"-"}:
                drop.append(c)
        if drop:
            v.drop(drop, axis=1, inplace=True)

    res = {k: v for k, v in res.items() if v.shape[1] > 0}
    if "TIME_ITER" in res:
        res["time_ITER"] = res["TIME_ITER"]
        del res["TIME_ITER"]

    if verbose:
        print(f"[merge_benchmark_reports] final number of metrics={len(res)}")

    # final fusion

    def _merge(res, merge, prefix, reverse=True, transpose=False):
        m = None
        for name in merge:
            df = res[name].T
            index_cols = df.index.names
            if index_cols == [None]:
                index_cols = ["index"]
            df["stat"] = name[len(prefix) :]
            df = df.reset_index(drop=False).set_index([*list(index_cols), "stat"])
            # Let's remove duplicated experiment, the last one is kept.
            df = df.T
            df = df[~df.index.duplicated(keep="last")].copy()
            if m is None:
                m = df
                continue
            m0 = m
            try:
                m = pandas.merge(
                    m,
                    df,
                    how="outer",
                    left_index=True,
                    right_index=True,
                    validate="1:1",
                )
            except ValueError as e:
                raise AssertionError(
                    f"Unable to join for name={name}, df.index={df.index.names}, "
                    f"m.index={m.index.names} e={e}\n----\n+ m0.index\n{m0.index[:3]}"
                    f"\n---\n+ df.index\n{df.index[:3]}"
                ) from e
            assert m.shape[0] <= df.shape[0] + m0.shape[0], (
                f"Something is going wrong for prefix {prefix!r} "
                f"(probably a same experiment reported twice), "
                f"df.shape={df.shape}, new_shape is {m.shape} "
                f"(before shape={m0.shape})"
                f"\n-- m0=\n{m0}\n-- df=\n{df}"
            )

        # We need to change the columns index order.
        if reverse:
            df = m.T
            index = set(df.index.names)
            if index == {"stat", "exporter"}:
                m = df.reset_index(drop=False).set_index(["stat", "exporter"]).T
            elif index == {"stat", "index"}:
                m = (
                    df.reset_index(drop=False)
                    .sort_index(axis=1)
                    .drop("index", axis=1)
                    .set_index(["stat"])
                    .T
                )
        else:
            m = m.T.sort_index()
            m = m.T
            if list(m.columns.names) == ["index", "stat"]:
                m.columns = m.columns.droplevel(level=0)

        if transpose:
            m0 = m
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                m = m.T.stack(level=list(range(len(m.index.names)))).reset_index(drop=False)
            cols = m.columns
            assert len(cols) >= 4, (
                f"Unexpected number of columns in {cols}, "
                f"prefix={prefix!r}, m.columns={m.columns}, "
                f"m0.index.names={m0.index.names}, "
                f"m0.columns.names={m0.columns.names}\n---\n{m0}"
            )
            exporter_column = [
                c for c in cols if c in ("exporter", "opt_patterns", "dynamic", "rtopt")
            ]
            assert "stat" in cols, (
                f"Unexpected columns {cols}, expecting 'stat', "
                f"{exporter_column!r}, {model!r}, reverse={reverse}, "
                f"transpose={transpose}, m0.index.names={m0.index.names}, "
                f"m0.columns.names={m0.columns.names}\n---"
                f"\n{m0.head()}\n---\n{m.head()}"
            )
            last = [c for c in cols if c not in {"stat", *exporter_column, *model}]
            assert last, (
                f"last cannot be empty, exporter_column={exporter_column}, "
                f"model={model}, cols={cols}\n----\n{m0.head()}"
            )
            added_columns = [c for c in last if c in new_keys]
            last = [c for c in last if c not in new_keys]
            if len(last) == 2 and last[0] == "index":
                last = last[1:]
            assert len(last) == 1, (
                f"Unexpected columns in {cols}, added={added_columns}, "
                f"last={last}, new_keys={new_keys}, exporter_column={exporter_column}, "
                f"prefix={prefix!r}"
                f"\nm0.index.names={m0.index.names}"
                f"\nm0.columns.names={m0.columns.names}"
                f"\nm0.columns[0]={m0.columns[0]}"
                f"\nm0.columns={m0.columns}\n----"
                f"\nm.index.names={m.index.names}"
                f"\nm.columns.names={m.columns.names}"
                f"\nm.columns[0]={m.columns[0]}"
                f"\nm.columns={m.columns}\n----"
            )
            m = m.pivot(
                index="stat",
                columns=[*model, *exporter_column, *added_columns],
                values=last[0],
            )
            m = m.T.sort_index().T
        return m

    if "speedup" in res:
        res["speedup_1speedup"] = res["speedup"]
        del res["speedup"]

    # verification

    set_model = set(model)
    for c, v in res.items():
        assert (
            c in {"0raw", "0main"} or set_model & set(v.index.names) == set_model
        ), f"There should not be any multiindex but c={c!r}, names={v.index.names}"

    # MODELS
    res["MODELS"] = _select_model_metrics(
        res,
        select=SELECTED_FEATURES,
        stack_levels=tuple(c for c in column_keys if c != "stat"),
    )

    # merging

    for prefix in [
        "status_",
        "discrepancies_",
        "memory_",
        "onnx_",
        "time_",
        "ERR_",
        "op_onnx_",
        "op_torch_",
        "op_opt_",
        "mempeak_",
        "speedup_",
        "bucket_",
        "config_",
        "torch_",
    ]:
        merge = [k for k in res if k.startswith(prefix)]
        merge.sort()

        if len(merge) == 0:
            continue

        if verbose:
            print(f"[merge_benchmark_reports] start fusion of {prefix!r}")

        sheet = _merge(
            res,
            merge,
            prefix,
            reverse=prefix != "status_",
            transpose=prefix.startswith("op_"),
        )
        assert (
            None not in sheet.index.names
        ), f"None in sheet.index.names={sheet.index.names}, prefix={prefix!r}"
        assert (
            None not in sheet.columns.names
        ), f"None in sheet.columns.names={sheet.columns.names}, prefix={prefix!r}"
        if verbose:
            print(f"[merge_benchmark_reports] done, shape of {prefix!r} is {sheet.shape}")

        res[prefix[:-1]] = sheet
        res = {k: v for k, v in res.items() if k not in set(merge)}

    # try to use numerical value everywhere
    if verbose:
        print("[merge_benchmark_reports] enforce numerical values")
    for k, v in res.items():
        if k in {"0main"}:
            continue
        if k.startswith("op_"):
            continue
        for c in v.columns:
            if "output_names" in c or "input_names" in c:
                continue
            if "date" in c:
                continue
            if k.startswith("op_"):
                continue
            if any((isinstance(_, str) and _.startswith("model_")) for _ in c):
                continue
            cc = v[c]
            if cc.dtype == np.object_:
                try:
                    dd = cc.astype(float)
                    v[c] = dd
                except (ValueError, TypeError):
                    types = [
                        type(_) for _ in cc if not isinstance(_, float) or not np.isnan(_)
                    ]
                    if set(types) == {str}:
                        continue
                    co = Counter(types)
                    assert str in co and co[str] >= len(types) // 4 + 1, (
                        f"Unexpected values for k={k!r}, columns={c!r}, "
                        f"types={set(types)}, values={v[c]}, co={co}"
                    )

    if verbose:
        print("[merge_benchmark_reports] done")
        print("[merge_benchmark_reports] create aggregated figures")

    # add pages for the summary
    res["AGG"], res["AGG2"] = _create_aggregation_figures(
        res,
        skip={
            "ERR",
            "0raw",
            "0main",
            "op_onnx",
            "op_torch",
            "op_opt",
            "MODELS",
            "torch",
        },
        model=model,
        exc=exc,
    )
    assert (
        None not in res["AGG"].index.names
    ), f"None in res['AGG'].index.names={res['AGG'].index.names}, prefix='AGG'"
    assert (
        None not in res["AGG2"].index.names
    ), f"None in res['AGG2'].index.names={res['AGG2'].index.names}, prefix='AGG2'"
    if verbose:
        print(
            f"[merge_benchmark_reports] done with shapes "
            f"{res['AGG'].shape} and {res['AGG2'].shape}"
        )

    names = res["AGG"].index.names
    new_names = []
    for c in ["cat", "stat"]:
        if c in names:
            new_names.append(c)
    last_names = []
    for c in column_keys:
        if c == "stat" or c not in names:
            continue
        last_names.append(c)
    if "agg" in names:
        last_names.append("agg")
    for c in names:
        if c not in new_names and c not in last_names:
            new_names.append(c)

    if verbose:
        print("[merge_benchmark_reports] reorder")

    res["AGG"] = _reorder_index_level(res["AGG"], new_names + last_names, prefix="AGG")
    res["AGG2"] = _reorder_index_level(res["AGG2"], new_names + last_names, prefix="AGG2")

    if verbose:
        print("[merge_benchmark_reports] done")

    final_res = {
        k: (
            v
            if k in {"0raw", "0main", "AGG", "AGG2", "op_onnx", "op_opt", "op_torch"}
            else _reorder_columns_level(v, column_keys, prefix=k)
        )
        for k, v in res.items()
    }

    if verbose:
        print(f"[merge_benchmark_reports] done with {len(final_res)} sheets")
        print("[merge_benchmark_reports] creates SUMMARY, SUMMARY2, SIMPLE")

    final_res["SUMMARY"], _suites = _select_metrics(
        res["AGG"], select=SELECTED_FEATURES, prefix="SUMMARY"
    )
    final_res["SIMPLE"], _suites = _select_metrics(
        res["AGG"],
        select=[f for f in SELECTED_FEATURES if f.get("simple", False)],
        prefix="SIMPLE",
    )
    final_res["SUMMARY2"], _suites = _select_metrics(
        res["AGG2"], select=SELECTED_FEATURES, prefix="SUMMARY2"
    )

    # adding dates
    df0 = final_res["0raw"]
    date_col = [
        c
        for c in ["DATE", "suite", "exporter", "opt_patterns", "dtype", "dynamic", "rtopt"]
        if c in df0.columns
    ]
    if verbose:
        print(f"[merge_benchmark_reports] add dates with columns={date_col}")
    date_col2 = [c for c in date_col if c != "DATE"]
    assert len(date_col2) != len(date_col), f"No date found in {sorted(df0.columns)}"
    final_res["dates"] = df0[date_col].groupby(date_col2).max().reset_index(drop=False)
    date_col2 = [c for c in date_col2 if c in final_res["SIMPLE"]]
    assert "suite" in date_col2, f"Unable to find 'suite' in {date_col2}"
    simple = final_res["SIMPLE"].merge(
        final_res["dates"], left_on=date_col2, right_on=date_col2, how="left"
    )
    assert (
        simple.shape[0] == final_res["SIMPLE"].shape[0]
    ), f"Some rows were added or deleted {final_res['SIMPLE'].shape} -> {simple.shape}"
    final_res["SIMPLE"] = simple

    # na to -
    if "ERR" in final_res:
        final_res["ERR"] = final_res["ERR"].fillna("-")

    if verbose:
        print(
            f"[merge_benchmark_reports] done with shapes "
            f"{final_res['SUMMARY'].shape}, {final_res['SUMMARY2'].shape}, "
            f"{final_res['SIMPLE'].shape}"
        )

    if base_dfs:

        def _float_(x):
            if isinstance(x, (int, float)):
                return x
            try:
                return float(x)
            except (ValueError, TypeError):
                return np.nan

        for name in {"0main", "0raw", "SUMMARY", "SUMMARY2", "MODELS", "SIMPLE"}:
            if name in base_dfs:
                final_res[f"{name}_base"] = base_dfs[name]

        for name in {"SUMMARY", "SUMMARY2", "MODELS", "SIMPLE"}:
            if name not in base_dfs or name not in final_res:
                continue
            drop = []
            df_str = final_res[name].select_dtypes("object")
            if df_str.shape[1] > 0:
                # The date may set the type of object just for one value
                # Let cast every columns to see if it can improved.
                df_base = base_dfs[name].copy()
                for c in df_base.columns:
                    cc = df_base[c].apply(_float_).astype(float)
                    if cc.isna().sum() <= 2:
                        df_base[c] = cc
                df_this = final_res[name].copy()
                for c in df_this.columns:
                    cc = df_this[c].apply(_float_).astype(float, errors="ignore")
                    if cc.isna().sum() <= 2:
                        df_this[c] = cc
                df_str = df_this.select_dtypes("object")
            else:
                df_this = final_res[name]
                df_base = base_dfs[name]

            if df_str.shape[1] > 0:
                stacked = list(df_str.columns)
                order_name = [c for c in df_this.columns if "#order" in c]
                if len(order_name) == 1:
                    stacked.extend(order_name)
                    drop.extend(order_name)
                assert all((c in df_this and c in df_base) for c in df_str.columns), (
                    f"Columns mismatch in sheet {name!r}, "
                    f"columns={list(df_str.columns)},\n"
                    f"[{[int(c in df_this) for c in df_str.columns]}], "
                    f"[{[int(c in df_base) for c in df_str.columns]}], "
                    f"\n{list(df_this.columns)}\n!=\n{list(df_base.columns)}"
                )
                df_this = df_this.set_index(stacked)
                df_base = df_base.set_index(stacked)
            else:
                stacked = None

            df_this = df_this.select_dtypes("number")
            df_base = df_base.select_dtypes("number")
            set_columns = sorted(set(df_this.columns) & set(df_base.columns))
            df_base = df_base[set_columns].sort_index(axis=0).sort_index(axis=1)
            df_this = df_this[set_columns].sort_index(axis=0).sort_index(axis=1)
            df_num = df_this.sub(df_base)
            if stacked:
                df_num = df_num.reset_index(drop=False)
                order_name = [c for c in df_num.columns if "#order" in c]
                if len(order_name) == 1:
                    df_num = df_num.sort_values(order_name)

            final_res[f"{name}_diff"] = df_num.sort_index(axis=1)

    # cleaning empty columns
    for v in final_res.values():
        v.dropna(axis=1, how="all", inplace=True)

    if export_simple and "SIMPLE" in final_res:
        if verbose:
            print(f"[merge_benchmark_reports-agg] start {export_simple!r}")
        for c in ("rtopt",):
            if (
                c in final_res["SIMPLE"].columns
                and len(set(final_res["SIMPLE"][c].dropna())) <= 1
            ):
                if verbose:
                    print(f"[merge_benchmark_reports-agg] drops {c!r} in SIMPLE")
                final_res["SIMPLE"] = final_res["SIMPLE"].drop(c, axis=1)
            elif verbose and c in final_res["SIMPLE"].columns:
                print(
                    f"[merge_benchmark_reports-agg] keeps {c!r} in SIMPLE: "
                    f"{set(final_res['SIMPLE'].dropna())}"
                )

        data_csv = final_res["SIMPLE"]
        _set_ = set(data_csv)

        # first pivot
        piv_index = tuple(c for c in ("dtype", "suite", "#order", "METRIC") if c in _set_)
        piv_columns = tuple(
            c for c in ("exporter", "opt_patterns", "dynamic", "rtopt") if c in _set_
        )
        ccc = [*piv_index, *piv_columns]
        gr = data_csv[[*ccc, "value"]].groupby(ccc).count()
        assert gr.values.max() <= 1, (
            f"Unexpected duplicated, piv_index={piv_index}, "
            f"piv_columns={piv_columns}, columns={data_csv.columns}, "
            f"set of columns you may want to skip to pass this test: "
            f"{dict((k,set(df[k])) for k in new_keys if k in df.columns)}, "  # noqa: C402
            f"issue=\n{gr[gr['value'] > 1]}"
        )

        piv = (
            data_csv.pivot(
                index=piv_index,
                columns=piv_columns,
                values="value",
            )
            .sort_index(axis=0)
            .sort_index(axis=1)
        )

        # total
        piv_index = tuple(c for c in ("#order", "METRIC") if c in _set_)
        piv_columns = (
            c for c in ("exporter", "opt_patterns", "dynamic", "rtopt") if c in _set_
        )
        piv_total = (
            pandas.pivot_table(
                data_csv,
                index=piv_index,
                columns=piv_columns,
                values="value",
                aggfunc="sum",
            )
            .sort_index(axis=0)
            .sort_index(axis=1)
        )
        piv, _weighted_speedup = _fix_report_piv(piv)
        piv_total, _ = _fix_report_piv(piv_total, agg=True)

        # concatenation with the all suite
        exporters = set(piv.columns)
        flat_piv = piv.reset_index(drop=False)
        not_exporters = {c for c in flat_piv.columns if c not in exporters}
        flat_piv_total = piv_total.reset_index(drop=False)
        set_piv = set(flat_piv_total.columns)
        for c in flat_piv.columns:
            if c not in set_piv:
                flat_piv_total[c] = "All"
        data_append = flat_piv_total

        data_csv = pandas.concat([flat_piv, data_append], axis=0)
        dates = data_csv[data_csv.METRIC == "date"][list(exporters)]
        data_csv = data_csv[data_csv.METRIC != "date"]
        nodata_data = data_csv.copy()
        dates_max = dates.astype(str).values.max() if dates.astype(str).shape[0] > 0 else None
        if dates_max is not None:
            data_csv["DATE"] = dates_max
        data_csv.columns = [
            "-".join(str(_).replace(".0", "") for _ in c if _ and _ != "none")
            for c in data_csv.columns
        ]
        data_csv.to_csv(export_simple.replace(".csv", ".exporter.csv"), index=False)

        # data_csv.to_csv(export_simple, index=True)

        reindexed = nodata_data.set_index(list(not_exporters))
        reindexed = reindexed.stack(
            list(range(len(reindexed.columns.names))), future_stack=True
        )
        if not isinstance(reindexed, pandas.DataFrame):
            reindexed = reindexed.to_frame()
        reindexed.columns = ["value"]
        reindexed = reindexed.reset_index(drop=False)
        if dates_max is not None:
            reindexed["DATE"] = dates_max
        reindexed.columns = ["".join(map(str, c)) for c in reindexed.columns]
        reindexed.to_csv(export_simple, index=False)

        export_simple_x = f"{export_simple}.xlsx"

        # We finally create a last page with the buckets only.
        mask_eager = [
            any(isinstance(_, str) and "speedup in" in _ for _ in row)
            for row in piv_total.index
        ]
        select_eager = piv_total.loc[mask_eager, :]
        mask_script = [
            any(isinstance(_, str) and "script in" in _ for _ in row)
            for row in piv_total.index
        ]
        select_script = piv_total.loc[mask_script, :]

        if verbose:
            print(
                f"[merge_benchmark_reports-agg] writes {export_simple_x!r} "
                f"shapes: {piv.shape}, {piv_total.shape}"
            )
        with pandas.ExcelWriter(export_simple_x, engine="openpyxl") as writer:
            piv.to_excel(writer, sheet_name="by_suite")
            piv_total.to_excel(writer, sheet_name="all_suites")
            select_eager.to_excel(writer, sheet_name="speedup_eager")
            _format_excel_cells(
                ["by_suite", "all_suites", "speedup_eager"], writer, verbose=verbose
            )
            if select_script.shape[0] > 0:
                select_script.to_excel(writer, sheet_name="speedup_script")
                _format_excel_cells(
                    ["by_suite", "all_suites", "speedup_eager", "speedup_script"],
                    writer,
                    verbose=verbose,
                )

    if export_correlations:
        models = [c for c in model if c in df.columns]
        exporter = [c for c in column_keys if c in df.columns]
        subset = [
            c
            for c in [
                "time_latency",
                "time_latency_eager",
                "time_export_unbiased",
                "discrepancies_abs",
            ]
            if c in df.columns
        ]
        if verbose:
            print(f"[merge_benchmark_reports-agg] compute correlations models={models}")
            print(f"[merge_benchmark_reports-agg] compute correlations exporter={exporter}")
            print(f"[merge_benchmark_reports-agg] compute correlations subset={subset}")
        corrs = _compute_correlations(
            df,
            model_column=models,
            exporter_column=exporter,
            columns=subset,
            verbose=verbose,
        )
        with pandas.ExcelWriter(export_correlations) as writer:
            for k, ev in corrs.items():
                ev.to_excel(
                    writer,
                    sheet_name=k,
                )

    if excel_output:
        if verbose:
            print(f"[merge_benchmark_reports] apply Excel style with {excel_output!r}")
        with pandas.ExcelWriter(excel_output, engine="openpyxl") as writer:
            no_index = {
                "0raw",
                "0main",
                "SUMMARY",
                "SUMMARY_base",
                "SIMPLE",
                "SIMPLE_base",
            }
            first_sheet = ["0main"]
            last_sheet = [
                "ERR",
                "SUMMARY",
                "SUMMARY_base",
                "SUMMARY_diff",
                "AGG",
                "AGG2",
                "0raw",
            ]
            ordered = set(first_sheet) | set(last_sheet)
            order = [
                *first_sheet,
                *sorted([k for k in final_res if k not in ordered]),
                *last_sheet,
            ]
            order = [k for k in order if k in final_res]
            for k in order:
                v = final_res[k]
                ev = _reverse_column_names_order(v, name=k)
                frow = len(ev.columns.names) if isinstance(ev.columns.names, list) else 1
                if k.startswith("SUMMARY"):
                    fcol = len(v.columns.names)
                else:
                    fcol = len(ev.index.names) if isinstance(ev.index.names, list) else 1

                if (
                    k in {"AGG2", "SUMMARY2", "SUMMARY2_base", "SUMMARY2_diff"}
                    and "suite" in ev.columns.names
                ):
                    if verbose:
                        print(f"[merge_benchmark_reports] reorder {k!r} (1)")
                    ev = _reorder_columns_level(ev, first_level=["suite"], prefix=k)
                elif k == "MODELS":
                    if verbose:
                        print(f"[merge_benchmark_reports] reorder {k!r} (2)")
                    ev = _reorder_columns_level(ev, first_level=["#order"], prefix=k)
                if verbose:
                    print(f"[merge_benchmark_reports] add {k!r} to {excel_output!r}")
                ev.to_excel(
                    writer,
                    sheet_name=k,
                    index=k not in no_index,
                    freeze_panes=(frow, fcol) if k not in no_index else None,
                )
                if verbose:
                    print(f"[merge_benchmark_reports] added {k!r} to {excel_output!r}")
            if apply_style:
                _apply_excel_style(final_res, writer, verbose=verbose)
            if verbose:
                print(f"[merge_benchmark_reports] save in {excel_output!r}")

    return final_res
