import glob
import itertools
import warnings
from collections import Counter
from typing import Optional, Dict, List, Set, Union
import numpy as np

SELECTED_FEATURES = [
    dict(
        cat="time",
        stat="ITER",
        agg="TOTAL",
        new_name="number of models",
        unit="N",
    ),
    dict(
        cat="speedup",
        agg="COUNT",
        stat="increase",
        new_name="number of running models",
        unit="N",
    ),
    dict(
        cat="time", agg="COUNT%", stat="export_success", new_name="pass rate", unit="%"
    ),
    dict(
        cat="time",
        agg="MEAN",
        stat="export_success",
        new_name="average export time",
        unit="s",
    ),
    dict(
        cat="speedup",
        agg="GEO-MEAN",
        stat="1speedup",
        new_name="average speedup (geo)",
        unit="x",
    ),
    dict(
        cat="status",
        agg="MEAN",
        stat="err<1e-1",
        new_name="discrepancies < 0.1",
        unit="%",
    ),
    dict(
        cat="status",
        agg="MEAN",
        stat="err<1e-2",
        new_name="discrepancies < 0.01",
        unit="%",
    ),
    dict(
        cat="status",
        agg="MEAN",
        stat="err_0<1e-1",
        new_name="discrepancies first output < 0.1",
        unit="%",
    ),
    dict(
        cat="status",
        agg="MEAN",
        stat="err_0<1e-2",
        new_name="discrepancies first output < 0.01",
        unit="%",
    ),
    dict(
        cat="status",
        agg="MEAN",
        stat="lat<=script+2%",
        new_name="model equal or faster than torch.script",
        unit="%",
    ),
    dict(
        cat="status",
        agg="MEAN",
        stat="lat<=eager+2%",
        new_name="model equal or faster than eager",
        unit="%",
    ),
    dict(
        cat="TIME_ITER",
        stat="TIME_ITER",
        agg="MEAN",
        new_name="average iteration time",
        unit="s",
    ),
    dict(
        cat="discrepancies",
        stat="abs",
        agg="MEAN",
        new_name="average absolute discrepancies",
        unit="f",
    ),
    dict(
        cat="time",
        agg="MEAN",
        stat="latency_eager",
        new_name="average latency eager",
        unit="s",
    ),
    dict(
        cat="time", agg="MEAN", stat="latency", new_name="average latency ort", unit="s"
    ),
    dict(
        cat="memory",
        agg="MEAN",
        stat="peak_gpu_eager_warmup",
        new_name="average GPU peak (eager warmup)",
        unit="bytes",
    ),
    dict(
        cat="memory",
        agg="MEAN",
        stat="peak_gpu_warmup",
        new_name="average GPU peak (warmup)",
        unit="bytes",
    ),
    dict(
        cat="memory",
        agg="MEAN",
        stat="peak_cpu_pp",
        new_name="average CPU peak",
        unit="Mb",
    ),
    dict(
        cat="memory",
        agg="MEAN",
        stat="peak_gpu_pp",
        new_name="average GPU peak",
        unit="Mb",
    ),
    dict(
        cat="memory",
        agg="MEAN",
        stat="peak_gpu_export",
        new_name="average GPU peak (export)",
        unit="bytes",
    ),
    dict(
        cat="speedup",
        agg="GEO-MEAN",
        stat="increase",
        new_name="average speedup increase",
        unit="%",
    ),
]


def _key(v):
    if isinstance(v, (int, float)):
        return v, ""
    if isinstance(v, str):
        return 1e10, v
    if isinstance(v, tuple):
        return tuple([1e10, *v])
    raise AssertionError(f"Unexpected type for v={v!r}, type is {type(v)}")


def sort_index_key(index):
    import pandas

    return pandas.Index(_key(v) for v in index)


def _isnan(x):
    if x is None:
        return True
    if isinstance(x, str):
        return x == ""
    try:
        return np.isnan(x)
    except TypeError:
        return False


def _isinf(x):
    if x is None:
        return True
    if isinstance(x, str):
        return x == "inf"
    try:
        return np.isinf(x)
    except TypeError:
        return False


def _apply_excel_style(
    res: Dict[str, "pandas.DataFrame"],  # noqa: F821
    writer: "ExcelWriter",  # noqa: F821
):
    from openpyxl.styles import Font, Alignment, numbers, PatternFill

    bold_font = Font(bold=True)
    alignment = Alignment(horizontal="left")
    center = Alignment(horizontal="center")
    right = Alignment(horizontal="right")
    red = Font(color="FF0000")
    yellow = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
    redf = PatternFill(start_color="FF0000", end_color="FF0000", fill_type="solid")
    lasts = {}
    for k, v in res.items():
        sheet = writer.sheets[k]
        if 0 in v.shape:
            continue

        if k == "0main":
            for c in "AB":
                sheet.column_dimensions[c].width = 40
                sheet.column_dimensions[c].alignment = alignment
            for cell in sheet[1]:
                cell.font = bold_font
                cell.alignment = alignment
            continue

        if k in {"0raw", "AGG", "AGG2"}:
            continue

        n_cols = (
            1
            if isinstance(v.index[0], (str, int, np.int64, np.int32))
            else len(v.index[0])
        )
        n_rows = (
            1
            if isinstance(v.columns[0], (str, int, np.int64, np.int32))
            else len(v.columns[0])
        )

        for i in range(n_cols):
            sheet.column_dimensions["ABCDEFGHIJKLMNOPQRSTUVWXYZ"[i]].width = 40

        first_row = None
        first_col = None
        look = v.iloc[0, 0]

        values = []
        for row in sheet.iter_rows(
            min_row=1,
            max_row=n_rows + n_cols + 1,
            min_col=1,
            max_col=n_rows + n_cols + 1,
        ):
            for cell in row:
                if hasattr(cell, "col_idx") and (
                    cell.value == look
                    or (_isnan(cell.value) and _isnan(look))
                    or (_isinf(cell.value) and _isinf(look))
                ):
                    first_row = cell.row
                    first_col = cell.col_idx if hasattr(cell, "col_idx") else first_col
                    break
                values.append(cell.value)
            if first_row is not None:
                break

        assert first_row is not None and first_col is not None, (
            f"Unable to find the first value in {k!r}, first_row={first_row}, "
            f"first_col={first_col}, look={look!r} ({type(look)}), values={values}"
        )

        last_row = first_row + v.shape[0] + 1
        last_col = first_col + v.shape[1] + 1
        lasts[k] = (last_row, last_col)
        for row in sheet.iter_rows(
            min_row=first_row,
            max_row=last_row,
            min_col=first_col,
            max_col=last_col,
        ):
            for cell in row:
                cell.alignment = alignment
                cell.font = bold_font
        for i in range(n_rows):
            for cell in sheet[i + 1]:
                cell.font = bold_font

        if k == "memory":
            for row in sheet.iter_rows(
                min_row=first_row,
                max_row=last_row,
                min_col=first_col,
                max_col=last_col,
            ):
                for cell in row:
                    cell.number_format = numbers.FORMAT_NUMBER
            continue

        if k == "ERR":
            for i in range(n_cols + v.shape[1]):
                sheet.column_dimensions["ABCDEFGHIJKLMNOPQRSTUVWXYZ"[i]].alignment = (
                    alignment
                )
                sheet.column_dimensions["ABCDEFGHIJKLMNOPQRSTUVWXYZ"[i]].width = 50
                if i >= 25:
                    break
            continue

        if k in ("TIME_ITER", "discrepancies", "speedup"):
            c1, c2 = None, None
            qpb, qpt, fmtp = None, None, None
            if k == "TIME_ITER":
                qt = np.quantile(v.values, 0.9)
                qb = None
                fmt = numbers.FORMAT_NUMBER_00
            elif k == "speedup":
                debug_values = []
                for row in sheet.iter_rows(
                    min_row=0,
                    max_row=30,
                    min_col=first_col,
                    max_col=last_col,
                ):
                    found = None
                    for cell in row:
                        debug_values.append(cell.value)
                        if cell.value == "increase":
                            c1 = cell.col_idx if c1 is None else min(cell.col_idx, c1)
                            c2 = cell.col_idx if c2 is None else max(cell.col_idx, c2)
                            found = cell
                            last_idx = cell.col_idx
                        elif cell.value is None:
                            # merged cell
                            if found is not None:
                                last_idx += 1
                                cid = getattr(cell, "col_idx", last_idx)
                                c2 = max(cid, c2)
                        else:
                            found = None

                assert (
                    c1 is not None and c2 is not None and c1 <= c2
                ), f"Unexpected value for c1={c1}, c2={c2}\ndebug_values={debug_values}"
                # ratio
                qb = 0.98
                qt = None
                fmt = "0.0000"
                # increase
                qpb = -0.02
                qpt = None
                fmtp = "0.000%"
            else:
                qt = 0.01
                qb = None
                fmt = "0.000000"

            for row in sheet.iter_rows(
                min_row=first_row,
                max_row=last_row,
                min_col=first_col,
                max_col=last_col,
            ):
                for cell in row:
                    if cell.value is None or isinstance(cell.value, str):
                        continue
                    if c1 is not None and cell.col_idx >= c1 and cell.col_idx <= c2:
                        if qt and cell.value > qpt:
                            cell.font = red
                        if qb and cell.value < qpb:
                            cell.font = red
                        cell.number_format = fmtp
                    else:
                        if qt and cell.value > qt:
                            cell.font = red
                        if qb and cell.value < qb:
                            cell.font = red
                        cell.number_format = fmt
            continue

        if k in ("bucket", "bucket_script", "status", "op_onnx", "op_torch"):
            has_convert = [("convert" in str(c)) for c in v.columns]
            has_20 = [("-20%" in str(c)) for c in v.columns]
            assert k != "status" or any(
                has_convert
            ), f"has_convert={has_convert} but df.columns={[str(c) for c in v.columns]}"
            assert not k.startswith("bucket") or any(
                has_20
            ), f"has_20={has_20} but df.columns={[str(c) for c in v.columns]}"
            for row in sheet.iter_rows(
                min_row=first_row,
                max_row=last_row,
                min_col=first_col,
                max_col=last_col,
            ):
                for cell in row:
                    if cell.value is None or isinstance(cell.value, str):
                        continue
                    if cell.value == 0:
                        cell.value = None
                    cell.number_format = numbers.FORMAT_NUMBER
                    cell.alignment = center
                    if k == "status":
                        idx = cell.col_idx - n_cols - 1
                        if has_convert[idx]:
                            cell.fill = yellow
                    elif k in ("bucket", "bucket_script"):
                        idx = cell.col_idx - n_cols - 1
                        if has_20[idx]:
                            cell.fill = redf
            continue

        if k == "time":
            for row in sheet.iter_rows(
                min_row=first_row,
                max_row=last_row,
                min_col=first_col,
                max_col=last_col,
            ):
                for cell in row:
                    if cell.value is None or isinstance(cell.value, str):
                        continue
                    cell.number_format = "0.000000"
            continue

        if k in {"SUMMARY", "SUMMARY2"}:
            fmt = {
                "x": "0.000",
                "%": "0.000%",
                "byes": "0 000 000 000",
                "Mb": "0.000",
                "N": "0",
                "f": "0.000",
            }
            for row in sheet.iter_rows(
                min_row=first_row,
                max_row=last_row,
                min_col=first_col,
                max_col=last_col,
            ):
                for cell in row:
                    if cell.value in fmt:
                        fcell = row[cell.col_idx - 2]
                        fcell.number_format = fmt[cell.value]
                        if cell.value == "x" and (
                            not isinstance(fcell, (float, int)) or fcell.value < 1.02
                        ):
                            fcell.font = red
            cols = {}
            for row in sheet.iter_rows(
                min_row=1,
                max_row=2,
                min_col=first_col,
                max_col=last_col,
            ):
                for cell in row:
                    if isinstance(cell.value, str):
                        cols[cell.value] = cell.col_idx

            maxc = None
            done = set()
            for k, ci in cols.items():
                if k is None or isinstance(k, int):
                    continue
                c = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[ci - 1]
                if k == "order":
                    sheet.column_dimensions[c].width = 5
                    for cell in sheet[c]:
                        cell.alignment = right
                    done.add(c)
                    continue
                if k == "METRIC":
                    sheet.column_dimensions[c].width = 50
                    for cell in sheet[c]:
                        cell.alignment = alignment
                    done.add(c)
                    continue
                if k in {"exporter", "opt_patterns"} or k.startswith("version"):
                    sheet.column_dimensions[c].width = 15
                    for cell in sheet[c]:
                        cell.alignment = alignment
                    done.add(c)
                    continue
                if k == "unit":
                    sheet.column_dimensions[c].width = 7
                    for cell in sheet[c]:
                        cell.alignment = right
                    maxc = c
                    done.add(c)
                    continue

            for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                if c == maxc:
                    break
                if c in done:
                    continue
                sheet.column_dimensions[c].width = 20
                for cell in sheet[c]:
                    cell.alignment = right
            continue

    for k, v in res.items():
        if k not in lasts:
            continue
        sheet = writer.sheets[k]
        last_row, last_col = lasts[k]


def merge_benchmark_reports(
    data: Union["pandas.DataFrame", List[str], str],  # noqa: F821
    model=("suite", "model_name"),
    keys=(
        "exporter",
        "opt_patterns",
        "device",
        "dtype",
        "dynamic",
        "flag_fake_tensor",
        "flag_no_grad",
        "flag_training",
        "version",
        "version_onnx",
        "version_onnxruntime",
        "version_onnxscript",
        "version_tag",
        "version_torch",
        "version_transformers",
        "machine",
    ),
    column_keys=("stat", "exporter", "opt_patterns"),
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
    ),
    formulas=("memory_peak", "buckets", "status", "memory_delta"),
    excel_output: Optional[str] = None,
    exc: bool = True,
) -> Dict[str, "pandas.DataFrame"]:  # noqa: F821
    """
    Merges multiple files produced by bash_benchmark...

    ::

        _index,DATE,ERR_export,ITER,TIME_ITER,capability,cpu,date_start,device,device_name,...
        101Dummy-custom,2024-07-08,,0,7.119158490095288,7.0,40,2024-07-08,cuda,...
        101Dummy-script,2024-07-08,,1,6.705480073112994,7.0,40,2024-07-08,cuda,...
        101Dummy16-custom,2024-07-08,,2,6.970448340754956,7.0,40,2024-07-08,cuda,...

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
    """
    import pandas

    if isinstance(data, str):
        data = [data]

    if isinstance(data, list):
        dfs = []
        for filename in data:
            if isinstance(filename, str):
                try:
                    df = pandas.read_csv(filename)
                except FileNotFoundError as e:
                    found = glob.glob(filename)
                    if not found:
                        raise AssertionError(f"Unable to find {filename!r}") from e
                    for f in found:
                        df = pandas.read_csv(f)
                        dfs.append(df)
                    continue
            elif isinstance(filename, pandas.DataFrame):
                df = filename
            else:
                raise AssertionError(
                    f"Unexpected type {type(filename)} for one element of data"
                )
            dfs.append(df)
        df = pandas.concat(dfs, axis=0)
    elif isinstance(data, pandas.DataFrame):
        df = data
    else:
        raise AssertionError(f"Unexpected type {type(data)} for data")

    if "STEP" in df.columns:
        # Experiment is run in two step. We remove the export rows
        # if it was successful as the metrics are reported in the row with the speedup.
        if any(df["STEP"].isna()):
            vals = set(df["STEP"][~df["STEP"].isna()])
            if vals == {"last"}:
                if "ERR_export" in df.columns:
                    df = df[(df["STEP"] == "last") | ~df["ERR_export"].isna()]
                else:
                    df = df[(df["STEP"] == "last")]
            elif "ERR_export" in df.columns:
                df = df[
                    (df["STEP"].isna())
                    | (df["STEP"] != "export")
                    | ~df["ERR_export"].isna()
                ]
            else:
                df = df[(df["STEP"].isna()) | (df["STEP"] != "export")]
        elif "ERR_export" in df.columns:
            df = df[(df["STEP"] != "export") | ~df["ERR_export"].isna()]
        else:
            df = df[df["STEP"] != "export"]

    if isinstance(model, str):
        model = [model]
    elif isinstance(model, tuple):
        model = list(model)
    assert isinstance(model, list), f"Unexpected type {type(model)} for model={model}"

    # checks all columns defining a model are available
    for m in model:
        if m not in df.columns:
            df = df.copy()
            df[m] = ""

    # let's remove the empty line
    df = df[~df[model].isna().max(axis=1)].copy()

    # replace nan values
    set_columns = set(df.columns)
    for c in ["opt_patterns", "ERR_export", "ERR_warmup"]:
        if c in set_columns:
            df[c] = df[c].fillna("-")

    res = {"0raw": df}

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

    unique = {}
    for c in df.columns:
        u = df[c].unique()
        if len(u) == 1:
            unique[c] = u.tolist()[0]
    if "exporter" in unique:
        del unique["exporter"]

    main = [dict(column="dates", value=", ".join(sorted(df["DATE"].unique().tolist())))]
    for k, v in unique.items():
        main.append(dict(column=k, value=v))
    res["0main"] = pandas.DataFrame(main)
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
    full_index = [s for s in df.columns if s in set([*column_keys, *keys, *model])]
    dupli = ~df.duplicated(full_index, keep="last")
    df = df[dupli].copy()

    # formulas
    bucket_columns = []
    for expr in formulas:
        if expr == "memory_delta":
            if (
                "memory_begin" in set_columns
                and "memory_peak" in set_columns
                and "memory_end" in set_columns
            ):
                df["memory_peak_cpu_pp"] = (
                    np.maximum(df["memory_peak"], df["memory_end"]) - df["memory_begin"]
                )
                report_on.append("memory_peak_cpu_pp")
            delta_gpu = None
            for i in range(1024):
                c1 = f"memory_gpu{i}_begin"
                c2 = f"memory_gpu{i}_peak"
                if c1 in set_columns and c2 in set_columns:
                    d = df[c2] - df[c1]
                    if delta_gpu is None:
                        delta_gpu = d
                    else:
                        delta_gpu += d
                else:
                    break
            if delta_gpu is not None:
                df["memory_peak_gpu_pp"] = delta_gpu
                report_on.append("memory_peak_gpu_pp")

        if expr == "memory_peak":
            if (
                "mema_gpu_5_after_export" in set_columns
                and "mema_gpu_4_reset" in set_columns
                and "mema_gpu_1_after_loading" in set_columns
                and "mema_gpu_2_after_warmup" in set_columns
                and "mema_gpu_6_before_session" in set_columns
                and "mema_gpu_8_after_export_warmup" in set_columns
            ):
                col_name = f"{expr}_gpu_export"
                df[col_name] = df["mema_gpu_5_after_export"] - df["mema_gpu_4_reset"]
                report_on.append(col_name)

                col_name = f"{expr}_gpu_eager_warmup"
                df[col_name] = (
                    df["mema_gpu_2_after_warmup"] - df["mema_gpu_0_before_loading"]
                )
                report_on.append(col_name)

                col_name = f"{expr}_gpu_warmup"
                df[col_name] = (
                    df["mema_gpu_8_after_export_warmup"]
                    - df["mema_gpu_6_before_session"]
                )
                report_on.append(col_name)
            continue

        if expr == "status":
            if "time_export_success" in set_columns:
                df["status_convert"] = (~df["time_export_success"].isna()).astype(int)
                report_on.append("status_convert")
            if "discrepancies_abs" in set_columns:
                df["status_convert_ort"] = (~df["discrepancies_abs"].isna()).astype(int)
                mets = []
                for th, mt in itertools.product(
                    ["1e-1", "1e-2", "1e-3", "1e-4"], ["abs", "abs_0", "abs_1+"]
                ):
                    met = f"status_err{mt[3:]}<{th}"
                    mets.append(met)
                    df[met] = (
                        ~df[f"discrepancies_{mt}"].isna()
                        & (df[f"discrepancies_{mt}"] < float(th))
                    ).astype(int)
                df["status_lat<=eager+2%"] = (
                    ~df["discrepancies_abs"].isna()
                    & (df["time_latency"] <= df["time_latency_eager"] * 1.02)
                ).astype(int)
                report_on.extend(
                    [
                        "status_convert_ort",
                        *mets,
                        "status_lat<=eager+2%",
                    ]
                )
            continue

        if expr == "buckets":
            if (
                "exporter" in set_columns
                and "speedup" in set_columns
                and "script" in set(df.exporter)
                and len(set(df.exporter)) > 1
            ):
                # Do the same with the exporter as a baseline.
                keep = [*model, *new_keys, "speedup"]
                gr = df[df.exporter == "script"][keep].copy()
                gr = gr[~gr["speedup"].isna()]
                gr["speedup_script"] = gr["speedup"]
                gr = gr.drop("speedup", axis=1)
                on = [k for k in keep[:-1] if k != "exporter"]
                joined = pandas.merge(df, gr, left_on=on, right_on=on, how="left")
                joined["speedup_increase_script"] = (
                    joined["speedup"] / joined["speedup_script"] - 1
                ).fillna(np.inf)
                assert joined.shape[0] == df.shape[0], (
                    f"Join issue df.shape={df.shape}, joined.shaped={joined.shape}, "
                    f"gr.shape={gr.shape}"
                )
                df = joined.drop("exporter_y", axis=1).copy()
                df["exporter"] = df["exporter_x"]
                df = df.drop("exporter_x", axis=1)
                set_columns = set(df.columns)
                df["status_lat<=script+2%"] = (
                    df["speedup_increase_script"] >= (1 / 1.02 - 1)
                ).astype(int)
                report_on.append("status_lat<=script+2%")

            for c in ["speedup_increase", "speedup_increase_script"]:
                if c not in set_columns:
                    continue
                scale = [-np.inf, -20, -10, -5, -2, 2, 5, 10, 20, np.inf]
                for i in range(1, len(scale)):
                    val = (df[c] >= scale[i - 1] / 100) & (df[c] < scale[i] / 100)
                    v1 = f"{scale[i-1]}%" if not np.isinf(scale[i - 1]) else ""
                    v2 = f"{scale[i]}%" if not np.isinf(scale[i]) else ""
                    d = f"[{v1},{v2}[" if v1 and v2 else (f"<{v2}" if v2 else f">={v1}")
                    if c == "speedup_increase_script":
                        d = f"script {d}"
                    bucket_columns.append(d)
                    df[d] = val.astype(int)
            continue

    # values
    for c in report_on:
        keep = [*model, *new_keys, c]
        dfc = df[keep]
        dfc = dfc[~dfc[model].isna().min(axis=1)]
        if new_keys:
            pivot = dfc.pivot(index=model, columns=new_keys, values=c)
        else:
            pivot = dfc.set_index(model)
        res[c] = pivot

    # buckets
    if bucket_columns:
        keepc = [*new_keys, *model, *bucket_columns, "speedup_increase"]
        table = df[keepc].copy()
        pcolumns = [c for c in new_keys if c not in model]
        index_col = model
        pivot = table[~table[index_col[0]].isna()].pivot(
            index=index_col, columns=pcolumns, values=bucket_columns
        )

        # the following code switches places between exporter and buckets
        tpiv = pivot.T.reset_index(drop=False)

        def _order(index):
            if index.name in pcolumns:
                return index
            order = [
                "<-20%",
                "[-20%,-10%[",
                "[-10%,-5%[",
                "[-5%,-2%[",
                "[-2%,2%[",
                "[2%,5%[",
                "[5%,10%[",
                "[10%,20%[",
                ">=20%",
                "script <-20%",
                "script [-20%,-10%[",
                "script [-10%,-5%[",
                "script [-5%,-2%[",
                "script [-2%,2%[",
                "script [2%,5%[",
                "script [5%,10%[",
                "script [10%,20%[",
                "script >=20%",
            ]
            position = {v: i for i, v in enumerate(order)}
            return [position[s] for s in index]

        if "level_0" in tpiv.columns:
            tpiv1 = tpiv[~tpiv.level_0.str.startswith("script")]
            tpiv2 = tpiv[tpiv.level_0.str.startswith("script")].copy()
            tpiv1 = (
                tpiv1.set_index([*pcolumns, "level_0"]).sort_index(key=_order).T.copy()
            )
            res["bucket"] = tpiv1.fillna(0).astype(int)

            if tpiv2.shape[0] > 0:
                tpiv2["level_0"] = tpiv2["level_0"].apply(lambda s: s[len("script ") :])
                tpiv2 = (
                    tpiv2.set_index([*pcolumns, "level_0"])
                    .sort_index(key=_order)
                    .T.copy()
                )
                res["bucket_script"] = tpiv2.fillna(0).astype(int)

    # let's remove empty variables
    for k, v in res.items():
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
                m = m.T.stack(level=list(range(len(m.index.names)))).reset_index(
                    drop=False
                )
            cols = m.columns
            assert len(cols) >= 4, (
                f"Unexpected number of columns in {cols}, "
                f"prefix={prefix!r}, m.columns={m.columns}, "
                f"m0.index.names={m0.index.names}, "
                f"m0.columns.names={m0.columns.names}\n---\n{m0}"
            )
            exporter_column = [c for c in cols if c in ("exporter", "opt_patterns")]
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
        "mempeak_",
        "speedup_",
    ]:
        merge = [k for k in res if k.startswith(prefix)]
        merge.sort()
        if len(merge) == 0:
            continue

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
        res[prefix[:-1]] = sheet
        res = {k: v for k, v in res.items() if k not in set(merge)}

    # try to use numerical value everywhere
    for k, v in res.items():
        if k in {"0main"}:
            continue
        for c in v.columns:
            if "output_names" in c or "input_names" in c:
                continue
            cc = v[c]
            if cc.dtype == np.object_:
                try:
                    dd = cc.astype(float)
                    v[c] = dd
                except ValueError:
                    types = [
                        type(_)
                        for _ in cc
                        if not isinstance(_, float) or not np.isnan(_)
                    ]
                    if set(types) == {str}:
                        continue
                    co = Counter(types)
                    assert str in co and co[str] >= len(types) // 4 + 1, (
                        f"Unexpected values for k={k!r}, columns={c!r}, "
                        f"types={set(types)}, values={v[c]}, co={co}"
                    )

    # add pages for the summary
    res["AGG"], res["AGG2"] = _create_aggregation_figures(
        res,
        skip={
            "onnx",
            "ERR",
            "0raw",
            "0main",
            "op_onnx",
            "op_torch",
            "bucket",
            "bucket_script",
        },
        model=model,
    )
    assert None not in res["AGG"].index.names, (
        f"None in res['AGG'].index.names={res['AGG'].index.names}, " f"prefix='AGG'"
    )
    assert None not in res["AGG2"].index.names, (
        f"None in res['AGG2'].index.names={res['AGG2'].index.names}, " f"prefix='AGG2'"
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

    res["AGG"] = _reorder_index_level(res["AGG"], new_names + last_names, prefix="AGG")
    res["AGG2"] = _reorder_index_level(
        res["AGG2"], new_names + last_names, prefix="AGG2"
    )

    final_res = {
        k: (
            v
            if k in {"0raw", "0main", "AGG", "AGG2", "op_onnx", "op_torch"}
            else _reorder_columns_level(v, column_keys, prefix=k)
        )
        for k, v in res.items()
    }

    final_res["SUMMARY"], suites = _select_metrics(res["AGG"], select=SELECTED_FEATURES)
    final_res["SUMMARY2"], suites = _select_metrics(
        res["AGG2"], select=SELECTED_FEATURES
    )

    if excel_output:
        with pandas.ExcelWriter(excel_output) as writer:
            no_index = {"0raw", "0main", "SUMMARY"}
            for k, v in sorted(final_res.items()):
                ev = _reverse_column_names_order(v, name=k)
                frow = (
                    len(ev.columns.names) if isinstance(ev.columns.names, list) else 1
                )
                fcol = len(ev.index.names) if isinstance(ev.index.names, list) else 1

                ev.to_excel(
                    writer,
                    sheet_name=k,
                    index=k not in no_index,
                    freeze_panes=(frow, fcol) if k not in no_index else None,
                )
            _apply_excel_style(final_res, writer)

    return final_res


def _geo_mean(serie):
    return np.exp(np.log(np.maximum(serie, 1e-10)).mean())


def _reorder_columns_level(
    df: "pandas.DataFrame",  # noqa: F821
    first_level: List[str],
    prefix: Optional[str] = None,
) -> "pandas.DataFrame":  # noqa: F821
    assert None not in df.index.names, f"None in df.index.names={df.index.names}"
    assert None not in df.columns.names, f"None in df.index.names={df.columns.names}"
    assert set(df.columns.names) & set(first_level), (
        f"Nothing to sort, prefix={prefix!r} "
        f"df.columns={df.columns}, first_level={first_level}\n--\n{df}"
    )

    c_in = [c for c in first_level if c in set(df.columns.names)]
    c_out = [c for c in df.columns.names if c not in set(c_in)]
    levels = c_in + c_out
    for i in range(len(levels)):
        if levels[i] == df.columns.names[i]:
            continue
        j = list(df.columns.names).index(levels[i])
        df = df.swaplevel(i, j, axis=1)
    assert (
        list(df.columns.names) == levels
    ), f"Issue levels={levels}, df.columns.names={df.columns.names}"
    assert None not in df.index.names, f"None in df.index.names={df.index.names}"
    assert None not in df.columns.names, f"None in df.index.names={df.columns.names}"
    return df.sort_index(axis=1)


def _sort_index_level(
    df: "pandas.DataFrame",  # noqa: F821
    debug: Optional[str] = None,
) -> "pandas.DataFrame":  # noqa: F821
    assert (
        None not in df.index.names
    ), f"None in df.index.names={df.index.names}, debug={debug!r}"
    assert (
        df.columns.names == [None] or None not in df.columns.names
    ), f"None in df.columns.names={df.columns.names}, debug={debug!r}"
    levels = list(df.index.names)
    levels.sort()
    for i in range(len(levels)):
        if levels[i] == df.index.names[i]:
            continue
        j = list(df.index.names).index(levels[i])
        df = df.swaplevel(i, j, axis=0)
    assert (
        list(df.index.names) == levels
    ), f"Issue levels={levels}, df.index.names={df.index.names}, debug={debug!r}"
    assert (
        None not in df.index.names
    ), f"None in df.index.names={df.index.names}, debug={debug!r}"
    assert (
        df.columns.names == [None] or None not in df.columns.names
    ), f"None in df.columns.names={df.columns.names}, debug={debug!r}"
    return df.sort_index(axis=0)


def _reorder_index_level(
    df: "pandas.DataFrame",  # noqa: F821
    first_level: List[str],
    prefix: Optional[str] = None,
) -> "pandas.DataFrame":  # noqa: F821
    assert (
        None not in df.index.names
    ), f"None in df.index.names={df.index.names}, prefix={prefix!r}"
    assert (
        None not in df.columns.names
    ), f"None in df.index.names={df.columns.names}, prefix={prefix!r}"
    assert set(df.index.names) & set(first_level), (
        f"Nothing to sort, prefix={prefix!r} "
        f"df.columns={df.index}, first_level={first_level}"
    )

    c_in = [c for c in first_level if c in set(df.index.names)]
    c_out = [c for c in df.index.names if c not in set(c_in)]
    levels = c_in + c_out
    for i in range(len(levels)):
        if levels[i] == df.index.names[i]:
            continue
        j = list(df.index.names).index(levels[i])
        df = df.swaplevel(i, j, axis=0)
    assert (
        list(df.index.names) == levels
    ), f"Issue levels={levels}, df.columns.names={df.index.names}"
    assert None not in df.index.names, f"None in df.index.names={df.index.names}"
    assert None not in df.columns.names, f"None in df.index.names={df.columns.names}"
    return df.sort_index(axis=0)


def _add_level(
    index: "pandas.MultiIndex", name: str, value: str  # noqa: F821
) -> "pandas.MultiIndex":  # noqa: F821
    import pandas

    if len(index.names) == 1:
        v = index.tolist()
        nv = [[value, _] for _ in v]
        nn = [name, index.names[0]]
        aa = np.array(nv).T.tolist()
        new_index = pandas.MultiIndex.from_arrays(aa, names=nn)
        return new_index

    v = index.tolist()
    nv = [[value, *_] for _ in v]
    nn = [name, *index.names]
    aa = np.array(nv).T.tolist()
    new_index = pandas.MultiIndex.from_arrays(aa, names=nn)
    return new_index


def _create_aggregation_figures(
    final_res: Dict[str, "pandas.DataFrame"],  # noqa: F821
    model: List[str],
    skip: Optional[Set[str]] = None,
    key: str = "suite",
) -> Dict[str, "pandas.DataFrame"]:  # noqa: F821
    import pandas

    assert key in model, f"Key {key!r} missing in model={model!r}"
    model_not_key = [c for c in model if c != key]

    aggs = {}
    for k, v in final_res.items():
        if k in skip:
            continue
        assert (
            v.select_dtypes(include=np.number).shape[1] > 0
        ), f"No numeric column for k={k!r}, dtypes=\n{v.dtypes}"
        assert (
            None not in v.index.names
        ), f"None in v.index.names={v.index.names}, k={k!r}"
        assert (
            None not in v.columns.names
        ), f"None in v.columns.names={v.columns.names}, k={k!r}"

        if key not in v.index.names:
            v = v.copy()
            v[key] = "?"
            v = v.reset_index(drop=False).set_index([key, *v.index.names])

        assert (
            key in v.index.names
        ), f"Unable to find key={key} in {v.index.names} for k={k!r}"
        assert len(v.index.names) == len(model), (
            f"Length mismatch for k={k!r}, "
            f"v.index.names={v.index.names}, model={model}"
        )

        # Let's drop any non numerical features.
        v = v.select_dtypes(include=[np.number])
        # gv = v.apply(lambda x: np.log(np.maximum(x, 1e-10).values))
        v = v.reset_index(drop=False).set_index(model_not_key)

        assert key in v.columns, f"Unable to find column {key!r} in {v.columns}"

        v = v.sort_index(axis=1)

        assert (
            None not in v.index.names
        ), f"None in v.index.names={v.index.names}, k={k!r}"
        assert (
            None not in v.columns.names
        ), f"None in v.columns.names={v.columns.names}, k={k!r}"
        try:
            gr = v.groupby(key)
        except ValueError as e:
            raise AssertionError(
                f"Unable to grouby by key={key!r}, "
                f"shape={v.shape} v.columns={v.columns}, "
                f"values={set(v[key])}\n---\n{v}"
            ) from e
        stats = [
            ("MEAN", gr.mean()),
            ("MEDIAN", gr.median()),
            ("SUM", gr.sum()),
            ("MIN", gr.min()),
            ("MAX", gr.max()),
            ("COUNT", gr.count()),
            ("COUNT%", gr.agg(lambda x: x.count() / len(x))),
            ("TOTAL", gr.agg(len)),
            ("GEO-MEAN", gr.agg(_geo_mean)),
        ]
        dfs = []
        for name, df in stats:
            assert isinstance(
                df, pandas.DataFrame
            ), f"Unexpected type {type(df)} for k={k!r} and name={name!r}"
            df.index = _add_level(df.index, "agg", name)
            df.index = _add_level(df.index, "cat", k)
            assert (
                None not in df.index.names
            ), f"None in df.index.names={df.index.names}, k={k!r}, name={name!r}"
            assert (
                None not in df.columns.names
            ), f"None in df.columns.names={df.columns.names}, k={k!r}, name={name!r}"
            dfs.append(df)

        if len(dfs) == 0:
            continue
        df = pandas.concat(dfs, axis=0)
        assert df.shape[0] > 0, f"Empty set for k={k!r}"
        assert df.shape[1] > 0, f"Empty columns for k={k!r}"

        assert (
            None not in df.index.names
        ), f"None in df.index.names={df.index.names}, k={k!r}"
        assert (
            None not in df.columns.names
        ), f"None in df.columns.names={df.columns.names}, k={k!r}"
        assert isinstance(
            df, pandas.DataFrame
        ), f"Unexpected type {type(df)} for k={k!r}"

        if "stat" in df.columns.names:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                df = df.stack("stat")
            if not isinstance(df, pandas.DataFrame):
                assert (
                    "opt_patterns" not in df.index.names
                ), f"Unexpected names for df.index.names={df.index.names} (k={k!r})"
                df = df.to_frame()
                if df.columns.names == [None]:
                    df.columns = pandas.MultiIndex.from_arrays(
                        [("_dummy_",)], names=["_dummy_"]
                    )
                    assert (
                        None not in df.columns.names
                    ), f"None in df.columns.names={df.columns.names}, k={k!r}, df={df}"
            assert isinstance(
                df, pandas.DataFrame
            ), f"Unexpected type {type(df)} for k={k!r}"
            assert (
                None not in df.index.names
            ), f"None in df.index.names={df.index.names}, k={k!r}"
            assert (
                None not in df.columns.names
            ), f"None in df.columns.names={df.columns.names}, k={k!r}, df={df}"
        assert isinstance(
            df, pandas.DataFrame
        ), f"Unexpected type {type(df)} for k={k!r}"
        assert (
            None not in df.index.names
        ), f"None in df.index.names={df.index.names}, k={k!r}"
        assert (
            None not in df.columns.names
        ), f"None in df.columns.names={df.columns.names}, k={k!r}"
        aggs[f"agg_{k}"] = df

    # check stat is part of the column otherwise the concatenation fails

    set_names = set()
    for df in aggs.values():
        set_names |= set(df.index.names)

    for k, df in aggs.items():
        assert (
            None not in df.index.names
        ), f"None in df.index.names={df.index.names}, k={k!r}"
        assert (
            None not in df.columns.names
        ), f"None in df.columns.names={df.columns.names}, k={k!r}"
        if len(df.index.names) == len(set_names):
            continue
        missing = set_names - set(df.index.names)
        for g in missing:
            df.index = _add_level(df.index, g, k.replace("agg_", ""))

    aggs = {k: _sort_index_level(df, debug=k) for k, df in aggs.items()}

    # concatenation
    dfs = pandas.concat([df for df in aggs.values()], axis=0)
    assert None not in dfs.index.names, f"None in dfs.index.names={dfs.index.names}"
    assert (
        None not in dfs.columns.names
    ), f"None in dfs.columns.names={dfs.columns.names}"
    names = list(dfs.columns.names)
    dfs = dfs.unstack(key)
    keep_columns = dfs
    assert None not in dfs.index.names, f"None in dfs.index.names={dfs.index.names}"
    for n in names:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            dfs = dfs.stack(n)
        assert (
            None not in dfs.index.names
        ), f"None in dfs.index.names={dfs.index.names}, n={n!r}"

    assert None not in dfs.index.names, f"None in dfs.index.names={dfs.index.names}"
    dfs = dfs.sort_index(axis=1).sort_index(axis=0)
    keep_columns = keep_columns.sort_index(axis=1).sort_index(axis=0)
    assert None not in dfs.index.names, f"None in dfs.index.names={dfs.index.names}"
    return dfs, keep_columns


def _reverse_column_names_order(
    df: "pandas.DataFrame",  # noqa:F821:
    name: Optional[str] = None,
) -> "pandas.DataFrame":  # noqa:F821:
    if len(df.columns.names) <= 1:
        return df
    col_names = df.columns.names
    assert isinstance(col_names, list), f"Unexpected type for {df.columns.names!r}"
    return df


def _select_metrics(
    df: "pandas.DataFrame", select: List[Dict[str, str]]  # noqa: F821
) -> "pandas.DataFrame":  # noqa: F821
    suites = set(df.columns)
    rows = []
    names = list(df.index.names)
    set_names = set(names)
    for i in df.index.tolist():
        rows.append(set(dict(zip(names, i)).items()))

    subset = [
        (s, set({k: v for k, v in s.items() if k in set_names}.items())) for s in select
    ]

    keep = []
    for i, row in enumerate(rows):
        for j, (d, s) in enumerate(subset):
            if (s & row) == s:
                keep.append((i, d["new_name"], d["unit"], d.get("order", j)))
                break

    dfi = df.iloc[[k[0] for k in keep]].reset_index(drop=False).copy()
    dfi["METRIC"] = [k[1] for k in keep]
    dfi["unit"] = [k[2] for k in keep]
    dfi["order"] = [k[3] for k in keep]
    dfi = dfi.copy()
    dd_ = set(select[0].keys())
    dd = set()
    for d in dd_:
        cs = [c for c in dfi.columns if d in c]
        if cs:
            dd.add(cs[0])
    cols = [
        *[c for c in dfi.columns if "order" in c],
        *[c for c in dfi.columns if "METRIC" in c],
    ]
    skip = set([*cols, *[c for c in dfi.columns if "unit" in c]])
    for c in dfi.columns:
        if c in skip or c in dd:
            continue
        cols.append(c)
    cols.extend([c for c in dfi.columns if "unit" in c])
    dfi = dfi[cols].sort_values(cols[:-1])
    return dfi, suites
