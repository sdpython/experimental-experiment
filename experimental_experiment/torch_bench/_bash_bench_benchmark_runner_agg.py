from typing import Optional, Dict, List, Sequence
import numpy as np


def _apply_excel_style(
    res: Dict[str, "pandas.DataFrame"],  # noqa: F821
    writer: "ExcelWriter",  # noqa: F821
):
    from openpyxl.styles import Font, Alignment, numbers, PatternFill

    def _isnan(x):
        if x is None:
            return True
        if isinstance(x, str):
            return False
        try:
            return np.isnan(x)
        except TypeError:
            return False

    bold_font = Font(bold=True)
    alignment = Alignment(horizontal="left")
    center = Alignment(horizontal="center")
    red = Font(color="FF0000")
    gray = PatternFill(start_color="AAAAAA", end_color="AAAAAA", fill_type="solid")
    yellow = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
    redf = PatternFill(start_color="FF0000", end_color="FF0000", fill_type="solid")
    lasts = {}
    for k, v in res.items():
        sheet = writer.sheets[k]
        if k == "0main":
            for c in "AB":
                sheet.column_dimensions[c].width = 40
                sheet.column_dimensions[c].alignment = alignment
            for cell in sheet[1]:
                cell.font = bold_font
                cell.alignment = alignment
            continue
        if k == "0raw":
            continue
        n_cols = 1 if isinstance(v.index[0], str) else len(v.index[0])
        n_rows = 1 if isinstance(v.columns[0], str) else len(v.columns[0])
        for i in range(n_cols):
            sheet.column_dimensions["ABCDEFGHIJ"[i]].width = 40

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
                    cell.value == look or (_isnan(cell.value) and _isnan(look))
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
        if k in ("TIME_ITER", "discrepancies", "speedup", "speedup_increase"):
            if k == "TIME_ITER":
                qt = np.quantile(v.values, 0.9)
                qb = None
                fmt = numbers.FORMAT_NUMBER_00
            elif k == "speedup":
                qb = 0.98
                qt = None
                fmt = "0.000"
            elif k == "speedup_increase":
                qb = -0.02
                qt = None
                fmt = "0.00%"
            else:
                qt = 0.01
                qb = None
                fmt = "0.00000"

            for row in sheet.iter_rows(
                min_row=first_row,
                max_row=last_row,
                min_col=first_col,
                max_col=last_col,
            ):
                for cell in row:
                    if cell.value is None or isinstance(cell.value, str):
                        continue
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
                    cell.number_format = "0.000"
            continue
    for k, v in res.items():
        if k not in lasts:
            continue
        sheet = writer.sheets[k]
        last_row, last_col = lasts[k]

        if (
            "~MEAN" in v.index
            or "~GMEAN" in v.index
            or "~COUNT" in v.index
            or "~SUM" in v.index
            or "~COUNT" in v.index
            or "~MED" in v.index
        ):
            for row in sheet.iter_rows(
                min_row=first_row,
                max_row=last_row,
                min_col=1,
                max_col=last_col,
            ):
                for cell in row:
                    if sheet.cell(row=cell.row, column=1).value in {
                        "~MEAN",
                        "~GMEAN",
                        "~SUM",
                        "~COUNT",
                        "~MED",
                    }:
                        cell.fill = gray
                        cell.font = bold_font
                        if "." not in cell.number_format:
                            cell.number_format = "0.000"


def merge_benchmark_reports(
    data: List[str],
    model="model_name",
    keys=(
        "suite",
        "exporter",
        "opt_patterns",
        "device",
        "dtype",
        "dynamic",
        "version",
        "version_onnxruntime",
        "version_torch",
        "version_transformers",
        "flag_fake_tensor",
        "flag_no_grad",
        "flag_training",
        "machine",
    ),
    column_keys=("stat", "exporter", "opt_patterns"),
    report_on=(
        "speedup",
        "speedup_increase",
        "discrepancies_*",
        "TIME_ITER",
        "time_*",
        "ERR_*",
        "onnx_*",
        "op_*",
        "memory_*",
        "mem_*",
    ),
    formulas=("memory_peak_load", "buckets", "status", "memory_delta"),
    excel_output: Optional[str] = None,
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
        filename
        flag_fake_tensor
        flag_no_grad
        flag_training
        has_cuda
        input_size
        machine
        mema_gpu_0_before_loading
        mema_gpu_1_after_loading
        mema_gpu_2_after_warmup
        mema_gpu_3_empty_cache
        mema_gpu_4_after_repeat
        mema_gpu_5_after_export
        mema_gpu_6_after_gcollect
        mema_gpu_7_after_session
        mema_gpu_8_after_export_warmup
        mema_gpu_9_after_export_repeat
        memory_begin,
        memory_end,
        memory_gpu0_begin,
        memory_gpu0_end,
        memory_gpu0_mean,
        memory_gpu0_n,
        memory_gpu0_peak,
        memory_mean,
        memory_n,
        memory_peak,
        model,
        model_name,
        onnx_filesize
        onnx_input_names,
        onnx_model
        onnx_n_inputs,
        onnx_n_outputs,
        onnx_optimized,
        onnx_output_names,
        opt_patterns,
        output_data,
        output_size,
        params_dtype,
        params_size,
        process,
        processor,
        providers,
        quiet,
        repeat,
        speedup,
        speedup_increase,
        target_opset,
        time_export,
        time_load,
        time_latency,
        time_latency_eager,
        time_session,
        time_total,
        time_warmup,
        time_warmup_eager,
        verbose,
        version,
        version_onnxruntime,
        version_torch,
        version_transformers,
        warmup,
        ERROR,
        OUTPUT,
        CMD
    """
    import pandas

    dfs = []
    for filename in data:
        df = pandas.read_csv(filename)
        dfs.append(df)

    df = pandas.concat(dfs, axis=0)

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

    main = [dict(column="dates", value=", ".join(sorted(df["DATE"].unique().tolist())))]
    for k, v in unique.items():
        main.append(dict(column=k, value=v))
    res["0main"] = pandas.DataFrame(main)
    new_keys = [k for k in keys if k not in unique]

    # formulas
    bucket_columns = []
    for expr in formulas:
        if expr == "memory_delta":
            if (
                "memory_begin" in set_columns
                and "memory_peak" in set_columns
                and "memory_end" in set_columns
            ):
                df["mempeak_cpu"] = (
                    np.maximum(df["memory_peak"], df["memory_end"]) - df["memory_begin"]
                )
                report_on.append("mempeak_cpu")
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
                df["mempeak_gpu"] = delta_gpu
                report_on.append("mempeak_gpu")
        if expr == "memory_peak_load":
            if (
                "mema_gpu_5_after_export" in set_columns
                and "mema_gpu_1_after_loading" in set_columns
            ):
                df[expr] = (
                    df["mema_gpu_5_after_export"] - df["mema_gpu_1_after_loading"]
                )
                report_on.append(expr)
            continue

        if expr == "status":
            if "discrepancies_abs" in set_columns:
                df["status_convert"] = (~df["discrepancies_abs"].isna()).astype(int)
                df["status_err<1e-2"] = (
                    ~df["discrepancies_abs"].isna() & (df["discrepancies_abs"] < 1e-2)
                ).astype(int)
                df["status_err<1e-3"] = (
                    ~df["discrepancies_abs"].isna() & (df["discrepancies_abs"] < 1e-3)
                ).astype(int)
                df["status_err<1e-4"] = (
                    ~df["discrepancies_abs"].isna() & (df["discrepancies_abs"] < 1e-4)
                ).astype(int)
                df["status_lat<=eager+2%"] = (
                    ~df["discrepancies_abs"].isna()
                    & (df["time_latency"] <= df["time_latency_eager"] * 1.02)
                ).astype(int)
                report_on.extend(
                    [
                        "status_convert",
                        "status_err<1e-2",
                        "status_err<1e-3",
                        "status_err<1e-4",
                        "status_lat<=eager+2%",
                    ]
                )
            continue

        if expr == "buckets":
            if "exporter" in set_columns and "script" in set(df.exporter):
                # Do the same with the exporter as a baseline.
                keep = [model, *new_keys, "speedup"]
                gr = df[df.exporter == "script"][keep].copy()
                gr["speedup_script"] = gr["speedup"]
                gr = gr.drop("speedup", axis=1)
                on = [k for k in keep[:-1] if k != "exporter"]
                joined = pandas.merge(df, gr, left_on=on, right_on=on, how="left")
                joined["speedup_increase_script"] = (
                    joined["speedup"] / joined["speedup_script"] - 1
                ).fillna(np.inf)
                assert df.shape[0] == joined.shape[0]
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
        keep = [model, *new_keys, c]
        dfc = df[keep]
        pivot = dfc.pivot(index=model, columns=new_keys, values=c)
        res[c] = pivot

    # buckets
    if bucket_columns:
        table = df[[*new_keys, model, *bucket_columns, "speedup_increase"]].copy()
        pcolumns = [c for c in ["exporter", "opt_patterns"] if c in new_keys]
        pivot = table.pivot(
            index=[
                *[c for c in new_keys if c not in ("exporter", "opt_patterns")],
                model,
            ],
            columns=pcolumns,
            values=bucket_columns,
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

        tpiv1 = tpiv[~tpiv.level_0.str.startswith("script")]
        tpiv2 = tpiv[tpiv.level_0.str.startswith("script")].copy()
        tpiv1 = tpiv1.set_index([*pcolumns, "level_0"]).sort_index(key=_order).T.copy()
        summ = tpiv1.sum(axis=0)
        mean = tpiv1.sum(axis=0)
        tpiv1.loc["~COUNT"] = tpiv1.shape[0]
        tpiv1.loc["~SUM"] = summ
        tpiv1.loc["~MEAN"] = summ / tpiv1.loc["~COUNT"]
        res["bucket"] = tpiv1.fillna(0).astype(int)

        if tpiv2.shape[0] > 0:
            tpiv2["level_0"] = tpiv2["level_0"].apply(lambda s: s[len("script ") :])
            tpiv2 = (
                tpiv2.set_index([*pcolumns, "level_0"]).sort_index(key=_order).T.copy()
            )
            summ = tpiv2.sum(axis=0)
            mean = tpiv2.mean(axis=0)
            tpiv2.loc["~COUNT"] = tpiv2.shape[0]
            tpiv2.loc["~SUM"] = summ
            tpiv2.loc["~MEAN"] = summ / tpiv2.loc["~COUNT"]
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

    # add summary at the end
    mean_med = [
        c
        for c in res
        if c.startswith("time_")
        or c.startswith("TIME_")
        or c.startswith("onnx_")
        or c.startswith("discrepancies_")
        or c.startswith("status_")
        or c.startswith("mempeak_")
        or c.startswith("memory_")
    ]
    for c in [*mean_med, "speedup_increase"]:
        num = all(
            [
                n
                in {
                    np.float32,
                    np.float64,
                    np.dtype("float64"),
                    np.dtype("float32"),
                    np.int32,
                    np.int64,
                    np.dtype("int64"),
                    np.dtype("int32"),
                }
                for n in set(res[c].dtypes)
            ]
        )
        if c in res:
            if num:
                mean = res[c].mean(axis=0).copy()
                med = res[c].median(axis=0)
                summ = res[c].sum(axis=0)
                res[c].loc["~COUNT"] = res[c].shape[0]
                res[c].loc["~SUM"] = summ
                res[c].loc["~MEAN"] = mean
                res[c].loc["~MED"] = med
    for c in ["speedup"]:
        if c in res:
            mean = np.exp(np.log(res[c]).mean(axis=0))
            med = res[c].median(axis=0)
            summ = res[c].sum(axis=0)
            res[c].loc["~COUNT"] = res[c].shape[0]
            res[c].loc["~SUM"] = summ
            res[c].loc["~GMEAN"] = mean
            res[c].loc["~MED"] = med

    # final fusion

    def _merge(res, merge, prefix, reverse=True, transpose=False):
        m = None
        for name in merge:
            df = res[name].T
            cols = set(df.columns)
            df = df.reset_index(drop=False).copy()
            index_cols = set(df.columns) - cols
            df["stat"] = name[len(prefix) :]
            df = df.set_index([*list(index_cols), "stat"]).T
            if m is None:
                m = df
                continue
            m = pandas.merge(m, df, how="outer", left_index=True, right_index=True)

        # We need to change the columns index order.
        if reverse:
            df = m.T
            setc = set(df.columns)
            df = df.reset_index(drop=False)
            index = set(df.columns) - setc
            if index == {"stat", "exporter"}:
                m = df.set_index(["stat", "exporter"]).T
        else:
            m = m.T.sort_index().T
        if transpose:
            m = m.T.stack().reset_index(drop=False)
            cols = m.columns
            assert len(cols) >= 4, f"Unexpected number of columns in {cols}"
            assert (
                "stat" in cols and "exporter" in cols and model in cols
            ), f"Unexpeted columns {cols}"
            last = [c for c in cols if c not in {"stat", "exporter", model}]
            added_columns = [c for c in last if c in new_keys]
            last = [c for c in last if c not in new_keys]
            assert (
                len(last) == 1
            ), f"Unexpected columns in {cols}, added={added_columns}, last={last}"
            m = m.pivot(
                index="stat",
                columns=[model, "exporter", *added_columns],
                values=last[0],
            )
            m = m.T.sort_index().T
        return m

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
    ]:
        merge = [k for k in res if k.startswith(prefix)]
        if len(merge) == 0:
            continue
        res[prefix[:-1]] = _merge(
            res,
            merge,
            prefix,
            reverse=prefix != "status_",
            transpose=prefix.startswith("op_"),
        )
        res = {k: v for k, v in res.items() if k not in set(merge)}

    for c in res:
        if c.startswith("op_"):
            summ = res[c].sum(axis=0)
            res[c].loc["~SUM"] = summ

    reorder = {
        "ERR",
        "TIME_ITER",
        "discrepancies",
        "memory",
        "mempeak",
        "onnx",
        "speedup",
        "speedup_increase",
        "status",
        "time",
    }
    final_res = {
        k: _reorder_indices(
            v,
            row_keys=[k for k in keys if k not in column_keys],
            column_keys=column_keys,
            name=k,
        )
        for k, v in res.items()
        if k in reorder
    }
    final_res.update({k: v for k, v in res.items() if k not in reorder})

    if excel_output:
        with pandas.ExcelWriter(excel_output) as writer:
            no_index = {"0raw", "0main"}
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

    return res


def _reorder_indices(
    df: "pandas.DataFrame",  # noqa: F821
    row_keys: Sequence[str],
    column_keys: Sequence[str],
    name: Optional[str] = None,
) -> "pandas.DataFrame":  # noqa: F821
    col_names = (
        df.columns.names if isinstance(df.columns.names, list) else [df.columns.names]
    )
    row_names = df.index.names if isinstance(df.index.names, list) else [df.index.names]
    column_keys = [k for k in column_keys if k in (set(col_names) | set(row_names))]

    if set(col_names) == set(column_keys):
        m = df.T.stack().reset_index(drop=False)
        first = list(column_keys)
        second = [c for c in m.columns if c not in first]
        m = m[first + second]
        m = m.set_index(first).sort_index().reset_index(drop=False)

        value = m.columns[-1]
        row = [c for c in m.columns[:-1] if c not in first]
        piv = m.pivot(index=row, columns=first, values=value)
        piv = piv.sort_index()

        return piv.copy()

    raise AssertionError(
        f"Not implemented for row_names={row_names!r}, "
        f"col_names={col_names!r}, column_keys={column_keys!r}, row_keys={row_keys!r}"
    )


def _reverse_column_names_order(
    df: "pandas.DataFrame",  # noqa:F821:
    name: Optional[str] = None,
) -> "pandas.DataFrame":  # noqa:F821:
    if len(df.columns.names) <= 1:
        return df
    col_names = df.columns.names
    assert isinstance(col_names, list), f"Unexpected type for {df.columns.names!r}"
    return df

    # m = df.T.stack().reset_index(drop=False)
    # new_names = list(reversed(col_names))
    # m = m.set_index(new_names).T
    # return m
