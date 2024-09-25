import time
import warnings
from typing import Dict, List, Optional, Sequence, Set

import numpy as np
import pandas
from pandas.errors import PerformanceWarning

BUCKET_SCALES = [-np.inf, -20, -10, -5, -2, 0, 2, 5, 10, 20, np.inf]
BUCKETS = [
    "<-20%",
    "[-20%,-10%[",
    "[-10%,-5%[",
    "[-5%,-2%[",
    "[-2%,0%[",
    "[0%,2%[",
    "[2%,5%[",
    "[5%,10%[",
    "[10%,20%[",
    ">=20%",
    "script <-20%",
    "script [-20%,-10%[",
    "script [-10%,-5%[",
    "script [-5%,-2%[",
    "script [-2%,0%[",
    "script [0%,2%[",
    "script [2%,5%[",
    "script [5%,10%[",
    "script [10%,20%[",
    "script >=20%",
]


def _SELECTED_FEATURES():
    features = [
        dict(
            cat="status",
            stat="date",
            agg="MAX",
            new_name="date",
            unit="date",
            help="Most recent date involved",
            simple=True,
        ),
        dict(
            cat="time",
            stat="ITER",
            agg="SUM",
            new_name="benchmark duration",
            unit="s",
            help="Total duration of the benchmark",
            simple=True,
        ),
        dict(
            cat="time",
            stat="ITER",
            agg="TOTAL",
            new_name="number of models",
            unit="N",
            help="Number of models evaluated in this document.",
            simple=True,
        ),
        dict(
            cat="time",
            stat="latency_eager",
            agg="COUNT",
            new_name="number of models eager mode",
            unit="N",
            help="Number of models running with eager mode",
            simple=True,
        ),
        dict(
            cat="status",
            stat="control_flow",
            agg="SUM",
            new_name="number of control flow",
            unit="N",
            help="torch.export.export does not work because of a "
            "control flow, in practice, this column means torch.export.export "
            "succeeds, this metric is only available if the data of exporter "
            "'export' or 'compile' is aggregated",
            simple=True,
        ),
        dict(
            cat="time",
            agg="COUNT",
            stat="export_success",
            new_name="export number",
            unit="N",
            help="Number of models successfully converted into ONNX. "
            "The ONNX model may not be run through onnxruntime or with "
            "significant discrepancies.",
            simple=True,
        ),
        dict(
            cat="speedup",
            agg="COUNT",
            stat="increase",
            new_name="number of running models",
            unit="N",
            help="Number of models converted and running with onnxruntime. "
            "The outputs may be right or wrong. Unit test ensures every aten functions "
            "is correctly converted but the combination may produce outputs "
            "with higher discrepancies than expected.",
            simple=True,
        ),
        dict(
            cat="status",
            agg="SUM",
            stat="accuracy_rate",
            new_name="accuracy number",
            unit="N",
            help="Number of models successfully converted into ONNX. "
            "It may be slow but the discrepancies are < 0.1.",
            simple=True,
        ),
        dict(
            cat="status",
            agg="SUM",
            stat="pass_rate",
            new_name="pass number",
            unit="N",
            help="Number of models successfully converted into ONNX, "
            "with a maximum discrepancy < 0.1 and a speedup > 0.98.",
            simple=True,
        ),
        dict(
            cat="time",
            agg="SUM",
            stat="export_success",
            new_name="total export time",
            unit="s",
            help="Total export time when the export succeeds. "
            "The model may not run through onnxruntime and the model "
            "may produce higher discrepancies than expected (lower is better).",
            simple=True,
        ),
        dict(
            cat="time",
            agg="SUM",
            stat="latency",
            new_name="total time exported model",
            unit="x",
            help="Total latency time with the exported model "
            "(onnxruntime, inductor, ...)",
            simple=True,
        ),
        dict(
            cat="time",
            agg="SUM",
            stat="latency_eager_if_exported_run",
            new_name="total time eager / exported model",
            unit="x",
            help="Total latency of eager mode knowing when the "
            "exported model (onnxruntime, inductor, ...) runs",
            simple=True,
        ),
        dict(
            cat="status",
            agg="SUM",
            stat="lat<=script+2%",
            new_name="number of models equal or faster than torch.script",
            unit="N",
            help="Number of models successfully converted with torch.script "
            "and the other exporter, and the second exporter is as fast or faster "
            "than torch.script.",
            simple=True,
        ),
        dict(
            cat="status",
            agg="SUM",
            stat="lat<=eager+2%",
            new_name="number of models equal or faster than eager",
            unit="N",
            help="Number of models as fast or faster than torch eager mode.",
            simple=True,
        ),
        dict(
            cat="status",
            agg="SUM",
            stat="lat<=inductor+2%",
            new_name="number of models equal or faster than inductor",
            unit="N",
            help="Number of models equal or faster than inductor (fullgraph=True)",
            simple=True,
        ),
        # average
        dict(
            cat="time",
            agg="MEAN",
            stat="export_success",
            new_name="average export time",
            unit="s",
            help="Average export time when the export succeeds. "
            "The model may not run through onnxruntime and the model "
            "may produce higher discrepancies than expected (lower is better).",
            simple=True,
        ),
        dict(
            cat="speedup",
            agg="GEO-MEAN",
            stat="1speedup",
            new_name="average speedup (geo)",
            unit="x",
            help="Geometric mean of all speedup for all model converted and runnning.",
            simple=True,
        ),
        # e-1
        dict(
            cat="status",
            agg="SUM",
            stat="err<1e-1",
            new_name="discrepancies < 0.1",
            unit="N",
            help="Number of models for which the maximum discrepancies is "
            "below 0.1 for all outputs.",
            simple=True,
        ),
        # e-2
        dict(
            cat="status",
            agg="SUM",
            stat="err<1e-2",
            new_name="discrepancies < 0.01",
            unit="N",
            help="Number of models for which the maximum discrepancies is "
            "below 0.01 for all outputs.",
            simple=True,
        ),
        dict(
            cat="time",
            stat="ITER",
            agg="MEAN",
            new_name="average iteration time",
            unit="s",
            help="Average total time per model and scenario. "
            "It usually reflects how long the export time is (lower is better).",
        ),
        dict(
            cat="discrepancies",
            stat="avg",
            agg="MEAN",
            new_name="average average discrepancies",
            unit="f",
            help="Average of average absolute discrepancies "
            "assuming it can be measured (lower is better).",
        ),
        dict(
            cat="discrepancies",
            stat="abs",
            agg="MEAN",
            new_name="average absolute discrepancies",
            unit="f",
            help="Average maximum absolute discrepancies "
            "assuming it can be measured (lower is better).",
        ),
        dict(
            cat="time",
            agg="MEAN",
            stat="latency_eager",
            new_name="average latency eager",
            unit="s",
            help="Average latency for eager mode (lower is better)",
        ),
        dict(
            cat="time",
            agg="MEAN",
            stat="latency",
            new_name="average latency ort",
            unit="s",
            help="Average latency for onnxruntime (lower is better)",
        ),
        dict(
            cat="onnx",
            agg="MEAN",
            stat="weight_size_torch",
            new_name="average weight size",
            unit="bytes",
            help="Average parameters size, this gives a kind of order "
            "of magnitude for the memory peak",
        ),
        dict(
            cat="onnx",
            agg="MEAN",
            stat="weight_size_proto",
            new_name="average weight size in ModelProto",
            unit="bytes",
            help="Average parameters size in the model proto, "
            "this gives a kind of order of magnitude for the memory peak "
            "this should be close to the parameter size",
        ),
        dict(
            cat="onnx",
            agg="MAX",
            stat="weight_size_torch",
            new_name="maximum weight size",
            unit="bytes",
            help="Maximum parameters size, "
            "useful to guess how much this machine can handle",
        ),
        dict(
            cat="onnx",
            agg="MAX",
            stat="weight_size_proto",
            new_name="maximum weight size in ModelProto",
            unit="bytes",
            help="Maximum parameters size in the model proto, "
            "useful to guess how much this machine can handle",
        ),
        dict(
            cat="memory",
            agg="MEAN",
            stat="peak_gpu_eager_warmup",
            new_name="average GPU peak (eager warmup)",
            unit="bytes",
            help="Average GPU peak while warming up eager mode (torch metric)",
        ),
        dict(
            cat="memory",
            agg="MEAN",
            stat="delta_peak_gpu_eager_warmup",
            new_name="average GPU delta peak (eager warmup)",
            unit="bytes",
            help="Average GPU peak of new allocated memory while warming up eager "
            "mode (torch metric)",
        ),
        dict(
            cat="memory",
            agg="MEAN",
            stat="peak_gpu_warmup",
            new_name="average GPU peak (warmup)",
            unit="bytes",
            help="Average GPU peak while warming up onnxruntime (torch metric)",
        ),
        dict(
            cat="memory",
            agg="MEAN",
            stat="delta_peak_gpu_warmup",
            new_name="average GPU delta peak (warmup)",
            unit="bytes",
            help="Average GPU peak of new allocated memory while warming up "
            "onnxruntime (torch metric)",
        ),
        dict(
            cat="memory",
            agg="MEAN",
            stat="delta_peak_cpu_pp",
            new_name="average CPU delta peak (export)",
            unit="bytes",
            help="Average CPU peak of new allocated memory while converting "
            "the model (measured in a secondary process)",
        ),
        dict(
            cat="memory",
            agg="MEAN",
            stat="delta_peak_gpu_export",
            new_name="average GPU delta peak (export) (torch)",
            unit="bytes",
            help="Average GPU peak of new allocated memory while "
            "converting the model (torch metric)",
            simple=True,
        ),
        dict(
            cat="memory",
            agg="MEAN",
            stat="delta_peak_gpu_pp",
            new_name="average GPU delta peak (export) (nvidia-smi)",
            unit="bytes",
            help="Average GPU peak of new allocated memory while "
            "converting the model (measured in a secondary process)",
            simple=True,
        ),
        dict(
            cat="memory",
            agg="MEAN",
            stat="peak_gpu_export",
            new_name="average GPU peak (export)",
            unit="bytes",
            help="Average GPU peak while converting the model (torch metric)",
        ),
        dict(
            cat="memory",
            agg="MEAN",
            stat="delta_peak_gpu_export",
            new_name="average delta GPU peak (export)",
            unit="bytes",
            help="Average GPU peak of new allocated memory "
            "while converting the model (torch metric)",
            simple=True,
        ),
        dict(
            cat="speedup",
            agg="MEAN",
            stat="increase",
            new_name="average speedup increase",
            unit="%",
            help="Average speedup increase compare to eager mode.",
        ),
    ]
    for b in BUCKETS:
        s = "script" in b
        bs = b.replace("script ", "")
        ag = "torch_script" if s else "eager"
        features.append(
            dict(
                cat="bucket",
                agg="SUM",
                stat=b,
                new_name=f"speedup/script in {bs}" if s else f"speedup in {bs}",
                unit="N",
                help=f"Number of models whose speedup against {ag} "
                f"falls into this interval",
                simple=True,
            )
        )
    return features


SELECTED_FEATURES = _SELECTED_FEATURES()

FILTERS = {
    "HG": [
        "AlbertForMaskedLM",
        "AlbertForQuestionAnswering",
        "AllenaiLongformerBase",
        "BartForCausalLM",
        "BartForConditionalGeneration",
        "BertForMaskedLM",
        "BertForQuestionAnswering",
        "BlenderbotForCausalLM",
        "BlenderbotSmallForCausalLM",
        "BlenderbotSmallForConditionalGeneration",
        "CamemBert",
        "DebertaForMaskedLM",
        "DebertaForQuestionAnswering",
        "DebertaV2ForMaskedLM",
        "DebertaV2ForQuestionAnswering",
        "DistilBertForMaskedLM",
        "DistilBertForQuestionAnswering",
        "DistillGPT2",
        "ElectraForCausalLM",
        "ElectraForQuestionAnswering",
        "GPT2ForSequenceClassification",
        "GoogleFnet",
        "LayoutLMForMaskedLM",
        "LayoutLMForSequenceClassification",
        "M2M100ForConditionalGeneration",
        "MBartForCausalLM",
        "MBartForConditionalGeneration",
        "MT5ForConditionalGeneration",
        "MegatronBertForCausalLM",
        "MegatronBertForQuestionAnswering",
        "MobileBertForMaskedLM",
        "MobileBertForQuestionAnswering",
        "OPTForCausalLM",
        "PLBartForCausalLM",
        "PLBartForConditionalGeneration",
        "PegasusForCausalLM",
        "PegasusForConditionalGeneration",
        "RobertaForCausalLM",
        "RobertaForQuestionAnswering",
        "Speech2Text2ForCausalLM",
        "T5ForConditionalGeneration",
        "T5Small",
        "TrOCRForCausalLM",
        "XGLMForCausalLM",
        "XLNetLMHeadModel",
        "YituTechConvBert",
    ]
}


def _nonone_(index):
    return None not in index.names


def _key(v):
    if isinstance(v, (int, float)):
        return v, ""
    if isinstance(v, str):
        return 1e10, v
    if isinstance(v, tuple):
        return (1e10, *v)
    raise AssertionError(f"Unexpected type for v={v!r}, type is {type(v)}")


def sort_index_key(index):
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


def _column_name(ci: int) -> str:
    if ci < 26:
        return chr(65 + ci)
    a = ci // 26
    b = ci % 26
    return f"{chr(65+a)}{chr(65+b)}"


def _apply_excel_style(
    res: Dict[str, pandas.DataFrame],
    writer: pandas.ExcelWriter,
    verbose: int = 0,
):
    from openpyxl.styles import Alignment, Font, PatternFill, numbers

    bold_font = Font(bold=True)
    alignment = Alignment(horizontal="left")
    center = Alignment(horizontal="center")
    right = Alignment(horizontal="right")
    red = Font(color="FF0000")
    yellow = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
    redf = PatternFill(start_color="FF0000", end_color="FF0000", fill_type="solid")
    lasts = {}
    for k, v in res.items():
        if verbose:
            print(f"[_apply_excel_style] apply style on {k!r}, shape={v.shape}")
        sheet = writer.sheets[k]
        if 0 in v.shape:
            continue

        if k in ("0main", "0main_base"):
            for c in "AB":
                sheet.column_dimensions[c].width = 40
                sheet.column_dimensions[c].alignment = alignment
            for cell in sheet[1]:
                cell.font = bold_font
                cell.alignment = alignment
            continue

        if k in {"0raw", "AGG", "AGG2", "0raw_base"}:
            continue

        n_cols = (
            1 if isinstance(v.index[0], (str, int, np.int64, np.int32)) else len(v.index[0])
        )
        n_rows = (
            1
            if isinstance(v.columns[0], (str, int, np.int64, np.int32))
            else len(v.columns[0])
        )

        for i in range(n_cols):
            sheet.column_dimensions[_column_name(i)].width = 40

        first_row = None
        first_col = None
        if v.index.names == [None]:
            first_row = len(v.columns.names)
            first_col = len(v.columns.names)
            if verbose > 1:
                print(
                    f"[_apply_excel_style] k={k!r}, first={first_row},{first_col}, "
                    f"columns.names={v.columns.names}"
                )
        else:
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

            if verbose > 1:
                print(
                    f"[_apply_excel_style] k={k!r}, first={first_row},{first_col}, "
                    f"index.names={v.index.names}, columns.names={v.columns.names}"
                )

            assert first_row is not None and first_col is not None, (
                f"Unable to find the first value in {k!r}, first_row={first_row}, "
                f"first_col={first_col}, look={look!r} ({type(look)}), "
                f"n_rows={n_rows}, n_cols={n_cols}, values={values}, "
                f"iloc[:3,:3]={v.iloc[:3, :3]}, v.index.names={v.index.names}, "
                f"v.columns={v.columns}"
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
                sheet.column_dimensions[_column_name(i)].alignment = alignment
                sheet.column_dimensions[_column_name(i)].width = 50
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

                if not (c1 is None and c2 is None):
                    assert c1 is not None and c2 is not None and c1 <= c2, (
                        f"Unexpected value for c1={c1}, c2={c2} (k={k!r})"
                        f"\ndebug_values={debug_values}"
                    )
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

        if k in ("bucket", "status", "op_onnx", "op_torch") and v.shape[1] > 3:
            has_convert = [("convert" in str(c)) for c in v.columns]
            has_20 = [("-20%" in str(c)) for c in v.columns]
            assert k != "status" or any(has_convert), (
                f"has_convert={has_convert} but k={k!r}, "
                f"v.columns={[str(c) for c in v.columns]}"
            )
            assert not k.startswith("bucket") or any(
                has_20
            ), f"has_20={has_20} but k={k!r}, v.columns={[str(c) for c in v.columns]}"
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
                    elif k in ("bucket",):
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

        if k in {
            "SUMMARY",
            "SUMMARY2",
            "SUMMARY_base",
            "SUMMARY2_base",
            "SUMMARY2_diff",
            "SUMMARY_diff",
            "SIMPLE",
            "SIMPLE_base",
            "SIMPLE_diff",
        }:
            fmt = {
                "x": "0.000",
                "%": "0.000%",
                "bytes": "# ##0",
                "Mb": "0.000",
                "N": "0",
                "f": "0.000",
                "s": "0.0000",
                "date": "aaaa-mm-dd",
            }
            for row in sheet.iter_rows(
                min_row=first_row,
                max_row=last_row,
                min_col=first_col,
                max_col=last_col,
            ):
                for cell in row:
                    if cell.value in fmt:
                        start, end = (
                            (0, cell.col_idx) if "SUMMARY" in k else (cell.col_idx, last_col)
                        )
                        for idx in range(start, end):
                            fcell = row[idx]
                            if isinstance(fcell.value, (int, float)):
                                f = fmt[cell.value]
                                if cell.value == "x":
                                    if (
                                        isinstance(fcell.value, (float, int))
                                        and fcell.value < 0.98
                                    ):
                                        fcell.font = red
                                elif cell.value in ("f", "s", "Mb"):
                                    if fcell.value >= 1000:
                                        f = "0 000"
                                    elif fcell.value >= 10:
                                        f = "0.00"
                                    elif fcell.value >= 1:
                                        f = "0.000"
                                elif cell.value == "date" and "SIMPLE" not in k:
                                    ts = time.gmtime(fcell.value)
                                    sval = time.strftime("%Y-%m-%d", ts)
                                    fcell.value = sval
                                fcell.number_format = f

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
                c = _column_name(ci - 1)
                if k in ("order", "#order"):
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
                if k in {
                    "exporter",
                    "opt_patterns",
                    "dynamic",
                    "rtopt",
                    "suite",
                } or k.startswith("version"):
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
                if k in ("help", "~help"):
                    sheet.column_dimensions[c].width = 20
                    for cell in sheet[c]:
                        cell.alignment = alignment
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

    if verbose:
        print("[_apply_excel_style] done")


def _reorder_columns_level(
    df: pandas.DataFrame,
    first_level: List[str],
    prefix: Optional[str] = None,
) -> pandas.DataFrame:
    assert _nonone_(df.columns), f"None in {df.columns.names}, prefix={prefix!r}"
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
    assert _nonone_(df.columns), f"None in {df.columns.names}"
    return df.sort_index(axis=1)


def _sort_index_level(
    df: pandas.DataFrame,
    debug: Optional[str] = None,
) -> pandas.DataFrame:
    assert _nonone_(df.index), f"None in {df.index.names}, debug={debug!r}"
    assert df.columns.names == [None] or _nonone_(
        df.columns
    ), f"None in {df.columns.names}, debug={debug!r}"
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
    assert _nonone_(df.index), f"None in {df.index.names}, debug={debug!r}"
    assert df.columns.names == [None] or _nonone_(
        df.columns
    ), f"None in {df.columns.names}, debug={debug!r}"
    return df.sort_index(axis=0)


def _reorder_index_level(
    df: pandas.DataFrame,
    first_level: List[str],
    prefix: Optional[str] = None,
) -> pandas.DataFrame:
    assert _nonone_(df.index), f"None in {df.index.names}, prefix={prefix!r}"
    assert _nonone_(df.columns), f"None in {df.columns.names}, prefix={prefix!r}"
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
    assert _nonone_(df.index), f"None in {df.index.names}"
    assert _nonone_(df.columns), f"None in {df.columns.names}"
    return df.sort_index(axis=0)


def _add_level(
    index: pandas.MultiIndex,
    name: str,
    value: str,
) -> pandas.MultiIndex:
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
    final_res: Dict[str, pandas.DataFrame],
    model: List[str],
    skip: Optional[Set[str]] = None,
    key: str = "suite",
    exc: bool = True,
) -> Dict[str, pandas.DataFrame]:
    assert key in model, f"Key {key!r} missing in model={model!r}"
    model_not_key = [c for c in model if c != key]

    aggs = {}
    for k, v in final_res.items():
        if k in skip:
            continue
        assert (
            v.select_dtypes(include=np.number).shape[1] > 0
        ), f"No numeric column for k={k!r}, dtypes=\n{v.dtypes}"
        assert None not in v.index.names, f"None in v.index.names={v.index.names}, k={k!r}"
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
        assert len(v.index.names) == len(
            model
        ), f"Length mismatch for k={k!r}, v.index.names={v.index.names}, model={model}"

        # Let's drop any non numerical features.
        v = v.select_dtypes(include=[np.number])
        # gv = v.apply(lambda x: np.log(np.maximum(x, 1e-10).values))
        v = v.reset_index(drop=False).set_index(model_not_key)

        assert key in v.columns, f"Unable to find column {key!r} in {v.columns}"

        v = v.sort_index(axis=1)

        assert None not in v.index.names, f"None in v.index.names={v.index.names}, k={k!r}"
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

        def _geo_mean(serie):
            nonan = serie.dropna()
            if len(nonan) == 0:
                return np.nan
            res = np.exp(np.log(np.maximum(nonan, 1e-10)).mean())
            return res

        def _propnan(df, is_nan):
            df[is_nan] = np.nan
            return df

        gr_no_nan = v.fillna(0).groupby(key)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=(FutureWarning, PerformanceWarning))
            total = gr_no_nan.count()
            is_nan = gr_no_nan.count() - gr.count() == total
        stats = [
            ("MEAN", gr.mean()),
            ("MEDIAN", gr.median()),
            ("SUM", _propnan(gr.sum(), is_nan)),
            ("MIN", gr.min()),
            ("MAX", gr.max()),
            ("COUNT", _propnan(gr.count(), is_nan)),
            ("COUNT%", _propnan(gr.count() / total, is_nan)),
            ("TOTAL", _propnan(total, is_nan)),
            ("NAN", _propnan(is_nan.astype(int), is_nan)),
        ]

        if k.startswith("speedup"):
            try:
                geo_mean = gr.agg(_geo_mean)
            except ValueError as e:
                if exc:
                    raise AssertionError(
                        f"Fails for geo_mean, k={k!r}, v=\n{v.head().T}"
                    ) from e
                geo_mean = None
            if geo_mean is not None:
                stats.append(("GEO-MEAN", geo_mean))

        dfs = []
        for name, df in stats:
            # avoid date to be numbers
            updates = {}
            drops = []
            for col in df.columns:
                if (isinstance(col, tuple) and "date" in col) or col == "date":
                    if name in {"SUM", "MEDIAN"}:
                        drops.append(col)
                    elif name in {"MIN", "MAX", "MEAN", "MEDIAM"}:
                        # maybe this code will fail someday but it seems that the cycle
                        # date -> int64 -> float64 -> int64 -> date
                        # keeps the date unchanged
                        vvv = df[col]
                        if vvv.dtype not in {object, np.object_, str, np.str_}:
                            updates[col] = vvv.apply(
                                lambda d: (
                                    np.nan
                                    if np.isnan(d)
                                    else time.strftime("%Y-%m-%d", time.gmtime(d))
                                )
                            )
            if drops:
                df.drop(drops, axis=1, inplace=True)
            if updates:
                for kc, vc in updates.items():
                    df[kc] = vc

            # then continue
            assert isinstance(
                df, pandas.DataFrame
            ), f"Unexpected type {type(df)} for k={k!r} and name={name!r}"
            df.index = _add_level(df.index, "agg", name)
            df.index = _add_level(df.index, "cat", k)
            assert _nonone_(df.index), f"None in {df.index.names}, k={k!r}, name={name!r}"
            assert _nonone_(
                df.columns
            ), f"None in {df.columns.names}, k={k!r}, name={name!r}"
            dfs.append(df)

        if len(dfs) == 0:
            continue
        df = pandas.concat(dfs, axis=0)

        assert df.shape[0] > 0, f"Empty set for k={k!r}"
        assert df.shape[1] > 0, f"Empty columns for k={k!r}"
        assert _nonone_(df.index), f"None in {df.index.names}, k={k!r}"
        assert _nonone_(df.columns), f"None in {df.columns.names}, k={k!r}"
        assert isinstance(df, pandas.DataFrame), f"Unexpected type {type(df)} for k={k!r}"

        if "stat" in df.columns.names:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=(FutureWarning, PerformanceWarning))
                df = df.stack("stat")
            if not isinstance(df, pandas.DataFrame):
                assert (
                    "opt_patterns" not in df.index.names
                    and "rtopt" not in df.index.names
                    and "dynamic" not in df.index.names
                ), f"Unexpected names for df.index.names={df.index.names} (k={k!r})"
                df = df.to_frame()
                if df.columns.names == [None]:
                    df.columns = pandas.MultiIndex.from_arrays(
                        [("_dummy_",)], names=["_dummy_"]
                    )
                    assert _nonone_(
                        df.columns
                    ), f"None in {df.columns.names}, k={k!r}, df={df}"
            assert isinstance(
                df, pandas.DataFrame
            ), f"Unexpected type {type(df)} for k={k!r}"
            assert _nonone_(df.index), f"None in {df.index.names}, k={k!r}"
            assert _nonone_(df.columns), f"None in {df.columns.names}, k={k!r}, df={df}"
        assert isinstance(df, pandas.DataFrame), f"Unexpected type {type(df)} for k={k!r}"
        assert _nonone_(df.index), f"None in {df.index.names}, k={k!r}"
        assert _nonone_(df.columns), f"None in {df.columns.names}, k={k!r}"
        aggs[f"agg_{k}"] = df

    # check stat is part of the column otherwise the concatenation fails

    set_names = set()
    for df in aggs.values():
        set_names |= set(df.index.names)

    for k, df in aggs.items():
        assert _nonone_(df.index), f"None in {df.index.names}, k={k!r}"
        assert _nonone_(df.columns), f"None in {df.columns.names}, k={k!r}"
        if len(df.index.names) == len(set_names):
            continue
        missing = set_names - set(df.index.names)
        for g in missing:
            df.index = _add_level(df.index, g, k.replace("agg_", ""))

    aggs = {k: _sort_index_level(df, debug=k) for k, df in aggs.items()}

    # concatenation
    dfs = pandas.concat(list(aggs.values()), axis=0)
    assert None not in dfs.index.names, f"None in dfs.index.names={dfs.index.names}"
    assert None not in dfs.columns.names, f"None in dfs.columns.names={dfs.columns.names}"
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
    df: pandas.DataFrame,  # :
    name: Optional[str] = None,
) -> pandas.DataFrame:  # :
    if len(df.columns.names) <= 1:
        return df
    col_names = df.columns.names
    assert isinstance(col_names, list), f"Unexpected type for {df.columns.names!r}"
    return df


def _select_metrics(
    df: pandas.DataFrame,
    select: List[Dict[str, str]],
    prefix: Optional[str] = None,
) -> pandas.DataFrame:
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
                keep.append((i, d["new_name"], d["unit"], d.get("order", j), d["help"]))
                break

    dfi = df.iloc[[k[0] for k in keep]].reset_index(drop=False).copy()
    has_suite = "suite" in dfi.columns.names and "exporter" in dfi.columns.names

    def _mk(c):
        if not has_suite:
            return c
        cc = []
        for n in dfi.columns.names:
            if n == "exporter":
                cc.append(c)
            elif c in ("#order", "METRIC"):
                cc.append("")
            else:
                cc.append("~")
        return tuple(cc)

    dfi[_mk("METRIC")] = [k[1] for k in keep]
    dfi[_mk("#order")] = [k[3] for k in keep]
    dfi[_mk("unit")] = [k[2] for k in keep]
    dfi[_mk("~help")] = [k[4] for k in keep]
    dfi = dfi.copy()

    dd_ = set(select[0].keys())
    dd = set()
    for d in dd_:
        cs = [c for c in dfi.columns if d in c]
        if cs:
            dd.add(cs[0])
    cols = [
        *[c for c in dfi.columns if "#order" in c],
        *[c for c in dfi.columns if "METRIC" in c],
    ]
    skip = {*cols, *[c for c in dfi.columns if "unit" in c or "~help" in c]}
    for c in dfi.columns:
        if c in skip or c in dd:
            continue
        cols.append(c)
    cols.extend([c for c in dfi.columns if "unit" in c])
    cols.extend([c for c in dfi.columns if "~help" in c])
    dfi = dfi[cols].sort_values(cols[:-2])
    if prefix == "SIMPLE":
        # flat table
        import pandas

        col_suites = [c for c in dfi.columns if c in suites]
        dfs = []
        for cs in col_suites:
            to_drop = [c for c in col_suites if c != cs]
            if to_drop:
                df = dfi.drop(to_drop, axis=1).copy()
            else:
                df = dfi.copy()
            df["suite"] = cs
            df["value"] = df[cs]
            df = df.drop(cs, axis=1)
            df.columns = [str(c) for c in df.columns]
            dfs.append(df[~df["value"].isna()])
        dfi = pandas.concat(dfs, axis=0).reset_index(drop=True)
    return dfi.sort_index(axis=1), suites


def _filter_data(
    df: pandas.DataFrame,
    filter_in: Optional[str] = None,
    filter_out: Optional[str] = None,
) -> pandas.DataFrame:
    """
    Argument `filter` follows the syntax
    ``<column1>:<fmt1>/<column2>:<fmt2>``.

    The format is the following:

    * a value or a set of values separated by ``;``
    """
    if not filter_in and not filter_out:
        return df

    def _f(fmt):
        cond = {}
        if isinstance(fmt, str):
            cols = fmt.split("/")
            for c in cols:
                assert ":" in c, f"Unexpected value {c!r} in fmt={fmt!r}"
                spl = c.split(":")
                assert len(spl) == 2, f"Unexpected value {c!r} in fmt={fmt!r}"
                name, fil = spl
                cond[name] = FILTERS[fil] if fil in FILTERS else set(fil.split(";"))
        return cond

    if filter_in:
        cond = _f(filter_in)
        assert isinstance(cond, dict), f"Unexpected type {type(cond)} for fmt={filter_in!r}"
        for k, v in cond.items():
            if k not in df.columns:
                continue
            df = df[df[k].isin(v)]

    if filter_out:
        cond = _f(filter_out)
        assert isinstance(cond, dict), f"Unexpected type {type(cond)} for fmt={filter_out!r}"
        for k, v in cond.items():
            if k not in df.columns:
                continue
            df = df[~df[k].isin(v)]
    return df


def _select_model_metrics(
    res: Dict[str, pandas.DataFrame],
    select: List[Dict[str, str]],
    stack_levels: Sequence[str],
) -> pandas.DataFrame:
    import pandas

    concat = []
    for i, metric in enumerate(select):
        cat, stat, new_name, agg = (
            metric["cat"],
            metric["stat"],
            metric["new_name"],
            metric["agg"],
        )
        if new_name.startswith("average "):
            new_name = new_name[len("average ") :]
        if agg in {"TOTAL", "COUNT", "COUNT%", "MAX", "SUM", "NAN"}:
            continue
        name = f"{cat}_{stat}"
        if name not in res:
            continue
        df = res[name].copy()
        cols = list(df.columns)
        if len(cols) == 1:
            col = (cols[0],) if isinstance(cols[0], str) else tuple(cols[0])
            col = (i, cat, stat, new_name, *col)
            names = ["#order", "cat", "stat", "full_name", *df.columns.names]
            df.columns = pandas.MultiIndex.from_tuples([col], names=names)
            concat.append(df)
        else:
            cols = [((c,) if isinstance(c, str) else tuple(c)) for c in cols]
            cols = [(i, cat, stat, new_name, *c) for c in cols]
            names = ["#order", "cat", "stat", "full_name", *df.columns.names]
            df.columns = pandas.MultiIndex.from_tuples(cols, names=names)
            concat.append(df)
    df = pandas.concat(concat, axis=1)
    if stack_levels:
        for c in stack_levels:
            if c in df.columns.names:
                with warnings.catch_warnings():
                    warnings.simplefilter(
                        "ignore", category=(FutureWarning, PerformanceWarning)
                    )
                    df = df.stack(c, dropna=np.nan)
    df = df.sort_index(axis=1)
    return df


def _compute_correlations(
    df: pandas.DataFrame,
    model_column: List[str],
    exporter_column: List[str],
    columns: List[str],
    verbose: int = 0,
) -> Dict[str, pandas.DataFrame]:
    """
    Computes correlations metrics.

    :param df: raw data
    :param model_column: what defines a model
    :param exporter_column: what defines an exporter
    :param columns: columns to look into
    :param verbose: verbosity
    :return: dictionary of dataframes
    """
    df = df.copy()
    df["_runs_"] = 1
    assert "_runs_" not in [
        *model_column,
        *exporter_column,
        *columns,
    ], f"Name '_runs_' is already taken in {sorted(df.columns)}"

    unique_exporter = df[[*exporter_column, "_runs_"]]
    n_runs = (
        unique_exporter.groupby(exporter_column, as_index=False).sum().reset_index(drop=True)
    )

    res = {"RUNS": n_runs}
    name_i = [f"{c}_i" for c in exporter_column]
    name_j = [f"{c}_j" for c in exporter_column]
    nonans = {}

    for c in columns:
        if verbose:
            print(f"[_compute_correlations] process {c!r}")
        piv = df.pivot(index=model_column, columns=exporter_column, values=[c])

        obs = []
        for i in range(piv.shape[1]):
            for j in range(piv.shape[1]):
                ci = piv.columns[i]
                cj = piv.columns[j]
                ni = ~piv[ci].isna()
                nj = ~piv[cj].isna()
                nonan_ = (ni & nj).apply(lambda x: 1 if x else 0)
                nonan = nonan_.sum()
                sumij = (piv[ci].fillna(0) * nonan_).sum()
                o = dict(
                    zip(
                        [*name_i, *name_j, f"nonan_{c}", f"sum_{c}"],
                        [*ci[1:], *cj[1:], nonan, sumij],
                    )
                )
                if c == "time_latency":
                    # Measuring the best
                    winners = (piv[ci].fillna(0) < (piv[cj].fillna(0) * 0.98)).astype(
                        int
                    ) * nonan_
                    o["win_latency"] = winners.sum()
                elif c == "discrepancies_abs":
                    # Measuring the best
                    winners = (piv[ci].fillna(0) < (piv[cj].fillna(0) * 0.75)).astype(
                        int
                    ) * nonan_
                    o["win_disc_abs"] = winners.sum()
                obs.append(o)
                key = ci[1:], cj[1:]
                if key not in nonans:
                    nonans[key] = nonan_
                else:
                    nonans[key] &= nonan_
        res[f"c_{c}"] = pandas.DataFrame(obs)

    obs = []
    for i in range(piv.shape[1]):
        for j in range(piv.shape[1]):
            ci = piv.columns[i][1:]
            cj = piv.columns[j][1:]
            o = dict(
                zip(
                    [*name_i, *name_j, "nonan"],
                    [*ci, *cj, nonans[ci, cj].sum()],
                )
            )
            obs.append(o)
    res_join = {"nonan": pandas.DataFrame(obs)}

    for c in columns:
        if verbose:
            print(f"[_compute_correlations] process 2 {c!r}")
        piv = df.pivot(index=model_column, columns=exporter_column, values=[c])

        obs = []
        for i in range(piv.shape[1]):
            for j in range(piv.shape[1]):
                ci = piv.columns[i]
                cj = piv.columns[j]
                sumij = (piv[ci].fillna(0) * nonans[ci[1:], cj[1:]]).sum()
                o = dict(
                    zip(
                        [*name_i, *name_j, f"sum_{c}"],
                        [*ci[1:], *cj[1:], sumij],
                    )
                )
                if c == "time_latency":
                    # Measuring the best
                    winners = (piv[ci].fillna(0) < (piv[cj].fillna(0) * 0.98)).astype(
                        int
                    ) * nonans[ci[1:], cj[1:]]
                    o["win_latency"] = winners.sum()
                elif c == "discrepancies_abs":
                    # Measuring the best
                    winners = (piv[ci].fillna(0) < (piv[cj].fillna(0) * 0.75)).astype(
                        int
                    ) * nonans[ci[1:], cj[1:]]
                    o["win_disc_abs"] = winners.sum()
                obs.append(o)
        res_join[f"c_{c}"] = pandas.DataFrame(obs)

    index_columns = [*name_i, *name_j]
    joined = res_join["nonan"].set_index(index_columns)
    for c in columns:
        if verbose:
            print(f"[_compute_correlations] joins {c!r}")
        joined = joined.join(res_join[f"c_{c}"].set_index(index_columns), how="outer")
    if "win_latency" in joined.columns:
        res["LATENCY"] = pandas.pivot_table(
            joined, index=name_i, columns=name_j, values="win_latency"
        )
    if "win_disc_abs" in joined.columns:
        res["DISC_ABS"] = pandas.pivot_table(
            joined, index=name_i, columns=name_j, values="win_disc_abs"
        )
    res["JOINED"] = joined.reset_index(drop=False)
    return res
