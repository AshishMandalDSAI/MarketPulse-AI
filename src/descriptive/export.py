"""Turn a DescriptiveReport into named tables and CSV / Excel downloads (Phase 8 Part 2).

Only values the engine actually calculated are exported. An undefined statistic is
an empty cell with its reason in the ``Undefined statistics`` column - never 0.
Spreadsheet formula injection is neutralized for values (``src.common.export``) and,
here, for column names that come from the uploaded dataset.
"""

from __future__ import annotations

import io
import re
import zipfile
from typing import Optional

import pandas as pd

from src.common.export import dataframe_to_csv_bytes, sheets_to_excel_bytes, _FORMULA_PREFIXES
from src.descriptive.contracts import DescriptiveReport, PairwiseMatrix
from src.descriptive.engine import METHODOLOGY, OUTLIER_DISCLAIMER

SHEET_ORDER = ("Overview", "Numeric Statistics", "Categorical Statistics", "Boolean Statistics",
               "Datetime Statistics", "Missing Values", "Outliers", "Correlation", "Covariance",
               "Undefined Pairs", "Frequency Tables", "Methodology")


def _neutral(name) -> str:
    text = str(name)
    return "'" + text if text.startswith(_FORMULA_PREFIXES) else text


def slug(label: str, default: str = "dataset") -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", str(label or "")).strip("_")[:40]
    return s or default


def export_stem(dataset_label: str, table: Optional[str] = None) -> str:
    stem = f"descriptive_statistics_{slug(dataset_label)}"
    return f"{stem}_{slug(table).lower()}" if table else stem


def _note_frame(text: str) -> pd.DataFrame:
    return pd.DataFrame({"Note": [text]})


MAX_LISTED = 25   # a spreadsheet cell holds at most 32,767 characters, so long tie lists are shortened


def _join(items, limit: int = MAX_LISTED) -> str:
    items = list(items)
    text = "; ".join(f"{v:g}" if isinstance(v, float) else str(v) for v in items[:limit])
    return text + f"; (+{len(items) - limit:,} more tied values not listed)" if len(items) > limit else text


def _matrix_frame(m: Optional[PairwiseMatrix], empty_note: str) -> pd.DataFrame:
    if m is None or not m.columns:
        return _note_frame(empty_note)
    cols = [_neutral(c) for c in m.columns]
    frame = pd.DataFrame([[(float("nan") if v is None else v) for v in row] for row in m.values],
                         columns=cols)
    frame.insert(0, "Column", cols)
    return frame


def build_tables(report: DescriptiveReport, dataset_label: str = "", dataset_id: str = "",
                 covariance: str = "sample") -> dict:
    """Ordered {sheet name: DataFrame} for the CURRENT report and settings."""
    st = report.settings
    # Overview ---------------------------------------------------------------
    q = report.quality
    overview = pd.DataFrame([
        ("Dataset", dataset_label or "(not named)"), ("Dataset id", dataset_id or "(not given)"),
        ("Rows", report.n_rows), ("Columns", report.n_cols),
        ("Numeric columns", len(report.numeric)), ("Categorical columns", len(report.categorical)),
        ("Yes/no columns", len(report.boolean)), ("Date columns", len(report.datetime)),
        ("Identifier-like columns", len(report.identifiers)), ("Empty columns", len(report.empty_columns)),
        ("Duplicate rows", q.get("duplicate_rows", "")),
        ("Missing cells", report.missing.missing_cells if report.missing else ""),
        ("Missing cells (%)", report.missing.missing_cell_pct if report.missing else ""),
        ("Percentiles shown", _join(st.get("percentiles", []))),
        ("Outlier multiplier (k)", st.get("outlier_multiplier", "")),
        ("Correlation method", st.get("correlation_method", "")),
        ("Covariance in this file", covariance),
    ], columns=["Item", "Value"])
    overview["Value"] = overview["Value"].astype(str)

    # Numeric ----------------------------------------------------------------
    pcts = list(st.get("percentiles", []))
    rows = []
    for n in report.numeric:
        row = {"Column": n.column, "Sample size (rows)": n.sample_size, "Valid count": n.valid_count,
               "Missing count": n.missing_count, "Missing %": n.missing_pct, "Infinite count": n.infinite_count,
               "Mean": n.mean, "Median": n.median, "Mode": _join(n.modes), "Mode frequency": n.mode_count,
               "Minimum": n.minimum, "Maximum": n.maximum, "Range": n.value_range, "Sum": n.total,
               "Sample variance": n.variance_sample, "Sample std dev": n.std_sample,
               "Population variance": n.variance_population, "Population std dev": n.std_population,
               "Q1": n.q1, "Q2 (median)": n.q2, "Q3": n.q3, "IQR": n.iqr}
        for p in pcts:
            row[f"P{p:g}"] = n.percentiles.get(float(p))
        row.update({"Skewness (G1)": n.skewness, "Excess kurtosis (G2)": n.kurtosis, "Zero values": n.zero_count,
                    "Negative values": n.negative_count,
                    "Undefined statistics": " | ".join(f"{k}: {v}" for k, v in n.undefined.items()),
                    "Notes": " | ".join(n.warnings)})
        rows.append(row)
    numeric = pd.DataFrame(rows) if rows else _note_frame("This dataset has no numeric columns.")

    # Categorical / boolean / datetime ---------------------------------------
    cat_rows = [{"Column": c.column, "Sample size (rows)": c.sample_size, "Valid count": c.valid_count,
                 "Missing count": c.missing_count, "Missing %": c.missing_pct, "Unique values": c.unique_count,
                 "Mode": _join(c.modes), "Mode frequency": c.mode_count, "Mode %": c.mode_pct,
                 "Top categories": "; ".join(f"{t['value']} ({t['count']})" for t in c.top_categories),
                 "Least frequent": "; ".join(f"{t['value']} ({t['count']})" for t in c.least_frequent),
                 "Mode note": c.mode_note, "Notes": " | ".join(c.warnings)} for c in report.categorical]
    cat = pd.DataFrame(cat_rows) if cat_rows else _note_frame("This dataset has no categorical columns.")
    bool_rows = [{"Column": b.column, "Sample size (rows)": b.sample_size, "Valid count": b.valid_count,
                  "Missing count": b.missing_count, "Missing %": b.missing_pct, "True count": b.true_count,
                  "False count": b.false_count, "True %": b.true_pct, "False %": b.false_pct,
                  "Undefined statistics": " | ".join(b.undefined.values()), "Notes": " | ".join(b.warnings)}
                 for b in report.boolean]
    boolean = pd.DataFrame(bool_rows) if bool_rows else _note_frame("This dataset has no yes/no columns.")
    dt_rows = [{"Column": d.column, "Sample size (rows)": d.sample_size, "Valid count": d.valid_count,
                "Missing count": d.missing_count, "Missing %": d.missing_pct, "Invalid dates": d.invalid_count,
                "Earliest": d.earliest, "Latest": d.latest, "Range (days)": d.range_days,
                "Distinct dates": d.unique_count, "Time zone": d.timezone, "Period used": d.period,
                "Undefined statistics": " | ".join(d.undefined.values()), "Notes": " | ".join(d.warnings)}
               for d in report.datetime]
    datetime_ = pd.DataFrame(dt_rows) if dt_rows else _note_frame("This dataset has no date columns.")

    # Missing, outliers -------------------------------------------------------
    missing = (pd.DataFrame(report.missing.rows).rename(columns={
        "column": "Column", "type": "Detected type", "rows": "Rows", "missing_count": "Missing count",
        "missing_pct": "Missing %", "blank_text": "Blank text", "infinite_count": "Infinite values",
        "invalid_dates": "Invalid dates", "valid_count": "Valid count", "valid_pct": "Valid %"})
        if report.missing and report.missing.rows else _note_frame("No columns to summarize."))
    out_rows = []
    for o in report.outliers:
        out_rows.append({"Column": o.column, "Method": f"IQR fences (k = {o.multiplier:g})",
                         "Valid count": o.valid_count, "Lower bound": o.lower_bound, "Upper bound": o.upper_bound,
                         "Q1": o.q1, "Q3": o.q3, "IQR": o.iqr, "Outliers below": o.n_low, "Outliers above": o.n_high,
                         "Outlier count": o.n_outliers if o.available else None, "Outlier %": o.outlier_pct,
                         "Example rows (row: value)": "; ".join(f"{e['row']}: {e['value']:g}" for e in o.examples),
                         "Not available because": o.reason, "Notes": " | ".join(o.warnings) + (
                             " | more outliers than the examples listed" if o.examples_truncated else "")})
    outliers = pd.DataFrame(out_rows) if out_rows else _note_frame("This dataset has no numeric columns.")
    if out_rows:
        outliers["Disclaimer"] = OUTLIER_DISCLAIMER

    # Matrices ----------------------------------------------------------------
    correlation = _matrix_frame(report.correlation, "Correlation needs at least two numeric, non-identifier columns.")
    cov = report.covariance_population if covariance == "population" else report.covariance_sample
    covariance_f = _matrix_frame(cov, "Covariance needs at least two numeric, non-identifier columns.")
    undef_rows = []
    for label, m in (("Correlation", report.correlation), ("Covariance (sample)", report.covariance_sample),
                     ("Covariance (population)", report.covariance_population)):
        if m is not None:
            undef_rows += [{"Matrix": label, "Pair": _neutral(k.replace(" || ", " vs ")), "Why undefined": v}
                           for k, v in m.reasons.items() if k.split(" || ")[0] != k.split(" || ")[1]]
            undef_rows += [{"Matrix": label, "Pair": _neutral(c), "Why undefined": r} for c, r in m.excluded.items()]
    undefined_pairs = (pd.DataFrame(undef_rows) if undef_rows
                       else _note_frame("Every pair of columns has a defined value."))

    freq_rows = [{"Column": c.column, "Value": t["value"], "Count": t["count"], "% of valid values": t["pct"],
                  "Table truncated": c.frequency_truncated} for c in report.categorical for t in c.frequency]
    freq_rows += [{"Column": b.column, "Value": v, "Count": cnt, "% of valid values": pc, "Table truncated": False}
                  for b in report.boolean for v, cnt, pc in (("True", b.true_count, b.true_pct),
                                                            ("False", b.false_count, b.false_pct))]
    frequency = pd.DataFrame(freq_rows) if freq_rows else _note_frame("No categorical or yes/no columns to count.")

    method = pd.DataFrame(list(METHODOLOGY.items()) + [("Outlier flags", OUTLIER_DISCLAIMER)] +
                          [("Type decision", c) for c in report.conversions] +
                          [("Warning", w) for w in report.warnings], columns=["Topic", "Description"])
    tables = {"Overview": overview, "Numeric Statistics": numeric, "Categorical Statistics": cat,
              "Boolean Statistics": boolean, "Datetime Statistics": datetime_, "Missing Values": missing,
              "Outliers": outliers, "Correlation": correlation, "Covariance": covariance_f,
              "Undefined Pairs": undefined_pairs, "Frequency Tables": frequency, "Methodology": method}
    return {name: tables[name] for name in SHEET_ORDER}


def tables_to_excel(tables: dict) -> bytes:
    return sheets_to_excel_bytes(tables)


def tables_to_csv_zip(tables: dict) -> bytes:
    """One CSV per table in a single ZIP (UTF-8, formula-safe)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, frame in tables.items():
            zf.writestr(f"{slug(name).lower()}.csv", dataframe_to_csv_bytes(frame))
    return buf.getvalue()


def table_to_csv(tables: dict, name: str) -> bytes:
    return dataframe_to_csv_bytes(tables[name])
