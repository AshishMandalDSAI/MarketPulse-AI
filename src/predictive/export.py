"""Exports built only from results that were actually calculated."""

from __future__ import annotations

import pandas as pd

from src.common.export import ExportError, dataframe_to_csv_bytes, sheets_to_excel_bytes
from src.common.headers import neutralize_headers


def _kv(d: dict) -> pd.DataFrame:
    return pd.DataFrame({"item": list(d.keys()), "value": [str(v) for v in d.values()]})


def supervised_sheets(res) -> dict:
    if res is None or res.status != "ok":
        raise ExportError("There is no completed model result to export.")
    s = {"Summary": _kv({"target": res.target, "task": res.task, "selected_model": res.selected,
                         "selection_rule": res.selection_reason, "selection_metric": res.selection_metric,
                         "seed": res.settings.seed, "baseline_beaten": res.baseline_beaten}),
         "Model comparison": res.comparison, "Held-out test": _kv(res.heldout)}
    if res.heldout_ci:
        s["Test intervals"] = pd.DataFrame([{"metric": k, "lower": v[0], "upper": v[1]} for k, v in res.heldout_ci.items()])
    if res.confusion is not None:
        s["Confusion matrix"] = res.confusion.reset_index(names="actual")
        s["Class metrics"] = res.class_metrics
    for name, df in (("Importance", res.importance), ("Leakage review", res.final_review),
                     ("Fold screening", res.fold_audit), ("Sensitivity", res.sensitivity),
                     ("Data preparation", _kv(res.data_prep)), ("Predictions", res.predictions)):
        if df is not None and len(df):
            s[name] = df
    s["Warnings"] = pd.DataFrame({"warning": res.warnings})
    return {k: neutralize_headers(v) for k, v in s.items()}


def supervised_excel(res) -> bytes:
    return sheets_to_excel_bytes(supervised_sheets(res))


def predictions_csv(res) -> bytes:
    if res is None or res.status != "ok" or res.predictions is None:
        raise ExportError("There are no predictions to export.")
    return dataframe_to_csv_bytes(neutralize_headers(res.predictions))


def segmentation_sheets(res) -> dict:
    if res is None or res.status != "ok":
        raise ExportError("There is no completed segmentation to export.")
    s = {"Summary": _kv({**res.metrics, "features": ", ".join(res.features)}), "Profiles": res.profiles,
         "Cluster counts tested": res.k_table, "Assignments": res.assignments,
         "Data preparation": _kv(res.data_prep), "Notes": pd.DataFrame({"note": res.warnings + [res.caveat]})}
    return {k: neutralize_headers(v) for k, v in s.items()}


def segmentation_excel(res) -> bytes:
    return sheets_to_excel_bytes(segmentation_sheets(res))


def assignments_csv(res) -> bytes:
    if res is None or res.status != "ok":
        raise ExportError("There are no assignments to export.")
    return dataframe_to_csv_bytes(neutralize_headers(res.assignments))
