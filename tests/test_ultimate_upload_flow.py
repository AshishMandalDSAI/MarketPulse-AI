"""Regression tests for defects found by the browser pass on an uploaded dataset.

1. Forecast page showed "not available" for any upload whose date column is not
   literally named "Date" (the loader passed the dataset's date column to
   read_csv(parse_dates=...) for a file whose date column is "Date").
2. Time-series charts on the Executive Overview drew one point per ROW, so
   row-level data (several orders per day) became a zig-zag.
3. Date Range KPI showed raw ISO timestamps ("2021-01-01T00:00:00").
4. Forecasting capability counted rows as "time points".
"""
from __future__ import annotations

import ast
import importlib
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


# ---- 1. forecast loader ----------------------------------------------------

def test_forecast_loader_reads_generic_forecast_with_custom_date_column(tmp_path):
    from dashboard.components import data_loader as dl
    processed = tmp_path / "processed"
    models = tmp_path / "models"
    processed.mkdir(); models.mkdir()
    pd.DataFrame({"Date": ["2021-01-31", "2021-02-01"], "Forecast_Revenue": [1.0, 2.0],
                  "Lower_90": [0.5, 1.5], "Upper_90": [1.5, 2.5]}).to_csv(processed / "revenue_forecast.csv",
                                                                         index=False)
    dl._forecast_worker.clear()
    out = dl._forecast_worker("upload_x", str(processed), str(models), "Order_Date", (1,), (0,))
    assert len(out) == 2
    assert pd.api.types.is_datetime64_any_dtype(out["Date"])


def test_forecast_loader_still_parses_dataset_named_date_column(tmp_path):
    from dashboard.components import data_loader as dl
    processed = tmp_path / "p"; models = tmp_path / "m"
    processed.mkdir(); models.mkdir()
    pd.DataFrame({"Week": ["2024-01-01"], "Forecast_Revenue": [3.0]}).to_csv(processed / "revenue_forecast.csv",
                                                                         index=False)
    dl._forecast_worker.clear()
    out = dl._forecast_worker("upload_y", str(processed), str(models), "Week", (2,), (0,))
    assert pd.api.types.is_datetime64_any_dtype(out["Week"])


def test_end_to_end_upload_forecast_is_loadable(tmp_path, monkeypatch):
    """run_full_analysis on row-level data with an 'Order_Date' column produces a
    forecast that the dashboard loader can read back (non-empty)."""
    import numpy as np
    from dashboard.components import data_loader as dl
    from src.data_engine import analysis_runner
    from src.data_engine.pipeline import profile_and_map

    rng = np.random.default_rng(0)
    dates = pd.date_range("2023-01-01", periods=40, freq="D")
    rows = pd.DataFrame({"Order_Date": rng.choice(dates, 400),
                         "Net_Sales": rng.uniform(100, 900, 400).round(2),
                         "Region": rng.choice(["N", "S"], 400)})
    _, mapping = profile_and_map(rows)
    did = "upload_testforecast01"
    from dashboard.components import session_paths  # noqa: F401  (paths module used by runner)
    result = analysis_runner.run_full_analysis(
        did, df=rows, mapping=mapping, confirmed_roles={"date", "revenue"},
        overrides={"date_column": "Order_Date", "revenue_column": "Net_Sales"})
    caps = result["capabilities"]
    fc = next(c for c in caps.capabilities if c.name == "Forecasting")
    assert "40 time points" in fc.reason          # distinct dates, not 400 rows
    from config.settings import dataset_processed_dir, dataset_models_dir
    p, m = dataset_processed_dir(did), dataset_models_dir(did)
    try:
        dl._forecast_worker.clear()
        out = dl._forecast_worker(did, str(p), str(m), "Order_Date", (3,), (0,))
        assert not out.empty and "Forecast_Revenue" in out.columns
    finally:
        import shutil
        shutil.rmtree(p.parent, ignore_errors=True)


# ---- 2. per-period aggregation ----------------------------------------------

def _per_period():
    src = (ROOT / "dashboard/pages/1_executive_overview.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "_per_period")
    ns: dict = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "overview", "exec"), ns)
    return ns["_per_period"]


def test_time_series_is_one_point_per_date_summed():
    f = _per_period()
    df = pd.DataFrame({"d": pd.to_datetime(["2021-01-02", "2021-01-01", "2021-01-01"]), "y": [5, 1, 2]})
    out = f(df, "d", "y")
    assert list(out["y"]) == [3, 5] and out["d"].is_monotonic_increasing


def test_time_series_mean_hint_and_unique_dates_untouched():
    f = _per_period()
    df = pd.DataFrame({"d": pd.to_datetime(["2021-01-01", "2021-01-01"]), "y": [2, 4]})
    assert list(f(df, "d", "y", "mean")["y"]) == [3]
    uniq = pd.DataFrame({"d": pd.to_datetime(["2021-01-01", "2021-01-02"]), "y": [7, None]})
    assert list(f(uniq, "d", "y")["y"]) == [7]          # missing values dropped, not zero-filled


# ---- 3. date range formatting -----------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("2021-01-01T00:00:00", "2021-01-01"),
    ("2021-01-01 13:45:00", "2021-01-01 13:45"),
    ("?", "?"),
])
def test_date_bound_formatting(raw, expected):
    from src.analytics.kpi_engine import _fmt_date_bound
    assert _fmt_date_bound(raw) == expected


# ---- 4. capability counts distinct dates -------------------------------------

def test_forecast_capability_counts_distinct_dates_not_rows():
    from src.data_engine import capability_engine as ce
    from config.constants import MIN_WEEKS_FOR_FORECAST
    src = (ROOT / "src/data_engine/capability_engine.py").read_text(encoding="utf-8")
    assert "nunique()" in src
    # few dates, many rows -> not available
    n_dates = max(MIN_WEEKS_FOR_FORECAST - 1, 1)
    dates = pd.date_range("2021-01-01", periods=n_dates, freq="W")
    df = pd.DataFrame({"Date": list(dates) * 50, "Revenue": range(n_dates * 50)})
    from src.data_engine.pipeline import profile_and_map
    from src.data_engine.data_standardizer import standardize
    _, mapping = profile_and_map(df)
    items = ce.detect_capabilities(standardize(df, mapping)).capabilities
    fc = next(c for c in items if c.name == "Forecasting")
    assert not fc.available
    assert f"Only {n_dates} time point" in fc.reason


# ---- 5. Forecast page on an upload: history line + readable dates ------------

def test_forecast_page_on_upload_shows_history_and_forecast(tmp_path):
    import json
    import shutil
    import uuid
    import numpy as np
    from streamlit.testing.v1 import AppTest
    from config.settings import dataset_processed_dir, dataset_upload_dir
    from src.data_engine.analysis_runner import run_full_analysis
    from src.data_engine.pipeline import profile_and_map

    rng = np.random.default_rng(1)
    dates = pd.date_range("2023-01-01", periods=40, freq="D")
    rows = pd.DataFrame({"Order_Date": rng.choice(dates, 400).astype("datetime64[ns]").astype(str),
                         "Net_Sales": rng.uniform(100, 900, 400).round(2)})
    _, mapping = profile_and_map(rows)
    did = f"upload_t{uuid.uuid4().hex[:10]}"
    run_full_analysis(did, df=rows, mapping=mapping, confirmed_roles={"date", "revenue"},
                      overrides={"date_column": "Order_Date", "revenue_column": "Net_Sales"})
    try:
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
        at.session_state["mp_active_dataset_id"] = did
        at.session_state["mp_active_dataset_filename"] = "orders.csv"
        at.session_state["mp_demo_explicitly_chosen"] = False
        at.run()
        label = next(o for o in at.sidebar.radio[0].options if "Forecast" in o)
        at.sidebar.radio[0].set_value(label).run()
        assert not at.exception
        assert not any("Forecast not available" in w.value for w in at.warning)
        charts = at.get("plotly_chart")
        assert charts, "forecast chart missing"
        names = [t.get("name") for t in json.loads(charts[0].proto.spec)["data"]]
        assert "Historical Net_Sales" in names and "Forecast" in names
    finally:
        shutil.rmtree(dataset_upload_dir(did), ignore_errors=True)
        shutil.rmtree(dataset_processed_dir(did), ignore_errors=True)


def test_show_table_formats_midnight_datetimes_as_dates(monkeypatch):
    from dashboard.components import ui
    captured = {}
    monkeypatch.setattr(ui.st, "dataframe", lambda df, **kw: captured.update(kw))
    ui.show_table(pd.DataFrame({"Date": pd.to_datetime(["2021-01-31", "2021-02-01"]), "v": [1.0, 2.0]}))
    cfg = captured["column_config"]["Date"]
    assert cfg["type_config"]["format"] == "YYYY-MM-DD"
