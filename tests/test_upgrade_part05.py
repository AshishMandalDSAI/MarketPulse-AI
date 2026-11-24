"""Upgrade PART 05 — app-wide chart/UI guard and final-audit regressions.

Every menu page (and every Statistics tab) is rendered with Streamlit AppTest for
the demo dataset (INR) and for an upload whose currency is unknown. Every Plotly
chart must have a title and, unless it is a pie/indicator/table/sunburst, both
axis titles. No chart may show a currency the dataset does not have, and no page
may raise an exception or fall into the safe-page fallback.
"""
from __future__ import annotations

import html
import json
import shutil
import uuid
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from config.settings import DEMO_DATASET_ID, dataset_processed_dir, dataset_upload_dir
from dashboard import navigation

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures" / "part02"
NO_AXES = {"pie", "indicator", "table", "sunburst"}


def _title(node) -> str:
    x = (node or {}).get("title")
    return ((x.get("text") if isinstance(x, dict) else x) or "").strip()


def _render(state: dict, label: str, tab: str | None):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=600)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    if tab:
        at.session_state[navigation.STATS_TAB_FORCE_KEY] = tab
    at.sidebar.radio[0].set_value(label).run()
    return at


def _targets():
    out = []
    for label in navigation.PAGES:
        if label == navigation.STATISTICS:
            out += [(label, x[1]) for x in navigation.STATISTICS_TABS]
        else:
            out.append((label, None))
    return out


@pytest.fixture(scope="module")
def unknown_currency_upload():
    from src.data_engine import role_engine as RE
    from src.data_engine.analysis_runner import run_full_analysis
    from src.data_engine.currency import choice_to_info, detect_currency, save_currency
    from src.data_engine.pipeline import profile_and_map

    df = pd.read_csv(FIX / "general_sales.csv")
    did = f"upload_p05{uuid.uuid4().hex[:8]}"
    _, m = profile_and_map(df)
    mm = RE.apply_user_roles(RE.suggest_roles(df), {}, targets=["Net_Sales"])
    c, o = RE.legacy_overrides(mm)
    run_full_analysis(did, df=df, mapping=m, confirmed_roles=c, overrides=o)
    std = pd.read_csv(dataset_processed_dir(did) / "standardized_data.csv")
    RE.save_mapping(dataset_processed_dir(did),
                    RE.apply_user_roles(RE.suggest_roles(std), {}, targets=["Net_Sales"]))
    save_currency(dataset_processed_dir(did),
                  choice_to_info("none", detect_currency(df), RE.monetary_columns(mm)))
    yield {"mp_active_dataset_id": did, "mp_active_dataset_filename": "general_sales.csv",
           "mp_demo_explicitly_chosen": False}
    shutil.rmtree(dataset_upload_dir(did), ignore_errors=True)
    shutil.rmtree(dataset_processed_dir(did), ignore_errors=True)


DEMO = {"mp_active_dataset_id": DEMO_DATASET_ID, "mp_demo_explicitly_chosen": True}


def _chart_problems(at, currency: str) -> list[str]:
    probs = []
    for c in at.get("plotly_chart"):
        f = json.loads(c.proto.spec)
        lay = f.get("layout", {})
        kinds = {d.get("type", "scatter") for d in f.get("data", [])}
        if not _title(lay):
            probs.append(f"chart without title {kinds}")
        if not kinds <= NO_AXES:
            if not _title(lay.get("xaxis")):
                probs.append(f"no x-axis title on '{_title(lay)}'")
            if not _title(lay.get("yaxis")):
                probs.append(f"no y-axis title on '{_title(lay)}'")
        words = " ".join([_title(lay), _title(lay.get("xaxis")), _title(lay.get("yaxis"))])
        if currency != "INR" and ("₹" in words or "INR" in words):
            probs.append(f"INR shown for a non-INR dataset: {words[:100]}")
        if currency != "USD" and ("USD" in words or "$" in words):
            probs.append(f"USD shown for a non-USD dataset: {words[:100]}")
    return probs


def _page_problems(at) -> list[str]:
    probs = []
    if at.exception:
        probs.append(f"exception: {at.exception[0].value[:200]}")
    for coll in (at.markdown, at.info, at.warning, at.error, at.caption):
        for e in coll:
            w = html.unescape(str(e.value))
            if "is not available for this dataset" in w or "Unexpected error" in w:
                probs.append("safe-page fallback: " + w[:160])
    return probs


@pytest.mark.parametrize("label,tab", _targets())
def test_demo_every_page_charts_labelled(label, tab):
    at = _render(DEMO, label, tab)
    probs = _page_problems(at) + _chart_problems(at, "INR")
    assert not probs, probs


@pytest.mark.parametrize("label,tab", _targets())
def test_unknown_currency_every_page_charts_labelled(unknown_currency_upload, label, tab):
    at = _render(unknown_currency_upload, label, tab)
    probs = _page_problems(at) + _chart_problems(at, "none")
    body = " ".join(html.unescape(str(e.value)) for e in at.markdown)
    if "₹" in body:
        probs.append("₹ in page text for an unknown-currency upload")
    assert not probs, probs


def _page_label(fragment: str) -> str:
    return next(p for p in navigation.PAGES if fragment.lower() in p.lower())


def test_forecast_chart_title_and_axes_demo():
    at = _render(DEMO, _page_label("Forecast"), None)
    figs = [json.loads(c.proto.spec)["layout"] for c in at.get("plotly_chart")]
    lay = next(f for f in figs if "forecast" in _title(f).lower())
    assert "history and forecast" in _title(lay)
    assert _title(lay["xaxis"]) in {"Date", "Week", "Month", "Quarter", "Year"}
    assert "(INR)" in _title(lay["yaxis"])  # demo revenue is INR


def test_forecast_chart_no_currency_when_unknown(unknown_currency_upload):
    at = _render(unknown_currency_upload, _page_label("Forecast"), None)
    for c in at.get("plotly_chart"):
        lay = json.loads(c.proto.spec)["layout"]
        y = _title(lay.get("yaxis"))
        assert "INR" not in y and "USD" not in y and "₹" not in y and "$" not in y


def test_driver_importance_chart_axes(unknown_currency_upload):
    # The bundled demo ships no Phase 5 output (baseline behaviour, documented as a
    # known limitation), so the chart is checked on an upload that has one.
    at = _render(unknown_currency_upload, _page_label("Anomalies"), None)
    figs = [json.loads(c.proto.spec)["layout"] for c in at.get("plotly_chart")]
    hits = [f for f in figs if _title(f) == "Driver Importance"]
    assert hits, [_title(f) for f in figs]
    for lay in hits:
        assert _title(lay["xaxis"]) == "Importance score"
        assert _title(lay["yaxis"]) == "Driver"


# --------------------------------------------------------------------------- security: HTML in column names
XSS = '<img src=x onerror=alert(1)>'


@pytest.fixture(scope="module")
def hostile_upload():
    from src.data_engine import role_engine as RE
    from src.data_engine.analysis_runner import run_full_analysis
    from src.data_engine.currency import choice_to_info, detect_currency, save_currency
    from src.data_engine.pipeline import profile_and_map

    df = pd.read_csv(FIX / "spend_weekly.csv").rename(columns={"TV_Spend": "TV_Spend" + XSS})
    did = f"upload_p05x{uuid.uuid4().hex[:8]}"
    _, m = profile_and_map(df)
    mm = RE.apply_user_roles(RE.suggest_roles(df), {}, targets=["Revenue"])
    c, o = RE.legacy_overrides(mm)
    run_full_analysis(did, df=df, mapping=m, confirmed_roles=c, overrides=o)
    std = pd.read_csv(dataset_processed_dir(did) / "standardized_data.csv")
    RE.save_mapping(dataset_processed_dir(did),
                    RE.apply_user_roles(RE.suggest_roles(std), {}, targets=["Revenue"]))
    save_currency(dataset_processed_dir(did),
                  choice_to_info("none", detect_currency(df), RE.monetary_columns(mm)))
    yield {"mp_active_dataset_id": did, "mp_active_dataset_filename": "x.csv",
           "mp_demo_explicitly_chosen": False}
    shutil.rmtree(dataset_upload_dir(did), ignore_errors=True)
    shutil.rmtree(dataset_processed_dir(did), ignore_errors=True)


@pytest.mark.parametrize("fragment", ["Effectiveness", "Channel Performance", "Resource Optimizer",
                                      "Scenario Simulator", "Executive", "Marketing Performance", "Statistics"])
def test_html_in_column_names_is_escaped(hostile_upload, fragment):
    """A column name containing HTML must never reach the page as live HTML."""
    at = _render(hostile_upload, _page_label(fragment), None)
    assert not at.exception
    for e in at.markdown:
        if getattr(e.proto, "allow_html", False) or getattr(e.proto, "unsafe_allow_html", False):
            assert "<img src=x" not in str(e.value), str(e.value)[:200]
