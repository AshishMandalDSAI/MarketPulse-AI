"""Upgrade PART 01: navigation, Statistics workspace, chart labelling, currency.

Test-list letters from the master specification: W navigation, V axis labels,
K currencies, D/E CSV/XLSX, F dataset switching, X forecast regression.
All data below are synthetic test fixtures built for these checks.
"""
from __future__ import annotations

import importlib
import io
import json
import math
import re
import shutil
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config.settings import dataset_processed_dir, dataset_upload_dir
from dashboard import navigation
from src.data_engine import currency as C

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app.py")

APPROVED = [
    "00 · 📤 Upload Data", "01 · 🧮 Statistics", "02 · 📊 Executive Overview", "03 · 📈 Marketing Performance",
    "04 · 🧠 Marketing Effectiveness Model", "05 · 📡 Channel Performance & Contribution",
    "06 · ⚖️ Marketing Resource Optimizer", "07 · 🧪 Marketing Scenario Simulator", "08 · 📅 Forecast",
    "09 · 🤖 AI Marketing Analyst", "10 · 💡 Recommendations", "11 · 📋 Reports", "12 · 🔮 Predictions",
    "13 · 🔍 Anomalies & Drivers", "14 · 🧠 Decision Intelligence", "15 · 🎯 Prescriptive Analytics",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _app(state=None):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=300)
    for k, v in (state or {}).items():
        at.session_state[k] = v
    return at


def _demo():
    from config.settings import DEMO_DATASET_ID
    return {"mp_active_dataset_id": DEMO_DATASET_ID, "mp_demo_explicitly_chosen": True}


def _upload_state(did, name="data.csv"):
    return {"mp_active_dataset_id": did, "mp_active_dataset_filename": name, "mp_demo_explicitly_chosen": False}


def _goto(at, label, tab=None):
    at.run()
    if tab is not None:
        at.session_state[navigation.STATS_TAB_FORCE_KEY] = tab
    at.sidebar.radio[0].set_value(label).run()
    return at


def _texts(at):
    import html
    out = []
    for coll in (at.markdown, at.info, at.warning, at.error, at.success, at.caption, at.subheader, at.title):
        out.extend(html.unescape(str(e.value)) for e in coll)
    return out


def _no_failure(at):
    assert not at.exception, at.exception
    # safe_page turns exceptions into a "... is not available" message; any exception name in it is a failure.
    bad = [t for t in _texts(at) if "Unexpected error" in t or re.search(r"\((KeyError|EmptyDataError)\b", t)]
    assert not bad, bad


def _figures(at):
    return [json.loads(c.proto.spec) for c in at.get("plotly_chart")]


def _axis_title(layout, axis):
    t = (layout.get(axis) or {}).get("title")
    if isinstance(t, dict):
        return t.get("text") or ""
    return t or ""


def _chart_title(layout):
    t = layout.get("title")
    return (t.get("text") if isinstance(t, dict) else t) or ""


def _assert_figures_labelled(at, min_figures=1):
    figs = _figures(at)
    assert len(figs) >= min_figures, f"expected at least {min_figures} chart(s), got {len(figs)}"
    for f in figs:
        layout = f.get("layout", {})
        kinds = {t.get("type", "scatter") for t in f.get("data", [])}
        assert _chart_title(layout).strip(), f"chart without title: {kinds}"
        if kinds <= {"pie"}:
            continue
        assert _axis_title(layout, "xaxis").strip(), f"no x-axis title on '{_chart_title(layout)}'"
        assert _axis_title(layout, "yaxis").strip(), f"no y-axis title on '{_chart_title(layout)}'"
    return figs


@pytest.fixture
def make_upload():
    """Run the real analysis on a synthetic frame; optionally store a currency record."""
    from src.data_engine.analysis_runner import run_full_analysis
    from src.data_engine.pipeline import profile_and_map
    made = []

    def make(df, currency_info=None, **kw):
        did = f"upload_p1{uuid.uuid4().hex[:10]}"
        _, mapping = profile_and_map(df)
        run_full_analysis(did, df=df, mapping=mapping, **kw)
        if currency_info is not None:
            C.save_currency(dataset_processed_dir(did), currency_info)
        made.append(did)
        return did

    yield make
    for did in made:
        shutil.rmtree(dataset_upload_dir(did), ignore_errors=True)
        shutil.rmtree(dataset_processed_dir(did), ignore_errors=True)


def _orders(n=400, seed=0, revenue_col="Net_Sales"):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    return pd.DataFrame({
        "Order_Date": rng.choice(dates, n).astype("datetime64[ns]").astype(str),
        revenue_col: rng.uniform(100, 900, n).round(2),
        "Units": rng.integers(1, 9, n),
        "Discount_Pct": rng.uniform(0, 30, n).round(1),
        "Region": rng.choice(["North", "South", "East", "West"], n),
        "Channel": rng.choice(["Web", "Store", "App"], n),
    })


# ---------------------------------------------------------------------------
# W · navigation
# ---------------------------------------------------------------------------

class TestNavigationRegistry:
    def test_menu_is_exactly_the_approved_structure(self):
        assert list(navigation.PAGES) == APPROVED

    def test_app_uses_registry_and_has_no_literal_labels(self):
        src = (ROOT / "app.py").read_text(encoding="utf-8")
        assert "PAGES = navigation.PAGES" in src
        assert not re.search(r'"\d\d · ', src)

    def test_no_page_hardcodes_menu_labels(self):
        offenders = []
        for p in list((ROOT / "dashboard").rglob("*.py")) + [ROOT / "src" / "ai" / "analyst.py"]:
            if p.name == "navigation.py":
                continue
            if re.search(r'"\d\d · [^"]+"', p.read_text(encoding="utf-8")):
                offenders.append(str(p.relative_to(ROOT)))
        assert offenders == []

    def test_group_headings_sit_on_their_first_page(self):
        groups = dict(navigation.NAV_GROUPS)
        assert groups == {0: "Data", 1: "Statistics", 2: "Overview", 4: "Marketing models",
                          8: "Forecast & insights", 12: "Advanced analytics"}
        from dashboard.styles.theme import THEME_CSS
        for pos, name in groups.items():
            assert f':nth-of-type({pos + 1})::before {{ content: "{name.upper()}"; }}' in THEME_CSS

    def test_every_menu_and_kept_module_imports(self):
        for module in list(navigation.PAGES.values()) + list(navigation.LEGACY_PAGES.values()):
            assert callable(getattr(importlib.import_module(module), "render"))

    def test_old_four_and_statistics_pages_are_out_of_the_menu_but_kept(self):
        for key in ("marketing_mix_model", "channel_roi", "budget_optimizer", "what_if_simulator",
                    "descriptive_statistics", "inferential_statistics", "predictive_analytics"):
            module = navigation.LEGACY_PAGES[key]
            assert module not in navigation.PAGES.values()
            assert (ROOT / (module.replace(".", "/") + ".py")).exists()

    def test_page_for_module(self):
        assert navigation.page_for_module("inferential") == (APPROVED[1], "🔬 Inferential Statistics")
        assert navigation.page_for_module("prescriptive") == (APPROVED[15], None)
        with pytest.raises(KeyError):
            navigation.page_for_module("nope")


class TestNavigationInApp:
    @pytest.mark.parametrize("label", APPROVED[1:])
    def test_every_menu_entry_renders_on_demo(self, label):
        at = _goto(_app(_demo()), label)
        _no_failure(at)
        assert at.sidebar.radio[0].value == label

    def test_every_menu_entry_renders_on_an_uploaded_dataset(self, make_upload):
        did = make_upload(_orders())
        for label in APPROVED[1:]:
            at = _goto(_app(_upload_state(did)), label)
            _no_failure(at)

    def test_stale_label_from_old_menu_falls_back_to_upload_page(self):
        at = _app({**_demo(), "mp_nav_page": "03 · 🧠 Marketing Mix Model"})
        at.run()
        _no_failure(at)
        assert at.sidebar.radio[0].value == APPROVED[0]

    def test_forced_jump_to_executive_overview(self):
        at = _app({**_demo(), "mp_force_page": navigation.EXECUTIVE_OVERVIEW})
        at.run()
        assert at.sidebar.radio[0].value == "02 · 📊 Executive Overview"

    def test_no_dataset_gate_still_blocks_pages(self):
        at = _goto(_app(), APPROVED[2])
        assert any("No dataset is currently active" in i.value for i in at.info)

    def test_prescriptive_badge_shows_menu_position(self):
        at = _goto(_app(_demo()), APPROVED[15])
        _no_failure(at)
        assert any("PAGE 15" in m.value for m in at.markdown)
        assert not any("MODULE 17" in m.value for m in at.markdown)


class TestStatisticsWorkspace:
    TABS = [t for _, t in navigation.STATISTICS_TABS]

    def test_three_tabs_with_required_names(self):
        assert self.TABS == ["📊 Descriptive Statistics", "🔬 Inferential Statistics", "🔮 Predictive Analysis"]

    def test_default_tab_is_descriptive_and_only_it_runs(self):
        at = _goto(_app(_demo()), APPROVED[1])
        _no_failure(at)
        heads = [h.value for h in at.subheader]
        assert "📊 Descriptive Statistics" in heads
        assert "🔬 Inferential Statistics" not in heads and "🔮 Predictive Analysis" not in heads

    @pytest.mark.parametrize("module_key,tab", list(navigation.STATISTICS_TABS))
    def test_each_tab_renders_its_own_module(self, module_key, tab):
        at = _goto(_app(_demo()), APPROVED[1], tab=tab)
        _no_failure(at)
        heads = [h.value for h in at.subheader]
        assert tab in heads
        assert all(t not in heads for t in self.TABS if t != tab)
        from src.common.modules import get_module_spec
        assert any(get_module_spec(module_key).title in i.value for i in at.info)   # module's own status note

    def test_tab_choice_persists_across_reruns(self):
        at = _goto(_app(_demo()), APPROVED[1], tab=self.TABS[2])
        at.run()
        assert self.TABS[2] in [h.value for h in at.subheader]

    def test_unknown_forced_tab_falls_back_without_error(self):
        at = _goto(_app(_demo()), APPROVED[1], tab="not a tab")
        _no_failure(at)
        assert "📊 Descriptive Statistics" in [h.value for h in at.subheader]

    def test_tabs_work_on_uploaded_data(self, make_upload):
        did = make_upload(_orders())
        for tab in self.TABS:
            at = _goto(_app(_upload_state(did)), APPROVED[1], tab=tab)
            _no_failure(at)
            assert tab in [h.value for h in at.subheader]


class TestInterimPages:
    PAGES = {APPROVED[4]: "Marketing Mix Model", APPROVED[5]: "Channel ROI",
             APPROVED[6]: "Budget Optimizer", APPROVED[7]: "What-If Simulator"}

    @pytest.mark.parametrize("label,legacy", list(PAGES.items()))
    def test_pending_notice_and_previous_module_not_shown_as_new(self, label, legacy):
        at = _goto(_app(_demo()), label)
        _no_failure(at)
        title = label.split(" ", 3)[3]
        text = " ".join(_texts(at))
        # PART 02 implemented these pages: the PART 01 "not implemented yet" notice is gone and the new
        # module's own content is shown (detailed checks: tests/test_upgrade_part02.py).
        assert f"{title}: not implemented yet" not in text
        assert "Detected analytics mode" in text and "What this model can do with this dataset" in text
        # The previous module does not run until the user switches it on.
        old_titles = {"Marketing Mix Model": "Marketing Mix Model (MMM)", "Channel ROI": "Channel ROI & Efficiency",
                      "Budget Optimizer": "AI Budget Optimizer", "What-If Simulator": "What-If Budget Simulator"}
        assert old_titles[legacy] not in [t.value for t in at.title]
        toggle = at.toggle[0]
        assert "previous" in toggle.label.lower() and legacy in toggle.label
        toggle.set_value(True).run()
        _no_failure(at)
        assert old_titles[legacy] in [t.value for t in at.title]
        assert any(f"Previous module · {legacy} · not the {title}" in c.value for c in at.caption)


# ---------------------------------------------------------------------------
# V · axis labels
# ---------------------------------------------------------------------------

class TestChartLabelHelpers:
    def test_humanize(self):
        from dashboard.components.chart_labels import humanize
        assert humanize("Net_Sales") == "Net Sales"
        assert humanize("totalMarketingSpend") == "Total Marketing Spend"
        assert humanize("ROI") == "ROI"

    @pytest.mark.parametrize("col,agg,expected", [
        ("Net_Sales", "sum", "Total Net Sales"), ("Net_Sales", "mean", "Average Net Sales"),
        ("Region", "count", "Number of records"), ("Discount_Pct", None, "Discount Pct (%)"),
        ("conversion_rate", "mean", "Average Conversion rate (%)"), ("Units", None, "Units"),
    ])
    def test_value_axis_title(self, col, agg, expected):
        from dashboard.components.chart_labels import value_axis_title
        assert value_axis_title(col, agg, monetary=False) == expected

    @pytest.mark.parametrize("freq,expected", [("D", "Date"), ("W-MON", "Week"), ("MS", "Month"),
                                               ("QS", "Quarter"), ("YS", "Year")])
    def test_time_axis_title_follows_frequency(self, freq, expected):
        from dashboard.components.chart_labels import time_axis_title
        assert time_axis_title(pd.date_range("2020-01-01", periods=12, freq=freq)) == expected

    def test_currency_not_repeated_when_already_in_the_name(self):
        from dashboard.components.chart_labels import value_axis_title
        assert value_axis_title("Net_Sales_USD", "sum", unit="USD") == "Total Net Sales (USD)"
        assert value_axis_title("Revenue (USD)", None, unit="USD") == "Revenue (USD)"
        assert value_axis_title("Revenue", "mean", unit="EUR") == "Average Revenue (EUR)"

    def test_chart_title_states_aggregation(self):
        from dashboard.components.chart_labels import chart_title
        assert chart_title("Net_Sales by Region", "mean", "Net_Sales") == "Average Net Sales by Region"
        assert chart_title("Net_Sales Trend", "sum", "Net_Sales") == "Total Net Sales Trend"
        assert chart_title("Net_Sales Trend", "none", "Net_Sales") == "Net_Sales Trend"

    def test_per_period_sums_and_orders(self):
        from dashboard.components.chart_labels import per_period
        df = pd.DataFrame({"d": ["2024-01-02", "2024-01-01", "2024-01-01"], "y": [5, 1, 2]})
        out, applied = per_period(df, "d", "y")
        assert applied == "sum" and list(out["y"]) == [3, 5]
        out, applied = per_period(df.drop_duplicates("d"), "d", "y")
        assert applied == "none"

    @pytest.mark.parametrize("chart_type,agg,expect_x,expect_y", [
        ("line", "none", "Date", "Total Net Sales"),
        ("area", "none", "Date", "Total Net Sales"),
        ("bar", "count", "Region", "Number of records"),
        ("bar", "mean", "Region", "Average Net Sales"),
        ("bar", "none", "Region", "Average Net Sales"),       # the renderer averages when unspecified
        ("horizontal_bar", "sum", "Total Net Sales", "Region"),
        ("scatter", "none", "Units", "Net Sales"),
        ("histogram", "none", "Net Sales", "Number of records"),
        ("box", "none", "Region", "Net Sales"),
        ("heatmap", "none", "Variable", "Variable"),
    ])
    def test_overview_axis_titles_for_every_chart_type(self, chart_type, agg, expect_x, expect_y, monkeypatch,
                                                       tmp_path):
        from dashboard.components import session_paths
        monkeypatch.setattr(session_paths, "is_user_dataset_mode", lambda: True)   # an upload ...
        monkeypatch.setattr(session_paths, "get_processed_dir", lambda: tmp_path)  # ... with no currency
        mod = importlib.import_module("dashboard.pages.1_executive_overview")
        df = _orders()
        df["Order_Date"] = pd.to_datetime(df["Order_Date"])
        x, y = {"line": ("Order_Date", "Net_Sales"), "area": ("Order_Date", "Net_Sales"), "bar": ("Region", "Net_Sales"),
                "horizontal_bar": ("Net_Sales", "Region"), "scatter": ("Units", "Net_Sales"),
                "histogram": ("Net_Sales", None), "box": ("Region", "Net_Sales"), "heatmap": (None, None)}[chart_type]
        xt, yt, _ = mod._axis_titles(chart_type, df, x, y, agg, "T")
        assert (xt, yt) == (expect_x, expect_y)


class TestChartsInApp:
    def test_executive_overview_demo_every_chart_labelled(self):
        at = _goto(_app(_demo()), APPROVED[2])
        _no_failure(at)
        figs = _assert_figures_labelled(at, min_figures=4)
        ys = [_axis_title(f["layout"], "yaxis") for f in figs]
        assert "Revenue (INR)" in ys and "Total Marketing Spend (INR)" in ys

    def test_executive_overview_upload_every_chart_labelled(self, make_upload):
        did = make_upload(_orders(), confirmed_roles={"date", "revenue"},
                          overrides={"date_column": "Order_Date", "revenue_column": "Net_Sales"})
        at = _goto(_app(_upload_state(did)), APPROVED[2])
        _no_failure(at)
        figs = _assert_figures_labelled(at, min_figures=3)
        for f in figs:          # no currency was identified, so no unit or symbol may appear
            for axis in ("xaxis", "yaxis"):
                assert "INR" not in _axis_title(f["layout"], axis) and "₹" not in _axis_title(f["layout"], axis)

    @pytest.mark.parametrize("fixture", ["a_marketing.csv", "b_sales.csv", "e_education.csv", "g_no_date.csv"])
    def test_executive_overview_diverse_uploads_labelled(self, make_upload, fixture):
        from tests.helpers import load_fixture
        did = make_upload(load_fixture(fixture))
        at = _goto(_app(_upload_state(did, fixture)), APPROVED[2])
        _no_failure(at)
        if at.get("plotly_chart"):
            _assert_figures_labelled(at)

    def test_marketing_performance_demo_labelled(self):
        at = _goto(_app(_demo()), APPROVED[3])
        _no_failure(at)
        _assert_figures_labelled(at, min_figures=5)

    def test_marketing_performance_upload_labelled_and_aggregated(self, make_upload):
        did = make_upload(_orders(), confirmed_roles={"date", "revenue"},
                          overrides={"date_column": "Order_Date", "revenue_column": "Net_Sales"})
        at = _goto(_app(_upload_state(did)), APPROVED[3])
        _no_failure(at)
        figs = _assert_figures_labelled(at)
        trend = [f for f in figs if "Trend" in _chart_title(f["layout"])]
        assert trend, "expected a trend chart"
        xs = trend[0]["data"][0]["x"]
        assert len(xs) == len(set(xs)), "trend must have one point per date"
        assert _axis_title(trend[0]["layout"], "yaxis").startswith("Total ")

    def test_currency_appears_on_axes_only_when_known(self, make_upload):
        df = _orders(revenue_col="Revenue_USD")
        info = C.choice_to_info("auto", C.detect_currency(df), monetary_columns=["Revenue_USD"])
        assert info["code"] == "USD"
        did = make_upload(df, currency_info=info, confirmed_roles={"date", "revenue"},
                          overrides={"date_column": "Order_Date", "revenue_column": "Revenue_USD"})
        at = _goto(_app(_upload_state(did)), APPROVED[2])
        _no_failure(at)
        titles = [_axis_title(f["layout"], a) for f in _figures(at) for a in ("xaxis", "yaxis")]
        assert any("(USD)" in t for t in titles)
        assert not any("INR" in t or "₹" in t for t in titles)
        assert not any("Units (USD)" in t for t in titles)       # non-monetary columns get no currency


# ---------------------------------------------------------------------------
# K · currency detection
# ---------------------------------------------------------------------------

class TestCurrencyDetection:
    @pytest.mark.parametrize("frame,code,symbol", [
        ({"Revenue (INR)": [1.0, 2.0]}, "INR", "₹"),
        ({"amount_usd": [1.0]}, "USD", "$"),
        ({"Sales": ["€1.200", "€30"]}, "EUR", "€"),
        ({"Turnover GBP": [3.0]}, "GBP", "£"),
        ({"Revenue (JPY)": [100]}, "JPY", "¥"),
        ({"Revenue USD": ["$1,200", "$3"]}, "USD", "$"),
        ({"Sales": ["₹1,200", "Rs. 300"]}, "INR", "₹"),
        ({"amt": ["EUR 300", "EUR 12"]}, "EUR", "€"),
        ({"Revenue (CAD)": [1.0]}, "CAD", "C$"),
    ])
    def test_identified(self, frame, code, symbol):
        r = C.detect_currency(pd.DataFrame(frame))
        assert (r.code, r.symbol) == (code, symbol) and not r.conflicts

    @pytest.mark.parametrize("frame,symbol", [
        ({"Sales": ["$1,200", "$3.50"]}, "$"), ({"amt": ["¥300"]}, "¥"), ({"Cost (Rs)": [1.0]}, "Rs"),
    ])
    def test_shared_symbol_keeps_symbol_but_no_code(self, frame, symbol):
        r = C.detect_currency(pd.DataFrame(frame))
        assert r.code is None and r.symbol == symbol and "several currencies" in r.note

    @pytest.mark.parametrize("frame", [
        {"rev_usd": [1.0], "cost_eur": [2.0]},                 # different columns disagree
        {"x": ["€5", "£6"]},                                    # one column mixes currencies
    ])
    def test_conflicts_give_no_currency(self, frame):
        r = C.detect_currency(pd.DataFrame(frame))
        assert not r.identified and r.conflicts

    @pytest.mark.parametrize("frame", [
        {"Price": [1.0], "Qty": [2]}, {"try_count": [1.0], "cad_hours": [2.0]}, {"users": [1], "hrs": [2]},
        {"Revenue": [100.0, 200.0]},                            # plain numbers: never assume INR
    ])
    def test_no_evidence_no_currency(self, frame):
        r = C.detect_currency(pd.DataFrame(frame))
        assert not r.identified and r.code is None and r.symbol is None

    def test_xlsx_upload_header_is_detected(self, tmp_path):
        path = tmp_path / "sales.xlsx"
        pd.DataFrame({"Week": ["2024-01-01"], "Revenue (EUR)": [10.0]}).to_excel(path, index=False)
        r = C.detect_currency(pd.read_excel(path))
        assert r.code == "EUR"

    def test_choice_to_info(self):
        det = C.detect_currency(pd.DataFrame({"amount_usd": [1.0]}))
        assert C.choice_to_info("auto", det, ["amount_usd"])["code"] == "USD"
        none = C.choice_to_info("none", det, ["amount_usd"])
        assert none["symbol"] is None and none["source"].startswith("user")
        gbp = C.choice_to_info("GBP", det, ["amount_usd"])
        assert gbp["code"] == "GBP" and gbp["symbol"] == "£" and gbp["source"] == "user selection"
        with pytest.raises(ValueError):
            C.choice_to_info("XXX", det)

    def test_save_and_load_roundtrip(self, tmp_path):
        C.save_currency(tmp_path, {"code": "EUR", "symbol": "€", "monetary_columns": ["a"]})
        assert C.load_currency(tmp_path)["symbol"] == "€"
        assert C.load_currency(tmp_path / "missing") is None


class TestCurrencyDisplay:
    def _set_dataset(self, monkeypatch, user_mode, processed):
        from dashboard.components import session_paths
        monkeypatch.setattr(session_paths, "is_user_dataset_mode", lambda: user_mode)
        monkeypatch.setattr(session_paths, "get_processed_dir", lambda: processed)

    def test_demo_money_columns_are_explicit(self, monkeypatch, tmp_path):
        from dashboard.components import currency as D
        self._set_dataset(monkeypatch, False, tmp_path)
        assert D.is_monetary("Revenue") and D.is_monetary("TV_Spend") and D.is_monetary("Total_Marketing_Spend")
        assert not D.is_monetary("Units") and not D.is_monetary("New_Customers") and not D.is_monetary("Promotion_Flag")

    def test_demo_is_inr_and_identical_to_previous_format(self, monkeypatch, tmp_path):
        from dashboard.components import currency as D
        from dashboard.components.kpi_card import format_inr
        self._set_dataset(monkeypatch, False, tmp_path)
        assert D.active_currency()["code"] == "INR"
        for v in (12, 999, 2.5e5, 1.5e7, -3e5, float("nan")):
            assert D.money(v) == format_inr(v)
            assert D.money(v, compact=False) == format_inr(v, compact=False)

    def test_upload_without_currency_shows_no_symbol(self, monkeypatch, tmp_path):
        from dashboard.components import currency as D
        self._set_dataset(monkeypatch, True, tmp_path)
        assert D.active_currency() is None and D.symbol() == "" and D.unit_label() == ""
        assert D.money(1_234_567) == "1.23M" and D.money(float("inf")) == "—"
        assert not D.is_monetary("Revenue")

    def test_upload_with_stored_currency(self, monkeypatch, tmp_path):
        from dashboard.components import currency as D
        C.save_currency(tmp_path, C.choice_to_info("USD", None, ["Revenue"]))
        self._set_dataset(monkeypatch, True, tmp_path)
        assert D.money(1500) == "$1.50K" and D.money(12) == "$12" and D.unit_label() == "USD"
        assert D.is_monetary("Revenue") and not D.is_monetary("Units")

    def test_dataset_switch_switches_currency(self, monkeypatch, tmp_path):
        from dashboard.components import currency as D
        a, b = tmp_path / "a", tmp_path / "b"
        a.mkdir(); b.mkdir()
        C.save_currency(a, C.choice_to_info("EUR", None, ["x"]))
        C.save_currency(b, C.choice_to_info("GBP", None, ["x"]))
        self._set_dataset(monkeypatch, True, a)
        assert D.symbol() == "€"
        self._set_dataset(monkeypatch, True, b)
        assert D.symbol() == "£"
        self._set_dataset(monkeypatch, False, b)
        assert D.symbol() == "₹"

    def test_chart_tick_prefix_follows_active_currency(self, monkeypatch, tmp_path):
        import plotly.graph_objects as go
        from dashboard.styles.theme import apply_chart_style
        self._set_dataset(monkeypatch, True, tmp_path)               # no currency stored
        fig = apply_chart_style(go.Figure(go.Bar(x=[1], y=[1])), currency="y", x_title="X", y_title="Y")
        assert (fig.layout.yaxis.tickprefix or "") == ""
        assert fig.layout.xaxis.title.text == "X" and fig.layout.yaxis.title.text == "Y"
        self._set_dataset(monkeypatch, False, tmp_path)              # demo
        fig = apply_chart_style(go.Figure(go.Bar(x=[1], y=[1])), currency="y")
        assert fig.layout.yaxis.tickprefix == "₹"

    def test_kpi_gets_currency_only_for_monetary_sums_and_means(self, monkeypatch, tmp_path):
        mod = importlib.import_module("dashboard.pages.1_executive_overview")
        C.save_currency(tmp_path, C.choice_to_info("EUR", None, ["Revenue"]))
        self._set_dataset(monkeypatch, True, tmp_path)
        assert mod._kpi_display_value({"formatted_value": "1.2K", "calculation": "sum(Revenue)",
                                       "source_column": "Revenue"}) == "€1.2K"
        assert mod._kpi_display_value({"formatted_value": "12", "calculation": "count(Revenue)",
                                       "source_column": "Revenue"}) == "12"
        assert mod._kpi_display_value({"formatted_value": "40", "calculation": "sum(Units)",
                                       "source_column": "Units"}) == "40"


class TestAnalystCurrency:
    def _analyst(self, did):
        from src.ai.analyst import MarketingAnalyst
        return MarketingAnalyst(data_dir=dataset_processed_dir(did), dataset_id=did)

    def test_uploaded_dataset_never_described_in_rupees(self, make_upload):
        from tests.helpers import load_fixture
        df = load_fixture("a_marketing.csv")
        did = make_upload(df, currency_info=C.choice_to_info("USD", None, list(df.columns)))
        a = self._analyst(did)
        assert a.cur == "$"
        for q in ("Which channel is the most efficient?", "Which channel performs worst?"):
            ans = a.answer(q)["answer"]
            assert "₹" not in ans and "rupee" not in ans.lower()

    def test_uploaded_dataset_without_currency_has_no_symbol(self, make_upload):
        from tests.helpers import load_fixture
        did = make_upload(load_fixture("a_marketing.csv"))
        a = self._analyst(did)
        assert a.cur == "" and a.per_unit == "unit of currency"

    def test_demo_still_inr(self):
        from src.ai.analyst import MarketingAnalyst
        a = MarketingAnalyst()
        assert a.is_demo and a.cur == "₹"


class TestUploadPageCurrencyStep:
    def test_monetary_columns(self):
        mod = importlib.import_module("dashboard.pages.0_upload_data")
        from src.data_engine.pipeline import profile_and_map
        df = pd.DataFrame({"week": pd.date_range("2024-01-01", periods=30, freq="W").astype(str),
                           "Revenue (EUR)": np.arange(30.0) + 100, "tv_spend": np.arange(30.0), "Units": np.arange(30)})
        _, mapping = profile_and_map(df)
        det = C.detect_currency(df)
        cols = mod._monetary_columns(mapping, {"Revenue (EUR)": "revenue", "tv_spend": "channel"}, det)
        assert "Revenue (EUR)" in cols and "tv_spend" in cols and "Units" not in cols

    def test_upload_page_offers_override_with_every_currency(self):
        src = (ROOT / "dashboard/pages/0_upload_data.py").read_text(encoding="utf-8")
        assert "2b · Currency" in src and '["auto", "none"] + sorted(CURRENCIES)' in src
        assert "save_currency(dataset_processed_dir(dataset_id)" in src


# ---------------------------------------------------------------------------
# X · forecast regression (page and loader unchanged by PART 01)
# ---------------------------------------------------------------------------

def test_forecast_page_renders_on_demo_and_upload(make_upload):
    at = _goto(_app(_demo()), APPROVED[8])
    _no_failure(at)
    assert at.get("plotly_chart")
    did = make_upload(_orders(), confirmed_roles={"date", "revenue"},
                      overrides={"date_column": "Order_Date", "revenue_column": "Net_Sales"})
    at = _goto(_app(_upload_state(did)), APPROVED[8])
    _no_failure(at)
    assert not any("Forecast not available" in w.value for w in at.warning)
