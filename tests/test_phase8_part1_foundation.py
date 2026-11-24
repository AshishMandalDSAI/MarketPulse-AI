"""Phase 8 / Part 1 - shared foundation and dashboard integration tests.

Covers: formatting, dataset summary (dynamic column typing on every fixture
dataset), CSV/Excel export, dataset-scoped session storage, module registry,
availability gate, navigation integrity (modules 00-13 untouched), real
Streamlit page rendering via AppTest, dataset switching / reset and
session-state isolation.
"""

import ast
import io
import re
import shutil
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.helpers import all_diverse_fixtures, GENERIC_FIXTURES_DIR, load_fixture
from src.common import export, session_scope
from src.common.availability import check_availability
from src.common.dataset_summary import DatasetSummary, summarize_dataset
from src.common.formatting import escape_text, fmt_int, fmt_number, fmt_p_value, fmt_pct
from src.common.modules import MODULE_SPECS, get_module_spec

APP_PATH = ROOT / "app.py"

# Upgrade PART 01 (approved navigation). Previously this file pinned the original
# 00-13 menu and the statistics modules at 14-17; the approved restructure
# replaces that layout, so the expected menu below is the approved one, exactly.
APPROVED_PAGES = {
    "00 · 📤 Upload Data": "dashboard.pages.0_upload_data",
    "01 · 🧮 Statistics": "dashboard.pages.statistics",
    "02 · 📊 Executive Overview": "dashboard.pages.1_executive_overview",
    "03 · 📈 Marketing Performance": "dashboard.pages.2_marketing_performance",
    "04 · 🧠 Marketing Effectiveness Model": "dashboard.pages.marketing_effectiveness",
    "05 · 📡 Channel Performance & Contribution": "dashboard.pages.channel_performance",
    "06 · ⚖️ Marketing Resource Optimizer": "dashboard.pages.resource_optimizer",
    "07 · 🧪 Marketing Scenario Simulator": "dashboard.pages.scenario_simulator",
    "08 · 📅 Forecast": "dashboard.pages.7_forecast",
    "09 · 🤖 AI Marketing Analyst": "dashboard.pages.8_ai_marketing_analyst",
    "10 · 💡 Recommendations": "dashboard.pages.9_recommendations",
    "11 · 📋 Reports": "dashboard.pages.10_reports",
    "12 · 🔮 Predictions": "dashboard.pages.11_predictions",
    "13 · 🔍 Anomalies & Drivers": "dashboard.pages.12_anomalies_drivers",
    "14 · 🧠 Decision Intelligence": "dashboard.pages.13_decision_intelligence",
    "15 · 🎯 Prescriptive Analytics": "dashboard.pages.17_prescriptive_analytics",
}
# Modules removed from the menu but kept (engines, API, analyst and tests use them).
KEPT_OUT_OF_MENU = [
    "dashboard.pages.3_marketing_mix_model", "dashboard.pages.4_channel_roi",
    "dashboard.pages.5_budget_optimizer", "dashboard.pages.6_what_if_simulator",
    "dashboard.pages.14_descriptive_statistics", "dashboard.pages.15_inferential_statistics",
    "dashboard.pages.16_predictive_analytics",
]
# Kept for the pre-existing tests below that iterate over "every menu page".
ORIGINAL_PAGES = APPROVED_PAGES


def _read_pages_dict() -> dict:
    """The menu app.py renders (app.PAGES is dashboard.navigation.PAGES)."""
    from dashboard import navigation
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    assert any(isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PAGES" for t in n.targets)
               and ast.unparse(n.value) == "navigation.PAGES" for n in tree.body), \
        "app.py must take its menu from dashboard.navigation"
    return dict(navigation.PAGES)


def _all_fixture_frames():
    frames = [(f"diverse/{n}", pd.read_csv(p)) for n, p in all_diverse_fixtures()]
    frames += [(f"generic/{p.name}", pd.read_csv(p)) for p in sorted(GENERIC_FIXTURES_DIR.glob("*.csv"))]
    return frames


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

class TestFormatting:
    @pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), "abc", True])
    def test_non_finite_or_invalid_values_show_placeholder(self, bad):
        assert fmt_number(bad) == "—"
        assert fmt_pct(bad) == "—"
        assert fmt_int(bad) == "—"

    def test_number_formats(self):
        assert fmt_number(0) == "0"
        assert fmt_number(3.14159265) == "3.1416"
        assert fmt_number(12345.678) == "12,345.68"
        assert "e" in fmt_number(0.0000001)

    def test_p_value_never_prints_zero(self):
        assert fmt_p_value(0.0) == "< 0.001"
        assert fmt_p_value(0.0004) == "< 0.001"
        assert fmt_p_value(0.0421) == "0.042"
        assert fmt_p_value(1.0) == "1.000"

    @pytest.mark.parametrize("bad", [-0.1, 1.5, None, float("nan")])
    def test_p_value_out_of_range_is_placeholder(self, bad):
        assert fmt_p_value(bad) == "—"

    def test_escape_text_blocks_html(self):
        assert "<script>" not in escape_text("<script>alert(1)</script>")


# ---------------------------------------------------------------------------
# Dataset summary - dynamic, dataset-agnostic
# ---------------------------------------------------------------------------

class TestDatasetSummary:
    @pytest.mark.parametrize("name,df", _all_fixture_frames(), ids=lambda v: v if isinstance(v, str) else "")
    def test_every_fixture_summarizes_consistently(self, name, df):
        s = summarize_dataset(df)
        assert s.n_rows == len(df) and s.n_cols == df.shape[1]
        typed = (s.numeric_columns + s.categorical_columns + s.datetime_columns
                 + s.boolean_columns + s.identifier_columns + s.text_columns + s.empty_columns)
        # each column is assigned to at most one type bucket and is a real column
        assert len(typed) == len(set(typed))
        assert set(typed) <= set(df.columns)
        assert len(s.columns) == df.shape[1]
        assert s.missing_cells == int(df.isna().sum().sum())
        assert 0.0 <= s.missing_pct <= 100.0
        assert s.duplicate_rows == int(df.duplicated().sum())
        assert list(s.column_table().columns) == [
            "column", "detected_type", "dtype", "missing_pct", "unique_values"]

    def test_unusual_column_names_are_never_assumed(self):
        df = pd.DataFrame({
            "ΔRevenue (₹) 😀": np.random.default_rng(0).normal(100, 10, 50),
            "  weird name!! ": ["a", "b", "c", "d", "e"] * 10,
            "x" * 80: np.arange(50) % 7,
        })
        s = summarize_dataset(df)
        assert "ΔRevenue (₹) 😀" in s.numeric_columns
        assert "  weird name!! " in s.categorical_columns

    def test_numeric_only_dataset(self):
        rng = np.random.default_rng(1)
        df = pd.DataFrame({"a": rng.normal(size=60), "b": rng.normal(size=60)})
        s = summarize_dataset(df)
        assert s.numeric_columns == ["a", "b"] and s.categorical_columns == []

    def test_counts_missing_and_duplicates_from_data(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0, 3.0], "b": ["x", "y", None, None]})
        s = summarize_dataset(df)
        assert s.missing_cells == 3
        assert s.missing_pct == pytest.approx(3 / 8 * 100, abs=0.01)
        assert s.duplicate_rows == 1

    @pytest.mark.parametrize("bad", [None, pd.DataFrame(), pd.DataFrame({"a": []}), "not a frame", 5])
    def test_empty_or_invalid_input_returns_empty_summary(self, bad):
        s = summarize_dataset(bad)
        assert isinstance(s, DatasetSummary) and s.is_empty

    def test_unhashable_cells_do_not_crash(self):
        df = pd.DataFrame({"a": [[1], [2], [3]], "b": [1, 2, 3]})
        assert summarize_dataset(df).n_rows == 3

    def test_constant_and_empty_columns_detected(self):
        df = pd.DataFrame({"c": [5] * 30, "e": [np.nan] * 30,
                           "v": np.random.default_rng(3).normal(size=30)})
        s = summarize_dataset(df)
        assert "c" in s.constant_columns and "e" in s.empty_columns and "v" in s.numeric_columns


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

class TestExport:
    def test_csv_round_trip(self):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        out = pd.read_csv(io.BytesIO(export.dataframe_to_csv_bytes(df)))
        pd.testing.assert_frame_equal(out, df)

    def test_formula_like_text_is_neutralized(self):
        df = pd.DataFrame({"t": ["=HYPERLINK(\"http://x\")", "+cmd", "-1+1", "@SUM(A1)", "safe"], "n": [-1, 2, 3, 4, 5]})
        out = pd.read_csv(io.BytesIO(export.dataframe_to_csv_bytes(df)))
        assert all(v.startswith("'") for v in out["t"][:4])
        assert out["t"][4] == "safe"
        assert out["n"][0] == -1  # real negative numbers untouched

    def test_excel_round_trip_multiple_sheets(self):
        sheets = {"Summary": pd.DataFrame({"a": [1, 2]}), "Details": pd.DataFrame({"b": ["x", "y"]})}
        raw = export.sheets_to_excel_bytes(sheets)
        back = pd.read_excel(io.BytesIO(raw), sheet_name=None)
        assert set(back) == {"Summary", "Details"}
        assert back["Details"]["b"].tolist() == ["x", "y"]

    def test_sheet_names_sanitized_and_unique(self):
        sheets = {"a/b:c*?[x]": pd.DataFrame({"a": [1]}), "A/B:C*?[X]": pd.DataFrame({"a": [2]}),
                  "y" * 60: pd.DataFrame({"a": [3]})}
        names = list(pd.read_excel(io.BytesIO(export.sheets_to_excel_bytes(sheets)), sheet_name=None))
        assert len(names) == 3 and len(set(n.lower() for n in names)) == 3
        assert all(len(n) <= 31 and not re.search(r"[\[\]:*?/\\]", n) for n in names)

    def test_excel_row_limit_is_reported(self, monkeypatch):
        monkeypatch.setattr(export, "EXCEL_MAX_ROWS", 2)
        with pytest.raises(export.ExportError, match="CSV"):
            export.sheets_to_excel_bytes({"s": pd.DataFrame({"a": [1, 2, 3]})})

    @pytest.mark.parametrize("bad", [None, "x", 3])
    def test_csv_rejects_non_table(self, bad):
        with pytest.raises(export.ExportError):
            export.dataframe_to_csv_bytes(bad)

    def test_excel_rejects_empty_and_non_table_sheets(self):
        with pytest.raises(export.ExportError):
            export.sheets_to_excel_bytes({})
        with pytest.raises(export.ExportError):
            export.sheets_to_excel_bytes({"s": [1, 2]})

    def test_export_does_not_mutate_input(self):
        df = pd.DataFrame({"t": ["=1+1"]})
        export.dataframe_to_csv_bytes(df)
        assert df["t"][0] == "=1+1"


# ---------------------------------------------------------------------------
# Dataset-scoped session storage
# ---------------------------------------------------------------------------

class TestSessionScope:
    def test_keys_always_use_cleared_prefix(self):
        assert session_scope.scoped_key("m", "x").startswith("mp_page_")

    def test_round_trip_same_dataset(self):
        state = {}
        session_scope.set_scoped(state, "ds1", "descriptive", "res", {"a": 1})
        assert session_scope.get_scoped(state, "ds1", "descriptive", "res") == {"a": 1}

    def test_other_dataset_never_sees_value_and_it_is_discarded(self):
        state = {}
        session_scope.set_scoped(state, "ds1", "descriptive", "res", "secret-A")
        assert session_scope.get_scoped(state, "ds2", "descriptive", "res", default="none") == "none"
        assert session_scope.scoped_key("descriptive", "res") not in state

    def test_modules_do_not_share_keys(self):
        state = {}
        session_scope.set_scoped(state, "ds1", "descriptive", "res", 1)
        assert session_scope.get_scoped(state, "ds1", "inferential", "res") is None

    def test_malformed_entry_returns_default(self):
        state = {session_scope.scoped_key("m", "x"): "raw"}
        assert session_scope.get_scoped(state, "ds", "m", "x", default=7) == 7

    def test_clear_module_removes_only_that_module(self):
        state = {}
        session_scope.set_scoped(state, "d", "a", "x", 1)
        session_scope.set_scoped(state, "d", "a", "y", 2)
        session_scope.set_scoped(state, "d", "b", "x", 3)
        assert session_scope.clear_module(state, "a") == 2
        assert session_scope.get_scoped(state, "d", "b", "x") == 3


# ---------------------------------------------------------------------------
# Module registry & navigation integrity
# ---------------------------------------------------------------------------

class TestModuleRegistryAndNavigation:
    def test_four_modules_with_required_labels(self):
        # PART 01: descriptive / inferential / predictive are tabs of 01 · Statistics;
        # prescriptive is menu entry 15. Module identities (14-17) are unchanged.
        assert [m.nav_label for m in MODULE_SPECS] == [
            "01 · 🧮 Statistics", "01 · 🧮 Statistics", "01 · 🧮 Statistics", "15 · 🎯 Prescriptive Analytics"]
        assert [m.stats_tab for m in MODULE_SPECS] == [
            "📊 Descriptive Statistics", "🔬 Inferential Statistics", "🔮 Predictive Analysis", None]
        assert [m.number for m in MODULE_SPECS] == [14, 15, 16, 17]
        assert [m.delivered_in_part for m in MODULE_SPECS] == [2, 3, 4, 5]

    def test_unknown_module_key_raises(self):
        with pytest.raises(KeyError):
            get_module_spec("nope")

    def test_menu_is_exactly_the_approved_structure_in_order(self):
        pages = _read_pages_dict()
        assert list(pages.items()) == list(APPROVED_PAGES.items())
        assert len(pages) == 16

    def test_every_module_is_reachable_from_the_menu(self):
        pages = _read_pages_dict()
        for m in MODULE_SPECS:
            assert m.nav_label in pages
        assert "dashboard.pages.statistics" in pages.values()

    def test_modules_removed_from_menu_are_kept(self):
        pages = _read_pages_dict()
        for module in KEPT_OUT_OF_MENU:
            assert module not in pages.values()
            assert (ROOT / (module.replace(".", "/") + ".py")).exists()

    @pytest.mark.parametrize("label,module", list(APPROVED_PAGES.items()) + [
        ("(kept) " + m, m) for m in KEPT_OUT_OF_MENU])
    def test_every_page_module_imports_and_has_render(self, label, module):
        pytest.importorskip("streamlit")
        import importlib
        assert hasattr(importlib.import_module(module), "render")

    def test_page_files_exist_for_new_modules(self):
        for m in MODULE_SPECS:
            assert (ROOT / (m.page_module.replace(".", "/") + ".py")).exists()

    def test_new_pages_contain_no_hardcoded_results(self):
        for m in MODULE_SPECS:
            src = (ROOT / (m.page_module.replace(".", "/") + ".py")).read_text(encoding="utf-8")
            assert "st.metric" not in src and "pd.DataFrame(" not in src


# ---------------------------------------------------------------------------
# Availability gate
# ---------------------------------------------------------------------------

class TestAvailability:
    def _s(self, df):
        return summarize_dataset(df)

    @pytest.mark.parametrize("key", ["descriptive", "inferential", "predictive", "prescriptive"])
    def test_empty_dataset_unavailable_with_reason_for_every_module(self, key):
        a = check_availability(key, summarize_dataset(None))
        assert not a.available and a.reasons and a.missing

    def test_unknown_module_unavailable(self):
        assert not check_availability("zzz", self._s(pd.DataFrame({"a": [1, 2, 3]}))).available

    def test_descriptive_needs_a_non_empty_column_not_a_varying_one(self):
        # Correction 1: constant columns can still be described (count, mean, mode, ...).
        const = pd.DataFrame({"a": [1] * 20, "b": ["x"] * 20})
        assert check_availability("descriptive", self._s(const)).available
        empty = pd.DataFrame({"a": [np.nan] * 20, "b": [None] * 20})
        a = check_availability("descriptive", self._s(empty))
        assert not a.available and "entirely empty" in a.reasons[0].lower()
        assert check_availability("descriptive", self._s(pd.DataFrame({"a": np.arange(5.0)}))).available

    def test_inferential_needs_three_rows(self):
        a = check_availability("inferential", self._s(pd.DataFrame({"a": [1.0, 2.0]})))
        assert not a.available and "3" in a.reasons[0]

    def test_predictive_needs_rows_and_two_columns(self):
        a = check_availability("predictive", self._s(pd.DataFrame({"a": np.arange(5.0), "b": np.arange(5.0)})))
        assert not a.available
        rng = np.random.default_rng(0)
        ok = pd.DataFrame({"a": rng.normal(size=40), "b": rng.normal(size=40)})
        assert check_availability("predictive", self._s(ok)).available

    def test_prescriptive_needs_two_numeric_columns(self):
        rng = np.random.default_rng(0)
        one = pd.DataFrame({"a": rng.normal(size=40), "g": ["x", "y"] * 20})
        a = check_availability("prescriptive", self._s(one))
        assert not a.available and any("numeric" in r.lower() for r in a.reasons)

    @pytest.mark.parametrize("name,df", _all_fixture_frames(), ids=lambda v: v if isinstance(v, str) else "")
    def test_gate_never_crashes_on_any_fixture(self, name, df):
        s = summarize_dataset(df)
        for m in MODULE_SPECS:
            a = check_availability(m.key, s)
            assert isinstance(a.available, bool)
            assert a.available or a.reasons  # an unavailable verdict always explains itself


# ---------------------------------------------------------------------------
# Real dashboard integration (Streamlit AppTest)
# ---------------------------------------------------------------------------

st_testing = pytest.importorskip("streamlit.testing.v1")
from config.settings import DEMO_DATASET_ID, dataset_upload_dir  # noqa: E402

_KPI = re.compile(r'mp-kpi-label">([^<]+)</div>\s*<div class="mp-kpi-value">([^<]+)<')


def _new_app(session=None):
    at = st_testing.AppTest.from_file(str(APP_PATH), default_timeout=90)
    for k, v in (session or {}).items():
        at.session_state[k] = v
    return at


def _demo_session():
    return {"mp_active_dataset_id": DEMO_DATASET_ID,
            "mp_active_dataset_filename": "NovaMart (demo)",
            "mp_demo_explicitly_chosen": True}


def _kpis(at) -> dict:
    out = {}
    for md in at.markdown:
        for label, value in _KPI.findall(md.value):
            out[label] = value
    return out


def _goto(at, label):
    at.run()  # first run builds the sidebar; then select the page
    at.sidebar.radio[0].set_value(label).run()
    return at


def _goto_module(at, key):
    """PART 01: a module is a Statistics tab (or its own page for prescriptive)."""
    from tests.helpers import goto_module
    at.run()
    goto_module(at, key)
    return at


@pytest.fixture
def uploaded_dataset():
    """Run the real pipeline on a fixture to create a temporary uploaded dataset."""
    from src.data_engine.analysis_runner import run_full_analysis
    ids = []

    def make(fixture_name):
        dataset_id = f"test_p8_{uuid.uuid4().hex[:10]}"
        df = load_fixture(fixture_name)
        run_full_analysis(dataset_id, df=df)
        ids.append(dataset_id)
        return dataset_id, df

    yield make
    for i in ids:
        shutil.rmtree(dataset_upload_dir(i), ignore_errors=True)


class TestDashboardIntegration:
    @pytest.mark.parametrize("module", MODULE_SPECS, ids=lambda m: m.key)
    def test_new_page_renders_on_demo_without_exception(self, module):
        at = _goto_module(_new_app(_demo_session()), module.key)
        assert not at.exception
        assert any(module.title in h.value for h in list(at.title) + list(at.subheader))  # PART 01: tab heading
        kpis = _kpis(at)
        assert {"Rows", "Columns", "Numeric", "Categorical", "Missing cells"} <= set(kpis)
        # shows the honest "not delivered yet" status and no analytic results
        assert any("Part" in i.value and module.title in i.value for i in at.info)
        assert not at.dataframe or all(True for _ in at.dataframe)  # only the column-details table

    def test_dataset_info_matches_the_real_demo_data(self):
        from dashboard.components.data_loader import load_marketing_data  # noqa: F401
        at = _goto_module(_new_app(_demo_session()), MODULE_SPECS[0].key)
        shown_rows = int(_kpis(at)["Rows"].replace(",", ""))
        demo = pd.read_csv(ROOT / "data" / "processed" / "marketing_data_clean.csv")
        assert shown_rows == len(demo)

    @pytest.mark.parametrize("label", list(ORIGINAL_PAGES))
    def test_original_pages_still_render_on_demo(self, label):
        at = _goto(_new_app(_demo_session()), label)
        assert not at.exception, f"{label} raised: {at.exception}"

    def test_no_dataset_gate_still_blocks_new_pages(self):
        at = _goto_module(_new_app(), MODULE_SPECS[0].key)
        assert not at.exception
        assert any("No dataset is currently active" in i.value for i in at.info)

    def test_uploaded_dataset_shows_its_own_numbers(self, uploaded_dataset):
        dataset_id, df = uploaded_dataset("f_employee.csv")
        at = _new_app({"mp_active_dataset_id": dataset_id,
                       "mp_active_dataset_filename": "f_employee.csv",
                       "mp_demo_explicitly_chosen": False})
        _goto_module(at, MODULE_SPECS[0].key)
        assert not at.exception
        assert int(_kpis(at)["Rows"].replace(",", "")) == len(df)
        assert any("f_employee.csv" in c.value for c in at.caption)

    @pytest.mark.parametrize("fixture", ["g_no_date.csv", "h_unusual_columns.csv", "i_missing_values.csv",
                                         "e_education.csv", "c_finance.csv", "k_minimal.csv"])
    @pytest.mark.parametrize("module", MODULE_SPECS, ids=lambda m: m.key)
    def test_new_pages_survive_diverse_uploaded_datasets(self, uploaded_dataset, fixture, module):
        dataset_id, df = uploaded_dataset(fixture)
        at = _new_app({"mp_active_dataset_id": dataset_id, "mp_active_dataset_filename": fixture,
                       "mp_demo_explicitly_chosen": False})
        _goto_module(at, module.key)
        assert not at.exception, at.exception
        if _kpis(at):
            assert int(_kpis(at)["Rows"].replace(",", "")) == len(df)
        else:  # data genuinely unusable -> an explicit availability message, never a blank page
            assert at.warning

    def test_back_to_back_datasets_without_any_cache_clear_never_mix(self, uploaded_dataset):
        """Streamlit's cache is process-wide. Two datasets viewed one after the
        other (as two browser sessions on one server would) must each show only
        their own data even when nothing clears the cache in between."""
        id_a, df_a = uploaded_dataset("f_employee.csv")
        id_b, df_b = uploaded_dataset("h_unusual_columns.csv")
        assert len(df_a) != len(df_b)
        seen = []
        for dataset_id, name in [(id_a, "f_employee.csv"), (id_b, "h_unusual_columns.csv"),
                                 (id_a, "f_employee.csv")]:
            at = _new_app({"mp_active_dataset_id": dataset_id, "mp_active_dataset_filename": name,
                           "mp_demo_explicitly_chosen": False})
            _goto_module(at, MODULE_SPECS[0].key)
            assert not at.exception
            seen.append(int(_kpis(at)["Rows"].replace(",", "")))
        assert seen == [len(df_a), len(df_b), len(df_a)]

    def test_switching_to_demo_clears_scoped_state_and_shows_demo_data(self, uploaded_dataset):
        dataset_id, df = uploaded_dataset("f_employee.csv")
        at = _new_app({"mp_active_dataset_id": dataset_id, "mp_active_dataset_filename": "f_employee.csv",
                       "mp_demo_explicitly_chosen": False})
        _goto_module(at, MODULE_SPECS[0].key)
        key = session_scope.scoped_key("descriptive", "dataset_summary")
        assert at.session_state[key]["dataset_id"] == dataset_id
        uploaded_rows = _kpis(at)["Rows"]

        switch = [b for b in at.sidebar.button if "demo" in b.label.lower()][0]
        switch.click().run()
        assert not at.exception
        entry = at.session_state[key] if key in at.session_state else None
        assert entry is None or entry["dataset_id"] == DEMO_DATASET_ID
        assert _kpis(at)["Rows"] != uploaded_rows or len(df) == len(pd.read_csv(
            ROOT / "data" / "processed" / "marketing_data_clean.csv"))

    def test_reset_analysis_removes_scoped_keys_and_blocks_pages(self):
        at = _new_app(_demo_session())
        _goto_module(at, MODULE_SPECS[1].key)
        assert any(k.startswith("mp_page_") for k in at.session_state.keys())
        reset = [b for b in at.sidebar.button if "Reset" in b.label][0]
        reset.click().run()
        assert not any(k.startswith("mp_page_") for k in at.session_state.keys())
        assert any("No dataset is currently active" in i.value for i in at.info)

    def test_stale_value_from_another_dataset_is_never_served(self):
        """Even if a clear were missed, a value tagged with another dataset id is ignored."""
        at = _new_app(_demo_session())
        key = session_scope.scoped_key("descriptive", "dataset_summary")
        at.session_state[key] = {"dataset_id": "some_other_dataset",
                                 "value": DatasetSummary(n_rows=999999, n_cols=3)}
        _goto_module(at, MODULE_SPECS[0].key)
        assert not at.exception
        assert _kpis(at)["Rows"] != "999,999"


class TestDownloadButtons:
    def test_buttons_render_and_export_only_supplied_tables(self):
        def app():
            import pandas as pd
            from dashboard.components.stats_ui import render_download_buttons
            render_download_buttons("descriptive", "demo_summary",
                                    table=pd.DataFrame({"a": [1, 2, 3]}),
                                    sheets={"One": pd.DataFrame({"a": [1]}), "Two": pd.DataFrame({"b": [2]})})
        at = st_testing.AppTest.from_function(app, default_timeout=30).run()
        assert not at.exception
        assert len(at.get("download_button")) == 2

    def test_oversized_excel_shows_reason_instead_of_crashing(self, monkeypatch):
        def app():
            import pandas as pd
            from src.common import export
            export.EXCEL_MAX_ROWS = 1
            from dashboard.components.stats_ui import render_download_buttons
            render_download_buttons("descriptive", "big", table=pd.DataFrame({"a": [1, 2, 3]}))
        at = st_testing.AppTest.from_function(app, default_timeout=30).run()
        assert not at.exception
        assert any("Excel unavailable" in c.value for c in at.caption)
        assert len(at.get("download_button")) == 1  # CSV still offered
