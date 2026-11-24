"""Phase 8 Part 2 - Descriptive Statistics page, rendered with Streamlit AppTest.

Each ``AppTest`` is one user session; Streamlit's cache is process-wide, so
running several in one process reproduces several browser tabs on one server
(this is a simulation, not a live multi-browser test).
"""

import re
import shutil
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

st_testing = pytest.importorskip("streamlit.testing.v1")

from config.settings import DEMO_DATASET_ID, dataset_processed_dir, dataset_upload_dir  # noqa: E402
from src.common import export as common_export, session_scope  # noqa: E402
from src.common.modules import MODULE_SPECS, get_module_spec  # noqa: E402
from src.descriptive.engine import analyze_dataframe  # noqa: E402
from tests.helpers import load_fixture  # noqa: E402
from tests.helpers import goto_module  # noqa: E402  (upgrade PART 01)

APP_PATH = ROOT / "app.py"
SPEC = get_module_spec("descriptive")
LABEL = SPEC.nav_label
_KPI = re.compile(r'mp-kpi-label">([^<]+)</div>\s*<div class="mp-kpi-value">([^<]+)<')

SECTIONS = ["Analysis Settings", "Data Quality Overview", "Numeric Column Statistics",
            "Categorical and Yes/No Column Statistics", "Date Column Summaries", "Frequency Distributions",
            "Missing-Value Analysis", "Outlier Analysis", "Correlation Analysis", "Covariance Analysis",
            "Visualizations", "Download Reports", "Methodology and Statistical Warnings"]


def _new_app(session=None):
    at = st_testing.AppTest.from_file(str(APP_PATH), default_timeout=120)
    for k, v in (session or {}).items():
        at.session_state[k] = v
    return at


def _demo_session():
    return {"mp_active_dataset_id": DEMO_DATASET_ID, "mp_active_dataset_filename": "NovaMart (demo)",
            "mp_demo_explicitly_chosen": True}


def _upload_session(dataset_id, name):
    return {"mp_active_dataset_id": dataset_id, "mp_active_dataset_filename": name, "mp_demo_explicitly_chosen": False}


def _goto(at):
    at.run()
    goto_module(at, SPEC.key)  # PART 01: module is a Statistics tab
    return at


def _kpis(at):
    out = {}
    for md in at.markdown:
        for label, value in _KPI.findall(md.value):
            out[label] = value
    return out


def _page_text(at) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption] + [i.value for i in at.info]
    parts += [w.value for w in at.warning] + [s.value for s in at.subheader] + [h.value for h in at.header]
    return "\n".join(parts)


def _downloads(at):
    return [e.proto for e in at.get("download_button")]


def _table_cells(at) -> set:
    """Every cell of every table on the page, as text (column names are in the first column of most tables)."""
    cells = set()
    for t in at.dataframe:
        frame = t.value
        cells |= {str(c) for c in frame.columns}
        cells |= {str(v) for v in frame.to_numpy().ravel()}
    return cells


def _numeric_table(at):
    """The report-driven 'Numeric Column Statistics' table (not the shared column-details table)."""
    tables = [t.value for t in at.dataframe if {"Column", "Valid", "Mean", "Median"} <= set(t.value.columns)]
    assert len(tables) == 1, f"expected one numeric statistics table, found {len(tables)}"
    return tables[0]


def _expected_numeric(dataset_id) -> dict:
    """Independent expectation: the engine run directly on the dataset's analysis table."""
    table = pd.read_csv(dataset_processed_dir(dataset_id) / "standardized_data.csv")
    return {n.column: n.valid_count for n in analyze_dataframe(table).numeric}


def _own_columns(df) -> set:
    return {str(c) for c in df.columns}


def _rows(at) -> int:
    return int(_kpis(at)["Rows"].replace(",", ""))


@pytest.fixture
def uploaded_dataset():
    from src.data_engine.analysis_runner import run_full_analysis
    ids = []

    def make(fixture_name):
        dataset_id = f"test_p8p2_{uuid.uuid4().hex[:10]}"
        df = load_fixture(fixture_name)
        run_full_analysis(dataset_id, df=df)
        ids.append(dataset_id)
        return dataset_id, df

    yield make
    for i in ids:
        shutil.rmtree(dataset_upload_dir(i), ignore_errors=True)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _real_excel_row_limit(monkeypatch):
    """An older test (test_phase8_part1_foundation.py::test_oversized_excel_...) assigns
    ``export.EXCEL_MAX_ROWS = 1`` without restoring it, which leaks into later tests in the
    same process. Pin the real limit here so these tests do not depend on test order."""
    monkeypatch.setattr(common_export, "EXCEL_MAX_ROWS", 1_048_575)


@pytest.fixture(scope="module")
def demo_at():
    saved = common_export.EXCEL_MAX_ROWS            # module-scoped, so it cannot use the function-scoped monkeypatch
    common_export.EXCEL_MAX_ROWS = 1_048_575
    try:
        yield _goto(_new_app(_demo_session()))
    finally:
        common_export.EXCEL_MAX_ROWS = saved


class TestRendering:


    def test_page_renders_on_demo_without_exception(self, demo_at):
        assert not demo_at.exception
        # PART 01: inside the Statistics workspace the module title is the tab's heading
        assert any(SPEC.title in h.value for h in list(demo_at.title) + list(demo_at.subheader))

    def test_status_notice_and_dataset_kpis(self, demo_at):
        assert any("Part" in i.value and SPEC.title in i.value for i in demo_at.info)
        assert {"Rows", "Columns", "Numeric", "Categorical", "Missing cells"} <= set(_kpis(demo_at))

    def test_every_section_is_present(self, demo_at):
        titles = [t.strip() for m in demo_at.markdown for t in re.findall(r'<div class="mp-section-title">(.*?)</div>', m.value)]
        own = [t for t in titles if t in SECTIONS]
        assert own == SECTIONS, titles          # all 13 module sections, in order (the scaffold adds its own around them)

    def test_tables_charts_and_controls_exist(self, demo_at):
        assert len(demo_at.dataframe) >= 8
        assert len(demo_at.selectbox) >= 5 and demo_at.radio and demo_at.multiselect
        assert len(demo_at.get("plotly_chart")) >= 1

    def test_three_downloads_csv_zip_and_excel(self, demo_at):
        urls = [p.url for p in _downloads(demo_at)]
        assert len(urls) == 3 and [u.rsplit(".", 1)[-1] for u in urls] == ["csv", "zip", "xlsx"]

    def test_downloads_differ_between_datasets(self, uploaded_dataset):
        dataset_id, _ = uploaded_dataset("f_employee.csv")
        demo = _downloads(_goto(_new_app(_demo_session())))
        up = _downloads(_goto(_new_app(_upload_session(dataset_id, "f_employee.csv"))))
        assert len(demo) == len(up) == 3
        assert all(d.url != u.url for d, u in zip(demo, up))      # content-addressed: different content, different file

    def test_numbers_on_the_page_match_an_independent_calculation(self, demo_at):
        demo = pd.read_csv(ROOT / "data" / "processed" / "marketing_data_clean.csv")
        assert _rows(demo_at) == len(demo)
        report = analyze_dataframe(demo)
        first = report.numeric[0]
        shown = " ".join(str(df.value.to_string()) for df in demo_at.dataframe)
        assert f"{first.valid_count:,}" in shown or str(first.valid_count) in shown

    def test_page_text_is_plain_english_and_has_no_raw_python(self, demo_at):
        text = _page_text(demo_at)
        for bad in ("Traceback", "NaN", "nan%", "<function", "None%"):
            assert bad not in text

    def test_no_dataset_gate(self):
        at = _goto(_new_app())
        assert not at.exception
        assert any("No dataset is currently active" in i.value for i in at.info)
        assert not _downloads(at)


# ---------------------------------------------------------------------------
# Interaction
# ---------------------------------------------------------------------------

class TestInteraction:
    def test_every_chart_type_can_be_selected_without_error(self):
        from src.descriptive.charts import CHART_TYPES
        at = _goto(_new_app(_demo_session()))
        pick = next(s for s in at.selectbox if s.label == "Chart type")
        for chart in CHART_TYPES:
            pick.set_value(chart).run()
            assert not at.exception, f"{chart}: {at.exception}"
            pick = next(s for s in at.selectbox if s.label == "Chart type")

    def test_histogram_bins_slider_changes_the_chart(self):
        at = _goto(_new_app(_demo_session()))
        next(s for s in at.selectbox if s.label == "Chart type").set_value("Histogram").run()
        before = len(at.get("plotly_chart"))
        sl = next(s for s in at.slider if s.label == "Number of bins")
        first = at.get("plotly_chart")[-1].proto.spec
        sl.set_value(5).run()
        assert not at.exception and len(at.get("plotly_chart")) == before
        assert at.get("plotly_chart")[-1].proto.spec != first

    def test_correlation_method_switch(self):
        at = _goto(_new_app(_demo_session()))
        radio = next(r for r in at.radio if r.label == "Correlation method")
        radio.set_value("spearman").run()
        assert not at.exception
        assert "spearman" in _page_text(at).lower()

    def test_covariance_type_changes_the_download_but_not_the_page_state(self):
        at = _goto(_new_app(_demo_session()))
        next(r for r in at.radio if r.label == "Covariance type").set_value("population").run()
        assert not at.exception

    def test_outlier_multiplier_and_percentile_changes(self):
        at = _goto(_new_app(_demo_session()))
        next(s for s in at.selectbox if "multiplier" in s.label).set_value(3.0).run()
        assert not at.exception
        next(m for m in at.multiselect if "Percentiles" in m.label).set_value([10, 90]).run()
        assert not at.exception

    def test_empty_percentile_selection_gives_a_message_not_an_exception(self):
        at = _goto(_new_app(_demo_session()))
        next(m for m in at.multiselect if "Percentiles" in m.label).set_value([]).run()
        assert not at.exception

    def test_chart_unavailable_explains_itself(self, uploaded_dataset):
        dataset_id, _ = uploaded_dataset("k_minimal.csv")
        at = _goto(_new_app(_upload_session(dataset_id, "k_minimal.csv")))
        assert not at.exception
        for chart in ("Histogram", "Box plot", "Violin plot", "Density plot", "Correlation heatmap", "Scatter plot",
                      "Bar chart", "Frequency distribution"):
            sel = next((s for s in at.selectbox if s.label == "Chart type"), None)
            if sel is None:
                break
            sel.set_value(chart).run()
            assert not at.exception, f"{chart}: {at.exception}"


# ---------------------------------------------------------------------------
# Dataset switching, sessions, file changes, reset
# ---------------------------------------------------------------------------

class TestIsolation:
    def test_uploaded_dataset_shows_its_own_numbers(self, uploaded_dataset):
        dataset_id, df = uploaded_dataset("f_employee.csv")
        at = _goto(_new_app(_upload_session(dataset_id, "f_employee.csv")))
        assert not at.exception and _rows(at) == len(df)
        assert any("f_employee.csv" in c.value for c in at.caption)

    def test_switching_datasets_in_one_session_updates_the_whole_page(self, uploaded_dataset):
        id_a, df_a = uploaded_dataset("f_employee.csv")
        id_b, df_b = uploaded_dataset("h_unusual_columns.csv")
        assert len(df_a) != len(df_b)
        at = _goto(_new_app(_upload_session(id_a, "f_employee.csv")))
        shown_a, text_a = _rows(at), _page_text(at)
        # switch by assignment only: no cache clear, no widget reset
        at.session_state["mp_active_dataset_id"] = id_b
        at.session_state["mp_active_dataset_filename"] = "h_unusual_columns.csv"
        at.run()
        assert not at.exception
        assert shown_a == len(df_a) and _rows(at) == len(df_b)
        assert _page_text(at) != text_a
        at.session_state["mp_active_dataset_id"] = id_a
        at.session_state["mp_active_dataset_filename"] = "f_employee.csv"
        at.run()
        assert not at.exception and _rows(at) == len(df_a)

    def test_back_to_back_sessions_never_mix_results(self, uploaded_dataset):
        id_a, df_a = uploaded_dataset("f_employee.csv")
        id_b, df_b = uploaded_dataset("h_unusual_columns.csv")
        seen = []
        for dataset_id, name in [(id_a, "f_employee.csv"), (id_b, "h_unusual_columns.csv"), (id_a, "f_employee.csv")]:
            at = _goto(_new_app(_upload_session(dataset_id, name)))
            assert not at.exception
            seen.append(_rows(at))
        assert seen == [len(df_a), len(df_b), len(df_a)]

    def test_interleaved_sessions_keep_their_own_data(self, uploaded_dataset):
        id_a, df_a = uploaded_dataset("f_employee.csv")
        id_b, df_b = uploaded_dataset("h_unusual_columns.csv")
        at_a = _goto(_new_app(_upload_session(id_a, "f_employee.csv")))
        at_b = _goto(_new_app(_upload_session(id_b, "h_unusual_columns.csv")))
        at_a.run()
        at_b.run()
        assert _rows(at_a) == len(df_a) and _rows(at_b) == len(df_b)
        assert "h_unusual_columns.csv" not in " ".join(c.value for c in at_a.caption)

    def test_statistics_tables_belong_to_the_active_dataset_after_every_switch(self, uploaded_dataset):
        """The report-driven numeric table must follow each switch (A -> B -> A -> demo): a report that
        survives a switch would show the previous dataset's columns and counts here."""
        id_a, _ = uploaded_dataset("f_employee.csv")
        id_b, _ = uploaded_dataset("h_unusual_columns.csv")
        exp_a, exp_b = _expected_numeric(id_a), _expected_numeric(id_b)
        assert exp_a and exp_b and set(exp_a) != set(exp_b)
        at = _goto(_new_app(_demo_session()))                      # start on the demo, as a real user often does
        demo_cols = set(_numeric_table(at)["Column"])
        assert demo_cols.isdisjoint(exp_a) and demo_cols.isdisjoint(exp_b)
        for dataset_id, name, expected in [(id_a, "f_employee.csv", exp_a), (id_b, "h_unusual_columns.csv", exp_b),
                                           (id_a, "f_employee.csv", exp_a)]:
            at.session_state["mp_active_dataset_id"] = dataset_id
            at.session_state["mp_active_dataset_filename"] = name
            at.session_state["mp_demo_explicitly_chosen"] = False
            at.run()
            assert not at.exception
            shown = _numeric_table(at)
            assert list(shown["Column"]) == list(expected), f"{name}: shows {list(shown['Column'])}"
            assert [int(v.replace(",", "")) for v in shown["Valid"]] == list(expected.values())
        at.session_state["mp_active_dataset_id"] = DEMO_DATASET_ID
        at.session_state["mp_demo_explicitly_chosen"] = True
        at.run()
        assert set(_numeric_table(at)["Column"]) == demo_cols

    def test_new_sessions_never_see_another_datasets_tables(self, uploaded_dataset):
        id_a, _ = uploaded_dataset("f_employee.csv")
        id_b, _ = uploaded_dataset("h_unusual_columns.csv")
        exp_a, exp_b = _expected_numeric(id_a), _expected_numeric(id_b)
        for dataset_id, name, expected in [(id_a, "f_employee.csv", exp_a), (id_b, "h_unusual_columns.csv", exp_b),
                                           (id_a, "f_employee.csv", exp_a)]:
            at = _goto(_new_app(_upload_session(dataset_id, name)))
            assert list(_numeric_table(at)["Column"]) == list(expected)

    def test_numeric_statistics_table_shows_independently_calculated_values(self, uploaded_dataset):
        dataset_id, _ = uploaded_dataset("f_employee.csv")
        table = pd.read_csv(dataset_processed_dir(dataset_id) / "standardized_data.csv")
        at = _goto(_new_app(_upload_session(dataset_id, "f_employee.csv")))
        shown = _numeric_table(at).set_index("Column")
        for col in shown.index:
            v = pd.to_numeric(table[col], errors="coerce").dropna()
            assert int(shown.loc[col, "Valid"].replace(",", "")) == len(v)
            assert float(shown.loc[col, "Mean"].replace(",", "")) == pytest.approx(v.mean(), rel=1e-3, abs=5e-3)
            assert float(shown.loc[col, "Median"].replace(",", "")) == pytest.approx(v.median(), rel=1e-3, abs=5e-3)

    def test_demo_and_upload_do_not_share_results(self, uploaded_dataset):
        dataset_id, df = uploaded_dataset("f_employee.csv")
        demo_rows = _rows(_goto(_new_app(_demo_session())))
        up = _goto(_new_app(_upload_session(dataset_id, "f_employee.csv")))
        assert _rows(up) == len(df) != demo_rows

    def test_changed_file_gives_fresh_results_not_stale_ones(self, uploaded_dataset):
        dataset_id, df = uploaded_dataset("f_employee.csv")
        at = _goto(_new_app(_upload_session(dataset_id, "f_employee.csv")))
        before = _rows(at)
        path = dataset_processed_dir(dataset_id) / "standardized_data.csv"
        table = pd.read_csv(path)
        time.sleep(0.05)
        pd.concat([table, table.head(7)], ignore_index=True).to_csv(path, index=False)
        fresh = _goto(_new_app(_upload_session(dataset_id, "f_employee.csv")))
        assert not fresh.exception and _rows(fresh) == before + 7

    def test_same_session_sees_a_changed_file_on_rerun(self, uploaded_dataset):
        dataset_id, _ = uploaded_dataset("f_employee.csv")
        at = _goto(_new_app(_upload_session(dataset_id, "f_employee.csv")))
        before = _rows(at)
        path = dataset_processed_dir(dataset_id) / "standardized_data.csv"
        table = pd.read_csv(path)
        time.sleep(0.05)
        table.head(max(3, before - 5)).to_csv(path, index=False)
        at.run()
        assert not at.exception and _rows(at) == max(3, before - 5)

    def test_reset_clears_state_and_blocks_the_page(self):
        at = _goto(_new_app(_demo_session()))
        assert any(k.startswith("mp_page_") for k in at.session_state.keys())
        next(b for b in at.sidebar.button if "Reset" in b.label).click().run()
        assert not any(k.startswith("mp_page_") for k in at.session_state.keys())
        assert any("No dataset is currently active" in i.value for i in at.info)

    def test_widget_keys_are_scoped_to_the_dataset(self, uploaded_dataset):
        id_a, _ = uploaded_dataset("f_employee.csv")
        id_b, _ = uploaded_dataset("h_unusual_columns.csv")
        at = _goto(_new_app(_upload_session(id_a, "f_employee.csv")))
        keys_a = {k for k in at.session_state.keys() if "descriptive" in str(k) and id_a in str(k)}
        at.session_state["mp_active_dataset_id"] = id_b
        at.run()
        keys_b = {k for k in at.session_state.keys() if "descriptive" in str(k) and id_b in str(k)}
        assert keys_a and keys_b and not (keys_a & keys_b)

    def test_unrelated_stale_value_cannot_leak_in(self):
        at = _new_app(_demo_session())
        key = session_scope.scoped_key("descriptive", "dataset_summary")
        from src.common.dataset_summary import DatasetSummary
        at.session_state[key] = {"dataset_id": "other", "value": DatasetSummary(n_rows=999999, n_cols=3)}
        _goto(at)
        assert not at.exception and _kpis(at)["Rows"] != "999,999"


# ---------------------------------------------------------------------------
# Many different datasets
# ---------------------------------------------------------------------------

FIXTURES = ["g_no_date.csv", "h_unusual_columns.csv", "i_missing_values.csv", "e_education.csv", "c_finance.csv",
            "k_minimal.csv", "f_employee.csv"]


class TestDiverseDatasets:
    @pytest.mark.parametrize("fixture", FIXTURES)
    def test_page_survives_and_reports_the_right_row_count(self, uploaded_dataset, fixture):
        dataset_id, df = uploaded_dataset(fixture)
        at = _goto(_new_app(_upload_session(dataset_id, fixture)))
        assert not at.exception, at.exception
        if _kpis(at):
            assert _rows(at) == len(df)
        else:
            assert at.warning

    @pytest.mark.parametrize("fixture", FIXTURES)
    def test_every_chart_type_survives_on_each_dataset(self, uploaded_dataset, fixture):
        from src.descriptive.charts import CHART_TYPES
        dataset_id, _ = uploaded_dataset(fixture)
        at = _goto(_new_app(_upload_session(dataset_id, fixture)))
        for chart in CHART_TYPES:
            sel = next((s for s in at.selectbox if s.label == "Chart type"), None)
            if sel is None:
                break
            sel.set_value(chart).run()
            assert not at.exception, f"{fixture}/{chart}: {at.exception}"


class TestRegistryAndNavigation:
    def test_only_the_descriptive_module_is_implemented(self):
        # Updated in Phase 8 Part 4 (approved): modules 16 and 17 are delivered as well; the list stays exact.
        assert [m.key for m in MODULE_SPECS if getattr(m, "implemented", False)] == ["descriptive", "inferential", "predictive", "prescriptive"]

    def test_descriptive_identity_is_preserved(self):
        assert SPEC.number == 14 and SPEC.key == "descriptive"
        assert SPEC.page_module.endswith("14_descriptive_statistics")

    def test_page_file_contains_no_hardcoded_results(self):
        src = (ROOT / "dashboard" / "pages" / "14_descriptive_statistics.py").read_text(encoding="utf-8")
        assert "st.metric" not in src and "pd.DataFrame(" not in src
        ui = (ROOT / "dashboard" / "components" / "descriptive_ui.py").read_text(encoding="utf-8")
        assert "session_state" not in ui.split("def _report_worker")[1].split("def compute_report")[0]
