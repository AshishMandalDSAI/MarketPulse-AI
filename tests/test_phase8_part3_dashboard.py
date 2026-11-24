"""Phase 8 Part 3 - Inferential Statistics page, rendered with Streamlit AppTest.

Each ``AppTest`` is one user session; Streamlit's cache is process-wide, so several in one process
reproduces several browser tabs on one server (a simulation, not a live multi-browser test).
Expected numbers come from running the engine directly on the dataset's own table.
"""

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

st_testing = pytest.importorskip("streamlit.testing.v1")

from config.settings import DEMO_DATASET_ID, dataset_processed_dir, dataset_upload_dir  # noqa: E402
from src.common import export as common_export  # noqa: E402
from src.common.modules import MODULE_SPECS, get_module_spec  # noqa: E402
from src.inferential import tests as T  # noqa: E402
from tests.helpers import load_fixture  # noqa: E402
from tests.helpers import goto_module  # noqa: E402  (upgrade PART 01)

APP_PATH = ROOT / "app.py"
SPEC = get_module_spec("inferential")
LABEL = SPEC.nav_label
TEST_LABEL = "Which test do you want to run?"

SECTIONS = ["Module Overview", "Dataset Information", "Statistical Test Selection", "Input Controls", "Assumption Checks",
            "Statistical Results", "Effect-Size Interpretation", "Confidence Intervals", "Plain-English Explanation", "Downloads"]


@pytest.fixture(autouse=True)
def _real_excel_row_limit(monkeypatch):
    monkeypatch.setattr(common_export, "EXCEL_MAX_ROWS", 1_048_575)


@pytest.fixture
def uploaded_dataset():
    from src.data_engine.analysis_runner import run_full_analysis
    ids = []

    def make(fixture_name):
        dataset_id = f"test_p8p3_{uuid.uuid4().hex[:10]}"
        df = load_fixture(fixture_name)
        run_full_analysis(dataset_id, df=df)
        ids.append(dataset_id)
        return dataset_id, df

    yield make
    for i in ids:
        shutil.rmtree(dataset_upload_dir(i), ignore_errors=True)
        shutil.rmtree(dataset_processed_dir(i), ignore_errors=True)


def _new_app(session=None):
    at = st_testing.AppTest.from_file(str(APP_PATH), default_timeout=120)
    for k, v in (session or {}).items():
        at.session_state[k] = v
    return at


def _demo_session():
    return {"mp_active_dataset_id": DEMO_DATASET_ID, "mp_active_dataset_filename": "NovaMart (demo)", "mp_demo_explicitly_chosen": True}


def _upload_session(dataset_id, name):
    return {"mp_active_dataset_id": dataset_id, "mp_active_dataset_filename": name, "mp_demo_explicitly_chosen": False}


def _goto(at):
    at.run()
    goto_module(at, SPEC.key)  # PART 01: module is a Statistics tab
    return at


def _switch(at, dataset_id, name):
    at.session_state["mp_active_dataset_id"] = dataset_id
    at.session_state["mp_active_dataset_filename"] = name
    at.run()
    return at


def _select_test(at, key):
    next(s for s in at.selectbox if s.label == TEST_LABEL).set_value(key).run()
    return at


def _sel(at, label):
    return next(s for s in at.selectbox if s.label == label)


def _offered(at):
    """Raw test keys offered by the page (the selectbox shows display labels, set_value needs the raw key)."""
    labels = list(_sel(at, TEST_LABEL).options)
    return [k for k in T.TEST_ORDER if T.TEST_SPECS[k] in labels]


def _titles(at):
    return [t.strip() for m in at.markdown for t in re.findall(r'<div class="mp-section-title">(.*?)</div>', m.value)]


def _page_text(at) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption] + [i.value for i in at.info]
    parts += [w.value for w in at.warning] + [e.value for e in at.error] + [s.value for s in at.success]
    return "\n".join(parts)


def _downloads(at):
    return [e.proto for e in at.get("download_button")]


def _kv(at, key_col) -> dict:
    """Merge every two-column st.table whose index is `key_col` into one dict."""
    out = {}
    for t in at.table:
        f = t.value
        if f.index.name == key_col:
            out.update({str(k): str(v) for k, v in f["Value"].items()})
    return out


def _results(at) -> dict:
    return _kv(at, "Result")


def _missing(at) -> dict:
    return _kv(at, "Measure")


def _analysis_table(dataset_id) -> pd.DataFrame:
    if dataset_id == DEMO_DATASET_ID:
        return pd.read_csv(ROOT / "data" / "processed" / "marketing_data_clean.csv")
    return pd.read_csv(dataset_processed_dir(dataset_id) / "standardized_data.csv")


def _expected_t1(dataset_id, column):
    s = pd.to_numeric(_analysis_table(dataset_id)[column], errors="coerce")
    return T.one_sample_t(s, 0.0, column=column)


def _num(text):
    return float(str(text).replace(",", ""))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def demo_at():
    saved = common_export.EXCEL_MAX_ROWS
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

    def test_status_notice(self, demo_at):
        assert any("Part" in i.value and SPEC.title in i.value for i in demo_at.info)

    def test_every_required_section_in_order(self, demo_at):
        own = [t for t in _titles(demo_at) if t in SECTIONS]
        assert own == SECTIONS, _titles(demo_at)

    def test_three_downloads_csv_zip_excel(self, demo_at):
        urls = [p.url for p in _downloads(demo_at)]
        assert len(urls) == 3 and [u.rsplit(".", 1)[-1] for u in urls] == ["csv", "zip", "xlsx"]

    def test_core_controls_exist(self, demo_at):
        labels = {s.label for s in demo_at.selectbox}
        assert {TEST_LABEL, "Significance level (α)", "Confidence level"} <= labels

    def test_alpha_and_confidence_choices(self, demo_at):
        assert [float(o) for o in _sel(demo_at, "Significance level (α)").options] == [0.01, 0.05, 0.10]
        assert [o.rstrip("%") for o in _sel(demo_at, "Confidence level").options] == ["90", "95", "99"]

    def test_page_has_no_raw_python_or_nan(self, demo_at):
        text = _page_text(demo_at)
        for bad in ("Traceback", "nan%", "<function", "None%"):
            assert bad not in text

    def test_page_file_has_no_hardcoded_results_or_columns(self):
        src = (ROOT / "dashboard" / "pages" / "15_inferential_statistics.py").read_text(encoding="utf-8")
        ui = (ROOT / "dashboard" / "components" / "inferential_ui.py").read_text(encoding="utf-8")
        assert "render_inferential_body" in src
        for demo_column in ("Marketing_Spend", "Revenue", "Campaign", "Channel"):
            assert demo_column not in ui

    def test_no_dataset_gate(self):
        at = _goto(_new_app())
        assert not at.exception
        assert any("No dataset is currently active" in i.value for i in at.info)
        assert not _downloads(at)

    def test_registry_marks_part3_module_only(self):
        assert [m.key for m in MODULE_SPECS if getattr(m, "implemented", False)] == ["descriptive", "inferential", "predictive", "prescriptive"]
        assert SPEC.number == 15 and SPEC.delivered_in_part == 3


# ---------------------------------------------------------------------------
# Every test is selectable and runs (or explains itself) on the demo dataset
# ---------------------------------------------------------------------------

class TestEveryTestOnDemo:
    @pytest.mark.parametrize("key", T.TEST_ORDER)
    def test_selectable_without_exception(self, key):
        at = _goto(_new_app(_demo_session()))
        if key not in _offered(at):                 # unavailable tests are explained, not offered
            assert any("not available for this dataset" in e.label for e in at.expander)
            return
        _select_test(at, key)
        assert not at.exception, at.exception
        assert [t for t in _titles(at) if t in SECTIONS] == SECTIONS
        assert len(_downloads(at)) == 3
        text = _page_text(at)
        assert ("Null hypothesis" in text) or ("was not run" in text) or key == "ci_mean"

    def test_z_test_without_sigma_is_refused_on_the_page(self):
        at = _select_test(_goto(_new_app(_demo_session())), "z")
        assert not at.exception
        text = _page_text(at)
        assert "was not run" in text and "population standard deviation" in text
        assert "Test statistic" not in " ".join(_results(at))
        assert "one-sample t-test" in text.lower()

    def test_z_test_with_sigma_runs(self):
        at = _select_test(_goto(_new_app(_demo_session())), "z")
        box = next(t for t in at.text_input if t.label.startswith("Known population standard deviation"))
        box.set_value("5").run()
        assert not at.exception
        res = _results(at)
        assert any(k.startswith("Test statistic (z") or k.startswith("Test statistic") for k in res)
        assert "P-value" in res

    def test_invalid_numeric_text_input_gives_a_message(self):
        at = _select_test(_goto(_new_app(_demo_session())), "t1")
        next(t for t in at.text_input if t.label.startswith("Hypothesised mean")).set_value("abc").run()
        assert not at.exception
        assert "not a number" in _page_text(at).lower() or "enter a number" in _page_text(at).lower() or at.error

    def test_alpha_changes_the_decision_text(self):
        at = _select_test(_goto(_new_app(_demo_session())), "t1")
        r05 = _results(at)
        _sel(at, "Significance level (α)").set_value(0.01).run()
        assert not at.exception
        assert _results(at)["Significance level (α)"] == "0.01" and r05["Significance level (α)"] == "0.05"

    def test_confidence_level_changes_the_interval_label(self):
        at = _select_test(_goto(_new_app(_demo_session())), "t1")
        _sel(at, "Confidence level").set_value(0.99).run()
        assert not at.exception and "99%" in _page_text(at)

    def test_page_numbers_equal_engine_numbers(self):
        at = _select_test(_goto(_new_app(_demo_session())), "t1")
        col = _sel(at, "Numeric column").value
        exp = _expected_t1(DEMO_DATASET_ID, col)
        res, miss = _results(at), _missing(at)
        stat_key = next(k for k in res if k.startswith("Test statistic"))
        assert _num(res[stat_key]) == pytest.approx(exp.statistic, abs=6e-3)
        assert _num(miss["Final observations used"]) == exp.missing.final_n
        assert _num(miss["Original observations"]) == exp.missing.original_n


# ---------------------------------------------------------------------------
# Diverse datasets
# ---------------------------------------------------------------------------

DIVERSE = ["a_marketing.csv", "b_sales.csv", "c_finance.csv", "d_operations.csv", "e_education.csv", "f_employee.csv",
           "g_no_date.csv", "h_unusual_columns.csv", "i_missing_values.csv", "j_duplicate_rows.csv", "k_minimal.csv"]


class TestDiverseDatasets:
    @pytest.mark.parametrize("fixture", DIVERSE)
    def test_every_fixture_renders_and_every_offered_test_runs(self, uploaded_dataset, fixture):
        dataset_id, _ = uploaded_dataset(fixture)
        at = _goto(_new_app(_upload_session(dataset_id, fixture)))
        assert not at.exception, (fixture, at.exception)
        offered = _offered(at)
        assert offered, fixture
        for key in offered:
            _select_test(at, key)
            assert not at.exception, (fixture, key, at.exception)
            assert [t for t in _titles(at) if t in SECTIONS] == SECTIONS, (fixture, key)
            assert 3 == len(_downloads(at)), (fixture, key)

    def test_unavailable_tests_are_listed_with_reasons(self, uploaded_dataset):
        dataset_id, _ = uploaded_dataset("k_minimal.csv")
        at = _goto(_new_app(_upload_session(dataset_id, "k_minimal.csv")))
        offered = set(_offered(at))
        hidden = [k for k in T.TEST_ORDER if k not in offered]
        if hidden:
            exp = next(e for e in at.expander if "not available for this dataset" in e.label)
            text = " ".join(m.value for m in exp.markdown)
            for k in hidden:
                assert T.TEST_SPECS[k] in text

    def test_missing_values_dataset_discloses_missing_counts(self, uploaded_dataset):
        dataset_id, _ = uploaded_dataset("i_missing_values.csv")
        at = _goto(_new_app(_upload_session(dataset_id, "i_missing_values.csv")))
        col = _sel(at, "Numeric column").value
        exp = _expected_t1(dataset_id, col)
        _select_test(at, "t1")
        miss = _missing(at)
        assert _num(miss["Original observations"]) == exp.missing.original_n
        assert _num(miss["Removed (missing or blank)"]) == exp.missing.removed_n
        assert _num(miss["Final observations used"]) == exp.missing.final_n

    def test_unusual_column_names_are_shown_verbatim_not_hardcoded(self, uploaded_dataset):
        dataset_id, df = uploaded_dataset("h_unusual_columns.csv")
        at = _select_test(_goto(_new_app(_upload_session(dataset_id, "h_unusual_columns.csv"))), "t1")
        col = _sel(at, "Numeric column")
        assert col.value in set(map(str, df.columns)) or col.value in set(map(str, _analysis_table(dataset_id).columns))


# ---------------------------------------------------------------------------
# Isolation: demo -> A -> B -> A -> demo, sessions, reset
# ---------------------------------------------------------------------------

def _snapshot(at):
    res, miss = _results(at), _missing(at)
    return {"col": _sel(at, "Numeric column").value, "stat": _num(res[next(k for k in res if k.startswith("Test statistic"))]),
            "final": _num(miss["Final observations used"]), "orig": _num(miss["Original observations"])}


class TestIsolation:
    def test_uploaded_dataset_shows_its_own_numbers(self, uploaded_dataset):
        dataset_id, _ = uploaded_dataset("f_employee.csv")
        at = _select_test(_goto(_new_app(_upload_session(dataset_id, "f_employee.csv"))), "t1")
        snap = _snapshot(at)
        exp = _expected_t1(dataset_id, snap["col"])
        assert snap["stat"] == pytest.approx(exp.statistic, abs=6e-3) and snap["final"] == exp.missing.final_n
        assert any("f_employee.csv" in c.value for c in at.caption)

    def test_demo_A_B_A_demo_sequence_never_shows_stale_results(self, uploaded_dataset):
        id_a, _ = uploaded_dataset("f_employee.csv")
        id_b, _ = uploaded_dataset("h_unusual_columns.csv")
        id_c, _ = uploaded_dataset("i_missing_values.csv")
        at = _select_test(_goto(_new_app(_demo_session())), "t1")
        order = [(DEMO_DATASET_ID, "NovaMart (demo)"), (id_a, "f_employee.csv"), (id_b, "h_unusual_columns.csv"),
                 (id_c, "i_missing_values.csv"), (id_a, "f_employee.csv"), (DEMO_DATASET_ID, "NovaMart (demo)")]
        seen = {}
        for dataset_id, name in order:
            _switch(at, dataset_id, name)
            assert not at.exception, (name, at.exception)
            if "t1" not in _offered(at):
                continue
            _select_test(at, "t1")
            snap = _snapshot(at)
            exp = _expected_t1(dataset_id, snap["col"])
            assert snap["stat"] == pytest.approx(exp.statistic, abs=6e-3), (name, snap)
            assert snap["final"] == exp.missing.final_n and snap["orig"] == exp.missing.original_n, (name, snap)
            if dataset_id in seen:
                assert seen[dataset_id] == snap, f"{name} changed on revisit"
            seen[dataset_id] = snap
        assert len({json_key(v) for v in seen.values()}) == len(seen), "different datasets produced identical results"

    def test_downloads_differ_between_datasets_and_return_on_revisit(self, uploaded_dataset):
        id_a, _ = uploaded_dataset("f_employee.csv")
        at = _select_test(_goto(_new_app(_demo_session())), "t1")
        demo = [d.url for d in _downloads(at)]
        _select_test(_switch(at, id_a, "f_employee.csv"), "t1")        # widget state of an unrendered dataset is dropped by Streamlit
        up = [d.url for d in _downloads(at)]
        _select_test(_switch(at, DEMO_DATASET_ID, "NovaMart (demo)"), "t1")
        again = [d.url for d in _downloads(at)]
        assert len(demo) == len(up) == 3 and all(d != u for d, u in zip(demo, up))
        # The CSV is byte-for-byte reproducible; the .zip and .xlsx embed file timestamps, so only the CSV is compared on revisit.
        assert again[0] == demo[0]

    def test_back_to_back_sessions_do_not_share_results(self, uploaded_dataset):
        id_a, _ = uploaded_dataset("f_employee.csv")
        id_b, _ = uploaded_dataset("i_missing_values.csv")
        snaps = []
        for dataset_id, name in [(id_a, "f_employee.csv"), (id_b, "i_missing_values.csv"), (id_a, "f_employee.csv")]:
            at = _select_test(_goto(_new_app(_upload_session(dataset_id, name))), "t1")
            snaps.append(_snapshot(at))
            exp = _expected_t1(dataset_id, snaps[-1]["col"])
            assert snaps[-1]["stat"] == pytest.approx(exp.statistic, abs=6e-3)
        assert snaps[0] == snaps[2] and snaps[0] != snaps[1]

    def test_interleaved_sessions_stay_independent(self, uploaded_dataset):
        id_a, _ = uploaded_dataset("f_employee.csv")
        id_b, _ = uploaded_dataset("i_missing_values.csv")
        s1 = _select_test(_goto(_new_app(_upload_session(id_a, "f_employee.csv"))), "t1")
        s2 = _select_test(_goto(_new_app(_upload_session(id_b, "i_missing_values.csv"))), "t1")
        s1.run()
        s2.run()
        e1, e2 = _snapshot(s1), _snapshot(s2)
        assert e1["stat"] == pytest.approx(_expected_t1(id_a, e1["col"]).statistic, abs=6e-3)
        assert e2["stat"] == pytest.approx(_expected_t1(id_b, e2["col"]).statistic, abs=6e-3)

    def test_changed_file_under_the_same_id_is_recomputed(self, uploaded_dataset):
        dataset_id, _ = uploaded_dataset("f_employee.csv")
        at = _select_test(_goto(_new_app(_upload_session(dataset_id, "f_employee.csv"))), "t1")
        before = _snapshot(at)
        path = dataset_processed_dir(dataset_id) / "standardized_data.csv"
        frame = pd.read_csv(path)
        frame = frame.iloc[: max(8, len(frame) // 2)]
        frame.to_csv(path, index=False)
        at.run()
        assert not at.exception
        after = _snapshot(at)
        exp = _expected_t1(dataset_id, after["col"])
        assert after["final"] == exp.missing.final_n and after["stat"] == pytest.approx(exp.statistic, abs=6e-3)
        assert after["final"] != before["final"] or after["stat"] != before["stat"]

    def test_widget_keys_are_scoped_by_module_and_dataset(self, uploaded_dataset):
        id_a, _ = uploaded_dataset("f_employee.csv")
        at = _select_test(_goto(_new_app(_upload_session(id_a, "f_employee.csv"))), "t1")
        keys = [k for k in at.session_state.keys() if k.startswith("mp_page_")]
        assert keys and all("inferential" in k for k in keys if "test" in k)
        assert any(id_a in k for k in keys)

    def test_inferential_state_does_not_touch_descriptive_state(self):
        at = _select_test(_goto(_new_app(_demo_session())), "t1")
        assert not [k for k in at.session_state.keys() if k.startswith("mp_page_descriptive")]

    def test_minimal_dataset_renders_without_exception(self, uploaded_dataset):
        id_a, _ = uploaded_dataset("k_minimal.csv")
        at = _goto(_new_app(_upload_session(id_a, "k_minimal.csv")))
        assert not at.exception

    def test_no_module_level_result_state(self):
        import importlib
        mods = [importlib.import_module(m) for m in ("src.inferential.tests", "src.inferential.assumptions", "src.inferential.interpret",
                                                     "src.inferential.effect_size", "src.inferential.export", "dashboard.components.inferential_ui")]
        for mod in mods:
            for name, val in vars(mod).items():
                if name.startswith("__") or name.isupper():
                    continue
                assert not isinstance(val, (dict, list, set)), f"{mod.__name__}.{name} is mutable module state"


def json_key(snap):
    return (snap["col"], round(snap["stat"], 6), snap["final"], snap["orig"])


# ---------------------------------------------------------------------------
# Regression: earlier modules untouched
# ---------------------------------------------------------------------------

class TestRegression:
    def test_descriptive_page_still_renders(self):
        at = _new_app(_demo_session())
        at.run()
        goto_module(at, "descriptive")
        assert not at.exception

    def test_later_modules_remain_placeholders(self):
        # Narrowed in Phase 8 Part 4 (approved): modules 16-17 are now delivered, so the placeholder text is no longer
        # expected; the page must still render without an exception and show real (not placeholder) content.
        for key in ("predictive", "prescriptive"):
            spec = get_module_spec(key)
            at = _new_app(_demo_session())
            at.run()
            goto_module(at, key)
            assert not at.exception
            assert not any("are delivered in Part" in i.value for i in at.info)
