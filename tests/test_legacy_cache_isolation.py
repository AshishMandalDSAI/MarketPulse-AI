"""Legacy cache isolation tests (dashboard/components/data_loader.py).

Streamlit's cache is shared by every browser session of a server process and is
keyed only by the function and its arguments. The 19 legacy loaders take no
arguments and read the active dataset from session state, so before the fix one
user could be served another user's data. These tests simulate separate user
sessions with Streamlit ``AppTest`` (each ``AppTest`` is one session; the cache
is process-wide, exactly as for two browser tabs on one server) and compare what
the public loaders return with what the dataset's files really contain.

Sections (A-H) follow the authorization for the fix:
A cross-session isolation, B dataset switching, C file modification and cache
hits, D upload replacement, E session reset, F public API compatibility,
G nested loader consistency, H all existing fixture datasets.
"""

import ast
import inspect
import json
import os
import shutil
import sys
import uuid
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

st_testing = pytest.importorskip("streamlit.testing.v1")
import streamlit as st  # noqa: E402

from config.settings import (  # noqa: E402
    DEMO_DATASET_ID, MODELS_DIR, PROCESSED_DATA_DIR,
    dataset_models_dir, dataset_processed_dir, dataset_upload_dir,
)
from dashboard.components import data_loader as dl  # noqa: E402
from src.data_engine.analysis_runner import run_full_analysis  # noqa: E402
from tests.helpers import GENERIC_FIXTURES_DIR, all_diverse_fixtures, load_fixture  # noqa: E402

# name -> declared return type of the 19 public loaders (unchanged by the fix)
LOADER_RETURNS = {
    "load_marketing_data": pd.DataFrame, "load_kpis": dict, "load_generic_kpis": list,
    "load_capabilities": dict, "load_channel_performance": pd.DataFrame,
    "load_channel_attribution": pd.DataFrame, "load_model_metrics": dict,
    "load_model_diagnostics": dict, "load_actual_vs_predicted": pd.DataFrame,
    "load_budget_optimization": dict, "load_budget_table": pd.DataFrame,
    "load_scenario_comparison": pd.DataFrame, "load_forecast": pd.DataFrame,
    "load_monthly_performance": pd.DataFrame, "load_quarterly_performance": pd.DataFrame,
    "load_canonical_metadata": dict, "load_dynamic_analytics": dict,
    "load_prediction_report": dict, "load_phase5_analysis": dict,
}
# loaders that read one file: name -> (directory kind, file name)
SIMPLE_FILES = {
    "load_kpis": ("p", "marketing_kpis.json"), "load_generic_kpis": ("p", "generic_kpis.json"),
    "load_capabilities": ("p", "capabilities.json"),
    "load_channel_performance": ("p", "channel_performance.csv"),
    "load_model_metrics": ("m", "model_metrics.json"),
    "load_model_diagnostics": ("m", "model_diagnostics.json"),
    "load_actual_vs_predicted": ("p", "actual_vs_predicted.csv"),
    "load_budget_optimization": ("p", "budget_optimization_summary.json"),
    "load_budget_table": ("p", "budget_optimization_result.csv"),
    "load_scenario_comparison": ("p", "scenario_comparison.csv"),
    "load_monthly_performance": ("p", "monthly_performance.csv"),
    "load_quarterly_performance": ("p", "quarterly_performance.csv"),
    "load_canonical_metadata": ("p", "canonical_metadata.json"),
    "load_prediction_report": ("p", "prediction_report.json"),
    "load_phase5_analysis": ("p", "phase5_analysis.json"),
}
CHECKED_LOADERS = ["load_marketing_data", "load_channel_attribution", "load_forecast", *SIMPLE_FILES]
EXPECTED_ANNOTATION = dict(LOADER_RETURNS)


# ---------------------------------------------------------------------------
# Ground truth: what the files really contain (independent of the loaders)
# ---------------------------------------------------------------------------

def _dirs(dataset_id):
    if dataset_id == DEMO_DATASET_ID:
        return PROCESSED_DATA_DIR, MODELS_DIR
    return dataset_processed_dir(dataset_id), dataset_models_dir(dataset_id)


def _csv_shape(path, date_col=None):
    """(rows, columns) of a CSV, or (0, ()) if it is missing/unreadable.

    ``date_col`` mirrors the loaders' existing behaviour: when it is given,
    pandas is asked to parse it, and a date column that is not in the file makes
    the read fail, which the loaders (before and after the fix) report as empty.
    """
    if not path.exists():
        return (0, ())
    try:
        df = pd.read_csv(path, parse_dates=[date_col]) if date_col else pd.read_csv(path)
    except Exception:
        return (0, ())
    return (len(df), tuple(df.columns))


def _date_column(dataset_id):
    if dataset_id == DEMO_DATASET_ID:
        return "Date"
    return _json_value(_dirs(dataset_id)[0] / "universal_kpis.json", {}).get("date_column")


def _json_value(path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8")) or default
    except Exception:
        return default


def truth(dataset_id, name):
    """Expected, normalized result of a public loader for a dataset."""
    p, m = _dirs(dataset_id)
    if name == "load_marketing_data":
        path = p / "marketing_data_clean.csv"
        return _csv_shape(path if path.exists() else p / "standardized_data.csv")
    if name == "load_channel_attribution":
        first = _csv_shape(p / "channel_attribution.csv")
        return first if first[0] else _csv_shape(m / "channel_attribution.csv")
    if name == "load_forecast":
        # Ultimate release: the forecast loader no longer asks pandas to parse the
        # dataset's date column in a file whose date column is "Date" (that made every
        # uploaded forecast read as empty — see tests/test_ultimate_upload_flow.py).
        # Ground truth is therefore the file's real shape.
        first = _csv_shape(p / "revenue_forecast.csv")
        return first if first[0] else _csv_shape(m / "revenue_forecast.csv")
    kind, fname = SIMPLE_FILES[name]
    path = (p if kind == "p" else m) / fname
    if name == "load_actual_vs_predicted":
        return _csv_shape(path, _date_column(dataset_id))
    if fname.endswith(".csv"):
        return _csv_shape(path)
    return _json_value(path, [] if name == "load_generic_kpis" else {})


def _normalize(value):
    return (len(value), tuple(value.columns)) if isinstance(value, pd.DataFrame) else value


def _without_ids(value):
    """Dynamic analytics stamps every kpi/chart/insight/relationship with a freshly
    generated random ``id`` (src/analytics/contracts.py, uuid4), so two computations of
    identical data differ only in those ids. Drop them to compare the actual content."""
    if isinstance(value, dict):
        return {k: _without_ids(v) for k, v in value.items() if k != "id"}
    if isinstance(value, list):
        return [_without_ids(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# Simulated user sessions
# ---------------------------------------------------------------------------

def _probe_app():
    import streamlit as st
    from dashboard.components import data_loader as dl, session_paths as sp
    ss = st.session_state
    if ss.get("_reset"):
        ss["_reset"] = False
        sp.reset_analysis()
    target = ss.get("_switch_to")
    if target:
        ss["_switch_to"] = None
        if target == "demo":
            sp.reset_to_demo_dataset()
        else:
            sp.set_active_dataset(target)
    out = {}
    for name in ss["_names"]:
        value = getattr(dl, name)()
        out[name] = (len(value), tuple(value.columns)) if hasattr(value, "columns") else value
    ss["_out"] = out


class User:
    """One browser session. ``switch`` uses the supported set_active_dataset()
    API (which clears the cache); ``dataset`` assigns session state directly,
    so NOTHING clears the cache - the key alone must keep datasets apart."""

    def __init__(self, names=("load_marketing_data",)):
        self.at = st_testing.AppTest.from_function(_probe_app, default_timeout=90)
        self.at.session_state["_names"] = list(names)

    def view(self, dataset=None, switch=None, reset=False):
        if dataset is not None:
            self.at.session_state["mp_active_dataset_id"] = dataset
        if switch is not None:
            self.at.session_state["_switch_to"] = switch
        if reset:
            self.at.session_state["_reset"] = True
        self.at.run()
        assert not self.at.exception, self.at.exception
        return self.at.session_state["_out"]

    def active(self):
        return self.at.session_state["mp_active_dataset_id"] if "mp_active_dataset_id" in self.at.session_state else None


def _view_as(user, dataset_id, mode):
    if mode == "switch_api":
        return user.view(switch=dataset_id)
    return user.view(dataset=dataset_id)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_dataset(fixture_name, subdir="diverse_schemas", df=None):
    dataset_id = f"test_cache_{uuid.uuid4().hex[:10]}"
    frame = df if df is not None else load_fixture(fixture_name, subdir)
    run_full_analysis(dataset_id, df=frame)
    return dataset_id, frame


@pytest.fixture(autouse=True)
def _empty_cache_before_each_test():
    st.cache_data.clear()
    yield


@pytest.fixture(scope="module")
def pair():
    """Two datasets that differ in row count AND column names."""
    ids = []
    a, df_a = _make_dataset("f_employee.csv")
    ids.append(a)
    b, df_b = _make_dataset("h_unusual_columns.csv")
    ids.append(b)
    assert len(df_a) != len(df_b) and list(df_a.columns) != list(df_b.columns)
    yield {"A": a, "B": b, "demo": DEMO_DATASET_ID}
    for i in ids:
        shutil.rmtree(dataset_upload_dir(i), ignore_errors=True)


@pytest.fixture
def scratch_dataset():
    """A dataset the test is allowed to modify on disk."""
    dataset_id, df = _make_dataset("h_unusual_columns.csv")
    yield dataset_id, df
    shutil.rmtree(dataset_upload_dir(dataset_id), ignore_errors=True)


def _table_path(dataset_id):
    p, _ = _dirs(dataset_id)
    path = p / "marketing_data_clean.csv"
    return path if path.exists() else p / "standardized_data.csv"


# ---------------------------------------------------------------------------
# A. Cross-session isolation
# ---------------------------------------------------------------------------

class TestCrossSessionIsolation:
    @pytest.mark.parametrize("mode", ["switch_api", "direct"])
    @pytest.mark.parametrize("first,second", [("A", "B"), ("B", "A")])
    def test_each_user_keeps_their_own_dataset(self, pair, first, second, mode):
        x, y = User(CHECKED_LOADERS), User(CHECKED_LOADERS)
        ds_x, ds_y = pair[first], pair[second]
        assert truth(ds_x, "load_marketing_data") != truth(ds_y, "load_marketing_data")

        out = _view_as(x, ds_x, mode)                          # 1-2 user X selects and loads
        assert {n: out[n] for n in CHECKED_LOADERS} == {n: truth(ds_x, n) for n in CHECKED_LOADERS}
        out = _view_as(y, ds_y, mode)                          # 3-4 user Y selects and loads
        assert {n: out[n] for n in CHECKED_LOADERS} == {n: truth(ds_y, n) for n in CHECKED_LOADERS}
        out = x.view()                                         # 5 X loads again, dataset unchanged
        assert x.active() == ds_x
        got = out["load_marketing_data"]                       # 6 X must still get X's data
        assert got == truth(ds_x, "load_marketing_data")
        assert got[0] == truth(ds_x, "load_marketing_data")[0]        # row count
        assert got[1] == truth(ds_x, "load_marketing_data")[1]        # column names
        assert {n: out[n] for n in CHECKED_LOADERS} == {n: truth(ds_x, n) for n in CHECKED_LOADERS}
        out = y.view()                                         # and Y is still unaffected
        assert {n: out[n] for n in CHECKED_LOADERS} == {n: truth(ds_y, n) for n in CHECKED_LOADERS}

    @pytest.mark.parametrize("upload_first", [True, False])
    def test_demo_user_and_upload_user_never_see_each_other(self, pair, upload_first):
        up, demo = User(), User()
        order = [(up, pair["A"]), (demo, pair["demo"])] if upload_first else [(demo, pair["demo"]), (up, pair["A"])]
        for user, ds in order:
            assert user.view(dataset=ds)["load_marketing_data"] == truth(ds, "load_marketing_data")
        assert up.view()["load_marketing_data"] == truth(pair["A"], "load_marketing_data")
        assert demo.view()["load_marketing_data"] == truth(pair["demo"], "load_marketing_data")


# ---------------------------------------------------------------------------
# B. Dataset switching
# ---------------------------------------------------------------------------

SEQUENCES = [["A", "B"], ["B", "A"], ["demo", "A", "demo"], ["A", "demo", "A"], ["A", "B", "demo", "B", "A"]]


class TestDatasetSwitching:
    @pytest.mark.parametrize("mode", ["switch_api", "direct"])
    @pytest.mark.parametrize("seq", SEQUENCES, ids=lambda s: "->".join(s))
    def test_switch_sequence_never_returns_another_datasets_data(self, pair, seq, mode):
        user = User(CHECKED_LOADERS)
        for key in seq:
            ds = pair[key]
            out = _view_as(user, ds, mode)
            for name in CHECKED_LOADERS:
                assert out[name] == truth(ds, name), f"{name} after switching to {key} ({mode})"


# ---------------------------------------------------------------------------
# C. File modification and cache hits
# ---------------------------------------------------------------------------

@pytest.fixture
def read_counter(monkeypatch):
    calls = []
    original = dl._safe_read_csv

    def counting(path, parse_dates=None):
        calls.append(str(path))
        return original(path, parse_dates=parse_dates)

    monkeypatch.setattr(dl, "_safe_read_csv", counting)
    return calls


class TestFileModification:
    def test_changed_file_is_reread_without_any_global_clear(self, scratch_dataset):
        dataset_id, df = scratch_dataset
        user = User()
        before = user.view(dataset=dataset_id)["load_marketing_data"]
        assert before[0] == len(df)

        path = _table_path(dataset_id)
        full = pd.read_csv(path)
        full.head(10).to_csv(path, index=False)                      # modify file, same dataset id
        assert user.view()["load_marketing_data"][0] == 10           # no st.cache_data.clear()
        full.to_csv(path, index=False)                               # restore
        assert user.view()["load_marketing_data"] == before

    def test_changed_json_file_is_reread_and_missing_file_that_appears_is_picked_up(self, scratch_dataset):
        dataset_id, _ = scratch_dataset
        report = dataset_processed_dir(dataset_id) / "prediction_report.json"
        report.unlink(missing_ok=True)
        user = User(["load_prediction_report"])
        assert user.view(dataset=dataset_id)["load_prediction_report"] == {}     # missing -> {}
        report.write_text(json.dumps({"k": 1}))
        assert user.view()["load_prediction_report"] == {"k": 1}                 # appeared
        report.write_text(json.dumps({"k": 2, "extra": True}))
        assert user.view()["load_prediction_report"] == {"k": 2, "extra": True}  # changed

    def test_cache_hits_when_dataset_path_mtime_and_size_are_unchanged(self, scratch_dataset, read_counter):
        dataset_id, _ = scratch_dataset
        user = User()
        user.view(dataset=dataset_id)
        user.view()
        user.view()
        table_reads = [c for c in read_counter if c == str(_table_path(dataset_id))]
        assert len(table_reads) == 1

    def test_mtime_change_alone_changes_the_key(self, scratch_dataset, read_counter):
        dataset_id, _ = scratch_dataset
        path = _table_path(dataset_id)
        user = User()
        user.view(dataset=dataset_id)
        info = os.stat(path)
        os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns + 5_000_000_000))   # same size, new mtime
        assert os.stat(path).st_size == info.st_size
        user.view()
        assert len([c for c in read_counter if c == str(path)]) == 2

    def test_size_change_alone_changes_the_key(self, scratch_dataset, read_counter):
        dataset_id, _ = scratch_dataset
        path = _table_path(dataset_id)
        user = User()
        user.view(dataset=dataset_id)
        info = os.stat(path)
        path.write_bytes(path.read_bytes() + b"\n")                                # one more byte
        os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns))                    # same mtime
        assert os.stat(path).st_size == info.st_size + 1
        user.view()
        assert len([c for c in read_counter if c == str(path)]) == 2

    def test_loader_clear_forces_a_reread(self, scratch_dataset, read_counter):
        dataset_id, _ = scratch_dataset
        user = User()
        user.view(dataset=dataset_id)
        dl.load_marketing_data.clear()
        user.view()
        assert len([c for c in read_counter if c == str(_table_path(dataset_id))]) == 2

    def test_fingerprint_is_deterministic_absolute_and_never_raises(self, tmp_path):
        f = tmp_path / "x.csv"
        missing = dl._fingerprint(f)
        assert missing == (str(f.resolve()), -1, -1) and dl._fingerprint(f) == missing
        f.write_text("a\n1\n")
        fp = dl._fingerprint(f)
        assert fp[0] == str(f.resolve()) and fp[2] == f.stat().st_size and fp[1] == f.stat().st_mtime_ns
        assert dl._fingerprint(tmp_path) != missing                       # a directory is just another path
        assert dl._fingerprint("bad\0name")[1:] == (-1, -1)               # unusable path: no exception
        assert os.path.isabs(dl._fingerprint("relative/path.csv")[0])


# Final-audit addition: every input file of every loader is checked on its own.
# (Mutation runs showed the tests above only exercised load_marketing_data and
# load_prediction_report: a broken fingerprint in any other loader went unnoticed.)

PROBE_FILES = [(name, kind, fname) for name, (kind, fname) in SIMPLE_FILES.items()] + [
    ("load_marketing_data", "p", "standardized_data.csv"),
    ("load_marketing_data", "p", "marketing_data_clean.csv"),
    ("load_channel_attribution", "p", "channel_attribution.csv"),
    ("load_channel_attribution", "m", "channel_attribution.csv"),
    ("load_forecast", "p", "revenue_forecast.csv"),
    ("load_forecast", "m", "revenue_forecast.csv"),
]


def _write_probe(path, dataset_id, version):
    """Write a small, version-specific file (version 2 is larger than version 1)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.name.endswith(".json"):
        if path.name == "generic_kpis.json":
            payload = [{"probe": version}] * version
        else:
            payload = {"probe": version, **({"extra": True} if version == 2 else {})}
        path.write_text(json.dumps(payload))
    else:
        date_col = _date_column(dataset_id) or "a"
        rows = 3 if version == 1 else 5
        pd.DataFrame({date_col: [f"2024-01-0{i + 1}" for i in range(rows)], "b": range(rows)}).to_csv(path, index=False)


class TestEveryInputFileInvalidatesItsLoader:
    @pytest.mark.parametrize("name,kind,fname", PROBE_FILES, ids=[f"{n}:{k}/{f}" for n, k, f in PROBE_FILES])
    def test_unchanged_file_hits_cache_and_changed_missing_or_restored_file_is_reread(
            self, scratch_dataset, monkeypatch, name, kind, fname):
        dataset_id, _ = scratch_dataset
        p, m = _dirs(dataset_id)
        target = (p if kind == "p" else m) / fname
        # make the file under test the one the loader actually uses
        if name == "load_channel_attribution" and kind == "m":
            (p / "channel_attribution.csv").unlink(missing_ok=True)
        if name == "load_forecast" and kind == "m":
            (p / "revenue_forecast.csv").unlink(missing_ok=True)
        if name == "load_marketing_data" and fname == "standardized_data.csv":
            (p / "marketing_data_clean.csv").unlink(missing_ok=True)

        calls = []                                    # independent count of real file reads
        for attr in ("_safe_read_csv", "_safe_read_json"):
            def make(original):
                def counting(path, *a, **k):
                    calls.append(str(path))
                    return original(path, *a, **k)
                return counting
            monkeypatch.setattr(dl, attr, make(getattr(dl, attr)))

        user = User([name])
        _write_probe(target, dataset_id, 1)
        first = user.view(dataset=dataset_id)[name]
        assert first == truth(dataset_id, name)
        assert (first[0] == 3) if isinstance(first, tuple) else ("probe" in str(first)), "probe file must be seen"
        reads = len(calls)
        assert reads >= 1

        assert user.view()[name] == first and len(calls) == reads, "unchanged file must be a cache hit"

        _write_probe(target, dataset_id, 2)           # changed file, no global clear
        second = user.view()[name]
        assert len(calls) > reads, "changed file must trigger a re-read"
        assert second == truth(dataset_id, name) and second != first

        target.unlink()                               # file goes missing
        gone = user.view()[name]
        assert gone == truth(dataset_id, name) and gone != second

        _write_probe(target, dataset_id, 1)           # file restored
        assert user.view()[name] == first

    def test_dynamic_analytics_follows_the_metadata_file_and_both_table_files(self, scratch_dataset):
        dataset_id, _ = scratch_dataset
        p, _m = _dirs(dataset_id)
        user = User(["load_dynamic_analytics"])
        before = user.view(dataset=dataset_id)["load_dynamic_analytics"]
        assert before and user.view()["load_dynamic_analytics"] == before          # non-empty, then a cache hit

        meta = p / "canonical_metadata.json"
        meta_bytes = meta.read_bytes()
        meta.write_text("{}")                                                       # metadata file changed, no clear
        assert user.view()["load_dynamic_analytics"] == {}
        meta.write_bytes(meta_bytes)                                                # restored
        assert _without_ids(user.view()["load_dynamic_analytics"]) == _without_ids(before)

        clean = p / "marketing_data_clean.csv"                                      # the other table file appears
        assert not clean.exists()
        pd.read_csv(_table_path(dataset_id)).head(12).to_csv(clean, index=False)
        appeared = user.view()["load_dynamic_analytics"]
        assert appeared and _without_ids(appeared) != _without_ids(before)
        st.cache_data.clear()                                                       # independent recompute
        assert _without_ids(appeared) == _without_ids(
            User(["load_dynamic_analytics"]).view(dataset=dataset_id)["load_dynamic_analytics"])
        clean.unlink()                                                              # and disappears again
        assert _without_ids(user.view()["load_dynamic_analytics"]) == _without_ids(before)


# ---------------------------------------------------------------------------
# D. Upload replacement
# ---------------------------------------------------------------------------

class TestUploadReplacement:
    def test_reupload_of_the_same_file_uses_the_new_id_and_never_the_old_cache(self):
        df = load_fixture("f_employee.csv")
        old_id, _ = _make_dataset("f_employee.csv", df=df)
        new_id, _ = _make_dataset("f_employee.csv", df=df.head(40))       # "same file", re-processed
        try:
            assert old_id != new_id
            user = User(CHECKED_LOADERS)
            assert user.view(dataset=old_id)["load_marketing_data"][0] == len(df)
            out = user.view(dataset=new_id)                               # no clear in between
            assert out["load_marketing_data"] == truth(new_id, "load_marketing_data")
            assert out["load_marketing_data"][0] != len(df)
            assert user.view(dataset=old_id)["load_marketing_data"][0] == len(df)
        finally:
            for i in (old_id, new_id):
                shutil.rmtree(dataset_upload_dir(i), ignore_errors=True)

    def test_upload_page_creates_a_fresh_dataset_id_for_every_analysis(self):
        src = (ROOT / "dashboard" / "pages" / "0_upload_data.py").read_text(encoding="utf-8")
        assert 'dataset_id = f"upload_{uuid.uuid4().hex[:12]}"' in src


# ---------------------------------------------------------------------------
# E. Session reset
# ---------------------------------------------------------------------------

class TestSessionReset:
    def test_reset_in_one_session_cannot_expose_or_remove_another_sessions_data(self, pair):
        a, b = User(CHECKED_LOADERS), User(CHECKED_LOADERS)
        assert a.view(dataset=pair["A"])["load_marketing_data"] == truth(pair["A"], "load_marketing_data")
        b.view(dataset=pair["B"])
        out = b.view(reset=True)                      # reset_analysis() also clears the process-wide cache
        assert "mp_active_dataset_id" not in b.at.session_state
        for name in CHECKED_LOADERS:                  # B's direct loader calls fall back to the demo, never to A's data
            assert out[name] == truth(DEMO_DATASET_ID, name), name
        again = a.view()
        for name in CHECKED_LOADERS:                  # A is unaffected and re-reads its own dataset
            assert again[name] == truth(pair["A"], name), name

    def test_after_reset_the_old_dataset_cannot_reappear(self, pair):
        user = User()
        user.view(dataset=pair["A"])
        out = user.view(reset=True)
        assert out["load_marketing_data"] == truth(DEMO_DATASET_ID, "load_marketing_data")
        assert out["load_marketing_data"] != truth(pair["A"], "load_marketing_data")


# ---------------------------------------------------------------------------
# F. Public API compatibility
# ---------------------------------------------------------------------------

class TestPublicApiCompatibility:
    def test_there_are_exactly_nineteen_public_loaders(self):
        assert len(LOADER_RETURNS) == 19

    @pytest.mark.parametrize("name", list(LOADER_RETURNS))
    def test_name_signature_annotation_and_clear_are_unchanged(self, name):
        fn = getattr(dl, name)
        assert callable(fn) and fn.__name__ == name
        sig = inspect.signature(fn)
        assert len(sig.parameters) == 0                                   # zero-argument, as before
        assert sig.return_annotation is EXPECTED_ANNOTATION[name]
        assert callable(fn.clear)
        fn.clear()                                                        # must not raise AttributeError or anything else

    @pytest.mark.parametrize("name", list(LOADER_RETURNS))
    def test_missing_files_return_the_same_empty_values_as_before(self, name):
        ghost = f"test_cache_ghost_{uuid.uuid4().hex[:8]}"                # dataset with no files at all
        value = User([name]).view(dataset=ghost)[name]
        expected = {pd.DataFrame: (0, ()), dict: {}, list: []}[LOADER_RETURNS[name]]
        assert value == expected

    @pytest.mark.parametrize("name", list(LOADER_RETURNS))
    def test_loader_is_a_thin_wrapper_not_itself_a_cached_function(self, name):
        fn = getattr(dl, name)
        assert not hasattr(fn, "__wrapped__")
        assert isinstance(fn, type(lambda: 0))

    def test_unchanged_helpers_still_exist(self):
        for name in ["has_capability", "capability_reason", "load_analysis_context", "pipeline_ready",
                     "_safe_read_csv", "_safe_read_json", "DATA_DIR", "MODELS_DIR"]:
            assert hasattr(dl, name)


# ---------------------------------------------------------------------------
# G. Nested loader consistency
# ---------------------------------------------------------------------------

class TestNestedLoaderConsistency:
    def test_workers_and_helpers_never_touch_session_context(self):
        tree = ast.parse((ROOT / "dashboard" / "components" / "data_loader.py").read_text(encoding="utf-8"))
        checked = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and (node.name.endswith("_worker") or node.name == "_read_marketing_df"):
                checked.append(node.name)
                names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
                attrs = {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}
                assert "session_paths" not in names and "session_state" not in attrs, node.name
                assert not names & set(LOADER_RETURNS), f"{node.name} calls a public loader"
        assert len(checked) == 20            # 19 workers + the shared uncached reader

        # --- follow the private helper call chain used by the cached workers ---
        funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        workers = sorted(n for n in funcs if n.endswith("_worker"))
        assert len(workers) == 19

        def callees(fn):
            """Module-level functions called (by plain name) from this function body."""
            return {c.func.id for c in ast.walk(fn)
                    if isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id in funcs}

        reachable, frontier = set(workers), list(workers)
        while frontier:
            for callee in callees(funcs[frontier.pop()]):
                if callee not in reachable:
                    reachable.add(callee)
                    frontier.append(callee)
        helpers = reachable - set(workers)
        # the chain really reaches the shared helpers (so the walk is not vacuous) ...
        assert {"_read_marketing_df", "_safe_read_csv", "_safe_read_json"} <= helpers, helpers
        # ... and never reaches a public wrapper or a context/fingerprint resolver
        assert not reachable & set(LOADER_RETURNS), reachable & set(LOADER_RETURNS)
        assert not reachable & {"_context", "has_capability", "capability_reason",
                                "load_analysis_context", "pipeline_ready"}

        session_attrs = {"session_state", "get_dataset_id", "get_processed_dir", "get_models_dir",
                         "get_date_column", "is_user_dataset_mode", "get_active_dataset"}
        session_names = {"session_paths", "_context", "st", *LOADER_RETURNS}

        def session_access(fn):
            """Every way a function body could reach the caller's session context."""
            found = set()
            # the body only: the @st.cache_data decorator is not part of what the function does
            for n in (x for stmt in fn.body for x in ast.walk(stmt)):
                if isinstance(n, ast.Name) and n.id in session_names:
                    found.add(n.id)
                elif isinstance(n, ast.Attribute) and n.attr in session_attrs:
                    found.add(n.attr)
                elif isinstance(n, ast.ImportFrom):
                    found |= {a.name for a in n.names if "session" in a.name} | (
                        {n.module} if n.module and "session" in n.module else set())
                elif isinstance(n, ast.Import):
                    found |= {a.name for a in n.names if "session" in a.name}
            return found

        for name in sorted(reachable):
            assert not session_access(funcs[name]), f"{name} reaches session context: {session_access(funcs[name])}"

        # negative control: the same inspection DOES flag the public wrappers, which
        # intentionally resolve the session context before calling a worker
        for public in ("load_marketing_data", "load_kpis", "load_dynamic_analytics"):
            assert "_context" in session_access(funcs[public]), public

    def test_dynamic_analytics_reads_every_input_from_one_dataset(self, pair, monkeypatch):
        seen = []
        original_json, original_df = dl._safe_read_json, dl._read_marketing_df
        monkeypatch.setattr(dl, "_safe_read_json", lambda p: (seen.append(Path(p)), original_json(p))[1])
        monkeypatch.setattr(dl, "_read_marketing_df",
                            lambda pdir, dc: (seen.append(Path(pdir)), original_df(pdir, dc))[1])
        User(["load_dynamic_analytics"]).view(dataset=pair["A"])
        assert seen, "worker did not run"
        pdir = dataset_processed_dir(pair["A"])
        assert all(pdir in (p, *p.parents) for p in seen), seen

    @pytest.mark.parametrize("order", [("A", "B", "A"), ("B", "A", "B")])
    def test_dynamic_analytics_result_belongs_to_the_active_dataset(self, pair, order):
        user = User(["load_dynamic_analytics", "load_canonical_metadata"])
        for key in order:                                              # no clear between datasets
            out = user.view(dataset=pair[key])
            meta = truth(pair[key], "load_canonical_metadata")
            assert out["load_canonical_metadata"] == meta
            dyn = out["load_dynamic_analytics"]
            if dyn:
                assert dyn["dataset_id"] == meta["dataset_id"]

    def test_dynamic_analytics_cache_follows_input_file_changes(self, scratch_dataset):
        dataset_id, df = scratch_dataset
        user = User(["load_dynamic_analytics"])
        before = user.view(dataset=dataset_id)["load_dynamic_analytics"]
        # an empty initial result would prove nothing: the fixture must give a real one
        assert before, "fixture must produce a non-empty dynamic analytics result"
        assert {"dataset_id", "kpis", "charts", "insights"} <= set(before)
        assert before["dataset_id"] == truth(dataset_id, "load_canonical_metadata")["dataset_id"]

        assert user.view()["load_dynamic_analytics"] == before   # unchanged files: a true cache hit (same ids too)

        path = _table_path(dataset_id)
        original_bytes = path.read_bytes()
        pd.read_csv(path).head(12).to_csv(path, index=False)     # change an input file, same dataset id
        try:
            after = user.view()["load_dynamic_analytics"]        # NO global cache clear
            assert after, "result after the file change must be recomputed, not emptied"
            assert after["dataset_id"] == before["dataset_id"]
            assert after != before                               # the stale cached object was not served
            assert _without_ids(after) != _without_ids(before)   # and the content really reflects the new file

            # independent recompute: empty the whole process cache, new session, same files
            st.cache_data.clear()
            recomputed = User(["load_dynamic_analytics"]).view(dataset=dataset_id)["load_dynamic_analytics"]
            assert recomputed
            assert _without_ids(after) == _without_ids(recomputed)   # the no-clear result IS the correct one
        finally:
            path.write_bytes(original_bytes)                     # restore the original file content
        restored = user.view()["load_dynamic_analytics"]         # again no clear: the key changed with the file
        assert restored, "restored file must give a non-empty result"
        assert _without_ids(restored) == _without_ids(before)    # back to the original content


# ---------------------------------------------------------------------------
# H. All existing fixture datasets
# ---------------------------------------------------------------------------

def _fixture_specs():
    specs = [(f"diverse/{n}", n, "diverse_schemas") for n, _ in all_diverse_fixtures()]
    specs += [(f"generic/{p.name}", p.name, "generic_schemas") for p in sorted(GENERIC_FIXTURES_DIR.glob("*.csv"))]
    return specs


@pytest.fixture(scope="module")
def all_fixture_datasets():
    made = {}
    for label, fname, subdir in _fixture_specs():
        made[label], _ = _make_dataset(fname, subdir)
    yield made
    for dataset_id in made.values():
        shutil.rmtree(dataset_upload_dir(dataset_id), ignore_errors=True)


class TestAllFixtureDatasets:
    def test_there_are_sixteen_fixtures(self):
        assert len(_fixture_specs()) == 16

    @pytest.mark.parametrize("label", [s[0] for s in _fixture_specs()])
    def test_every_public_loader_matches_the_datasets_files(self, all_fixture_datasets, label):
        ds = all_fixture_datasets[label]
        user = User([*LOADER_RETURNS])
        out = user.view(dataset=ds)
        for name, expected_type in LOADER_RETURNS.items():
            value = out[name]
            if expected_type is pd.DataFrame:
                assert isinstance(value, tuple) and len(value) == 2
            else:
                assert isinstance(value, expected_type), f"{name}: {type(value)}"
        for name in CHECKED_LOADERS:
            assert out[name] == truth(ds, name), name

    @pytest.mark.parametrize("reverse", [False, True])
    def test_walking_through_all_datasets_in_one_session_never_mixes_them(self, all_fixture_datasets, reverse):
        labels = [s[0] for s in _fixture_specs()]
        labels = list(reversed(labels)) if reverse else labels
        user = User(CHECKED_LOADERS)
        for label in labels:                                           # no cache clear between datasets
            ds = all_fixture_datasets[label]
            out = user.view(dataset=ds)
            for name in CHECKED_LOADERS:
                assert out[name] == truth(ds, name), f"{name} on {label}"

    def test_two_sessions_interleaved_over_all_datasets_stay_isolated(self, all_fixture_datasets):
        labels = [s[0] for s in _fixture_specs()]
        x, y = User(CHECKED_LOADERS), User(CHECKED_LOADERS)
        for lx, ly in zip(labels, reversed(labels)):
            dx, dy = all_fixture_datasets[lx], all_fixture_datasets[ly]
            ox, oy = x.view(dataset=dx), y.view(dataset=dy)
            assert ox["load_marketing_data"] == truth(dx, "load_marketing_data")
            assert oy["load_marketing_data"] == truth(dy, "load_marketing_data")
            assert x.view()["load_marketing_data"] == truth(dx, "load_marketing_data")
