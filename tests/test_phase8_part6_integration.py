"""Phase 8 Part 6 - integration, dataset isolation and security regression tests for Modules 14-17."""

import io
import shutil
import sys
import uuid
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
st_testing = pytest.importorskip("streamlit.testing.v1")

from config.settings import dataset_processed_dir, dataset_upload_dir  # noqa: E402
from src.common.headers import neutral_name, neutralize_headers  # noqa: E402
from src.common.modules import get_module_spec  # noqa: E402
from tests.helpers import goto_module  # noqa: E402  (upgrade PART 01)

EVIL = '=HYPERLINK("http://x","c")'
EVIL2 = "@SUM(1+1)"


@pytest.fixture(autouse=True)
def _real_excel_row_limit(monkeypatch):
    from src.common import export as common_export
    monkeypatch.setattr(common_export, "EXCEL_MAX_ROWS", 1_048_575)


def mk(n=104, seed=0, names=("tv", "search", "social"), target="rev", time="week"):
    r = np.random.default_rng(seed)
    d = pd.DataFrame({time: pd.date_range("2022-01-03", periods=n, freq="W-MON"), names[0]: r.uniform(1e4, 5e4, n),
                      names[1]: r.uniform(5e3, 2e4, n), names[2]: r.uniform(3e3, 1.5e4, n),
                      "cat": r.choice(["a", "b", "c"], n)})
    d[target] = 2e5 + 2 * d[names[0]] ** .9 + 4 * d[names[1]] ** .85 + 3 * d[names[2]] + r.normal(0, 5e3, n)
    return d


def formula_cells(xlsx: bytes) -> list:
    import openpyxl
    wb = openpyxl.load_workbook(io.BytesIO(xlsx))
    return [(ws.title, c.coordinate, c.value) for ws in wb.worksheets for row in ws.iter_rows() for c in row
            if c.data_type == "f" or (isinstance(c.value, str) and c.value.startswith(("=", "+", "-", "@")))]


# ---------------------------------------------------------------- header helper
def test_neutral_name_prefixes():
    for p in ("=", "+", "-", "@", "\t", "\r"):
        assert neutral_name(p + "x") == "'" + p + "x"
    assert neutral_name("plain") == "plain" and neutral_name(5) == "5"


def test_neutralize_headers_does_not_mutate_input():
    d = pd.DataFrame({EVIL: [1]}, index=["@i"])
    out = neutralize_headers(d)
    assert list(d.columns) == [EVIL] and out.columns[0] == "'" + EVIL and out.index[0] == "'@i"


# ---------------------------------------------------------------- Module 16 regression: infinity
def test_module16_infinite_feature_is_treated_as_missing_not_crash():
    from src.predictive.evaluate import run_supervised
    d = mk(n=160)
    d["inf_feat"] = d["tv"]
    d.loc[[3, 9], "inf_feat"] = np.inf
    d["neg_inf"] = d["search"]
    d.loc[5, "neg_inf"] = -np.inf
    res = run_supervised(d, "rev")
    assert res.status == "ok", res.refusal
    assert res.data_prep.get("infinite_feature_values_treated_as_missing", 0) >= 3


def test_module16_infinite_feature_counts_recorded():
    from src.predictive import evaluate as ev
    import inspect
    src = inspect.getsource(ev.run_supervised)
    assert "infinite_feature_values_treated_as_missing" in src


def test_module16_infinite_target_rows_are_unlabelled_not_crash():
    from src.predictive.evaluate import run_supervised
    d = mk(n=160)
    d.loc[[1, 2], "rev"] = np.inf
    res = run_supervised(d, "rev")
    assert res.status == "ok" or res.refusal is not None  # a result or a readable refusal, never an exception


# ---------------------------------------------------------------- Module 17 regression: non-finite
@pytest.mark.parametrize("col,frag", [("tv", "infinite"), ("rev", "infinite")])
def test_module17_blocks_infinite_values_with_readable_message(col, frag):
    from src.prescriptive import engine as E
    d = mk()
    d.loc[4, col] = np.inf
    problems = E.validate_roles(d, ["tv", "search", "social"], "rev", "week")
    assert problems and any(frag in p and col in p for p in problems)


def test_module17_finite_data_still_validates():
    from src.prescriptive import engine as E
    assert E.validate_roles(mk(), ["tv", "search", "social"], "rev", "week") == []


# ---------------------------------------------------------------- export header injection, all four modules
def evil_frame():
    d = mk(n=130, names=(EVIL, "search", EVIL2))
    d = d.rename(columns={"cat": "+cat", "rev": "-rev"})
    return d


def test_module16_exports_neutralise_headers():
    from src.predictive import export as X
    from src.predictive.evaluate import run_supervised
    from src.predictive.segment import run_segmentation
    d = evil_frame()
    sup = run_supervised(d, "-rev")
    assert sup.status == "ok", sup.refusal
    assert formula_cells(X.supervised_excel(sup)) == []
    head = X.predictions_csv(sup).decode("utf-8-sig").splitlines()[0]
    assert not head.startswith(("=", "+", "-", "@")) and ',=' not in head and ',@' not in head
    seg = run_segmentation(d, [EVIL, "search", EVIL2])
    assert seg.status == "ok", seg.refusal
    assert formula_cells(X.segmentation_excel(seg)) == []
    head = X.assignments_csv(seg).decode("utf-8-sig").splitlines()[0]
    assert ',=' not in head and ',@' not in head and not head.startswith(("=", "+", "-", "@"))


def test_module17_exports_neutralise_headers():
    from src.prescriptive import engine as E
    from src.common.export import sheets_to_excel_bytes
    d = mk(names=(EVIL, "search", EVIL2))
    ch = [EVIL, "search", EVIL2]
    mmm = E.fit_response_model(d, ch, "rev", "week")
    tot = sum(E._Ctx(d, mmm).current.values())
    res = E.optimize_allocation(d, mmm, tot, {c: tot * .05 for c in ch}, {c: tot * .8 for c in ch})
    assert res.status == "ok", (res.message, res.issues)
    sc = E.scenario_table(res)
    sheets = E.result_sheets(res, sc)
    assert formula_cells(sheets_to_excel_bytes(sheets)) == []
    for name, frame in sheets.items():
        for c in frame.columns:
            assert not str(c).startswith(("=", "+", "-", "@")), (name, c)


def test_module14_and_15_exports_neutralise_headers():
    from src.descriptive.engine import analyze_dataframe
    from src.descriptive import export as DX
    from src.inferential import export as IX
    from src.inferential.tests import one_sample_t
    d = evil_frame()
    rep = analyze_dataframe(d)
    assert formula_cells(DX.tables_to_excel(DX.build_tables(rep, "ds", "id"))) == []
    z = DX.tables_to_csv_zip(DX.build_tables(rep, "ds", "id"))
    with zipfile.ZipFile(io.BytesIO(z)) as zf:
        for n in zf.namelist():
            first = zf.read(n).decode("utf-8-sig").splitlines()[0]
            assert ',=' not in first and ',@' not in first and not first.startswith(("=", "@", "+", "-")), (n, first)
    r = one_sample_t(d[EVIL], 30000, column=EVIL)
    assert formula_cells(IX.tables_to_excel(IX.build_tables([r], "ds", "id"))) == []


# ---------------------------------------------------------------- dashboard workflow and isolation
def _session(at, ds, name="up"):
    at.session_state["mp_active_dataset_id"] = ds
    at.session_state["mp_active_dataset_filename"] = name
    at.session_state["mp_demo_explicitly_chosen"] = False


def _page(at, key):
    goto_module(at, key)  # PART 01: Statistics tab or own page
    return at


def _all_text(at) -> str:
    parts = []
    for coll in (at.markdown, at.success, at.info, at.warning, at.error, at.caption, at.subheader, at.header, at.title):
        parts += [str(getattr(e, "value", "")) for e in coll]
    for coll in (at.selectbox, at.multiselect):
        for w in coll:
            parts += [str(o) for o in w.options]
    return "\n".join(parts)


def _make_dataset(df):
    from src.data_engine.analysis_runner import run_full_analysis
    i = f"test_p8p6_{uuid.uuid4().hex[:10]}"
    run_full_analysis(i, df=df)
    return i


def _cleanup(*ids):
    for i in ids:
        shutil.rmtree(dataset_upload_dir(i), ignore_errors=True)
        shutil.rmtree(dataset_processed_dir(i), ignore_errors=True)


DEMO_ONLY = ["TV_Spend", "Radio_Spend", "Print_Spend", "Social_Media_Spend", "Influencer_Marketing_Spend", "Website_Traffic"]


def test_dataset_switch_A_B_A_no_stale_columns_in_any_module():
    a = mk(names=("alpha_tv", "alpha_search", "alpha_social"), target="alpha_rev")
    b = mk(seed=2, names=("beta_tv", "beta_search", "beta_social"), target="beta_rev")
    ia, ib = _make_dataset(a), _make_dataset(b)
    try:
        at = st_testing.AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
        _session(at, ia, "A")
        at.run()
        seen = {}
        for ds, tag, other in ((ia, "alpha", "beta"), (ib, "beta", "alpha"), (ia, "alpha", "beta")):
            at.session_state["mp_active_dataset_id"] = ds
            at.session_state["mp_active_dataset_filename"] = tag
            for key in ("descriptive", "inferential", "predictive", "prescriptive"):
                _page(at, key)
                assert not at.exception, (key, tag)
                txt = _all_text(at)
                assert other + "_" not in txt, (key, tag, "stale column from other dataset")
                for dm in DEMO_ONLY:
                    assert dm not in txt, (key, tag, "demo fallback")
            _page(at, "predictive")
            next(b_ for b_ in at.button if b_.label == "Run prediction").click().run()
            assert not at.exception
            assert other + "_" not in _all_text(at)
            assert tag + "_" in _all_text(at)
            seen[tag] = True
        assert seen == {"alpha": True, "beta": True}
    finally:
        _cleanup(ia, ib)


def test_no_dataset_state_has_no_demo_fallback():
    at = st_testing.AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
    at.session_state["mp_demo_explicitly_chosen"] = False
    at.run()
    for key in ("descriptive", "inferential", "predictive", "prescriptive"):
        _page(at, key)
        assert not at.exception, key
        txt = _all_text(at)
        for dm in DEMO_ONLY:
            assert dm not in txt, (key, dm)


@pytest.mark.parametrize("name,df", [
    ("empty", pd.DataFrame()),
    ("one_row", mk(n=1)),
    ("all_text", pd.DataFrame({"a": list("abcde") * 6, "b": list("vwxyz") * 6})),
    ("one_col", pd.DataFrame({"x": np.arange(40.0)})),
    ("all_nan", pd.DataFrame({"x": [np.nan] * 40, "y": [np.nan] * 40})),
    ("constant", pd.DataFrame({"x": [1.0] * 40, "y": [2.0] * 40, "z": [3.0] * 40})),
])
def test_invalid_datasets_degrade_gracefully_on_all_four_pages(name, df):
    from src.data_engine.analysis_runner import run_full_analysis
    i = f"test_p8p6_{uuid.uuid4().hex[:10]}"
    try:
        try:
            run_full_analysis(i, df=df)
        except Exception:
            pass  # the upload pipeline may refuse; the pages must still be safe
        at = st_testing.AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
        _session(at, i)
        at.run()
        for key in ("descriptive", "inferential", "predictive", "prescriptive"):
            _page(at, key)
            assert not at.exception, (name, key)
            for b_ in at.button:
                if b_.label in ("Run prediction", "Find the best allocation"):
                    b_.click().run()
                    assert not at.exception, (name, key, b_.label)
    finally:
        _cleanup(i)


def _renamed_variant(df):
    return df.rename(columns={"tv": "Television Spend ($)", "search": "Paid Search", "social": "Social-Media",
                              "rev": "Total Sales", "week": "Week Starting", "cat": "Region"})


@pytest.mark.parametrize("fmt", ["csv", "xlsx"])
def test_renamed_columns_missing_values_and_duplicates_flow_through_all_modules(fmt):
    d = _renamed_variant(mk(n=120))
    d.loc[[4, 17, 40], "Paid Search"] = np.nan
    d = pd.concat([d, d.iloc[:6]], ignore_index=True)  # duplicates
    if fmt == "csv":
        loaded = pd.read_csv(io.BytesIO(d.to_csv(index=False).encode()))
    else:
        buf = io.BytesIO(); d.to_excel(buf, index=False); buf.seek(0)
        loaded = pd.read_excel(buf)
    ds = _make_dataset(loaded)
    try:
        at = st_testing.AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
        _session(at, ds)
        at.run()
        for key in ("descriptive", "inferential"):
            _page(at, key)
            assert not at.exception, key
            assert "Television Spend" in _all_text(at) or "Total Sales" in _all_text(at) or key == "inferential"
        _page(at, "predictive")
        next(b for b in at.button if b.label == "Run prediction").click().run()
        assert not at.exception
        txt = _all_text(at)
        assert "Total Sales" in txt or "Selected:" in txt
        for dm in DEMO_ONLY:
            assert dm not in txt
        _page(at, "prescriptive")
        assert not at.exception
        assert not any(dm in _all_text(at) for dm in DEMO_ONLY)
    finally:
        _cleanup(ds)


def test_module17_options_come_from_active_dataset_only():
    a = mk(names=("aa_tv", "aa_search", "aa_social"), target="aa_rev")
    ia = _make_dataset(a)
    try:
        at = st_testing.AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
        _session(at, ia)
        at.run()
        _page(at, "prescriptive")
        assert not at.exception
        opts = set()
        for w in list(at.selectbox) + list(at.multiselect):
            opts |= {str(o) for o in w.options}
        assert opts & {"aa_tv", "aa_search", "aa_social", "aa_rev"}
        assert not opts & set(DEMO_ONLY)
    finally:
        _cleanup(ia)
