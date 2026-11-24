"""Phase 8 Part 3 - FINAL CORRECTION tests (dashboard level, Streamlit AppTest).

Issue 1: the chi-square pages must show an explicit independence warning for time-ordered rows and never present
the test as having satisfied independence. Issue 2: the Spearman page must label its degrees of freedom approximate.
Expected values come from calling the engine directly on the dataset's own table.
"""

import ast
import shutil
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("streamlit.testing.v1")

from config.settings import DEMO_DATASET_ID, dataset_processed_dir, dataset_upload_dir  # noqa: E402
from src.common import export as common_export  # noqa: E402
from src.inferential import tests as T  # noqa: E402
from tests.test_phase8_part3_dashboard import (  # noqa: E402
    _analysis_table, _demo_session, _goto, _new_app, _page_text, _results, _sel, _select_test, _upload_session,
)

TIME_BOX = "My rows are in time order (for example daily or weekly records)"
UI_FILE = ROOT / "dashboard" / "components" / "inferential_ui.py"


@pytest.fixture(autouse=True)
def _real_excel_row_limit(monkeypatch):
    monkeypatch.setattr(common_export, "EXCEL_MAX_ROWS", 1_048_575)


def _frame(sticky: bool, n: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    region = ["North"]
    for _ in range(n - 1):
        region.append(region[-1] if (sticky and rng.random() < 0.9) else str(rng.choice(["North", "South", "East"])))
    return pd.DataFrame({"order_date": pd.date_range("2024-01-01", periods=n, freq="D"), "region": region,
                         "channel": rng.choice(["Web", "Store"], size=n), "revenue": rng.normal(100, 15, size=n).round(2),
                         "units": rng.integers(5, 60, size=n)})


@pytest.fixture
def make_dataset():
    from src.data_engine.analysis_runner import run_full_analysis
    ids = []

    def make(df):
        did = f"test_p8p3c_{uuid.uuid4().hex[:8]}"
        run_full_analysis(did, df=df)
        ids.append(did)
        return did

    yield make
    for i in ids:
        shutil.rmtree(dataset_upload_dir(i), ignore_errors=True)
        shutil.rmtree(dataset_processed_dir(i), ignore_errors=True)


def _checkbox(at):
    return next(c for c in at.checkbox if c.label == TIME_BOX)


def _warn_text(at):
    return "\n".join(w.value for w in at.warning)


def _err_text(at):
    return "\n".join(e.value for e in at.error)


def _expander_labels(at):
    return [e.label for e in at.expander]


# ---------------------------------------------------------------------------
# Issue 1
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", ["chi_gof", "chi_ind"])
class TestChiSquareTimeOrderedPage:
    def test_default_on_a_dated_dataset_shows_control_and_result_warnings(self, make_dataset, key):
        did = make_dataset(_frame(sticky=False))
        at = _select_test(_goto(_new_app(_upload_session(did, "iid.csv"))), key)
        assert not at.exception and _checkbox(at).value is True            # dataset has a date column -> time-ordered default
        text = _warn_text(at)
        assert "Independence warning (time-ordered rows)" in text           # at the control
        assert "Independence NOT established" in text                       # in the results
        assert "does NOT establish independence" in text and "Serial dependence may invalidate ordinary chi-square inference" in text
        assert "Usable with caution" in text and "No assumption problems were found" not in _page_text(at)
        assert any("Independence of observations - caution" in l and l.startswith("⚠️") for l in _expander_labels(at))
        assert not any("Independence of observations - met" in l for l in _expander_labels(at))

    def test_plain_english_section_repeats_the_warning(self, make_dataset, key):
        did = make_dataset(_frame(sticky=False))
        at = _select_test(_goto(_new_app(_upload_session(did, "iid.csv"))), key)
        md = "\n".join(m.value for m in at.markdown)
        assert "**Independence warning.**" in md and "does NOT establish independence" in md

    def test_unchecking_removes_warnings_and_reports_not_checked(self, make_dataset, key):
        did = make_dataset(_frame(sticky=False))
        at = _select_test(_goto(_new_app(_upload_session(did, "iid.csv"))), key)
        _checkbox(at).set_value(False).run()
        assert not at.exception
        assert "Independence warning (time-ordered rows)" not in _warn_text(at)
        assert "Independence NOT established" not in _warn_text(at) + _err_text(at)
        labels = _expander_labels(at)
        assert any("Independence of observations - not checked" in l for l in labels)    # never 'met'
        assert not any("Independence of observations - met" in l for l in labels)

    def test_serial_dependence_is_an_error_banner_but_the_test_still_runs(self, make_dataset, key):
        did = make_dataset(_frame(sticky=True))
        at = _select_test(_goto(_new_app(_upload_session(did, "sticky.csv"))), key)
        assert not at.exception
        err = _err_text(at)
        assert "Independence NOT established" in err and "detected serial dependence" in err
        assert "is **unreliable**" in err
        assert "Test statistic" in " ".join(_results(at))                    # a number is shown (with the warning), not refused
        assert any("Independence of observations - violated" in l for l in _expander_labels(at))

    def test_page_numbers_and_validity_equal_the_engine(self, make_dataset, key):
        did = make_dataset(_frame(sticky=True))
        at = _select_test(_goto(_new_app(_upload_session(did, "sticky.csv"))), key)
        df = _analysis_table(did)
        if key == "chi_gof":
            col = _sel(at, "Categorical column").value
            exp = T.chi_square_gof(df[col], column=col, time_ordered=True)
        else:
            a, b = _sel(at, "First categorical column").value, _sel(at, "Second categorical column").value
            exp = T.chi_square_independence(df[a], df[b], label_a=a, label_b=b, time_ordered=True)
        res = _results(at)
        stat_key = next(k for k in res if k.startswith("Test statistic"))
        assert float(res[stat_key].replace(",", "")) == pytest.approx(exp.statistic, rel=2e-3, abs=2e-3)
        assert exp.validity in ("unreliable", "usable_with_caution")
        assert (exp.validity == "unreliable") == ("is **unreliable**" in _err_text(at))

    def test_demo_dataset_page_never_claims_independence(self, key):
        at = _select_test(_goto(_new_app(_demo_session())), key)
        assert not at.exception
        assert "Independence NOT established" in _warn_text(at) + _err_text(at)
        assert "No assumption problems were found" not in _page_text(at)

    def test_downloads_carry_the_warning(self, make_dataset, key):
        did = make_dataset(_frame(sticky=False))
        at = _select_test(_goto(_new_app(_upload_session(did, "iid.csv"))), key)
        assert len(at.get("download_button")) == 3 and not at.exception


class TestUiPassesTheFlagToBothChiSquareFunctions:
    """Static guard: the checkbox value must reach both chi-square calls (a dropped argument would silently hide the warning)."""

    def test_both_calls_forward_time_ordered(self):
        tree = ast.parse(UI_FILE.read_text(encoding="utf-8"))
        calls = {n.func.attr: n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and n.func.attr in ("chi_square_gof", "chi_square_independence")}
        assert set(calls) == {"chi_square_gof", "chi_square_independence"}
        for name, call in calls.items():
            kw = {k.arg: ast.unparse(k.value) for k in call.keywords}
            assert kw.get("time_ordered") == "time_ordered", name

    def test_chi_square_checkbox_help_and_control_warning_exist(self):
        src = UI_FILE.read_text(encoding="utf-8")
        assert "T.CHI_TIME_ORDER_WARNING" in src and "Independence NOT established" in src


# ---------------------------------------------------------------------------
# Issue 2
# ---------------------------------------------------------------------------

class TestSpearmanApproximateDfPage:
    def test_spearman_page_labels_df_approximate_and_explains(self):
        at = _select_test(_goto(_new_app(_demo_session())), "spearman")
        assert not at.exception
        res = _results(at)
        assert "Degrees of freedom (approximate)" in res and "Degrees of freedom" not in res
        a, b = _sel(at, "First numeric column").value, _sel(at, "Second numeric column").value
        exp = T.spearman_test(_analysis_table(DEMO_DATASET_ID)[a], _analysis_table(DEMO_DATASET_ID)[b], x_label=a, y_label=b)
        assert float(res["Degrees of freedom (approximate)"]) == exp.df
        captions = "\n".join(c.value for c in at.caption)
        assert "About the degrees of freedom" in captions and "Approximate." in captions
        assert "not an exact property of rank correlation" in captions
        md = "\n".join(m.value for m in at.markdown)
        assert "**About the degrees of freedom.**" in md                     # plain-English section too
        assert "APPROXIMATE" in captions                                      # the warning line shown with the results

    def test_pearson_page_keeps_the_exact_label(self):
        at = _select_test(_goto(_new_app(_demo_session())), "pearson")
        res = _results(at)
        assert "Degrees of freedom" in res and "Degrees of freedom (approximate)" not in res
        assert "About the degrees of freedom" not in "\n".join(c.value for c in at.caption)

    def test_chi_square_and_t_pages_keep_the_exact_label(self):
        for key in ("t1", "chi_gof"):
            at = _select_test(_goto(_new_app(_demo_session())), key)
            assert "Degrees of freedom (approximate)" not in _results(at), key
