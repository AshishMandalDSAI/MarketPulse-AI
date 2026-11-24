"""Phase 8 Part 4 - Predictive Analytics page with Streamlit AppTest (synthetic datasets, run through the real upload analysis)."""

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

from config.settings import dataset_processed_dir, dataset_upload_dir  # noqa: E402
from src.common.modules import get_module_spec  # noqa: E402
from tests.helpers import goto_module  # noqa: E402  (upgrade PART 01)

APP_PATH = ROOT / "app.py"
LABEL = get_module_spec("predictive").nav_label

@pytest.fixture(autouse=True)
def _real_excel_row_limit(monkeypatch):
    from src.common import export as common_export
    monkeypatch.setattr(common_export, "EXCEL_MAX_ROWS", 1_048_575)



def _frame(seed=0, n=240, target="outcome_value"):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"alpha_x": rng.normal(size=n), "beta_x": rng.normal(size=n), "group": rng.choice(list("pqr"), n)})
    d[target] = 2 * d.alpha_x - d.beta_x + rng.normal(size=n) * 0.3
    return d


@pytest.fixture
def make_dataset():
    from src.data_engine.analysis_runner import run_full_analysis
    ids = []

    def make(df):
        i = f"test_p8p4_{uuid.uuid4().hex[:10]}"
        run_full_analysis(i, df=df)
        ids.append(i)
        return i
    yield make
    for i in ids:
        shutil.rmtree(dataset_upload_dir(i), ignore_errors=True)
        shutil.rmtree(dataset_processed_dir(i), ignore_errors=True)


def _open(dataset_id, name="data.csv"):
    at = st_testing.AppTest.from_file(str(APP_PATH), default_timeout=180)
    at.session_state["mp_active_dataset_id"] = dataset_id
    at.session_state["mp_active_dataset_filename"] = name
    at.session_state["mp_demo_explicitly_chosen"] = False
    at.run()
    goto_module(at, "predictive")  # PART 01: module is a Statistics tab
    return at


def _text(at):
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption] + [i.value for i in at.info]
    parts += [w.value for w in at.warning] + [e.value for e in at.error] + [s.value for s in at.success]
    return "\n".join(parts)


def _run_button(at, label):
    return next(b for b in at.button if b.label == label)


def test_page_renders_without_exception_and_no_result_before_run(make_dataset):
    at = _open(make_dataset(_frame()))
    assert not at.exception
    assert "Press **Run prediction**" in _text(at) or "press **Run prediction**" in _text(at)
    assert not at.get("download_button")


def test_run_prediction_shows_selected_model_and_downloads(make_dataset):
    at = _open(make_dataset(_frame()))
    _run_button(at, "Run prediction").click().run()
    assert not at.exception
    t = _text(at)
    assert "Selected:" in t and "Test rows played no part" in t
    assert len(at.get("download_button")) >= 2


def test_leaky_copy_is_reported_not_used(make_dataset):
    df = _frame()
    df["outcome_copy"] = df["outcome_value"]
    at = _open(make_dataset(df))
    sel = next(s for s in at.selectbox if s.label.startswith("Target"))
    sel.set_value("outcome_value").run()
    _run_button(at, "Run prediction").click().run()
    assert not at.exception
    assert any("outcome_copy" in str(d.value) for d in at.dataframe)


def test_dataset_switch_does_not_show_stale_result(make_dataset):
    a = make_dataset(_frame(1))
    b = make_dataset(_frame(2, target="other_value"))
    at = _open(a)
    _run_button(at, "Run prediction").click().run()
    assert "Selected:" in _text(at)
    at.session_state["mp_active_dataset_id"] = b
    at.session_state["mp_active_dataset_filename"] = "b.csv"
    at.run()
    assert "Selected:" not in _text(at)


def test_segmentation_tab_runs(make_dataset):
    at = _open(make_dataset(_frame()))
    _run_button(at, "Run segmentation").click().run()
    assert not at.exception
    assert any("Silhouette" in m.label for m in at.metric)


def test_tiny_dataset_shows_refusal_not_crash(make_dataset):
    at = _open(make_dataset(_frame(n=12)))
    if any(b.label == "Run prediction" for b in at.button):
        _run_button(at, "Run prediction").click().run()
        assert not at.exception
        assert "Cannot run" in _text(at)
