"""Phase 8 Part 4 - end-to-end workflow: CSV and Excel uploads through the real analysis, then Modules 14-17 on the same data."""

import io
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

@pytest.fixture(autouse=True)
def _real_excel_row_limit(monkeypatch):
    from src.common import export as common_export
    monkeypatch.setattr(common_export, "EXCEL_MAX_ROWS", 1_048_575)



def data(n=104, seed=4):
    r = np.random.default_rng(seed)
    d = pd.DataFrame({"week": pd.date_range("2022-01-03", periods=n, freq="W-MON"), "tv_spend": r.uniform(1e4, 5e4, n),
                      "search_spend": r.uniform(5e3, 2e4, n), "social_spend": r.uniform(3e3, 1.5e4, n),
                      "region": r.choice(["north", "south", "east"], n)})
    d["revenue"] = 2e5 + 2 * d.tv_spend ** .9 + 4 * d.search_spend ** .85 + 3 * d.social_spend + r.normal(0, 5e3, n)
    return d


def as_csv(d):
    return pd.read_csv(io.BytesIO(d.to_csv(index=False).encode()))


def as_xlsx(d):
    b = io.BytesIO(); d.to_excel(b, index=False); b.seek(0)
    return pd.read_excel(b)


@pytest.mark.parametrize("loader", [as_csv, as_xlsx], ids=["csv", "excel"])
def test_all_four_modules_work_on_uploaded_file(loader):
    from src.data_engine.analysis_runner import run_full_analysis
    i = f"test_p8p4w_{uuid.uuid4().hex[:10]}"
    run_full_analysis(i, df=loader(data()))
    try:
        at = st_testing.AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
        at.session_state["mp_active_dataset_id"] = i
        at.session_state["mp_active_dataset_filename"] = "up"
        at.session_state["mp_demo_explicitly_chosen"] = False
        at.run()
        for key in ("descriptive", "inferential", "predictive", "prescriptive"):
            goto_module(at, key)
            assert not at.exception, key
        goto_module(at, "predictive")
        next(b for b in at.button if b.label == "Run prediction").click().run()
        assert not at.exception and any("Selected:" in s.value for s in at.success)
        goto_module(at, "prescriptive")
        next(b for b in at.button if b.label == "Find the best allocation").click().run()
        assert not at.exception and any("recommended" in m.label.lower() for m in at.metric)
    finally:
        shutil.rmtree(dataset_upload_dir(i), ignore_errors=True)
        shutil.rmtree(dataset_processed_dir(i), ignore_errors=True)
