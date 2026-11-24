"""Final stabilization — dashboard page regressions.

1. No page module may call render() at import time: app.py imports the module
   and then calls render() itself, so an import-time call drew the page twice
   on its first visit (duplicate elements; StreamlitDuplicateElementId crash).
2. Pages 01, 11 and 12 must render for an uploaded (non-demo) marketing
   dataset, including the forecasting engine's flat-list prediction output.
"""

from __future__ import annotations

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
sys.path.insert(0, str(ROOT / "tests"))
from dashboard import navigation as _nav  # noqa: E402  (upgrade PART 01)

PAGES_DIR = ROOT / "dashboard" / "pages"


def _top_level_render_calls(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    lines = []
    for node in tree.body:  # only unguarded module-level statements
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            func = node.value.func
            if isinstance(func, ast.Name) and func.id == "render":
                lines.append(node.lineno)
    return lines


@pytest.mark.parametrize("page", sorted(PAGES_DIR.glob("*.py")), ids=lambda p: p.name)
def test_page_does_not_render_at_import_time(page):
    assert _top_level_render_calls(page) == [], (
        f"{page.name} calls render() at module level; guard it with "
        f"`if __name__ == \"__main__\":` (app.py already calls render())."
    )


@pytest.fixture(scope="module")
def uploaded_marketing_dataset():
    from config.settings import dataset_upload_dir
    from src.data_engine.analysis_runner import run_full_analysis
    from src.data_engine.pipeline import profile_and_map

    rng = np.random.default_rng(5)
    n = 80
    df = pd.DataFrame({
        "Transaction_Time": pd.date_range("2023-01-02", periods=n, freq="W").strftime("%Y-%m-%d"),
        "Turnover": rng.normal(80_000, 9_000, n).round(2),
        "Facebook_Spend": rng.uniform(2_000, 8_000, n).round(2),
        "Google_Advertising": rng.uniform(1_000, 6_000, n).round(2),
    })
    dataset_id = f"test_final_{uuid.uuid4().hex[:10]}"
    _, mapping = profile_and_map(df)
    run_full_analysis(dataset_id, df=df, mapping=mapping,
                      confirmed_roles={"date", "revenue", "channels"})
    yield dataset_id
    shutil.rmtree(dataset_upload_dir(dataset_id), ignore_errors=True)


# Upgrade PART 01: labels come from the navigation registry (the menu was renumbered).
@pytest.mark.parametrize("label", [
    _nav.label("executive_overview"),
    _nav.label("predictions"),
    _nav.label("anomalies_drivers"),
])
def test_pages_render_for_uploaded_marketing_dataset(uploaded_marketing_dataset, label):
    from test_phase8_part1_foundation import _new_app, _goto

    at = _goto(_new_app({
        "mp_active_dataset_id": uploaded_marketing_dataset,
        "mp_active_dataset_filename": "tx.csv",
        "mp_demo_explicitly_chosen": False,
    }), label)
    assert not at.exception, f"{label} raised: {at.exception}"


def test_predictions_page_draws_forecast_chart(uploaded_marketing_dataset):
    from test_phase8_part1_foundation import _new_app, _goto

    at = _goto(_new_app({
        "mp_active_dataset_id": uploaded_marketing_dataset,
        "mp_active_dataset_filename": "tx.csv",
        "mp_demo_explicitly_chosen": False,
    }), _nav.label("predictions"))
    assert not at.exception
    charts = at.get("plotly_chart")
    assert len(charts) >= 1, "forecast chart should be drawn from the engine's flat-list output"
