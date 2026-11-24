"""Ultimate release — AI analyst honesty and dataset awareness."""

from __future__ import annotations

import os
import shutil
import sys
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ai.analyst import MarketingAnalyst  # noqa: E402

QUESTIONS = ["Which channel should receive more budget?", "Which channel has the best efficiency?",
             "What happens if Search spending increases by 20%?", "Why is revenue declining?",
             "What is the recommended budget allocation?", "What is the capital of France?"]


@pytest.fixture(scope="module")
def uploaded_dataset():
    from config.settings import dataset_processed_dir, dataset_upload_dir
    from src.data_engine.analysis_runner import run_full_analysis
    from src.data_engine.pipeline import profile_and_map
    rng = np.random.default_rng(2)
    n = 70
    df = pd.DataFrame({"Order_Date": pd.date_range("2023-01-02", periods=n, freq="W-MON").strftime("%Y-%m-%d"),
                       "Net_Sales": rng.normal(50_000, 4_000, n).round(2),
                       "Meta_Ads": rng.uniform(1_000, 5_000, n).round(2),
                       "Search_Ad_Cost": rng.uniform(500, 3_000, n).round(2)})
    ds = f"test_ai_{uuid.uuid4().hex[:10]}"
    _, mapping = profile_and_map(df)
    run_full_analysis(ds, df=df, mapping=mapping, confirmed_roles={"date", "revenue", "channels"})
    yield ds, dataset_processed_dir(ds)
    shutil.rmtree(dataset_upload_dir(ds), ignore_errors=True)


def test_never_claims_llm_even_with_key_configured():
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test-not-real"}):
        a = MarketingAnalyst()
        assert a.api_key_available and a.llm_enabled is False
        for q in QUESTIONS:
            r = a.answer(q)
            assert r["llm_used"] is False
            assert "Demo AI Mode" in r["mode"] and "LLM" not in r["mode"]


def test_every_answer_states_its_evidence_type():
    a = MarketingAnalyst()
    kinds = {a.answer(q)["evidence_type"] for q in QUESTIONS}
    assert kinds <= {"observed fact", "model estimate", "model scenario", "recommendation", "guidance"}
    assert a.answer("What happens if Search spending increases by 20%?")["evidence_type"] == "model scenario"


def test_no_causal_incremental_claims():
    a = MarketingAnalyst()
    for q in QUESTIONS:
        assert "incremental revenue" not in a.answer(q)["answer"].lower()


def test_uploaded_dataset_never_uses_demo_simulator(uploaded_dataset):
    ds, data_dir = uploaded_dataset
    a = MarketingAnalyst(data_dir=data_dir, dataset_id=ds)
    assert a.is_demo is False
    r = a.answer("What happens if Search spending increases by 20%?")
    assert "Prescriptive Analytics" in r["answer"]
    for demo_channel in ("TV", "Radio", "Influencer"):
        assert demo_channel not in r["answer"]


def test_uploaded_dataset_answers_without_demo_only_columns(uploaded_dataset):
    ds, data_dir = uploaded_dataset
    a = MarketingAnalyst(data_dir=data_dir, dataset_id=ds)
    r = a.answer("Which channel has the best efficiency?")
    assert any(ch in r["answer"] for ch in ("Meta_Ads", "Search_Ad_Cost"))


def test_fallback_never_reports_fabricated_zero_revenue(uploaded_dataset):
    ds, data_dir = uploaded_dataset
    r = MarketingAnalyst(data_dir=data_dir, dataset_id=ds).answer("Tell me a joke")
    assert "₹0 across" not in r["answer"]
    assert "not a general-purpose chatbot" in r["answer"]
