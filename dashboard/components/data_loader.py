"""Cached data-loading helpers shared across all dashboard pages.

Centralizing loads here means expensive operations (reading CSVs, unpickling
the trained model) happen once per session via st.cache_data / st.cache_resource
rather than on every page render (Phase 15 — Performance requirement).
"""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"


@st.cache_data
def load_marketing_data() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "marketing_data_clean.csv", parse_dates=["Date"])


@st.cache_data
def load_kpis() -> dict:
    path = DATA_DIR / "marketing_kpis.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_channel_performance() -> pd.DataFrame:
    path = DATA_DIR / "channel_performance.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_channel_attribution() -> pd.DataFrame:
    path = DATA_DIR / "channel_attribution.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_model_metrics() -> dict:
    path = MODELS_DIR / "model_metrics.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_model_diagnostics() -> dict:
    path = MODELS_DIR / "model_diagnostics.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_actual_vs_predicted() -> pd.DataFrame:
    path = DATA_DIR / "actual_vs_predicted.csv"
    return pd.read_csv(path, parse_dates=["Date"]) if path.exists() else pd.DataFrame()


@st.cache_data
def load_budget_optimization() -> dict:
    path = DATA_DIR / "budget_optimization_summary.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_budget_table() -> pd.DataFrame:
    path = DATA_DIR / "budget_optimization_result.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_scenario_comparison() -> pd.DataFrame:
    path = DATA_DIR / "scenario_comparison.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_forecast() -> pd.DataFrame:
    path = DATA_DIR / "revenue_forecast.csv"
    return pd.read_csv(path, parse_dates=["Date"]) if path.exists() else pd.DataFrame()


@st.cache_data
def load_monthly_performance() -> pd.DataFrame:
    path = DATA_DIR / "monthly_performance.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_quarterly_performance() -> pd.DataFrame:
    path = DATA_DIR / "quarterly_performance.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def pipeline_ready() -> bool:
    """Whether the core pipeline outputs exist (used to show a friendly setup
    message instead of crashing with tracebacks if the pipeline hasn't run)."""
    required = [
        DATA_DIR / "marketing_data_clean.csv",
        DATA_DIR / "channel_attribution.csv",
        MODELS_DIR / "model_metrics.json",
    ]
    return all(p.exists() for p in required)
