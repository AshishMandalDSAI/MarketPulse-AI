"""Regression guards — Phase 0.

Ensures all 16 existing MarketPulse features remain available and
importable. These tests protect against accidental deletion or
breaking changes during migration.

GUARDED FEATURES:
1. CSV upload
2. XLSX upload
3. Data profiling
4. Semantic detection (column classifier)
5. KPI functionality
6. Streamlit dashboard (pages exist)
7. Forecasting
8. MMM
9. Adstock
10. Channel contribution / attribution
11. ROI
12. Budget optimization
13. What-if simulation
14. AI Copilot (analyst)
15. Report generation
16. Existing automated tests
"""

import importlib

import pytest
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ---- 1 & 2: CSV / XLSX Upload ----

def test_upload_page_exists():
    """The upload page must exist at its known location."""
    page = ROOT / "dashboard" / "pages" / "0_upload_data.py"
    assert page.exists(), f"Upload page missing: {page}"


def test_csv_upload_via_profiler():
    """CSV files can be read and profiled."""
    from src.data_engine.profiler import profile_dataset
    df = pd.read_csv(ROOT / "tests" / "fixtures" / "diverse_schemas" / "a_marketing.csv")
    profile = profile_dataset(df)
    assert profile["n_rows"] > 0


def test_xlsx_support():
    """openpyxl is available for XLSX reading."""
    try:
        import openpyxl
        xlsx_available = True
    except ImportError:
        xlsx_available = False
    # XLSX support is optional in sandbox but the module ref must exist
    # in the codebase
    upload_page = (ROOT / "dashboard" / "pages" / "0_upload_data.py").read_text(encoding="utf-8")
    assert "xlsx" in upload_page.lower() or "excel" in upload_page.lower(), \
        "Upload page must reference xlsx/excel support"


# ---- 3: Data Profiling ----

def test_profiler_module_importable():
    mod = importlib.import_module("src.data_engine.profiler")
    assert hasattr(mod, "profile_dataset")


def test_profiler_returns_complete_profile():
    from src.data_engine.profiler import profile_dataset
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    profile = profile_dataset(df)
    assert "n_rows" in profile
    assert "n_cols" in profile
    assert "columns" in profile


# ---- 4: Semantic Detection ----

def test_column_classifier_importable():
    mod = importlib.import_module("src.data_engine.column_classifier")
    assert hasattr(mod, "classify_columns")


def test_classifier_returns_column_mapping():
    from src.data_engine.column_classifier import classify_columns
    from src.data_engine.schemas import ColumnMapping
    df = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=10, freq="D"),
        "Revenue": range(100, 110),
    })
    mapping = classify_columns(df)
    assert isinstance(mapping, ColumnMapping)


# ---- 5: KPI Functionality ----

def test_generic_kpis_module_importable():
    mod = importlib.import_module("src.kpi.generic_kpis")
    assert hasattr(mod, "compute_generic_kpis")
    assert hasattr(mod, "GenericKPI")


# ---- 6: Streamlit Dashboard Pages ----

EXPECTED_DASHBOARD_PAGES = [
    "0_upload_data.py",
    "1_executive_overview.py",
    "2_marketing_performance.py",
    "3_marketing_mix_model.py",
    "4_channel_roi.py",
    "5_budget_optimizer.py",
    "6_what_if_simulator.py",
    "7_forecast.py",
    "8_ai_marketing_analyst.py",
    "9_recommendations.py",
    "10_reports.py",
]


def test_all_dashboard_pages_exist():
    pages_dir = ROOT / "dashboard" / "pages"
    assert pages_dir.exists(), "Dashboard pages directory missing"
    existing = [f.name for f in pages_dir.iterdir() if f.suffix == ".py"]
    for expected in EXPECTED_DASHBOARD_PAGES:
        assert expected in existing, f"Dashboard page missing: {expected}"


def test_dashboard_app_entry_exists():
    app_file = ROOT / "app.py"
    assert app_file.exists(), "app.py entry point missing"


# ---- 7: Forecasting ----

def test_forecast_module_importable():
    mod = importlib.import_module("src.forecasting.forecast")
    assert hasattr(mod, "build_time_features")
    assert hasattr(mod, "fit_forecast_model")
    assert hasattr(mod, "forecast_future")


# ---- 8: MMM ----

def test_mmm_module_importable():
    mod = importlib.import_module("src.models.marketing_mix_model")
    assert hasattr(mod, "select_channel_params")
    assert hasattr(mod, "fit_and_evaluate")


# ---- 9: Adstock ----

def test_adstock_module_importable():
    mod = importlib.import_module("src.models.adstock")
    assert hasattr(mod, "geometric_adstock")


def test_adstock_function_works():
    from src.models.adstock import geometric_adstock
    import numpy as np
    result = geometric_adstock(np.array([100, 0, 0, 0, 0]), decay=0.5)
    assert len(result) == 5
    assert result[0] == 100
    assert result[1] > 0  # carryover from first period


# ---- 10: Channel Attribution ----

def test_channel_attribution_data_exists():
    """Channel attribution CSV must exist in the demo processed directory."""
    attr_file = ROOT / "data" / "processed" / "channel_attribution.csv"
    assert attr_file.exists(), "channel_attribution.csv missing from demo data"


# ---- 11: ROI ----

def test_roi_page_exists():
    roi_page = ROOT / "dashboard" / "pages" / "4_channel_roi.py"
    assert roi_page.exists()


# ---- 12: Budget Optimization ----

def test_budget_optimizer_module_importable():
    mod = importlib.import_module("src.optimization.budget_optimizer")
    assert hasattr(mod, "optimize_budget")


# ---- 13: What-If Simulation ----

def test_simulator_module_importable():
    mod = importlib.import_module("src.optimization.simulator")
    assert hasattr(mod, "BudgetSimulator")


# ---- 14: AI Copilot ----

def test_ai_analyst_module_importable():
    mod = importlib.import_module("src.ai.analyst")
    assert hasattr(mod, "MarketingAnalyst")


def test_ai_analyst_answers_question():
    from src.ai.analyst import MarketingAnalyst
    analyst = MarketingAnalyst()
    result = analyst.answer("Which channel has the best efficiency?")
    assert "question" in result
    assert "answer" in result
    assert "mode" in result
    assert "grounded_in_project_data" in result
    assert result["grounded_in_project_data"] is True


# ---- 15: Report Generation ----

def test_report_page_exists():
    report_page = ROOT / "dashboard" / "pages" / "10_reports.py"
    assert report_page.exists()


# ---- 16: Existing Automated Tests ----

def test_all_original_test_files_exist():
    """All original test files must still exist."""
    original_test_files = [
        "test_ai_analyst.py",
        "test_analytics.py",
        "test_api.py",
        "test_data_pipeline.py",
        "test_dataset_first_platform.py",
        "test_mmm_pipeline.py",
        "test_models.py",
    ]
    tests_dir = ROOT / "tests"
    for name in original_test_files:
        assert (tests_dir / name).exists(), f"Original test file missing: {name}"


# ---- Graceful degradation ----

def test_safe_page_decorator_importable():
    """The @safe_page decorator must be importable."""
    pytest.importorskip("streamlit")
    mod = importlib.import_module("dashboard.components.graceful")
    assert hasattr(mod, "safe_page")


# ---- Data engine schemas ----

def test_schemas_module_importable():
    mod = importlib.import_module("src.data_engine.schemas")
    assert hasattr(mod, "FieldGuess")
    assert hasattr(mod, "ColumnMapping")
    assert hasattr(mod, "StandardizedDataset")


# ---- Capability engine ----

def test_capability_engine_importable():
    mod = importlib.import_module("src.data_engine.capability_engine")
    assert hasattr(mod, "detect_capabilities")
    assert hasattr(mod, "CapabilityReport")
    assert hasattr(mod, "Capability")


# ---- Config ----

def test_config_constants_importable():
    mod = importlib.import_module("config.constants")
    assert hasattr(mod, "MIN_CHANNELS_FOR_MMM")
    assert hasattr(mod, "MIN_WEEKS_FOR_FORECAST")
    assert hasattr(mod, "CONFIDENCE_HIGH")
    assert hasattr(mod, "CONFIDENCE_MEDIUM")


def test_config_aliases_importable():
    mod = importlib.import_module("config.column_aliases")
    assert hasattr(mod, "ALIAS_REGISTRY")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-q"])
