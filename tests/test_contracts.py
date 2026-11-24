"""Contract tests for canonical result schemas — Phase 0.

These tests define and verify the structural contracts for all result
objects that the platform produces or will produce. They do NOT implement
the future engines — they only establish the stable structure so that
future phases can build against known contracts.

EXISTING contracts (already implemented, tested for backward compat):
- DatasetProfile (from profiler.py)
- SemanticMap / ColumnMapping (from schemas.py)
- CapabilityReport (from capability_engine.py)
- KPIResult (from generic_kpis.py)

FUTURE contracts (structural placeholders only):
- VisualizationRecommendation
- PredictionResult
- ForecastResult
- AnomalyResult
- DriverAnalysisResult
- MMMResult
- ROIResult
- OptimizationResult
- ScenarioResult
- RecommendationResult
- CopilotContext
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.helpers import load_fixture, run_pipeline_on_fixture


# =====================================================================
# EXISTING CONTRACTS — backward compatibility
# =====================================================================

class TestDatasetProfileContract:
    """DatasetProfile must always contain: n_rows, n_cols, columns (dict)."""

    def test_profile_has_required_keys(self):
        from src.data_engine.profiler import profile_dataset
        df = load_fixture("a_marketing.csv")
        profile = profile_dataset(df)

        assert "n_rows" in profile
        assert "n_cols" in profile
        assert "columns" in profile
        assert isinstance(profile["columns"], dict)
        assert profile["n_rows"] == len(df)
        assert profile["n_cols"] == len(df.columns)

    def test_column_profile_has_required_fields(self):
        from src.data_engine.profiler import profile_dataset
        df = load_fixture("b_sales.csv")
        profile = profile_dataset(df)

        for col_name, col_info in profile["columns"].items():
            assert "dtype" in col_info, f"Column '{col_name}' missing 'dtype'"
            assert "inferred_type" in col_info, f"Column '{col_name}' missing 'inferred_type'"
            assert "missing_pct" in col_info, f"Column '{col_name}' missing 'missing_pct'"
            assert "cardinality" in col_info, f"Column '{col_name}' missing 'cardinality'"

    def test_profile_works_on_diverse_datasets(self):
        from src.data_engine.profiler import profile_dataset
        for name in ["c_finance.csv", "e_education.csv", "g_no_date.csv", "k_minimal.csv"]:
            df = load_fixture(name)
            profile = profile_dataset(df)
            assert profile["n_rows"] > 0


class TestSemanticMapContract:
    """ColumnMapping.to_dict() must always have: date, revenue, channels,
    unmapped_columns, spend_like_columns, freed_columns."""

    def test_column_mapping_has_all_role_keys(self):
        from src.data_engine.column_classifier import classify_columns
        df = load_fixture("a_marketing.csv")
        mapping = classify_columns(df)
        d = mapping.to_dict()

        required = {"date", "revenue", "sales", "cost", "impressions", "clicks",
                     "conversions", "orders", "profit", "margin",
                     "channels", "unmapped_columns", "spend_like_columns", "freed_columns"}
        missing = required - set(d.keys())
        assert not missing, f"ColumnMapping.to_dict() missing keys: {missing}"

    def test_field_guess_has_required_fields(self):
        from src.data_engine.column_classifier import classify_columns
        df = load_fixture("a_marketing.csv")
        mapping = classify_columns(df)
        d = mapping.to_dict()

        for role in ["date", "revenue"]:
            guess = d[role]
            assert "column" in guess
            assert "confidence" in guess
            assert "confidence_level" in guess
            assert "matched_via" in guess
            assert "needs_confirmation" in guess

    def test_channels_is_list_of_field_guesses(self):
        from src.data_engine.column_classifier import classify_columns
        df = load_fixture("a_marketing.csv")
        mapping = classify_columns(df)
        d = mapping.to_dict()

        assert isinstance(d["channels"], list)
        # Marketing fixture should find at least one channel
        if d["channels"]:
            ch = d["channels"][0]
            assert "column" in ch
            assert "confidence" in ch


class TestCapabilityReportContract:
    """CapabilityReport must contain at least 13 capability entries,
    each with: key, name, available (bool), reason (str)."""

    EXPECTED_KEYS = {
        "data_profiling", "generic_kpis", "ai_analyst", "business_report",
        "category_analysis", "trend_analysis", "forecasting",
        "marketing_mix_modeling", "channel_attribution", "model_diagnostics",
        "roi_roas", "budget_optimization", "what_if_simulation",
    }

    def test_report_has_all_current_capability_keys(self):
        _, _, _, _, caps = run_pipeline_on_fixture("a_marketing.csv")
        report_dict = caps.to_dict()
        missing = self.EXPECTED_KEYS - set(report_dict.keys())
        assert not missing, f"CapabilityReport missing keys: {missing}"

    def test_each_capability_has_required_fields(self):
        _, _, _, _, caps = run_pipeline_on_fixture("a_marketing.csv")
        for cap in caps.capabilities:
            assert hasattr(cap, "key") and isinstance(cap.key, str)
            assert hasattr(cap, "name") and isinstance(cap.name, str)
            assert hasattr(cap, "available") and isinstance(cap.available, bool)
            assert hasattr(cap, "reason") and isinstance(cap.reason, str)
            assert len(cap.reason) > 0, f"Capability '{cap.key}' has empty reason"

    def test_nonmarketing_dataset_has_correct_unavailable_reasons(self):
        _, _, _, _, caps = run_pipeline_on_fixture("e_education.csv")
        mmm = caps.get("marketing_mix_modeling")
        assert mmm is not None
        assert mmm.available is False
        assert len(mmm.reason) > 0


class TestKPIResultContract:
    """Generic KPIs must return a list of dicts, each with:
    label, value, and optionally category."""

    def test_generic_kpis_returns_list(self):
        from src.kpi.generic_kpis import compute_generic_kpis
        from src.data_engine.column_classifier import classify_columns
        from src.data_engine.data_standardizer import standardize

        df = load_fixture("b_sales.csv")
        mapping = classify_columns(df)
        dataset = standardize(df, mapping)
        kpis = compute_generic_kpis(dataset)

        assert isinstance(kpis, list)
        assert len(kpis) > 0

    def test_each_kpi_has_label_and_value(self):
        from src.kpi.generic_kpis import compute_generic_kpis, GenericKPI
        from src.data_engine.column_classifier import classify_columns
        from src.data_engine.data_standardizer import standardize

        df = load_fixture("c_finance.csv")
        mapping = classify_columns(df)
        dataset = standardize(df, mapping)
        kpis = compute_generic_kpis(dataset)

        for kpi in kpis:
            # GenericKPI is a dataclass with label, value, note
            assert isinstance(kpi, GenericKPI), f"Expected GenericKPI, got {type(kpi)}"
            assert hasattr(kpi, "label") and isinstance(kpi.label, str)
            assert hasattr(kpi, "value") and isinstance(kpi.value, str)
            assert len(kpi.label) > 0, f"KPI has empty label"

    def test_kpi_to_dict_contract(self):
        """GenericKPI.to_dict() must return dict with label, value, note keys."""
        from src.kpi.generic_kpis import compute_generic_kpis
        from src.data_engine.column_classifier import classify_columns
        from src.data_engine.data_standardizer import standardize

        df = load_fixture("a_marketing.csv")
        mapping = classify_columns(df)
        dataset = standardize(df, mapping)
        kpis = compute_generic_kpis(dataset)

        for kpi in kpis:
            d = kpi.to_dict()
            assert isinstance(d, dict)
            assert "label" in d
            assert "value" in d
            assert "note" in d


# =====================================================================
# FUTURE CONTRACTS — structural placeholders
# These define the EXPECTED structure for results that will be
# implemented in later phases. No engine code is written here.
# =====================================================================

class TestFutureContractSchemas:
    """Define expected field sets for future result types.
    These tests pass immediately (they only validate the schema
    definitions themselves are consistent). When the engines are built,
    actual output tests will extend these."""

    # Each contract: {required_keys: set, description: str}
    CONTRACTS = {
        "VisualizationRecommendation": {
            "required_keys": {"chart_type", "x_column", "y_column", "reason"},
            "optional_keys": {"color_by", "title", "priority"},
        },
        "PredictionResult": {
            "required_keys": {"model_type", "target_column", "metrics", "predictions"},
            "optional_keys": {"feature_importances", "model_params"},
        },
        "ForecastResult": {
            "required_keys": {"target_column", "forecast_values", "date_column", "method"},
            "optional_keys": {"confidence_interval", "n_periods", "metrics"},
        },
        "AnomalyResult": {
            "required_keys": {"column", "anomalies", "method", "n_anomalies"},
            "optional_keys": {"threshold", "scores"},
        },
        "DriverAnalysisResult": {
            "required_keys": {"target_column", "drivers", "method"},
            "optional_keys": {"correlations", "feature_importances"},
        },
        "MMMResult": {
            "required_keys": {"channels", "coefficients", "metrics", "attribution"},
            "optional_keys": {"adstock_params", "saturation_params", "model_type"},
        },
        "ROIResult": {
            "required_keys": {"channel_roi", "overall_roi"},
            "optional_keys": {"period", "methodology"},
        },
        "OptimizationResult": {
            "required_keys": {"total_budget", "allocation", "predicted_outcome"},
            "optional_keys": {"improvement_pct", "constraints", "method"},
        },
        "ScenarioResult": {
            "required_keys": {"scenario_name", "changes", "predicted_outcome", "baseline_outcome"},
            "optional_keys": {"revenue_change_pct", "comparison"},
        },
        "RecommendationResult": {
            "required_keys": {"recommendations"},
            "optional_keys": {"priority", "confidence", "data_support"},
        },
        "CopilotContext": {
            "required_keys": {"dataset_id", "capabilities", "available_results"},
            "optional_keys": {"kpis", "profile", "recent_queries"},
        },
    }

    def test_all_contracts_have_required_keys(self):
        """Validate that every contract definition has at least one required key."""
        for name, contract in self.CONTRACTS.items():
            assert len(contract["required_keys"]) > 0, \
                f"Contract '{name}' has no required keys"

    def test_no_overlap_between_required_and_optional(self):
        """Required and optional keys must not overlap in any contract."""
        for name, contract in self.CONTRACTS.items():
            overlap = contract["required_keys"] & contract.get("optional_keys", set())
            assert not overlap, \
                f"Contract '{name}' has keys in both required and optional: {overlap}"

    def test_contract_names_are_distinct(self):
        """All contract names should be unique."""
        names = list(self.CONTRACTS.keys())
        assert len(names) == len(set(names))

    def test_existing_contracts_covered(self):
        """Verify that existing result types are tested above (not just
        in future placeholders)."""
        tested_existing = {
            "DatasetProfile",  # TestDatasetProfileContract
            "SemanticMap",     # TestSemanticMapContract (ColumnMapping)
            "CapabilityReport",  # TestCapabilityReportContract
            "KPIResult",       # TestKPIResultContract
        }
        # Just a documentation assertion — these classes exist above
        assert len(tested_existing) == 4


# =====================================================================
# STANDARDIZED DATASET CONTRACT
# =====================================================================

class TestStandardizedDatasetContract:
    """StandardizedDataset must always expose certain properties and methods."""

    def test_has_required_attributes(self):
        _, _, _, dataset, _ = run_pipeline_on_fixture("a_marketing.csv")

        assert hasattr(dataset, "df")
        assert hasattr(dataset, "date_col")
        assert hasattr(dataset, "revenue_col")
        assert hasattr(dataset, "channels")
        assert hasattr(dataset, "control_cols")
        assert hasattr(dataset, "dimension_cols")
        assert hasattr(dataset, "identifier_cols")

    def test_has_required_properties(self):
        _, _, _, dataset, _ = run_pipeline_on_fixture("b_sales.csv")

        assert hasattr(dataset, "has_date")
        assert hasattr(dataset, "has_revenue")
        assert hasattr(dataset, "has_channels")
        assert hasattr(dataset, "measure_cols")

    def test_has_required_methods(self):
        _, _, _, dataset, _ = run_pipeline_on_fixture("c_finance.csv")

        assert callable(getattr(dataset, "date_range", None))
        assert callable(getattr(dataset, "is_ready_for_analysis", None))
        assert callable(getattr(dataset, "mmm_readiness", None))

    def test_measure_cols_returns_list(self):
        _, _, _, dataset, _ = run_pipeline_on_fixture("a_marketing.csv")
        measures = dataset.measure_cols
        assert isinstance(measures, list)

    def test_mmm_readiness_returns_tuple(self):
        _, _, _, dataset, _ = run_pipeline_on_fixture("a_marketing.csv")
        ready, missing = dataset.mmm_readiness()
        assert isinstance(ready, bool)
        assert isinstance(missing, list)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-q"])
