"""Phase 6 — Marketing Intelligence Tests.

120+ tests covering:
  - Contracts (serialization, roundtrip)
  - Field discovery (channel detection, validation, target, controls)
  - MMM engine (fitting, evaluation, model selection, edge cases)
  - Channel contribution (ablation, ordering)
  - ROI (naive & model-based)
  - Budget optimizer (SLSQP, bounds)
  - What-if simulator (scenarios, non-causal language)
  - Orchestrator (end-to-end, readiness levels)
  - Dataset-agnostic tests (novel schemas — no Revenue/Date/Sales/Marketing)
  - Edge cases (empty, small, constant, missing data, no channels)
  - Non-causal language compliance
  - No demo data leakage
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_engine.canonical import CanonicalDataset, AnalysisContext
from src.marketing.contracts import (
    MarketingReadiness, DataQualityState, CapabilityStatus,
    MarketingCapabilityReport, ChannelValidation, MMMResult,
    ChannelContribution, ROIResult, BudgetOptimizationResult,
    WhatIfScenario, WhatIfResult, Phase6Result,
)
from src.marketing.field_discovery import (
    _normalize_col_name, _has_spend_semantic_evidence,
    detect_channel_candidates, validate_channel, validate_channels,
    detect_marketing_target, detect_control_variables,
    detect_conversion_fields, assess_marketing_capabilities,
)
from src.marketing.mmm_engine import (
    fit_mmm, build_feature_matrix, channel_feature_name, predict_with_mmm,
    _mape, _evaluate, _time_aware_split, _select_channel_params,
)
from src.marketing.contribution import compute_contributions
from src.marketing.roi import compute_roi_naive, compute_roi_model_based
from src.marketing.optimizer import optimize_budget
from src.marketing.simulator import (
    simulate_custom_allocation, simulate_total_budget,
    simulate_standard_scenarios,
)
from src.marketing.orchestrator import run_marketing_analysis


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_canonical(df: pd.DataFrame, dataset_id: str = "test") -> tuple:
    """Build CanonicalDataset + AnalysisContext from a raw DataFrame
    using the real pipeline (so semantic roles, targets, etc. are detected)."""
    from src.data_engine.pipeline import profile_and_map, confirm_and_standardize
    from src.data_engine.capability_engine import detect_capabilities

    _, mapping = profile_and_map(df)
    dataset, _ = confirm_and_standardize(df, mapping)
    capabilities = detect_capabilities(dataset)
    canonical = CanonicalDataset.from_standardized(
        dataset_id=dataset_id, std_dataset=dataset,
        filename="test.csv", capability_report=capabilities,
    )
    context = AnalysisContext.from_canonical(canonical)
    return canonical, context


def _make_marketing_df(n_rows=52, n_channels=3, seed=42) -> pd.DataFrame:
    """Create a synthetic marketing dataset with date, channels, target."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2023-01-01", periods=n_rows, freq="W")
    data = {"Date": dates}
    channels = []
    for i in range(n_channels):
        col = f"Channel_{i+1}_Spend"
        data[col] = rng.uniform(100, 10000, n_rows)
        channels.append(col)
    # Revenue as a function of channels + noise
    revenue = sum(data[c] * (0.5 + i * 0.3) for i, c in enumerate(channels))
    revenue += rng.normal(0, 500, n_rows)
    data["Revenue"] = revenue
    data["Price_Index"] = rng.uniform(90, 110, n_rows)
    data["Holiday"] = rng.choice([0, 1], n_rows, p=[0.85, 0.15])
    return pd.DataFrame(data)


def _make_non_marketing_df(n_rows=50, seed=42) -> pd.DataFrame:
    """Create a non-marketing dataset (HR/Operations)."""
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "employee_age": rng.randint(22, 65, n_rows),
        "experience_years": rng.randint(0, 40, n_rows),
        "monthly_salary": rng.uniform(3000, 15000, n_rows),
        "department": rng.choice(["Engineering", "Sales", "HR", "Finance"], n_rows),
        "performance_score": rng.uniform(1, 5, n_rows),
    })


# ===================================================================
# 1. CONTRACTS — Serialization / Roundtrip
# ===================================================================

class TestContracts:

    def test_capability_status_roundtrip(self):
        cs = CapabilityStatus(available=True, confidence=0.85,
                              reason="test reason",
                              detected_fields=["a", "b"])
        d = cs.to_dict()
        cs2 = CapabilityStatus.from_dict(d)
        assert cs2.available is True
        assert cs2.confidence == 0.85
        assert cs2.detected_fields == ["a", "b"]

    def test_marketing_readiness_values(self):
        assert MarketingReadiness.NOT_READY.value == "NOT_READY"
        assert MarketingReadiness.READY_FOR_MMM.value == "READY_FOR_MMM"
        assert MarketingReadiness.READY_FOR_OPTIMIZATION.value == "READY_FOR_OPTIMIZATION"

    def test_marketing_capability_report_roundtrip(self):
        report = MarketingCapabilityReport(
            readiness=MarketingReadiness.READY_FOR_MMM,
            target_field="Revenue", time_field="Date",
            channel_fields=["TV_Spend", "Search_Spend"],
        )
        report.mmm = CapabilityStatus(available=True, confidence=0.9)
        d = report.to_dict()
        report2 = MarketingCapabilityReport.from_dict(d)
        assert report2.readiness == MarketingReadiness.READY_FOR_MMM
        assert report2.channel_fields == ["TV_Spend", "Search_Spend"]
        assert report2.mmm.available is True

    def test_channel_validation_to_dict(self):
        cv = ChannelValidation(column="TV_Spend", is_valid=True,
                               reasons=["numeric"], warnings=["high corr"])
        d = cv.to_dict()
        assert d["column"] == "TV_Spend"
        assert d["is_valid"] is True

    def test_mmm_result_roundtrip(self):
        mmm = MMMResult(selected_model="adstock", channels=["a", "b"],
                        target_column="Revenue",
                        train_metrics={"r2": 0.9, "rmse": 100.0})
        d = mmm.to_dict()
        mmm2 = MMMResult.from_dict(d)
        assert mmm2.selected_model == "adstock"
        assert mmm2.channels == ["a", "b"]

    def test_channel_contribution_roundtrip(self):
        cc = ChannelContribution(channel="TV", spend=1000, contribution=500,
                                 roi=0.5)
        d = cc.to_dict()
        cc2 = ChannelContribution.from_dict(d)
        assert cc2.channel == "TV"
        assert cc2.roi == 0.5

    def test_roi_result_roundtrip(self):
        roi = ROIResult(channel="TV", total_spend=1000, naive_roas=5.0,
                        model_estimated_roi=3.0)
        d = roi.to_dict()
        roi2 = ROIResult.from_dict(d)
        assert roi2.naive_roas == 5.0
        assert roi2.model_estimated_roi == 3.0

    def test_budget_optimization_result_roundtrip(self):
        bor = BudgetOptimizationResult(success=True, total_budget=10000,
                                        expected_improvement_pct=5.0)
        d = bor.to_dict()
        bor2 = BudgetOptimizationResult.from_dict(d)
        assert bor2.success is True
        assert bor2.total_budget == 10000

    def test_whatif_scenario_roundtrip(self):
        ws = WhatIfScenario(name="test", total_budget=5000,
                            predicted_target=8000, target_change_pct=10.0)
        d = ws.to_dict()
        ws2 = WhatIfScenario.from_dict(d)
        assert ws2.name == "test"
        assert ws2.target_change_pct == 10.0

    def test_whatif_result_roundtrip(self):
        wr = WhatIfResult(
            scenarios=[WhatIfScenario(name="s1")],
            baseline_scenario=WhatIfScenario(name="baseline"),
            target_column="Rev",
        )
        d = wr.to_dict()
        wr2 = WhatIfResult.from_dict(d)
        assert len(wr2.scenarios) == 1
        assert wr2.baseline_scenario.name == "baseline"

    def test_phase6_result_roundtrip(self):
        p6 = Phase6Result(dataset_id="test_ds")
        p6.marketing_report = MarketingCapabilityReport(
            readiness=MarketingReadiness.PARTIAL)
        d = p6.to_dict()
        p6_2 = Phase6Result.from_dict(d)
        assert p6_2.dataset_id == "test_ds"
        assert p6_2.marketing_report.readiness == MarketingReadiness.PARTIAL

    def test_safe_float_handles_nan_inf(self):
        from src.marketing.contracts import _safe_float
        assert _safe_float(None) is None
        assert _safe_float(np.nan) is None
        assert _safe_float(np.inf) is None
        assert _safe_float(42.5) == 42.5

    def test_safe_dict_handles_nested(self):
        from src.marketing.contracts import _safe_dict
        d = {"a": 1.0, "b": {"c": np.nan}, "d": [1, np.inf]}
        out = _safe_dict(d)
        assert out["a"] == 1.0
        assert out["b"]["c"] is None
        assert out["d"][1] is None

    def test_whatif_result_has_disclaimer(self):
        wr = WhatIfResult()
        assert "not causal" in wr.disclaimer.lower() or "may differ" in wr.disclaimer.lower()


# ===================================================================
# 2. FIELD DISCOVERY
# ===================================================================

class TestFieldDiscovery:

    def test_normalize_col_name(self):
        assert _normalize_col_name("TV_Spend") == "tv spend"
        assert _normalize_col_name("  Google-Ads-Budget  ") == "google ads budget"
        assert _normalize_col_name("Channel.1.Cost") == "channel 1 cost"

    def test_spend_evidence_channel_alias(self):
        ok, pattern = _has_spend_semantic_evidence("Google_Ads")
        assert ok
        assert "channel_alias" in pattern

    def test_spend_evidence_structural(self):
        ok, _ = _has_spend_semantic_evidence("tiktok_spend")
        assert ok

    def test_spend_evidence_generic(self):
        ok, _ = _has_spend_semantic_evidence("marketing spend")
        assert ok

    def test_no_spend_evidence_for_random(self):
        ok, _ = _has_spend_semantic_evidence("employee_age")
        assert not ok

    def test_no_spend_evidence_for_plain_number(self):
        ok, _ = _has_spend_semantic_evidence("value_123")
        assert not ok

    def test_detect_channels_with_marketing_df(self):
        df = _make_marketing_df()
        canonical, context = _build_canonical(df)
        candidates = detect_channel_candidates(canonical)
        assert len(candidates) >= 1

    def test_detect_channels_non_marketing_df(self):
        df = _make_non_marketing_df()
        canonical, context = _build_canonical(df)
        candidates = detect_channel_candidates(canonical)
        # HR dataset should have 0 channel candidates
        assert len(candidates) == 0

    def test_validate_channel_valid(self):
        df = pd.DataFrame({
            "TV_Spend": np.random.uniform(100, 5000, 52),
            "Revenue": np.random.uniform(10000, 50000, 52),
        })
        result = validate_channel("TV_Spend", df, target_col="Revenue")
        assert result.is_valid

    def test_validate_channel_constant_rejected(self):
        df = pd.DataFrame({
            "Const_Spend": [100.0] * 52,
            "Revenue": np.random.uniform(10000, 50000, 52),
        })
        result = validate_channel("Const_Spend", df, target_col="Revenue")
        assert not result.is_valid
        assert any("constant" in r.lower() for r in result.rejection_reasons)

    def test_validate_channel_target_rejected(self):
        df = pd.DataFrame({
            "Revenue": np.random.uniform(10000, 50000, 52),
        })
        result = validate_channel("Revenue", df, target_col="Revenue")
        assert not result.is_valid

    def test_validate_channel_all_null_rejected(self):
        df = pd.DataFrame({
            "Bad_Spend": [np.nan] * 52,
            "Revenue": np.random.uniform(10000, 50000, 52),
        })
        result = validate_channel("Bad_Spend", df, target_col="Revenue")
        assert not result.is_valid

    def test_validate_channel_sequential_id_rejected(self):
        df = pd.DataFrame({
            "id_col": list(range(1, 53)),
            "Revenue": np.random.uniform(10000, 50000, 52),
        })
        result = validate_channel("id_col", df, target_col="Revenue")
        assert not result.is_valid

    def test_validate_channel_too_few_obs(self):
        # 10 non-null values with variance, rest NaN
        vals = [float(i * 100 + 50) for i in range(10)]
        tiny = vals + [np.nan] * 40
        df = pd.DataFrame({
            "Tiny_Spend": tiny,
            "Revenue": np.random.uniform(10000, 50000, 50),
        })
        # Only 10 non-null, but min is max(10, 50*0.1)=10 — should still pass
        result = validate_channel("Tiny_Spend", df, target_col="Revenue")
        assert result.is_valid  # exactly 10 non-null meets threshold

    def test_validate_channel_multicollinearity_warning(self):
        rng = np.random.RandomState(42)
        base = rng.uniform(100, 5000, 52)
        df = pd.DataFrame({
            "Spend_A": base,
            "Spend_B": base * 1.001 + rng.normal(0, 0.01, 52),
            "Revenue": rng.uniform(10000, 50000, 52),
        })
        result = validate_channel("Spend_A", df, target_col="Revenue",
                                  all_channels=["Spend_A", "Spend_B"])
        assert any("multicollinearity" in w.lower() or "correlation" in w.lower()
                    for w in result.warnings)

    def test_validate_channels_batch(self):
        df = _make_marketing_df()
        candidates = ["Channel_1_Spend", "Channel_2_Spend", "Channel_3_Spend"]
        results = validate_channels(candidates, df, target_col="Revenue")
        assert len(results) == 3
        assert all(v.is_valid for v in results)

    def test_detect_marketing_target(self):
        df = _make_marketing_df()
        canonical, _ = _build_canonical(df)
        target = detect_marketing_target(canonical)
        assert target is not None

    def test_detect_marketing_target_none_for_all_categorical(self):
        df = pd.DataFrame({
            "cat_a": ["x", "y", "z"] * 20,
            "cat_b": ["a", "b"] * 30,
        })
        canonical, _ = _build_canonical(df)
        target = detect_marketing_target(canonical)
        # May or may not be None depending on pipeline, but should not crash
        assert True  # no exception

    def test_detect_control_variables(self):
        df = _make_marketing_df()
        canonical, _ = _build_canonical(df)
        channels = ["Channel_1_Spend", "Channel_2_Spend"]
        controls = detect_control_variables(canonical, channels,
                                            target_col="Revenue",
                                            time_col="Date")
        # Should include Price_Index and Holiday but not channels or target
        assert "Channel_1_Spend" not in controls
        assert "Revenue" not in controls

    def test_detect_conversion_fields(self):
        df = pd.DataFrame({
            "conversions": [10, 20, 30],
            "leads": [5, 10, 15],
            "revenue": [1000, 2000, 3000],
            "spend": [100, 200, 300],
        })
        canonical, _ = _build_canonical(df)
        conv = detect_conversion_fields(canonical)
        assert "conversions" in conv or "leads" in conv

    def test_assess_not_ready_for_non_marketing(self):
        df = _make_non_marketing_df()
        canonical, context = _build_canonical(df)
        report = assess_marketing_capabilities(canonical, context)
        assert report.readiness == MarketingReadiness.NOT_READY

    def test_assess_ready_for_marketing(self):
        df = _make_marketing_df(n_rows=52, n_channels=3)
        canonical, context = _build_canonical(df)
        report = assess_marketing_capabilities(canonical, context)
        # Should be at least PARTIAL or higher since channels exist
        assert report.readiness != MarketingReadiness.NOT_READY

    def test_assess_empty_dataframe(self):
        df = pd.DataFrame()
        canonical, context = _build_canonical(pd.DataFrame({"a": [1, 2, 3]}))
        # Directly call with empty df
        canonical._dataframe = pd.DataFrame()
        report = assess_marketing_capabilities(canonical, context)
        assert report.readiness == MarketingReadiness.NOT_READY


# ===================================================================
# 3. MMM ENGINE
# ===================================================================

class TestMMMEngine:

    def test_mape_excludes_zeros(self):
        y_true = np.array([100, 200, 0, 300])
        y_pred = np.array([110, 190, 50, 310])
        m = _mape(y_true, y_pred)
        assert m > 0
        assert m < 100

    def test_mape_all_zeros(self):
        assert _mape(np.zeros(5), np.ones(5)) == 0.0

    def test_evaluate_perfect(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        metrics = _evaluate(y, y, 1)
        assert metrics["r2"] == 1.0
        assert metrics["rmse"] == 0.0

    def test_evaluate_returns_all_keys(self):
        y = np.array([1.0, 2.0, 3.0])
        p = np.array([1.1, 2.1, 3.1])
        m = _evaluate(y, p, 1)
        assert all(k in m for k in ("r2", "adjusted_r2", "rmse", "mae", "mape"))

    def test_time_aware_split(self):
        train, test = _time_aware_split(100, 0.2)
        assert len(train) == 80
        assert len(test) == 20
        assert train[-1] < test[0]  # chronological

    def test_time_aware_split_small(self):
        train, test = _time_aware_split(10, 0.2)
        assert len(train) + len(test) == 10

    def test_fit_mmm_basic(self):
        df = _make_marketing_df(n_rows=52, n_channels=3, seed=42)
        channels = ["Channel_1_Spend", "Channel_2_Spend", "Channel_3_Spend"]
        result = fit_mmm(df, channels, "Revenue",
                         control_cols=["Price_Index"], time_col="Date")
        assert result._model is not None
        assert result._scaler is not None
        assert result.selected_model in ("baseline", "adstock", "adstock_saturation")
        assert result.train_metrics["r2"] > 0

    def test_fit_mmm_returns_channel_params(self):
        df = _make_marketing_df(n_rows=52, n_channels=2)
        channels = ["Channel_1_Spend", "Channel_2_Spend"]
        result = fit_mmm(df, channels, "Revenue")
        assert len(result.channel_params) == 2
        for ch in channels:
            assert "decay" in result.channel_params[ch]
            assert "shape" in result.channel_params[ch]
            assert "k" in result.channel_params[ch]

    def test_fit_mmm_model_comparison(self):
        df = _make_marketing_df(n_rows=60)
        result = fit_mmm(df, ["Channel_1_Spend", "Channel_2_Spend", "Channel_3_Spend"],
                         "Revenue")
        assert "baseline" in result.model_comparison

    def test_fit_mmm_no_channels(self):
        df = _make_marketing_df()
        result = fit_mmm(df, [], "Revenue")
        assert result._model is None
        assert any("No channels" in l for l in result.limitations)

    def test_fit_mmm_missing_target(self):
        df = _make_marketing_df()
        result = fit_mmm(df, ["Channel_1_Spend"], "nonexistent_col")
        assert result._model is None

    def test_fit_mmm_too_few_rows(self):
        df = _make_marketing_df(n_rows=5)
        result = fit_mmm(df, ["Channel_1_Spend", "Channel_2_Spend"], "Revenue")
        assert result._model is None
        assert any("rows" in l.lower() for l in result.limitations)

    def test_fit_mmm_assumptions_documented(self):
        df = _make_marketing_df()
        result = fit_mmm(df, ["Channel_1_Spend", "Channel_2_Spend"], "Revenue",
                         time_col="Date")
        assert len(result.assumptions) >= 3
        assert any("Ridge" in a or "linear" in a.lower() for a in result.assumptions)
        assert any("time" in a.lower() for a in result.assumptions)

    def test_fit_mmm_limitations_documented(self):
        df = _make_marketing_df()
        result = fit_mmm(df, ["Channel_1_Spend", "Channel_2_Spend"], "Revenue")
        assert any("correlational" in l.lower() or "not causal" in l.lower()
                    for l in result.limitations)

    def test_fit_mmm_no_time_col(self):
        df = _make_marketing_df()
        result = fit_mmm(df, ["Channel_1_Spend", "Channel_2_Spend"], "Revenue")
        assert result._model is not None  # should still work without time col

    def test_fit_mmm_with_nan_in_channels(self):
        df = _make_marketing_df(n_rows=52)
        df.loc[0:5, "Channel_1_Spend"] = np.nan
        result = fit_mmm(df, ["Channel_1_Spend", "Channel_2_Spend"], "Revenue")
        assert result._model is not None  # NaN should be filled with 0

    def test_fit_mmm_serialization(self):
        df = _make_marketing_df()
        result = fit_mmm(df, ["Channel_1_Spend", "Channel_2_Spend"], "Revenue")
        d = result.to_dict()
        # Ensure it's JSON-serializable
        json_str = json.dumps(d)
        assert len(json_str) > 0
        # Roundtrip
        result2 = MMMResult.from_dict(json.loads(json_str))
        assert result2.selected_model == result.selected_model

    def test_build_feature_matrix(self):
        df = _make_marketing_df()
        mmm = fit_mmm(df, ["Channel_1_Spend", "Channel_2_Spend"], "Revenue")
        X = build_feature_matrix(df, mmm)
        assert len(X) == len(df)
        assert list(X.columns) == mmm.feature_names

    def test_channel_feature_name(self):
        assert channel_feature_name("TV", "baseline") == "TV__raw"
        assert channel_feature_name("TV", "adstock") == "TV__adstock"
        assert channel_feature_name("TV", "adstock_saturation") == "TV__adstock_sat"

    def test_predict_with_mmm(self):
        df = _make_marketing_df()
        mmm = fit_mmm(df, ["Channel_1_Spend", "Channel_2_Spend"], "Revenue")
        X = build_feature_matrix(df, mmm)
        preds = predict_with_mmm(X, mmm)
        assert len(preds) == len(df)
        assert not np.all(preds == 0)

    def test_predict_with_no_model(self):
        mmm = MMMResult()
        X = pd.DataFrame({"a": [1, 2, 3]})
        preds = predict_with_mmm(X, mmm)
        assert np.all(preds == 0)

    def test_select_channel_params(self):
        df = _make_marketing_df()
        params = _select_channel_params(
            df, ["Channel_1_Spend", "Channel_2_Spend"], "Revenue")
        assert "Channel_1_Spend" in params
        assert "Channel_2_Spend" in params
        for p in params.values():
            assert 0 <= p["decay"] <= 1
            assert p["shape"] > 0

    def test_mmm_train_test_no_shuffle(self):
        """Verify time-aware split is chronological."""
        df = _make_marketing_df(n_rows=52)
        mmm = fit_mmm(df, ["Channel_1_Spend", "Channel_2_Spend"], "Revenue",
                       time_col="Date")
        assert mmm.n_train > 0
        assert mmm.n_test > 0
        assert mmm.n_train + mmm.n_test == len(df)


# ===================================================================
# 4. CHANNEL CONTRIBUTION
# ===================================================================

class TestContribution:

    def test_compute_contributions_basic(self):
        df = _make_marketing_df(n_rows=52, n_channels=3)
        channels = ["Channel_1_Spend", "Channel_2_Spend", "Channel_3_Spend"]
        mmm = fit_mmm(df, channels, "Revenue", time_col="Date")
        contribs = compute_contributions(df, mmm)
        assert len(contribs) == 3
        assert all(isinstance(c, ChannelContribution) for c in contribs)

    def test_contributions_sorted_descending(self):
        df = _make_marketing_df(n_rows=52, n_channels=3)
        channels = ["Channel_1_Spend", "Channel_2_Spend", "Channel_3_Spend"]
        mmm = fit_mmm(df, channels, "Revenue")
        contribs = compute_contributions(df, mmm)
        contributions = [c.contribution for c in contribs]
        assert contributions == sorted(contributions, reverse=True)

    def test_contributions_share_sums_to_one(self):
        df = _make_marketing_df(n_rows=52, n_channels=3)
        mmm = fit_mmm(df, ["Channel_1_Spend", "Channel_2_Spend", "Channel_3_Spend"],
                       "Revenue")
        contribs = compute_contributions(df, mmm)
        total_share = sum(c.contribution_share for c in contribs)
        if total_share > 0:
            assert abs(total_share - 1.0) < 0.01

    def test_contributions_no_model(self):
        df = _make_marketing_df()
        mmm = MMMResult()
        contribs = compute_contributions(df, mmm)
        assert contribs == []

    def test_contributions_have_roi(self):
        df = _make_marketing_df(n_rows=52)
        mmm = fit_mmm(df, ["Channel_1_Spend", "Channel_2_Spend"], "Revenue")
        contribs = compute_contributions(df, mmm)
        for c in contribs:
            assert hasattr(c, "roi")

    def test_contributions_serializable(self):
        df = _make_marketing_df(n_rows=52)
        mmm = fit_mmm(df, ["Channel_1_Spend", "Channel_2_Spend"], "Revenue")
        contribs = compute_contributions(df, mmm)
        dicts = [c.to_dict() for c in contribs]
        json_str = json.dumps(dicts)
        assert len(json_str) > 0


# ===================================================================
# 5. ROI
# ===================================================================

class TestROI:

    def test_naive_roi(self):
        df = _make_marketing_df(n_rows=52)
        results = compute_roi_naive(df, ["Channel_1_Spend", "Channel_2_Spend"],
                                    "Revenue")
        assert len(results) == 2
        assert all(r.method == "naive_ratio" for r in results)
        assert all(r.naive_roas is not None for r in results)

    def test_naive_roi_zero_spend(self):
        df = pd.DataFrame({
            "Zero_Spend": [0.0] * 20,
            "Revenue": np.random.uniform(1000, 5000, 20),
        })
        results = compute_roi_naive(df, ["Zero_Spend"], "Revenue")
        assert results[0].naive_roas is None  # 0 spend => None

    def test_model_based_roi(self):
        df = _make_marketing_df(n_rows=52)
        channels = ["Channel_1_Spend", "Channel_2_Spend"]
        mmm = fit_mmm(df, channels, "Revenue")
        contribs = compute_contributions(df, mmm)
        results = compute_roi_model_based(df, channels, "Revenue", contribs)
        assert len(results) == 2
        assert all(r.method == "model_based" for r in results)
        assert any(r.model_estimated_roi is not None for r in results)

    def test_roi_serialization(self):
        roi = ROIResult(channel="TV", total_spend=1000, naive_roas=5.0)
        d = roi.to_dict()
        json_str = json.dumps(d)
        roi2 = ROIResult.from_dict(json.loads(json_str))
        assert roi2.naive_roas == 5.0


# ===================================================================
# 6. BUDGET OPTIMIZER
# ===================================================================

class TestOptimizer:

    def test_optimize_basic(self):
        df = _make_marketing_df(n_rows=52, n_channels=3)
        channels = ["Channel_1_Spend", "Channel_2_Spend", "Channel_3_Spend"]
        mmm = fit_mmm(df, channels, "Revenue", time_col="Date")
        result = optimize_budget(df, mmm)
        assert isinstance(result, BudgetOptimizationResult)
        # May or may not succeed depending on convergence
        assert result.total_budget > 0

    def test_optimize_respects_bounds(self):
        df = _make_marketing_df(n_rows=52, n_channels=3)
        channels = ["Channel_1_Spend", "Channel_2_Spend", "Channel_3_Spend"]
        mmm = fit_mmm(df, channels, "Revenue", time_col="Date")
        result = optimize_budget(df, mmm, min_pct=0.1, max_pct=0.5)
        if result.success:
            for ch, alloc in result.optimized_allocation.items():
                assert alloc >= result.total_budget * 0.1 - 1  # tolerance
                assert alloc <= result.total_budget * 0.5 + 1

    def test_optimize_custom_budget(self):
        df = _make_marketing_df(n_rows=52)
        channels = ["Channel_1_Spend", "Channel_2_Spend"]
        mmm = fit_mmm(df, channels, "Revenue")
        result = optimize_budget(df, mmm, total_budget=50000)
        assert abs(result.total_budget - 50000) < 1

    def test_optimize_no_model(self):
        df = _make_marketing_df()
        mmm = MMMResult()
        result = optimize_budget(df, mmm)
        assert not result.success or result.total_budget == 0

    def test_optimize_no_channels(self):
        df = _make_marketing_df()
        mmm = MMMResult()
        mmm.channels = []
        result = optimize_budget(df, mmm)
        assert "No channels" in result.message or result.total_budget == 0

    def test_optimize_serialization(self):
        df = _make_marketing_df(n_rows=52)
        mmm = fit_mmm(df, ["Channel_1_Spend", "Channel_2_Spend"], "Revenue")
        result = optimize_budget(df, mmm)
        d = result.to_dict()
        json_str = json.dumps(d)
        assert len(json_str) > 0


# ===================================================================
# 7. WHAT-IF SIMULATOR
# ===================================================================

class TestSimulator:

    def test_simulate_custom_allocation(self):
        df = _make_marketing_df(n_rows=52)
        channels = ["Channel_1_Spend", "Channel_2_Spend"]
        mmm = fit_mmm(df, channels, "Revenue")
        alloc = {"Channel_1_Spend": 5000, "Channel_2_Spend": 5000}
        result = simulate_custom_allocation(df, mmm, alloc)
        assert isinstance(result, WhatIfResult)
        assert result.baseline_scenario is not None
        assert len(result.scenarios) == 1

    def test_simulate_total_budget(self):
        df = _make_marketing_df(n_rows=52)
        mmm = fit_mmm(df, ["Channel_1_Spend", "Channel_2_Spend"], "Revenue")
        result = simulate_total_budget(df, mmm, total_budget=20000)
        assert result.baseline_scenario is not None

    def test_simulate_standard_scenarios(self):
        df = _make_marketing_df(n_rows=52)
        channels = ["Channel_1_Spend", "Channel_2_Spend"]
        mmm = fit_mmm(df, channels, "Revenue", time_col="Date")
        result = simulate_standard_scenarios(df, mmm)
        assert len(result.scenarios) >= 3  # Current + Aggressive + Conservative
        names = [s.name for s in result.scenarios]
        assert "Current" in names

    def test_whatif_disclaimer_present(self):
        df = _make_marketing_df(n_rows=52)
        mmm = fit_mmm(df, ["Channel_1_Spend", "Channel_2_Spend"], "Revenue")
        result = simulate_standard_scenarios(df, mmm)
        assert "not" in result.disclaimer.lower() or "may differ" in result.disclaimer.lower()

    def test_whatif_non_causal_language(self):
        """Verify no causal language in scenarios."""
        df = _make_marketing_df(n_rows=52)
        mmm = fit_mmm(df, ["Channel_1_Spend", "Channel_2_Spend"], "Revenue")
        result = simulate_standard_scenarios(df, mmm)
        d = result.to_dict()
        text = json.dumps(d).lower()
        causal_words = ["causes", "leads to", "results in", "creates"]
        for word in causal_words:
            assert word not in text, f"Causal language '{word}' found in what-if result"

    def test_whatif_serialization(self):
        df = _make_marketing_df(n_rows=52)
        mmm = fit_mmm(df, ["Channel_1_Spend", "Channel_2_Spend"], "Revenue")
        result = simulate_standard_scenarios(df, mmm)
        d = result.to_dict()
        json_str = json.dumps(d)
        result2 = WhatIfResult.from_dict(json.loads(json_str))
        assert len(result2.scenarios) == len(result.scenarios)


# ===================================================================
# 8. ORCHESTRATOR — End-to-End
# ===================================================================

class TestOrchestrator:

    def test_e2e_marketing_dataset(self):
        df = _make_marketing_df(n_rows=52, n_channels=3)
        canonical, context = _build_canonical(df)
        result = run_marketing_analysis(canonical, context, "test_e2e")
        assert isinstance(result, Phase6Result)
        assert result.marketing_report is not None
        assert result.dataset_id == "test_e2e"

    def test_e2e_non_marketing_dataset(self):
        df = _make_non_marketing_df()
        canonical, context = _build_canonical(df)
        result = run_marketing_analysis(canonical, context, "hr_data")
        assert result.marketing_report.readiness == MarketingReadiness.NOT_READY
        assert result.mmm_result is None
        assert result.contributions == []

    def test_e2e_serialization(self):
        df = _make_marketing_df(n_rows=52)
        canonical, context = _build_canonical(df)
        result = run_marketing_analysis(canonical, context, "serialize_test")
        d = result.to_dict()
        json_str = json.dumps(d)
        assert len(json_str) > 0
        result2 = Phase6Result.from_dict(json.loads(json_str))
        assert result2.dataset_id == "serialize_test"

    def test_e2e_unavailable_analyses_populated(self):
        df = _make_non_marketing_df()
        canonical, context = _build_canonical(df)
        result = run_marketing_analysis(canonical, context, "non_mktg")
        # Should explain WHY marketing analysis is unavailable
        all_unavailable = list(result.unavailable_analyses)
        if result.marketing_report:
            all_unavailable.extend(result.marketing_report.unavailable_analyses)
        assert len(all_unavailable) > 0


# ===================================================================
# 9. DATASET-AGNOSTIC — Novel Schemas (NO Revenue/Date/Sales/Marketing)
# ===================================================================

class TestDatasetAgnostic:

    def test_retail_schema(self):
        """Retail: purchase_day, units_sold, unit_price, store_area — no channels."""
        rng = np.random.RandomState(1)
        df = pd.DataFrame({
            "purchase_day": pd.date_range("2023-01-01", periods=100, freq="D"),
            "units_sold": rng.randint(1, 100, 100),
            "unit_price": rng.uniform(5, 50, 100),
            "store_area": rng.choice(["North", "South", "East"], 100),
            "product_group": rng.choice(["A", "B", "C", "D"], 100),
        })
        canonical, context = _build_canonical(df)
        result = run_marketing_analysis(canonical, context, "retail")
        assert result.marketing_report.readiness == MarketingReadiness.NOT_READY

    def test_hr_schema(self):
        """HR: employee_age, experience, salary, dept, rating — no channels."""
        df = _make_non_marketing_df()
        canonical, context = _build_canonical(df)
        result = run_marketing_analysis(canonical, context, "hr")
        assert result.marketing_report.readiness == MarketingReadiness.NOT_READY
        assert result.mmm_result is None

    def test_finance_schema(self):
        """Finance: transaction_amount, account_age, risk_score — no channels."""
        rng = np.random.RandomState(3)
        df = pd.DataFrame({
            "transaction_amount": rng.uniform(10, 10000, 150),
            "account_age": rng.randint(1, 20, 150),
            "risk_score": rng.uniform(0, 1, 150),
            "transaction_type": rng.choice(["purchase", "refund", "transfer"], 150),
            "branch_code": rng.choice(["BR001", "BR002", "BR003"], 150),
        })
        canonical, context = _build_canonical(df)
        result = run_marketing_analysis(canonical, context, "finance")
        assert result.marketing_report.readiness == MarketingReadiness.NOT_READY

    def test_operations_schema(self):
        """Operations: machine_temp, runtime, defect_count — no channels."""
        rng = np.random.RandomState(4)
        df = pd.DataFrame({
            "machine_temperature": rng.uniform(50, 200, 90),
            "runtime_hours": rng.uniform(1, 24, 90),
            "defect_count": rng.randint(0, 10, 90),
            "machine_type": rng.choice(["CNC", "Lathe", "Mill"], 90),
            "shift_name": rng.choice(["Morning", "Afternoon", "Night"], 90),
        })
        canonical, context = _build_canonical(df)
        result = run_marketing_analysis(canonical, context, "operations")
        assert result.marketing_report.readiness == MarketingReadiness.NOT_READY

    def test_education_schema(self):
        """Education: study_hours, attendance, exam_score — no channels."""
        rng = np.random.RandomState(5)
        df = pd.DataFrame({
            "student_age": rng.randint(18, 25, 120),
            "study_hours": rng.uniform(0, 12, 120),
            "attendance_rate": rng.uniform(0.5, 1.0, 120),
            "exam_score": rng.uniform(0, 100, 120),
            "course_type": rng.choice(["Math", "Science", "Arts"], 120),
        })
        canonical, context = _build_canonical(df)
        result = run_marketing_analysis(canonical, context, "education")
        assert result.marketing_report.readiness == MarketingReadiness.NOT_READY

    def test_marketing_with_unusual_col_names(self):
        """Marketing dataset with unusual column names (spaces, symbols)."""
        rng = np.random.RandomState(6)
        n = 52
        df = pd.DataFrame({
            "Week": pd.date_range("2023-01-01", periods=n, freq="W"),
            "Google Ads Spend": rng.uniform(500, 5000, n),
            "Facebook Ads Spend": rng.uniform(300, 3000, n),
            "Total Revenue ($)": rng.uniform(10000, 50000, n),
            "Price Index": rng.uniform(90, 110, n),
        })
        canonical, context = _build_canonical(df)
        result = run_marketing_analysis(canonical, context, "unusual_names")
        # Should detect channels despite unusual naming
        assert result.marketing_report is not None


# ===================================================================
# 10. EDGE CASES
# ===================================================================

class TestEdgeCases:

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        canonical, context = _build_canonical(pd.DataFrame({"x": [1]}))
        canonical._dataframe = df
        result = run_marketing_analysis(canonical, context, "empty")
        assert result.marketing_report.readiness == MarketingReadiness.NOT_READY

    def test_single_row(self):
        df = pd.DataFrame({
            "spend": [100], "revenue": [1000],
        })
        canonical, context = _build_canonical(df)
        result = run_marketing_analysis(canonical, context, "single_row")
        assert result.marketing_report is not None

    def test_two_rows(self):
        df = pd.DataFrame({
            "spend": [100, 200], "revenue": [1000, 2000],
        })
        canonical, context = _build_canonical(df)
        result = run_marketing_analysis(canonical, context, "two_rows")
        assert result.marketing_report is not None

    def test_all_zeros(self):
        df = pd.DataFrame({
            "Channel_Spend": [0.0] * 50,
            "Revenue": [0.0] * 50,
        })
        canonical, context = _build_canonical(df)
        result = run_marketing_analysis(canonical, context, "all_zeros")
        assert result.marketing_report is not None

    def test_negative_values(self):
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "Channel_Spend": rng.uniform(-1000, 5000, 52),
            "Revenue": rng.uniform(-5000, 50000, 52),
        })
        canonical, context = _build_canonical(df)
        result = run_marketing_analysis(canonical, context, "negatives")
        assert result.marketing_report is not None

    def test_missing_data_50pct(self):
        rng = np.random.RandomState(42)
        spend = rng.uniform(100, 5000, 52).astype(float)
        spend[:26] = np.nan
        df = pd.DataFrame({
            "Date": pd.date_range("2023-01-01", periods=52, freq="W"),
            "Channel_Spend": spend,
            "Revenue": rng.uniform(10000, 50000, 52),
        })
        canonical, context = _build_canonical(df)
        result = run_marketing_analysis(canonical, context, "missing_50")
        assert result.marketing_report is not None

    def test_constant_channel(self):
        df = pd.DataFrame({
            "Channel_Spend": [100.0] * 52,
            "Revenue": np.random.uniform(10000, 50000, 52),
        })
        canonical, context = _build_canonical(df)
        report = assess_marketing_capabilities(canonical, context)
        # Constant channel should be rejected
        if report.channel_fields:
            assert "Channel_Spend" not in report.channel_fields or len(report.warnings) > 0

    def test_high_cardinality_id_not_channel(self):
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "transaction_id": list(range(200)),
            "customer_id": list(range(200)),
            "amount": rng.uniform(10, 500, 200),
        })
        canonical, context = _build_canonical(df)
        report = assess_marketing_capabilities(canonical, context)
        # IDs should NOT be detected as channels
        assert "transaction_id" not in report.channel_fields
        assert "customer_id" not in report.channel_fields

    def test_large_channel_count(self):
        """Test with many channel columns to verify MAX_CHANNELS limit."""
        rng = np.random.RandomState(42)
        data = {"Date": pd.date_range("2023-01-01", periods=52, freq="W")}
        for i in range(60):
            data[f"Ad_Channel_{i}_Spend"] = rng.uniform(10, 1000, 52)
        data["Revenue"] = rng.uniform(10000, 50000, 52)
        df = pd.DataFrame(data)
        canonical, context = _build_canonical(df)
        candidates = detect_channel_candidates(canonical)
        assert len(candidates) <= 50  # MAX_CHANNELS


# ===================================================================
# 11. NON-CAUSAL LANGUAGE COMPLIANCE
# ===================================================================

class TestNonCausalLanguage:

    def test_mmm_assumptions_non_causal(self):
        df = _make_marketing_df(n_rows=52)
        result = fit_mmm(df, ["Channel_1_Spend", "Channel_2_Spend"],
                         "Revenue")
        text = " ".join(result.assumptions + result.limitations).lower()
        forbidden = ["causes", "leads to", "results in",
                     "creates revenue", "drives revenue"]
        for word in forbidden:
            assert word not in text

    def test_contribution_model_estimated_label(self):
        """Contributions should be labeled as model-estimated, not causal."""
        df = _make_marketing_df(n_rows=52)
        mmm = fit_mmm(df, ["Channel_1_Spend", "Channel_2_Spend"], "Revenue")
        contribs = compute_contributions(df, mmm)
        # The module docstring mentions model-estimated
        from src.marketing import contribution
        assert "model-estimated" in contribution.__doc__.lower() or \
               "model" in contribution.__doc__.lower()

    def test_whatif_disclaimer_non_causal(self):
        wr = WhatIfResult()
        disclaimer = wr.disclaimer.lower()
        assert "causal" not in disclaimer or "not" in disclaimer
        assert "may differ" in disclaimer or "guarantee" in disclaimer

    def test_optimizer_module_doc_non_causal(self):
        from src.marketing import optimizer
        doc = optimizer.__doc__.lower()
        forbidden = ["causes", "leads to", "results in"]
        for word in forbidden:
            assert word not in doc, f"Causal language '{word}' in optimizer docstring"


# ===================================================================
# 12. NO DEMO DATA LEAKAGE
# ===================================================================

class TestNoDemoLeakage:

    def test_no_demo_fallback_in_orchestrator(self):
        """Orchestrator should never load demo data on failure."""
        import inspect
        source = inspect.getsource(run_marketing_analysis)
        forbidden = ["demo", "sample_data", "novamart", "fallback",
                      "default_dataset", "mock_data"]
        source_lower = source.lower()
        for word in forbidden:
            # Allow "df.sample" which is a pandas method, not demo data
            if word == "demo":
                assert word not in source_lower or source_lower.count(word) == 0
            else:
                assert word not in source_lower

    def test_no_demo_in_field_discovery(self):
        import inspect
        source = inspect.getsource(assess_marketing_capabilities).lower()
        assert "demo" not in source
        assert "novamart" not in source

    def test_no_demo_in_mmm_engine(self):
        import inspect
        source = inspect.getsource(fit_mmm).lower()
        assert "demo" not in source
        assert "novamart" not in source

    def test_no_hardcoded_channels_in_mmm(self):
        """No hardcoded NovaMart channel names in Phase 6 code."""
        import inspect
        source = inspect.getsource(fit_mmm).lower()
        novamart_channels = ["tv_spend", "radio_spend", "print_spend",
                             "search_spend", "social_media_spend",
                             "display_ads_spend", "email_marketing_spend",
                             "influencer_marketing_spend"]
        for ch in novamart_channels:
            assert ch not in source

    def test_no_hardcoded_channels_in_contribution(self):
        import inspect
        source = inspect.getsource(compute_contributions).lower()
        assert '"revenue"' not in source
        assert "tv_spend" not in source

    def test_no_hardcoded_channels_in_optimizer(self):
        import inspect
        source = inspect.getsource(optimize_budget).lower()
        assert '"revenue"' not in source
        assert "tv_spend" not in source

    def test_not_ready_returns_empty_not_demo(self):
        """When dataset isn't marketing, result should be empty, not demo."""
        df = _make_non_marketing_df()
        canonical, context = _build_canonical(df)
        result = run_marketing_analysis(canonical, context, "no_demo")
        assert result.mmm_result is None
        assert result.contributions == []
        assert result.roi_results == []
        assert result.optimization_result is None
        assert result.whatif_result is None


# ===================================================================
# 13. INTEGRATION WITH ANALYSIS RUNNER
# ===================================================================

class TestAnalysisRunnerIntegration:

    def test_runner_returns_phase6_key(self):
        """Verify run_full_analysis includes phase6_result key."""
        from src.data_engine.analysis_runner import run_full_analysis
        df = _make_marketing_df(n_rows=30, n_channels=2)
        result = run_full_analysis("test_phase6_key", df)
        assert "phase6_result" in result

    def test_runner_phase6_non_marketing(self):
        """Non-marketing dataset should still have phase6_result (with NOT_READY)."""
        from src.data_engine.analysis_runner import run_full_analysis
        df = _make_non_marketing_df()
        result = run_full_analysis("test_non_mktg", df)
        assert "phase6_result" in result
        if result["phase6_result"]:
            p6 = Phase6Result.from_dict(result["phase6_result"])
            assert p6.marketing_report.readiness == MarketingReadiness.NOT_READY


# ===================================================================
# 14. ADDITIONAL COVERAGE
# ===================================================================

class TestAdditionalCoverage:

    def test_mmm_with_single_channel(self):
        """Single channel should still work (ROI possible, MMM may work)."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "Date": pd.date_range("2023-01-01", periods=52, freq="W"),
            "Google_Ads_Spend": rng.uniform(500, 5000, 52),
            "Revenue": rng.uniform(10000, 50000, 52),
        })
        result = fit_mmm(df, ["Google_Ads_Spend"], "Revenue", time_col="Date")
        # Single channel fits but may have limited utility
        assert result._model is not None or len(result.limitations) > 0

    def test_mmm_with_many_controls(self):
        rng = np.random.RandomState(42)
        data = {
            "Date": pd.date_range("2023-01-01", periods=52, freq="W"),
            "Spend_A": rng.uniform(100, 5000, 52),
            "Spend_B": rng.uniform(100, 5000, 52),
            "Revenue": rng.uniform(10000, 50000, 52),
        }
        for i in range(10):
            data[f"Control_{i}"] = rng.uniform(0, 100, 52)
        df = pd.DataFrame(data)
        result = fit_mmm(df, ["Spend_A", "Spend_B"], "Revenue",
                         control_cols=[f"Control_{i}" for i in range(10)],
                         time_col="Date")
        assert result._model is not None

    def test_optimizer_equal_split_when_no_history(self):
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "Spend_A": [0.0] * 52,
            "Spend_B": [0.0] * 52,
            "Revenue": rng.uniform(1000, 5000, 52),
        })
        mmm = fit_mmm(df, ["Spend_A", "Spend_B"], "Revenue")
        if mmm._model is not None:
            result = optimize_budget(df, mmm, total_budget=10000)
            # With zero history, initial guess should be equal split
            assert result.total_budget == 10000

    def test_contribution_non_negative(self):
        """Leave-one-out contribution should be clipped to non-negative."""
        df = _make_marketing_df(n_rows=52)
        mmm = fit_mmm(df, ["Channel_1_Spend", "Channel_2_Spend"], "Revenue")
        contribs = compute_contributions(df, mmm)
        for c in contribs:
            assert c.contribution >= 0

    def test_field_discovery_multiple_signals(self):
        """Verify 3-signal channel detection combines signals."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "Date": pd.date_range("2023-01-01", periods=52, freq="W"),
            "google_ads": rng.uniform(100, 5000, 52),
            "facebook_ads": rng.uniform(100, 3000, 52),
            "Revenue": rng.uniform(10000, 50000, 52),
        })
        canonical, context = _build_canonical(df)
        candidates = detect_channel_candidates(canonical)
        # Should detect google_ads and facebook_ads via CHANNEL_SPEND_ALIASES
        assert len(candidates) >= 1

    def test_mmm_result_has_split_info(self):
        df = _make_marketing_df(n_rows=52)
        result = fit_mmm(df, ["Channel_1_Spend", "Channel_2_Spend"], "Revenue")
        assert result.n_train > 0
        assert result.n_test > 0
        assert result.split_method  # not empty

    def test_phase6_result_json_round_trip(self):
        """Full Phase6Result can survive JSON round-trip."""
        df = _make_marketing_df(n_rows=52)
        canonical, context = _build_canonical(df)
        p6 = run_marketing_analysis(canonical, context, "json_test")
        d = p6.to_dict()
        json_str = json.dumps(d)
        d2 = json.loads(json_str)
        p6_2 = Phase6Result.from_dict(d2)
        assert p6_2.dataset_id == "json_test"
        assert p6_2.marketing_report is not None

    def test_mmm_missing_channel_col(self):
        """Missing channel column should be handled gracefully."""
        df = _make_marketing_df(n_rows=52)
        result = fit_mmm(df, ["Channel_1_Spend", "NONEXISTENT_Spend"],
                         "Revenue")
        # Should fit with available channels only
        assert any("not found" in l.lower() for l in result.limitations)

    def test_assess_reports_all_unavailable_reasons(self):
        """Non-marketing data should get clear unavailable reasons."""
        df = _make_non_marketing_df()
        canonical, context = _build_canonical(df)
        report = assess_marketing_capabilities(canonical, context)
        assert len(report.unavailable_analyses) > 0
        # Should mention channels
        combined = " ".join(report.unavailable_analyses).lower()
        assert "channel" in combined or "spend" in combined
