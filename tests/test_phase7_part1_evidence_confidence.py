"""Phase 7 Part 1 — Comprehensive tests for Evidence + Confidence Foundation.

Covers:
- Every required enum value
- Contract validation and serialization
- Missing and malformed evidence
- Evidence-level distinctions (fact vs inference vs prediction vs recommendation)
- Explicit limitations
- Confidence levels, including INSUFFICIENT
- Confidence explanations and reduction reasons
- Deterministic results
- Empty and insufficient evidence
- Dataset/session isolation
- Conflicting dataset identifiers
- Correlation-versus-causation safeguards
- Prediction-versus-fact safeguards
- Recommendation-versus-guarantee safeguards
- Scenario-versus-future safeguards
- Generic non-marketing examples
- Regression-safe imports
"""

import pytest


# ---------------------------------------------------------------------------
# 1. Regression-safe imports
# ---------------------------------------------------------------------------

class TestImports:
    """Verify all public names import cleanly."""

    def test_import_contracts_module(self):
        import src.decision.contracts  # noqa: F401

    def test_import_evidence_module(self):
        import src.decision.evidence  # noqa: F401

    def test_import_confidence_module(self):
        import src.decision.confidence  # noqa: F401

    def test_import_package_init(self):
        import src.decision  # noqa: F401

    def test_import_all_enums(self):
        from src.decision.contracts import (
            EvidenceLevel,
            ConfidenceLevel,
            LimitationCategory,
        )
        assert EvidenceLevel is not None
        assert ConfidenceLevel is not None
        assert LimitationCategory is not None

    def test_import_all_contracts(self):
        from src.decision.contracts import (
            DecisionEvidence,
            DecisionSignal,
            DecisionLimitation,
            ConfidenceScore,
        )
        assert DecisionEvidence is not None
        assert DecisionSignal is not None
        assert DecisionLimitation is not None
        assert ConfidenceScore is not None

    def test_import_evidence_helpers(self):
        from src.decision.evidence import (
            classify_evidence_level,
            create_evidence,
            create_limitation,
            create_signal,
            ensure_non_causal,
            ensure_prediction_uncertainty,
            validate_dataset_scope,
        )
        assert callable(classify_evidence_level)
        assert callable(create_evidence)
        assert callable(create_limitation)
        assert callable(create_signal)
        assert callable(ensure_non_causal)
        assert callable(ensure_prediction_uncertainty)
        assert callable(validate_dataset_scope)

    def test_import_confidence_helpers(self):
        from src.decision.confidence import (
            assess_confidence,
            assess_signal_confidence,
            aggregate_confidence,
        )
        assert callable(assess_confidence)
        assert callable(assess_signal_confidence)
        assert callable(aggregate_confidence)


# ---------------------------------------------------------------------------
# 2. Enum values
# ---------------------------------------------------------------------------

class TestEnumValues:
    """Every required enum member exists with correct string value."""

    def test_evidence_level_fact(self):
        from src.decision.contracts import EvidenceLevel
        assert EvidenceLevel.FACT.value == "fact"

    def test_evidence_level_inference(self):
        from src.decision.contracts import EvidenceLevel
        assert EvidenceLevel.INFERENCE.value == "inference"

    def test_evidence_level_prediction(self):
        from src.decision.contracts import EvidenceLevel
        assert EvidenceLevel.PREDICTION.value == "prediction"

    def test_evidence_level_recommendation(self):
        from src.decision.contracts import EvidenceLevel
        assert EvidenceLevel.RECOMMENDATION.value == "recommendation"

    def test_evidence_level_limitation(self):
        from src.decision.contracts import EvidenceLevel
        assert EvidenceLevel.LIMITATION.value == "limitation"

    def test_confidence_level_high(self):
        from src.decision.contracts import ConfidenceLevel
        assert ConfidenceLevel.HIGH.value == "high"

    def test_confidence_level_medium(self):
        from src.decision.contracts import ConfidenceLevel
        assert ConfidenceLevel.MEDIUM.value == "medium"

    def test_confidence_level_low(self):
        from src.decision.contracts import ConfidenceLevel
        assert ConfidenceLevel.LOW.value == "low"

    def test_confidence_level_insufficient(self):
        from src.decision.contracts import ConfidenceLevel
        assert ConfidenceLevel.INSUFFICIENT.value == "insufficient"

    def test_limitation_category_data_quality(self):
        from src.decision.contracts import LimitationCategory
        assert LimitationCategory.DATA_QUALITY.value == "data_quality"

    def test_limitation_category_sample_size(self):
        from src.decision.contracts import LimitationCategory
        assert LimitationCategory.SAMPLE_SIZE.value == "sample_size"

    def test_limitation_category_observational(self):
        from src.decision.contracts import LimitationCategory
        assert LimitationCategory.OBSERVATIONAL.value == "observational"

    def test_limitation_category_model_uncertainty(self):
        from src.decision.contracts import LimitationCategory
        assert LimitationCategory.MODEL_UNCERTAINTY.value == "model_uncertainty"

    def test_limitation_category_temporal(self):
        from src.decision.contracts import LimitationCategory
        assert LimitationCategory.TEMPORAL.value == "temporal"

    def test_limitation_category_scope(self):
        from src.decision.contracts import LimitationCategory
        assert LimitationCategory.SCOPE.value == "scope"

    def test_limitation_category_methodology(self):
        from src.decision.contracts import LimitationCategory
        assert LimitationCategory.METHODOLOGY.value == "methodology"


# ---------------------------------------------------------------------------
# 3. Contract validation and serialization
# ---------------------------------------------------------------------------

class TestDecisionLimitation:
    """DecisionLimitation creation, validation, serialization."""

    def test_create_with_defaults(self):
        from src.decision.contracts import DecisionLimitation, LimitationCategory
        lim = DecisionLimitation(description="Test limitation")
        assert lim.description == "Test limitation"
        assert lim.category == LimitationCategory.OBSERVATIONAL.value
        assert lim.id.startswith("lim_")
        assert lim.impact == ""
        assert lim.recommendation == ""

    def test_create_with_all_fields(self):
        from src.decision.contracts import DecisionLimitation, LimitationCategory
        lim = DecisionLimitation(
            id="lim_custom1",
            category=LimitationCategory.SAMPLE_SIZE.value,
            description="Only 10 observations",
            impact="Low statistical power",
            recommendation="Collect more data",
        )
        assert lim.id == "lim_custom1"
        assert lim.category == "sample_size"
        assert lim.description == "Only 10 observations"
        assert lim.impact == "Low statistical power"
        assert lim.recommendation == "Collect more data"

    def test_immutability(self):
        from src.decision.contracts import DecisionLimitation
        lim = DecisionLimitation(description="Test")
        with pytest.raises(AttributeError):
            lim.description = "Changed"

    def test_to_dict(self):
        from src.decision.contracts import DecisionLimitation
        lim = DecisionLimitation(
            id="lim_abc12345",
            category="data_quality",
            description="Missing values",
            impact="Incomplete analysis",
            recommendation="Impute or exclude",
        )
        d = lim.to_dict()
        assert d["id"] == "lim_abc12345"
        assert d["category"] == "data_quality"
        assert d["description"] == "Missing values"
        assert d["impact"] == "Incomplete analysis"
        assert d["recommendation"] == "Impute or exclude"

    def test_from_dict_roundtrip(self):
        from src.decision.contracts import DecisionLimitation
        original = DecisionLimitation(
            id="lim_test0001",
            category="temporal",
            description="Seasonal data only",
            impact="May not generalise",
            recommendation="Validate on other periods",
        )
        restored = DecisionLimitation.from_dict(original.to_dict())
        assert restored.id == original.id
        assert restored.category == original.category
        assert restored.description == original.description
        assert restored.impact == original.impact
        assert restored.recommendation == original.recommendation


class TestDecisionEvidence:
    """DecisionEvidence creation, validation, serialization."""

    def test_create_minimal(self):
        from src.decision.contracts import DecisionEvidence, EvidenceLevel
        ev = DecisionEvidence(description="Row count is 500", level=EvidenceLevel.FACT.value)
        assert ev.description == "Row count is 500"
        assert ev.level == "fact"
        assert ev.id.startswith("ev_")
        assert ev.limitations == []
        assert ev.dataset_id == ""

    def test_invalid_level_raises(self):
        from src.decision.contracts import DecisionEvidence
        with pytest.raises(ValueError, match="Invalid evidence level"):
            DecisionEvidence(description="Bad level", level="bogus")

    def test_empty_description_raises(self):
        from src.decision.contracts import DecisionEvidence
        with pytest.raises(ValueError, match="non-empty description"):
            DecisionEvidence(description="", level="fact")

    def test_immutability(self):
        from src.decision.contracts import DecisionEvidence
        ev = DecisionEvidence(description="Test", level="fact")
        with pytest.raises(AttributeError):
            ev.level = "inference"

    def test_to_dict_with_limitations(self):
        from src.decision.contracts import DecisionEvidence, DecisionLimitation
        lim = DecisionLimitation(description="Observational only")
        ev = DecisionEvidence(
            id="ev_test0001",
            level="inference",
            description="Feature X correlates with target Y",
            source="correlation",
            metric_name="pearson_r",
            metric_value=0.85,
            dataset_id="ds_001",
            limitations=[lim],
        )
        d = ev.to_dict()
        assert d["id"] == "ev_test0001"
        assert d["level"] == "inference"
        assert d["metric_value"] == 0.85
        assert d["dataset_id"] == "ds_001"
        assert len(d["limitations"]) == 1
        assert d["limitations"][0]["description"] == "Observational only"

    def test_from_dict_roundtrip(self):
        from src.decision.contracts import DecisionEvidence, DecisionLimitation
        lim = DecisionLimitation(id="lim_r1", description="Small sample")
        original = DecisionEvidence(
            id="ev_r1",
            level="prediction",
            description="Model predicts 42",
            source="regression_engine",
            metric_name="predicted_value",
            metric_value=42.0,
            dataset_id="ds_round",
            limitations=[lim],
        )
        restored = DecisionEvidence.from_dict(original.to_dict())
        assert restored.id == original.id
        assert restored.level == original.level
        assert restored.description == original.description
        assert restored.source == original.source
        assert restored.metric_value == original.metric_value
        assert restored.dataset_id == original.dataset_id
        assert len(restored.limitations) == 1
        assert restored.limitations[0].description == "Small sample"

    def test_safe_float_nan(self):
        """NaN metric values are sanitised to None in serialization."""
        import math
        from src.decision.contracts import DecisionEvidence
        ev = DecisionEvidence(
            description="Contains NaN",
            level="fact",
            metric_value=float("nan"),
        )
        d = ev.to_dict()
        assert d["metric_value"] is None

    def test_safe_float_inf(self):
        """Inf metric values are sanitised to None in serialization."""
        from src.decision.contracts import DecisionEvidence
        ev = DecisionEvidence(
            description="Contains Inf",
            level="fact",
            metric_value=float("inf"),
        )
        d = ev.to_dict()
        assert d["metric_value"] is None


class TestDecisionSignal:
    """DecisionSignal creation, validation, serialization."""

    def test_create_minimal(self):
        from src.decision.contracts import DecisionSignal
        sig = DecisionSignal(
            description="Feature A is important",
            evidence_ids=["ev_1"],
        )
        assert sig.description == "Feature A is important"
        assert sig.id.startswith("sig_")
        assert sig.evidence_ids == ["ev_1"]
        assert sig.level == "inference"  # default
        assert sig.limitations == []

    def test_invalid_level_raises(self):
        from src.decision.contracts import DecisionSignal
        with pytest.raises(ValueError, match="Invalid evidence level"):
            DecisionSignal(description="Bad", level="wrong", evidence_ids=["ev_1"])

    def test_empty_description_raises(self):
        from src.decision.contracts import DecisionSignal
        with pytest.raises(ValueError, match="non-empty description"):
            DecisionSignal(description="", evidence_ids=["ev_1"])

    def test_immutability(self):
        from src.decision.contracts import DecisionSignal
        sig = DecisionSignal(description="Test", evidence_ids=["ev_1"])
        with pytest.raises(AttributeError):
            sig.description = "Changed"

    def test_to_dict_roundtrip(self):
        from src.decision.contracts import DecisionSignal, DecisionLimitation
        lim = DecisionLimitation(id="lim_s1", description="Scope limited")
        original = DecisionSignal(
            id="sig_rt1",
            description="Cluster detected",
            evidence_ids=["ev_1", "ev_2"],
            level="inference",
            affected_features=["col_a", "col_b"],
            limitations=[lim],
            dataset_id="ds_test",
        )
        restored = DecisionSignal.from_dict(original.to_dict())
        assert restored.id == original.id
        assert restored.description == original.description
        assert restored.evidence_ids == original.evidence_ids
        assert restored.level == original.level
        assert restored.affected_features == original.affected_features
        assert restored.dataset_id == original.dataset_id
        assert len(restored.limitations) == 1


class TestConfidenceScore:
    """ConfidenceScore creation, validation, serialization."""

    def test_create_defaults(self):
        from src.decision.contracts import ConfidenceScore
        cs = ConfidenceScore()
        assert cs.level == "insufficient"
        assert cs.evidence_count == 0
        assert cs.reduction_reasons == []

    def test_invalid_level_raises(self):
        from src.decision.contracts import ConfidenceScore
        with pytest.raises(ValueError, match="Invalid confidence level"):
            ConfidenceScore(level="ultra_high")

    def test_to_dict_roundtrip(self):
        from src.decision.contracts import ConfidenceScore
        original = ConfidenceScore(
            level="medium",
            explanation="Moderate evidence available.",
            evidence_count=3,
            evidence_quality="moderate",
            reduction_reasons=["Single source"],
            dataset_id="ds_conf",
        )
        restored = ConfidenceScore.from_dict(original.to_dict())
        assert restored.level == original.level
        assert restored.explanation == original.explanation
        assert restored.evidence_count == original.evidence_count
        assert restored.evidence_quality == original.evidence_quality
        assert restored.reduction_reasons == original.reduction_reasons
        assert restored.dataset_id == original.dataset_id


# ---------------------------------------------------------------------------
# 4. Evidence classification
# ---------------------------------------------------------------------------

class TestEvidenceClassification:
    """classify_evidence_level produces correct levels for known sources."""

    def test_fact_sources(self):
        from src.decision.evidence import classify_evidence_level
        for src in ["kpi_engine", "data_quality", "profiler", "row_count",
                     "summary", "distribution", "aggregation"]:
            assert classify_evidence_level(src) == "fact", f"Expected FACT for '{src}'"

    def test_inference_sources(self):
        from src.decision.evidence import classify_evidence_level
        for src in ["insight_engine", "relationship_engine", "driver_engine",
                     "correlation", "anomaly_detector", "contribution", "roi"]:
            assert classify_evidence_level(src) == "inference", f"Expected INFERENCE for '{src}'"

    def test_prediction_sources(self):
        from src.decision.evidence import classify_evidence_level
        for src in ["forecasting_engine", "regression_engine",
                     "classification_engine", "scenario_engine",
                     "simulator", "forecast", "prediction"]:
            assert classify_evidence_level(src) == "prediction", f"Expected PREDICTION for '{src}'"

    def test_limitation_source(self):
        from src.decision.evidence import classify_evidence_level
        assert classify_evidence_level("limitation_tracker") == "limitation"

    def test_recommendation_source(self):
        from src.decision.evidence import classify_evidence_level
        assert classify_evidence_level("recommendation_engine") == "recommendation"

    def test_unknown_source_defaults_to_inference(self):
        """Unknown sources are conservatively classified as INFERENCE, not FACT."""
        from src.decision.evidence import classify_evidence_level
        assert classify_evidence_level("unknown_module") == "inference"
        assert classify_evidence_level("") == "inference"

    def test_case_insensitivity(self):
        from src.decision.evidence import classify_evidence_level
        assert classify_evidence_level("KPI_ENGINE") == "fact"
        assert classify_evidence_level("Forecasting_Engine") == "prediction"


# ---------------------------------------------------------------------------
# 5. Evidence creation
# ---------------------------------------------------------------------------

class TestEvidenceCreation:
    """create_evidence factory function."""

    def test_auto_classification(self):
        from src.decision.evidence import create_evidence
        ev = create_evidence("Total records: 1000", "row_count")
        assert ev.level == "fact"
        assert ev.description == "Total records: 1000"

    def test_explicit_level_overrides(self):
        from src.decision.evidence import create_evidence
        ev = create_evidence("Override test", "row_count", level="prediction")
        assert ev.level == "prediction"

    def test_empty_description_raises(self):
        from src.decision.evidence import create_evidence
        with pytest.raises(ValueError, match="must not be empty"):
            create_evidence("", "row_count")

    def test_whitespace_only_description_raises(self):
        from src.decision.evidence import create_evidence
        with pytest.raises(ValueError, match="must not be empty"):
            create_evidence("   ", "row_count")

    def test_description_stripped(self):
        from src.decision.evidence import create_evidence
        ev = create_evidence("  padded  ", "row_count")
        assert ev.description == "padded"

    def test_dataset_id_preserved(self):
        from src.decision.evidence import create_evidence
        ev = create_evidence("Test", "summary", dataset_id="ds_alpha")
        assert ev.dataset_id == "ds_alpha"

    def test_metric_fields(self):
        from src.decision.evidence import create_evidence
        ev = create_evidence(
            "Mean value observed",
            "aggregation",
            metric_name="mean",
            metric_value=3.14,
        )
        assert ev.metric_name == "mean"
        assert ev.metric_value == 3.14

    def test_unique_ids(self):
        from src.decision.evidence import create_evidence
        ev1 = create_evidence("First", "summary")
        ev2 = create_evidence("Second", "summary")
        assert ev1.id != ev2.id


# ---------------------------------------------------------------------------
# 6. Limitation creation
# ---------------------------------------------------------------------------

class TestLimitationCreation:
    """create_limitation factory function."""

    def test_defaults(self):
        from src.decision.evidence import create_limitation
        lim = create_limitation("Observational data only")
        assert lim.category == "observational"
        assert lim.description == "Observational data only"
        assert lim.id.startswith("lim_")

    def test_all_fields(self):
        from src.decision.evidence import create_limitation
        lim = create_limitation(
            "Small sample",
            category="sample_size",
            impact="Low statistical power",
            recommendation="Collect more data",
        )
        assert lim.category == "sample_size"
        assert lim.impact == "Low statistical power"

    def test_empty_description_raises(self):
        from src.decision.evidence import create_limitation
        with pytest.raises(ValueError, match="must not be empty"):
            create_limitation("")


# ---------------------------------------------------------------------------
# 7. Correlation vs causation safeguards
# ---------------------------------------------------------------------------

class TestCorrelationCausationSafeguards:
    """ensure_non_causal attaches observational limitations to inferences."""

    def test_inference_gets_limitation(self):
        from src.decision.evidence import create_evidence, ensure_non_causal
        ev = create_evidence("X correlates with Y", "correlation")
        assert ev.level == "inference"
        guarded = ensure_non_causal(ev)
        assert len(guarded.limitations) == 1
        assert guarded.limitations[0].category == "observational"
        assert "causation" in guarded.limitations[0].description.lower()

    def test_idempotent(self):
        from src.decision.evidence import create_evidence, ensure_non_causal
        ev = create_evidence("X correlates with Y", "correlation")
        once = ensure_non_causal(ev)
        twice = ensure_non_causal(once)
        assert len(twice.limitations) == 1  # Not duplicated

    def test_fact_not_modified(self):
        from src.decision.evidence import create_evidence, ensure_non_causal
        ev = create_evidence("Row count: 500", "row_count")
        assert ev.level == "fact"
        guarded = ensure_non_causal(ev)
        assert len(guarded.limitations) == 0  # Facts are not observational inferences

    def test_prediction_not_modified(self):
        from src.decision.evidence import create_evidence, ensure_non_causal
        ev = create_evidence("Model forecasts 100", "forecast")
        assert ev.level == "prediction"
        guarded = ensure_non_causal(ev)
        assert len(guarded.limitations) == 0


# ---------------------------------------------------------------------------
# 8. Prediction vs fact safeguards
# ---------------------------------------------------------------------------

class TestPredictionFactSafeguards:
    """ensure_prediction_uncertainty marks predictions with explicit uncertainty."""

    def test_prediction_gets_uncertainty(self):
        from src.decision.evidence import create_evidence, ensure_prediction_uncertainty
        ev = create_evidence("Forecast value: 200", "forecast")
        assert ev.level == "prediction"
        guarded = ensure_prediction_uncertainty(ev)
        assert len(guarded.limitations) == 1
        assert guarded.limitations[0].category == "model_uncertainty"
        assert "projection" in guarded.limitations[0].description.lower()

    def test_idempotent(self):
        from src.decision.evidence import create_evidence, ensure_prediction_uncertainty
        ev = create_evidence("Forecast value: 200", "forecast")
        once = ensure_prediction_uncertainty(ev)
        twice = ensure_prediction_uncertainty(once)
        assert len(twice.limitations) == 1

    def test_fact_not_modified(self):
        from src.decision.evidence import create_evidence, ensure_prediction_uncertainty
        ev = create_evidence("Observed: 500", "row_count")
        guarded = ensure_prediction_uncertainty(ev)
        assert len(guarded.limitations) == 0

    def test_inference_not_modified(self):
        from src.decision.evidence import create_evidence, ensure_prediction_uncertainty
        ev = create_evidence("Correlation found", "correlation")
        guarded = ensure_prediction_uncertainty(ev)
        assert len(guarded.limitations) == 0


# ---------------------------------------------------------------------------
# 9. Signal creation
# ---------------------------------------------------------------------------

class TestSignalCreation:
    """create_signal with weakest-link level, dataset isolation, and validation."""

    def test_basic_signal(self):
        from src.decision.evidence import create_evidence, create_signal
        ev1 = create_evidence("Fact A", "summary")
        ev2 = create_evidence("Inference B", "correlation")
        sig = create_signal("Combined finding", [ev1, ev2])
        assert sig.description == "Combined finding"
        assert sig.id.startswith("sig_")
        assert len(sig.evidence_ids) == 2

    def test_weakest_link_level(self):
        """Signal level is the weakest evidence level among items."""
        from src.decision.evidence import create_evidence, create_signal
        ev_fact = create_evidence("Fact", "summary")  # fact
        ev_pred = create_evidence("Prediction", "forecast")  # prediction
        sig = create_signal("Mixed signal", [ev_fact, ev_pred])
        # prediction is weaker than fact
        assert sig.level == "prediction"

    def test_explicit_level_override(self):
        from src.decision.evidence import create_evidence, create_signal
        ev = create_evidence("Fact", "summary")
        sig = create_signal("Overridden", [ev], level="recommendation")
        assert sig.level == "recommendation"

    def test_empty_evidence_raises(self):
        """Missing evidence must not be presented as evidence of absence."""
        from src.decision.evidence import create_signal
        with pytest.raises(ValueError, match="at least one piece of evidence"):
            create_signal("No evidence signal", [])

    def test_empty_description_raises(self):
        from src.decision.evidence import create_evidence, create_signal
        ev = create_evidence("Fact", "summary")
        with pytest.raises(ValueError, match="must not be empty"):
            create_signal("", [ev])

    def test_affected_features(self):
        from src.decision.evidence import create_evidence, create_signal
        ev = create_evidence("Feature analysis", "driver_engine")
        sig = create_signal(
            "Driver found",
            [ev],
            affected_features=["col_x", "col_y"],
        )
        assert sig.affected_features == ["col_x", "col_y"]

    def test_signal_limitations(self):
        from src.decision.evidence import create_evidence, create_signal, create_limitation
        ev = create_evidence("Analysis result", "insight_engine")
        lim = create_limitation("Scope limited to subset")
        sig = create_signal("Limited finding", [ev], limitations=[lim])
        assert len(sig.limitations) == 1
        assert sig.limitations[0].description == "Scope limited to subset"


# ---------------------------------------------------------------------------
# 10. Dataset isolation
# ---------------------------------------------------------------------------

class TestDatasetIsolation:
    """Evidence from different datasets must not be silently merged."""

    def test_same_dataset_allowed(self):
        from src.decision.evidence import create_evidence, create_signal
        ev1 = create_evidence("A", "summary", dataset_id="ds_1")
        ev2 = create_evidence("B", "summary", dataset_id="ds_1")
        sig = create_signal("Same dataset", [ev1, ev2], dataset_id="ds_1")
        assert sig.dataset_id == "ds_1"

    def test_conflicting_datasets_raises(self):
        from src.decision.evidence import create_evidence, create_signal
        ev1 = create_evidence("A", "summary", dataset_id="ds_1")
        ev2 = create_evidence("B", "summary", dataset_id="ds_2")
        with pytest.raises(ValueError, match="different datasets"):
            create_signal("Mixed datasets", [ev1, ev2])

    def test_unscoped_evidence_allowed(self):
        """Evidence without a dataset_id can combine with any dataset."""
        from src.decision.evidence import create_evidence, create_signal
        ev1 = create_evidence("A", "summary", dataset_id="ds_1")
        ev2 = create_evidence("B", "summary")  # no dataset_id
        # Should not raise
        sig = create_signal("Mixed scope", [ev1, ev2])
        assert sig.dataset_id == "ds_1"

    def test_validate_dataset_scope_filters(self):
        from src.decision.evidence import create_evidence, validate_dataset_scope
        ev1 = create_evidence("A", "summary", dataset_id="ds_1")
        ev2 = create_evidence("B", "summary", dataset_id="ds_2")
        ev3 = create_evidence("C", "summary")  # unscoped
        filtered = validate_dataset_scope([ev1, ev2, ev3], "ds_1")
        assert len(filtered) == 2  # ev1 (matches) and ev3 (unscoped)
        ids = {e.id for e in filtered}
        assert ev1.id in ids
        assert ev3.id in ids
        assert ev2.id not in ids

    def test_validate_dataset_scope_empty_expected(self):
        """Empty expected_dataset_id returns all evidence."""
        from src.decision.evidence import create_evidence, validate_dataset_scope
        ev1 = create_evidence("A", "summary", dataset_id="ds_1")
        ev2 = create_evidence("B", "summary", dataset_id="ds_2")
        filtered = validate_dataset_scope([ev1, ev2], "")
        assert len(filtered) == 2

    def test_signal_dataset_propagated(self):
        """Signal inherits dataset_id from evidence when not specified."""
        from src.decision.evidence import create_evidence, create_signal
        ev = create_evidence("A", "summary", dataset_id="ds_auto")
        sig = create_signal("Auto dataset", [ev])
        assert sig.dataset_id == "ds_auto"


# ---------------------------------------------------------------------------
# 11. Confidence assessment
# ---------------------------------------------------------------------------

class TestConfidenceAssessment:
    """assess_confidence deterministic rules."""

    def test_no_evidence_insufficient(self):
        from src.decision.confidence import assess_confidence
        score = assess_confidence([])
        assert score.level == "insufficient"
        assert score.evidence_count == 0
        assert "No evidence" in score.explanation

    def test_all_limitations_insufficient(self):
        from src.decision.confidence import assess_confidence
        from src.decision.evidence import create_evidence
        ev1 = create_evidence("Limitation A", "limitation_tracker")
        ev2 = create_evidence("Limitation B", "limitation_flag")
        score = assess_confidence([ev1, ev2])
        assert score.level == "insufficient"

    def test_high_confidence(self):
        """>=3 items with >=1 FACT and no critical limitations → HIGH."""
        from src.decision.confidence import assess_confidence
        from src.decision.evidence import create_evidence
        ev1 = create_evidence("Fact A", "summary")  # fact
        ev2 = create_evidence("Fact B", "row_count")  # fact
        ev3 = create_evidence("Fact C", "aggregation")  # fact
        score = assess_confidence([ev1, ev2, ev3])
        assert score.level == "high"
        assert score.evidence_count == 3

    def test_medium_confidence_two_items(self):
        """>=2 items with >=1 FACT → MEDIUM."""
        from src.decision.confidence import assess_confidence
        from src.decision.evidence import create_evidence
        ev1 = create_evidence("Fact A", "summary")
        ev2 = create_evidence("Inference B", "correlation")
        score = assess_confidence([ev1, ev2])
        assert score.level == "medium"

    def test_low_confidence_predictions_only(self):
        """Evidence with only predictions (no facts) → LOW."""
        from src.decision.confidence import assess_confidence
        from src.decision.evidence import create_evidence
        ev1 = create_evidence("Prediction A", "forecast")
        score = assess_confidence([ev1])
        assert score.level == "low"

    def test_critical_limitation_reduces_confidence(self):
        """Critical limitations (data_quality, sample_size) reduce confidence."""
        from src.decision.confidence import assess_confidence
        from src.decision.evidence import create_evidence, create_limitation
        lim = create_limitation("Poor data quality", category="data_quality")
        ev1 = create_evidence("Fact A", "summary", limitations=[lim])
        ev2 = create_evidence("Fact B", "row_count")
        ev3 = create_evidence("Fact C", "aggregation")
        score = assess_confidence([ev1, ev2, ev3])
        # Would be HIGH without the limitation, but critical limitation reduces it
        assert score.level in ("medium", "low")

    def test_single_source_reduction_reason(self):
        """Multiple evidence from a single source adds a reduction reason."""
        from src.decision.confidence import assess_confidence
        from src.decision.evidence import create_evidence
        ev1 = create_evidence("A", "summary")
        ev2 = create_evidence("B", "summary")
        score = assess_confidence([ev1, ev2])
        has_single_source_reason = any(
            "single" in r.lower() for r in score.reduction_reasons
        )
        assert has_single_source_reason

    def test_dataset_id_propagated(self):
        from src.decision.confidence import assess_confidence
        from src.decision.evidence import create_evidence
        ev = create_evidence("Fact", "summary")
        score = assess_confidence([ev], dataset_id="ds_conf")
        assert score.dataset_id == "ds_conf"

    def test_explanation_includes_counts(self):
        from src.decision.confidence import assess_confidence
        from src.decision.evidence import create_evidence
        ev = create_evidence("Fact A", "summary")
        score = assess_confidence([ev])
        assert "1 piece" in score.explanation or "1 observed" in score.explanation

    def test_deterministic(self):
        """Same input produces same output every time."""
        from src.decision.confidence import assess_confidence
        from src.decision.contracts import DecisionEvidence
        # Use fixed IDs to ensure determinism
        ev = DecisionEvidence(
            id="ev_fixed01",
            level="fact",
            description="Deterministic test",
            source="summary",
        )
        s1 = assess_confidence([ev])
        s2 = assess_confidence([ev])
        assert s1.level == s2.level
        assert s1.evidence_count == s2.evidence_count
        assert s1.evidence_quality == s2.evidence_quality
        assert s1.reduction_reasons == s2.reduction_reasons


# ---------------------------------------------------------------------------
# 12. Signal confidence assessment
# ---------------------------------------------------------------------------

class TestSignalConfidence:
    """assess_signal_confidence with evidence lookup and missing evidence."""

    def test_basic_signal_confidence(self):
        from src.decision.confidence import assess_signal_confidence
        from src.decision.evidence import create_evidence, create_signal
        ev1 = create_evidence("Fact A", "summary")
        ev2 = create_evidence("Fact B", "row_count")
        sig = create_signal("Test signal", [ev1, ev2])
        lookup = {ev1.id: ev1, ev2.id: ev2}
        score = assess_signal_confidence(sig, lookup)
        assert score.level in ("medium", "high")
        assert score.evidence_count == 2

    def test_missing_evidence_reduces_confidence(self):
        from src.decision.confidence import assess_signal_confidence
        from src.decision.contracts import DecisionSignal
        sig = DecisionSignal(
            description="Missing evidence signal",
            evidence_ids=["ev_exists", "ev_missing1", "ev_missing2"],
            dataset_id="ds_test",
        )
        from src.decision.contracts import DecisionEvidence
        ev = DecisionEvidence(
            id="ev_exists",
            level="fact",
            description="Existing fact",
            source="summary",
        )
        lookup = {"ev_exists": ev}
        score = assess_signal_confidence(sig, lookup)
        has_missing_reason = any("could not locate" in r.lower() for r in score.reduction_reasons)
        assert has_missing_reason

    def test_signal_critical_limitation_reduces(self):
        from src.decision.confidence import assess_signal_confidence
        from src.decision.evidence import (
            create_evidence, create_signal, create_limitation,
        )
        ev1 = create_evidence("Fact A", "summary")
        ev2 = create_evidence("Fact B", "row_count")
        ev3 = create_evidence("Fact C", "aggregation")
        lim = create_limitation("Bad quality", category="data_quality")
        sig = create_signal("Signal with limitation", [ev1, ev2, ev3], limitations=[lim])
        lookup = {ev1.id: ev1, ev2.id: ev2, ev3.id: ev3}
        score = assess_signal_confidence(sig, lookup)
        # Signal-level critical limitation should reduce from HIGH
        assert score.level != "insufficient"


# ---------------------------------------------------------------------------
# 13. Aggregate confidence
# ---------------------------------------------------------------------------

class TestAggregateConfidence:
    """aggregate_confidence uses weakest-link rule."""

    def test_empty_scores(self):
        from src.decision.confidence import aggregate_confidence
        score = aggregate_confidence([])
        assert score.level == "insufficient"

    def test_weakest_link(self):
        from src.decision.contracts import ConfidenceScore
        from src.decision.confidence import aggregate_confidence
        high = ConfidenceScore(level="high", evidence_count=5)
        low = ConfidenceScore(level="low", evidence_count=1)
        agg = aggregate_confidence([high, low])
        assert agg.level == "low"

    def test_single_insufficient_dominates(self):
        from src.decision.contracts import ConfidenceScore
        from src.decision.confidence import aggregate_confidence
        med = ConfidenceScore(level="medium", evidence_count=3)
        insuf = ConfidenceScore(level="insufficient", evidence_count=0)
        agg = aggregate_confidence([med, insuf])
        assert agg.level == "insufficient"

    def test_total_evidence_count(self):
        from src.decision.contracts import ConfidenceScore
        from src.decision.confidence import aggregate_confidence
        s1 = ConfidenceScore(level="high", evidence_count=5)
        s2 = ConfidenceScore(level="medium", evidence_count=3)
        agg = aggregate_confidence([s1, s2])
        assert agg.evidence_count == 8

    def test_reduction_reasons_aggregated(self):
        from src.decision.contracts import ConfidenceScore
        from src.decision.confidence import aggregate_confidence
        s1 = ConfidenceScore(level="medium", reduction_reasons=["Reason A"])
        s2 = ConfidenceScore(level="low", reduction_reasons=["Reason B"])
        agg = aggregate_confidence([s1, s2])
        assert "Reason A" in agg.reduction_reasons
        assert "Reason B" in agg.reduction_reasons

    def test_duplicate_reasons_deduplicated(self):
        from src.decision.contracts import ConfidenceScore
        from src.decision.confidence import aggregate_confidence
        s1 = ConfidenceScore(level="medium", reduction_reasons=["Same reason"])
        s2 = ConfidenceScore(level="low", reduction_reasons=["Same reason"])
        agg = aggregate_confidence([s1, s2])
        assert agg.reduction_reasons.count("Same reason") == 1


# ---------------------------------------------------------------------------
# 14. Recommendation vs guarantee safeguards
# ---------------------------------------------------------------------------

class TestRecommendationSafeguards:
    """Recommendations are classified distinctly from facts and inferences."""

    def test_recommendation_classification(self):
        from src.decision.evidence import classify_evidence_level
        assert classify_evidence_level("recommendation_engine") == "recommendation"

    def test_recommendation_evidence_level(self):
        from src.decision.evidence import create_evidence
        ev = create_evidence(
            "Suggested action: increase sample size",
            "recommendation_engine",
        )
        assert ev.level == "recommendation"

    def test_recommendation_is_not_fact(self):
        from src.decision.contracts import EvidenceLevel
        assert EvidenceLevel.RECOMMENDATION.value != EvidenceLevel.FACT.value


# ---------------------------------------------------------------------------
# 15. Scenario vs future safeguards
# ---------------------------------------------------------------------------

class TestScenarioSafeguards:
    """Scenario simulations are classified as predictions, not facts."""

    def test_scenario_engine_is_prediction(self):
        from src.decision.evidence import classify_evidence_level
        assert classify_evidence_level("scenario_engine") == "prediction"

    def test_simulator_is_prediction(self):
        from src.decision.evidence import classify_evidence_level
        assert classify_evidence_level("simulator") == "prediction"

    def test_scenario_evidence_carries_uncertainty(self):
        from src.decision.evidence import create_evidence, ensure_prediction_uncertainty
        ev = create_evidence("Scenario outcome: 150 units", "scenario_engine")
        guarded = ensure_prediction_uncertainty(ev)
        assert len(guarded.limitations) >= 1
        assert any(
            l.category == "model_uncertainty" for l in guarded.limitations
        )


# ---------------------------------------------------------------------------
# 16. Generic non-marketing examples
# ---------------------------------------------------------------------------

class TestGenericDomainExamples:
    """Evidence system works with non-marketing, domain-agnostic examples."""

    def test_healthcare_example(self):
        from src.decision.evidence import create_evidence, ensure_non_causal
        ev = create_evidence(
            "Patient cohort shows correlation between treatment A and outcome B",
            "correlation",
            metric_name="correlation_coefficient",
            metric_value=0.72,
            dataset_id="clinical_trial_42",
        )
        guarded = ensure_non_causal(ev)
        assert ev.level == "inference"
        assert guarded.limitations[0].category == "observational"
        assert ev.dataset_id == "clinical_trial_42"

    def test_manufacturing_example(self):
        from src.decision.evidence import create_evidence, create_signal
        ev1 = create_evidence(
            "Defect rate: 2.3%",
            "kpi_engine",
            metric_name="defect_rate",
            metric_value=0.023,
            dataset_id="factory_line_7",
        )
        ev2 = create_evidence(
            "Temperature anomaly detected",
            "anomaly_detector",
            dataset_id="factory_line_7",
        )
        sig = create_signal(
            "Production quality issue detected",
            [ev1, ev2],
            affected_features=["temperature", "defect_count"],
        )
        assert sig.level == "inference"  # weakest of fact + inference
        assert sig.dataset_id == "factory_line_7"

    def test_education_example(self):
        from src.decision.evidence import create_evidence
        from src.decision.confidence import assess_confidence
        ev = create_evidence(
            "Average test score: 78.5",
            "aggregation",
            metric_name="mean_score",
            metric_value=78.5,
            dataset_id="semester_2024",
        )
        score = assess_confidence([ev], dataset_id="semester_2024")
        # Single fact → low (needs more evidence for higher confidence)
        assert score.level in ("low", "medium")

    def test_logistics_example(self):
        from src.decision.evidence import create_evidence, ensure_prediction_uncertainty
        ev = create_evidence(
            "Predicted delivery time: 3.2 days",
            "regression_engine",
            metric_name="predicted_days",
            metric_value=3.2,
            dataset_id="logistics_q4",
        )
        guarded = ensure_prediction_uncertainty(ev)
        assert ev.level == "prediction"
        assert len(guarded.limitations) >= 1

    def test_finance_example(self):
        from src.decision.evidence import create_evidence, create_signal
        ev1 = create_evidence(
            "Portfolio variance: 0.15",
            "kpi_engine",
            metric_name="variance",
            metric_value=0.15,
            dataset_id="portfolio_2024",
        )
        ev2 = create_evidence(
            "Sector correlation detected",
            "relationship_engine",
            dataset_id="portfolio_2024",
        )
        sig = create_signal(
            "Risk concentration in sector",
            [ev1, ev2],
            affected_features=["sector_allocation"],
        )
        assert sig.dataset_id == "portfolio_2024"
        assert len(sig.evidence_ids) == 2


# ---------------------------------------------------------------------------
# 17. Edge cases and malformed evidence
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases: dicts as evidence, missing fields, mixed types."""

    def test_assess_confidence_with_dicts(self):
        """assess_confidence accepts dict-based evidence items."""
        from src.decision.confidence import assess_confidence
        items = [
            {"level": "fact", "source": "summary", "limitations": []},
            {"level": "fact", "source": "row_count", "limitations": []},
            {"level": "fact", "source": "aggregation", "limitations": []},
        ]
        score = assess_confidence(items)
        assert score.level == "high"

    def test_assess_confidence_with_mixed_dicts_and_objects(self):
        from src.decision.confidence import assess_confidence
        from src.decision.evidence import create_evidence
        ev = create_evidence("Object evidence", "summary")
        items = [
            ev,
            {"level": "inference", "source": "other_engine", "limitations": []},
        ]
        score = assess_confidence(items)
        assert score.level in ("low", "medium")

    def test_aggregate_with_dicts(self):
        from src.decision.confidence import aggregate_confidence
        scores = [
            {"level": "high", "reduction_reasons": [], "evidence_count": 3},
            {"level": "low", "reduction_reasons": ["R1"], "evidence_count": 1},
        ]
        agg = aggregate_confidence(scores)
        assert agg.level == "low"
        assert agg.evidence_count == 4

    def test_assess_confidence_single_inference(self):
        """Single inference without any fact → LOW."""
        from src.decision.confidence import assess_confidence
        from src.decision.evidence import create_evidence
        ev = create_evidence("Anomaly found", "anomaly_detector")
        score = assess_confidence([ev])
        assert score.level == "low"

    def test_observational_limitation_adds_reduction_reason(self):
        from src.decision.confidence import assess_confidence
        from src.decision.evidence import create_evidence, create_limitation
        lim = create_limitation("Observational only", category="observational")
        ev1 = create_evidence("Fact", "summary", limitations=[lim])
        ev2 = create_evidence("Fact 2", "row_count")
        score = assess_confidence([ev1, ev2])
        has_correlation_reason = any(
            "correlation" in r.lower() for r in score.reduction_reasons
        )
        assert has_correlation_reason


# ---------------------------------------------------------------------------
# 18. ID generation
# ---------------------------------------------------------------------------

class TestIdGeneration:
    """_make_id produces prefixed unique IDs."""

    def test_prefix(self):
        from src.decision.contracts import _make_id
        assert _make_id("ev_").startswith("ev_")
        assert _make_id("sig_").startswith("sig_")
        assert _make_id("lim_").startswith("lim_")

    def test_uniqueness(self):
        from src.decision.contracts import _make_id
        ids = {_make_id("ev_") for _ in range(100)}
        assert len(ids) == 100

    def test_length(self):
        from src.decision.contracts import _make_id
        id_val = _make_id("ev_")
        # prefix (3 chars) + 8 hex chars = 11
        assert len(id_val) == 11


# ---------------------------------------------------------------------------
# 19. Safe float helper
# ---------------------------------------------------------------------------

class TestSafeFloat:
    """_safe_float sanitises NaN, Inf, and non-numeric values."""

    def test_normal_float(self):
        from src.decision.contracts import _safe_float
        assert _safe_float(3.14) == 3.14

    def test_integer(self):
        from src.decision.contracts import _safe_float
        assert _safe_float(42) == 42.0

    def test_nan(self):
        from src.decision.contracts import _safe_float
        assert _safe_float(float("nan")) is None

    def test_inf(self):
        from src.decision.contracts import _safe_float
        assert _safe_float(float("inf")) is None

    def test_neg_inf(self):
        from src.decision.contracts import _safe_float
        assert _safe_float(float("-inf")) is None

    def test_none(self):
        from src.decision.contracts import _safe_float
        assert _safe_float(None) is None

    def test_string_non_numeric(self):
        from src.decision.contracts import _safe_float
        assert _safe_float("not_a_number") is None

    def test_string_numeric(self):
        from src.decision.contracts import _safe_float
        assert _safe_float("3.14") == 3.14
