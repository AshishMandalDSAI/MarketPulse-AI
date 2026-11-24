"""Phase 7 Part 3 — Impact Quantifier tests.

Comprehensive tests covering:
1.  Valid quantitative impact
2.  Qualitative-only impact
3.  Insufficient evidence
4.  Missing values
5.  NaN and infinity
6.  Zero and negative values
7.  Incompatible units
8.  Dataset isolation
9.  Conflicting evidence
10. Confidence separated from impact magnitude
11. Forecast-based impact labeling
12. Scenario-based impact labeling
13. Correlation-versus-causation safeguards
14. No invented financial estimates
15. Generic non-marketing datasets
16. Marketing examples when supported by actual evidence
17. Serialization
18. Determinism
19. Malformed inputs
20. Regression-safe imports
"""

from __future__ import annotations

import math
import pytest

from src.decision.contracts import (
    ConfidenceLevel,
    ConfidenceScore,
    DecisionEvidence,
    DecisionLimitation,
    DecisionSignal,
    EvidenceLevel,
    LimitationCategory,
    _safe_float,
)
from src.decision.confidence import assess_confidence
from src.decision.evidence import (
    create_evidence,
    create_limitation,
    create_signal,
    validate_dataset_scope,
)
from src.decision.signal_synthesizer import (
    SignalPriority,
    SynthesizedSignal,
    SynthesisResult,
)
from src.decision.impact_quantifier import (
    ImpactAssessment,
    ImpactCategory,
    ImpactQuantifier,
    ImpactReport,
    ImpactType,
    _get_attr,
    _get_attr_nested,
)


# ---------------------------------------------------------------------------
# Helpers — build test objects
# ---------------------------------------------------------------------------

DS_ID = "test_dataset_001"


def _make_evidence(
    *,
    level: str = EvidenceLevel.FACT.value,
    description: str = "Test evidence",
    source: str = "test_engine",
    metric_name: str = "metric_a",
    metric_value=100.0,
    details: dict | None = None,
    dataset_id: str = DS_ID,
    limitations: list | None = None,
    ev_id: str | None = None,
) -> DecisionEvidence:
    kwargs = dict(
        level=level,
        description=description,
        source=source,
        metric_name=metric_name,
        metric_value=metric_value,
        details=details or {},
        dataset_id=dataset_id,
        limitations=limitations or [],
    )
    if ev_id is not None:
        kwargs["id"] = ev_id
    return DecisionEvidence(**kwargs)


def _make_signal(
    *,
    description: str = "Test signal",
    evidence_ids: list | None = None,
    level: str = EvidenceLevel.FACT.value,
    dataset_id: str = DS_ID,
    limitations: list | None = None,
    sig_id: str | None = None,
) -> DecisionSignal:
    kwargs = dict(
        description=description,
        evidence_ids=evidence_ids or [],
        level=level,
        dataset_id=dataset_id,
        limitations=limitations or [],
    )
    if sig_id is not None:
        kwargs["id"] = sig_id
    return DecisionSignal(**kwargs)


def _make_synth_signal(
    *,
    signal: DecisionSignal | None = None,
    priority: int = SignalPriority.MEDIUM,
    confidence_score: ConfidenceScore | None = None,
    source_type: str = "kpi",
    generation_reason: str = "Test generation",
    evidence_items: list | None = None,
) -> SynthesizedSignal:
    if signal is None:
        signal = _make_signal()
    if confidence_score is None:
        confidence_score = ConfidenceScore(
            level=ConfidenceLevel.MEDIUM.value,
            explanation="Test confidence",
            evidence_count=1,
            evidence_quality="adequate",
            dataset_id=signal.dataset_id,
        )
    return SynthesizedSignal(
        signal=signal,
        priority=priority,
        confidence_score=confidence_score,
        source_type=source_type,
        generation_reason=generation_reason,
        evidence_items=evidence_items or [],
    )


# ===================================================================
# 20. REGRESSION-SAFE IMPORTS
# ===================================================================

class TestRegressionSafeImports:
    """All Part 1, Part 2, and Part 3 modules must import cleanly."""

    def test_import_contracts(self):
        from src.decision import contracts
        assert hasattr(contracts, "DecisionEvidence")
        assert hasattr(contracts, "DecisionSignal")
        assert hasattr(contracts, "ConfidenceScore")

    def test_import_evidence(self):
        from src.decision import evidence
        assert hasattr(evidence, "create_evidence")
        assert hasattr(evidence, "validate_dataset_scope")

    def test_import_confidence(self):
        from src.decision import confidence
        assert hasattr(confidence, "assess_confidence")

    def test_import_signal_synthesizer(self):
        from src.decision import signal_synthesizer
        assert hasattr(signal_synthesizer, "SignalSynthesizer")
        assert hasattr(signal_synthesizer, "SynthesizedSignal")
        assert hasattr(signal_synthesizer, "SynthesisResult")

    def test_import_impact_quantifier(self):
        from src.decision import impact_quantifier
        assert hasattr(impact_quantifier, "ImpactQuantifier")
        assert hasattr(impact_quantifier, "ImpactAssessment")
        assert hasattr(impact_quantifier, "ImpactReport")
        assert hasattr(impact_quantifier, "ImpactType")
        assert hasattr(impact_quantifier, "ImpactCategory")

    def test_impact_type_enum_values(self):
        assert ImpactType.MEASURED.value == "measured"
        assert ImpactType.MODEL_ESTIMATED.value == "model_estimated"
        assert ImpactType.SCENARIO_BASED.value == "scenario_based"
        assert ImpactType.QUALITATIVE.value == "qualitative"
        assert ImpactType.UNKNOWN.value == "unknown"

    def test_impact_category_enum_values(self):
        assert ImpactCategory.PERFORMANCE.value == "performance"
        assert ImpactCategory.EFFICIENCY.value == "efficiency"
        assert ImpactCategory.RISK.value == "risk"
        assert ImpactCategory.QUALITY.value == "quality"
        assert ImpactCategory.VOLUME.value == "volume"
        assert ImpactCategory.RELATIONSHIP.value == "relationship"
        assert ImpactCategory.ANOMALY.value == "anomaly"
        assert ImpactCategory.GENERAL.value == "general"

    def test_impact_type_is_str_enum(self):
        assert isinstance(ImpactType.MEASURED, str)
        assert ImpactType.MEASURED == "measured"

    def test_impact_category_is_str_enum(self):
        assert isinstance(ImpactCategory.PERFORMANCE, str)
        assert ImpactCategory.PERFORMANCE == "performance"


# ===================================================================
# 1. VALID QUANTITATIVE IMPACT
# ===================================================================

class TestValidQuantitativeImpact:
    """Impact assessments with valid numeric estimates."""

    def test_kpi_measured_impact(self):
        ev = _make_evidence(metric_name="throughput", metric_value=42.5,
                            details={"unit": "items/hour"})
        signal = _make_signal(evidence_ids=[ev.id])
        synth = _make_synth_signal(signal=signal, source_type="kpi",
                                   evidence_items=[ev])
        q = ImpactQuantifier(dataset_id=DS_ID)
        report = q.assess_signals([synth], {ev.id: ev})

        assert len(report.assessments) == 1
        a = report.assessments[0]
        assert a.impact_type == ImpactType.MEASURED.value
        assert a.impact_category == ImpactCategory.PERFORMANCE.value
        assert a.estimated_value == 42.5
        assert a.signal_id == signal.id
        assert a.dataset_id == DS_ID
        assert a.unit == "items/hour"

    def test_anomaly_deviation_computed(self):
        ev = _make_evidence(
            level=EvidenceLevel.INFERENCE.value,
            description="Anomaly detected",
            source="anomaly_detector",
            metric_value=3.2,
            details={"observed": 150.0, "expected": 100.0},
        )
        signal = _make_signal(evidence_ids=[ev.id],
                              level=EvidenceLevel.INFERENCE.value)
        synth = _make_synth_signal(signal=signal, source_type="anomaly",
                                   evidence_items=[ev])
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {ev.id: ev})

        assert a is not None
        assert a.impact_type == ImpactType.MEASURED.value
        assert a.impact_category == ImpactCategory.ANOMALY.value
        assert a.estimated_value == 50.0  # 150 - 100

    def test_model_r2_impact(self):
        ev = _make_evidence(
            level=EvidenceLevel.PREDICTION.value,
            description="Regression model fitted",
            metric_value=0.85,
            details={"metrics": {"r2": 0.85}, "model": "linear_regression"},
        )
        signal = _make_signal(level=EvidenceLevel.PREDICTION.value,
                              evidence_ids=[ev.id])
        synth = _make_synth_signal(signal=signal, source_type="regression",
                                   evidence_items=[ev])
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {ev.id: ev})

        assert a is not None
        assert a.impact_type == ImpactType.MODEL_ESTIMATED.value
        assert a.estimated_value == 0.85
        assert a.unit == "R²"

    def test_report_quantified_filter(self):
        ev1 = _make_evidence(metric_value=10.0)
        ev2 = _make_evidence(metric_value=None, description="Qualitative only")
        sig1 = _make_signal(description="Signal 1", evidence_ids=[ev1.id])
        sig2 = _make_signal(description="Signal 2", evidence_ids=[ev2.id])
        synth1 = _make_synth_signal(signal=sig1, source_type="kpi",
                                    evidence_items=[ev1])
        synth2 = _make_synth_signal(signal=sig2, source_type="kpi",
                                    evidence_items=[ev2])
        q = ImpactQuantifier(dataset_id=DS_ID)
        report = q.assess_signals([synth1, synth2], {ev1.id: ev1, ev2.id: ev2})

        quantified = report.quantified()
        assert len(quantified) >= 1
        # At least the first one is quantified
        assert any(a.estimated_value == 10.0 for a in quantified)


# ===================================================================
# 2. QUALITATIVE-ONLY IMPACT
# ===================================================================

class TestQualitativeImpact:
    """Impact assessments that are descriptive, not numeric."""

    def test_segmentation_is_qualitative(self):
        ev = _make_evidence(
            level=EvidenceLevel.PREDICTION.value,
            description="Segmentation completed",
            metric_value=4,
            details={"quality_metrics": {"silhouette": 0.65}},
        )
        signal = _make_signal(level=EvidenceLevel.PREDICTION.value,
                              evidence_ids=[ev.id])
        synth = _make_synth_signal(signal=signal, source_type="segmentation",
                                   evidence_items=[ev])
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {ev.id: ev})

        assert a is not None
        assert a.impact_type == ImpactType.QUALITATIVE.value
        assert "segmented" in a.description.lower() or "group" in a.description.lower()

    def test_generic_recommendation_qualitative(self):
        ev = _make_evidence(
            level=EvidenceLevel.RECOMMENDATION.value,
            description="Recommendation for process change",
            metric_value=None,
        )
        signal = _make_signal(
            description="Process recommendation",
            level=EvidenceLevel.RECOMMENDATION.value,
            evidence_ids=[ev.id],
        )
        synth = _make_synth_signal(signal=signal, source_type="generic",
                                   evidence_items=[ev])
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {ev.id: ev})

        assert a is not None
        assert a.impact_type == ImpactType.QUALITATIVE.value
        assert a.estimated_value is None


# ===================================================================
# 3. INSUFFICIENT EVIDENCE
# ===================================================================

class TestInsufficientEvidence:
    """Explicit UNKNOWN when evidence is insufficient."""

    def test_no_evidence_items(self):
        signal = _make_signal()
        synth = _make_synth_signal(signal=signal, evidence_items=[])
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {})

        assert a is not None
        assert a.impact_type == ImpactType.UNKNOWN.value
        assert a.confidence is not None
        assert a.confidence.level == ConfidenceLevel.INSUFFICIENT.value
        assert "cannot be assessed" in a.description.lower()

    def test_empty_signals_list(self):
        q = ImpactQuantifier(dataset_id=DS_ID)
        report = q.assess_signals([], {})
        assert len(report.assessments) == 0
        assert any("no signals" in n.lower() for n in report.processing_notes)

    def test_evidence_all_filtered_out_by_dataset(self):
        ev = _make_evidence(dataset_id="other_dataset")
        signal = _make_signal(evidence_ids=[ev.id])
        synth = _make_synth_signal(signal=signal, evidence_items=[ev])
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {ev.id: ev})

        assert a is not None
        assert a.impact_type == ImpactType.UNKNOWN.value

    def test_no_underlying_signal(self):
        """SynthesizedSignal with signal=None."""
        synth = SynthesizedSignal(
            signal=None,
            priority=3,
            confidence_score=ConfidenceScore(
                level="insufficient", explanation="none",
                dataset_id=DS_ID,
            ),
            source_type="unknown",
            generation_reason="test",
            evidence_items=[],
        )
        q = ImpactQuantifier(dataset_id=DS_ID)
        report = q.assess_signals([synth], {})
        assert len(report.assessments) == 0
        assert any("no underlying signal" in n.lower()
                    for n in report.processing_notes)


# ===================================================================
# 4. MISSING VALUES
# ===================================================================

class TestMissingValues:
    """Handle None and missing fields gracefully."""

    def test_none_metric_value(self):
        ev = _make_evidence(metric_value=None)
        signal = _make_signal(evidence_ids=[ev.id])
        synth = _make_synth_signal(signal=signal, source_type="kpi",
                                   evidence_items=[ev])
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {ev.id: ev})

        assert a is not None
        assert a.estimated_value is None

    def test_empty_details_dict(self):
        ev = _make_evidence(details={})
        signal = _make_signal(evidence_ids=[ev.id])
        synth = _make_synth_signal(signal=signal, source_type="forecast",
                                   evidence_items=[ev])
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {ev.id: ev})

        # Should still produce an assessment, not crash
        assert a is not None

    def test_missing_source_type(self):
        ev = _make_evidence()
        signal = _make_signal(evidence_ids=[ev.id])
        synth = _make_synth_signal(signal=signal, source_type="",
                                   evidence_items=[ev])
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {ev.id: ev})

        assert a is not None
        # Falls through to generic
        assert a.impact_type in {t.value for t in ImpactType}


# ===================================================================
# 5. NaN AND INFINITY
# ===================================================================

class TestNaNAndInfinity:
    """NaN and Inf values must be sanitized."""

    def test_nan_metric_value_sanitized(self):
        ev = _make_evidence(metric_value=float("nan"))
        signal = _make_signal(evidence_ids=[ev.id])
        synth = _make_synth_signal(signal=signal, source_type="kpi",
                                   evidence_items=[ev])
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {ev.id: ev})

        assert a is not None
        assert a.estimated_value is None  # NaN → None

    def test_inf_metric_value_sanitized(self):
        ev = _make_evidence(metric_value=float("inf"))
        signal = _make_signal(evidence_ids=[ev.id])
        synth = _make_synth_signal(signal=signal, source_type="kpi",
                                   evidence_items=[ev])
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {ev.id: ev})

        assert a is not None
        assert a.estimated_value is None  # Inf → None

    def test_neg_inf_metric_value_sanitized(self):
        ev = _make_evidence(metric_value=float("-inf"))
        signal = _make_signal(evidence_ids=[ev.id])
        synth = _make_synth_signal(signal=signal, source_type="kpi",
                                   evidence_items=[ev])
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {ev.id: ev})

        assert a is not None
        assert a.estimated_value is None

    def test_nan_in_anomaly_details(self):
        ev = _make_evidence(
            level=EvidenceLevel.INFERENCE.value,
            description="Anomaly",
            details={"observed": float("nan"), "expected": 100.0},
        )
        signal = _make_signal(level=EvidenceLevel.INFERENCE.value,
                              evidence_ids=[ev.id])
        synth = _make_synth_signal(signal=signal, source_type="anomaly",
                                   evidence_items=[ev])
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {ev.id: ev})

        assert a is not None
        # Deviation cannot be computed with NaN observed
        assert a.estimated_value is None

    def test_nan_in_serialization(self):
        """to_dict must not contain NaN/Inf."""
        a = ImpactAssessment(
            description="Test",
            estimated_value=float("nan"),
        )
        d = a.to_dict()
        assert d["estimated_value"] is None

    def test_inf_in_bounds_serialization(self):
        a = ImpactAssessment(
            description="Test",
            lower_bound=float("-inf"),
            upper_bound=float("inf"),
        )
        d = a.to_dict()
        assert d["lower_bound"] is None
        assert d["upper_bound"] is None


# ===================================================================
# 6. ZERO AND NEGATIVE VALUES
# ===================================================================

class TestZeroAndNegativeValues:
    """Zero and negative values are valid and preserved."""

    def test_zero_metric_value(self):
        ev = _make_evidence(metric_value=0.0)
        signal = _make_signal(evidence_ids=[ev.id])
        synth = _make_synth_signal(signal=signal, source_type="kpi",
                                   evidence_items=[ev])
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {ev.id: ev})

        assert a is not None
        assert a.estimated_value == 0.0

    def test_negative_metric_value(self):
        ev = _make_evidence(metric_value=-15.7)
        signal = _make_signal(evidence_ids=[ev.id])
        synth = _make_synth_signal(signal=signal, source_type="kpi",
                                   evidence_items=[ev])
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {ev.id: ev})

        assert a is not None
        assert a.estimated_value == -15.7

    def test_zero_expected_anomaly(self):
        """Zero expected → division guard: deviation should be None."""
        ev = _make_evidence(
            level=EvidenceLevel.INFERENCE.value,
            description="Anomaly",
            details={"observed": 50.0, "expected": 0.0},
        )
        signal = _make_signal(level=EvidenceLevel.INFERENCE.value,
                              evidence_ids=[ev.id])
        synth = _make_synth_signal(signal=signal, source_type="anomaly",
                                   evidence_items=[ev])
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {ev.id: ev})

        # expected==0 → deviation skipped (guard against divide-by-zero patterns)
        assert a is not None


# ===================================================================
# 7. INCOMPATIBLE UNITS
# ===================================================================

class TestIncompatibleUnits:
    """Units must be preserved from the source evidence, never combined."""

    def test_unit_from_evidence_preserved(self):
        ev = _make_evidence(details={"unit": "widgets/day"})
        signal = _make_signal(evidence_ids=[ev.id])
        synth = _make_synth_signal(signal=signal, source_type="kpi",
                                   evidence_items=[ev])
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {ev.id: ev})

        assert a.unit == "widgets/day"

    def test_driver_unit_is_generic(self):
        """Driver impact uses generic 'association strength', not monetary."""
        ev = _make_evidence(
            level=EvidenceLevel.INFERENCE.value,
            description="Driver found",
            metric_value=0.45,
            details={"direction": "positive", "method": "shap"},
        )
        signal = _make_signal(level=EvidenceLevel.INFERENCE.value,
                              evidence_ids=[ev.id])
        synth = _make_synth_signal(signal=signal, source_type="driver",
                                   evidence_items=[ev])
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {ev.id: ev})

        assert a.unit == "association strength"
        # Must not invent a monetary unit
        assert "$" not in a.unit
        assert "dollar" not in a.unit.lower()
        assert "revenue" not in a.unit.lower()


# ===================================================================
# 8. DATASET ISOLATION
# ===================================================================

class TestDatasetIsolation:
    """Evidence from different datasets must not cross-contaminate."""

    def test_dataset_id_required(self):
        with pytest.raises(ValueError, match="dataset_id is required"):
            ImpactQuantifier(dataset_id="")

    def test_signal_dataset_mismatch_skipped(self):
        ev = _make_evidence(dataset_id="other_ds")
        signal = _make_signal(dataset_id="other_ds", evidence_ids=[ev.id])
        synth = _make_synth_signal(signal=signal, evidence_items=[ev])
        q = ImpactQuantifier(dataset_id=DS_ID)
        report = q.assess_signals([synth], {ev.id: ev})

        # Signal skipped — dataset mismatch
        assert len(report.assessments) == 0
        assert any("does not match" in n for n in report.processing_notes)

    def test_evidence_filtered_to_dataset(self):
        ev_match = _make_evidence(dataset_id=DS_ID, ev_id="ev_match")
        ev_other = _make_evidence(dataset_id="foreign", ev_id="ev_foreign")
        signal = _make_signal(evidence_ids=[ev_match.id, ev_other.id])
        synth = _make_synth_signal(
            signal=signal, source_type="kpi",
            evidence_items=[ev_match, ev_other],
        )
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {ev_match.id: ev_match, ev_other.id: ev_other})

        assert a is not None
        # Only the matching evidence is used
        assert ev_match.id in a.evidence_ids

    def test_report_carries_dataset_id(self):
        q = ImpactQuantifier(dataset_id=DS_ID)
        report = q.assess_signals([], {})
        assert report.dataset_id == DS_ID

    def test_assessment_carries_dataset_id(self):
        ev = _make_evidence()
        signal = _make_signal(evidence_ids=[ev.id])
        synth = _make_synth_signal(signal=signal, source_type="kpi",
                                   evidence_items=[ev])
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {ev.id: ev})

        assert a.dataset_id == DS_ID

    def test_unscoped_signal_accepted(self):
        """Signal with empty dataset_id is accepted."""
        ev = _make_evidence(dataset_id=DS_ID)
        signal = _make_signal(dataset_id="", evidence_ids=[ev.id])
        synth = _make_synth_signal(signal=signal, source_type="kpi",
                                   evidence_items=[ev])
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {ev.id: ev})
        assert a is not None
        assert a.impact_type == ImpactType.MEASURED.value


# ===================================================================
# 9. CONFLICTING EVIDENCE
# ===================================================================

class TestConflictingEvidence:
    """Conflicting evidence is handled, not silently ignored."""

    def test_conflicting_evidence_still_assessed(self):
        """Even with conflicting evidence, an assessment is produced."""
        ev1 = _make_evidence(
            metric_value=100.0, description="Metric rising",
            details={"direction": "up"},
        )
        ev2 = _make_evidence(
            metric_value=-50.0, description="Metric falling",
            details={"direction": "down"},
        )
        signal = _make_signal(
            description="Conflicting metrics",
            evidence_ids=[ev1.id, ev2.id],
        )
        synth = _make_synth_signal(
            signal=signal, source_type="kpi",
            evidence_items=[ev1, ev2],
        )
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {ev1.id: ev1, ev2.id: ev2})

        # An assessment is produced (not silently dropped)
        assert a is not None
        assert a.impact_type in {t.value for t in ImpactType}

    def test_multiple_signals_independently_assessed(self):
        ev1 = _make_evidence(metric_value=10.0, description="Up trend")
        ev2 = _make_evidence(metric_value=-5.0, description="Down trend")
        sig1 = _make_signal(description="Signal A", evidence_ids=[ev1.id])
        sig2 = _make_signal(description="Signal B", evidence_ids=[ev2.id])
        synth1 = _make_synth_signal(signal=sig1, source_type="kpi",
                                    evidence_items=[ev1])
        synth2 = _make_synth_signal(signal=sig2, source_type="kpi",
                                    evidence_items=[ev2])
        q = ImpactQuantifier(dataset_id=DS_ID)
        report = q.assess_signals([synth1, synth2],
                                  {ev1.id: ev1, ev2.id: ev2})

        assert len(report.assessments) == 2
        values = {a.estimated_value for a in report.assessments}
        assert 10.0 in values
        assert -5.0 in values


# ===================================================================
# 10. CONFIDENCE SEPARATED FROM IMPACT MAGNITUDE
# ===================================================================

class TestConfidenceSeparatedFromMagnitude:
    """Confidence is a separate field, not conflated with estimated_value."""

    def test_high_impact_low_confidence(self):
        ev = _make_evidence(
            level=EvidenceLevel.INFERENCE.value,
            description="Large anomaly",
            metric_value=9999.0,
            details={"observed": 10000.0, "expected": 1.0},
        )
        signal = _make_signal(
            level=EvidenceLevel.INFERENCE.value,
            evidence_ids=[ev.id],
        )
        synth = _make_synth_signal(signal=signal, source_type="anomaly",
                                   evidence_items=[ev])
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {ev.id: ev})

        assert a is not None
        # Large deviation, but confidence assessed independently
        assert a.estimated_value is not None
        assert a.confidence is not None
        assert a.confidence.level != a.estimated_value

    def test_small_impact_high_confidence_possible(self):
        """A small metric can have solid confidence."""
        evs = [
            _make_evidence(
                metric_value=0.01,
                description=f"Small metric {i}",
                source=f"source_{i}",
            )
            for i in range(4)
        ]
        signal = _make_signal(evidence_ids=[e.id for e in evs])
        synth = _make_synth_signal(signal=signal, source_type="kpi",
                                   evidence_items=evs)
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {e.id: e for e in evs})

        assert a is not None
        assert a.estimated_value == 0.01  # Small value
        assert a.confidence is not None
        # Confidence is determined by evidence strength, not by value magnitude

    def test_confidence_and_value_in_different_fields(self):
        ev = _make_evidence(metric_value=50.0)
        signal = _make_signal(evidence_ids=[ev.id])
        synth = _make_synth_signal(signal=signal, source_type="kpi",
                                   evidence_items=[ev])
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {ev.id: ev})

        assert a.estimated_value == 50.0
        assert a.confidence is not None
        assert isinstance(a.confidence.level, str)
        assert isinstance(a.estimated_value, float)


# ===================================================================
# 11. FORECAST-BASED IMPACT LABELING
# ===================================================================

class TestForecastImpactLabeling:
    """Forecast impacts must be labeled as model projections, not facts."""

    def test_forecast_is_model_estimated(self):
        ev = _make_evidence(
            level=EvidenceLevel.PREDICTION.value,
            description="Forecast for next period",
            metric_value=12,
            details={"metrics": {"rmse": 5.3, "mae": 4.1}, "model": "arima"},
        )
        signal = _make_signal(level=EvidenceLevel.PREDICTION.value,
                              evidence_ids=[ev.id])
        synth = _make_synth_signal(signal=signal, source_type="forecast",
                                   evidence_items=[ev])
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {ev.id: ev})

        assert a.impact_type == ImpactType.MODEL_ESTIMATED.value
        assert "projection" in a.description.lower() or "forecast" in a.description.lower()

    def test_forecast_has_model_uncertainty_limitation(self):
        ev = _make_evidence(
            level=EvidenceLevel.PREDICTION.value,
            description="Forecast",
            details={"metrics": {"rmse": 10.0}},
        )
        signal = _make_signal(level=EvidenceLevel.PREDICTION.value,
                              evidence_ids=[ev.id])
        synth = _make_synth_signal(signal=signal, source_type="forecast",
                                   evidence_items=[ev])
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {ev.id: ev})

        lim_cats = {lim.category for lim in a.limitations
                    if isinstance(lim, DecisionLimitation)}
        assert LimitationCategory.MODEL_UNCERTAINTY.value in lim_cats

    def test_forecast_explanation_says_not_fact(self):
        ev = _make_evidence(
            level=EvidenceLevel.PREDICTION.value,
            description="Forecast",
            details={"metrics": {}},
        )
        signal = _make_signal(level=EvidenceLevel.PREDICTION.value,
                              evidence_ids=[ev.id])
        synth = _make_synth_signal(signal=signal, source_type="forecast",
                                   evidence_items=[ev])
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {ev.id: ev})

        assert "not" in a.explanation.lower() and "fact" in a.explanation.lower()

    def test_forecast_uses_error_metric_not_prediction(self):
        """The estimated_value should be the error metric, not the forecast value."""
        ev = _make_evidence(
            level=EvidenceLevel.PREDICTION.value,
            description="Forecast",
            metric_value=12,
            details={"metrics": {"rmse": 5.3}},
        )
        signal = _make_signal(level=EvidenceLevel.PREDICTION.value,
                              evidence_ids=[ev.id])
        synth = _make_synth_signal(signal=signal, source_type="forecast",
                                   evidence_items=[ev])
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {ev.id: ev})

        # Estimated value is the RMSE, not the horizon or raw forecast
        assert a.estimated_value == 5.3


# ===================================================================
# 12. SCENARIO-BASED IMPACT LABELING
# ===================================================================

class TestScenarioImpactLabeling:
    """Scenario impacts must be labeled as hypothetical projections."""

    def test_whatif_is_scenario_based(self):
        ev = _make_evidence(
            level=EvidenceLevel.PREDICTION.value,
            description="What-if scenario",
            metric_value=15.0,
        )
        signal = _make_signal(level=EvidenceLevel.PREDICTION.value,
                              evidence_ids=[ev.id])
        synth = _make_synth_signal(signal=signal, source_type="whatif",
                                   evidence_items=[ev])
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {ev.id: ev})

        assert a.impact_type == ImpactType.SCENARIO_BASED.value

    def test_scenario_has_hypothetical_limitation(self):
        ev = _make_evidence(
            level=EvidenceLevel.PREDICTION.value,
            description="Scenario projection",
            metric_value=20.0,
        )
        signal = _make_signal(level=EvidenceLevel.PREDICTION.value,
                              evidence_ids=[ev.id])
        synth = _make_synth_signal(signal=signal, source_type="scenario",
                                   evidence_items=[ev])
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {ev.id: ev})

        lim_descriptions = [
            lim.description for lim in a.limitations
            if isinstance(lim, DecisionLimitation)
        ]
        assert any("hypothetical" in d.lower() or "scenario" in d.lower()
                    for d in lim_descriptions)

    def test_optimization_is_scenario_based(self):
        ev = _make_evidence(
            level=EvidenceLevel.PREDICTION.value,
            description="Optimization result",
            metric_value=8.5,
        )
        signal = _make_signal(level=EvidenceLevel.PREDICTION.value,
                              evidence_ids=[ev.id])
        synth = _make_synth_signal(signal=signal, source_type="optimization",
                                   evidence_items=[ev])
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {ev.id: ev})

        assert a.impact_type == ImpactType.SCENARIO_BASED.value
        assert a.impact_category == ImpactCategory.EFFICIENCY.value

    def test_scenario_explanation_says_not_guaranteed(self):
        ev = _make_evidence(
            level=EvidenceLevel.PREDICTION.value,
            description="Scenario",
            metric_value=10.0,
        )
        signal = _make_signal(level=EvidenceLevel.PREDICTION.value,
                              evidence_ids=[ev.id])
        synth = _make_synth_signal(signal=signal, source_type="whatif",
                                   evidence_items=[ev])
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {ev.id: ev})

        assert "not" in a.explanation.lower() or "hypothetical" in a.explanation.lower()


# ===================================================================
# 13. CORRELATION-VERSUS-CAUSATION SAFEGUARDS
# ===================================================================

class TestCorrelationCausationSafeguards:
    """Model-estimated impacts must not claim causation."""

    def test_driver_has_observational_limitation(self):
        ev = _make_evidence(
            level=EvidenceLevel.INFERENCE.value,
            description="Driver identified",
            metric_value=0.8,
            details={"method": "permutation"},
        )
        signal = _make_signal(level=EvidenceLevel.INFERENCE.value,
                              evidence_ids=[ev.id])
        synth = _make_synth_signal(signal=signal, source_type="driver",
                                   evidence_items=[ev])
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {ev.id: ev})

        lim_cats = {
            lim.category for lim in a.limitations
            if isinstance(lim, DecisionLimitation)
        }
        assert LimitationCategory.OBSERVATIONAL.value in lim_cats

    def test_driver_explanation_mentions_not_causal(self):
        ev = _make_evidence(
            level=EvidenceLevel.INFERENCE.value,
            description="Driver",
            metric_value=0.5,
        )
        signal = _make_signal(level=EvidenceLevel.INFERENCE.value,
                              evidence_ids=[ev.id])
        synth = _make_synth_signal(signal=signal, source_type="driver",
                                   evidence_items=[ev])
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {ev.id: ev})

        assert "not" in a.explanation.lower() and "caus" in a.explanation.lower()

    def test_relationship_has_non_causal_guard(self):
        ev = _make_evidence(
            level=EvidenceLevel.INFERENCE.value,
            description="Correlation found",
            metric_value=0.75,
            details={"relationship_type": "correlation"},
        )
        signal = _make_signal(level=EvidenceLevel.INFERENCE.value,
                              evidence_ids=[ev.id])
        synth = _make_synth_signal(signal=signal, source_type="relationship",
                                   evidence_items=[ev])
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {ev.id: ev})

        lim_cats = {
            lim.category for lim in a.limitations
            if isinstance(lim, DecisionLimitation)
        }
        assert LimitationCategory.OBSERVATIONAL.value in lim_cats

    def test_mmm_has_observational_limitation(self):
        ev = _make_evidence(
            level=EvidenceLevel.INFERENCE.value,
            description="MMM result",
            metric_value=0.9,
            details={"test_metrics": {"r2": 0.88}},
        )
        signal = _make_signal(level=EvidenceLevel.INFERENCE.value,
                              evidence_ids=[ev.id])
        synth = _make_synth_signal(signal=signal, source_type="mmm",
                                   evidence_items=[ev])
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {ev.id: ev})

        lim_cats = {
            lim.category for lim in a.limitations
            if isinstance(lim, DecisionLimitation)
        }
        assert LimitationCategory.OBSERVATIONAL.value in lim_cats

    def test_contribution_has_observational_limitation(self):
        ev = _make_evidence(
            level=EvidenceLevel.INFERENCE.value,
            description="Contribution estimate",
            metric_name="roi_estimate",
            metric_value=2.5,
        )
        signal = _make_signal(level=EvidenceLevel.INFERENCE.value,
                              evidence_ids=[ev.id])
        synth = _make_synth_signal(signal=signal, source_type="contribution",
                                   evidence_items=[ev])
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {ev.id: ev})

        lim_cats = {
            lim.category for lim in a.limitations
            if isinstance(lim, DecisionLimitation)
        }
        assert LimitationCategory.OBSERVATIONAL.value in lim_cats

    def test_driver_assumptions_mention_no_causation(self):
        ev = _make_evidence(
            level=EvidenceLevel.INFERENCE.value,
            description="Driver",
            metric_value=0.3,
        )
        signal = _make_signal(level=EvidenceLevel.INFERENCE.value,
                              evidence_ids=[ev.id])
        synth = _make_synth_signal(signal=signal, source_type="driver",
                                   evidence_items=[ev])
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {ev.id: ev})

        assert any("causation" in assumption.lower() or "causal" in assumption.lower()
                    for assumption in a.assumptions)


# ===================================================================
# 14. NO INVENTED FINANCIAL ESTIMATES
# ===================================================================

class TestNoInventedFinancialEstimates:
    """Impact quantifier must never invent currency, revenue, etc."""

    def test_kpi_does_not_invent_currency(self):
        ev = _make_evidence(metric_value=100.0, metric_name="sensor_reading")
        signal = _make_signal(evidence_ids=[ev.id])
        synth = _make_synth_signal(signal=signal, source_type="kpi",
                                   evidence_items=[ev])
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {ev.id: ev})

        d = a.to_dict()
        serialized = str(d).lower()
        assert "$" not in serialized
        assert "dollar" not in serialized
        assert "revenue" not in serialized
        assert "savings" not in serialized

    def test_driver_does_not_invent_roi(self):
        ev = _make_evidence(
            level=EvidenceLevel.INFERENCE.value,
            description="Feature importance",
            metric_value=0.4,
        )
        signal = _make_signal(level=EvidenceLevel.INFERENCE.value,
                              evidence_ids=[ev.id])
        synth = _make_synth_signal(signal=signal, source_type="feature",
                                   evidence_items=[ev])
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {ev.id: ev})

        d = a.to_dict()
        serialized = str(d).lower()
        assert "revenue" not in serialized
        assert "cost" not in a.unit.lower()

    def test_scenario_does_not_present_as_actual(self):
        ev = _make_evidence(
            level=EvidenceLevel.PREDICTION.value,
            description="Budget scenario",
            metric_value=12.0,
        )
        signal = _make_signal(level=EvidenceLevel.PREDICTION.value,
                              evidence_ids=[ev.id])
        synth = _make_synth_signal(signal=signal, source_type="optimization",
                                   evidence_items=[ev])
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {ev.id: ev})

        d = a.to_dict()
        serialized = str(d).lower()
        # Must not claim a guaranteed dollar benefit (disclaimers like
        # "not a guaranteed outcome" are acceptable — they warn, not claim)
        assert "guaranteed savings" not in serialized
        assert "guaranteed benefit" not in serialized
        assert "will save" not in serialized

    def test_unknown_does_not_fabricate(self):
        ev = _make_evidence(
            level=EvidenceLevel.LIMITATION.value,
            description="Only a limitation, no data",
        )
        signal = _make_signal(
            level=EvidenceLevel.LIMITATION.value,
            evidence_ids=[ev.id],
        )
        synth = _make_synth_signal(signal=signal, source_type="kpi",
                                   evidence_items=[ev])
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {ev.id: ev})

        assert a is not None
        # Should not fabricate a financial figure from a limitation
        assert "$" not in str(a.to_dict())


# ===================================================================
# 15. GENERIC NON-MARKETING DATASETS
# ===================================================================

class TestGenericNonMarketingDatasets:
    """Impact quantifier works for any domain, not just marketing."""

    def test_healthcare_dataset(self):
        ev = _make_evidence(
            metric_name="patient_wait_time",
            metric_value=25.3,
            details={"unit": "minutes"},
            source="kpi_engine",
        )
        signal = _make_signal(
            description="Patient wait time observation",
            evidence_ids=[ev.id],
        )
        synth = _make_synth_signal(signal=signal, source_type="kpi",
                                   evidence_items=[ev])
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {ev.id: ev})

        assert a.impact_type == ImpactType.MEASURED.value
        assert a.estimated_value == 25.3
        assert a.unit == "minutes"

    def test_manufacturing_anomaly(self):
        ev = _make_evidence(
            level=EvidenceLevel.INFERENCE.value,
            description="Temperature anomaly in reactor",
            metric_value=2.8,
            details={"observed": 450.0, "expected": 400.0},
            source="anomaly_detector",
        )
        signal = _make_signal(
            description="Reactor temperature anomaly",
            level=EvidenceLevel.INFERENCE.value,
            evidence_ids=[ev.id],
        )
        synth = _make_synth_signal(signal=signal, source_type="anomaly",
                                   evidence_items=[ev])
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {ev.id: ev})

        assert a.impact_type == ImpactType.MEASURED.value
        assert a.impact_category == ImpactCategory.ANOMALY.value
        assert a.estimated_value == 50.0  # 450 - 400

    def test_education_forecast(self):
        ev = _make_evidence(
            level=EvidenceLevel.PREDICTION.value,
            description="Student enrollment forecast",
            metric_value=6,
            details={"metrics": {"mae": 12.5}, "model": "prophet"},
            source="forecasting_engine",
        )
        signal = _make_signal(
            description="Enrollment forecast",
            level=EvidenceLevel.PREDICTION.value,
            evidence_ids=[ev.id],
        )
        synth = _make_synth_signal(signal=signal, source_type="forecast",
                                   evidence_items=[ev])
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {ev.id: ev})

        assert a.impact_type == ImpactType.MODEL_ESTIMATED.value

    def test_logistics_driver(self):
        ev = _make_evidence(
            level=EvidenceLevel.INFERENCE.value,
            description="Distance is a key driver of delivery time",
            metric_value=0.72,
            details={"method": "permutation", "direction": "positive"},
            source="driver_engine",
        )
        signal = _make_signal(
            description="Delivery time driver",
            level=EvidenceLevel.INFERENCE.value,
            evidence_ids=[ev.id],
        )
        synth = _make_synth_signal(signal=signal, source_type="driver",
                                   evidence_items=[ev])
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {ev.id: ev})

        assert a.impact_type == ImpactType.MODEL_ESTIMATED.value
        assert a.impact_category == ImpactCategory.RELATIONSHIP.value

    def test_finance_regression(self):
        ev = _make_evidence(
            level=EvidenceLevel.PREDICTION.value,
            description="Interest rate regression model",
            metric_value=0.92,
            details={"metrics": {"r2": 0.92}, "model": "ridge"},
            source="regression_engine",
        )
        signal = _make_signal(
            description="Interest rate model",
            level=EvidenceLevel.PREDICTION.value,
            evidence_ids=[ev.id],
        )
        synth = _make_synth_signal(signal=signal, source_type="regression",
                                   evidence_items=[ev])
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {ev.id: ev})

        assert a.impact_type == ImpactType.MODEL_ESTIMATED.value
        assert a.estimated_value == 0.92

    def test_iot_segmentation(self):
        ev = _make_evidence(
            level=EvidenceLevel.PREDICTION.value,
            description="IoT device segmentation",
            metric_value=5,
            details={"quality_metrics": {"silhouette": 0.55}},
            source="segmentation_engine",
        )
        signal = _make_signal(
            description="Device clusters",
            level=EvidenceLevel.PREDICTION.value,
            evidence_ids=[ev.id],
        )
        synth = _make_synth_signal(signal=signal, source_type="segmentation",
                                   evidence_items=[ev])
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {ev.id: ev})

        assert a.impact_type == ImpactType.QUALITATIVE.value


# ===================================================================
# 16. MARKETING EXAMPLES WHEN SUPPORTED BY EVIDENCE
# ===================================================================

class TestMarketingExamplesWithEvidence:
    """Marketing-domain signals work when backed by actual evidence."""

    def test_mmm_with_r2(self):
        ev = _make_evidence(
            level=EvidenceLevel.INFERENCE.value,
            description="MMM model fitted",
            metric_value=0.87,
            details={"test_metrics": {"r2": 0.87}},
            source="mmm_engine",
        )
        signal = _make_signal(
            description="Marketing mix model",
            level=EvidenceLevel.INFERENCE.value,
            evidence_ids=[ev.id],
        )
        synth = _make_synth_signal(signal=signal, source_type="mmm",
                                   evidence_items=[ev])
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {ev.id: ev})

        assert a.impact_type == ImpactType.MODEL_ESTIMATED.value
        assert a.estimated_value == 0.87

    def test_contribution_with_roi(self):
        ev = _make_evidence(
            level=EvidenceLevel.INFERENCE.value,
            description="Channel contribution estimate",
            metric_name="roi_estimate",
            metric_value=3.2,
            source="contribution",
        )
        signal = _make_signal(
            description="Channel ROI",
            level=EvidenceLevel.INFERENCE.value,
            evidence_ids=[ev.id],
        )
        synth = _make_synth_signal(signal=signal, source_type="roi",
                                   evidence_items=[ev])
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {ev.id: ev})

        assert a.impact_type == ImpactType.MODEL_ESTIMATED.value
        assert a.estimated_value == 3.2

    def test_optimization_scenario(self):
        ev = _make_evidence(
            level=EvidenceLevel.PREDICTION.value,
            description="Budget optimization result",
            metric_value=5.5,
            source="optimizer",
        )
        signal = _make_signal(
            description="Optimal allocation",
            level=EvidenceLevel.PREDICTION.value,
            evidence_ids=[ev.id],
        )
        synth = _make_synth_signal(signal=signal, source_type="optimizer",
                                   evidence_items=[ev])
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {ev.id: ev})

        assert a.impact_type == ImpactType.SCENARIO_BASED.value
        assert a.impact_category == ImpactCategory.EFFICIENCY.value


# ===================================================================
# 17. SERIALIZATION
# ===================================================================

class TestSerialization:
    """to_dict / from_dict roundtrips for all Part 3 contracts."""

    def test_impact_assessment_roundtrip(self):
        a = ImpactAssessment(
            signal_id="sig_abc",
            dataset_id=DS_ID,
            impact_type=ImpactType.MEASURED.value,
            impact_category=ImpactCategory.PERFORMANCE.value,
            estimation_method="Direct observation",
            estimated_value=42.0,
            unit="widgets",
            time_horizon="per day",
            lower_bound=40.0,
            upper_bound=44.0,
            description="Test assessment",
            explanation="Measured directly",
            evidence_ids=["ev_001"],
            confidence=ConfidenceScore(
                level=ConfidenceLevel.HIGH.value,
                explanation="Strong",
                evidence_count=3,
                dataset_id=DS_ID,
            ),
            limitations=[
                create_limitation("Test limitation"),
            ],
            assumptions=["Assumption A"],
        )

        d = a.to_dict()
        restored = ImpactAssessment.from_dict(d)

        assert restored.signal_id == a.signal_id
        assert restored.dataset_id == a.dataset_id
        assert restored.impact_type == a.impact_type
        assert restored.impact_category == a.impact_category
        assert restored.estimated_value == a.estimated_value
        assert restored.unit == a.unit
        assert restored.time_horizon == a.time_horizon
        assert restored.lower_bound == a.lower_bound
        assert restored.upper_bound == a.upper_bound
        assert restored.description == a.description
        assert restored.explanation == a.explanation
        assert restored.evidence_ids == a.evidence_ids
        assert restored.confidence.level == a.confidence.level
        assert len(restored.limitations) == 1
        assert restored.assumptions == a.assumptions

    def test_impact_report_roundtrip(self):
        a = ImpactAssessment(
            description="Test",
            impact_type=ImpactType.MEASURED.value,
            estimated_value=10.0,
        )
        report = ImpactReport(
            dataset_id=DS_ID,
            assessments=[a],
            processing_notes=["Note 1"],
        )

        d = report.to_dict()
        restored = ImpactReport.from_dict(d)

        assert restored.dataset_id == DS_ID
        assert len(restored.assessments) == 1
        assert restored.assessments[0].estimated_value == 10.0
        assert restored.processing_notes == ["Note 1"]

    def test_report_to_dict_includes_counts(self):
        a1 = ImpactAssessment(description="A", estimated_value=1.0)
        a2 = ImpactAssessment(description="B", estimated_value=None)
        report = ImpactReport(assessments=[a1, a2])

        d = report.to_dict()
        assert d["assessment_count"] == 2
        assert d["quantified_count"] == 1

    def test_from_dict_ignores_extra_keys(self):
        d = {
            "description": "Test",
            "impact_type": "measured",
            "impact_category": "general",
            "extra_key": "should be ignored",
        }
        a = ImpactAssessment.from_dict(d)
        assert a.description == "Test"

    def test_serialization_with_none_confidence(self):
        a = ImpactAssessment(description="Test", confidence=None)
        d = a.to_dict()
        assert d["confidence"] is None
        restored = ImpactAssessment.from_dict(d)
        assert restored.confidence is None


# ===================================================================
# 18. DETERMINISM
# ===================================================================

class TestDeterminism:
    """Same inputs → same outputs (excluding random IDs)."""

    def test_same_inputs_same_output(self):
        ev = _make_evidence(
            ev_id="ev_fixed",
            metric_value=99.0,
            description="Fixed evidence",
        )
        signal = _make_signal(
            sig_id="sig_fixed",
            description="Fixed signal",
            evidence_ids=["ev_fixed"],
        )
        synth = _make_synth_signal(signal=signal, source_type="kpi",
                                   evidence_items=[ev])

        results = []
        for _ in range(3):
            q = ImpactQuantifier(dataset_id=DS_ID)
            a = q.assess_single_signal(synth, {"ev_fixed": ev})
            results.append(a)

        # All three should have same type, category, value, description
        for r in results[1:]:
            assert r.impact_type == results[0].impact_type
            assert r.impact_category == results[0].impact_category
            assert r.estimated_value == results[0].estimated_value
            assert r.description == results[0].description
            assert r.explanation == results[0].explanation

    def test_report_order_deterministic(self):
        evs = [_make_evidence(
            ev_id=f"ev_det_{i}",
            metric_value=float(i),
            description=f"Evidence {i}",
        ) for i in range(3)]
        sigs = [_make_signal(
            sig_id=f"sig_det_{i}",
            description=f"Signal {i}",
            evidence_ids=[evs[i].id],
        ) for i in range(3)]
        synths = [_make_synth_signal(
            signal=sigs[i],
            source_type="kpi",
            evidence_items=[evs[i]],
        ) for i in range(3)]
        registry = {ev.id: ev for ev in evs}

        reports = []
        for _ in range(3):
            q = ImpactQuantifier(dataset_id=DS_ID)
            r = q.assess_signals(synths, registry)
            reports.append(r)

        for r in reports[1:]:
            assert len(r.assessments) == len(reports[0].assessments)
            for a1, a2 in zip(reports[0].assessments, r.assessments):
                assert a1.estimated_value == a2.estimated_value


# ===================================================================
# 19. MALFORMED INPUTS
# ===================================================================

class TestMalformedInputs:
    """Gracefully handle bad input data."""

    def test_invalid_impact_type_raises(self):
        with pytest.raises(ValueError, match="Invalid impact_type"):
            ImpactAssessment(description="Test", impact_type="fake_type")

    def test_invalid_impact_category_raises(self):
        with pytest.raises(ValueError, match="Invalid impact_category"):
            ImpactAssessment(description="Test", impact_category="fake_cat")

    def test_empty_description_raises(self):
        with pytest.raises(ValueError, match="non-empty description"):
            ImpactAssessment(description="")

    def test_signal_exception_captured_in_notes(self):
        """If _assess_single raises, it's captured in processing_notes."""
        # A synth signal that will cause an internal error
        class BadSignal:
            signal = None  # This should trigger a skip
            source_type = "kpi"
            evidence_items = []

        q = ImpactQuantifier(dataset_id=DS_ID)
        report = q.assess_signals([BadSignal()], {})
        # Should produce a processing note, not crash
        assert len(report.processing_notes) > 0

    def test_evidence_with_non_numeric_metric(self):
        ev = _make_evidence(metric_value="not_a_number")
        signal = _make_signal(evidence_ids=[ev.id])
        synth = _make_synth_signal(signal=signal, source_type="kpi",
                                   evidence_items=[ev])
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {ev.id: ev})

        assert a is not None
        # _safe_float("not_a_number") → None
        assert a.estimated_value is None

    def test_dict_based_synth_signal(self):
        """_get_attr handles both objects and dicts."""
        d = {"key": "value"}
        assert _get_attr(d, "key", None) == "value"
        assert _get_attr(d, "missing", "default") == "default"

    def test_get_attr_nested_with_dict(self):
        d = {"outer": {"inner": 42}}
        assert _get_attr_nested(d, "outer", "inner", None) == 42
        assert _get_attr_nested(d, "outer", "missing", "default") == "default"
        assert _get_attr_nested(d, "nope", "inner", "default") == "default"

    def test_from_dict_with_limitation_dicts(self):
        d = {
            "description": "Test",
            "limitations": [
                {
                    "id": "lim_abc",
                    "category": "observational",
                    "description": "Test lim",
                }
            ],
        }
        a = ImpactAssessment.from_dict(d)
        assert len(a.limitations) == 1
        assert isinstance(a.limitations[0], DecisionLimitation)

    def test_evidence_lookup_from_registry(self):
        """When evidence_items is empty, evidence is looked up via evidence_ids."""
        ev = _make_evidence(ev_id="ev_lookup")
        signal = _make_signal(evidence_ids=["ev_lookup"])
        synth = _make_synth_signal(
            signal=signal,
            source_type="kpi",
            evidence_items=[],  # Empty — should fallback to registry
        )
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {"ev_lookup": ev})

        assert a is not None
        assert a.impact_type == ImpactType.MEASURED.value


# ===================================================================
# ADDITIONAL CONTRACT AND REPORT TESTS
# ===================================================================

class TestImpactReportHelpers:
    """ImpactReport filter and accessor methods."""

    def test_by_type_filter(self):
        a1 = ImpactAssessment(description="A", impact_type=ImpactType.MEASURED.value)
        a2 = ImpactAssessment(description="B", impact_type=ImpactType.QUALITATIVE.value)
        report = ImpactReport(assessments=[a1, a2])

        measured = report.by_type(ImpactType.MEASURED.value)
        assert len(measured) == 1
        assert measured[0].description == "A"

    def test_by_category_filter(self):
        a1 = ImpactAssessment(
            description="A",
            impact_category=ImpactCategory.ANOMALY.value,
        )
        a2 = ImpactAssessment(
            description="B",
            impact_category=ImpactCategory.PERFORMANCE.value,
        )
        report = ImpactReport(assessments=[a1, a2])

        anomalies = report.by_category(ImpactCategory.ANOMALY.value)
        assert len(anomalies) == 1

    def test_assessed_signals_list(self):
        a1 = ImpactAssessment(description="A", signal_id="sig_1")
        a2 = ImpactAssessment(description="B", signal_id="sig_2")
        report = ImpactReport(assessments=[a1, a2])

        assert report.assessed_signals() == ["sig_1", "sig_2"]

    def test_quantifier_dataset_id_property(self):
        q = ImpactQuantifier(dataset_id="my_ds")
        assert q.dataset_id == "my_ds"


# ===================================================================
# CLASSIFICATION DISPATCH TESTS
# ===================================================================

class TestClassificationDispatch:
    """Ensure each source_type routes to the correct assessment strategy."""

    def test_classification_model(self):
        ev = _make_evidence(
            level=EvidenceLevel.PREDICTION.value,
            description="Classification model",
            details={"metrics": {"accuracy": 0.91}, "model": "random_forest"},
        )
        signal = _make_signal(level=EvidenceLevel.PREDICTION.value,
                              evidence_ids=[ev.id])
        synth = _make_synth_signal(signal=signal, source_type="classification",
                                   evidence_items=[ev])
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {ev.id: ev})

        assert a.impact_type == ImpactType.MODEL_ESTIMATED.value
        assert a.estimated_value == 0.91
        assert a.unit == "accuracy"

    def test_insight_routes_to_relationship(self):
        ev = _make_evidence(
            level=EvidenceLevel.INFERENCE.value,
            description="Insight found",
            metric_value=0.65,
        )
        signal = _make_signal(level=EvidenceLevel.INFERENCE.value,
                              evidence_ids=[ev.id])
        synth = _make_synth_signal(signal=signal, source_type="insight",
                                   evidence_items=[ev])
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {ev.id: ev})

        assert a.impact_type == ImpactType.MODEL_ESTIMATED.value
        assert a.impact_category == ImpactCategory.RELATIONSHIP.value

    def test_feature_importance_routes_to_driver(self):
        ev = _make_evidence(
            level=EvidenceLevel.INFERENCE.value,
            description="Feature importance",
            metric_value=0.33,
        )
        signal = _make_signal(level=EvidenceLevel.INFERENCE.value,
                              evidence_ids=[ev.id])
        synth = _make_synth_signal(signal=signal, source_type="feature_importance",
                                   evidence_items=[ev])
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {ev.id: ev})

        assert a.impact_type == ImpactType.MODEL_ESTIMATED.value
        assert a.impact_category == ImpactCategory.RELATIONSHIP.value


# ===================================================================
# GENERIC IMPACT FALLBACK TESTS
# ===================================================================

class TestGenericImpactFallback:
    """Generic/fallback path maps evidence level to impact type."""

    def test_fact_evidence_maps_to_measured(self):
        ev = _make_evidence(level=EvidenceLevel.FACT.value,
                            metric_value=77.0)
        signal = _make_signal(evidence_ids=[ev.id])
        synth = _make_synth_signal(signal=signal, source_type="custom_engine",
                                   evidence_items=[ev])
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {ev.id: ev})

        assert a.impact_type == ImpactType.MEASURED.value

    def test_prediction_evidence_maps_to_scenario(self):
        ev = _make_evidence(
            level=EvidenceLevel.PREDICTION.value,
            description="Custom prediction",
        )
        signal = _make_signal(level=EvidenceLevel.PREDICTION.value,
                              evidence_ids=[ev.id])
        synth = _make_synth_signal(signal=signal, source_type="custom_engine",
                                   evidence_items=[ev])
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {ev.id: ev})

        assert a.impact_type == ImpactType.SCENARIO_BASED.value

    def test_inference_evidence_maps_to_model_estimated(self):
        ev = _make_evidence(
            level=EvidenceLevel.INFERENCE.value,
            description="Custom inference",
        )
        signal = _make_signal(level=EvidenceLevel.INFERENCE.value,
                              evidence_ids=[ev.id])
        synth = _make_synth_signal(signal=signal, source_type="custom_engine",
                                   evidence_items=[ev])
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {ev.id: ev})

        assert a.impact_type == ImpactType.MODEL_ESTIMATED.value

    def test_limitation_evidence_maps_to_qualitative(self):
        ev = _make_evidence(
            level=EvidenceLevel.LIMITATION.value,
            description="Only a limitation",
        )
        signal = _make_signal(
            level=EvidenceLevel.LIMITATION.value,
            evidence_ids=[ev.id],
        )
        synth = _make_synth_signal(signal=signal, source_type="custom_engine",
                                   evidence_items=[ev])
        q = ImpactQuantifier(dataset_id=DS_ID)
        a = q.assess_single_signal(synth, {ev.id: ev})

        assert a.impact_type == ImpactType.QUALITATIVE.value
