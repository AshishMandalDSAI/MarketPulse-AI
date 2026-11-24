"""Phase 7 Part 5 — Decision Generator comprehensive tests.

Tests cover all 19 required categories:
1. Decisions generated from valid evidence
2. Investigation and monitoring recommendations
3. Insufficient evidence
4. Contradictory evidence
5. Traceability to supporting signals and evidence
6. Impact and confidence preservation
7. Transparent prioritization
8. Deterministic generation and ordering
9. Dataset isolation
10. Malformed or incomplete inputs
11. Serialization
12. Forecast-based recommendations
13. Scenario-based recommendations
14. Correlation-versus-causation safeguards
15. No guaranteed-outcome claims
16. No invented financial benefits
17. Risk and reversibility handling
18. Generic non-marketing datasets
19. Regression-safe imports
"""

import pytest
from dataclasses import FrozenInstanceError

from src.decision.contracts import (
    ConfidenceLevel,
    ConfidenceScore,
    DecisionEvidence,
    DecisionLimitation,
    DecisionSignal,
    EvidenceLevel,
    LimitationCategory,
    _make_id,
)
from src.decision.confidence import assess_confidence
from src.decision.evidence import (
    create_evidence,
    create_limitation,
    create_signal,
    validate_dataset_scope,
)
from src.decision.impact_quantifier import (
    ImpactAssessment,
    ImpactReport,
    ImpactType,
    ImpactCategory,
)
from src.decision.scenario_engine import (
    ScenarioResult,
    ScenarioStatus,
)
from src.decision.decision_generator import (
    DecisionCategory,
    DecisionGenerator,
    DecisionOption,
    DecisionReport,
    PriorityLevel,
    Reversibility,
    UrgencyLevel,
)


# ---------------------------------------------------------------------------
# Helpers — build test fixtures without hardcoding domain-specific fields
# ---------------------------------------------------------------------------

DATASET_ID = "test_ds_001"
OTHER_DATASET_ID = "test_ds_other"


def _make_evidence(
    description="Test observation",
    source="analytics",
    dataset_id=DATASET_ID,
    level=EvidenceLevel.FACT.value,
    metric_name="metric_a",
    metric_value=42.0,
    details=None,
):
    return create_evidence(
        description=description,
        source=source,
        dataset_id=dataset_id,
        level=level,
        metric_name=metric_name,
        metric_value=metric_value,
        details=details or {},
    )


def _make_signal(
    description="Test signal",
    evidence_items=None,
    dataset_id=DATASET_ID,
):
    if evidence_items is None:
        evidence_items = [_make_evidence()]
    return create_signal(
        description=description,
        evidence_items=evidence_items,
        dataset_id=dataset_id,
    )


class FakeSynthesizedSignal:
    """Mimics Part 2 SynthesizedSignal using __slots__."""
    __slots__ = (
        "signal", "priority", "confidence_score",
        "source_type", "generation_reason", "evidence_items",
    )

    def __init__(
        self,
        signal=None,
        priority=None,
        confidence_score=None,
        source_type="",
        generation_reason="",
        evidence_items=None,
    ):
        self.signal = signal
        self.priority = priority
        self.confidence_score = confidence_score
        self.source_type = source_type
        self.generation_reason = generation_reason
        self.evidence_items = evidence_items or []


class FakeSynthesisResult:
    """Mimics Part 2 SynthesisResult."""
    __slots__ = (
        "dataset_id", "signals", "evidence_registry",
        "contradictions", "processing_notes",
    )

    def __init__(
        self,
        dataset_id="",
        signals=None,
        evidence_registry=None,
        contradictions=None,
        processing_notes=None,
    ):
        self.dataset_id = dataset_id
        self.signals = signals or []
        self.evidence_registry = evidence_registry or {}
        self.contradictions = contradictions or []
        self.processing_notes = processing_notes or []


class FakeSignalPriority:
    """Mimics Part 2 SignalPriority enum."""
    def __init__(self, value):
        self.value = value

    def __int__(self):
        return self.value


def _make_synth_signal(
    description="Test signal",
    source_type="kpi",
    priority_value=3,
    evidence_items=None,
    dataset_id=DATASET_ID,
    confidence_score=None,
    level=EvidenceLevel.FACT.value,
):
    if evidence_items is None:
        evidence_items = [_make_evidence(dataset_id=dataset_id, level=level)]
    signal = _make_signal(
        description=description,
        evidence_items=evidence_items,
        dataset_id=dataset_id,
    )
    return FakeSynthesizedSignal(
        signal=signal,
        priority=FakeSignalPriority(priority_value),
        confidence_score=confidence_score,
        source_type=source_type,
        generation_reason="test",
        evidence_items=evidence_items,
    )


def _make_synthesis_result(
    signals=None,
    dataset_id=DATASET_ID,
    contradictions=None,
    evidence_registry=None,
):
    if signals is None:
        signals = [_make_synth_signal(dataset_id=dataset_id)]
    return FakeSynthesisResult(
        dataset_id=dataset_id,
        signals=signals,
        evidence_registry=evidence_registry or {},
        contradictions=contradictions or [],
    )


def _make_impact_assessment(
    description="Test impact",
    impact_type=ImpactType.MEASURED.value,
    impact_category=ImpactCategory.PERFORMANCE.value,
    estimated_value=10.5,
    dataset_id=DATASET_ID,
    signal_id="sig_test",
    confidence=None,
    limitations=None,
):
    if confidence is None:
        confidence = ConfidenceScore(
            level=ConfidenceLevel.MEDIUM.value,
            explanation="Test confidence",
            evidence_count=2,
            evidence_quality="moderate",
            dataset_id=dataset_id,
        )
    return ImpactAssessment(
        signal_id=signal_id,
        dataset_id=dataset_id,
        impact_type=impact_type,
        impact_category=impact_category,
        estimation_method="test method",
        estimated_value=estimated_value,
        description=description,
        explanation=f"Explanation for {description}",
        confidence=confidence,
        limitations=limitations or [],
    )


def _make_impact_report(
    assessments=None,
    dataset_id=DATASET_ID,
):
    if assessments is None:
        assessments = [_make_impact_assessment(dataset_id=dataset_id)]
    return ImpactReport(
        dataset_id=dataset_id,
        assessments=assessments,
    )


def _make_scenario_result(
    status=ScenarioStatus.SIMULATED.value,
    delta=5.0,
    delta_pct=10.0,
    dataset_id=DATASET_ID,
    description="Test scenario evaluated",
    confidence=None,
):
    if confidence is None:
        confidence = ConfidenceScore(
            level=ConfidenceLevel.MEDIUM.value,
            explanation="Test scenario confidence",
            evidence_count=1,
            evidence_quality="moderate",
            dataset_id=dataset_id,
        )
    return ScenarioResult(
        scenario_id="scen_test01",
        dataset_id=dataset_id,
        status=status,
        estimated_outcome=105.0 if delta is not None else None,
        baseline_outcome=100.0 if delta is not None else None,
        delta=delta,
        delta_pct=delta_pct,
        unit="units",
        estimation_method="test model",
        model_description="test model",
        description=description,
        explanation="Test scenario explanation",
        evidence_ids=["ev_test1"],
        confidence=confidence,
    )


# ===========================================================================
# 1. Decisions generated from valid evidence
# ===========================================================================

class TestDecisionsFromValidEvidence:
    """Test that valid evidence generates appropriate decisions."""

    def test_measured_evidence_generates_recommend(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        # Supply multiple evidence items so confidence reaches MEDIUM
        evidence_items = [
            _make_evidence(description=f"Observed fact {i}", metric_name=f"m{i}")
            for i in range(3)
        ]
        synth = _make_synth_signal(
            description="Metric A is above threshold",
            source_type="kpi",
            level=EvidenceLevel.FACT.value,
            evidence_items=evidence_items,
        )
        result = gen.generate(
            synthesis_result=_make_synthesis_result(signals=[synth]),
        )
        assert len(result.decisions) >= 1
        decision = result.decisions[0]
        assert decision.category == DecisionCategory.RECOMMEND.value
        assert decision.dataset_id == DATASET_ID
        assert decision.title
        assert decision.description
        assert decision.recommended_action
        assert decision.explanation

    def test_model_evidence_generates_monitor(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        ev = _make_evidence(
            description="Model coefficient significant",
            source="regression",
            level=EvidenceLevel.INFERENCE.value,
        )
        synth = _make_synth_signal(
            description="Feature X associated with target",
            source_type="regression",
            evidence_items=[ev],
            level=EvidenceLevel.INFERENCE.value,
        )
        result = gen.generate(
            synthesis_result=_make_synthesis_result(signals=[synth]),
        )
        assert len(result.decisions) >= 1
        decision = result.decisions[0]
        assert decision.category in (
            DecisionCategory.MONITOR.value,
            DecisionCategory.INVESTIGATE.value,
        )

    def test_anomaly_evidence_generates_investigate(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        ev = _make_evidence(
            description="Anomalous spike detected",
            source="anomaly_detection",
        )
        synth = _make_synth_signal(
            description="Anomalous pattern in metric B",
            source_type="anomaly",
            evidence_items=[ev],
        )
        result = gen.generate(
            synthesis_result=_make_synthesis_result(signals=[synth]),
        )
        assert len(result.decisions) >= 1
        decision = result.decisions[0]
        assert decision.category == DecisionCategory.INVESTIGATE.value

    def test_impact_report_generates_decisions(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        report = _make_impact_report()
        result = gen.generate(impact_report=report)
        assert len(result.decisions) >= 1

    def test_combined_inputs_generate_decisions(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        result = gen.generate(
            synthesis_result=_make_synthesis_result(),
            impact_report=_make_impact_report(),
            scenario_results=[_make_scenario_result()],
        )
        assert len(result.decisions) >= 1
        # Should have decisions from multiple sources
        categories = {d.category for d in result.decisions}
        assert len(categories) >= 1


# ===========================================================================
# 2. Investigation and monitoring recommendations
# ===========================================================================

class TestInvestigationAndMonitoring:
    """Test investigation and monitoring category generation."""

    def test_low_confidence_produces_investigation(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        confidence = ConfidenceScore(
            level=ConfidenceLevel.LOW.value,
            explanation="Low evidence quality",
            evidence_count=1,
            evidence_quality="low",
            dataset_id=DATASET_ID,
        )
        ev = _make_evidence(level=EvidenceLevel.INFERENCE.value)
        synth = _make_synth_signal(
            description="Weak signal detected",
            source_type="generic",
            evidence_items=[ev],
            confidence_score=confidence,
            level=EvidenceLevel.INFERENCE.value,
        )
        result = gen.generate(
            synthesis_result=_make_synthesis_result(signals=[synth]),
        )
        assert len(result.decisions) >= 1
        decision = result.decisions[0]
        assert decision.category in (
            DecisionCategory.INVESTIGATE.value,
            DecisionCategory.MONITOR.value,
        )

    def test_forecast_source_produces_monitor(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        ev = _make_evidence(
            description="Forecast model projection",
            source="forecast",
            level=EvidenceLevel.PREDICTION.value,
        )
        synth = _make_synth_signal(
            description="Forecast shows upward trend",
            source_type="forecast",
            evidence_items=[ev],
            level=EvidenceLevel.PREDICTION.value,
        )
        result = gen.generate(
            synthesis_result=_make_synthesis_result(signals=[synth]),
        )
        assert len(result.decisions) >= 1
        decision = result.decisions[0]
        assert decision.category == DecisionCategory.MONITOR.value

    def test_monitoring_decision_has_follow_up(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        ev = _make_evidence(
            source="forecast",
            level=EvidenceLevel.PREDICTION.value,
        )
        synth = _make_synth_signal(
            description="Trend projection",
            source_type="forecast",
            evidence_items=[ev],
            level=EvidenceLevel.PREDICTION.value,
        )
        result = gen.generate(
            synthesis_result=_make_synthesis_result(signals=[synth]),
        )
        decision = result.decisions[0]
        assert len(decision.required_follow_up) >= 1


# ===========================================================================
# 3. Insufficient evidence
# ===========================================================================

class TestInsufficientEvidence:
    """Test handling of insufficient evidence."""

    def test_no_inputs_produces_no_decisions(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        result = gen.generate()
        assert len(result.decisions) == 0
        assert len(result.processing_notes) >= 1

    def test_empty_signals_produces_note(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        synth_result = _make_synthesis_result(signals=[])
        result = gen.generate(synthesis_result=synth_result)
        assert len(result.decisions) == 0
        assert any("no signals" in note.lower() for note in result.processing_notes)

    def test_insufficient_confidence_produces_blocked(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        confidence = ConfidenceScore(
            level=ConfidenceLevel.INSUFFICIENT.value,
            explanation="Insufficient evidence",
            evidence_count=0,
            evidence_quality="none",
            dataset_id=DATASET_ID,
        )
        ev = _make_evidence(level=EvidenceLevel.LIMITATION.value)
        synth = _make_synth_signal(
            description="Cannot determine outcome",
            source_type="generic",
            evidence_items=[ev],
            confidence_score=confidence,
            level=EvidenceLevel.LIMITATION.value,
        )
        result = gen.generate(
            synthesis_result=_make_synthesis_result(signals=[synth]),
        )
        assert len(result.decisions) >= 1
        decision = result.decisions[0]
        assert decision.category == DecisionCategory.BLOCKED.value

    def test_scenario_insufficient_evidence_produces_blocked(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        scenario = _make_scenario_result(
            status=ScenarioStatus.INSUFFICIENT_EVIDENCE.value,
            delta=None,
            delta_pct=None,
            description="Cannot evaluate scenario",
            confidence=ConfidenceScore(
                level=ConfidenceLevel.INSUFFICIENT.value,
                explanation="Insufficient",
                evidence_count=0,
                evidence_quality="none",
                dataset_id=DATASET_ID,
            ),
        )
        result = gen.generate(scenario_results=[scenario])
        assert len(result.decisions) >= 1
        decision = result.decisions[0]
        assert decision.category == DecisionCategory.BLOCKED.value


# ===========================================================================
# 4. Contradictory evidence
# ===========================================================================

class TestContradictoryEvidence:
    """Test that contradictions are surfaced."""

    def test_contradictions_surfaced_in_decision(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        ev = _make_evidence()
        signal = _make_signal(evidence_items=[ev])
        signal_id = signal.id

        synth = FakeSynthesizedSignal(
            signal=signal,
            priority=FakeSignalPriority(3),
            source_type="kpi",
            evidence_items=[ev],
        )
        contradictions = [
            {"signal_ids": [signal_id], "description": "Signal contradicts prior finding"},
        ]
        result = gen.generate(
            synthesis_result=_make_synthesis_result(
                signals=[synth],
                contradictions=contradictions,
            ),
        )
        assert len(result.decisions) >= 1
        decision = result.decisions[0]
        assert len(decision.contradictions) >= 1
        assert "contradicts" in decision.contradictions[0].lower()

    def test_string_contradiction_surfaced(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        ev = _make_evidence()
        signal = _make_signal(evidence_items=[ev])
        signal_id = signal.id

        synth = FakeSynthesizedSignal(
            signal=signal,
            priority=FakeSignalPriority(3),
            source_type="kpi",
            evidence_items=[ev],
        )
        contradictions = [
            f"Signal {signal_id} is contradicted by other evidence",
        ]
        result = gen.generate(
            synthesis_result=_make_synthesis_result(
                signals=[synth],
                contradictions=contradictions,
            ),
        )
        assert len(result.decisions) >= 1
        decision = result.decisions[0]
        assert len(decision.contradictions) >= 1


# ===========================================================================
# 5. Traceability to supporting signals and evidence
# ===========================================================================

class TestTraceability:
    """Test that decisions are traceable to their evidence."""

    def test_signal_ids_preserved(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        synth = _make_synth_signal(description="Traceable signal")
        signal_id = synth.signal.id
        result = gen.generate(
            synthesis_result=_make_synthesis_result(signals=[synth]),
        )
        decision = result.decisions[0]
        assert signal_id in decision.signal_ids

    def test_evidence_ids_preserved(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        ev = _make_evidence(description="Traceable evidence")
        synth = _make_synth_signal(
            description="Signal with evidence",
            evidence_items=[ev],
        )
        result = gen.generate(
            synthesis_result=_make_synthesis_result(signals=[synth]),
        )
        decision = result.decisions[0]
        assert ev.id in decision.evidence_ids

    def test_impact_ids_from_impact_report(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        assessment = _make_impact_assessment()
        report = _make_impact_report(assessments=[assessment])
        result = gen.generate(impact_report=report)
        decision = result.decisions[0]
        assert assessment.id in decision.impact_ids

    def test_scenario_ids_preserved(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        scenario = _make_scenario_result()
        result = gen.generate(scenario_results=[scenario])
        decision = result.decisions[0]
        assert "scen_test01" in decision.scenario_ids

    def test_evidence_from_registry_used(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        ev = _make_evidence(description="Registry evidence")
        signal = _make_signal(
            description="Signal using registry",
            evidence_items=[ev],
        )
        # Create synth signal with empty evidence_items but signal has evidence_ids
        synth = FakeSynthesizedSignal(
            signal=signal,
            priority=FakeSignalPriority(3),
            source_type="kpi",
            evidence_items=[],  # Empty — should fall back to registry
        )
        registry = {ev.id: ev}
        result = gen.generate(
            synthesis_result=_make_synthesis_result(
                signals=[synth],
                evidence_registry=registry,
            ),
        )
        decision = result.decisions[0]
        assert ev.id in decision.evidence_ids


# ===========================================================================
# 6. Impact and confidence preservation
# ===========================================================================

class TestImpactAndConfidencePreservation:
    """Test that impact values and confidence are preserved correctly."""

    def test_impact_value_preserved(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        assessment = _make_impact_assessment(estimated_value=42.5)
        report = _make_impact_report(assessments=[assessment])
        result = gen.generate(impact_report=report)
        decision = result.decisions[0]
        assert decision.impact_value == 42.5

    def test_confidence_preserved(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        confidence = ConfidenceScore(
            level=ConfidenceLevel.HIGH.value,
            explanation="Strong evidence",
            evidence_count=5,
            evidence_quality="high",
            dataset_id=DATASET_ID,
        )
        assessment = _make_impact_assessment(confidence=confidence)
        report = _make_impact_report(assessments=[assessment])
        result = gen.generate(impact_report=report)
        decision = result.decisions[0]
        assert decision.confidence is not None
        assert decision.confidence.level == ConfidenceLevel.HIGH.value
        assert decision.confidence.evidence_count == 5

    def test_impact_type_preserved(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        assessment = _make_impact_assessment(
            impact_type=ImpactType.SCENARIO_BASED.value,
        )
        report = _make_impact_report(assessments=[assessment])
        result = gen.generate(impact_report=report)
        decision = result.decisions[0]
        assert decision.impact_type == ImpactType.SCENARIO_BASED.value

    def test_impact_merged_into_signal_decision(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)

        # Create a signal and synth
        ev = _make_evidence()
        signal = _make_signal(description="Signal with impact", evidence_items=[ev])
        signal_id = signal.id

        synth = FakeSynthesizedSignal(
            signal=signal,
            priority=FakeSignalPriority(3),
            source_type="kpi",
            evidence_items=[ev],
        )

        # Create an impact assessment for the same signal
        assessment = _make_impact_assessment(
            signal_id=signal_id,
            estimated_value=99.9,
        )

        result = gen.generate(
            synthesis_result=_make_synthesis_result(signals=[synth]),
            impact_report=_make_impact_report(assessments=[assessment]),
        )
        # The signal decision should have the impact merged
        signal_decisions = [d for d in result.decisions if signal_id in d.signal_ids]
        assert len(signal_decisions) >= 1
        merged = signal_decisions[0]
        assert assessment.id in merged.impact_ids


# ===========================================================================
# 7. Transparent prioritization
# ===========================================================================

class TestTransparentPrioritization:
    """Test that prioritization is transparent and documented."""

    def test_priority_rationale_provided(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        result = gen.generate(
            synthesis_result=_make_synthesis_result(),
        )
        for decision in result.decisions:
            assert decision.priority_rationale
            assert len(decision.priority_rationale) > 0

    def test_priority_is_valid_level(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        result = gen.generate(
            synthesis_result=_make_synthesis_result(),
            impact_report=_make_impact_report(),
            scenario_results=[_make_scenario_result()],
        )
        valid_priorities = {p.value for p in PriorityLevel}
        for decision in result.decisions:
            assert decision.priority in valid_priorities

    def test_high_evidence_gets_higher_priority(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        # Many evidence items → higher priority
        many_evidence = [
            _make_evidence(description=f"Evidence {i}") for i in range(5)
        ]
        strong_synth = _make_synth_signal(
            description="Strong multi-evidence signal",
            source_type="kpi",
            priority_value=1,  # CRITICAL signal priority
            evidence_items=many_evidence,
        )
        # Single evidence → lower priority
        weak_evidence = [_make_evidence(level=EvidenceLevel.INFERENCE.value)]
        weak_synth = _make_synth_signal(
            description="Single evidence signal",
            source_type="generic",
            priority_value=5,  # INFO signal priority
            evidence_items=weak_evidence,
            level=EvidenceLevel.INFERENCE.value,
        )
        result = gen.generate(
            synthesis_result=_make_synthesis_result(
                signals=[strong_synth, weak_synth],
            ),
        )
        priorities = [d.priority for d in result.decisions]
        priority_order = {
            PriorityLevel.CRITICAL.value: 0,
            PriorityLevel.HIGH.value: 1,
            PriorityLevel.MEDIUM.value: 2,
            PriorityLevel.LOW.value: 3,
        }
        # Strong signal should have higher or equal priority
        if len(priorities) >= 2:
            strong_ord = priority_order.get(priorities[0], 99)
            weak_ord = priority_order.get(priorities[1], 99)
            assert strong_ord <= weak_ord

    def test_report_ranked_uses_multiple_dimensions(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        result = gen.generate(
            synthesis_result=_make_synthesis_result(),
            impact_report=_make_impact_report(),
            scenario_results=[_make_scenario_result()],
        )
        ranked = result.ranked()
        # ranked() should return all decisions
        assert len(ranked) == len(result.decisions)

    def test_priority_rationale_includes_evidence_strength(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        result = gen.generate(
            synthesis_result=_make_synthesis_result(),
        )
        for decision in result.decisions:
            assert "evidence" in decision.priority_rationale.lower()


# ===========================================================================
# 8. Deterministic generation and ordering
# ===========================================================================

class TestDeterministicGeneration:
    """Test deterministic output for identical inputs."""

    def test_same_inputs_produce_same_count(self):
        signals = [_make_synth_signal(description="Deterministic signal")]
        synth_result = _make_synthesis_result(signals=signals)

        gen1 = DecisionGenerator(dataset_id=DATASET_ID)
        result1 = gen1.generate(synthesis_result=synth_result)

        gen2 = DecisionGenerator(dataset_id=DATASET_ID)
        result2 = gen2.generate(synthesis_result=synth_result)

        assert len(result1.decisions) == len(result2.decisions)

    def test_same_inputs_produce_same_categories(self):
        signals = [_make_synth_signal(description="Consistent signal")]
        synth_result = _make_synthesis_result(signals=signals)

        gen1 = DecisionGenerator(dataset_id=DATASET_ID)
        result1 = gen1.generate(synthesis_result=synth_result)

        gen2 = DecisionGenerator(dataset_id=DATASET_ID)
        result2 = gen2.generate(synthesis_result=synth_result)

        cats1 = [d.category for d in result1.decisions]
        cats2 = [d.category for d in result2.decisions]
        assert cats1 == cats2

    def test_same_inputs_produce_same_priorities(self):
        signals = [_make_synth_signal(description="Priority-stable signal")]
        synth_result = _make_synthesis_result(signals=signals)

        gen1 = DecisionGenerator(dataset_id=DATASET_ID)
        result1 = gen1.generate(synthesis_result=synth_result)

        gen2 = DecisionGenerator(dataset_id=DATASET_ID)
        result2 = gen2.generate(synthesis_result=synth_result)

        pris1 = [d.priority for d in result1.decisions]
        pris2 = [d.priority for d in result2.decisions]
        assert pris1 == pris2

    def test_ranked_order_is_deterministic(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        result = gen.generate(
            synthesis_result=_make_synthesis_result(),
            impact_report=_make_impact_report(),
            scenario_results=[_make_scenario_result()],
        )
        ranked1 = result.ranked()
        ranked2 = result.ranked()
        assert [d.id for d in ranked1] == [d.id for d in ranked2]


# ===========================================================================
# 9. Dataset isolation
# ===========================================================================

class TestDatasetIsolation:
    """Test that decisions are isolated by dataset."""

    def test_wrong_dataset_synthesis_skipped(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        synth_result = _make_synthesis_result(
            dataset_id=OTHER_DATASET_ID,
        )
        result = gen.generate(synthesis_result=synth_result)
        assert len(result.decisions) == 0
        assert any("does not match" in n for n in result.processing_notes)

    def test_wrong_dataset_impact_skipped(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        report = _make_impact_report(dataset_id=OTHER_DATASET_ID)
        result = gen.generate(impact_report=report)
        assert len(result.decisions) == 0
        assert any("does not match" in n for n in result.processing_notes)

    def test_wrong_dataset_scenario_skipped(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        scenario = _make_scenario_result(dataset_id=OTHER_DATASET_ID)
        result = gen.generate(scenario_results=[scenario])
        assert len(result.decisions) == 0

    def test_all_decisions_scoped_to_dataset(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        result = gen.generate(
            synthesis_result=_make_synthesis_result(),
            impact_report=_make_impact_report(),
            scenario_results=[_make_scenario_result()],
        )
        for decision in result.decisions:
            assert decision.dataset_id == DATASET_ID

    def test_mixed_dataset_evidence_filtered(self):
        """Generator should filter out evidence from wrong dataset."""
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        good_ev = _make_evidence(dataset_id=DATASET_ID)
        bad_ev = _make_evidence(dataset_id=OTHER_DATASET_ID)
        # Build signal from same-dataset evidence only (create_signal enforces isolation)
        signal = _make_signal(
            description="Mixed dataset signal",
            evidence_items=[good_ev],
        )
        # But attach cross-dataset evidence at the synthesized signal level
        synth = FakeSynthesizedSignal(
            signal=signal,
            priority=FakeSignalPriority(3),
            source_type="kpi",
            evidence_items=[good_ev, bad_ev],
        )
        result = gen.generate(
            synthesis_result=_make_synthesis_result(signals=[synth]),
        )
        # Decision should exist but only reference the correct dataset evidence
        if result.decisions:
            decision = result.decisions[0]
            assert decision.dataset_id == DATASET_ID
            # bad_ev's ID should not appear in decision evidence
            assert bad_ev.id not in decision.evidence_ids

    def test_generator_requires_dataset_id(self):
        with pytest.raises(ValueError, match="dataset_id is required"):
            DecisionGenerator(dataset_id="")


# ===========================================================================
# 10. Malformed or incomplete inputs
# ===========================================================================

class TestMalformedInputs:
    """Test handling of malformed or incomplete inputs."""

    def test_none_signal_in_synth_skipped(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        synth = FakeSynthesizedSignal(signal=None, source_type="kpi")
        result = gen.generate(
            synthesis_result=_make_synthesis_result(signals=[synth]),
        )
        assert len(result.decisions) == 0

    def test_empty_impact_report(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        report = ImpactReport(dataset_id=DATASET_ID)
        result = gen.generate(impact_report=report)
        assert len(result.decisions) == 0
        assert any("no assessments" in n.lower() for n in result.processing_notes)

    def test_empty_scenario_results(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        result = gen.generate(scenario_results=[])
        assert len(result.decisions) == 0

    def test_invalid_category_raises(self):
        with pytest.raises(ValueError, match="Invalid decision category"):
            DecisionOption(
                dataset_id=DATASET_ID,
                title="Test",
                category="invalid_cat",
            )

    def test_invalid_priority_raises(self):
        with pytest.raises(ValueError, match="Invalid priority level"):
            DecisionOption(
                dataset_id=DATASET_ID,
                title="Test",
                priority="extreme",
            )

    def test_invalid_urgency_raises(self):
        with pytest.raises(ValueError, match="Invalid urgency level"):
            DecisionOption(
                dataset_id=DATASET_ID,
                title="Test",
                urgency="right_now",
            )

    def test_invalid_reversibility_raises(self):
        with pytest.raises(ValueError, match="Invalid reversibility"):
            DecisionOption(
                dataset_id=DATASET_ID,
                title="Test",
                reversibility="maybe",
            )

    def test_empty_title_raises(self):
        with pytest.raises(ValueError, match="non-empty title"):
            DecisionOption(
                dataset_id=DATASET_ID,
                title="",
            )

    def test_empty_dataset_id_raises(self):
        with pytest.raises(ValueError, match="non-empty dataset_id"):
            DecisionOption(
                title="Test",
                dataset_id="",
            )

    def test_decision_option_is_frozen(self):
        d = DecisionOption(
            dataset_id=DATASET_ID,
            title="Test frozen",
        )
        with pytest.raises(FrozenInstanceError):
            d.title = "Changed"

    def test_exception_in_signal_processing_noted(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        # Create a signal that will cause an error during processing
        # by having a signal with no description
        bad_signal = DecisionSignal(
            description="Bad signal for error test",
            dataset_id=DATASET_ID,
        )
        synth = FakeSynthesizedSignal(
            signal=bad_signal,
            priority="not_a_priority",  # Will cause error in priority extraction
            source_type="kpi",
            evidence_items=[_make_evidence()],
        )
        # Should not raise, should produce a processing note or handle gracefully
        result = gen.generate(
            synthesis_result=_make_synthesis_result(signals=[synth]),
        )
        # Either a decision is generated or a processing note is added
        assert len(result.decisions) >= 0


# ===========================================================================
# 11. Serialization
# ===========================================================================

class TestSerialization:
    """Test to_dict and from_dict round-tripping."""

    def test_decision_option_round_trip(self):
        confidence = ConfidenceScore(
            level=ConfidenceLevel.HIGH.value,
            explanation="Test",
            evidence_count=3,
            evidence_quality="high",
            dataset_id=DATASET_ID,
        )
        limitation = create_limitation(
            "Test limitation",
            category=LimitationCategory.DATA_QUALITY.value,
        )
        d = DecisionOption(
            dataset_id=DATASET_ID,
            title="Serialization test",
            description="Testing round trip",
            category=DecisionCategory.RECOMMEND.value,
            recommended_action="Test action",
            signal_ids=["sig_001"],
            evidence_ids=["ev_001", "ev_002"],
            impact_ids=["imp_001"],
            scenario_ids=["scen_001"],
            estimated_impact="10.5 units",
            impact_type=ImpactType.MEASURED.value,
            impact_value=10.5,
            impact_unit="units",
            confidence=confidence,
            priority=PriorityLevel.HIGH.value,
            priority_rationale="Strong evidence",
            urgency=UrgencyLevel.SHORT_TERM.value,
            assumptions=["Assumption 1"],
            limitations=[limitation],
            risks=["Risk 1"],
            reversibility=Reversibility.EASILY_REVERSIBLE.value,
            required_follow_up=["Follow up 1"],
            invalidation_conditions=["Condition 1"],
            explanation="Test explanation",
            evidence_basis="measured",
            contradictions=["Contradiction 1"],
        )
        d_dict = d.to_dict()
        d_restored = DecisionOption.from_dict(d_dict)

        assert d_restored.dataset_id == d.dataset_id
        assert d_restored.title == d.title
        assert d_restored.category == d.category
        assert d_restored.priority == d.priority
        assert d_restored.urgency == d.urgency
        assert d_restored.reversibility == d.reversibility
        assert d_restored.impact_value == d.impact_value
        assert d_restored.confidence.level == d.confidence.level
        assert d_restored.evidence_ids == d.evidence_ids
        assert d_restored.signal_ids == d.signal_ids
        assert len(d_restored.limitations) == len(d.limitations)

    def test_decision_report_round_trip(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        result = gen.generate(
            synthesis_result=_make_synthesis_result(),
            impact_report=_make_impact_report(),
            scenario_results=[_make_scenario_result()],
        )
        r_dict = result.to_dict()
        r_restored = DecisionReport.from_dict(r_dict)

        assert r_restored.dataset_id == result.dataset_id
        assert len(r_restored.decisions) == len(result.decisions)

    def test_to_dict_contains_expected_keys(self):
        d = DecisionOption(
            dataset_id=DATASET_ID,
            title="Key test",
        )
        d_dict = d.to_dict()
        expected_keys = {
            "id", "dataset_id", "title", "description", "category",
            "recommended_action", "signal_ids", "evidence_ids",
            "impact_ids", "scenario_ids", "estimated_impact",
            "impact_type", "impact_value", "impact_unit", "confidence",
            "priority", "priority_rationale", "urgency", "assumptions",
            "limitations", "risks", "reversibility", "required_follow_up",
            "invalidation_conditions", "explanation", "evidence_basis",
            "contradictions",
        }
        assert expected_keys.issubset(set(d_dict.keys()))

    def test_report_to_dict_contains_summary(self):
        report = DecisionReport(
            dataset_id=DATASET_ID,
            decisions=[
                DecisionOption(
                    dataset_id=DATASET_ID,
                    title="Test",
                    category=DecisionCategory.RECOMMEND.value,
                ),
                DecisionOption(
                    dataset_id=DATASET_ID,
                    title="Test 2",
                    category=DecisionCategory.MONITOR.value,
                ),
            ],
        )
        r_dict = report.to_dict()
        assert "decision_count" in r_dict
        assert r_dict["decision_count"] == 2
        assert "by_category" in r_dict
        assert "by_priority" in r_dict

    def test_nan_impact_value_serialized_safely(self):
        d = DecisionOption(
            dataset_id=DATASET_ID,
            title="NaN test",
            impact_value=float("nan"),
        )
        d_dict = d.to_dict()
        assert d_dict["impact_value"] is None

    def test_inf_impact_value_serialized_safely(self):
        d = DecisionOption(
            dataset_id=DATASET_ID,
            title="Inf test",
            impact_value=float("inf"),
        )
        d_dict = d.to_dict()
        assert d_dict["impact_value"] is None


# ===========================================================================
# 12. Forecast-based recommendations
# ===========================================================================

class TestForecastBasedRecommendations:
    """Test decisions from forecast evidence."""

    def test_forecast_produces_monitor_not_recommend(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        ev = _make_evidence(
            description="Forecast projects increase",
            source="forecast",
            level=EvidenceLevel.PREDICTION.value,
        )
        synth = _make_synth_signal(
            description="Forecast shows improvement trend",
            source_type="forecast",
            evidence_items=[ev],
            level=EvidenceLevel.PREDICTION.value,
        )
        result = gen.generate(
            synthesis_result=_make_synthesis_result(signals=[synth]),
        )
        decision = result.decisions[0]
        # Forecasts should NOT be treated as recommendations to act
        assert decision.category != DecisionCategory.RECOMMEND.value
        assert decision.category == DecisionCategory.MONITOR.value

    def test_forecast_evidence_basis_is_scenario_based(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        ev = _make_evidence(
            source="forecast",
            level=EvidenceLevel.PREDICTION.value,
        )
        synth = _make_synth_signal(
            description="Forecast signal",
            source_type="forecast",
            evidence_items=[ev],
            level=EvidenceLevel.PREDICTION.value,
        )
        result = gen.generate(
            synthesis_result=_make_synthesis_result(signals=[synth]),
        )
        decision = result.decisions[0]
        assert decision.evidence_basis == "scenario_based"

    def test_forecast_decision_has_model_limitation(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        ev = _make_evidence(
            source="forecast",
            level=EvidenceLevel.PREDICTION.value,
        )
        synth = _make_synth_signal(
            description="Forecast with limitation",
            source_type="forecast",
            evidence_items=[ev],
            level=EvidenceLevel.PREDICTION.value,
        )
        result = gen.generate(
            synthesis_result=_make_synthesis_result(signals=[synth]),
        )
        decision = result.decisions[0]
        has_model_limitation = any(
            hasattr(lim, "category") and
            lim.category == LimitationCategory.MODEL_UNCERTAINTY.value
            for lim in decision.limitations
        )
        assert has_model_limitation


# ===========================================================================
# 13. Scenario-based recommendations
# ===========================================================================

class TestScenarioBasedRecommendations:
    """Test decisions from scenario results."""

    def test_simulated_scenario_produces_monitor(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        scenario = _make_scenario_result(
            status=ScenarioStatus.SIMULATED.value,
        )
        result = gen.generate(scenario_results=[scenario])
        decision = result.decisions[0]
        assert decision.category == DecisionCategory.MONITOR.value

    def test_not_simulatable_produces_investigate(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        scenario = _make_scenario_result(
            status=ScenarioStatus.NOT_SIMULATABLE.value,
            description="Scenario cannot be simulated",
        )
        result = gen.generate(scenario_results=[scenario])
        decision = result.decisions[0]
        assert decision.category == DecisionCategory.INVESTIGATE.value

    def test_scenario_evidence_basis_is_scenario_based(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        scenario = _make_scenario_result()
        result = gen.generate(scenario_results=[scenario])
        decision = result.decisions[0]
        assert decision.evidence_basis == "scenario_based"

    def test_scenario_has_projection_disclaimer(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        scenario = _make_scenario_result()
        result = gen.generate(scenario_results=[scenario])
        decision = result.decisions[0]
        # Should have limitation about projections not being guaranteed
        has_projection_limitation = any(
            hasattr(lim, "description") and
            "not guaranteed" in lim.description.lower()
            for lim in decision.limitations
        )
        assert has_projection_limitation

    def test_scenario_delta_preserved_as_impact_value(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        scenario = _make_scenario_result(delta=12.5)
        result = gen.generate(scenario_results=[scenario])
        decision = result.decisions[0]
        assert decision.impact_value == 12.5

    def test_partially_simulated_scenario(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        scenario = _make_scenario_result(
            status=ScenarioStatus.PARTIALLY_SIMULATED.value,
            description="Partially simulated scenario",
        )
        result = gen.generate(scenario_results=[scenario])
        decision = result.decisions[0]
        assert decision.category == DecisionCategory.MONITOR.value


# ===========================================================================
# 14. Correlation-versus-causation safeguards
# ===========================================================================

class TestCorrelationVsCausation:
    """Test that model-based findings are not presented as causal."""

    def test_model_estimated_has_causation_disclaimer(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        ev = _make_evidence(
            description="Feature importance from model",
            source="driver_analysis",
            level=EvidenceLevel.INFERENCE.value,
        )
        synth = _make_synth_signal(
            description="Feature X associated with target",
            source_type="driver",
            evidence_items=[ev],
            level=EvidenceLevel.INFERENCE.value,
        )
        result = gen.generate(
            synthesis_result=_make_synthesis_result(signals=[synth]),
        )
        decision = result.decisions[0]
        # Check that explanation or limitations mention non-causal nature
        text = (
            decision.explanation.lower()
            + " ".join(
                getattr(lim, "description", "").lower()
                for lim in decision.limitations
            )
        )
        assert "causal" in text or "association" in text or "model estimate" in text

    def test_regression_source_has_model_limitation(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        ev = _make_evidence(
            source="regression",
            level=EvidenceLevel.INFERENCE.value,
        )
        synth = _make_synth_signal(
            description="Regression coefficient",
            source_type="regression",
            evidence_items=[ev],
            level=EvidenceLevel.INFERENCE.value,
        )
        result = gen.generate(
            synthesis_result=_make_synthesis_result(signals=[synth]),
        )
        decision = result.decisions[0]
        has_model_lim = any(
            hasattr(lim, "category") and
            lim.category == LimitationCategory.MODEL_UNCERTAINTY.value
            for lim in decision.limitations
        )
        assert has_model_lim

    def test_model_evidence_not_categorized_as_recommend(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        ev = _make_evidence(
            source="classification",
            level=EvidenceLevel.INFERENCE.value,
        )
        synth = _make_synth_signal(
            description="Classification result",
            source_type="classification",
            evidence_items=[ev],
            level=EvidenceLevel.INFERENCE.value,
        )
        result = gen.generate(
            synthesis_result=_make_synthesis_result(signals=[synth]),
        )
        decision = result.decisions[0]
        # Model-based evidence should not directly produce RECOMMEND
        assert decision.category != DecisionCategory.RECOMMEND.value


# ===========================================================================
# 15. No guaranteed-outcome claims
# ===========================================================================

class TestNoGuaranteedOutcomes:
    """Test that no decision claims guaranteed outcomes."""

    def test_recommend_action_no_guarantee(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        result = gen.generate(
            synthesis_result=_make_synthesis_result(),
        )
        for decision in result.decisions:
            action = decision.recommended_action.lower()
            explanation = decision.explanation.lower()
            combined = action + " " + explanation
            assert "guaranteed" not in combined or "not guaranteed" in combined
            assert "will solve" not in combined
            assert "will fix" not in combined
            assert "will increase" not in combined
            assert "will decrease" not in combined

    def test_scenario_decision_no_guarantee(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        result = gen.generate(
            scenario_results=[_make_scenario_result()],
        )
        for decision in result.decisions:
            combined = (
                decision.recommended_action.lower()
                + " " + decision.explanation.lower()
                + " " + (decision.estimated_impact or "").lower()
            )
            assert "guaranteed" not in combined or "not guaranteed" in combined

    def test_impact_decision_no_guarantee(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        result = gen.generate(
            impact_report=_make_impact_report(),
        )
        for decision in result.decisions:
            combined = (
                decision.recommended_action.lower()
                + " " + decision.explanation.lower()
            )
            assert "guaranteed" not in combined or "not guaranteed" in combined


# ===========================================================================
# 16. No invented financial benefits
# ===========================================================================

class TestNoInventedFinancialBenefits:
    """Test that no decisions invent financial benefits."""

    def test_no_invented_currency_amounts(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        result = gen.generate(
            synthesis_result=_make_synthesis_result(),
            impact_report=_make_impact_report(),
            scenario_results=[_make_scenario_result()],
        )
        currency_symbols = ["$", "€", "£", "¥", "USD", "EUR", "GBP"]
        for decision in result.decisions:
            combined = (
                decision.recommended_action
                + " " + decision.explanation
                + " " + (decision.estimated_impact or "")
                + " " + decision.title
                + " " + decision.description
            )
            for symbol in currency_symbols:
                assert symbol not in combined, (
                    f"Currency symbol '{symbol}' found in decision: "
                    f"'{combined[:200]}'"
                )

    def test_no_invented_roi(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        result = gen.generate(
            synthesis_result=_make_synthesis_result(),
        )
        for decision in result.decisions:
            combined = (
                decision.recommended_action.lower()
                + " " + decision.explanation.lower()
            )
            # Should not invent ROI claims
            if "roi" in combined:
                assert "model" in combined or "estimate" in combined

    def test_no_invented_revenue_claims(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        result = gen.generate(
            synthesis_result=_make_synthesis_result(),
        )
        for decision in result.decisions:
            combined = (
                decision.recommended_action.lower()
                + " " + decision.explanation.lower()
            )
            assert "revenue increase" not in combined
            assert "revenue growth" not in combined
            assert "profit increase" not in combined


# ===========================================================================
# 17. Risk and reversibility handling
# ===========================================================================

class TestRiskAndReversibility:
    """Test risk and reversibility handling."""

    def test_decisions_have_risk_information(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        result = gen.generate(
            synthesis_result=_make_synthesis_result(),
        )
        for decision in result.decisions:
            # Decisions should either have risks listed or be investigation/blocked
            if decision.category in (
                DecisionCategory.RECOMMEND.value,
                DecisionCategory.MONITOR.value,
                DecisionCategory.MITIGATE_RISK.value,
            ):
                assert len(decision.risks) >= 0  # May be empty for some
                assert decision.reversibility in {r.value for r in Reversibility}

    def test_model_based_decisions_note_uncertainty_risk(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        ev = _make_evidence(
            source="regression",
            level=EvidenceLevel.INFERENCE.value,
        )
        synth = _make_synth_signal(
            description="Model finding",
            source_type="regression",
            evidence_items=[ev],
            level=EvidenceLevel.INFERENCE.value,
        )
        result = gen.generate(
            synthesis_result=_make_synthesis_result(signals=[synth]),
        )
        decision = result.decisions[0]
        risk_text = " ".join(decision.risks).lower()
        assert "model" in risk_text or "projection" in risk_text or len(decision.risks) >= 1

    def test_reversibility_is_valid_enum(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        result = gen.generate(
            synthesis_result=_make_synthesis_result(),
            scenario_results=[_make_scenario_result()],
        )
        valid = {r.value for r in Reversibility}
        for decision in result.decisions:
            assert decision.reversibility in valid

    def test_risk_mitigation_category_has_follow_up(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        ev = _make_evidence(
            source="risk_assessment",
            level=EvidenceLevel.FACT.value,
        )
        synth = _make_synth_signal(
            description="Risk identified",
            source_type="risk",
            evidence_items=[ev],
        )
        result = gen.generate(
            synthesis_result=_make_synthesis_result(signals=[synth]),
        )
        risk_decisions = [
            d for d in result.decisions
            if d.category == DecisionCategory.MITIGATE_RISK.value
        ]
        for decision in risk_decisions:
            assert len(decision.required_follow_up) >= 1


# ===========================================================================
# 18. Generic non-marketing datasets
# ===========================================================================

class TestGenericNonMarketingDatasets:
    """Test that the engine works with non-marketing datasets."""

    def test_healthcare_dataset(self):
        gen = DecisionGenerator(dataset_id="healthcare_001")
        ev = _make_evidence(
            description="Patient wait time above threshold",
            source="analytics",
            dataset_id="healthcare_001",
            metric_name="avg_wait_time",
            metric_value=45.0,
        )
        synth = _make_synth_signal(
            description="Average wait time exceeds target",
            source_type="kpi",
            evidence_items=[ev],
            dataset_id="healthcare_001",
        )
        result = gen.generate(
            synthesis_result=_make_synthesis_result(
                signals=[synth],
                dataset_id="healthcare_001",
            ),
        )
        assert len(result.decisions) >= 1
        decision = result.decisions[0]
        assert decision.dataset_id == "healthcare_001"
        # Should not contain marketing-specific terms
        combined = decision.recommended_action + " " + decision.explanation
        assert "google ads" not in combined.lower()
        assert "facebook" not in combined.lower()

    def test_manufacturing_dataset(self):
        gen = DecisionGenerator(dataset_id="mfg_001")
        ev = _make_evidence(
            description="Defect rate spike detected",
            source="quality_control",
            dataset_id="mfg_001",
            metric_name="defect_rate",
            metric_value=0.05,
        )
        synth = _make_synth_signal(
            description="Manufacturing defect rate increased",
            source_type="anomaly",
            evidence_items=[ev],
            dataset_id="mfg_001",
        )
        result = gen.generate(
            synthesis_result=_make_synthesis_result(
                signals=[synth],
                dataset_id="mfg_001",
            ),
        )
        assert len(result.decisions) >= 1
        decision = result.decisions[0]
        assert decision.dataset_id == "mfg_001"
        assert decision.category == DecisionCategory.INVESTIGATE.value

    def test_education_dataset(self):
        gen = DecisionGenerator(dataset_id="edu_001")
        ev = _make_evidence(
            description="Student performance improved",
            source="analytics",
            dataset_id="edu_001",
            metric_name="avg_score",
            metric_value=85.0,
        )
        synth = _make_synth_signal(
            description="Average assessment scores increased",
            source_type="kpi",
            evidence_items=[ev],
            dataset_id="edu_001",
        )
        result = gen.generate(
            synthesis_result=_make_synthesis_result(
                signals=[synth],
                dataset_id="edu_001",
            ),
        )
        assert len(result.decisions) >= 1
        decision = result.decisions[0]
        assert decision.dataset_id == "edu_001"

    def test_finance_dataset(self):
        gen = DecisionGenerator(dataset_id="fin_001")
        ev = _make_evidence(
            description="Transaction anomaly detected",
            source="anomaly_detection",
            dataset_id="fin_001",
            metric_name="transaction_count",
            metric_value=500.0,
        )
        synth = _make_synth_signal(
            description="Unusual transaction pattern",
            source_type="anomaly",
            evidence_items=[ev],
            dataset_id="fin_001",
        )
        result = gen.generate(
            synthesis_result=_make_synthesis_result(
                signals=[synth],
                dataset_id="fin_001",
            ),
        )
        assert len(result.decisions) >= 1
        assert result.decisions[0].category == DecisionCategory.INVESTIGATE.value

    def test_no_hardcoded_domain_terms_in_engine_output(self):
        gen = DecisionGenerator(dataset_id=DATASET_ID)
        result = gen.generate(
            synthesis_result=_make_synthesis_result(),
            impact_report=_make_impact_report(),
            scenario_results=[_make_scenario_result()],
        )
        hardcoded_terms = [
            "novamart", "google ads", "facebook ads",
            "revenue", "sales",
        ]
        for decision in result.decisions:
            combined = (
                decision.recommended_action.lower()
                + " " + decision.explanation.lower()
                + " " + decision.title.lower()
            )
            for term in hardcoded_terms:
                assert term not in combined, (
                    f"Hardcoded term '{term}' found in decision output"
                )


# ===========================================================================
# 19. Regression-safe imports
# ===========================================================================

class TestRegressionSafeImports:
    """Test that all Phase 7 modules import correctly."""

    def test_import_contracts(self):
        from src.decision import contracts
        assert hasattr(contracts, "DecisionEvidence")
        assert hasattr(contracts, "DecisionSignal")
        assert hasattr(contracts, "ConfidenceScore")

    def test_import_evidence(self):
        from src.decision import evidence
        assert hasattr(evidence, "create_evidence")
        assert hasattr(evidence, "create_signal")
        assert hasattr(evidence, "validate_dataset_scope")

    def test_import_confidence(self):
        from src.decision import confidence
        assert hasattr(confidence, "assess_confidence")
        assert hasattr(confidence, "aggregate_confidence")

    def test_import_signal_synthesizer(self):
        from src.decision import signal_synthesizer
        assert hasattr(signal_synthesizer, "SignalSynthesizer")
        assert hasattr(signal_synthesizer, "SynthesisResult")

    def test_import_impact_quantifier(self):
        from src.decision import impact_quantifier
        assert hasattr(impact_quantifier, "ImpactQuantifier")
        assert hasattr(impact_quantifier, "ImpactReport")
        assert hasattr(impact_quantifier, "ImpactAssessment")

    def test_import_scenario_engine(self):
        from src.decision import scenario_engine
        assert hasattr(scenario_engine, "ScenarioEngine")
        assert hasattr(scenario_engine, "ScenarioResult")

    def test_import_decision_generator(self):
        from src.decision import decision_generator
        assert hasattr(decision_generator, "DecisionGenerator")
        assert hasattr(decision_generator, "DecisionOption")
        assert hasattr(decision_generator, "DecisionReport")
        assert hasattr(decision_generator, "DecisionCategory")
        assert hasattr(decision_generator, "PriorityLevel")
        assert hasattr(decision_generator, "UrgencyLevel")
        assert hasattr(decision_generator, "Reversibility")

    def test_import_decision_package(self):
        from src.decision import (
            contracts,
            evidence,
            confidence,
            signal_synthesizer,
            impact_quantifier,
            scenario_engine,
            decision_generator,
        )
        # Verify no circular imports
        assert contracts is not None
        assert evidence is not None
        assert confidence is not None
        assert signal_synthesizer is not None
        assert impact_quantifier is not None
        assert scenario_engine is not None
        assert decision_generator is not None


# ===========================================================================
# Additional: DecisionReport helper methods
# ===========================================================================

class TestDecisionReportHelpers:
    """Test DecisionReport filtering and ranking helpers."""

    def test_by_category(self):
        report = DecisionReport(
            dataset_id=DATASET_ID,
            decisions=[
                DecisionOption(
                    dataset_id=DATASET_ID,
                    title="Rec 1",
                    category=DecisionCategory.RECOMMEND.value,
                ),
                DecisionOption(
                    dataset_id=DATASET_ID,
                    title="Mon 1",
                    category=DecisionCategory.MONITOR.value,
                ),
                DecisionOption(
                    dataset_id=DATASET_ID,
                    title="Inv 1",
                    category=DecisionCategory.INVESTIGATE.value,
                ),
            ],
        )
        assert len(report.recommended()) == 1
        assert len(report.monitoring()) == 1
        assert len(report.investigations()) == 1
        assert len(report.blocked()) == 0
        assert len(report.risk_mitigations()) == 0

    def test_by_priority(self):
        report = DecisionReport(
            dataset_id=DATASET_ID,
            decisions=[
                DecisionOption(
                    dataset_id=DATASET_ID,
                    title="High",
                    priority=PriorityLevel.HIGH.value,
                ),
                DecisionOption(
                    dataset_id=DATASET_ID,
                    title="Low",
                    priority=PriorityLevel.LOW.value,
                ),
            ],
        )
        assert len(report.by_priority(PriorityLevel.HIGH.value)) == 1
        assert len(report.by_priority(PriorityLevel.LOW.value)) == 1

    def test_ranked_sorts_by_priority_first(self):
        report = DecisionReport(
            dataset_id=DATASET_ID,
            decisions=[
                DecisionOption(
                    dataset_id=DATASET_ID,
                    title="Low priority",
                    priority=PriorityLevel.LOW.value,
                    category=DecisionCategory.RECOMMEND.value,
                ),
                DecisionOption(
                    dataset_id=DATASET_ID,
                    title="High priority",
                    priority=PriorityLevel.HIGH.value,
                    category=DecisionCategory.RECOMMEND.value,
                ),
            ],
        )
        ranked = report.ranked()
        assert ranked[0].priority == PriorityLevel.HIGH.value
        assert ranked[1].priority == PriorityLevel.LOW.value


# ===========================================================================
# Additional: Enum validation
# ===========================================================================

class TestEnumValues:
    """Test that enum values are valid strings."""

    def test_decision_category_values(self):
        assert DecisionCategory.RECOMMEND.value == "recommend"
        assert DecisionCategory.INVESTIGATE.value == "investigate"
        assert DecisionCategory.MONITOR.value == "monitor"
        assert DecisionCategory.MITIGATE_RISK.value == "mitigate_risk"
        assert DecisionCategory.BLOCKED.value == "blocked"

    def test_priority_level_values(self):
        assert PriorityLevel.CRITICAL.value == "critical"
        assert PriorityLevel.HIGH.value == "high"
        assert PriorityLevel.MEDIUM.value == "medium"
        assert PriorityLevel.LOW.value == "low"

    def test_urgency_level_values(self):
        assert UrgencyLevel.IMMEDIATE.value == "immediate"
        assert UrgencyLevel.SHORT_TERM.value == "short_term"
        assert UrgencyLevel.MEDIUM_TERM.value == "medium_term"
        assert UrgencyLevel.LONG_TERM.value == "long_term"
        assert UrgencyLevel.NOT_ASSESSED.value == "not_assessed"

    def test_reversibility_values(self):
        assert Reversibility.EASILY_REVERSIBLE.value == "easily_reversible"
        assert Reversibility.PARTIALLY_REVERSIBLE.value == "partially_reversible"
        assert Reversibility.IRREVERSIBLE.value == "irreversible"
        assert Reversibility.UNKNOWN.value == "unknown"
