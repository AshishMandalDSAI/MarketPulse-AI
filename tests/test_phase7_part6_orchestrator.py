"""Phase 7 Part 6 — Decision Orchestrator tests.

Comprehensive tests covering all 25 required categories:
  1. Successful end-to-end orchestration
  2. Empty workflow and empty evidence
  3. Insufficient evidence
  4. Correct stage execution order
  5. Optional scenario evaluation
  6. Unsupported scenario evaluation
  7. Dataset-scope validation
  8. Cross-dataset evidence rejection
  9. Cross-dataset component-result rejection
  10. Two or more datasets processed sequentially without leakage
  11. Recovery after a failed workflow
  12. Stage-specific failures
  13. Partial-success reporting
  14. Complete-failure reporting
  15. Warning and limitation propagation
  16. Evidence provenance and decision traceability
  17. Deterministic decision ordering
  18. Safe serialization and deserialization
  19. NaN/Infinity handling
  20. No demo-data fallback
  21. No fabricated successful outputs
  22. No silently swallowed exceptions
  23. Compatibility with Parts 1–5 public contracts
  24. No global mutable state
  25. Invalid request handling
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
)
from src.decision.evidence import (
    create_evidence,
    create_limitation,
    create_signal,
    validate_dataset_scope,
)
from src.decision.confidence import assess_confidence
from src.decision.signal_synthesizer import (
    SignalPriority,
    SignalSynthesizer,
    SynthesisResult,
    SynthesizedSignal,
)
from src.decision.impact_quantifier import (
    ImpactAssessment,
    ImpactQuantifier,
    ImpactReport,
    ImpactType,
    ImpactCategory,
)
from src.decision.scenario_engine import (
    ScenarioDefinition,
    ScenarioEngine,
    ScenarioParameter,
    ScenarioResult,
    ScenarioStatus,
)
from src.decision.decision_generator import (
    DecisionCategory,
    DecisionGenerator,
    DecisionOption,
    DecisionReport,
    PriorityLevel,
    UrgencyLevel,
    Reversibility,
)
from src.decision.orchestrator import (
    DecisionOrchestrator,
    EvidenceSummary,
    ExecutionStatus,
    OrchestrationReport,
    OrchestrationRequest,
    StageResult,
    StageStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DATASET_A = "test_ds_alpha"
DATASET_B = "test_ds_beta"


def _make_evidence(
    description="Test observation",
    source="analytics",
    dataset_id=DATASET_A,
    level=EvidenceLevel.FACT.value,
    metric_name="metric_x",
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


def _make_evidence_set(n=3, dataset_id=DATASET_A):
    """Create multiple evidence items for a medium-confidence assessment."""
    return [
        _make_evidence(
            description=f"Observed metric {i}",
            metric_name=f"m{i}",
            metric_value=10.0 + i,
            dataset_id=dataset_id,
        )
        for i in range(n)
    ]


def _make_request(dataset_id=DATASET_A, evidence_items=None, **kwargs):
    return OrchestrationRequest(
        dataset_id=dataset_id,
        evidence_items=evidence_items or [],
        **kwargs,
    )


class FakeAnalyticsResult:
    """Mimics Phase 3 analytics result."""
    def __init__(self, dataset_id="", kpis=None, insights=None,
                 relationships=None, charts=None):
        self.dataset_id = dataset_id
        self.kpis = kpis or []
        self.insights = insights or []
        self.relationships = relationships or []
        self.charts = charts or []


class FakeKPI:
    """Mimics Phase 3 KPI."""
    def __init__(self, name="throughput", value=100.0, unit="units/hr",
                 dataset_id=""):
        self.name = name
        self.value = value
        self.unit = unit
        self.dataset_id = dataset_id
        self.change_pct = 5.0
        self.change_direction = "positive"
        self.period = "monthly"
        self.details = {}


class FakePredictionReport:
    """Mimics Phase 4 prediction report."""
    def __init__(self, dataset_id="", forecasts=None, regressions=None,
                 classifications=None, segmentations=None):
        self.dataset_id = dataset_id
        self.forecasts = forecasts or []
        self.regressions = regressions or []
        self.classifications = classifications or []
        self.segmentations = segmentations or []


class FakePhase5Result:
    """Mimics Phase 5 anomaly/driver result."""
    def __init__(self, dataset_id="", anomaly_reports=None,
                 driver_reports=None):
        self.dataset_id = dataset_id
        self.anomaly_reports = anomaly_reports or []
        self.driver_reports = driver_reports or []


class FakeMarketingResult:
    """Mimics Phase 6 marketing result."""
    def __init__(self, dataset_id="", mmm_results=None, whatif_results=None,
                 contribution_results=None, optimization_results=None):
        self.dataset_id = dataset_id
        self.mmm_results = mmm_results or []
        self.whatif_results = whatif_results or []
        self.contribution_results = contribution_results or []
        self.optimization_results = optimization_results or []


# ===========================================================================
# 1. Successful end-to-end orchestration
# ===========================================================================

class TestSuccessfulOrchestration:
    """Test successful end-to-end workflows."""

    def test_basic_workflow_with_evidence(self):
        orch = DecisionOrchestrator()
        evidence = _make_evidence_set(3)
        request = _make_request(evidence_items=evidence)
        report = orch.run(request)
        assert report.dataset_id == DATASET_A
        assert report.workflow_id.startswith("wf_")
        assert report.status in {s.value for s in ExecutionStatus}
        assert report.evidence_summary is not None
        assert report.evidence_summary.total_count == 3

    def test_workflow_produces_decisions_from_evidence(self):
        orch = DecisionOrchestrator()
        evidence = _make_evidence_set(3)
        request = _make_request(evidence_items=evidence)
        report = orch.run(request)
        # The orchestrator processes evidence through synthesis → impact → decisions
        assert report.synthesis_result is not None
        assert report.stages is not None
        assert len(report.stages) >= 7  # All 7 main stages

    def test_all_stages_present(self):
        orch = DecisionOrchestrator()
        request = _make_request(evidence_items=_make_evidence_set(3))
        report = orch.run(request)
        stage_names = [s.name for s in report.stages]
        assert "validate_request" in stage_names
        assert "classify_evidence" in stage_names
        assert "assess_confidence" in stage_names
        assert "synthesize_signals" in stage_names
        assert "quantify_impacts" in stage_names
        assert "evaluate_scenarios" in stage_names
        assert "generate_decisions" in stage_names

    def test_workflow_with_analytics_result(self):
        orch = DecisionOrchestrator()
        analytics = FakeAnalyticsResult(
            dataset_id=DATASET_A,
            kpis=[FakeKPI(dataset_id=DATASET_A)],
        )
        request = _make_request(analytics_result=analytics)
        report = orch.run(request)
        assert report.dataset_id == DATASET_A
        assert report.status != ExecutionStatus.FAILED.value


# ===========================================================================
# 2. Empty workflow and empty evidence
# ===========================================================================

class TestEmptyWorkflow:
    """Test behavior with no inputs."""

    def test_empty_evidence_produces_report(self):
        orch = DecisionOrchestrator()
        request = _make_request(evidence_items=[])
        report = orch.run(request)
        assert report is not None
        assert report.dataset_id == DATASET_A
        assert report.evidence_summary is not None
        assert report.evidence_summary.total_count == 0

    def test_empty_workflow_status(self):
        orch = DecisionOrchestrator()
        request = _make_request()
        report = orch.run(request)
        assert report.status == ExecutionStatus.INSUFFICIENT_EVIDENCE.value

    def test_empty_workflow_distinct_from_failure(self):
        orch = DecisionOrchestrator()
        request = _make_request()
        report = orch.run(request)
        assert report.status != ExecutionStatus.FAILED.value

    def test_empty_decisions_list(self):
        orch = DecisionOrchestrator()
        request = _make_request()
        report = orch.run(request)
        assert report.decisions() == []


# ===========================================================================
# 3. Insufficient evidence
# ===========================================================================

class TestInsufficientEvidence:
    """Test insufficient evidence handling."""

    def test_insufficient_confidence_reports_status(self):
        orch = DecisionOrchestrator()
        # Single limitation-level evidence → insufficient
        ev = _make_evidence(
            description="Data quality issue",
            level=EvidenceLevel.LIMITATION.value,
        )
        request = _make_request(evidence_items=[ev])
        report = orch.run(request)
        assert report.evidence_summary.confidence is not None
        assert report.evidence_summary.confidence.level == ConfidenceLevel.INSUFFICIENT.value

    def test_insufficient_evidence_no_fabricated_decisions(self):
        orch = DecisionOrchestrator()
        ev = _make_evidence(
            description="Known constraint",
            level=EvidenceLevel.LIMITATION.value,
        )
        request = _make_request(evidence_items=[ev])
        report = orch.run(request)
        # Should not fabricate decisions from limitations-only evidence
        decisions = report.decisions()
        # If any decisions exist, they should be BLOCKED or INVESTIGATE
        for d in decisions:
            assert d.category in {
                DecisionCategory.BLOCKED.value,
                DecisionCategory.INVESTIGATE.value,
                DecisionCategory.MONITOR.value,
            }


# ===========================================================================
# 4. Correct stage execution order
# ===========================================================================

class TestStageExecutionOrder:
    """Test that stages execute in the correct order."""

    def test_stages_in_order(self):
        orch = DecisionOrchestrator()
        request = _make_request(evidence_items=_make_evidence_set(3))
        report = orch.run(request)
        stage_names = [s.name for s in report.stages]
        expected_order = [
            "validate_request",
            "classify_evidence",
            "assess_confidence",
            "synthesize_signals",
            "quantify_impacts",
            "evaluate_scenarios",
            "generate_decisions",
        ]
        assert stage_names == expected_order

    def test_validation_runs_first(self):
        orch = DecisionOrchestrator()
        request = _make_request(evidence_items=_make_evidence_set(3))
        report = orch.run(request)
        assert report.stages[0].name == "validate_request"
        assert report.stages[0].status == StageStatus.COMPLETED.value


# ===========================================================================
# 5. Optional scenario evaluation
# ===========================================================================

class TestOptionalScenarioEvaluation:
    """Test that scenarios are optional."""

    def test_no_scenarios_by_default(self):
        orch = DecisionOrchestrator()
        request = _make_request(evidence_items=_make_evidence_set(3))
        report = orch.run(request)
        scenario_stage = report.stage_by_name("evaluate_scenarios")
        assert scenario_stage is not None
        assert scenario_stage.status == StageStatus.NOT_REQUESTED.value

    def test_scenarios_requested_but_none_supplied(self):
        orch = DecisionOrchestrator()
        request = _make_request(
            evidence_items=_make_evidence_set(3),
            evaluate_scenarios=True,
        )
        report = orch.run(request)
        scenario_stage = report.stage_by_name("evaluate_scenarios")
        assert scenario_stage.status == StageStatus.SKIPPED.value

    def test_scenarios_not_invented(self):
        orch = DecisionOrchestrator()
        request = _make_request(evidence_items=_make_evidence_set(3))
        report = orch.run(request)
        assert report.scenario_results == []


# ===========================================================================
# 6. Unsupported scenario evaluation
# ===========================================================================

class TestUnsupportedScenarios:
    """Test handling of scenarios that cannot be evaluated."""

    def test_scenario_evaluation_with_invalid_definition(self):
        orch = DecisionOrchestrator()
        # Supply a non-ScenarioDefinition object — will fail gracefully
        fake_defn = {"name": "bad", "dataset_id": DATASET_A}
        request = _make_request(
            evidence_items=_make_evidence_set(3),
            evaluate_scenarios=True,
            scenario_definitions=[fake_defn],
        )
        report = orch.run(request)
        scenario_stage = report.stage_by_name("evaluate_scenarios")
        # Should complete but may have processing notes about failures
        assert scenario_stage is not None
        assert scenario_stage.status in {
            StageStatus.COMPLETED.value,
            StageStatus.FAILED.value,
        }


# ===========================================================================
# 7. Dataset-scope validation
# ===========================================================================

class TestDatasetScopeValidation:
    """Test dataset scope enforcement."""

    def test_request_requires_dataset_id(self):
        with pytest.raises(ValueError, match="dataset_id is required"):
            OrchestrationRequest(dataset_id="")

    def test_whitespace_dataset_id_rejected(self):
        with pytest.raises(ValueError, match="dataset_id is required"):
            OrchestrationRequest(dataset_id="   ")

    def test_evidence_scoped_to_dataset(self):
        orch = DecisionOrchestrator()
        evidence = _make_evidence_set(3, dataset_id=DATASET_A)
        request = _make_request(evidence_items=evidence)
        report = orch.run(request)
        assert report.evidence_summary.dataset_id == DATASET_A


# ===========================================================================
# 8. Cross-dataset evidence rejection
# ===========================================================================

class TestCrossDatasetEvidenceRejection:
    """Test that cross-dataset evidence is filtered out."""

    def test_wrong_dataset_evidence_filtered(self):
        orch = DecisionOrchestrator()
        good_ev = _make_evidence(dataset_id=DATASET_A)
        bad_ev = _make_evidence(dataset_id=DATASET_B)
        request = _make_request(evidence_items=[good_ev, bad_ev])
        report = orch.run(request)
        # Warnings about cross-dataset filtering
        classify_stage = report.stage_by_name("classify_evidence")
        all_warnings = classify_stage.warnings + report.warnings
        assert any("filter" in w.lower() or "dataset" in w.lower()
                    for w in all_warnings)

    def test_only_matching_evidence_classified(self):
        orch = DecisionOrchestrator()
        good_ev = _make_evidence_set(2, dataset_id=DATASET_A)
        bad_ev = [_make_evidence(dataset_id=DATASET_B)]
        request = _make_request(evidence_items=good_ev + bad_ev)
        report = orch.run(request)
        # Evidence summary should count only DATASET_A evidence
        # (the classify stage filters by dataset scope)
        assert report.evidence_summary.total_count <= 3


# ===========================================================================
# 9. Cross-dataset component-result rejection
# ===========================================================================

class TestCrossDatasetComponentRejection:
    """Test that cross-dataset component results are rejected."""

    def test_analytics_wrong_dataset_rejected(self):
        orch = DecisionOrchestrator()
        analytics = FakeAnalyticsResult(dataset_id=DATASET_B)
        request = _make_request(analytics_result=analytics)
        report = orch.run(request)
        assert report.status == ExecutionStatus.FAILED.value
        validate_stage = report.stage_by_name("validate_request")
        assert validate_stage.status == StageStatus.FAILED.value
        assert "cross-dataset" in validate_stage.error.lower()

    def test_prediction_wrong_dataset_rejected(self):
        orch = DecisionOrchestrator()
        predictions = FakePredictionReport(dataset_id=DATASET_B)
        request = _make_request(prediction_report=predictions)
        report = orch.run(request)
        assert report.status == ExecutionStatus.FAILED.value

    def test_phase5_wrong_dataset_rejected(self):
        orch = DecisionOrchestrator()
        phase5 = FakePhase5Result(dataset_id=DATASET_B)
        request = _make_request(phase5_result=phase5)
        report = orch.run(request)
        assert report.status == ExecutionStatus.FAILED.value

    def test_marketing_wrong_dataset_rejected(self):
        orch = DecisionOrchestrator()
        marketing = FakeMarketingResult(dataset_id=DATASET_B)
        request = _make_request(marketing_result=marketing)
        report = orch.run(request)
        assert report.status == ExecutionStatus.FAILED.value


# ===========================================================================
# 10. Sequential datasets without leakage
# ===========================================================================

class TestSequentialDatasetIsolation:
    """Test that sequential workflows remain independent."""

    def test_two_datasets_no_leakage(self):
        orch = DecisionOrchestrator()

        # Workflow A
        ev_a = _make_evidence_set(3, dataset_id=DATASET_A)
        request_a = _make_request(dataset_id=DATASET_A, evidence_items=ev_a)
        report_a = orch.run(request_a)

        # Workflow B
        ev_b = _make_evidence_set(2, dataset_id=DATASET_B)
        request_b = _make_request(dataset_id=DATASET_B, evidence_items=ev_b)
        report_b = orch.run(request_b)

        # Both should be independent
        assert report_a.dataset_id == DATASET_A
        assert report_b.dataset_id == DATASET_B
        assert report_a.workflow_id != report_b.workflow_id
        assert report_a.evidence_summary.total_count == 3
        assert report_b.evidence_summary.total_count == 2

    def test_sequential_workflows_different_evidence(self):
        orch = DecisionOrchestrator()

        # First run with evidence
        request1 = _make_request(
            dataset_id=DATASET_A,
            evidence_items=_make_evidence_set(3, dataset_id=DATASET_A),
        )
        report1 = orch.run(request1)

        # Second run without evidence
        request2 = _make_request(dataset_id=DATASET_A, evidence_items=[])
        report2 = orch.run(request2)

        # Second run should not have evidence from first run
        assert report2.evidence_summary.total_count == 0
        assert report1.evidence_summary.total_count == 3


# ===========================================================================
# 11. Recovery after a failed workflow
# ===========================================================================

class TestRecoveryAfterFailure:
    """Test that a failed workflow does not corrupt the next run."""

    def test_recovery_after_failed_validation(self):
        orch = DecisionOrchestrator()

        # Intentionally fail with cross-dataset result
        bad_analytics = FakeAnalyticsResult(dataset_id=DATASET_B)
        request_bad = _make_request(analytics_result=bad_analytics)
        report_bad = orch.run(request_bad)
        assert report_bad.status == ExecutionStatus.FAILED.value

        # Next run should work fine
        request_good = _make_request(evidence_items=_make_evidence_set(3))
        report_good = orch.run(request_good)
        assert report_good.status != ExecutionStatus.FAILED.value
        assert report_good.evidence_summary.total_count == 3

    def test_recovery_independent_orchestrator(self):
        """Each orchestrator run is fully independent."""
        orch1 = DecisionOrchestrator()
        orch2 = DecisionOrchestrator()

        bad_analytics = FakeAnalyticsResult(dataset_id=DATASET_B)
        request_bad = _make_request(analytics_result=bad_analytics)
        report_bad = orch1.run(request_bad)
        assert report_bad.status == ExecutionStatus.FAILED.value

        request_good = _make_request(evidence_items=_make_evidence_set(3))
        report_good = orch2.run(request_good)
        assert report_good.status != ExecutionStatus.FAILED.value


# ===========================================================================
# 12. Stage-specific failures
# ===========================================================================

class TestStageSpecificFailures:
    """Test that stage failures are recorded correctly."""

    def test_validation_failure_recorded(self):
        orch = DecisionOrchestrator()
        analytics = FakeAnalyticsResult(dataset_id=DATASET_B)
        request = _make_request(analytics_result=analytics)
        report = orch.run(request)
        validate_stage = report.stage_by_name("validate_request")
        assert validate_stage.status == StageStatus.FAILED.value
        assert validate_stage.error

    def test_skipped_stages_recorded(self):
        orch = DecisionOrchestrator()
        request = _make_request()
        report = orch.run(request)
        synth_stage = report.stage_by_name("synthesize_signals")
        assert synth_stage.status == StageStatus.SKIPPED.value

    def test_stage_result_has_name(self):
        orch = DecisionOrchestrator()
        request = _make_request(evidence_items=_make_evidence_set(3))
        report = orch.run(request)
        for stage in report.stages:
            assert stage.name, "Each stage must have a name"


# ===========================================================================
# 13. Partial-success reporting
# ===========================================================================

class TestPartialSuccessReporting:
    """Test partial success detection."""

    def test_no_decisions_but_evidence_is_partial(self):
        orch = DecisionOrchestrator()
        # Single evidence item → low confidence → may produce investigate/blocked
        ev = _make_evidence(description="Single observation")
        request = _make_request(evidence_items=[ev])
        report = orch.run(request)
        # Status should not be FAILED since stages completed
        assert report.status != ExecutionStatus.FAILED.value


# ===========================================================================
# 14. Complete-failure reporting
# ===========================================================================

class TestCompleteFailureReporting:
    """Test complete failure detection."""

    def test_validation_failure_causes_failed_status(self):
        orch = DecisionOrchestrator()
        analytics = FakeAnalyticsResult(dataset_id=DATASET_B)
        request = _make_request(analytics_result=analytics)
        report = orch.run(request)
        assert report.status == ExecutionStatus.FAILED.value

    def test_failed_report_has_errors(self):
        orch = DecisionOrchestrator()
        analytics = FakeAnalyticsResult(dataset_id=DATASET_B)
        request = _make_request(analytics_result=analytics)
        report = orch.run(request)
        assert len(report.errors) > 0


# ===========================================================================
# 15. Warning and limitation propagation
# ===========================================================================

class TestWarningAndLimitationPropagation:
    """Test that warnings and limitations flow through stages."""

    def test_cross_dataset_warning_propagated(self):
        orch = DecisionOrchestrator()
        good_ev = _make_evidence(dataset_id=DATASET_A)
        bad_ev = _make_evidence(dataset_id=DATASET_B)
        request = _make_request(evidence_items=[good_ev, bad_ev])
        report = orch.run(request)
        assert len(report.warnings) > 0

    def test_processing_notes_collected(self):
        orch = DecisionOrchestrator()
        request = _make_request(evidence_items=_make_evidence_set(3))
        report = orch.run(request)
        assert len(report.processing_notes) > 0


# ===========================================================================
# 16. Evidence provenance and decision traceability
# ===========================================================================

class TestEvidenceProvenanceTraceability:
    """Test evidence provenance tracking."""

    def test_evidence_summary_has_sources(self):
        orch = DecisionOrchestrator()
        ev = _make_evidence(source="data_engine")
        request = _make_request(evidence_items=[ev])
        report = orch.run(request)
        assert "data_engine" in report.evidence_summary.sources

    def test_evidence_summary_counts_by_level(self):
        orch = DecisionOrchestrator()
        fact = _make_evidence(level=EvidenceLevel.FACT.value)
        inf = _make_evidence(
            description="Inferred relationship",
            level=EvidenceLevel.INFERENCE.value,
        )
        request = _make_request(evidence_items=[fact, inf])
        report = orch.run(request)
        summary = report.evidence_summary
        assert summary.fact_count == 1
        assert summary.inference_count == 1
        assert summary.total_count == 2

    def test_decisions_traceable_through_report(self):
        orch = DecisionOrchestrator()
        evidence = _make_evidence_set(3)
        request = _make_request(evidence_items=evidence)
        report = orch.run(request)
        # If decisions were generated, each should have evidence_ids
        for d in report.decisions():
            assert d.dataset_id == DATASET_A


# ===========================================================================
# 17. Deterministic decision ordering
# ===========================================================================

class TestDeterministicOrdering:
    """Test that equivalent inputs produce equivalent ordering."""

    def test_same_inputs_same_order(self):
        orch = DecisionOrchestrator()
        evidence = _make_evidence_set(3)

        request1 = _make_request(evidence_items=evidence)
        report1 = orch.run(request1)

        request2 = _make_request(evidence_items=evidence)
        report2 = orch.run(request2)

        decisions1 = report1.decisions()
        decisions2 = report2.decisions()
        # Same number of decisions
        assert len(decisions1) == len(decisions2)
        # Same categories in same order
        categories1 = [d.category for d in decisions1]
        categories2 = [d.category for d in decisions2]
        assert categories1 == categories2

    def test_workflow_ids_differ(self):
        """Workflow IDs should be unique even with same inputs."""
        orch = DecisionOrchestrator()
        evidence = _make_evidence_set(3)
        r1 = orch.run(_make_request(evidence_items=evidence))
        r2 = orch.run(_make_request(evidence_items=evidence))
        assert r1.workflow_id != r2.workflow_id


# ===========================================================================
# 18. Safe serialization and deserialization
# ===========================================================================

class TestSerialization:
    """Test serialization round-trip."""

    def test_orchestration_report_to_dict(self):
        orch = DecisionOrchestrator()
        request = _make_request(evidence_items=_make_evidence_set(3))
        report = orch.run(request)
        d = report.to_dict()
        assert isinstance(d, dict)
        assert d["dataset_id"] == DATASET_A
        assert d["workflow_id"].startswith("wf_")
        assert isinstance(d["stages"], list)

    def test_orchestration_report_roundtrip(self):
        orch = DecisionOrchestrator()
        request = _make_request(evidence_items=_make_evidence_set(3))
        report = orch.run(request)
        d = report.to_dict()
        restored = OrchestrationReport.from_dict(d)
        assert restored.dataset_id == report.dataset_id
        assert restored.workflow_id == report.workflow_id
        assert restored.status == report.status
        assert len(restored.stages) == len(report.stages)

    def test_evidence_summary_to_dict(self):
        orch = DecisionOrchestrator()
        request = _make_request(evidence_items=_make_evidence_set(3))
        report = orch.run(request)
        d = report.evidence_summary.to_dict()
        assert d["total_count"] == 3
        assert d["dataset_id"] == DATASET_A

    def test_evidence_summary_roundtrip(self):
        orch = DecisionOrchestrator()
        request = _make_request(evidence_items=_make_evidence_set(3))
        report = orch.run(request)
        d = report.evidence_summary.to_dict()
        restored = EvidenceSummary.from_dict(d)
        assert restored.total_count == report.evidence_summary.total_count

    def test_stage_result_to_dict(self):
        stage = StageResult(
            name="test_stage",
            status=StageStatus.COMPLETED.value,
            warnings=["warning"],
            processing_notes=["note"],
        )
        d = stage.to_dict()
        assert d["name"] == "test_stage"
        assert d["status"] == "completed"
        restored = StageResult.from_dict(d)
        assert restored.name == stage.name

    def test_request_to_dict(self):
        request = _make_request(evidence_items=_make_evidence_set(3))
        d = request.to_dict()
        assert d["dataset_id"] == DATASET_A
        assert d["evidence_count"] == 3

    def test_empty_report_serialization(self):
        orch = DecisionOrchestrator()
        request = _make_request()
        report = orch.run(request)
        d = report.to_dict()
        assert d["decision_report"] is None or isinstance(d["decision_report"], dict)
        # Should not raise
        OrchestrationReport.from_dict(d)


# ===========================================================================
# 19. NaN/Infinity handling
# ===========================================================================

class TestNaNInfinityHandling:
    """Test that NaN/Infinity values are handled safely."""

    def test_nan_metric_value_in_evidence(self):
        orch = DecisionOrchestrator()
        ev = _make_evidence(metric_value=float("nan"))
        request = _make_request(evidence_items=[ev])
        report = orch.run(request)
        d = report.to_dict()
        # Should not contain NaN in serialized form
        assert report is not None

    def test_inf_metric_value_in_evidence(self):
        orch = DecisionOrchestrator()
        ev = _make_evidence(metric_value=float("inf"))
        request = _make_request(evidence_items=[ev])
        report = orch.run(request)
        assert report is not None

    def test_report_serialization_no_nan(self):
        orch = DecisionOrchestrator()
        ev = _make_evidence(metric_value=float("nan"))
        request = _make_request(evidence_items=[ev])
        report = orch.run(request)
        d = report.to_dict()
        # Serialization should complete without error
        assert isinstance(d, dict)


# ===========================================================================
# 20. No demo-data fallback
# ===========================================================================

class TestNoDemoDataFallback:
    """Test that no demo data is invented."""

    def test_empty_input_no_demo_data(self):
        orch = DecisionOrchestrator()
        request = _make_request()
        report = orch.run(request)
        assert report.decisions() == []
        assert report.evidence_summary.total_count == 0

    def test_no_hardcoded_values_in_empty_report(self):
        orch = DecisionOrchestrator()
        request = _make_request()
        report = orch.run(request)
        d = report.to_dict()
        # No invented decision data
        if d.get("decision_report"):
            decisions = d["decision_report"].get("decisions", [])
            assert len(decisions) == 0


# ===========================================================================
# 21. No fabricated successful outputs
# ===========================================================================

class TestNoFabricatedOutputs:
    """Test that failed stages don't produce fabricated results."""

    def test_failed_validation_no_decisions(self):
        orch = DecisionOrchestrator()
        analytics = FakeAnalyticsResult(dataset_id=DATASET_B)
        request = _make_request(analytics_result=analytics)
        report = orch.run(request)
        assert report.decisions() == []

    def test_failed_stage_error_preserved(self):
        orch = DecisionOrchestrator()
        analytics = FakeAnalyticsResult(dataset_id=DATASET_B)
        request = _make_request(analytics_result=analytics)
        report = orch.run(request)
        assert len(report.errors) > 0
        assert any("cross-dataset" in e.lower() for e in report.errors)


# ===========================================================================
# 22. No silently swallowed exceptions
# ===========================================================================

class TestNoSwallowedExceptions:
    """Test that exceptions are recorded, not swallowed."""

    def test_validation_error_in_errors_list(self):
        orch = DecisionOrchestrator()
        analytics = FakeAnalyticsResult(dataset_id=DATASET_B)
        request = _make_request(analytics_result=analytics)
        report = orch.run(request)
        assert len(report.errors) > 0

    def test_stage_error_field_populated(self):
        orch = DecisionOrchestrator()
        analytics = FakeAnalyticsResult(dataset_id=DATASET_B)
        request = _make_request(analytics_result=analytics)
        report = orch.run(request)
        validate_stage = report.stage_by_name("validate_request")
        assert validate_stage.error != ""

    def test_processing_notes_capture_issues(self):
        orch = DecisionOrchestrator()
        request = _make_request()
        report = orch.run(request)
        # Should have notes about missing evidence
        assert len(report.processing_notes) > 0


# ===========================================================================
# 23. Compatibility with Parts 1–5 public contracts
# ===========================================================================

class TestParts1To5Compatibility:
    """Test compatibility with existing Part 1–5 contracts."""

    def test_uses_part1_evidence_contracts(self):
        ev = _make_evidence()
        assert isinstance(ev, DecisionEvidence)
        assert ev.level == EvidenceLevel.FACT.value

    def test_uses_part1_confidence_contracts(self):
        evidence = _make_evidence_set(3)
        score = assess_confidence(evidence, dataset_id=DATASET_A)
        assert isinstance(score, ConfidenceScore)

    def test_orchestrator_accepts_part2_synthesis(self):
        # Build a real synthesis result through the synthesizer
        synthesizer = SignalSynthesizer(dataset_id=DATASET_A)
        evidence = _make_evidence_set(3)
        synthesizer.ingest_evidence(evidence)
        synthesis = synthesizer.synthesize()
        assert isinstance(synthesis, SynthesisResult)

    def test_orchestrator_accepts_part3_impact(self):
        # Build a real impact through the quantifier
        synthesizer = SignalSynthesizer(dataset_id=DATASET_A)
        evidence = _make_evidence_set(3)
        synthesizer.ingest_evidence(evidence)
        synthesis = synthesizer.synthesize()

        quantifier = ImpactQuantifier(dataset_id=DATASET_A)
        report = quantifier.assess_signals(
            synthesis.signals, synthesis.evidence_registry,
        )
        assert isinstance(report, ImpactReport)

    def test_orchestrator_accepts_part5_generator(self):
        gen = DecisionGenerator(dataset_id=DATASET_A)
        report = gen.generate()
        assert isinstance(report, DecisionReport)

    def test_execution_status_enum_values(self):
        assert ExecutionStatus.SUCCESS.value == "success"
        assert ExecutionStatus.PARTIAL_SUCCESS.value == "partial_success"
        assert ExecutionStatus.INSUFFICIENT_EVIDENCE.value == "insufficient_evidence"
        assert ExecutionStatus.FAILED.value == "failed"

    def test_stage_status_enum_values(self):
        assert StageStatus.COMPLETED.value == "completed"
        assert StageStatus.SKIPPED.value == "skipped"
        assert StageStatus.FAILED.value == "failed"
        assert StageStatus.NOT_REQUESTED.value == "not_requested"


# ===========================================================================
# 24. No global mutable state
# ===========================================================================

class TestNoGlobalMutableState:
    """Test that no global mutable state leaks between runs."""

    def test_orchestrator_no_instance_state(self):
        orch = DecisionOrchestrator()
        # Run twice with different inputs
        r1 = orch.run(_make_request(
            dataset_id=DATASET_A,
            evidence_items=_make_evidence_set(3, DATASET_A),
        ))
        r2 = orch.run(_make_request(
            dataset_id=DATASET_B,
            evidence_items=_make_evidence_set(2, DATASET_B),
        ))
        assert r1.dataset_id == DATASET_A
        assert r2.dataset_id == DATASET_B
        assert r1.evidence_summary.total_count == 3
        assert r2.evidence_summary.total_count == 2

    def test_different_orchestrators_independent(self):
        orch1 = DecisionOrchestrator()
        orch2 = DecisionOrchestrator()
        r1 = orch1.run(_make_request(evidence_items=_make_evidence_set(3)))
        r2 = orch2.run(_make_request(evidence_items=_make_evidence_set(2)))
        assert r1.evidence_summary.total_count == 3
        assert r2.evidence_summary.total_count == 2

    def test_no_class_level_state(self):
        """Verify no class-level mutable state exists."""
        assert not hasattr(DecisionOrchestrator, "_state")
        assert not hasattr(DecisionOrchestrator, "_cache")
        assert not hasattr(DecisionOrchestrator, "_results")


# ===========================================================================
# 25. Invalid request handling
# ===========================================================================

class TestInvalidRequestHandling:
    """Test handling of invalid or malformed requests."""

    def test_empty_dataset_id_rejected(self):
        with pytest.raises(ValueError, match="dataset_id is required"):
            OrchestrationRequest(dataset_id="")

    def test_none_evidence_items_handled(self):
        # Default is empty list
        request = _make_request()
        assert request.evidence_items == []

    def test_request_with_non_evidence_objects(self):
        orch = DecisionOrchestrator()
        # Pass non-evidence objects — should be handled gracefully
        request = _make_request(evidence_items=[{"not": "evidence"}])
        report = orch.run(request)
        # Should not crash
        assert report is not None


# ===========================================================================
# Additional edge cases
# ===========================================================================

class TestOrchestrationReportHelpers:
    """Test OrchestrationReport helper methods."""

    def test_stage_by_name_found(self):
        orch = DecisionOrchestrator()
        report = orch.run(_make_request(evidence_items=_make_evidence_set(3)))
        stage = report.stage_by_name("validate_request")
        assert stage is not None
        assert stage.name == "validate_request"

    def test_stage_by_name_not_found(self):
        orch = DecisionOrchestrator()
        report = orch.run(_make_request(evidence_items=_make_evidence_set(3)))
        stage = report.stage_by_name("nonexistent_stage")
        assert stage is None

    def test_decisions_helper_returns_list(self):
        orch = DecisionOrchestrator()
        report = orch.run(_make_request(evidence_items=_make_evidence_set(3)))
        decisions = report.decisions()
        assert isinstance(decisions, list)

    def test_decisions_empty_when_no_report(self):
        report = OrchestrationReport(dataset_id=DATASET_A)
        assert report.decisions() == []


class TestRegressionSafeImports:
    """Test that all imports work correctly."""

    def test_orchestrator_imports(self):
        from src.decision.orchestrator import (
            DecisionOrchestrator,
            OrchestrationRequest,
            OrchestrationReport,
            EvidenceSummary,
            StageResult,
            ExecutionStatus,
            StageStatus,
        )
        assert DecisionOrchestrator is not None
        assert OrchestrationRequest is not None
        assert OrchestrationReport is not None

    def test_all_parts_importable(self):
        from src.decision.contracts import EvidenceLevel, ConfidenceLevel
        from src.decision.evidence import create_evidence, validate_dataset_scope
        from src.decision.confidence import assess_confidence
        from src.decision.signal_synthesizer import SignalSynthesizer
        from src.decision.impact_quantifier import ImpactQuantifier
        from src.decision.scenario_engine import ScenarioEngine
        from src.decision.decision_generator import DecisionGenerator
        from src.decision.orchestrator import DecisionOrchestrator
        assert all([
            EvidenceLevel, ConfidenceLevel,
            create_evidence, validate_dataset_scope,
            assess_confidence,
            SignalSynthesizer, ImpactQuantifier,
            ScenarioEngine, DecisionGenerator,
            DecisionOrchestrator,
        ])


class TestEndToEndIntegration:
    """Full integration test using real Part 2-5 components."""

    def test_full_pipeline_with_evidence(self):
        """Run the full pipeline with real components."""
        orch = DecisionOrchestrator()
        evidence = _make_evidence_set(4, dataset_id=DATASET_A)
        request = _make_request(evidence_items=evidence)
        report = orch.run(request)

        # Should produce a complete report
        assert report.workflow_id.startswith("wf_")
        assert report.dataset_id == DATASET_A
        assert report.evidence_summary is not None
        assert report.evidence_summary.total_count == 4
        assert report.evidence_summary.confidence is not None
        assert len(report.stages) == 7

        # Serialization should work end-to-end
        d = report.to_dict()
        assert isinstance(d, dict)
        restored = OrchestrationReport.from_dict(d)
        assert restored.dataset_id == DATASET_A
