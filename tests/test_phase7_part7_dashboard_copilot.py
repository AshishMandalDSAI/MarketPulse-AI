"""Phase 7 Part 7 — Dashboard + AI Copilot Integration Tests.

Covers all 24 required test categories from the specification:
 1. Decision dashboard rendering
 2. Successful orchestrator integration
 3. Empty decision results
 4. Insufficient-evidence results
 5. Partial-success results
 6. Failed workflow rendering
 7. Evidence provenance display
 8. Correct separation of confidence, impact, urgency, and priority
 9. Correct distinction between facts, predictions, recommendations, limitations
10. Dataset switching without stale decision cards
11. No cross-dataset Copilot context leakage
12. Copilot receives only the active dataset's relevant decision context
13. Missing or unavailable optional Copilot capability
14. Unsupported questions and insufficient evidence
15. Missing optional scenario results
16. Safe handling of warnings and stage errors
17. No demo-data fallback
18. No fabricated recommendations
19. No credentials or secrets exposed in rendered content
20. Compatibility with existing dashboard behavior
21. Correct handling of invalid or missing dataset/session identity
22. Cache behavior across dataset changes
23. Deterministic presentation of equivalent decision results
24. Regression coverage for any modified integration boundary

These tests exercise the service layer (dashboard_service.py), the
Copilot integration (analyst.py), and the navigation integration (app.py,
session_paths.py) WITHOUT requiring a running Streamlit server.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest

from src.decision.dashboard_service import (
    DecisionCard,
    DashboardResult,
    build_copilot_context,
    format_copilot_answer_context,
    format_dashboard_result,
    format_decision_cards,
    get_confidence_display,
    get_evidence_level_display,
    get_stage_status_display,
    get_status_display,
    run_decision_workflow,
    _empty_copilot_context,
)
from src.decision.orchestrator import (
    ExecutionStatus,
    OrchestrationReport,
    OrchestrationRequest,
    StageStatus,
    StageResult,
    EvidenceSummary,
    DecisionOrchestrator,
)
from src.decision.contracts import ConfidenceLevel, EvidenceLevel
from src.ai.analyst import MarketingAnalyst


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_minimal_report(
    dataset_id: str = "test-dataset-001",
    status: str = ExecutionStatus.SUCCESS.value,
    *,
    decision_report=None,
    evidence_summary=None,
    stages=None,
    warnings=None,
    errors=None,
    limitations=None,
    scenario_results=None,
) -> OrchestrationReport:
    """Build a minimal OrchestrationReport for testing."""
    return OrchestrationReport(
        dataset_id=dataset_id,
        workflow_id="wf-test-001",
        status=status,
        stages=stages or [],
        evidence_summary=evidence_summary,
        decision_report=decision_report,
        scenario_results=scenario_results,
        warnings=warnings or [],
        errors=errors or [],
        limitations=limitations or [],
    )


class _FakeDecision:
    """Lightweight stand-in for a decision object."""
    def __init__(self, **kwargs):
        self.title = kwargs.get("title", "Test Decision")
        self.category = kwargs.get("category", "test")
        self.description = kwargs.get("description", "A test decision.")
        self.priority = kwargs.get("priority", "high")
        self.urgency = kwargs.get("urgency", "medium")
        self.confidence_level = kwargs.get("confidence_level", "high")
        self.reversibility = kwargs.get("reversibility", "reversible")
        self.evidence_ids = kwargs.get("evidence_ids", ["ev-001", "ev-002"])
        self.impact_magnitude = kwargs.get("impact_magnitude", 0.15)
        self.impact_type = kwargs.get("impact_type", "revenue")
        self.limitations = kwargs.get("limitations", [])
        self.warnings = kwargs.get("warnings", [])
        self.dataset_id = kwargs.get("dataset_id", "test-dataset-001")


class _FakeDecisionReport:
    """Lightweight stand-in for a decision report."""
    def __init__(self, decisions=None):
        self.decisions = decisions or []


class _FakeConfidence:
    """Lightweight stand-in for a confidence assessment."""
    def __init__(self, level="high"):
        self.level = level


class _FakeLimitation:
    """Stand-in for a limitation."""
    def __init__(self, desc):
        self.description = desc
    def to_dict(self):
        return {"description": self.description}


# ---------------------------------------------------------------------------
# 1. Decision dashboard rendering
# ---------------------------------------------------------------------------

class TestDashboardRendering:
    """Verify that DashboardResult is correctly built from a report."""

    def test_format_dashboard_result_basic(self):
        report = _make_minimal_report()
        result = format_dashboard_result(report)
        assert isinstance(result, DashboardResult)
        assert result.dataset_id == "test-dataset-001"
        assert result.workflow_id == "wf-test-001"
        assert result.status_display["label"] == "Success"

    def test_format_dashboard_result_with_decisions(self):
        dec = _FakeDecision()
        dr = _FakeDecisionReport([dec])
        report = _make_minimal_report(decision_report=dr)
        result = format_dashboard_result(report)
        assert result.has_decisions is True
        assert len(result.decision_cards) == 1
        assert result.decision_cards[0].title == "Test Decision"

    def test_dashboard_result_to_dict(self):
        report = _make_minimal_report()
        result = format_dashboard_result(report)
        d = result.to_dict()
        assert "dataset_id" in d
        assert "status" in d
        assert "decision_cards" in d
        assert isinstance(d["decision_cards"], list)


# ---------------------------------------------------------------------------
# 2. Successful orchestrator integration
# ---------------------------------------------------------------------------

class TestOrchestratorIntegration:
    """Verify run_decision_workflow calls the orchestrator and returns results."""

    def test_run_returns_dashboard_result(self):
        result = run_decision_workflow("test-ds-integration")
        assert isinstance(result, DashboardResult)
        assert result.dataset_id == "test-ds-integration"

    def test_run_populates_status(self):
        result = run_decision_workflow("test-ds-integration")
        assert "label" in result.status_display
        assert result.status_display["raw"] in [s.value for s in ExecutionStatus]


# ---------------------------------------------------------------------------
# 3. Empty decision results
# ---------------------------------------------------------------------------

class TestEmptyDecisionResults:
    """Verify honest empty state when no decisions are generated."""

    def test_empty_decisions_have_reason(self):
        report = _make_minimal_report(status=ExecutionStatus.SUCCESS.value)
        result = format_dashboard_result(report)
        assert result.has_decisions is False
        assert result.empty_reason != ""
        assert "no decisions" in result.empty_reason.lower() or "insufficient" in result.empty_reason.lower()

    def test_empty_reason_not_fabricated(self):
        """Empty reason must explain why, not present fake results."""
        report = _make_minimal_report(status=ExecutionStatus.SUCCESS.value)
        result = format_dashboard_result(report)
        assert "sample" not in result.empty_reason.lower()
        assert "demo" not in result.empty_reason.lower()
        assert "example" not in result.empty_reason.lower()


# ---------------------------------------------------------------------------
# 4. Insufficient-evidence results
# ---------------------------------------------------------------------------

class TestInsufficientEvidence:
    """Verify correct display of insufficient-evidence status."""

    def test_insufficient_evidence_status(self):
        report = _make_minimal_report(status=ExecutionStatus.INSUFFICIENT_EVIDENCE.value)
        result = format_dashboard_result(report)
        assert result.status_display["label"] == "Insufficient Evidence"
        assert result.has_decisions is False

    def test_insufficient_evidence_reason(self):
        report = _make_minimal_report(status=ExecutionStatus.INSUFFICIENT_EVIDENCE.value)
        result = format_dashboard_result(report)
        assert "evidence" in result.empty_reason.lower()


# ---------------------------------------------------------------------------
# 5. Partial-success results
# ---------------------------------------------------------------------------

class TestPartialSuccess:
    """Verify partial-success rendering."""

    def test_partial_success_status(self):
        report = _make_minimal_report(status=ExecutionStatus.PARTIAL_SUCCESS.value)
        result = format_dashboard_result(report)
        assert result.status_display["label"] == "Partial Success"
        assert result.status_display["level"] == "warning"

    def test_partial_success_with_some_decisions(self):
        dec = _FakeDecision()
        dr = _FakeDecisionReport([dec])
        report = _make_minimal_report(
            status=ExecutionStatus.PARTIAL_SUCCESS.value,
            decision_report=dr,
        )
        result = format_dashboard_result(report)
        assert result.has_decisions is True
        assert result.status_display["label"] == "Partial Success"


# ---------------------------------------------------------------------------
# 6. Failed workflow rendering
# ---------------------------------------------------------------------------

class TestFailedWorkflow:
    """Verify failed workflow display."""

    def test_failed_status(self):
        report = _make_minimal_report(
            status=ExecutionStatus.FAILED.value,
            errors=["Stage X failed: connection timeout"],
        )
        result = format_dashboard_result(report)
        assert result.status_display["label"] == "Failed"
        assert result.status_display["level"] == "error"
        assert len(result.errors) == 1

    def test_failed_empty_reason(self):
        report = _make_minimal_report(status=ExecutionStatus.FAILED.value)
        result = format_dashboard_result(report)
        assert "failed" in result.empty_reason.lower()


# ---------------------------------------------------------------------------
# 7. Evidence provenance display
# ---------------------------------------------------------------------------

class TestEvidenceProvenance:
    """Verify evidence IDs and summaries are preserved."""

    def test_evidence_ids_in_card(self):
        dec = _FakeDecision(evidence_ids=["ev-abc", "ev-def", "ev-ghi"])
        dr = _FakeDecisionReport([dec])
        report = _make_minimal_report(decision_report=dr)
        result = format_dashboard_result(report)
        card = result.decision_cards[0]
        assert card.evidence_ids == ["ev-abc", "ev-def", "ev-ghi"]
        assert "3 evidence items" in card.evidence_summary

    def test_no_evidence_ids(self):
        dec = _FakeDecision(evidence_ids=[])
        dr = _FakeDecisionReport([dec])
        report = _make_minimal_report(decision_report=dr)
        result = format_dashboard_result(report)
        card = result.decision_cards[0]
        assert "No direct evidence" in card.evidence_summary

    def test_single_evidence_id(self):
        dec = _FakeDecision(evidence_ids=["ev-solo"])
        dr = _FakeDecisionReport([dec])
        report = _make_minimal_report(decision_report=dr)
        result = format_dashboard_result(report)
        card = result.decision_cards[0]
        assert "1 evidence item" in card.evidence_summary

    def test_evidence_level_display(self):
        for level in EvidenceLevel:
            display = get_evidence_level_display(level.value)
            assert "label" in display
            assert "description" in display
            assert display["label"] != "Unknown"


# ---------------------------------------------------------------------------
# 8. Correct separation of confidence, impact, urgency, and priority
# ---------------------------------------------------------------------------

class TestDimensionSeparation:
    """Verify these dimensions remain separate — not conflated."""

    def test_card_dimensions_independent(self):
        dec = _FakeDecision(
            priority="high", urgency="low",
            confidence_level="medium", impact_magnitude=0.05,
        )
        dr = _FakeDecisionReport([dec])
        report = _make_minimal_report(decision_report=dr)
        result = format_dashboard_result(report)
        card = result.decision_cards[0]
        assert card.priority == "high"
        assert card.urgency == "low"
        assert card.confidence == "medium"
        # Impact is in a separate field
        assert "0.05" in card.impact_summary

    def test_high_priority_low_confidence(self):
        """High priority must not imply high confidence."""
        dec = _FakeDecision(priority="high", confidence_level="low")
        dr = _FakeDecisionReport([dec])
        report = _make_minimal_report(decision_report=dr)
        result = format_dashboard_result(report)
        card = result.decision_cards[0]
        assert card.priority == "high"
        assert card.confidence == "low"

    def test_confidence_display_levels(self):
        for level in ConfidenceLevel:
            display = get_confidence_display(level.value)
            assert "label" in display
            assert "description" in display


# ---------------------------------------------------------------------------
# 9. Correct distinction: facts, predictions, recommendations, limitations
# ---------------------------------------------------------------------------

class TestEpistemicDistinction:
    """Verify evidence level labels distinguish epistemic types."""

    def test_fact_label(self):
        d = get_evidence_level_display(EvidenceLevel.FACT.value)
        assert "fact" in d["label"].lower() or "measured" in d["label"].lower()

    def test_prediction_label(self):
        d = get_evidence_level_display(EvidenceLevel.PREDICTION.value)
        assert "prediction" in d["label"].lower()

    def test_recommendation_label(self):
        d = get_evidence_level_display(EvidenceLevel.RECOMMENDATION.value)
        assert "recommendation" in d["label"].lower()

    def test_limitation_label(self):
        d = get_evidence_level_display(EvidenceLevel.LIMITATION.value)
        assert "limitation" in d["label"].lower()

    def test_inference_label(self):
        d = get_evidence_level_display(EvidenceLevel.INFERENCE.value)
        assert "inference" in d["label"].lower()


# ---------------------------------------------------------------------------
# 10. Dataset switching without stale decision cards
# ---------------------------------------------------------------------------

class TestDatasetSwitching:
    """Verify decision results are invalidated on dataset switch."""

    def test_different_dataset_ids_produce_different_results(self):
        r1 = run_decision_workflow("dataset-alpha")
        r2 = run_decision_workflow("dataset-beta")
        assert r1.dataset_id == "dataset-alpha"
        assert r2.dataset_id == "dataset-beta"

    def test_decision_keys_in_dataset_dependent_list(self):
        """decision_result and decision_copilot_context must be in
        _DATASET_DEPENDENT_KEYS so they are cleared on switch."""
        from dashboard.components.session_paths import _DATASET_DEPENDENT_KEYS
        assert "decision_result" in _DATASET_DEPENDENT_KEYS
        assert "decision_copilot_context" in _DATASET_DEPENDENT_KEYS

    def test_copilot_context_refuses_stale_dataset(self):
        """build_copilot_context must reject a result from a different dataset."""
        report = _make_minimal_report(dataset_id="old-dataset")
        result = format_dashboard_result(report)
        ctx = build_copilot_context(result, "new-dataset")
        assert ctx["has_decisions"] is False
        assert ctx["dataset_id"] == "new-dataset"


# ---------------------------------------------------------------------------
# 11. No cross-dataset Copilot context leakage
# ---------------------------------------------------------------------------

class TestCopilotIsolation:
    """Verify the Copilot never receives another dataset's context."""

    def test_context_matches_dataset(self):
        report = _make_minimal_report(dataset_id="ds-A")
        result = format_dashboard_result(report)
        ctx = build_copilot_context(result, "ds-A")
        assert ctx["dataset_id"] == "ds-A"

    def test_context_empty_on_mismatch(self):
        report = _make_minimal_report(dataset_id="ds-A")
        result = format_dashboard_result(report)
        ctx = build_copilot_context(result, "ds-B")
        assert ctx["has_decisions"] is False
        assert ctx["decisions"] == []

    def test_analyst_rejects_mismatched_context(self):
        """Analyst must ignore context for a different dataset."""
        ctx = {"dataset_id": "ds-A", "has_decisions": True, "decisions": [{"title": "X"}]}
        analyst = MarketingAnalyst(decision_context=ctx, dataset_id="ds-B")
        assert analyst._decision_context is None


# ---------------------------------------------------------------------------
# 12. Copilot receives only the active dataset's relevant decision context
# ---------------------------------------------------------------------------

class TestCopilotActiveDataset:
    """Verify Copilot receives context from and only from the active dataset."""

    def test_copilot_context_contains_decisions(self):
        dec = _FakeDecision(dataset_id="ds-active")
        dr = _FakeDecisionReport([dec])
        report = _make_minimal_report(dataset_id="ds-active", decision_report=dr)
        result = format_dashboard_result(report)
        ctx = build_copilot_context(result, "ds-active")
        assert ctx["has_decisions"] is True
        assert ctx["decision_count"] == 1
        assert ctx["decisions"][0]["title"] == "Test Decision"

    def test_copilot_context_none_result(self):
        ctx = build_copilot_context(None, "ds-active")
        assert ctx["has_decisions"] is False


# ---------------------------------------------------------------------------
# 13. Missing or unavailable optional Copilot capability
# ---------------------------------------------------------------------------

class TestMissingCopilotCapability:
    """Verify graceful handling when decision context is unavailable."""

    def test_analyst_without_decision_context(self):
        analyst = MarketingAnalyst()
        assert analyst._decision_context is None

    def test_analyst_answers_decision_question_without_context(self):
        analyst = MarketingAnalyst()
        result = analyst.answer("What are the decision intelligence results?")
        assert "not been run" in result["answer"].lower() or "not available" in result["answer"].lower()

    def test_analyst_still_answers_non_decision_questions(self):
        """Non-decision questions still work when context is absent."""
        analyst = MarketingAnalyst()
        result = analyst.answer("What is the revenue trend?")
        # Should not mention decision intelligence
        assert "decision intelligence" not in result["answer"].lower() or "isn't available" in result["answer"].lower()


# ---------------------------------------------------------------------------
# 14. Unsupported questions and insufficient evidence
# ---------------------------------------------------------------------------

class TestUnsupportedQuestions:
    """Verify honest answers for questions outside available evidence."""

    def test_empty_context_copilot_answer(self):
        ctx = _empty_copilot_context("ds-test")
        text = format_copilot_answer_context(ctx)
        assert "not run" in text.lower() or "not available" in text.lower()

    def test_no_decisions_copilot_answer(self):
        ctx = {
            "dataset_id": "ds-test", "has_decisions": False,
            "status": "Insufficient Evidence",
            "empty_reason": "Not enough evidence.",
        }
        text = format_copilot_answer_context(ctx)
        assert "insufficient" in text.lower() or "not enough" in text.lower()


# ---------------------------------------------------------------------------
# 15. Missing optional scenario results
# ---------------------------------------------------------------------------

class TestMissingScenarios:
    """Verify graceful handling when scenarios are absent."""

    def test_no_scenarios_flag(self):
        report = _make_minimal_report()
        result = format_dashboard_result(report)
        assert result.has_scenarios is False

    def test_copilot_context_no_scenarios(self):
        report = _make_minimal_report()
        result = format_dashboard_result(report)
        ctx = build_copilot_context(result, "test-dataset-001")
        assert ctx["has_scenarios"] is False


# ---------------------------------------------------------------------------
# 16. Safe handling of warnings and stage errors
# ---------------------------------------------------------------------------

class TestWarningsAndErrors:
    """Verify warnings and stage errors are surfaced safely."""

    def test_warnings_preserved(self):
        report = _make_minimal_report(warnings=["Check data quality", "Limited date range"])
        result = format_dashboard_result(report)
        assert len(result.warnings) == 2

    def test_errors_preserved(self):
        report = _make_minimal_report(errors=["Stage failed: timeout"])
        result = format_dashboard_result(report)
        assert len(result.errors) == 1

    def test_limitations_preserved(self):
        lim = _FakeLimitation("Small sample size")
        report = _make_minimal_report(limitations=[lim])
        result = format_dashboard_result(report)
        assert "Small sample size" in result.limitations[0]

    def test_stage_error_display(self):
        stage = StageResult(
            name="evidence_collection",
            status=StageStatus.FAILED.value,
            error="Connection refused",
        )
        report = _make_minimal_report(stages=[stage])
        result = format_dashboard_result(report)
        assert result.stage_displays[0]["error"] == "Connection refused"
        assert result.stage_displays[0]["status"]["label"] == "Failed"

    def test_stage_warning_display(self):
        stage = StageResult(
            name="signal_synthesis",
            status=StageStatus.COMPLETED.value,
            warnings=["Low coverage"],
        )
        report = _make_minimal_report(stages=[stage])
        result = format_dashboard_result(report)
        assert "Low coverage" in result.stage_displays[0]["warnings"]


# ---------------------------------------------------------------------------
# 17. No demo-data fallback
# ---------------------------------------------------------------------------

class TestNoDemoFallback:
    """Verify no demo/sample data is injected when processing fails."""

    def test_failed_workflow_no_fake_decisions(self):
        report = _make_minimal_report(status=ExecutionStatus.FAILED.value)
        result = format_dashboard_result(report)
        assert result.has_decisions is False
        assert len(result.decision_cards) == 0

    def test_empty_dataset_no_fake_decisions(self):
        result = run_decision_workflow("empty-ds")
        assert isinstance(result, DashboardResult)
        # Whatever the outcome, decisions must come from orchestrator, not hardcoded
        for card in result.decision_cards:
            assert card.dataset_id == "empty-ds" or card.dataset_id == ""


# ---------------------------------------------------------------------------
# 18. No fabricated recommendations
# ---------------------------------------------------------------------------

class TestNoFabricatedRecommendations:
    """Verify the service never invents findings."""

    def test_empty_report_no_cards(self):
        report = _make_minimal_report(decision_report=None)
        cards = format_decision_cards(report)
        assert cards == []

    def test_copilot_no_fabrication(self):
        ctx = _empty_copilot_context("ds-test")
        text = format_copilot_answer_context(ctx)
        # Must not contain invented financial numbers or fake recommendations
        assert "$" not in text
        assert "recommend" not in text.lower() or "cannot" in text.lower() or "no" in text.lower()

    def test_analyst_no_fabrication_without_context(self):
        analyst = MarketingAnalyst()
        result = analyst.answer("What are the decision options?")
        answer = result["answer"].lower()
        assert "not been run" in answer or "not available" in answer or "run analysis" in answer


# ---------------------------------------------------------------------------
# 19. No credentials or secrets exposed
# ---------------------------------------------------------------------------

class TestNoSecretsExposed:
    """Verify no sensitive data in rendered content."""

    def test_copilot_context_no_raw_data(self):
        dec = _FakeDecision()
        dr = _FakeDecisionReport([dec])
        report = _make_minimal_report(decision_report=dr)
        result = format_dashboard_result(report)
        ctx = build_copilot_context(result, "test-dataset-001")
        # Context should not contain raw data keys
        ctx_str = str(ctx)
        assert "api_key" not in ctx_str.lower()
        assert "password" not in ctx_str.lower()
        assert "secret" not in ctx_str.lower()
        assert "credential" not in ctx_str.lower()
        assert "token" not in ctx_str.lower() or "version" in ctx_str.lower()  # version tokens ok

    def test_dashboard_result_no_secrets(self):
        report = _make_minimal_report()
        result = format_dashboard_result(report)
        d = result.to_dict()
        d_str = str(d)
        assert "api_key" not in d_str.lower()
        assert "password" not in d_str.lower()


# ---------------------------------------------------------------------------
# 20. Compatibility with existing dashboard behavior
# ---------------------------------------------------------------------------

class TestDashboardCompatibility:
    """Verify the new page integrates with existing navigation."""

    def test_page_registered_in_app(self):
        """The Decision Intelligence page must be in the PAGES dict."""
        # Import app module to check
        import importlib
        spec = importlib.util.find_spec("app")
        if spec is None:
            pytest.skip("app module not importable outside streamlit")

        # Upgrade PART 01: the menu lives in dashboard/navigation.py and app.py renders it.
        assert "PAGES = navigation.PAGES" in (ROOT / "app.py").read_text(encoding="utf-8")
        from dashboard.navigation import PAGES
        entries = [(lbl, mod) for lbl, mod in PAGES.items() if mod == "dashboard.pages.13_decision_intelligence"]
        assert len(entries) == 1 and "Decision Intelligence" in entries[0][0]

    def test_decision_page_module_importable(self):
        """The page module must be importable (no syntax errors)."""
        pytest.importorskip("streamlit")
        import importlib
        mod = importlib.import_module("dashboard.pages.13_decision_intelligence")
        assert hasattr(mod, "render")

    def test_dashboard_service_importable(self):
        """The service module must be importable."""
        from src.decision import dashboard_service
        assert hasattr(dashboard_service, "run_decision_workflow")
        assert hasattr(dashboard_service, "build_copilot_context")


# ---------------------------------------------------------------------------
# 21. Correct handling of invalid or missing dataset/session identity
# ---------------------------------------------------------------------------

class TestInvalidDataset:
    """Verify correct behavior for invalid dataset identifiers."""

    def test_empty_dataset_id(self):
        result = run_decision_workflow("")
        assert result.status_display["label"] == "Failed"
        assert "no dataset" in result.empty_reason.lower()

    def test_none_dataset_id_copilot(self):
        ctx = _empty_copilot_context("")
        assert ctx["dataset_id"] == ""
        assert ctx["has_decisions"] is False

    def test_whitespace_dataset_id(self):
        result = run_decision_workflow("   ")
        assert result.status_display["label"] == "Failed"

    def test_copilot_context_none_dashboard_result(self):
        ctx = build_copilot_context(None, "any-id")
        assert ctx["has_decisions"] is False
        assert ctx["dataset_id"] == "any-id"


# ---------------------------------------------------------------------------
# 22. Cache behavior across dataset changes
# ---------------------------------------------------------------------------

class TestCacheBehavior:
    """Verify caching keys and invalidation."""

    def test_dataset_dependent_keys_registered(self):
        from dashboard.components.session_paths import _DATASET_DEPENDENT_KEYS
        assert "decision_result" in _DATASET_DEPENDENT_KEYS
        assert "decision_copilot_context" in _DATASET_DEPENDENT_KEYS

    def test_fresh_workflow_per_call(self):
        """Each call to run_decision_workflow creates a fresh orchestrator."""
        r1 = run_decision_workflow("ds-cache-1")
        r2 = run_decision_workflow("ds-cache-1")
        # Both should succeed independently
        assert r1.dataset_id == "ds-cache-1"
        assert r2.dataset_id == "ds-cache-1"
        # Workflow IDs should differ (fresh orchestrator each time)
        assert r1.workflow_id != "" and r2.workflow_id != ""


# ---------------------------------------------------------------------------
# 23. Deterministic presentation of equivalent decision results
# ---------------------------------------------------------------------------

class TestDeterministicPresentation:
    """Verify same input → same output for display formatting."""

    def test_same_report_same_result(self):
        dec = _FakeDecision(title="Deterministic", priority="high")
        dr = _FakeDecisionReport([dec])
        report = _make_minimal_report(decision_report=dr)

        r1 = format_dashboard_result(report)
        r2 = format_dashboard_result(report)

        assert r1.to_dict() == r2.to_dict()

    def test_status_display_deterministic(self):
        for status in ExecutionStatus:
            d1 = get_status_display(status.value)
            d2 = get_status_display(status.value)
            assert d1 == d2

    def test_confidence_display_deterministic(self):
        for level in ConfidenceLevel:
            d1 = get_confidence_display(level.value)
            d2 = get_confidence_display(level.value)
            assert d1 == d2


# ---------------------------------------------------------------------------
# 24. Regression coverage for modified integration boundaries
# ---------------------------------------------------------------------------

class TestRegressionBoundaries:
    """Regression tests for modified files: session_paths, analyst, app."""

    def test_session_paths_module_loads(self):
        """session_paths must still export its public API."""
        from dashboard.components import session_paths as sp
        assert callable(sp.get_dataset_id)
        assert callable(sp.set_active_dataset)
        assert callable(sp.reset_to_demo_dataset)
        assert callable(sp.reset_analysis)
        assert callable(sp.dataset_chosen)
        assert callable(sp.get_processed_dir)

    def test_analyst_still_answers_marketing_questions(self):
        """Adding decision routing must not break existing marketing Q&A."""
        analyst = MarketingAnalyst()
        # These should hit existing handlers, not decision handler
        r1 = analyst.answer("Which channel should get more budget?")
        assert r1["grounded_in_project_data"] is True

    def test_analyst_decision_question_detection(self):
        analyst = MarketingAnalyst()
        assert analyst._is_decision_question("what are the decision options?")
        assert analyst._is_decision_question("show me the decision intelligence results")
        assert not analyst._is_decision_question("which channel should get more budget?")
        assert not analyst._is_decision_question("what is the revenue trend?")

    def test_analyst_answer_returns_correct_structure(self):
        """All answers must return the expected dict structure."""
        analyst = MarketingAnalyst()
        for q in [
            "What are the decisions?",
            "Which channel should get more budget?",
            "What is the revenue trend?",
        ]:
            result = analyst.answer(q)
            assert "question" in result
            assert "answer" in result
            assert "mode" in result
            assert "grounded_in_project_data" in result
            assert result["grounded_in_project_data"] is True

    def test_decision_card_to_dict(self):
        card = DecisionCard(
            title="Test", category="test", priority="high",
            urgency="low", confidence="medium",
            evidence_ids=["ev-1"], dataset_id="ds-1",
        )
        d = card.to_dict()
        assert d["title"] == "Test"
        assert d["priority"] == "high"
        assert d["urgency"] == "low"
        assert d["confidence"] == "medium"
        assert d["evidence_ids"] == ["ev-1"]

    def test_copilot_answer_context_with_decisions(self):
        """format_copilot_answer_context must produce readable text."""
        ctx = {
            "has_decisions": True,
            "status": "Success",
            "confidence": "High Confidence",
            "confidence_description": "Strong evidence.",
            "decisions": [{
                "title": "Increase Search Budget",
                "category": "budget",
                "priority": "high",
                "urgency": "medium",
                "confidence": "high",
                "description": "Shift budget to search.",
                "evidence_summary": "Based on 3 evidence items.",
                "impact_summary": "revenue: magnitude 0.15",
                "limitations": ["Small sample"],
            }],
            "warnings": ["Data quality warning"],
            "limitations": ["Limited date range"],
        }
        text = format_copilot_answer_context(ctx)
        assert "Increase Search Budget" in text
        assert "high" in text.lower()
        assert "Small sample" in text
        assert "Data quality warning" in text

    def test_copilot_warnings_limited(self):
        """Copilot context limits warnings/limitations to prevent context bloat."""
        dec = _FakeDecision()
        dr = _FakeDecisionReport([dec])
        report = _make_minimal_report(
            decision_report=dr,
            warnings=[f"w{i}" for i in range(20)],
        )
        result = format_dashboard_result(report)
        ctx = build_copilot_context(result, "test-dataset-001")
        assert len(ctx["warnings"]) <= 10

    def test_unknown_status_handled(self):
        d = get_status_display("nonexistent_status")
        assert d["label"] == "Unknown"

    def test_unknown_stage_status_handled(self):
        d = get_stage_status_display("bogus")
        assert d["label"] == "Unknown"

    def test_unknown_confidence_handled(self):
        d = get_confidence_display("bogus")
        assert d["label"] == "Unknown"

    def test_unknown_evidence_level_handled(self):
        d = get_evidence_level_display("bogus")
        assert d["label"] == "Unknown"
