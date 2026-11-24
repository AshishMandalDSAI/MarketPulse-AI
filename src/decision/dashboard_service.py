"""Phase 7 Part 7 — Decision Intelligence Dashboard Service.

Thin, stateless service layer that bridges the Decision Orchestrator
(Part 6) to the dashboard UI and AI Copilot. All decision results are
scoped to a single dataset and workflow — no cross-dataset leakage,
no global mutable state, no demo-data fallback.

This module:
  - Accepts raw analytical outputs and runs the orchestrator.
  - Formats orchestration results for dashboard rendering.
  - Builds Copilot context from decision results.
  - Provides helpers for rendering cards, statuses, and warnings.

No Streamlit dependency in this module — it is pure Python so tests
run without importing streamlit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

from src.decision.contracts import (
    ConfidenceLevel,
    EvidenceLevel,
    _safe_float,
)
from src.decision.orchestrator import (
    DecisionOrchestrator,
    ExecutionStatus,
    OrchestrationReport,
    OrchestrationRequest,
    StageStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_str(value: Any) -> str:
    """Convert a value to a safe display string."""
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.2f}"
    return str(value)


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


# ---------------------------------------------------------------------------
# Status display helpers
# ---------------------------------------------------------------------------

_STATUS_LABELS = {
    ExecutionStatus.SUCCESS.value: ("Success", "success",
        "All stages completed — decisions are available."),
    ExecutionStatus.PARTIAL_SUCCESS.value: ("Partial Success", "warning",
        "Some stages completed but others encountered issues."),
    ExecutionStatus.INSUFFICIENT_EVIDENCE.value: ("Insufficient Evidence", "info",
        "Not enough evidence to generate confident decisions."),
    ExecutionStatus.FAILED.value: ("Failed", "error",
        "The workflow could not complete. Check errors below."),
}

_STAGE_STATUS_LABELS = {
    StageStatus.COMPLETED.value: ("Completed", "success"),
    StageStatus.SKIPPED.value: ("Skipped", "info"),
    StageStatus.FAILED.value: ("Failed", "error"),
    StageStatus.NOT_REQUESTED.value: ("Not Requested", "secondary"),
}

_CONFIDENCE_LABELS = {
    ConfidenceLevel.HIGH.value: ("High Confidence", "success",
        "Multiple strong evidence items support these findings."),
    ConfidenceLevel.MEDIUM.value: ("Medium Confidence", "warning",
        "Moderate evidence available. Findings should be reviewed."),
    ConfidenceLevel.LOW.value: ("Low Confidence", "warning",
        "Limited evidence. Treat findings as preliminary."),
    ConfidenceLevel.INSUFFICIENT.value: ("Insufficient", "error",
        "Not enough evidence to draw conclusions."),
}

_EVIDENCE_LEVEL_LABELS = {
    EvidenceLevel.FACT.value: ("Measured Fact", "Directly observed or measured data."),
    EvidenceLevel.INFERENCE.value: ("Statistical Inference", "Derived from analysis; correlation, not causation."),
    EvidenceLevel.PREDICTION.value: ("Model Prediction", "Forward-looking estimate with inherent uncertainty."),
    EvidenceLevel.RECOMMENDATION.value: ("Recommendation", "Suggested action based on available evidence."),
    EvidenceLevel.LIMITATION.value: ("Limitation", "Known constraint or gap in the analysis."),
}


def get_status_display(status: str) -> dict:
    """Return display info for an execution status."""
    label, level, desc = _STATUS_LABELS.get(
        status, ("Unknown", "secondary", "Status could not be determined.")
    )
    return {"label": label, "level": level, "description": desc, "raw": status}


def get_stage_status_display(status: str) -> dict:
    """Return display info for a stage status."""
    label, level = _STAGE_STATUS_LABELS.get(status, ("Unknown", "secondary"))
    return {"label": label, "level": level, "raw": status}


def get_confidence_display(confidence_level: str) -> dict:
    """Return display info for a confidence level."""
    label, level, desc = _CONFIDENCE_LABELS.get(
        confidence_level, ("Unknown", "secondary", "Confidence could not be assessed.")
    )
    return {"label": label, "level": level, "description": desc, "raw": confidence_level}


def get_evidence_level_display(level: str) -> dict:
    """Return display info for an evidence level."""
    label, desc = _EVIDENCE_LEVEL_LABELS.get(level, ("Unknown", "Unknown evidence level."))
    return {"label": label, "description": desc, "raw": level}


# ---------------------------------------------------------------------------
# Decision card formatting
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DecisionCard:
    """A single decision formatted for dashboard display.

    Keeps evidence strength, confidence, impact, urgency, and priority
    semantically separate — as required by the spec.
    """
    title: str = ""
    category: str = ""
    description: str = ""
    priority: str = ""
    urgency: str = ""
    confidence: str = ""
    reversibility: str = ""
    evidence_ids: list = field(default_factory=list)
    evidence_summary: str = ""
    impact_summary: str = ""
    limitations: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    dataset_id: str = ""

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "category": self.category,
            "description": self.description,
            "priority": self.priority,
            "urgency": self.urgency,
            "confidence": self.confidence,
            "reversibility": self.reversibility,
            "evidence_ids": list(self.evidence_ids),
            "evidence_summary": self.evidence_summary,
            "impact_summary": self.impact_summary,
            "limitations": list(self.limitations),
            "warnings": list(self.warnings),
            "dataset_id": self.dataset_id,
        }


def format_decision_cards(report: OrchestrationReport) -> list[DecisionCard]:
    """Extract decision options from an OrchestrationReport into display cards."""
    cards = []
    if report.decision_report is None:
        return cards

    decisions = _get_attr(report.decision_report, "decisions", [])
    for dec in decisions:
        title = _get_attr(dec, "title", "Untitled Decision")
        category = _get_attr(dec, "category", "")
        description = _get_attr(dec, "description", "")
        priority = _get_attr(dec, "priority", "")
        urgency = _get_attr(dec, "urgency", "")
        confidence = _get_attr(dec, "confidence_level", "")
        reversibility = _get_attr(dec, "reversibility", "")
        evidence_ids = list(_get_attr(dec, "evidence_ids", []))
        dataset_id = _get_attr(dec, "dataset_id", report.dataset_id)

        # Impact summary — from the decision's quantified impact
        impact_magnitude = _get_attr(dec, "impact_magnitude", None)
        impact_type = _get_attr(dec, "impact_type", "")
        if impact_magnitude is not None:
            safe_mag = _safe_float(impact_magnitude)
            if safe_mag is not None:
                impact_summary = f"{impact_type}: magnitude {safe_mag:.2f}"
            else:
                impact_summary = f"{impact_type}: magnitude unavailable"
        else:
            impact_summary = ""

        # Limitations from the decision
        dec_limitations = [
            _safe_str(_get_attr(lim, "description", str(lim)))
            for lim in _get_attr(dec, "limitations", [])
        ]

        # Warnings
        dec_warnings = list(_get_attr(dec, "warnings", []))

        # Evidence summary text
        ev_count = len(evidence_ids)
        if ev_count == 0:
            evidence_summary = "No direct evidence linked."
        elif ev_count == 1:
            evidence_summary = "Based on 1 evidence item."
        else:
            evidence_summary = f"Based on {ev_count} evidence items."

        cards.append(DecisionCard(
            title=title,
            category=category,
            description=description,
            priority=priority,
            urgency=urgency,
            confidence=confidence,
            reversibility=reversibility,
            evidence_ids=evidence_ids,
            evidence_summary=evidence_summary,
            impact_summary=impact_summary,
            limitations=dec_limitations,
            warnings=dec_warnings,
            dataset_id=dataset_id,
        ))

    return cards


# ---------------------------------------------------------------------------
# Dashboard result container
# ---------------------------------------------------------------------------

@dataclass
class DashboardResult:
    """Complete formatted result for the dashboard UI.

    This is a mutable container used only within a single render cycle.
    Not persisted — rebuilt from the OrchestrationReport each time.
    """
    report: Optional[OrchestrationReport] = None
    dataset_id: str = ""
    workflow_id: str = ""
    status_display: dict = field(default_factory=dict)
    confidence_display: dict = field(default_factory=dict)
    decision_cards: list = field(default_factory=list)
    stage_displays: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    limitations: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    has_decisions: bool = False
    has_scenarios: bool = False
    empty_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "workflow_id": self.workflow_id,
            "status": self.status_display,
            "confidence": self.confidence_display,
            "decision_cards": [c.to_dict() for c in self.decision_cards],
            "stages": self.stage_displays,
            "warnings": self.warnings,
            "limitations": self.limitations,
            "errors": self.errors,
            "has_decisions": self.has_decisions,
            "has_scenarios": self.has_scenarios,
            "empty_reason": self.empty_reason,
        }


def format_dashboard_result(report: OrchestrationReport) -> DashboardResult:
    """Convert an OrchestrationReport into a DashboardResult for rendering."""
    result = DashboardResult()
    result.report = report
    result.dataset_id = report.dataset_id
    result.workflow_id = report.workflow_id
    result.status_display = get_status_display(report.status)

    # Confidence
    conf_level = ""
    if report.evidence_summary and report.evidence_summary.confidence:
        conf_level = _get_attr(report.evidence_summary.confidence, "level", "")
    result.confidence_display = get_confidence_display(conf_level)

    # Decision cards
    result.decision_cards = format_decision_cards(report)
    result.has_decisions = bool(result.decision_cards)

    # Scenarios
    result.has_scenarios = bool(report.scenario_results)

    # Stage displays
    for stage in report.stages:
        s_name = _get_attr(stage, "name", "")
        s_status = _get_attr(stage, "status", "")
        s_error = _get_attr(stage, "error", "")
        s_warnings = _get_attr(stage, "warnings", [])
        s_notes = _get_attr(stage, "processing_notes", [])

        result.stage_displays.append({
            "name": s_name,
            "status": get_stage_status_display(s_status),
            "error": s_error,
            "warnings": list(s_warnings),
            "notes": list(s_notes),
        })

    # Warnings and limitations
    result.warnings = list(report.warnings)
    result.errors = list(report.errors)

    # Limitations
    for lim in report.limitations:
        if hasattr(lim, "to_dict"):
            result.limitations.append(_get_attr(lim, "description", str(lim)))
        else:
            result.limitations.append(str(lim))

    # Empty-state explanation
    if not result.has_decisions:
        if report.status == ExecutionStatus.INSUFFICIENT_EVIDENCE.value:
            result.empty_reason = (
                "The workflow completed but there is not enough evidence to "
                "generate decisions. Upload more data or run additional "
                "analysis to provide evidence."
            )
        elif report.status == ExecutionStatus.FAILED.value:
            result.empty_reason = (
                "The workflow failed before decisions could be generated. "
                "Check the errors below for details."
            )
        else:
            result.empty_reason = (
                "No decisions were generated from the available data. "
                "This may indicate insufficient analytical results."
            )

    return result


# ---------------------------------------------------------------------------
# Run orchestration for a dataset
# ---------------------------------------------------------------------------

def run_decision_workflow(
    dataset_id: str,
    *,
    evidence_items: list | None = None,
    analytics_result: Any = None,
    prediction_report: Any = None,
    phase5_result: Any = None,
    marketing_result: Any = None,
    scenario_definitions: list | None = None,
    scenario_model_info: Any = None,
    evaluate_scenarios: bool = False,
) -> DashboardResult:
    """Run the full decision orchestration for a dataset and return
    a formatted DashboardResult.

    This is the main entry point for the dashboard page. It creates
    a fresh orchestrator per call — no state leaks between calls.
    """
    if not dataset_id or not str(dataset_id).strip():
        # Return a failed result for invalid dataset
        result = DashboardResult()
        result.dataset_id = ""
        result.status_display = get_status_display(ExecutionStatus.FAILED.value)
        result.errors = ["No dataset is currently active."]
        result.empty_reason = "No dataset is active. Upload or select a dataset first."
        return result

    request = OrchestrationRequest(
        dataset_id=dataset_id,
        evidence_items=evidence_items or [],
        analytics_result=analytics_result,
        prediction_report=prediction_report,
        phase5_result=phase5_result,
        marketing_result=marketing_result,
        scenario_definitions=scenario_definitions or [],
        scenario_model_info=scenario_model_info,
        evaluate_scenarios=evaluate_scenarios,
    )

    orchestrator = DecisionOrchestrator()
    report = orchestrator.run(request)

    return format_dashboard_result(report)


# ---------------------------------------------------------------------------
# Copilot context builder
# ---------------------------------------------------------------------------

def build_copilot_context(
    dashboard_result: Optional[DashboardResult],
    dataset_id: str,
) -> dict:
    """Build a context dict for the AI Copilot from decision results.

    Rules:
      - Only the active dataset's results are included.
      - No secrets, credentials, or raw data exposed.
      - Facts, inferences, predictions, and recommendations are
        distinguished.
      - Confidence and limitations are communicated.
      - Returns an empty context if no results are available.
    """
    if dashboard_result is None:
        return _empty_copilot_context(dataset_id)

    # Verify dataset match
    if dashboard_result.dataset_id != dataset_id:
        return _empty_copilot_context(dataset_id)

    context = {
        "dataset_id": dataset_id,
        "workflow_id": dashboard_result.workflow_id,
        "has_decisions": dashboard_result.has_decisions,
        "status": dashboard_result.status_display.get("label", "Unknown"),
        "confidence": dashboard_result.confidence_display.get("label", "Unknown"),
        "confidence_description": dashboard_result.confidence_display.get("description", ""),
        "decision_count": len(dashboard_result.decision_cards),
        "decisions": [],
        "warnings": dashboard_result.warnings[:10],  # Limit to prevent context bloat
        "limitations": dashboard_result.limitations[:10],
        "has_scenarios": dashboard_result.has_scenarios,
        "empty_reason": dashboard_result.empty_reason,
    }

    for card in dashboard_result.decision_cards:
        context["decisions"].append({
            "title": card.title,
            "category": card.category,
            "description": card.description,
            "priority": card.priority,
            "urgency": card.urgency,
            "confidence": card.confidence,
            "evidence_summary": card.evidence_summary,
            "impact_summary": card.impact_summary,
            "limitations": card.limitations[:5],
        })

    return context


def _empty_copilot_context(dataset_id: str) -> dict:
    """Return an empty context when no decision results are available."""
    return {
        "dataset_id": dataset_id,
        "workflow_id": "",
        "has_decisions": False,
        "status": "Not Run",
        "confidence": "",
        "confidence_description": "",
        "decision_count": 0,
        "decisions": [],
        "warnings": [],
        "limitations": [],
        "has_scenarios": False,
        "empty_reason": "Decision Intelligence has not been run for this dataset.",
    }


def format_copilot_answer_context(context: dict) -> str:
    """Format the copilot context into a text block the AI analyst
    can use to answer decision-related questions.

    Keeps facts, inferences, predictions, and recommendations
    semantically separate. Never fabricates findings.
    """
    if not context.get("has_decisions"):
        reason = context.get("empty_reason", "No decision results are available.")
        return (
            f"Decision Intelligence status: {context.get('status', 'Not Run')}. "
            f"{reason}"
        )

    parts = []
    parts.append(
        f"Decision Intelligence status: {context.get('status', 'Unknown')}. "
        f"Confidence: {context.get('confidence', 'Unknown')}. "
        f"{context.get('confidence_description', '')}"
    )

    for i, dec in enumerate(context.get("decisions", []), 1):
        parts.append(
            f"\nDecision {i}: {dec.get('title', 'Untitled')}\n"
            f"  Category: {dec.get('category', 'N/A')}\n"
            f"  Priority: {dec.get('priority', 'N/A')} | "
            f"Urgency: {dec.get('urgency', 'N/A')} | "
            f"Confidence: {dec.get('confidence', 'N/A')}\n"
            f"  {dec.get('description', '')}\n"
            f"  Evidence: {dec.get('evidence_summary', 'None')}\n"
            f"  Impact: {dec.get('impact_summary', 'Not quantified')}"
        )
        if dec.get("limitations"):
            parts.append(f"  Limitations: {'; '.join(dec['limitations'])}")

    if context.get("warnings"):
        parts.append(f"\nWarnings: {'; '.join(context['warnings'][:5])}")
    if context.get("limitations"):
        parts.append(f"\nAnalysis Limitations: {'; '.join(context['limitations'][:5])}")

    return "\n".join(parts)
