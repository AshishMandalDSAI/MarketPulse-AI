"""Phase 7 Part 6 — Decision Orchestrator.

Deterministic, domain-agnostic orchestrator that coordinates Parts 1–5
components into a single end-to-end decision workflow:

    1. Validate the request and dataset scope.
    2. Classify and validate available evidence.
    3. Assess evidence confidence and limitations.
    4. Synthesize relevant signals.
    5. Quantify impacts.
    6. Evaluate scenarios (when supplied).
    7. Generate structured decisions.
    8. Assemble a traceable orchestration report.

No LLM calls, no global mutable state, no demo-data fallback.
"""

from __future__ import annotations

import enum
import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Imports from Parts 1–5
# ---------------------------------------------------------------------------

from src.decision.contracts import (
    ConfidenceLevel,
    ConfidenceScore,
    DecisionEvidence,
    DecisionLimitation,
    DecisionSignal,
    EvidenceLevel,
    LimitationCategory,
    _make_id,
    _safe_float,
)
from src.decision.evidence import (
    classify_evidence_level,
    create_evidence,
    create_limitation,
    create_signal,
    ensure_non_causal,
    ensure_prediction_uncertainty,
    validate_dataset_scope,
)
from src.decision.confidence import (
    assess_confidence,
    assess_signal_confidence,
    aggregate_confidence,
)
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
    ScenarioResult,
    ScenarioComparison,
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    """Get attribute from an object or dict, with a default."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _safe_float_value(value: Any) -> Optional[float]:
    """Convert to float, replacing NaN/Inf with None."""
    return _safe_float(value)


# ---------------------------------------------------------------------------
# Execution Status
# ---------------------------------------------------------------------------

class ExecutionStatus(str, enum.Enum):
    """Overall orchestration execution status."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Stage Status
# ---------------------------------------------------------------------------

class StageStatus(str, enum.Enum):
    """Status of an individual orchestration stage."""
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"
    NOT_REQUESTED = "not_requested"


# ---------------------------------------------------------------------------
# Stage Result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StageResult:
    """Result of a single orchestration stage."""
    name: str = ""
    status: str = StageStatus.SKIPPED.value
    error: str = ""
    warnings: list = field(default_factory=list)
    processing_notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "error": self.error,
            "warnings": list(self.warnings),
            "processing_notes": list(self.processing_notes),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StageResult":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Orchestration Request
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OrchestrationRequest:
    """Immutable request for a decision orchestration workflow.

    All inputs are optional except dataset_id. The orchestrator generates
    decisions from whatever is provided.
    """
    dataset_id: str = ""
    evidence_items: list = field(default_factory=list)
    analytics_result: Any = None
    prediction_report: Any = None
    phase5_result: Any = None
    marketing_result: Any = None
    scenario_definitions: list = field(default_factory=list)
    scenario_model_info: Any = None
    evaluate_scenarios: bool = False

    def __post_init__(self):
        if not self.dataset_id or not str(self.dataset_id).strip():
            raise ValueError(
                "dataset_id is required. Every workflow must be explicitly "
                "scoped to a dataset."
            )

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "evidence_count": len(self.evidence_items),
            "has_analytics": self.analytics_result is not None,
            "has_predictions": self.prediction_report is not None,
            "has_phase5": self.phase5_result is not None,
            "has_marketing": self.marketing_result is not None,
            "scenario_count": len(self.scenario_definitions),
            "evaluate_scenarios": self.evaluate_scenarios,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OrchestrationRequest":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Evidence Summary
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvidenceSummary:
    """Summary of evidence used in the orchestration."""
    total_count: int = 0
    fact_count: int = 0
    inference_count: int = 0
    prediction_count: int = 0
    limitation_count: int = 0
    recommendation_count: int = 0
    dataset_id: str = ""
    sources: list = field(default_factory=list)
    confidence: Optional[ConfidenceScore] = None

    def to_dict(self) -> dict:
        return {
            "total_count": self.total_count,
            "fact_count": self.fact_count,
            "inference_count": self.inference_count,
            "prediction_count": self.prediction_count,
            "limitation_count": self.limitation_count,
            "recommendation_count": self.recommendation_count,
            "dataset_id": self.dataset_id,
            "sources": list(self.sources),
            "confidence": self.confidence.to_dict() if self.confidence else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EvidenceSummary":
        raw = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if "confidence" in raw and isinstance(raw["confidence"], dict):
            raw["confidence"] = ConfidenceScore.from_dict(raw["confidence"])
        return cls(**raw)


# ---------------------------------------------------------------------------
# Orchestration Report
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OrchestrationReport:
    """Complete result of a decision orchestration workflow.

    Captures every stage's status, all intermediate results, and the
    final decision report — traceable from evidence to decisions.
    """
    workflow_id: str = field(default_factory=lambda: _make_id("wf_"))
    dataset_id: str = ""
    status: str = ExecutionStatus.FAILED.value
    evidence_summary: Optional[EvidenceSummary] = None
    synthesis_result: Optional[SynthesisResult] = None
    impact_report: Optional[ImpactReport] = None
    scenario_results: list = field(default_factory=list)
    decision_report: Optional[DecisionReport] = None
    stages: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    limitations: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    processing_notes: list = field(default_factory=list)

    def decisions(self) -> list:
        """Return all generated decisions."""
        if self.decision_report is not None:
            return list(self.decision_report.decisions)
        return []

    def stage_by_name(self, name: str) -> Optional[StageResult]:
        """Find a stage result by name."""
        for stage in self.stages:
            s_name = _get_attr(stage, "name", "")
            if s_name == name:
                return stage
        return None

    def to_dict(self) -> dict:
        return {
            "workflow_id": self.workflow_id,
            "dataset_id": self.dataset_id,
            "status": self.status,
            "evidence_summary": (
                self.evidence_summary.to_dict()
                if self.evidence_summary else None
            ),
            "synthesis_result": (
                self.synthesis_result.to_dict()
                if self.synthesis_result and hasattr(self.synthesis_result, "to_dict")
                else None
            ),
            "impact_report": (
                self.impact_report.to_dict()
                if self.impact_report and hasattr(self.impact_report, "to_dict")
                else None
            ),
            "scenario_results": [
                sr.to_dict() if hasattr(sr, "to_dict") else sr
                for sr in self.scenario_results
            ],
            "decision_report": (
                self.decision_report.to_dict()
                if self.decision_report else None
            ),
            "stages": [
                s.to_dict() if hasattr(s, "to_dict") else s
                for s in self.stages
            ],
            "warnings": list(self.warnings),
            "limitations": [
                lim.to_dict() if hasattr(lim, "to_dict") else lim
                for lim in self.limitations
            ],
            "errors": list(self.errors),
            "processing_notes": list(self.processing_notes),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OrchestrationReport":
        raw = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if "evidence_summary" in raw and isinstance(raw["evidence_summary"], dict):
            raw["evidence_summary"] = EvidenceSummary.from_dict(
                raw["evidence_summary"]
            )
        if "decision_report" in raw and isinstance(raw["decision_report"], dict):
            raw["decision_report"] = DecisionReport.from_dict(
                raw["decision_report"]
            )
        if "stages" in raw and isinstance(raw["stages"], list):
            raw["stages"] = [
                StageResult.from_dict(s) if isinstance(s, dict) else s
                for s in raw["stages"]
            ]
        if "limitations" in raw and isinstance(raw["limitations"], list):
            raw["limitations"] = [
                DecisionLimitation.from_dict(lim) if isinstance(lim, dict) else lim
                for lim in raw["limitations"]
            ]
        # scenario_results and synthesis_result are complex — kept as dicts
        # from_dict is best-effort for serialization round-trip
        return cls(**raw)


# ---------------------------------------------------------------------------
# Decision Orchestrator
# ---------------------------------------------------------------------------

class DecisionOrchestrator:
    """Coordinates Parts 1–5 into a single end-to-end decision workflow.

    Usage:
        orchestrator = DecisionOrchestrator()
        request = OrchestrationRequest(
            dataset_id="ds_001",
            evidence_items=[...],
            analytics_result=analytics_result,
        )
        report = orchestrator.run(request)

    The orchestrator is stateless between runs — each run is independent.
    No global mutable state, no caches, no demo-data fallback.
    """

    def run(self, request: OrchestrationRequest) -> OrchestrationReport:
        """Execute the full orchestration workflow.

        Steps:
            1. Validate request and dataset scope.
            2. Classify and validate evidence.
            3. Assess evidence confidence.
            4. Synthesize signals.
            5. Quantify impacts.
            6. Evaluate scenarios (if requested).
            7. Generate decisions.
            8. Assemble report.

        Each stage records its status. A failed stage does not prevent
        subsequent stages from running where possible.

        Returns:
            OrchestrationReport with complete results and traceability.
        """
        # Fresh state per run — no leakage between workflows
        dataset_id = request.dataset_id
        workflow_id = _make_id("wf_")
        stages: list[StageResult] = []
        warnings: list[str] = []
        limitations: list[DecisionLimitation] = []
        errors: list[str] = []
        processing_notes: list[str] = []

        # Intermediate results
        classified_evidence: list[DecisionEvidence] = []
        confidence_score: Optional[ConfidenceScore] = None
        synthesis_result: Optional[SynthesisResult] = None
        impact_report: Optional[ImpactReport] = None
        scenario_results: list[ScenarioResult] = []
        decision_report: Optional[DecisionReport] = None

        # -----------------------------------------------------------------
        # Stage 1: Validate request and dataset scope
        # -----------------------------------------------------------------
        stage1 = self._stage_validate_request(request, dataset_id)
        stages.append(stage1)

        if stage1.status == StageStatus.FAILED.value:
            errors.extend(stage1.warnings)
            if stage1.error:
                errors.append(stage1.error)
            return OrchestrationReport(
                workflow_id=workflow_id,
                dataset_id=dataset_id,
                status=ExecutionStatus.FAILED.value,
                stages=stages,
                warnings=warnings,
                limitations=limitations,
                errors=errors,
                processing_notes=processing_notes,
            )

        # -----------------------------------------------------------------
        # Stage 2: Classify and validate evidence
        # -----------------------------------------------------------------
        stage2, classified_evidence = self._stage_classify_evidence(
            request, dataset_id,
        )
        stages.append(stage2)
        warnings.extend(stage2.warnings)
        processing_notes.extend(stage2.processing_notes)

        # -----------------------------------------------------------------
        # Stage 3: Assess confidence
        # -----------------------------------------------------------------
        stage3, confidence_score = self._stage_assess_confidence(
            classified_evidence, dataset_id,
        )
        stages.append(stage3)
        warnings.extend(stage3.warnings)
        processing_notes.extend(stage3.processing_notes)

        # Check for insufficient evidence
        is_insufficient = (
            confidence_score is not None
            and confidence_score.level == ConfidenceLevel.INSUFFICIENT.value
            and not request.analytics_result
            and not request.prediction_report
            and not request.phase5_result
            and not request.marketing_result
        )

        # -----------------------------------------------------------------
        # Stage 4: Synthesize signals
        # -----------------------------------------------------------------
        stage4, synthesis_result = self._stage_synthesize_signals(
            request, classified_evidence, dataset_id,
        )
        stages.append(stage4)
        warnings.extend(stage4.warnings)
        processing_notes.extend(stage4.processing_notes)
        if stage4.error:
            errors.append(stage4.error)

        # -----------------------------------------------------------------
        # Stage 5: Quantify impacts
        # -----------------------------------------------------------------
        stage5, impact_report = self._stage_quantify_impacts(
            synthesis_result, dataset_id,
        )
        stages.append(stage5)
        warnings.extend(stage5.warnings)
        processing_notes.extend(stage5.processing_notes)
        if stage5.error:
            errors.append(stage5.error)

        # -----------------------------------------------------------------
        # Stage 6: Evaluate scenarios (optional)
        # -----------------------------------------------------------------
        stage6, scenario_results = self._stage_evaluate_scenarios(
            request, classified_evidence, dataset_id,
        )
        stages.append(stage6)
        warnings.extend(stage6.warnings)
        processing_notes.extend(stage6.processing_notes)
        if stage6.error:
            errors.append(stage6.error)

        # -----------------------------------------------------------------
        # Stage 7: Generate decisions
        # -----------------------------------------------------------------
        stage7, decision_report = self._stage_generate_decisions(
            synthesis_result, impact_report, scenario_results, dataset_id,
        )
        stages.append(stage7)
        warnings.extend(stage7.warnings)
        processing_notes.extend(stage7.processing_notes)
        if stage7.error:
            errors.append(stage7.error)

        # -----------------------------------------------------------------
        # Stage 8: Assemble report
        # -----------------------------------------------------------------

        # Build evidence summary
        evidence_summary = self._build_evidence_summary(
            classified_evidence, confidence_score, dataset_id,
        )

        # Collect limitations from all stages
        if synthesis_result is not None:
            for sig in _get_attr(synthesis_result, "signals", []):
                signal_obj = _get_attr(sig, "signal", None)
                if signal_obj:
                    for lim in _get_attr(signal_obj, "limitations", []):
                        if hasattr(lim, "to_dict"):
                            limitations.append(lim)

        # Determine overall status
        status = self._determine_status(
            stages, classified_evidence, decision_report,
            is_insufficient,
        )

        return OrchestrationReport(
            workflow_id=workflow_id,
            dataset_id=dataset_id,
            status=status,
            evidence_summary=evidence_summary,
            synthesis_result=synthesis_result,
            impact_report=impact_report,
            scenario_results=scenario_results,
            decision_report=decision_report,
            stages=stages,
            warnings=warnings,
            limitations=limitations,
            errors=errors,
            processing_notes=processing_notes,
        )

    # -------------------------------------------------------------------
    # Stage implementations
    # -------------------------------------------------------------------

    def _stage_validate_request(
        self,
        request: OrchestrationRequest,
        dataset_id: str,
    ) -> StageResult:
        """Stage 1: Validate request and dataset scope."""
        warnings = []
        notes = []

        # Check for cross-dataset evidence
        cross_dataset_items = []
        for ev in request.evidence_items:
            ev_ds = _get_attr(ev, "dataset_id", "")
            if ev_ds and ev_ds != dataset_id:
                cross_dataset_items.append(ev_ds)

        if cross_dataset_items:
            warnings.append(
                f"Evidence items from other datasets detected and will be "
                f"filtered: {sorted(set(cross_dataset_items))}"
            )

        # Check for cross-dataset component results
        for label, obj in [
            ("analytics_result", request.analytics_result),
            ("prediction_report", request.prediction_report),
            ("phase5_result", request.phase5_result),
            ("marketing_result", request.marketing_result),
        ]:
            if obj is not None:
                obj_ds = _get_attr(obj, "dataset_id", "")
                if obj_ds and obj_ds != dataset_id:
                    return StageResult(
                        name="validate_request",
                        status=StageStatus.FAILED.value,
                        error=(
                            f"Cross-dataset component result rejected: "
                            f"{label} has dataset_id '{obj_ds}', expected "
                            f"'{dataset_id}'."
                        ),
                        warnings=warnings,
                    )

        has_any_input = (
            bool(request.evidence_items)
            or request.analytics_result is not None
            or request.prediction_report is not None
            or request.phase5_result is not None
            or request.marketing_result is not None
        )

        if not has_any_input:
            notes.append("No evidence or analytical results provided.")

        return StageResult(
            name="validate_request",
            status=StageStatus.COMPLETED.value,
            warnings=warnings,
            processing_notes=notes,
        )

    def _stage_classify_evidence(
        self,
        request: OrchestrationRequest,
        dataset_id: str,
    ) -> tuple[StageResult, list]:
        """Stage 2: Classify and validate evidence."""
        warnings = []
        notes = []
        classified = []

        if not request.evidence_items:
            notes.append("No evidence items provided for classification.")
            return (
                StageResult(
                    name="classify_evidence",
                    status=StageStatus.COMPLETED.value,
                    processing_notes=notes,
                ),
                classified,
            )

        # Filter to dataset scope
        scoped = validate_dataset_scope(request.evidence_items, dataset_id)

        filtered_count = len(request.evidence_items) - len(scoped)
        if filtered_count > 0:
            warnings.append(
                f"Filtered {filtered_count} evidence item(s) from other datasets."
            )

        # Apply safety guards
        for ev in scoped:
            try:
                level = _get_attr(ev, "level", "")
                if level == EvidenceLevel.INFERENCE.value:
                    ev = ensure_non_causal(ev)
                elif level == EvidenceLevel.PREDICTION.value:
                    ev = ensure_prediction_uncertainty(ev)
                classified.append(ev)
            except Exception as exc:
                notes.append(f"Failed to classify evidence: {exc}")

        notes.append(f"Classified {len(classified)} evidence item(s).")

        return (
            StageResult(
                name="classify_evidence",
                status=StageStatus.COMPLETED.value,
                warnings=warnings,
                processing_notes=notes,
            ),
            classified,
        )

    def _stage_assess_confidence(
        self,
        evidence_items: list,
        dataset_id: str,
    ) -> tuple[StageResult, Optional[ConfidenceScore]]:
        """Stage 3: Assess evidence confidence."""
        notes = []

        if not evidence_items:
            score = ConfidenceScore(
                level=ConfidenceLevel.INSUFFICIENT.value,
                explanation="No evidence available for assessment.",
                evidence_count=0,
                evidence_quality="none",
                dataset_id=dataset_id,
            )
            notes.append("No evidence to assess confidence on.")
            return (
                StageResult(
                    name="assess_confidence",
                    status=StageStatus.COMPLETED.value,
                    processing_notes=notes,
                ),
                score,
            )

        try:
            score = assess_confidence(evidence_items, dataset_id=dataset_id)
            notes.append(
                f"Confidence assessed as {score.level} based on "
                f"{score.evidence_count} evidence item(s)."
            )
            return (
                StageResult(
                    name="assess_confidence",
                    status=StageStatus.COMPLETED.value,
                    processing_notes=notes,
                ),
                score,
            )
        except Exception as exc:
            return (
                StageResult(
                    name="assess_confidence",
                    status=StageStatus.FAILED.value,
                    error=f"Confidence assessment failed: {exc}",
                ),
                None,
            )

    def _stage_synthesize_signals(
        self,
        request: OrchestrationRequest,
        evidence_items: list,
        dataset_id: str,
    ) -> tuple[StageResult, Optional[SynthesisResult]]:
        """Stage 4: Synthesize signals from analytical results."""
        notes = []

        has_ingestible = (
            request.analytics_result is not None
            or request.prediction_report is not None
            or request.phase5_result is not None
            or request.marketing_result is not None
            or bool(evidence_items)
        )

        if not has_ingestible:
            notes.append("No analytical results or evidence to synthesize.")
            return (
                StageResult(
                    name="synthesize_signals",
                    status=StageStatus.SKIPPED.value,
                    processing_notes=notes,
                ),
                None,
            )

        try:
            synthesizer = SignalSynthesizer(dataset_id=dataset_id)

            # Ingest analytical results from each phase
            if request.analytics_result is not None:
                try:
                    synthesizer.ingest_analytics(request.analytics_result)
                except Exception as exc:
                    notes.append(f"Analytics ingestion warning: {exc}")

            if request.prediction_report is not None:
                try:
                    synthesizer.ingest_predictions(request.prediction_report)
                except Exception as exc:
                    notes.append(f"Prediction ingestion warning: {exc}")

            if request.phase5_result is not None:
                try:
                    synthesizer.ingest_anomalies(request.phase5_result)
                except Exception as exc:
                    notes.append(f"Phase 5 ingestion warning: {exc}")

            if request.marketing_result is not None:
                try:
                    synthesizer.ingest_marketing(request.marketing_result)
                except Exception as exc:
                    notes.append(f"Marketing ingestion warning: {exc}")

            # Ingest raw evidence items
            if evidence_items:
                try:
                    synthesizer.ingest_evidence(evidence_items)
                except Exception as exc:
                    notes.append(f"Evidence ingestion warning: {exc}")

            result = synthesizer.synthesize()
            signal_count = len(_get_attr(result, "signals", []))
            notes.append(f"Synthesized {signal_count} signal(s).")

            return (
                StageResult(
                    name="synthesize_signals",
                    status=StageStatus.COMPLETED.value,
                    processing_notes=notes,
                ),
                result,
            )
        except Exception as exc:
            return (
                StageResult(
                    name="synthesize_signals",
                    status=StageStatus.FAILED.value,
                    error=f"Signal synthesis failed: {exc}",
                    processing_notes=notes,
                ),
                None,
            )

    def _stage_quantify_impacts(
        self,
        synthesis_result: Optional[SynthesisResult],
        dataset_id: str,
    ) -> tuple[StageResult, Optional[ImpactReport]]:
        """Stage 5: Quantify impacts from synthesized signals."""
        notes = []

        if synthesis_result is None:
            notes.append("No synthesis result available for impact quantification.")
            return (
                StageResult(
                    name="quantify_impacts",
                    status=StageStatus.SKIPPED.value,
                    processing_notes=notes,
                ),
                None,
            )

        signals = _get_attr(synthesis_result, "signals", [])
        if not signals:
            notes.append("No signals to quantify impact for.")
            return (
                StageResult(
                    name="quantify_impacts",
                    status=StageStatus.SKIPPED.value,
                    processing_notes=notes,
                ),
                None,
            )

        try:
            quantifier = ImpactQuantifier(dataset_id=dataset_id)
            evidence_registry = _get_attr(
                synthesis_result, "evidence_registry", {}
            )
            report = quantifier.assess_signals(signals, evidence_registry)
            assessment_count = len(_get_attr(report, "assessments", []))
            notes.append(f"Quantified {assessment_count} impact assessment(s).")

            return (
                StageResult(
                    name="quantify_impacts",
                    status=StageStatus.COMPLETED.value,
                    processing_notes=notes,
                ),
                report,
            )
        except Exception as exc:
            return (
                StageResult(
                    name="quantify_impacts",
                    status=StageStatus.FAILED.value,
                    error=f"Impact quantification failed: {exc}",
                    processing_notes=notes,
                ),
                None,
            )

    def _stage_evaluate_scenarios(
        self,
        request: OrchestrationRequest,
        evidence_items: list,
        dataset_id: str,
    ) -> tuple[StageResult, list]:
        """Stage 6: Evaluate scenarios (optional)."""
        notes = []
        results = []

        if not request.evaluate_scenarios:
            return (
                StageResult(
                    name="evaluate_scenarios",
                    status=StageStatus.NOT_REQUESTED.value,
                    processing_notes=["Scenario evaluation not requested."],
                ),
                results,
            )

        if not request.scenario_definitions:
            notes.append(
                "Scenario evaluation requested but no scenario definitions supplied."
            )
            return (
                StageResult(
                    name="evaluate_scenarios",
                    status=StageStatus.SKIPPED.value,
                    warnings=["No scenario definitions provided."],
                    processing_notes=notes,
                ),
                results,
            )

        try:
            engine = ScenarioEngine(dataset_id=dataset_id)

            for defn in request.scenario_definitions:
                try:
                    # Validate dataset scope
                    defn_ds = _get_attr(defn, "dataset_id", "")
                    if defn_ds and defn_ds != dataset_id:
                        notes.append(
                            f"Skipped scenario with dataset_id '{defn_ds}' "
                            f"(expected '{dataset_id}')."
                        )
                        continue

                    result = engine.evaluate_scenario(
                        defn,
                        evidence_items,
                        request.scenario_model_info,
                    )
                    results.append(result)
                except Exception as exc:
                    defn_name = _get_attr(defn, "name", "unknown")
                    notes.append(
                        f"Scenario '{defn_name}' evaluation failed: {exc}"
                    )

            notes.append(f"Evaluated {len(results)} scenario(s).")

            return (
                StageResult(
                    name="evaluate_scenarios",
                    status=StageStatus.COMPLETED.value,
                    processing_notes=notes,
                ),
                results,
            )
        except Exception as exc:
            return (
                StageResult(
                    name="evaluate_scenarios",
                    status=StageStatus.FAILED.value,
                    error=f"Scenario evaluation failed: {exc}",
                    processing_notes=notes,
                ),
                results,
            )

    def _stage_generate_decisions(
        self,
        synthesis_result: Optional[SynthesisResult],
        impact_report: Optional[ImpactReport],
        scenario_results: list,
        dataset_id: str,
    ) -> tuple[StageResult, Optional[DecisionReport]]:
        """Stage 7: Generate decisions from all available inputs."""
        notes = []

        has_any = (
            synthesis_result is not None
            or impact_report is not None
            or bool(scenario_results)
        )

        if not has_any:
            notes.append("No inputs available for decision generation.")
            return (
                StageResult(
                    name="generate_decisions",
                    status=StageStatus.SKIPPED.value,
                    processing_notes=notes,
                ),
                None,
            )

        try:
            generator = DecisionGenerator(dataset_id=dataset_id)
            report = generator.generate(
                synthesis_result=synthesis_result,
                impact_report=impact_report,
                scenario_results=scenario_results,
            )
            decision_count = len(_get_attr(report, "decisions", []))
            notes.append(f"Generated {decision_count} decision(s).")

            return (
                StageResult(
                    name="generate_decisions",
                    status=StageStatus.COMPLETED.value,
                    processing_notes=notes,
                ),
                report,
            )
        except Exception as exc:
            return (
                StageResult(
                    name="generate_decisions",
                    status=StageStatus.FAILED.value,
                    error=f"Decision generation failed: {exc}",
                    processing_notes=notes,
                ),
                None,
            )

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _build_evidence_summary(
        self,
        evidence_items: list,
        confidence: Optional[ConfidenceScore],
        dataset_id: str,
    ) -> EvidenceSummary:
        """Build a summary of evidence used in the workflow."""
        sources = set()
        fact_count = 0
        inference_count = 0
        prediction_count = 0
        limitation_count = 0
        recommendation_count = 0

        for ev in evidence_items:
            level = _get_attr(ev, "level", "")
            source = _get_attr(ev, "source", "")
            if source:
                sources.add(source)

            if level == EvidenceLevel.FACT.value:
                fact_count += 1
            elif level == EvidenceLevel.INFERENCE.value:
                inference_count += 1
            elif level == EvidenceLevel.PREDICTION.value:
                prediction_count += 1
            elif level == EvidenceLevel.LIMITATION.value:
                limitation_count += 1
            elif level == EvidenceLevel.RECOMMENDATION.value:
                recommendation_count += 1

        return EvidenceSummary(
            total_count=len(evidence_items),
            fact_count=fact_count,
            inference_count=inference_count,
            prediction_count=prediction_count,
            limitation_count=limitation_count,
            recommendation_count=recommendation_count,
            dataset_id=dataset_id,
            sources=sorted(sources),
            confidence=confidence,
        )

    def _determine_status(
        self,
        stages: list,
        evidence_items: list,
        decision_report: Optional[DecisionReport],
        is_insufficient: bool,
    ) -> str:
        """Determine overall orchestration status."""
        # Check for any failed stages
        failed_stages = [
            s for s in stages
            if _get_attr(s, "status", "") == StageStatus.FAILED.value
        ]
        completed_stages = [
            s for s in stages
            if _get_attr(s, "status", "") == StageStatus.COMPLETED.value
        ]

        # All stages failed
        if failed_stages and not completed_stages:
            return ExecutionStatus.FAILED.value

        # Insufficient evidence with no results from analytical stages
        if is_insufficient and (
            decision_report is None
            or not _get_attr(decision_report, "decisions", [])
        ):
            return ExecutionStatus.INSUFFICIENT_EVIDENCE.value

        # Some stages failed
        if failed_stages:
            return ExecutionStatus.PARTIAL_SUCCESS.value

        # No decisions generated but stages completed
        if decision_report is None or not _get_attr(
            decision_report, "decisions", []
        ):
            if not evidence_items:
                return ExecutionStatus.INSUFFICIENT_EVIDENCE.value
            return ExecutionStatus.PARTIAL_SUCCESS.value

        return ExecutionStatus.SUCCESS.value
