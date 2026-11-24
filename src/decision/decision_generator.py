"""Phase 7 Part 5 — Domain-Agnostic Decision Generator.

Deterministic engine that converts supported evidence, decision signals,
impact assessments, and scenario results into structured decision options
and actionable recommendations.

Key design principles:
1. Generate decisions only from available, relevant evidence.
2. Never invent evidence, numerical benefits, financial returns, or business facts.
3. Distinguish measured outcomes from model estimates and hypothetical scenarios.
4. Do not present recommendations as guaranteed results.
5. Do not convert correlation into causation.
6. Do not interpret model coefficients or feature importance as causal impact.
7. Do not rank decisions by confidence alone.
8. Keep confidence, impact magnitude, urgency, and priority conceptually separate.
9. Explain the basis for every recommendation.
10. Surface contradictory evidence and material limitations.
11. When evidence is insufficient, prefer investigation or monitoring over unsupported action.
12. Avoid irreversible or high-impact recommendations when the evidence is weak.
13. Do not claim that an action will solve a problem unless the evidence supports that claim.
14. Keep recommendations domain-agnostic.
15. Never silently mix evidence from different datasets.
16. Do not use an LLM to invent or rank recommendations.
17. Do not introduce global mutable caches or demo-data fallback behavior.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Optional

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
from src.decision.confidence import (
    assess_confidence,
    aggregate_confidence,
)
from src.decision.evidence import (
    create_limitation,
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


# ---------------------------------------------------------------------------
# Decision Category — what kind of action is recommended
# ---------------------------------------------------------------------------

class DecisionCategory(str, enum.Enum):
    """Category of a decision option."""
    RECOMMEND = "recommend"                # Action recommended based on evidence
    INVESTIGATE = "investigate"            # Further investigation needed
    MONITOR = "monitor"                    # Monitor and track over time
    MITIGATE_RISK = "mitigate_risk"        # Risk mitigation action
    BLOCKED = "blocked"                    # Decision blocked by insufficient evidence


# ---------------------------------------------------------------------------
# Priority Level — urgency-independent priority
# ---------------------------------------------------------------------------

class PriorityLevel(str, enum.Enum):
    """Priority level for a decision option.

    Priority is determined by a composite of evidence strength,
    impact magnitude, urgency, and risk — NOT by confidence alone.
    """
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ---------------------------------------------------------------------------
# Urgency Level
# ---------------------------------------------------------------------------

class UrgencyLevel(str, enum.Enum):
    """Urgency level for a decision option.

    Urgency is conceptually separate from priority and confidence.
    """
    IMMEDIATE = "immediate"
    SHORT_TERM = "short_term"
    MEDIUM_TERM = "medium_term"
    LONG_TERM = "long_term"
    NOT_ASSESSED = "not_assessed"


# ---------------------------------------------------------------------------
# Reversibility
# ---------------------------------------------------------------------------

class Reversibility(str, enum.Enum):
    """How reversible a recommended action is."""
    EASILY_REVERSIBLE = "easily_reversible"
    PARTIALLY_REVERSIBLE = "partially_reversible"
    IRREVERSIBLE = "irreversible"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# DecisionOption — one decision recommendation or action
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DecisionOption:
    """A single decision option generated from evidence.

    Captures the recommendation, its supporting evidence, the basis
    for the recommendation, and all limitations and assumptions.
    """
    # Identity
    id: str = field(default_factory=lambda: _make_id("dec_"))
    dataset_id: str = ""

    # Title and description
    title: str = ""
    description: str = ""

    # Decision / action category
    category: str = DecisionCategory.INVESTIGATE.value

    # Recommended action
    recommended_action: str = ""

    # Supporting references
    signal_ids: list = field(default_factory=list)       # list[str]
    evidence_ids: list = field(default_factory=list)     # list[str]
    impact_ids: list = field(default_factory=list)       # list[str]
    scenario_ids: list = field(default_factory=list)     # list[str]

    # Expected or estimated impact (only when defensible)
    estimated_impact: Optional[str] = None               # Human-readable impact summary
    impact_type: str = ""                                 # from ImpactType
    impact_value: Optional[float] = None                  # Numeric if available
    impact_unit: str = ""

    # Confidence
    confidence: Optional[ConfidenceScore] = None

    # Priority
    priority: str = PriorityLevel.MEDIUM.value
    priority_rationale: str = ""

    # Urgency (conceptually separate from priority)
    urgency: str = UrgencyLevel.NOT_ASSESSED.value

    # Assumptions and limitations
    assumptions: list = field(default_factory=list)       # list[str]
    limitations: list = field(default_factory=list)       # list[DecisionLimitation]

    # Risks and trade-offs
    risks: list = field(default_factory=list)              # list[str]
    reversibility: str = Reversibility.UNKNOWN.value

    # Follow-up
    required_follow_up: list = field(default_factory=list) # list[str]

    # Invalidation conditions
    invalidation_conditions: list = field(default_factory=list)  # list[str]

    # Human-readable explanation
    explanation: str = ""

    # Evidence type distinction
    evidence_basis: str = ""  # "measured", "model_estimated", "scenario_based", "qualitative"

    # Contradictions surfaced
    contradictions: list = field(default_factory=list)  # list[str]

    def __post_init__(self):
        if self.category not in {c.value for c in DecisionCategory}:
            raise ValueError(
                f"Invalid decision category: {self.category!r}. "
                f"Must be one of: {[c.value for c in DecisionCategory]}"
            )
        if self.priority not in {p.value for p in PriorityLevel}:
            raise ValueError(
                f"Invalid priority level: {self.priority!r}. "
                f"Must be one of: {[p.value for p in PriorityLevel]}"
            )
        if self.urgency not in {u.value for u in UrgencyLevel}:
            raise ValueError(
                f"Invalid urgency level: {self.urgency!r}. "
                f"Must be one of: {[u.value for u in UrgencyLevel]}"
            )
        if self.reversibility not in {r.value for r in Reversibility}:
            raise ValueError(
                f"Invalid reversibility: {self.reversibility!r}. "
                f"Must be one of: {[r.value for r in Reversibility]}"
            )
        if not self.title:
            raise ValueError("DecisionOption requires a non-empty title.")
        if not self.dataset_id:
            raise ValueError(
                "DecisionOption requires a non-empty dataset_id. "
                "Decisions must be scoped to a dataset."
            )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "dataset_id": self.dataset_id,
            "title": self.title,
            "description": self.description,
            "category": self.category,
            "recommended_action": self.recommended_action,
            "signal_ids": list(self.signal_ids),
            "evidence_ids": list(self.evidence_ids),
            "impact_ids": list(self.impact_ids),
            "scenario_ids": list(self.scenario_ids),
            "estimated_impact": self.estimated_impact,
            "impact_type": self.impact_type,
            "impact_value": _safe_float(self.impact_value),
            "impact_unit": self.impact_unit,
            "confidence": self.confidence.to_dict() if self.confidence else None,
            "priority": self.priority,
            "priority_rationale": self.priority_rationale,
            "urgency": self.urgency,
            "assumptions": list(self.assumptions),
            "limitations": [
                lim.to_dict() if hasattr(lim, "to_dict") else lim
                for lim in self.limitations
            ],
            "risks": list(self.risks),
            "reversibility": self.reversibility,
            "required_follow_up": list(self.required_follow_up),
            "invalidation_conditions": list(self.invalidation_conditions),
            "explanation": self.explanation,
            "evidence_basis": self.evidence_basis,
            "contradictions": list(self.contradictions),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DecisionOption":
        raw = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if "confidence" in raw and isinstance(raw["confidence"], dict):
            raw["confidence"] = ConfidenceScore.from_dict(raw["confidence"])
        if "limitations" in raw and isinstance(raw["limitations"], list):
            raw["limitations"] = [
                DecisionLimitation.from_dict(lim) if isinstance(lim, dict) else lim
                for lim in raw["limitations"]
            ]
        return cls(**raw)


# ---------------------------------------------------------------------------
# DecisionReport — aggregate output
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DecisionReport:
    """Aggregate output of the Decision Generator for a dataset.

    Contains all decision options, processing notes, and metadata.
    """
    dataset_id: str = ""
    decisions: list = field(default_factory=list)          # list[DecisionOption]
    processing_notes: list = field(default_factory=list)   # list[str]

    def by_category(self, category: str) -> list:
        """Filter decisions by category."""
        return [d for d in self.decisions if d.category == category]

    def by_priority(self, priority: str) -> list:
        """Filter decisions by priority level."""
        return [d for d in self.decisions if d.priority == priority]

    def recommended(self) -> list:
        """Return RECOMMEND category decisions."""
        return self.by_category(DecisionCategory.RECOMMEND.value)

    def investigations(self) -> list:
        """Return INVESTIGATE category decisions."""
        return self.by_category(DecisionCategory.INVESTIGATE.value)

    def monitoring(self) -> list:
        """Return MONITOR category decisions."""
        return self.by_category(DecisionCategory.MONITOR.value)

    def risk_mitigations(self) -> list:
        """Return MITIGATE_RISK category decisions."""
        return self.by_category(DecisionCategory.MITIGATE_RISK.value)

    def blocked(self) -> list:
        """Return BLOCKED category decisions."""
        return self.by_category(DecisionCategory.BLOCKED.value)

    def ranked(self) -> list:
        """Return decisions ranked by the multi-dimensional priority order.

        Ranking dimensions (kept distinct):
        1. Priority level (critical > high > medium > low)
        2. Category weight (recommend > mitigate_risk > investigate > monitor > blocked)
        3. Confidence level (high > medium > low > insufficient)
        4. Evidence count (more evidence ranks higher)
        5. Stable tie-breaking by decision ID
        """
        priority_order = {
            PriorityLevel.CRITICAL.value: 0,
            PriorityLevel.HIGH.value: 1,
            PriorityLevel.MEDIUM.value: 2,
            PriorityLevel.LOW.value: 3,
        }
        category_order = {
            DecisionCategory.RECOMMEND.value: 0,
            DecisionCategory.MITIGATE_RISK.value: 1,
            DecisionCategory.INVESTIGATE.value: 2,
            DecisionCategory.MONITOR.value: 3,
            DecisionCategory.BLOCKED.value: 4,
        }
        confidence_order = {
            ConfidenceLevel.HIGH.value: 0,
            ConfidenceLevel.MEDIUM.value: 1,
            ConfidenceLevel.LOW.value: 2,
            ConfidenceLevel.INSUFFICIENT.value: 3,
        }

        def sort_key(d: DecisionOption):
            conf_level = ""
            ev_count = 0
            if d.confidence:
                conf_level = d.confidence.level
                ev_count = d.confidence.evidence_count
            return (
                priority_order.get(d.priority, 99),
                category_order.get(d.category, 99),
                confidence_order.get(conf_level, 99),
                -ev_count,  # Higher evidence count sorts first
                d.id,       # Deterministic tie-break
            )

        return sorted(self.decisions, key=sort_key)

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "decisions": [d.to_dict() for d in self.decisions],
            "processing_notes": list(self.processing_notes),
            "decision_count": len(self.decisions),
            "by_category": {
                cat.value: len(self.by_category(cat.value))
                for cat in DecisionCategory
            },
            "by_priority": {
                pri.value: len(self.by_priority(pri.value))
                for pri in PriorityLevel
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DecisionReport":
        raw = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if "decisions" in raw and isinstance(raw["decisions"], list):
            raw["decisions"] = [
                DecisionOption.from_dict(opt) if isinstance(opt, dict) else opt
                for opt in raw["decisions"]
            ]
        return cls(**raw)


# ---------------------------------------------------------------------------
# DecisionGenerator — the engine
# ---------------------------------------------------------------------------

class DecisionGenerator:
    """Deterministic, domain-agnostic decision generation engine.

    Converts supported evidence, decision signals, impact assessments,
    and scenario results into structured decision options and actionable
    recommendations.

    Usage:
        generator = DecisionGenerator(dataset_id="ds_001")
        report = generator.generate(
            synthesis_result=synthesis_result,  # from Part 2
            impact_report=impact_report,        # from Part 3
            scenario_results=scenario_results,  # from Part 4
        )
    """

    def __init__(self, dataset_id: str = ""):
        if not dataset_id:
            raise ValueError(
                "dataset_id is required. Decisions must be scoped to a dataset."
            )
        self._dataset_id = dataset_id
        self._decisions: list[DecisionOption] = []
        self._processing_notes: list[str] = []

    @property
    def dataset_id(self) -> str:
        return self._dataset_id

    @property
    def processing_notes(self) -> list[str]:
        return list(self._processing_notes)

    # -------------------------------------------------------------------
    # Main entry point
    # -------------------------------------------------------------------

    def generate(
        self,
        synthesis_result=None,
        impact_report: Optional[ImpactReport] = None,
        scenario_results: Optional[list] = None,
    ) -> DecisionReport:
        """Generate decisions from available inputs.

        All inputs are optional — the engine generates decisions from
        whatever is provided and notes what is missing.

        Args:
            synthesis_result: SynthesisResult from Part 2 signal synthesizer
            impact_report: ImpactReport from Part 3 impact quantifier
            scenario_results: list of ScenarioResult from Part 4 scenario engine

        Returns:
            DecisionReport with all generated decisions.
        """
        has_any_input = False

        # Process synthesis result (Part 2)
        if synthesis_result is not None:
            has_any_input = True
            self._process_synthesis(synthesis_result)

        # Process impact report (Part 3)
        if impact_report is not None:
            has_any_input = True
            self._process_impacts(impact_report)

        # Process scenario results (Part 4)
        if scenario_results:
            has_any_input = True
            self._process_scenarios(scenario_results)

        if not has_any_input:
            self._processing_notes.append(
                "No inputs provided for decision generation."
            )

        if not self._decisions:
            self._processing_notes.append(
                "No decisions could be generated from the available inputs."
            )

        return self._build_report()

    # -------------------------------------------------------------------
    # Process synthesis result (Part 2 signals)
    # -------------------------------------------------------------------

    def _process_synthesis(self, synthesis_result) -> None:
        """Generate decisions from synthesized signals."""
        ds_id = _get_attr(synthesis_result, "dataset_id", "")
        if ds_id and ds_id != self._dataset_id:
            self._processing_notes.append(
                f"Synthesis result dataset_id '{ds_id}' does not match "
                f"generator dataset_id '{self._dataset_id}', skipping."
            )
            return

        signals = _get_attr(synthesis_result, "signals", [])
        evidence_registry = _get_attr(synthesis_result, "evidence_registry", {})
        contradictions = _get_attr(synthesis_result, "contradictions", [])

        if not signals:
            self._processing_notes.append(
                "Synthesis result contains no signals."
            )
            return

        for synth_signal in signals:
            try:
                decision = self._decision_from_signal(
                    synth_signal, evidence_registry, contradictions,
                )
                if decision is not None:
                    self._decisions.append(decision)
            except Exception as exc:
                source_type = _get_attr(synth_signal, "source_type", "unknown")
                self._processing_notes.append(
                    f"Failed to generate decision from signal "
                    f"'{source_type}': {exc}"
                )

    def _decision_from_signal(
        self,
        synth_signal,
        evidence_registry: dict,
        contradictions: list,
    ) -> Optional[DecisionOption]:
        """Generate a decision option from a single synthesized signal."""
        signal = _get_attr(synth_signal, "signal", None)
        if signal is None:
            return None

        # Validate dataset scope
        sig_ds = _get_attr(signal, "dataset_id", "")
        if sig_ds and sig_ds != self._dataset_id:
            return None

        # Gather evidence
        evidence_items = _get_attr(synth_signal, "evidence_items", [])
        if not evidence_items:
            ev_ids = _get_attr(signal, "evidence_ids", [])
            for eid in ev_ids:
                ev = evidence_registry.get(eid)
                if ev is not None:
                    evidence_items.append(ev)

        # Scope evidence to dataset
        scoped_evidence = (
            validate_dataset_scope(evidence_items, self._dataset_id)
            if evidence_items else []
        )

        # Get signal metadata
        priority_obj = _get_attr(synth_signal, "priority", None)
        source_type = _get_attr(synth_signal, "source_type", "")
        generation_reason = _get_attr(synth_signal, "generation_reason", "")
        confidence_score = _get_attr(synth_signal, "confidence_score", None)

        signal_desc = _get_attr(signal, "description", "")
        signal_id = _get_attr(signal, "id", "")

        # Assess confidence from evidence if not already available
        if confidence_score is None and scoped_evidence:
            confidence_score = assess_confidence(
                scoped_evidence, dataset_id=self._dataset_id,
            )

        # Determine evidence basis
        evidence_basis = self._determine_evidence_basis(scoped_evidence)

        # Determine category, action, and priority
        category = self._determine_category(
            scoped_evidence, confidence_score, source_type,
        )
        priority, priority_rationale = self._determine_priority(
            scoped_evidence, confidence_score, source_type,
            priority_obj, category,
        )

        # Build limitations
        limitations = list(_get_attr(signal, "limitations", []))
        if evidence_basis in ("model_estimated", "scenario_based"):
            limitations.append(create_limitation(
                "This recommendation is based on model estimates, "
                "not directly measured outcomes.",
                category=LimitationCategory.MODEL_UNCERTAINTY.value,
            ))

        # Surface contradictions
        contradiction_notes = self._find_relevant_contradictions(
            signal_id, contradictions,
        )

        # Build recommended action
        recommended_action = self._build_action(
            category, signal_desc, evidence_basis, source_type,
        )

        # Build explanation
        explanation = self._build_explanation(
            category, signal_desc, evidence_basis,
            len(scoped_evidence), source_type,
        )

        # Build invalidation conditions
        invalidation_conditions = self._build_invalidation_conditions(
            evidence_basis, source_type,
        )

        # Determine urgency from signal priority
        urgency = self._determine_urgency(priority_obj)

        # Build title
        title = self._build_title(signal_desc, category)

        # Evidence IDs
        evidence_ids = [
            _get_attr(ev, "id", "") for ev in scoped_evidence
        ]

        return DecisionOption(
            dataset_id=self._dataset_id,
            title=title,
            description=signal_desc,
            category=category,
            recommended_action=recommended_action,
            signal_ids=[signal_id] if signal_id else [],
            evidence_ids=evidence_ids,
            confidence=confidence_score,
            priority=priority,
            priority_rationale=priority_rationale,
            urgency=urgency,
            assumptions=self._build_assumptions(evidence_basis),
            limitations=limitations,
            risks=self._build_risks(category, evidence_basis),
            reversibility=Reversibility.UNKNOWN.value,
            required_follow_up=self._build_follow_up(category, evidence_basis),
            invalidation_conditions=invalidation_conditions,
            explanation=explanation,
            evidence_basis=evidence_basis,
            contradictions=contradiction_notes,
        )

    # -------------------------------------------------------------------
    # Process impact report (Part 3)
    # -------------------------------------------------------------------

    def _process_impacts(self, impact_report: ImpactReport) -> None:
        """Generate decisions from impact assessments."""
        ds_id = _get_attr(impact_report, "dataset_id", "")
        if ds_id and ds_id != self._dataset_id:
            self._processing_notes.append(
                f"Impact report dataset_id '{ds_id}' does not match "
                f"generator dataset_id '{self._dataset_id}', skipping."
            )
            return

        assessments = _get_attr(impact_report, "assessments", [])
        if not assessments:
            self._processing_notes.append(
                "Impact report contains no assessments."
            )
            return

        for assessment in assessments:
            try:
                decision = self._decision_from_impact(assessment)
                if decision is not None:
                    # Avoid duplicate decisions for the same signal
                    if not self._has_signal_decision(
                        _get_attr(assessment, "signal_id", ""),
                    ):
                        self._decisions.append(decision)
                    else:
                        # Merge impact information into existing decision
                        self._merge_impact_into_existing(assessment)
            except Exception as exc:
                self._processing_notes.append(
                    f"Failed to generate decision from impact assessment: {exc}"
                )

    def _decision_from_impact(
        self,
        assessment: ImpactAssessment,
    ) -> Optional[DecisionOption]:
        """Generate a decision from an impact assessment."""
        # Validate dataset scope
        ds_id = _get_attr(assessment, "dataset_id", "")
        if ds_id and ds_id != self._dataset_id:
            return None

        impact_type = _get_attr(assessment, "impact_type", "")
        impact_category = _get_attr(assessment, "impact_category", "")
        description = _get_attr(assessment, "description", "")
        explanation = _get_attr(assessment, "explanation", "")
        estimated_value = _safe_float(_get_attr(assessment, "estimated_value", None))
        unit = _get_attr(assessment, "unit", "")
        confidence = _get_attr(assessment, "confidence", None)
        limitations = list(_get_attr(assessment, "limitations", []))
        assumptions = list(_get_attr(assessment, "assumptions", []))
        evidence_ids = list(_get_attr(assessment, "evidence_ids", []))
        assessment_id = _get_attr(assessment, "id", "")
        signal_id = _get_attr(assessment, "signal_id", "")

        # Determine evidence basis from impact type
        evidence_basis = self._impact_type_to_evidence_basis(impact_type)

        # Determine category
        if impact_type == ImpactType.UNKNOWN.value:
            category = DecisionCategory.INVESTIGATE.value
        elif confidence and confidence.level == ConfidenceLevel.INSUFFICIENT.value:
            category = DecisionCategory.INVESTIGATE.value
        elif impact_category == ImpactCategory.ANOMALY.value:
            category = DecisionCategory.INVESTIGATE.value
        elif impact_category == ImpactCategory.RISK.value:
            category = DecisionCategory.MITIGATE_RISK.value
        elif impact_type == ImpactType.MEASURED.value:
            category = DecisionCategory.RECOMMEND.value
        elif impact_type in (ImpactType.MODEL_ESTIMATED.value,
                             ImpactType.SCENARIO_BASED.value):
            category = DecisionCategory.MONITOR.value
        else:
            category = DecisionCategory.MONITOR.value

        # Determine priority
        priority, priority_rationale = self._determine_priority_from_impact(
            impact_type, confidence, estimated_value, category,
        )

        # Build impact summary
        estimated_impact = None
        if estimated_value is not None:
            estimated_impact = f"{description}"
            if impact_type != ImpactType.MEASURED.value:
                estimated_impact += " (model estimate, not guaranteed)"

        # Build action
        if category == DecisionCategory.RECOMMEND.value:
            action = f"Act on measured finding: {description}"
        elif category == DecisionCategory.INVESTIGATE.value:
            action = f"Investigate further: {description}"
        elif category == DecisionCategory.MONITOR.value:
            action = f"Monitor and track: {description}"
        elif category == DecisionCategory.MITIGATE_RISK.value:
            action = f"Assess and mitigate: {description}"
        else:
            action = f"Review: {description}"

        # Title
        title = self._build_title(description, category)

        return DecisionOption(
            dataset_id=self._dataset_id,
            title=title,
            description=description,
            category=category,
            recommended_action=action,
            signal_ids=[signal_id] if signal_id else [],
            evidence_ids=evidence_ids,
            impact_ids=[assessment_id] if assessment_id else [],
            estimated_impact=estimated_impact,
            impact_type=impact_type,
            impact_value=estimated_value,
            impact_unit=unit,
            confidence=confidence,
            priority=priority,
            priority_rationale=priority_rationale,
            urgency=UrgencyLevel.NOT_ASSESSED.value,
            assumptions=assumptions,
            limitations=limitations,
            risks=self._build_risks(category, evidence_basis),
            reversibility=Reversibility.UNKNOWN.value,
            required_follow_up=self._build_follow_up(category, evidence_basis),
            invalidation_conditions=self._build_invalidation_conditions(
                evidence_basis, "",
            ),
            explanation=explanation,
            evidence_basis=evidence_basis,
        )

    def _has_signal_decision(self, signal_id: str) -> bool:
        """Check if a decision already exists for this signal."""
        if not signal_id:
            return False
        return any(
            signal_id in d.signal_ids
            for d in self._decisions
        )

    def _merge_impact_into_existing(self, assessment: ImpactAssessment) -> None:
        """Merge impact information into an existing decision for the same signal."""
        signal_id = _get_attr(assessment, "signal_id", "")
        assessment_id = _get_attr(assessment, "id", "")
        if not signal_id:
            return

        for i, d in enumerate(self._decisions):
            if signal_id in d.signal_ids:
                # Build updated fields (frozen dataclass — must reconstruct)
                new_impact_ids = list(d.impact_ids)
                if assessment_id and assessment_id not in new_impact_ids:
                    new_impact_ids.append(assessment_id)

                estimated_value = _safe_float(
                    _get_attr(assessment, "estimated_value", None)
                )
                impact_type = _get_attr(assessment, "impact_type", "")
                unit = _get_attr(assessment, "unit", "")
                description = _get_attr(assessment, "description", "")

                estimated_impact = d.estimated_impact
                if estimated_value is not None and not estimated_impact:
                    estimated_impact = description
                    if impact_type != ImpactType.MEASURED.value:
                        estimated_impact += " (model estimate, not guaranteed)"

                self._decisions[i] = DecisionOption(
                    id=d.id,
                    dataset_id=d.dataset_id,
                    title=d.title,
                    description=d.description,
                    category=d.category,
                    recommended_action=d.recommended_action,
                    signal_ids=d.signal_ids,
                    evidence_ids=d.evidence_ids,
                    impact_ids=new_impact_ids,
                    scenario_ids=d.scenario_ids,
                    estimated_impact=estimated_impact,
                    impact_type=impact_type or d.impact_type,
                    impact_value=(
                        estimated_value
                        if estimated_value is not None
                        else d.impact_value
                    ),
                    impact_unit=unit or d.impact_unit,
                    confidence=d.confidence,
                    priority=d.priority,
                    priority_rationale=d.priority_rationale,
                    urgency=d.urgency,
                    assumptions=d.assumptions,
                    limitations=d.limitations,
                    risks=d.risks,
                    reversibility=d.reversibility,
                    required_follow_up=d.required_follow_up,
                    invalidation_conditions=d.invalidation_conditions,
                    explanation=d.explanation,
                    evidence_basis=d.evidence_basis,
                    contradictions=d.contradictions,
                )
                break

    # -------------------------------------------------------------------
    # Process scenario results (Part 4)
    # -------------------------------------------------------------------

    def _process_scenarios(self, scenario_results: list) -> None:
        """Generate decisions from scenario results."""
        for result in scenario_results:
            try:
                decision = self._decision_from_scenario(result)
                if decision is not None:
                    self._decisions.append(decision)
            except Exception as exc:
                scenario_id = _get_attr(result, "scenario_id", "unknown")
                self._processing_notes.append(
                    f"Failed to generate decision from scenario "
                    f"'{scenario_id}': {exc}"
                )

    def _decision_from_scenario(
        self,
        result: ScenarioResult,
    ) -> Optional[DecisionOption]:
        """Generate a decision from a scenario result."""
        ds_id = _get_attr(result, "dataset_id", "")
        if ds_id and ds_id != self._dataset_id:
            self._processing_notes.append(
                f"Scenario result dataset_id '{ds_id}' does not match "
                f"generator dataset_id '{self._dataset_id}', skipping."
            )
            return None

        status = _get_attr(result, "status", "")
        scenario_id = _get_attr(result, "scenario_id", "")
        description = _get_attr(result, "description", "")
        explanation = _get_attr(result, "explanation", "")
        delta = _safe_float(_get_attr(result, "delta", None))
        delta_pct = _safe_float(_get_attr(result, "delta_pct", None))
        unit = _get_attr(result, "unit", "")
        confidence = _get_attr(result, "confidence", None)
        limitations = list(_get_attr(result, "limitations", []))
        assumptions = list(_get_attr(result, "assumptions", []))
        evidence_ids = list(_get_attr(result, "evidence_ids", []))
        result_id = _get_attr(result, "id", "")

        # Determine category based on scenario status
        if status == ScenarioStatus.INSUFFICIENT_EVIDENCE.value:
            category = DecisionCategory.BLOCKED.value
        elif status == ScenarioStatus.NOT_SIMULATABLE.value:
            category = DecisionCategory.INVESTIGATE.value
        elif status == ScenarioStatus.PARTIALLY_SIMULATED.value:
            category = DecisionCategory.MONITOR.value
        else:
            # SIMULATED — but still scenario-based, not guaranteed
            category = DecisionCategory.MONITOR.value

        # Scenarios are always scenario-based evidence
        evidence_basis = "scenario_based"

        # Build impact summary
        estimated_impact = None
        impact_value = None
        if delta is not None:
            impact_value = delta
            estimated_impact = (
                f"Projected change: {delta:.4f}"
                f" ({delta_pct:.2f}%)" if delta_pct is not None else ""
            )
            if estimated_impact:
                estimated_impact += " (scenario projection, not guaranteed)"

        # Priority
        priority, priority_rationale = self._determine_priority_from_scenario(
            status, confidence, delta, category,
        )

        # Build action
        if category == DecisionCategory.BLOCKED.value:
            action = (
                f"Cannot evaluate scenario: insufficient evidence. "
                f"{description}"
            )
        elif category == DecisionCategory.INVESTIGATE.value:
            action = (
                f"Investigate further to enable scenario evaluation. "
                f"{description}"
            )
        else:
            action = (
                f"Monitor scenario outcomes and validate projections. "
                f"{description}"
            )

        # Add scenario-specific limitations
        limitations.append(create_limitation(
            "Scenario projections are hypothetical model outputs. "
            "They are not guaranteed outcomes and should not be treated as such.",
            category=LimitationCategory.MODEL_UNCERTAINTY.value,
        ))

        title = self._build_title(description, category)

        return DecisionOption(
            dataset_id=self._dataset_id,
            title=title,
            description=description,
            category=category,
            recommended_action=action,
            evidence_ids=evidence_ids,
            scenario_ids=[scenario_id] if scenario_id else [],
            estimated_impact=estimated_impact,
            impact_type=ImpactType.SCENARIO_BASED.value,
            impact_value=impact_value,
            impact_unit=unit,
            confidence=confidence,
            priority=priority,
            priority_rationale=priority_rationale,
            urgency=UrgencyLevel.NOT_ASSESSED.value,
            assumptions=assumptions,
            limitations=limitations,
            risks=[
                "Scenario projections may not materialize in practice.",
                "External factors not modeled could change outcomes.",
            ],
            reversibility=Reversibility.UNKNOWN.value,
            required_follow_up=[
                "Validate scenario assumptions against current conditions.",
                "Monitor actual outcomes against projected values.",
            ],
            invalidation_conditions=[
                "Underlying model assumptions no longer hold.",
                "New data contradicts the scenario's evidence base.",
            ],
            explanation=explanation,
            evidence_basis=evidence_basis,
        )

    # -------------------------------------------------------------------
    # Category determination
    # -------------------------------------------------------------------

    def _determine_category(
        self,
        evidence_items: list,
        confidence: Optional[ConfidenceScore],
        source_type: str,
    ) -> str:
        """Determine decision category from evidence and confidence."""
        # Insufficient evidence → blocked or investigate
        if not evidence_items:
            return DecisionCategory.BLOCKED.value

        if confidence:
            if confidence.level == ConfidenceLevel.INSUFFICIENT.value:
                return DecisionCategory.BLOCKED.value

        # Anomaly sources → investigate
        if "anomaly" in source_type:
            return DecisionCategory.INVESTIGATE.value

        # Forecasts and scenarios → monitor
        if "forecast" in source_type or "scenario" in source_type:
            return DecisionCategory.MONITOR.value

        # Risk-related → mitigate
        if "risk" in source_type:
            return DecisionCategory.MITIGATE_RISK.value

        # Strong evidence from measured sources → recommend
        if confidence and confidence.level in (
            ConfidenceLevel.HIGH.value,
            ConfidenceLevel.MEDIUM.value,
        ):
            # Check evidence level — only recommend based on facts/observations
            has_measured = any(
                _get_attr(ev, "level", "") == EvidenceLevel.FACT.value
                for ev in evidence_items
            )
            if has_measured:
                return DecisionCategory.RECOMMEND.value

        # Model-based sources → monitor
        if any(kw in source_type for kw in (
            "driver", "feature", "regression", "classification",
            "mmm", "contribution", "roi", "model",
        )):
            return DecisionCategory.MONITOR.value

        # Low confidence → investigate
        if confidence and confidence.level == ConfidenceLevel.LOW.value:
            return DecisionCategory.INVESTIGATE.value

        # Default: investigate (err on side of caution)
        return DecisionCategory.INVESTIGATE.value

    # -------------------------------------------------------------------
    # Priority determination
    # -------------------------------------------------------------------

    def _determine_priority(
        self,
        evidence_items: list,
        confidence: Optional[ConfidenceScore],
        source_type: str,
        priority_obj,
        category: str,
    ) -> tuple:
        """Determine priority level and rationale.

        Priority is a composite of evidence strength, potential impact
        indication, urgency indicators, and risk. It is NOT determined
        by confidence alone.

        Returns:
            (priority_level, rationale_string)
        """
        rationale_parts = []

        # Evidence strength component
        evidence_score = 0
        if evidence_items:
            evidence_score = min(len(evidence_items), 5)  # Cap at 5
            has_facts = any(
                _get_attr(ev, "level", "") == EvidenceLevel.FACT.value
                for ev in evidence_items
            )
            if has_facts:
                evidence_score += 2
            rationale_parts.append(
                f"Evidence strength: {len(evidence_items)} item(s)"
                + (" including measured facts" if has_facts else "")
            )
        else:
            rationale_parts.append("Evidence strength: no evidence available")

        # Confidence component (contributes but does not dominate)
        confidence_score = 0
        if confidence:
            conf_map = {
                ConfidenceLevel.HIGH.value: 3,
                ConfidenceLevel.MEDIUM.value: 2,
                ConfidenceLevel.LOW.value: 1,
                ConfidenceLevel.INSUFFICIENT.value: 0,
            }
            confidence_score = conf_map.get(confidence.level, 0)
            rationale_parts.append(f"Confidence: {confidence.level}")

        # Signal priority component (from Part 2)
        signal_priority_score = 0
        if priority_obj is not None:
            sig_pri_value = _get_attr(priority_obj, "value", None)
            if sig_pri_value is None:
                # Priority might be the enum itself
                try:
                    sig_pri_value = int(priority_obj)
                except (TypeError, ValueError):
                    sig_pri_value = None

            if sig_pri_value is not None:
                # SignalPriority: CRITICAL=1, HIGH=2, MEDIUM=3, LOW=4, INFO=5
                # Invert so higher signal priority → higher score
                signal_priority_score = max(0, 6 - sig_pri_value)
                rationale_parts.append(
                    f"Signal priority: level {sig_pri_value}"
                )

        # Category adjustment (blocked/investigate get lower priority)
        category_score = 0
        if category == DecisionCategory.RECOMMEND.value:
            category_score = 2
        elif category == DecisionCategory.MITIGATE_RISK.value:
            category_score = 3  # Risk mitigation gets a boost
        elif category == DecisionCategory.MONITOR.value:
            category_score = 1
        elif category == DecisionCategory.INVESTIGATE.value:
            category_score = 0
        elif category == DecisionCategory.BLOCKED.value:
            category_score = 0

        # Composite score
        total_score = (
            evidence_score
            + confidence_score
            + signal_priority_score
            + category_score
        )

        # Map to priority level
        if total_score >= 12:
            priority = PriorityLevel.CRITICAL.value
        elif total_score >= 8:
            priority = PriorityLevel.HIGH.value
        elif total_score >= 4:
            priority = PriorityLevel.MEDIUM.value
        else:
            priority = PriorityLevel.LOW.value

        rationale = "; ".join(rationale_parts)
        return priority, rationale

    def _determine_priority_from_impact(
        self,
        impact_type: str,
        confidence: Optional[ConfidenceScore],
        estimated_value: Optional[float],
        category: str,
    ) -> tuple:
        """Determine priority from impact assessment attributes."""
        rationale_parts = []

        score = 0

        # Impact type contribution
        if impact_type == ImpactType.MEASURED.value:
            score += 4
            rationale_parts.append("Impact: directly measured")
        elif impact_type == ImpactType.MODEL_ESTIMATED.value:
            score += 2
            rationale_parts.append("Impact: model-estimated")
        elif impact_type == ImpactType.SCENARIO_BASED.value:
            score += 1
            rationale_parts.append("Impact: scenario-based projection")
        elif impact_type == ImpactType.QUALITATIVE.value:
            score += 1
            rationale_parts.append("Impact: qualitative assessment")
        else:
            rationale_parts.append("Impact: type unknown")

        # Confidence contribution
        if confidence:
            conf_map = {
                ConfidenceLevel.HIGH.value: 3,
                ConfidenceLevel.MEDIUM.value: 2,
                ConfidenceLevel.LOW.value: 1,
                ConfidenceLevel.INSUFFICIENT.value: 0,
            }
            score += conf_map.get(confidence.level, 0)
            rationale_parts.append(f"Confidence: {confidence.level}")

        # Category contribution
        if category == DecisionCategory.MITIGATE_RISK.value:
            score += 2
            rationale_parts.append("Category: risk mitigation")

        if score >= 8:
            priority = PriorityLevel.CRITICAL.value
        elif score >= 5:
            priority = PriorityLevel.HIGH.value
        elif score >= 3:
            priority = PriorityLevel.MEDIUM.value
        else:
            priority = PriorityLevel.LOW.value

        rationale = "; ".join(rationale_parts)
        return priority, rationale

    def _determine_priority_from_scenario(
        self,
        status: str,
        confidence: Optional[ConfidenceScore],
        delta: Optional[float],
        category: str,
    ) -> tuple:
        """Determine priority from scenario result attributes."""
        rationale_parts = []

        score = 0

        if status == ScenarioStatus.SIMULATED.value:
            score += 3
            rationale_parts.append("Status: fully simulated")
        elif status == ScenarioStatus.PARTIALLY_SIMULATED.value:
            score += 2
            rationale_parts.append("Status: partially simulated")
        elif status == ScenarioStatus.NOT_SIMULATABLE.value:
            score += 1
            rationale_parts.append("Status: not simulatable")
        else:
            rationale_parts.append("Status: insufficient evidence")

        if confidence:
            conf_map = {
                ConfidenceLevel.HIGH.value: 3,
                ConfidenceLevel.MEDIUM.value: 2,
                ConfidenceLevel.LOW.value: 1,
                ConfidenceLevel.INSUFFICIENT.value: 0,
            }
            score += conf_map.get(confidence.level, 0)
            rationale_parts.append(f"Confidence: {confidence.level}")

        if score >= 5:
            priority = PriorityLevel.HIGH.value
        elif score >= 3:
            priority = PriorityLevel.MEDIUM.value
        else:
            priority = PriorityLevel.LOW.value

        rationale = "; ".join(rationale_parts)
        return priority, rationale

    # -------------------------------------------------------------------
    # Evidence basis determination
    # -------------------------------------------------------------------

    def _determine_evidence_basis(self, evidence_items: list) -> str:
        """Determine the evidence basis type from evidence items."""
        if not evidence_items:
            return "qualitative"

        levels = set()
        for ev in evidence_items:
            level = _get_attr(ev, "level", "")
            if level:
                levels.add(level)

        if EvidenceLevel.FACT.value in levels:
            return "measured"
        if EvidenceLevel.PREDICTION.value in levels:
            return "scenario_based"
        if EvidenceLevel.INFERENCE.value in levels:
            return "model_estimated"
        return "qualitative"

    def _impact_type_to_evidence_basis(self, impact_type: str) -> str:
        """Convert ImpactType to evidence basis string."""
        mapping = {
            ImpactType.MEASURED.value: "measured",
            ImpactType.MODEL_ESTIMATED.value: "model_estimated",
            ImpactType.SCENARIO_BASED.value: "scenario_based",
            ImpactType.QUALITATIVE.value: "qualitative",
            ImpactType.UNKNOWN.value: "qualitative",
        }
        return mapping.get(impact_type, "qualitative")

    # -------------------------------------------------------------------
    # Urgency determination
    # -------------------------------------------------------------------

    def _determine_urgency(self, priority_obj) -> str:
        """Determine urgency from signal priority."""
        if priority_obj is None:
            return UrgencyLevel.NOT_ASSESSED.value

        sig_pri_value = _get_attr(priority_obj, "value", None)
        if sig_pri_value is None:
            try:
                sig_pri_value = int(priority_obj)
            except (TypeError, ValueError):
                return UrgencyLevel.NOT_ASSESSED.value

        # SignalPriority: CRITICAL=1, HIGH=2, MEDIUM=3, LOW=4, INFO=5
        if sig_pri_value <= 1:
            return UrgencyLevel.IMMEDIATE.value
        elif sig_pri_value <= 2:
            return UrgencyLevel.SHORT_TERM.value
        elif sig_pri_value <= 3:
            return UrgencyLevel.MEDIUM_TERM.value
        else:
            return UrgencyLevel.LONG_TERM.value

    # -------------------------------------------------------------------
    # Contradiction surfacing
    # -------------------------------------------------------------------

    def _find_relevant_contradictions(
        self,
        signal_id: str,
        contradictions: list,
    ) -> list:
        """Find contradictions relevant to a signal."""
        if not contradictions or not signal_id:
            return []

        relevant = []
        for contradiction in contradictions:
            if isinstance(contradiction, str):
                # Simple string contradictions — include if signal_id mentioned
                if signal_id in contradiction:
                    relevant.append(contradiction)
            elif isinstance(contradiction, dict):
                # Dict-based contradictions
                involved = contradiction.get("signal_ids", [])
                if signal_id in involved:
                    note = contradiction.get("description", str(contradiction))
                    relevant.append(note)

        return relevant

    # -------------------------------------------------------------------
    # Title, action, and explanation builders
    # -------------------------------------------------------------------

    def _build_title(self, description: str, category: str) -> str:
        """Build a concise title from description and category."""
        # Truncate description for title
        title_desc = description[:80].strip()
        if len(description) > 80:
            title_desc = title_desc[:77] + "..."

        prefix_map = {
            DecisionCategory.RECOMMEND.value: "Action",
            DecisionCategory.INVESTIGATE.value: "Investigate",
            DecisionCategory.MONITOR.value: "Monitor",
            DecisionCategory.MITIGATE_RISK.value: "Mitigate",
            DecisionCategory.BLOCKED.value: "Blocked",
        }
        prefix = prefix_map.get(category, "Review")
        return f"{prefix}: {title_desc}" if title_desc else prefix

    def _build_action(
        self,
        category: str,
        description: str,
        evidence_basis: str,
        source_type: str,
    ) -> str:
        """Build the recommended action text."""
        if category == DecisionCategory.BLOCKED.value:
            return (
                "Collect additional evidence before proceeding. "
                "Current evidence is insufficient to support a recommendation."
            )

        if category == DecisionCategory.INVESTIGATE.value:
            return (
                f"Investigate further: {description}. "
                "Gather additional data to confirm or refute the observation."
            )

        if category == DecisionCategory.MONITOR.value:
            qualifier = ""
            if evidence_basis == "model_estimated":
                qualifier = (
                    " The underlying model estimate should be validated "
                    "against observed outcomes."
                )
            elif evidence_basis == "scenario_based":
                qualifier = (
                    " The scenario projection should be tracked against "
                    "actual results."
                )
            return (
                f"Monitor and track: {description}.{qualifier}"
            )

        if category == DecisionCategory.MITIGATE_RISK.value:
            return (
                f"Assess and mitigate identified risk: {description}. "
                "Develop a mitigation plan based on available evidence."
            )

        if category == DecisionCategory.RECOMMEND.value:
            qualifier = ""
            if evidence_basis != "measured":
                qualifier = (
                    " Note: this recommendation is based on "
                    f"{evidence_basis.replace('_', ' ')} evidence."
                )
            return (
                f"Take action based on evidence: {description}.{qualifier}"
            )

        return f"Review: {description}"

    def _build_explanation(
        self,
        category: str,
        description: str,
        evidence_basis: str,
        evidence_count: int,
        source_type: str,
    ) -> str:
        """Build the human-readable explanation."""
        parts = []

        parts.append(
            f"This {category.replace('_', ' ')} decision is based on "
            f"{evidence_count} piece(s) of {evidence_basis.replace('_', ' ')} "
            f"evidence."
        )

        if evidence_basis == "measured":
            parts.append(
                "The evidence is directly observed from the dataset."
            )
        elif evidence_basis == "model_estimated":
            parts.append(
                "The evidence is based on model estimates. "
                "Model estimates reflect statistical associations, "
                "not necessarily causal relationships."
            )
        elif evidence_basis == "scenario_based":
            parts.append(
                "The evidence is based on scenario projections. "
                "Projected outcomes are hypothetical and may not materialize."
            )
        elif evidence_basis == "qualitative":
            parts.append(
                "The evidence provides qualitative insights "
                "without numeric quantification."
            )

        if category == DecisionCategory.BLOCKED.value:
            parts.append(
                "The evidence is insufficient to support any recommendation. "
                "Additional data collection is required."
            )

        return " ".join(parts)

    def _build_assumptions(self, evidence_basis: str) -> list:
        """Build assumptions list based on evidence basis."""
        assumptions = []
        if evidence_basis == "measured":
            assumptions.append(
                "Observed patterns in the data represent ongoing conditions."
            )
        elif evidence_basis == "model_estimated":
            assumptions.extend([
                "Model relationships hold for the current conditions.",
                "Statistical association does not establish causation.",
            ])
        elif evidence_basis == "scenario_based":
            assumptions.extend([
                "Scenario assumptions remain valid.",
                "No external factors intervene during the projection period.",
            ])
        return assumptions

    def _build_risks(self, category: str, evidence_basis: str) -> list:
        """Build risk list based on category and evidence basis."""
        risks = []
        if evidence_basis in ("model_estimated", "scenario_based"):
            risks.append(
                "Outcome may differ from model projections."
            )
        if evidence_basis == "model_estimated":
            risks.append(
                "Model associations may not reflect causal relationships."
            )
        if category == DecisionCategory.RECOMMEND.value:
            risks.append(
                "Acting on this recommendation involves inherent uncertainty."
            )
        return risks

    def _build_follow_up(self, category: str, evidence_basis: str) -> list:
        """Build follow-up actions based on category."""
        follow_up = []
        if category == DecisionCategory.INVESTIGATE.value:
            follow_up.append("Collect additional evidence.")
            follow_up.append("Re-evaluate after new data is available.")
        elif category == DecisionCategory.MONITOR.value:
            follow_up.append("Track relevant metrics over time.")
            follow_up.append("Compare actual outcomes to projections.")
        elif category == DecisionCategory.MITIGATE_RISK.value:
            follow_up.append("Develop and document a mitigation plan.")
            follow_up.append("Reassess risk after mitigation actions.")
        elif category == DecisionCategory.BLOCKED.value:
            follow_up.append("Identify and collect missing evidence.")
            follow_up.append("Re-run decision generation when evidence is available.")
        elif category == DecisionCategory.RECOMMEND.value:
            follow_up.append("Validate outcomes after action is taken.")
            if evidence_basis != "measured":
                follow_up.append(
                    "Monitor whether actual results align with "
                    f"{evidence_basis.replace('_', ' ')} projections."
                )
        return follow_up

    def _build_invalidation_conditions(
        self,
        evidence_basis: str,
        source_type: str,
    ) -> list:
        """Build conditions that would invalidate the recommendation."""
        conditions = [
            "New evidence contradicts the current findings.",
        ]
        if evidence_basis == "model_estimated":
            conditions.append(
                "The underlying model is retrained with substantially different results."
            )
        elif evidence_basis == "scenario_based":
            conditions.append(
                "Scenario assumptions are proven invalid by new observations."
            )
        conditions.append(
            "The dataset this decision is based on is found to contain "
            "errors or is no longer representative."
        )
        return conditions

    # -------------------------------------------------------------------
    # Build report
    # -------------------------------------------------------------------

    def _build_report(self) -> DecisionReport:
        return DecisionReport(
            dataset_id=self._dataset_id,
            decisions=list(self._decisions),
            processing_notes=list(self._processing_notes),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_attr(obj, name: str, default=None):
    """Get attribute from an object or dict."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)
