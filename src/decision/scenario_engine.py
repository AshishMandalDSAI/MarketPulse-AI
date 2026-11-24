"""Phase 7 Part 4 — Domain-Agnostic Scenario Engine.

Deterministic engine for defining, validating, and comparing hypothetical
scenarios using available evidence and supported analytical capabilities.

Key design principles:
1. A scenario is not an observed fact.
2. A simulation is not a guaranteed future outcome.
3. Correlation does not establish causation.
4. Model coefficients are not automatically causal effects.
5. Unsupported assumptions must be disclosed.
6. No invented financial benefits, savings, ROI, or currency.
7. No incompatible unit combination.
8. No silent extrapolation outside a model's supported range.
9. Confidence is separate from impact magnitude.
10. Contradictory evidence and limitations preserved.
11. Dataset isolation enforced.
12. No global mutable caches. No demo-data fallbacks. No LLM calls.
"""

from __future__ import annotations

import enum
import math
import uuid
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
from src.decision.confidence import assess_confidence
from src.decision.evidence import (
    create_limitation,
    validate_dataset_scope,
)
from src.decision.impact_quantifier import (
    ImpactType,
    ImpactCategory,
)


# ---------------------------------------------------------------------------
# Scenario Status — outcome of evaluation
# ---------------------------------------------------------------------------

class ScenarioStatus(str, enum.Enum):
    """Result state of a scenario evaluation."""
    SIMULATED = "simulated"                        # Fully numerically evaluated
    PARTIALLY_SIMULATED = "partially_simulated"    # Some parameters evaluated, some not
    NOT_SIMULATABLE = "not_simulatable"             # Cannot numerically evaluate
    INSUFFICIENT_EVIDENCE = "insufficient_evidence" # Not enough evidence to evaluate


# ---------------------------------------------------------------------------
# ScenarioParameter — one parameter change in a scenario
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScenarioParameter:
    """A single parameter change within a scenario definition.

    Captures the baseline value (if known), the proposed alternative,
    the unit, and whether the change is supported by evidence.
    """
    id: str = field(default_factory=lambda: _make_id("sp_"))
    name: str = ""                          # Parameter name (generic, not hardcoded)
    baseline_value: Optional[float] = None  # Current / observed value
    alternative_value: Optional[float] = None  # Proposed change
    unit: str = ""                          # Unit of measurement (preserved from evidence)
    supported: bool = True                  # Whether evidence supports this change
    unsupported_reason: str = ""            # Why the change is unsupported
    change_pct: Optional[float] = None      # Percentage change (computed, not assumed)

    def __post_init__(self):
        if not self.name:
            raise ValueError("ScenarioParameter requires a non-empty name.")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "baseline_value": _safe_float(self.baseline_value),
            "alternative_value": _safe_float(self.alternative_value),
            "unit": self.unit,
            "supported": self.supported,
            "unsupported_reason": self.unsupported_reason,
            "change_pct": _safe_float(self.change_pct),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ScenarioParameter":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# ScenarioDefinition — a complete scenario specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScenarioDefinition:
    """A complete scenario specification with parameters and assumptions.

    A scenario describes a hypothetical situation: what would happen if
    certain parameters were changed from their baseline values?
    """
    id: str = field(default_factory=lambda: _make_id("scen_"))
    name: str = ""
    description: str = ""
    dataset_id: str = ""
    parameters: list = field(default_factory=list)     # list[ScenarioParameter]
    assumptions: list = field(default_factory=list)     # list[str]
    evidence_ids: list = field(default_factory=list)    # Supporting evidence references
    limitations: list = field(default_factory=list)     # list[DecisionLimitation]

    def __post_init__(self):
        if not self.name:
            raise ValueError("ScenarioDefinition requires a non-empty name.")
        if not self.description:
            raise ValueError("ScenarioDefinition requires a non-empty description.")
        if not self.dataset_id:
            raise ValueError(
                "ScenarioDefinition requires a non-empty dataset_id. "
                "Scenarios must be scoped to a dataset."
            )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "dataset_id": self.dataset_id,
            "parameters": [
                p.to_dict() if hasattr(p, "to_dict") else p
                for p in self.parameters
            ],
            "assumptions": list(self.assumptions),
            "evidence_ids": list(self.evidence_ids),
            "limitations": [
                lim.to_dict() if hasattr(lim, "to_dict") else lim
                for lim in self.limitations
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ScenarioDefinition":
        raw = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if "parameters" in raw and isinstance(raw["parameters"], list):
            raw["parameters"] = [
                ScenarioParameter.from_dict(p) if isinstance(p, dict) else p
                for p in raw["parameters"]
            ]
        if "limitations" in raw and isinstance(raw["limitations"], list):
            raw["limitations"] = [
                DecisionLimitation.from_dict(lim) if isinstance(lim, dict) else lim
                for lim in raw["limitations"]
            ]
        return cls(**raw)


# ---------------------------------------------------------------------------
# ScenarioResult — the outcome of evaluating a scenario
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScenarioResult:
    """The outcome of evaluating a scenario.

    Contains the evaluation status, any numeric result (only when fully
    supported), uncertainty bounds, confidence, limitations, and a clear
    explanation of what was and was not evaluated.
    """
    id: str = field(default_factory=lambda: _make_id("sr_"))
    scenario_id: str = ""
    dataset_id: str = ""

    # Evaluation status
    status: str = ScenarioStatus.INSUFFICIENT_EVIDENCE.value

    # Numeric result (only when status is SIMULATED or PARTIALLY_SIMULATED)
    estimated_outcome: Optional[float] = None
    baseline_outcome: Optional[float] = None
    delta: Optional[float] = None          # estimated - baseline
    delta_pct: Optional[float] = None      # (delta / baseline) * 100
    unit: str = ""

    # Uncertainty
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None

    # Evaluation method
    estimation_method: str = ""            # How the outcome was computed
    model_description: str = ""            # Which model / calculation was used

    # Human-readable
    description: str = ""
    explanation: str = ""

    # What's missing (for NOT_SIMULATABLE / INSUFFICIENT_EVIDENCE)
    missing_requirements: list = field(default_factory=list)  # list[str]
    unsupported_parameters: list = field(default_factory=list)  # list[str]

    # Evidence, confidence, limitations
    evidence_ids: list = field(default_factory=list)
    confidence: Optional[ConfidenceScore] = None
    limitations: list = field(default_factory=list)   # list[DecisionLimitation]
    assumptions: list = field(default_factory=list)    # list[str]

    def __post_init__(self):
        if self.status not in {s.value for s in ScenarioStatus}:
            raise ValueError(
                f"Invalid scenario status: {self.status!r}. "
                f"Must be one of: {[s.value for s in ScenarioStatus]}"
            )
        if not self.description:
            raise ValueError("ScenarioResult requires a non-empty description.")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "scenario_id": self.scenario_id,
            "dataset_id": self.dataset_id,
            "status": self.status,
            "estimated_outcome": _safe_float(self.estimated_outcome),
            "baseline_outcome": _safe_float(self.baseline_outcome),
            "delta": _safe_float(self.delta),
            "delta_pct": _safe_float(self.delta_pct),
            "unit": self.unit,
            "lower_bound": _safe_float(self.lower_bound),
            "upper_bound": _safe_float(self.upper_bound),
            "estimation_method": self.estimation_method,
            "model_description": self.model_description,
            "description": self.description,
            "explanation": self.explanation,
            "missing_requirements": list(self.missing_requirements),
            "unsupported_parameters": list(self.unsupported_parameters),
            "evidence_ids": list(self.evidence_ids),
            "confidence": self.confidence.to_dict() if self.confidence else None,
            "limitations": [
                lim.to_dict() if hasattr(lim, "to_dict") else lim
                for lim in self.limitations
            ],
            "assumptions": list(self.assumptions),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ScenarioResult":
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
# ScenarioComparison — comparing two or more scenario results
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScenarioComparison:
    """Comparison between two or more scenario results.

    Compares outcomes, highlights differences, and surfaces limitations.
    """
    id: str = field(default_factory=lambda: _make_id("sc_"))
    dataset_id: str = ""
    scenario_ids: list = field(default_factory=list)   # list[str]
    results: list = field(default_factory=list)         # list[ScenarioResult]
    comparison_notes: list = field(default_factory=list)  # list[str]
    limitations: list = field(default_factory=list)     # list[DecisionLimitation]

    def __post_init__(self):
        if len(self.results) < 2:
            raise ValueError(
                "ScenarioComparison requires at least two ScenarioResult objects."
            )

    @property
    def best_outcome(self) -> Optional["ScenarioResult"]:
        """Return the result with the highest estimated outcome, if any."""
        quantified = [
            r for r in self.results if r.estimated_outcome is not None
        ]
        if not quantified:
            return None
        return max(quantified, key=lambda r: r.estimated_outcome)

    @property
    def all_simulated(self) -> bool:
        """Whether all scenarios were fully simulated."""
        return all(
            r.status == ScenarioStatus.SIMULATED.value for r in self.results
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "dataset_id": self.dataset_id,
            "scenario_ids": list(self.scenario_ids),
            "results": [r.to_dict() for r in self.results],
            "comparison_notes": list(self.comparison_notes),
            "limitations": [
                lim.to_dict() if hasattr(lim, "to_dict") else lim
                for lim in self.limitations
            ],
            "all_simulated": self.all_simulated,
            "result_count": len(self.results),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ScenarioComparison":
        raw = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if "results" in raw and isinstance(raw["results"], list):
            raw["results"] = [
                ScenarioResult.from_dict(r) if isinstance(r, dict) else r
                for r in raw["results"]
            ]
        if "limitations" in raw and isinstance(raw["limitations"], list):
            raw["limitations"] = [
                DecisionLimitation.from_dict(lim) if isinstance(lim, dict) else lim
                for lim in raw["limitations"]
            ]
        return cls(**raw)


# ---------------------------------------------------------------------------
# ScenarioEngine — the main engine
# ---------------------------------------------------------------------------

class ScenarioEngine:
    """Deterministic, domain-agnostic scenario evaluation engine.

    Defines, validates, and evaluates hypothetical scenarios using available
    evidence and model relationships. Does not invent formulas or assume
    linear responses.

    Usage:
        engine = ScenarioEngine(dataset_id="ds_001")
        definition = engine.define_scenario(
            name="Increase throughput",
            description="What if throughput increased by 20%?",
            parameters=[ScenarioParameter(name="throughput", ...)],
            evidence_items=[...],
        )
        result = engine.evaluate_scenario(definition, evidence_items, model_info)
        comparison = engine.compare_scenarios([result_a, result_b])
    """

    def __init__(self, dataset_id: str = ""):
        if not dataset_id:
            raise ValueError(
                "dataset_id is required. Scenarios must be scoped to a dataset."
            )
        self._dataset_id = dataset_id
        self._processing_notes: list[str] = []

    @property
    def dataset_id(self) -> str:
        return self._dataset_id

    @property
    def processing_notes(self) -> list[str]:
        return list(self._processing_notes)

    # -------------------------------------------------------------------
    # Define a scenario
    # -------------------------------------------------------------------

    def define_scenario(
        self,
        name: str,
        description: str,
        parameters: list,
        evidence_items: Optional[list] = None,
        assumptions: Optional[list] = None,
    ) -> ScenarioDefinition:
        """Create a validated scenario definition.

        Validates that parameters are well-formed and evidence is scoped
        to the correct dataset.

        Args:
            name: Human-readable scenario name
            description: What this scenario explores
            parameters: List of ScenarioParameter objects
            evidence_items: Supporting evidence (filtered by dataset)
            assumptions: Explicit assumptions for this scenario

        Returns:
            ScenarioDefinition with validated parameters
        """
        # Validate parameters
        validated_params = []
        for param in parameters:
            validated = self._validate_parameter(param)
            validated_params.append(validated)

        # Collect evidence IDs from dataset-scoped evidence
        evidence_ids = []
        if evidence_items:
            scoped = validate_dataset_scope(evidence_items, self._dataset_id)
            evidence_ids = [_get_attr(ev, "id", "") for ev in scoped]
            if len(scoped) < len(evidence_items):
                self._processing_notes.append(
                    f"Filtered {len(evidence_items) - len(scoped)} evidence items "
                    f"outside dataset '{self._dataset_id}'."
                )

        # Build limitations from unsupported parameters
        lims = []
        unsupported_names = [
            p.name for p in validated_params if not p.supported
        ]
        if unsupported_names:
            lims.append(create_limitation(
                f"Unsupported parameter changes: {', '.join(unsupported_names)}. "
                "These parameters cannot be numerically evaluated.",
                category=LimitationCategory.METHODOLOGY.value,
            ))

        return ScenarioDefinition(
            name=name,
            description=description,
            dataset_id=self._dataset_id,
            parameters=validated_params,
            assumptions=assumptions or [],
            evidence_ids=evidence_ids,
            limitations=lims,
        )

    # -------------------------------------------------------------------
    # Validate a parameter
    # -------------------------------------------------------------------

    def _validate_parameter(self, param) -> ScenarioParameter:
        """Validate a single scenario parameter.

        Checks for NaN/Inf values, computes change_pct if possible.
        """
        if isinstance(param, dict):
            param = ScenarioParameter.from_dict(param)

        name = _get_attr(param, "name", "")
        baseline = _safe_float(_get_attr(param, "baseline_value", None))
        alternative = _safe_float(_get_attr(param, "alternative_value", None))
        unit = _get_attr(param, "unit", "")
        supported = _get_attr(param, "supported", True)
        unsupported_reason = _get_attr(param, "unsupported_reason", "")

        # Validate values
        if alternative is None and supported:
            supported = False
            unsupported_reason = (
                unsupported_reason or
                "Alternative value is missing or non-finite."
            )

        # Compute change percentage if both values are available
        change_pct = None
        if baseline is not None and alternative is not None and baseline != 0.0:
            change_pct = ((alternative - baseline) / abs(baseline)) * 100.0
            change_pct = _safe_float(change_pct)

        return ScenarioParameter(
            name=name,
            baseline_value=baseline,
            alternative_value=alternative,
            unit=unit,
            supported=supported,
            unsupported_reason=unsupported_reason,
            change_pct=change_pct,
        )

    # -------------------------------------------------------------------
    # Evaluate a scenario
    # -------------------------------------------------------------------

    def evaluate_scenario(
        self,
        scenario: ScenarioDefinition,
        evidence_items: Optional[list] = None,
        model_info: Optional[dict] = None,
    ) -> ScenarioResult:
        """Evaluate a scenario against available evidence and model information.

        A scenario can be numerically evaluated ONLY when:
        1. All required parameters have baseline and alternative values.
        2. A defensible calculation method is available (model, formula).
        3. Evidence supports the relationship between parameters.

        When any of these conditions is not met, returns NOT_SIMULATABLE
        or INSUFFICIENT_EVIDENCE with a clear explanation.

        Args:
            scenario: The scenario definition to evaluate
            evidence_items: Available evidence for evaluation
            model_info: Optional model information dict with keys:
                - 'coefficients': dict mapping parameter name → coefficient
                - 'intercept': float baseline intercept
                - 'r_squared': float model fit metric
                - 'supported_range': dict mapping param → (min, max)
                - 'model_type': str describing the model
                - 'unit': str unit of the model's output
                All keys are optional; missing info limits evaluation.

        Returns:
            ScenarioResult with evaluation outcome
        """
        # Validate dataset scope
        if scenario.dataset_id != self._dataset_id:
            return self._make_insufficient_result(
                scenario,
                f"Scenario dataset_id '{scenario.dataset_id}' does not match "
                f"engine dataset_id '{self._dataset_id}'.",
            )

        # Filter evidence by dataset
        scoped_evidence = []
        if evidence_items:
            scoped_evidence = validate_dataset_scope(
                evidence_items, self._dataset_id
            )

        # Check if we have enough to evaluate
        if not scenario.parameters:
            return self._make_insufficient_result(
                scenario,
                "No parameters defined for this scenario.",
                evidence_items=scoped_evidence,
            )

        # Check parameter support
        supported_params = [p for p in scenario.parameters if p.supported]
        unsupported_params = [p for p in scenario.parameters if not p.supported]

        if not supported_params:
            return self._make_not_simulatable_result(
                scenario,
                "No supported parameters for numerical evaluation.",
                unsupported=[p.name for p in unsupported_params],
                evidence_items=scoped_evidence,
            )

        # Attempt numerical evaluation if model_info is available
        if model_info and isinstance(model_info, dict):
            return self._evaluate_with_model(
                scenario, supported_params, unsupported_params,
                scoped_evidence, model_info,
            )

        # No model available — check if evidence alone supports calculation
        if scoped_evidence:
            return self._evaluate_qualitative(
                scenario, supported_params, unsupported_params,
                scoped_evidence,
            )

        # Nothing to evaluate with
        return self._make_insufficient_result(
            scenario,
            "No model information or evidence available for evaluation.",
            evidence_items=scoped_evidence,
        )

    # -------------------------------------------------------------------
    # Evaluate with model information
    # -------------------------------------------------------------------

    def _evaluate_with_model(
        self,
        scenario: ScenarioDefinition,
        supported_params: list,
        unsupported_params: list,
        evidence_items: list,
        model_info: dict,
    ) -> ScenarioResult:
        """Evaluate scenario using provided model information."""
        coefficients = model_info.get("coefficients", {})
        intercept = _safe_float(model_info.get("intercept", 0.0)) or 0.0
        r_squared = _safe_float(model_info.get("r_squared"))
        supported_range = model_info.get("supported_range", {})
        model_type = model_info.get("model_type", "linear model")
        model_unit = model_info.get("unit", "")

        if not coefficients:
            return self._make_not_simulatable_result(
                scenario,
                "Model information provided but no coefficients available.",
                missing=["Model coefficients for parameter relationships"],
                unsupported=[p.name for p in unsupported_params],
                evidence_items=evidence_items,
            )

        # Check which supported params have coefficients
        evaluable_params = []
        unevaluable_params = []
        out_of_range_warnings = []

        for param in supported_params:
            if param.name in coefficients:
                coeff = _safe_float(coefficients[param.name])
                if coeff is None:
                    unevaluable_params.append(param.name)
                    continue

                # Check range
                if param.name in supported_range:
                    param_range = supported_range[param.name]
                    if (isinstance(param_range, (list, tuple))
                            and len(param_range) == 2):
                        range_min, range_max = param_range
                        alt = param.alternative_value
                        if alt is not None:
                            if alt < range_min or alt > range_max:
                                out_of_range_warnings.append(
                                    f"Parameter '{param.name}' value {alt} is "
                                    f"outside supported range [{range_min}, {range_max}]."
                                )

                evaluable_params.append((param, coeff))
            else:
                unevaluable_params.append(param.name)

        if not evaluable_params:
            return self._make_not_simulatable_result(
                scenario,
                "No model coefficients match the scenario parameters.",
                missing=[
                    f"Coefficient for parameter '{p.name}'"
                    for p in supported_params
                ],
                unsupported=[p.name for p in unsupported_params],
                evidence_items=evidence_items,
            )

        # Compute baseline and alternative outcomes
        baseline_outcome = intercept
        alternative_outcome = intercept

        for param, coeff in evaluable_params:
            if param.baseline_value is not None:
                baseline_outcome += coeff * param.baseline_value
            if param.alternative_value is not None:
                alternative_outcome += coeff * param.alternative_value

        baseline_outcome = _safe_float(baseline_outcome)
        alternative_outcome = _safe_float(alternative_outcome)

        delta = None
        delta_pct = None
        if baseline_outcome is not None and alternative_outcome is not None:
            delta = alternative_outcome - baseline_outcome
            delta = _safe_float(delta)
            if baseline_outcome != 0.0 and delta is not None:
                delta_pct = _safe_float((delta / abs(baseline_outcome)) * 100.0)

        # Determine status
        if unevaluable_params or unsupported_params:
            status = ScenarioStatus.PARTIALLY_SIMULATED.value
        else:
            status = ScenarioStatus.SIMULATED.value

        # Build confidence
        confidence = assess_confidence(
            evidence_items, dataset_id=self._dataset_id
        ) if evidence_items else ConfidenceScore(
            level=ConfidenceLevel.LOW.value,
            explanation="No supporting evidence; result based on model only.",
            evidence_count=0,
            evidence_quality="none",
            reduction_reasons=["No evidence items supporting the model."],
            dataset_id=self._dataset_id,
        )

        # Build limitations
        lims = list(scenario.limitations)
        lims.append(create_limitation(
            "Scenario results are model-based projections, not observed outcomes. "
            "Actual results depend on factors beyond the model.",
            category=LimitationCategory.MODEL_UNCERTAINTY.value,
        ))
        if out_of_range_warnings:
            for warning in out_of_range_warnings:
                lims.append(create_limitation(
                    f"Extrapolation warning: {warning}",
                    category=LimitationCategory.SCOPE.value,
                ))

        # Build assumptions
        assumptions = list(scenario.assumptions)
        assumptions.append(
            "Model relationships hold under the hypothetical conditions."
        )
        assumptions.append(
            "No external factors intervene during the projection period."
        )
        if model_type:
            assumptions.append(
                f"The {model_type} accurately represents the response surface."
            )

        # Description
        desc_parts = [f"Scenario '{scenario.name}' evaluated using {model_type}"]
        if delta is not None:
            desc_parts.append(f": projected delta = {delta:.4f}")
            if model_unit:
                desc_parts.append(f" {model_unit}")
        description = "".join(desc_parts)

        explanation = (
            f"Evaluation used a {model_type} with "
            f"{len(evaluable_params)} parameter(s). "
            "The result is a model-based projection, not a measured outcome. "
        )
        if r_squared is not None:
            explanation += f"Model fit (R²) = {r_squared:.4f}. "
        if unevaluable_params:
            explanation += (
                f"Parameters without coefficients: {', '.join(unevaluable_params)}. "
            )
        if out_of_range_warnings:
            explanation += (
                "Warning: some parameter values fall outside the model's "
                "supported range — results may be unreliable. "
            )

        return ScenarioResult(
            scenario_id=scenario.id,
            dataset_id=self._dataset_id,
            status=status,
            estimated_outcome=alternative_outcome,
            baseline_outcome=baseline_outcome,
            delta=delta,
            delta_pct=delta_pct,
            unit=model_unit,
            estimation_method=f"Model-based projection ({model_type})",
            model_description=model_type,
            description=description,
            explanation=explanation,
            missing_requirements=(
                [f"Coefficient for '{n}'" for n in unevaluable_params]
                if unevaluable_params else []
            ),
            unsupported_parameters=[p.name for p in unsupported_params],
            evidence_ids=[_get_attr(ev, "id", "") for ev in evidence_items],
            confidence=confidence,
            limitations=lims,
            assumptions=assumptions,
        )

    # -------------------------------------------------------------------
    # Qualitative evaluation (evidence but no model)
    # -------------------------------------------------------------------

    def _evaluate_qualitative(
        self,
        scenario: ScenarioDefinition,
        supported_params: list,
        unsupported_params: list,
        evidence_items: list,
    ) -> ScenarioResult:
        """Evaluate scenario qualitatively when no model is available."""
        confidence = assess_confidence(
            evidence_items, dataset_id=self._dataset_id
        )

        lims = list(scenario.limitations)
        lims.append(create_limitation(
            "No defensible model or calculation available for numerical "
            "evaluation. Only qualitative assessment is possible.",
            category=LimitationCategory.METHODOLOGY.value,
        ))

        return ScenarioResult(
            scenario_id=scenario.id,
            dataset_id=self._dataset_id,
            status=ScenarioStatus.NOT_SIMULATABLE.value,
            description=(
                f"Scenario '{scenario.name}' cannot be numerically evaluated. "
                "Evidence is available but no calculation model is provided."
            ),
            explanation=(
                "Numerical simulation requires a defensible model relating "
                "the scenario parameters to outcomes. Without such a model, "
                "the engine does not invent a formula or assume a linear "
                "response. The available evidence provides context but cannot "
                "produce a projected outcome."
            ),
            missing_requirements=[
                "Model coefficients or transfer function for parameter relationships",
                "Baseline outcome measurement",
            ],
            unsupported_parameters=[p.name for p in unsupported_params],
            evidence_ids=[_get_attr(ev, "id", "") for ev in evidence_items],
            confidence=confidence,
            limitations=lims,
            assumptions=list(scenario.assumptions),
        )

    # -------------------------------------------------------------------
    # Compare scenarios
    # -------------------------------------------------------------------

    def compare_scenarios(
        self,
        results: list,
    ) -> ScenarioComparison:
        """Compare two or more scenario results.

        Args:
            results: List of ScenarioResult objects to compare

        Returns:
            ScenarioComparison with notes and limitations
        """
        if len(results) < 2:
            raise ValueError(
                "Scenario comparison requires at least two results."
            )

        # Validate dataset scope
        for r in results:
            if r.dataset_id != self._dataset_id:
                raise ValueError(
                    f"Cannot compare scenarios from different datasets. "
                    f"Result dataset_id '{r.dataset_id}' does not match "
                    f"engine dataset_id '{self._dataset_id}'."
                )

        # Build comparison notes
        notes = []
        lims = []

        # Check if all are simulated
        statuses = [r.status for r in results]
        if ScenarioStatus.NOT_SIMULATABLE.value in statuses:
            notes.append(
                "Some scenarios could not be numerically evaluated. "
                "Comparison is limited to qualitative assessment."
            )
            lims.append(create_limitation(
                "Not all scenarios were numerically evaluated.",
                category=LimitationCategory.METHODOLOGY.value,
            ))

        if ScenarioStatus.INSUFFICIENT_EVIDENCE.value in statuses:
            notes.append(
                "Some scenarios lacked sufficient evidence for evaluation."
            )

        # Check unit compatibility
        units = set()
        for r in results:
            if r.unit:
                units.add(r.unit)
        if len(units) > 1:
            notes.append(
                f"Scenarios use different units ({', '.join(sorted(units))}). "
                "Direct numeric comparison may be misleading."
            )
            lims.append(create_limitation(
                f"Incompatible units: {', '.join(sorted(units))}. "
                "Scenarios measured in different units cannot be directly ranked.",
                category=LimitationCategory.METHODOLOGY.value,
            ))

        # Compare quantified results
        quantified = [r for r in results if r.estimated_outcome is not None]
        if len(quantified) >= 2:
            values = [(r.scenario_id, r.estimated_outcome) for r in quantified]
            best = max(values, key=lambda x: x[1])
            worst = min(values, key=lambda x: x[1])
            if best[1] != worst[1]:
                notes.append(
                    f"Highest projected outcome: scenario '{best[0]}' "
                    f"({best[1]:.4f}). "
                    f"Lowest: scenario '{worst[0]}' ({worst[1]:.4f})."
                )
        elif quantified:
            notes.append(
                "Only one scenario produced a numeric outcome; "
                "quantitative comparison is limited."
            )
        else:
            notes.append(
                "No scenarios produced numeric outcomes. "
                "Comparison is qualitative only."
            )

        # Add disclaimers
        notes.append(
            "All projections are model-based estimates, not guaranteed outcomes."
        )

        return ScenarioComparison(
            dataset_id=self._dataset_id,
            scenario_ids=[r.scenario_id for r in results],
            results=list(results),
            comparison_notes=notes,
            limitations=lims,
        )

    # -------------------------------------------------------------------
    # Helpers for insufficient / not-simulatable results
    # -------------------------------------------------------------------

    def _make_insufficient_result(
        self,
        scenario: ScenarioDefinition,
        reason: str,
        evidence_items: Optional[list] = None,
    ) -> ScenarioResult:
        """Create a result indicating insufficient evidence."""
        confidence = ConfidenceScore(
            level=ConfidenceLevel.INSUFFICIENT.value,
            explanation=reason,
            evidence_count=len(evidence_items) if evidence_items else 0,
            evidence_quality="none",
            dataset_id=self._dataset_id,
        )

        return ScenarioResult(
            scenario_id=scenario.id,
            dataset_id=self._dataset_id,
            status=ScenarioStatus.INSUFFICIENT_EVIDENCE.value,
            description=f"Scenario '{scenario.name}' cannot be evaluated: {reason}",
            explanation=reason,
            missing_requirements=[reason],
            evidence_ids=(
                [_get_attr(ev, "id", "") for ev in evidence_items]
                if evidence_items else []
            ),
            confidence=confidence,
            limitations=list(scenario.limitations),
            assumptions=list(scenario.assumptions),
        )

    def _make_not_simulatable_result(
        self,
        scenario: ScenarioDefinition,
        reason: str,
        missing: Optional[list] = None,
        unsupported: Optional[list] = None,
        evidence_items: Optional[list] = None,
    ) -> ScenarioResult:
        """Create a result indicating the scenario cannot be simulated."""
        confidence = assess_confidence(
            evidence_items or [], dataset_id=self._dataset_id
        )

        lims = list(scenario.limitations)
        lims.append(create_limitation(
            reason,
            category=LimitationCategory.METHODOLOGY.value,
        ))

        return ScenarioResult(
            scenario_id=scenario.id,
            dataset_id=self._dataset_id,
            status=ScenarioStatus.NOT_SIMULATABLE.value,
            description=f"Scenario '{scenario.name}' cannot be numerically evaluated.",
            explanation=(
                f"{reason} "
                "A defensible model or calculation method is required for "
                "numerical scenario evaluation."
            ),
            missing_requirements=missing or [reason],
            unsupported_parameters=unsupported or [],
            evidence_ids=(
                [_get_attr(ev, "id", "") for ev in evidence_items]
                if evidence_items else []
            ),
            confidence=confidence,
            limitations=lims,
            assumptions=list(scenario.assumptions),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_attr(obj, name: str, default=None):
    """Get attribute from an object or dict."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)
