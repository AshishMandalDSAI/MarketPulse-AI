"""Phase 7 Part 3 — Impact Quantifier.

Deterministic, domain-agnostic engine that assesses the potential impact
of DecisionSignal objects using available evidence and analytical results.

Impact types:
- MEASURED: Directly computed from data (aggregations, observed differences)
- MODEL_ESTIMATED: Derived from a fitted model (regression coefficients, etc.)
- SCENARIO_BASED: Projected from a what-if scenario or optimization
- QUALITATIVE: Described but not numerically quantifiable from evidence
- UNKNOWN: Insufficient evidence to assess impact

Design principles:
1. Quantify only when evidence supports the calculation.
2. Preserve original units; never combine incompatible units.
3. Correlation is not causation — do not describe model coefficients as causal.
4. Do not present forecasts as observed outcomes.
5. Do not invent currency, percentages, or financial estimates.
6. Insufficient evidence → explicit UNKNOWN, not fabricated numbers.
7. Impact magnitude is separate from confidence and priority.
8. Dataset isolation enforced at every step.
9. No LLM calls. No global mutable caches. No demo-data fallback.
"""

from __future__ import annotations

import enum
import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from src.decision.contracts import (
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


# ---------------------------------------------------------------------------
# Impact Type — how the impact was estimated
# ---------------------------------------------------------------------------

class ImpactType(str, enum.Enum):
    """How the impact was estimated or derived."""
    MEASURED = "measured"              # Directly computed from data
    MODEL_ESTIMATED = "model_estimated"  # Derived from a fitted model
    SCENARIO_BASED = "scenario_based"   # Projected from a what-if or simulation
    QUALITATIVE = "qualitative"         # Described but not numerically quantifiable
    UNKNOWN = "unknown"                 # Insufficient evidence


# ---------------------------------------------------------------------------
# Impact Category — what area the impact affects
# ---------------------------------------------------------------------------

class ImpactCategory(str, enum.Enum):
    """Domain-agnostic impact category."""
    PERFORMANCE = "performance"         # Metric performance (KPIs, targets)
    EFFICIENCY = "efficiency"           # Operational efficiency
    RISK = "risk"                       # Risk or exposure
    QUALITY = "quality"                 # Quality (data, process, product)
    VOLUME = "volume"                   # Volume or throughput
    RELATIONSHIP = "relationship"       # Statistical relationship strength
    ANOMALY = "anomaly"                 # Anomalous deviation
    GENERAL = "general"                 # Uncategorised


# ---------------------------------------------------------------------------
# ImpactAssessment — the core output contract
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ImpactAssessment:
    """A single impact assessment for a DecisionSignal.

    Immutable. Carries the estimated impact, its type, supporting evidence,
    limitations, and an explanation of how the assessment was derived.
    """
    id: str = field(default_factory=lambda: _make_id("imp_"))
    signal_id: str = ""
    dataset_id: str = ""

    # Impact classification
    impact_type: str = ImpactType.UNKNOWN.value
    impact_category: str = ImpactCategory.GENERAL.value
    estimation_method: str = ""          # Brief description of how impact was assessed

    # Numeric estimate (only when evidence supports it)
    estimated_value: Optional[float] = None
    unit: str = ""                        # Original unit from the evidence
    time_horizon: str = ""                # e.g. "per period", "forecast horizon"

    # Range / uncertainty
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None

    # Human-readable
    description: str = ""                 # What the impact is
    explanation: str = ""                 # How the impact was assessed

    # Supporting evidence
    evidence_ids: list = field(default_factory=list)

    # Confidence (from Part 1)
    confidence: Optional[ConfidenceScore] = None

    # Limitations and assumptions
    limitations: list = field(default_factory=list)   # list[DecisionLimitation]
    assumptions: list = field(default_factory=list)    # list[str]

    def __post_init__(self):
        if self.impact_type not in {e.value for e in ImpactType}:
            raise ValueError(
                f"Invalid impact_type: {self.impact_type!r}. "
                f"Must be one of: {[e.value for e in ImpactType]}"
            )
        if self.impact_category not in {e.value for e in ImpactCategory}:
            raise ValueError(
                f"Invalid impact_category: {self.impact_category!r}. "
                f"Must be one of: {[e.value for e in ImpactCategory]}"
            )
        if not self.description:
            raise ValueError("ImpactAssessment requires a non-empty description.")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "signal_id": self.signal_id,
            "dataset_id": self.dataset_id,
            "impact_type": self.impact_type,
            "impact_category": self.impact_category,
            "estimation_method": self.estimation_method,
            "estimated_value": _safe_float(self.estimated_value),
            "unit": self.unit,
            "time_horizon": self.time_horizon,
            "lower_bound": _safe_float(self.lower_bound),
            "upper_bound": _safe_float(self.upper_bound),
            "description": self.description,
            "explanation": self.explanation,
            "evidence_ids": list(self.evidence_ids),
            "confidence": self.confidence.to_dict() if self.confidence else None,
            "limitations": [
                lim.to_dict() if hasattr(lim, "to_dict") else lim
                for lim in self.limitations
            ],
            "assumptions": list(self.assumptions),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ImpactAssessment":
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
# ImpactReport — aggregate output
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ImpactReport:
    """Aggregate output of the Impact Quantifier for a dataset.

    Contains all impact assessments, an evidence registry subset,
    and processing notes.
    """
    dataset_id: str = ""
    assessments: list = field(default_factory=list)     # list[ImpactAssessment]
    processing_notes: list = field(default_factory=list)  # list[str]

    def assessed_signals(self) -> list:
        """Return signal IDs that have assessments."""
        return [a.signal_id for a in self.assessments]

    def by_type(self, impact_type: str) -> list:
        """Filter assessments by impact type."""
        return [a for a in self.assessments if a.impact_type == impact_type]

    def by_category(self, category: str) -> list:
        """Filter assessments by impact category."""
        return [a for a in self.assessments if a.impact_category == category]

    def quantified(self) -> list:
        """Return assessments that have a numeric estimate."""
        return [a for a in self.assessments if a.estimated_value is not None]

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "assessments": [a.to_dict() for a in self.assessments],
            "processing_notes": list(self.processing_notes),
            "assessment_count": len(self.assessments),
            "quantified_count": len(self.quantified()),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ImpactReport":
        raw = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if "assessments" in raw and isinstance(raw["assessments"], list):
            raw["assessments"] = [
                ImpactAssessment.from_dict(a) if isinstance(a, dict) else a
                for a in raw["assessments"]
            ]
        return cls(**raw)


# ---------------------------------------------------------------------------
# ImpactQuantifier — the engine
# ---------------------------------------------------------------------------

class ImpactQuantifier:
    """Deterministic, domain-agnostic impact quantification engine.

    Takes SynthesizedSignal objects (from Part 2) and their supporting
    evidence, and produces ImpactAssessment objects.

    Usage:
        quantifier = ImpactQuantifier(dataset_id="ds_001")
        report = quantifier.assess_signals(
            synthesized_signals,
            evidence_registry,
        )
    """

    def __init__(self, dataset_id: str = ""):
        if not dataset_id:
            raise ValueError(
                "dataset_id is required. Impact assessments must be scoped to a dataset."
            )
        self._dataset_id = dataset_id
        self._assessments: list[ImpactAssessment] = []
        self._processing_notes: list[str] = []

    @property
    def dataset_id(self) -> str:
        return self._dataset_id

    # -------------------------------------------------------------------
    # Main entry point
    # -------------------------------------------------------------------

    def assess_signals(
        self,
        synthesized_signals: list,
        evidence_registry: dict,
    ) -> ImpactReport:
        """Assess impact of each synthesized signal.

        Args:
            synthesized_signals: list of SynthesizedSignal (from Part 2)
            evidence_registry: dict mapping evidence ID → DecisionEvidence

        Returns:
            ImpactReport with all assessments.
        """
        if not synthesized_signals:
            self._processing_notes.append(
                "No signals provided for impact assessment."
            )
            return self._build_report()

        for synth_signal in synthesized_signals:
            try:
                assessment = self._assess_single(synth_signal, evidence_registry)
                if assessment is not None:
                    self._assessments.append(assessment)
            except Exception as exc:
                self._processing_notes.append(
                    f"Failed to assess signal "
                    f"'{_get_attr(synth_signal, 'source_type', 'unknown')}': {exc}"
                )

        return self._build_report()

    def assess_single_signal(
        self,
        synthesized_signal,
        evidence_registry: dict,
    ) -> Optional[ImpactAssessment]:
        """Assess impact of a single signal. Public API for targeted use."""
        return self._assess_single(synthesized_signal, evidence_registry)

    # -------------------------------------------------------------------
    # Per-signal assessment
    # -------------------------------------------------------------------

    def _assess_single(
        self,
        synth_signal,
        evidence_registry: dict,
    ) -> Optional[ImpactAssessment]:
        """Assess one SynthesizedSignal's impact."""
        signal = _get_attr(synth_signal, "signal", None)
        if signal is None:
            self._processing_notes.append(
                "SynthesizedSignal has no underlying signal, skipping."
            )
            return None

        # Validate dataset scope
        sig_ds = _get_attr(signal, "dataset_id", "")
        if sig_ds and sig_ds != self._dataset_id:
            self._processing_notes.append(
                f"Signal dataset_id '{sig_ds}' does not match "
                f"quantifier dataset_id '{self._dataset_id}', skipping."
            )
            return None

        # Collect supporting evidence
        evidence_items = _get_attr(synth_signal, "evidence_items", [])
        if not evidence_items:
            # Try evidence_ids lookup
            ev_ids = _get_attr(signal, "evidence_ids", [])
            for eid in ev_ids:
                ev = evidence_registry.get(eid)
                if ev is not None:
                    evidence_items.append(ev)

        if not evidence_items:
            return self._make_unknown_assessment(
                signal,
                "No supporting evidence available for impact assessment.",
            )

        # Validate evidence dataset scope
        scoped_evidence = validate_dataset_scope(evidence_items, self._dataset_id)
        if not scoped_evidence:
            return self._make_unknown_assessment(
                signal,
                f"All evidence filtered out for dataset '{self._dataset_id}'.",
            )

        # Dispatch based on source type
        source_type = _get_attr(synth_signal, "source_type", "")
        return self._dispatch_assessment(
            signal, synth_signal, scoped_evidence, source_type,
        )

    # -------------------------------------------------------------------
    # Dispatch by source type
    # -------------------------------------------------------------------

    def _dispatch_assessment(
        self,
        signal: DecisionSignal,
        synth_signal,
        evidence_items: list,
        source_type: str,
    ) -> ImpactAssessment:
        """Route to the appropriate impact assessment strategy."""
        # KPI — measured impact from data
        if "kpi" in source_type:
            return self._assess_kpi_impact(signal, evidence_items)

        # Anomaly — measured deviation
        if "anomaly" in source_type:
            return self._assess_anomaly_impact(signal, evidence_items)

        # Driver / feature importance — model-estimated association
        if "driver" in source_type or "feature" in source_type.lower():
            return self._assess_driver_impact(signal, evidence_items)

        # Relationship / insight — observational association
        if "relationship" in source_type or "insight" in source_type:
            return self._assess_relationship_impact(signal, evidence_items)

        # Forecast — model projection
        if "forecast" in source_type:
            return self._assess_forecast_impact(signal, evidence_items)

        # Regression / classification — model-estimated
        if "regression" in source_type or "classification" in source_type:
            return self._assess_model_impact(signal, evidence_items)

        # Segmentation — qualitative
        if "segmentation" in source_type:
            return self._assess_segmentation_impact(signal, evidence_items)

        # What-if / optimization — scenario-based
        if "whatif" in source_type or "scenario" in source_type:
            return self._assess_scenario_impact(signal, evidence_items)
        if "optimization" in source_type or "optimizer" in source_type:
            return self._assess_optimization_impact(signal, evidence_items)

        # MMM — model-estimated
        if "mmm" in source_type:
            return self._assess_mmm_impact(signal, evidence_items)

        # Contribution / ROI — model-estimated
        if "contribution" in source_type or "roi" in source_type:
            return self._assess_contribution_impact(signal, evidence_items)

        # Default: qualitative from description
        return self._assess_generic_impact(signal, evidence_items)

    # -------------------------------------------------------------------
    # KPI impact
    # -------------------------------------------------------------------

    def _assess_kpi_impact(
        self, signal: DecisionSignal, evidence_items: list,
    ) -> ImpactAssessment:
        """Assess impact from a KPI observation (MEASURED)."""
        ev = evidence_items[0]
        metric_name = _get_attr(ev, "metric_name", "")
        metric_value = _safe_float(_get_attr(ev, "metric_value", None))
        unit = _get_attr_nested(ev, "details", "unit", "")

        confidence = assess_confidence(evidence_items, dataset_id=self._dataset_id)

        return ImpactAssessment(
            signal_id=signal.id,
            dataset_id=self._dataset_id,
            impact_type=ImpactType.MEASURED.value,
            impact_category=ImpactCategory.PERFORMANCE.value,
            estimation_method="Direct observation from dataset",
            estimated_value=metric_value,
            unit=unit,
            description=(
                f"Observed value: {metric_name} = {metric_value}"
                if metric_value is not None
                else f"Observed metric: {metric_name}"
            ),
            explanation=(
                "Impact is directly measured from the dataset. "
                "This is an observed fact, not a projection."
            ),
            evidence_ids=[_get_attr(e, "id", "") for e in evidence_items],
            confidence=confidence,
            limitations=list(signal.limitations),
        )

    # -------------------------------------------------------------------
    # Anomaly impact
    # -------------------------------------------------------------------

    def _assess_anomaly_impact(
        self, signal: DecisionSignal, evidence_items: list,
    ) -> ImpactAssessment:
        """Assess impact from an anomaly detection (MEASURED deviation)."""
        ev = evidence_items[0]
        details = _get_attr(ev, "details", {})
        observed = _safe_float(details.get("observed"))
        expected = _safe_float(details.get("expected"))
        score = _safe_float(_get_attr(ev, "metric_value", None))

        deviation = None
        if observed is not None and expected is not None and expected != 0.0:
            deviation = observed - expected

        confidence = assess_confidence(evidence_items, dataset_id=self._dataset_id)

        description = f"Anomalous deviation detected"
        if deviation is not None:
            description += f": deviation = {deviation:.4f} (observed={observed}, expected={expected})"

        return ImpactAssessment(
            signal_id=signal.id,
            dataset_id=self._dataset_id,
            impact_type=ImpactType.MEASURED.value,
            impact_category=ImpactCategory.ANOMALY.value,
            estimation_method="Anomaly detection comparison (observed vs expected)",
            estimated_value=deviation,
            description=description,
            explanation=(
                "Impact measured as the difference between the observed and "
                "expected value. The anomaly score indicates how unusual "
                "the observation is relative to normal patterns."
            ),
            evidence_ids=[_get_attr(e, "id", "") for e in evidence_items],
            confidence=confidence,
            limitations=list(signal.limitations),
        )

    # -------------------------------------------------------------------
    # Driver / feature importance impact
    # -------------------------------------------------------------------

    def _assess_driver_impact(
        self, signal: DecisionSignal, evidence_items: list,
    ) -> ImpactAssessment:
        """Assess impact from driver analysis or feature importance.

        MODEL_ESTIMATED: the importance score is from a model, not causal.
        """
        ev = evidence_items[0]
        metric_value = _safe_float(_get_attr(ev, "metric_value", None))
        details = _get_attr(ev, "details", {})
        direction = details.get("direction", "")
        method = details.get("method", "")

        confidence = assess_confidence(evidence_items, dataset_id=self._dataset_id)

        lims = list(signal.limitations)
        lims.append(create_limitation(
            "This is a model-estimated association, not a causal effect. "
            "Feature importance reflects predictive power, not causation.",
            category=LimitationCategory.OBSERVATIONAL.value,
        ))

        return ImpactAssessment(
            signal_id=signal.id,
            dataset_id=self._dataset_id,
            impact_type=ImpactType.MODEL_ESTIMATED.value,
            impact_category=ImpactCategory.RELATIONSHIP.value,
            estimation_method=f"Driver analysis via {method}" if method else "Driver analysis",
            estimated_value=metric_value,
            unit="association strength",
            description=(
                f"Model-estimated association strength: {metric_value}"
                if metric_value is not None
                else "Association detected but strength not quantifiable"
            ),
            explanation=(
                "Impact estimated from model-based driver analysis. "
                "The value represents the strength of statistical association "
                "between the feature and the target variable. "
                "This does not establish a causal relationship."
            ),
            evidence_ids=[_get_attr(e, "id", "") for e in evidence_items],
            confidence=confidence,
            limitations=lims,
            assumptions=[
                "Association does not imply causation.",
                "Importance reflects model-specific predictive contribution.",
            ],
        )

    # -------------------------------------------------------------------
    # Relationship / insight impact
    # -------------------------------------------------------------------

    def _assess_relationship_impact(
        self, signal: DecisionSignal, evidence_items: list,
    ) -> ImpactAssessment:
        """Assess impact from a statistical relationship or insight."""
        ev = evidence_items[0]
        metric_value = _safe_float(_get_attr(ev, "metric_value", None))
        details = _get_attr(ev, "details", {})
        rel_type = details.get("relationship_type", "")

        confidence = assess_confidence(evidence_items, dataset_id=self._dataset_id)

        lims = list(signal.limitations)
        lims.append(create_limitation(
            "Statistical relationship does not establish causation.",
            category=LimitationCategory.OBSERVATIONAL.value,
        ))

        return ImpactAssessment(
            signal_id=signal.id,
            dataset_id=self._dataset_id,
            impact_type=ImpactType.MODEL_ESTIMATED.value,
            impact_category=ImpactCategory.RELATIONSHIP.value,
            estimation_method=f"Statistical analysis ({rel_type})" if rel_type else "Statistical analysis",
            estimated_value=metric_value,
            unit="correlation strength" if rel_type == "correlation" else "association strength",
            description=(
                f"Relationship strength: {metric_value}"
                if metric_value is not None
                else "Statistical relationship detected"
            ),
            explanation=(
                "Impact quantified as the strength of the observed "
                "statistical relationship. This reflects correlation "
                "or association, not a measured causal effect."
            ),
            evidence_ids=[_get_attr(e, "id", "") for e in evidence_items],
            confidence=confidence,
            limitations=lims,
        )

    # -------------------------------------------------------------------
    # Forecast impact
    # -------------------------------------------------------------------

    def _assess_forecast_impact(
        self, signal: DecisionSignal, evidence_items: list,
    ) -> ImpactAssessment:
        """Assess impact from a forecast (MODEL_ESTIMATED projection)."""
        ev = evidence_items[0]
        details = _get_attr(ev, "details", {})
        metrics = details.get("metrics", {})
        model_name = details.get("model", "")
        horizon = _safe_float(_get_attr(ev, "metric_value", None))

        # Try to extract prediction interval width as uncertainty indicator
        mae = _safe_float(metrics.get("mae") or metrics.get("MAE"))
        rmse = _safe_float(metrics.get("rmse") or metrics.get("RMSE"))

        confidence = assess_confidence(evidence_items, dataset_id=self._dataset_id)

        lims = list(signal.limitations)
        lims.append(create_limitation(
            "Forecast values are model projections, not observed outcomes. "
            "Actual results may differ significantly.",
            category=LimitationCategory.MODEL_UNCERTAINTY.value,
        ))

        time_horizon = ""
        if horizon is not None and horizon > 0:
            time_horizon = f"{int(horizon)} periods ahead"

        return ImpactAssessment(
            signal_id=signal.id,
            dataset_id=self._dataset_id,
            impact_type=ImpactType.MODEL_ESTIMATED.value,
            impact_category=ImpactCategory.PERFORMANCE.value,
            estimation_method=f"Forecast model ({model_name})" if model_name else "Forecast model",
            estimated_value=rmse or mae,
            unit="forecast error (RMSE or MAE)",
            time_horizon=time_horizon,
            description=(
                f"Forecast projection for {time_horizon}"
                if time_horizon
                else "Forecast projection"
            ),
            explanation=(
                "Impact estimated from the forecast model's error metrics. "
                "The estimated value represents the model's typical error "
                "(RMSE or MAE), indicating the range of uncertainty. "
                "Forecast values are projections, not facts."
            ),
            evidence_ids=[_get_attr(e, "id", "") for e in evidence_items],
            confidence=confidence,
            limitations=lims,
            assumptions=[
                "Future patterns follow historical trends.",
                "The forecast model's assumptions hold for the projection period.",
            ],
        )

    # -------------------------------------------------------------------
    # Regression / classification model impact
    # -------------------------------------------------------------------

    def _assess_model_impact(
        self, signal: DecisionSignal, evidence_items: list,
    ) -> ImpactAssessment:
        """Assess impact from regression or classification models."""
        ev = evidence_items[0]
        details = _get_attr(ev, "details", {})
        metrics = details.get("metrics", {})
        model_name = details.get("model", "")
        metric_value = _safe_float(_get_attr(ev, "metric_value", None))

        r2 = _safe_float(metrics.get("r2") or metrics.get("R2"))
        accuracy = _safe_float(metrics.get("accuracy"))
        primary_metric = r2 if r2 is not None else accuracy

        confidence = assess_confidence(evidence_items, dataset_id=self._dataset_id)

        lims = list(signal.limitations)
        lims.append(create_limitation(
            "Model fit metrics describe how well the model explains "
            "variation in the data, not the magnitude of any causal effect.",
            category=LimitationCategory.MODEL_UNCERTAINTY.value,
        ))

        description = f"Model explains"
        if r2 is not None:
            description += f" {r2:.1%} of target variance (R²={r2:.4f})"
        elif accuracy is not None:
            description += f" target with {accuracy:.1%} accuracy"
        else:
            description = "Model fitted but explanatory power not available"

        return ImpactAssessment(
            signal_id=signal.id,
            dataset_id=self._dataset_id,
            impact_type=ImpactType.MODEL_ESTIMATED.value,
            impact_category=ImpactCategory.PERFORMANCE.value,
            estimation_method=f"Predictive model ({model_name})" if model_name else "Predictive model",
            estimated_value=primary_metric,
            unit="R²" if r2 is not None else ("accuracy" if accuracy is not None else ""),
            description=description,
            explanation=(
                "Impact measured by the model's explanatory power. "
                "This indicates how much of the target variation the model "
                "captures, not the business impact of any particular feature."
            ),
            evidence_ids=[_get_attr(e, "id", "") for e in evidence_items],
            confidence=confidence,
            limitations=lims,
        )

    # -------------------------------------------------------------------
    # Segmentation impact
    # -------------------------------------------------------------------

    def _assess_segmentation_impact(
        self, signal: DecisionSignal, evidence_items: list,
    ) -> ImpactAssessment:
        """Assess impact from segmentation (QUALITATIVE)."""
        ev = evidence_items[0]
        details = _get_attr(ev, "details", {})
        quality = details.get("quality_metrics", {})
        silhouette = _safe_float(quality.get("silhouette"))
        cluster_count = _safe_float(_get_attr(ev, "metric_value", None))

        confidence = assess_confidence(evidence_items, dataset_id=self._dataset_id)

        return ImpactAssessment(
            signal_id=signal.id,
            dataset_id=self._dataset_id,
            impact_type=ImpactType.QUALITATIVE.value,
            impact_category=ImpactCategory.GENERAL.value,
            estimation_method="Cluster analysis",
            estimated_value=silhouette,
            unit="silhouette score" if silhouette is not None else "",
            description=(
                f"Dataset segmented into {int(cluster_count)} groups"
                if cluster_count is not None
                else "Dataset segmented into groups"
            ),
            explanation=(
                "Segmentation identifies groups within the data. "
                "The silhouette score measures cluster separation quality. "
                "Business impact depends on how these segments are used."
            ),
            evidence_ids=[_get_attr(e, "id", "") for e in evidence_items],
            confidence=confidence,
            limitations=list(signal.limitations),
        )

    # -------------------------------------------------------------------
    # What-if / scenario impact
    # -------------------------------------------------------------------

    def _assess_scenario_impact(
        self, signal: DecisionSignal, evidence_items: list,
    ) -> ImpactAssessment:
        """Assess impact from a what-if scenario (SCENARIO_BASED)."""
        ev = evidence_items[0]
        metric_value = _safe_float(_get_attr(ev, "metric_value", None))
        details = _get_attr(ev, "details", {})
        disclaimer = details.get("disclaimer", "")

        confidence = assess_confidence(evidence_items, dataset_id=self._dataset_id)

        lims = list(signal.limitations)
        lims.append(create_limitation(
            "Scenario projections are hypothetical model outputs. "
            "Actual outcomes depend on factors beyond the model.",
            category=LimitationCategory.MODEL_UNCERTAINTY.value,
        ))
        if disclaimer:
            lims.append(create_limitation(
                disclaimer,
                category=LimitationCategory.METHODOLOGY.value,
            ))

        return ImpactAssessment(
            signal_id=signal.id,
            dataset_id=self._dataset_id,
            impact_type=ImpactType.SCENARIO_BASED.value,
            impact_category=ImpactCategory.PERFORMANCE.value,
            estimation_method="What-if scenario simulation",
            estimated_value=metric_value,
            unit="projected change (%)" if metric_value is not None else "",
            description=(
                f"Scenario projects {metric_value:.2f}% change in target"
                if metric_value is not None
                else "Scenario simulation completed"
            ),
            explanation=(
                "Impact derived from a what-if scenario simulation. "
                "The projected change is a model estimate based on "
                "hypothetical input modifications, not an observed outcome."
            ),
            evidence_ids=[_get_attr(e, "id", "") for e in evidence_items],
            confidence=confidence,
            limitations=lims,
            assumptions=[
                "Model relationships hold under the simulated conditions.",
                "No external factors intervene during the projection period.",
            ],
        )

    # -------------------------------------------------------------------
    # Optimization impact
    # -------------------------------------------------------------------

    def _assess_optimization_impact(
        self, signal: DecisionSignal, evidence_items: list,
    ) -> ImpactAssessment:
        """Assess impact from budget/resource optimization (SCENARIO_BASED)."""
        ev = evidence_items[0]
        metric_value = _safe_float(_get_attr(ev, "metric_value", None))

        confidence = assess_confidence(evidence_items, dataset_id=self._dataset_id)

        lims = list(signal.limitations)
        lims.append(create_limitation(
            "Optimization results are model-based recommendations. "
            "Actual improvement depends on market conditions and execution.",
            category=LimitationCategory.MODEL_UNCERTAINTY.value,
        ))

        return ImpactAssessment(
            signal_id=signal.id,
            dataset_id=self._dataset_id,
            impact_type=ImpactType.SCENARIO_BASED.value,
            impact_category=ImpactCategory.EFFICIENCY.value,
            estimation_method="Mathematical optimization on model surface",
            estimated_value=metric_value,
            unit="projected improvement (%)" if metric_value is not None else "",
            description=(
                f"Optimization suggests {metric_value:.2f}% potential improvement"
                if metric_value is not None
                else "Optimization completed but improvement not quantifiable"
            ),
            explanation=(
                "Impact estimated from an optimization algorithm applied to "
                "the model's prediction surface. The result is a theoretical "
                "optimum, not a guaranteed outcome."
            ),
            evidence_ids=[_get_attr(e, "id", "") for e in evidence_items],
            confidence=confidence,
            limitations=lims,
            assumptions=[
                "Model predictions accurately represent the response surface.",
                "Constraints and market conditions remain as modeled.",
            ],
        )

    # -------------------------------------------------------------------
    # MMM impact
    # -------------------------------------------------------------------

    def _assess_mmm_impact(
        self, signal: DecisionSignal, evidence_items: list,
    ) -> ImpactAssessment:
        """Assess impact from a Marketing Mix Model (MODEL_ESTIMATED)."""
        ev = evidence_items[0]
        details = _get_attr(ev, "details", {})
        test_metrics = details.get("test_metrics", {})
        r2 = _safe_float(
            test_metrics.get("r2") or test_metrics.get("R2")
        )
        metric_value = _safe_float(_get_attr(ev, "metric_value", None))

        confidence = assess_confidence(evidence_items, dataset_id=self._dataset_id)

        lims = list(signal.limitations)
        lims.append(create_limitation(
            "MMM results are model estimates based on observational data. "
            "Channel effects are associations, not proven causal impacts.",
            category=LimitationCategory.OBSERVATIONAL.value,
        ))

        return ImpactAssessment(
            signal_id=signal.id,
            dataset_id=self._dataset_id,
            impact_type=ImpactType.MODEL_ESTIMATED.value,
            impact_category=ImpactCategory.PERFORMANCE.value,
            estimation_method="Marketing Mix Model",
            estimated_value=r2 if r2 is not None else metric_value,
            unit="R² (model fit)" if r2 is not None else "",
            description=(
                f"MMM explains {r2:.1%} of target variance"
                if r2 is not None
                else "Marketing Mix Model fitted"
            ),
            explanation=(
                "Impact assessed via the Marketing Mix Model's explanatory "
                "power. Channel effects are model-estimated associations "
                "from observational data, not controlled experiments."
            ),
            evidence_ids=[_get_attr(e, "id", "") for e in evidence_items],
            confidence=confidence,
            limitations=lims,
        )

    # -------------------------------------------------------------------
    # Contribution / ROI impact
    # -------------------------------------------------------------------

    def _assess_contribution_impact(
        self, signal: DecisionSignal, evidence_items: list,
    ) -> ImpactAssessment:
        """Assess impact from channel contribution or ROI estimates."""
        ev = evidence_items[0]
        metric_value = _safe_float(_get_attr(ev, "metric_value", None))
        metric_name = _get_attr(ev, "metric_name", "")
        details = _get_attr(ev, "details", {})

        confidence = assess_confidence(evidence_items, dataset_id=self._dataset_id)

        lims = list(signal.limitations)
        lims.append(create_limitation(
            "Contribution and ROI estimates are model-derived and depend "
            "on model assumptions. They reflect association, not causation.",
            category=LimitationCategory.OBSERVATIONAL.value,
        ))

        unit = ""
        if "roi" in metric_name.lower():
            unit = "ROI ratio"
        elif "share" in metric_name.lower():
            unit = "proportion"

        return ImpactAssessment(
            signal_id=signal.id,
            dataset_id=self._dataset_id,
            impact_type=ImpactType.MODEL_ESTIMATED.value,
            impact_category=ImpactCategory.PERFORMANCE.value,
            estimation_method="Model-based contribution/ROI estimation",
            estimated_value=metric_value,
            unit=unit,
            description=(
                f"Model-estimated {metric_name}: {metric_value}"
                if metric_value is not None
                else f"Model-estimated {metric_name or 'contribution'}"
            ),
            explanation=(
                "Impact estimated from model-based contribution analysis. "
                "These values are derived from the model's decomposition "
                "of effects, not from a controlled experiment."
            ),
            evidence_ids=[_get_attr(e, "id", "") for e in evidence_items],
            confidence=confidence,
            limitations=lims,
        )

    # -------------------------------------------------------------------
    # Generic / fallback impact
    # -------------------------------------------------------------------

    def _assess_generic_impact(
        self, signal: DecisionSignal, evidence_items: list,
    ) -> ImpactAssessment:
        """Assess impact when source type is unknown or generic."""
        ev = evidence_items[0]
        metric_value = _safe_float(_get_attr(ev, "metric_value", None))
        ev_level = _get_attr(ev, "level", "")

        confidence = assess_confidence(evidence_items, dataset_id=self._dataset_id)

        # Determine type from evidence level
        if ev_level == EvidenceLevel.FACT.value:
            impact_type = ImpactType.MEASURED.value
        elif ev_level == EvidenceLevel.PREDICTION.value:
            impact_type = ImpactType.SCENARIO_BASED.value
        elif ev_level == EvidenceLevel.INFERENCE.value:
            impact_type = ImpactType.MODEL_ESTIMATED.value
        else:
            impact_type = ImpactType.QUALITATIVE.value

        return ImpactAssessment(
            signal_id=signal.id,
            dataset_id=self._dataset_id,
            impact_type=impact_type,
            impact_category=ImpactCategory.GENERAL.value,
            estimation_method="Evidence-level classification",
            estimated_value=metric_value,
            description=(
                f"Evidence-based finding (metric value: {metric_value})"
                if metric_value is not None
                else f"Evidence-based finding: {signal.description[:100]}"
            ),
            explanation=(
                "Impact assessed from available evidence. "
                "The impact type was determined by the evidence level."
            ),
            evidence_ids=[_get_attr(e, "id", "") for e in evidence_items],
            confidence=confidence,
            limitations=list(signal.limitations),
        )

    # -------------------------------------------------------------------
    # Unknown / insufficient impact
    # -------------------------------------------------------------------

    def _make_unknown_assessment(
        self,
        signal: DecisionSignal,
        reason: str,
    ) -> ImpactAssessment:
        """Create an explicit UNKNOWN impact assessment."""
        return ImpactAssessment(
            signal_id=signal.id,
            dataset_id=self._dataset_id,
            impact_type=ImpactType.UNKNOWN.value,
            impact_category=ImpactCategory.GENERAL.value,
            estimation_method="Insufficient evidence",
            description=f"Impact cannot be assessed: {reason}",
            explanation=reason,
            confidence=ConfidenceScore(
                level="insufficient",
                explanation=reason,
                evidence_count=0,
                evidence_quality="none",
                dataset_id=self._dataset_id,
            ),
            limitations=[
                create_limitation(
                    reason,
                    category=LimitationCategory.DATA_QUALITY.value,
                ),
            ],
        )

    # -------------------------------------------------------------------
    # Build report
    # -------------------------------------------------------------------

    def _build_report(self) -> ImpactReport:
        return ImpactReport(
            dataset_id=self._dataset_id,
            assessments=list(self._assessments),
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


def _get_attr_nested(obj, attr1: str, attr2: str, default=None):
    """Get a nested attribute: obj.attr1.attr2 or obj[attr1][attr2]."""
    intermediate = _get_attr(obj, attr1, {})
    if isinstance(intermediate, dict):
        return intermediate.get(attr2, default)
    return getattr(intermediate, attr2, default)
