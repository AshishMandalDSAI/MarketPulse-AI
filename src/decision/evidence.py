"""Phase 7 Part 1 — Evidence classification helpers.

Deterministic functions that classify analytical findings by their
epistemic status: FACT, INFERENCE, PREDICTION, RECOMMENDATION, LIMITATION.

Safety rules enforced:
1. Correlation does not establish causation.
2. Association does not prove a cause.
3. A feature's importance is not automatically a causal effect.
4. A prediction is not an observed fact.
5. A recommendation is not a guaranteed outcome.
6. A scenario simulation is not the actual future.
7. A forecast is not certain.
8. Missing evidence must not be presented as evidence of absence.
9. Insufficient data → explicit INSUFFICIENT, not invented confidence.
10. Evidence from one dataset must not be silently attributed to another.

No LLM calls, no brittle keyword replacements. Purely deterministic.
"""

from __future__ import annotations

from typing import Any, Optional

from src.decision.contracts import (
    DecisionEvidence,
    DecisionLimitation,
    DecisionSignal,
    EvidenceLevel,
    LimitationCategory,
    _make_id,
)


# ---------------------------------------------------------------------------
# Evidence Classification Rules
# ---------------------------------------------------------------------------

# Source keywords that indicate each evidence level.
# These are matched against the `source` field — NOT against freetext.
_FACT_SOURCES = frozenset({
    "kpi_engine", "data_quality", "profiler", "row_count",
    "date_range", "summary", "distribution", "aggregation",
})

_INFERENCE_SOURCES = frozenset({
    "insight_engine", "relationship_engine", "driver_engine",
    "correlation", "anomaly_detector", "field_discovery",
    "contribution", "roi", "mmm_engine",
})

_PREDICTION_SOURCES = frozenset({
    "forecasting_engine", "regression_engine", "classification_engine",
    "segmentation_engine", "scenario_engine", "simulator",
    "optimizer", "forecast", "prediction",
})


def classify_evidence_level(source: str, details: Optional[dict] = None) -> str:
    """Classify a finding's evidence level from its source identifier.

    Returns the EvidenceLevel value string.

    The classification is deterministic:
    - Direct aggregations / data-quality metrics → FACT
    - Statistical relationships, drivers, anomalies → INFERENCE
    - Model projections, forecasts, scenarios → PREDICTION
    - Explicit limitations → LIMITATION
    - Otherwise → INFERENCE (conservative default)
    """
    source_lower = source.lower().strip()

    # Check explicit limitation first
    if "limitation" in source_lower:
        return EvidenceLevel.LIMITATION.value

    # Check explicit recommendation
    if "recommendation" in source_lower:
        return EvidenceLevel.RECOMMENDATION.value

    # Check prediction sources
    for kw in _PREDICTION_SOURCES:
        if kw in source_lower:
            return EvidenceLevel.PREDICTION.value

    # Check fact sources
    for kw in _FACT_SOURCES:
        if kw in source_lower:
            return EvidenceLevel.FACT.value

    # Check inference sources
    for kw in _INFERENCE_SOURCES:
        if kw in source_lower:
            return EvidenceLevel.INFERENCE.value

    # Conservative default: treat unknowns as inferences, not facts
    return EvidenceLevel.INFERENCE.value


# ---------------------------------------------------------------------------
# Evidence Creation Helpers
# ---------------------------------------------------------------------------

def create_evidence(
    description: str,
    source: str,
    *,
    level: Optional[str] = None,
    metric_name: str = "",
    metric_value: Any = None,
    details: Optional[dict] = None,
    dataset_id: str = "",
    limitations: Optional[list] = None,
) -> DecisionEvidence:
    """Create a DecisionEvidence with auto-classified level.

    If `level` is provided, it overrides the auto-classification.
    If `description` is empty, raises ValueError.
    """
    if not description or not description.strip():
        raise ValueError("Evidence description must not be empty.")

    resolved_level = level if level else classify_evidence_level(source, details)

    return DecisionEvidence(
        id=_make_id("ev_"),
        level=resolved_level,
        description=description.strip(),
        source=source,
        metric_name=metric_name,
        metric_value=metric_value,
        details=details or {},
        dataset_id=dataset_id,
        limitations=limitations or [],
    )


def create_limitation(
    description: str,
    *,
    category: str = LimitationCategory.OBSERVATIONAL.value,
    impact: str = "",
    recommendation: str = "",
) -> DecisionLimitation:
    """Create a DecisionLimitation with validation."""
    if not description or not description.strip():
        raise ValueError("Limitation description must not be empty.")

    return DecisionLimitation(
        id=_make_id("lim_"),
        category=category,
        description=description.strip(),
        impact=impact,
        recommendation=recommendation,
    )


# ---------------------------------------------------------------------------
# Safety: Correlation / Causation Guard
# ---------------------------------------------------------------------------

_OBSERVATIONAL_LIMITATION = DecisionLimitation(
    id=_make_id("lim_"),
    category=LimitationCategory.OBSERVATIONAL.value,
    description=(
        "This finding is based on observational data. "
        "Correlation or association does not establish causation."
    ),
    impact="The observed relationship may not be causal.",
    recommendation=(
        "A controlled experiment or quasi-experimental design would be "
        "needed to establish a causal relationship."
    ),
)


def ensure_non_causal(evidence: DecisionEvidence) -> DecisionEvidence:
    """Return evidence with an observational-data limitation attached.

    Used for any INFERENCE-level evidence derived from observational data
    (correlations, driver analysis, feature importance, model coefficients).
    Idempotent: does not duplicate the limitation if already present.
    """
    if evidence.level != EvidenceLevel.INFERENCE.value:
        return evidence

    # Check if already has an observational limitation
    for lim in evidence.limitations:
        cat = lim.category if hasattr(lim, "category") else lim.get("category", "")
        if cat == LimitationCategory.OBSERVATIONAL.value:
            return evidence

    # Frozen dataclass — rebuild with the new limitation appended
    new_limitations = list(evidence.limitations) + [_OBSERVATIONAL_LIMITATION]
    return DecisionEvidence(
        id=evidence.id,
        level=evidence.level,
        description=evidence.description,
        source=evidence.source,
        metric_name=evidence.metric_name,
        metric_value=evidence.metric_value,
        details=evidence.details,
        dataset_id=evidence.dataset_id,
        limitations=new_limitations,
    )


def ensure_prediction_uncertainty(evidence: DecisionEvidence) -> DecisionEvidence:
    """Return prediction-level evidence with an explicit uncertainty limitation.

    Ensures predictions are never presented as certain outcomes.
    Idempotent.
    """
    if evidence.level != EvidenceLevel.PREDICTION.value:
        return evidence

    for lim in evidence.limitations:
        cat = lim.category if hasattr(lim, "category") else lim.get("category", "")
        if cat == LimitationCategory.MODEL_UNCERTAINTY.value:
            return evidence

    uncertainty_lim = DecisionLimitation(
        id=_make_id("lim_"),
        category=LimitationCategory.MODEL_UNCERTAINTY.value,
        description=(
            "This is a model-generated projection, not an observed fact. "
            "Actual outcomes may differ from the prediction."
        ),
        impact="The predicted value is subject to model uncertainty.",
        recommendation="Consider the prediction interval and model fit metrics.",
    )

    new_limitations = list(evidence.limitations) + [uncertainty_lim]
    return DecisionEvidence(
        id=evidence.id,
        level=evidence.level,
        description=evidence.description,
        source=evidence.source,
        metric_name=evidence.metric_name,
        metric_value=evidence.metric_value,
        details=evidence.details,
        dataset_id=evidence.dataset_id,
        limitations=new_limitations,
    )


# ---------------------------------------------------------------------------
# Signal Construction
# ---------------------------------------------------------------------------

def create_signal(
    description: str,
    evidence_items: list,
    *,
    level: Optional[str] = None,
    affected_features: Optional[list] = None,
    limitations: Optional[list] = None,
    dataset_id: str = "",
) -> DecisionSignal:
    """Create a DecisionSignal from a list of evidence items.

    The signal's level defaults to the weakest (highest ordinal) evidence
    level among the supporting evidence — a chain is only as strong as
    its weakest link.

    Raises ValueError if:
    - description is empty
    - evidence_items is empty (missing evidence cannot be presented as
      evidence of absence)
    - evidence items span different dataset_ids (cross-dataset contamination)
    """
    if not description or not description.strip():
        raise ValueError("Signal description must not be empty.")

    if not evidence_items:
        raise ValueError(
            "A signal must be supported by at least one piece of evidence. "
            "Missing evidence must not be presented as evidence of absence."
        )

    # Enforce dataset isolation
    evidence_ids = []
    dataset_ids_seen = set()
    for ev in evidence_items:
        ev_id = ev.id if hasattr(ev, "id") else ev.get("id", "")
        ev_ds = ev.dataset_id if hasattr(ev, "dataset_id") else ev.get("dataset_id", "")
        evidence_ids.append(ev_id)
        if ev_ds:
            dataset_ids_seen.add(ev_ds)

    if dataset_id:
        dataset_ids_seen.add(dataset_id)

    if len(dataset_ids_seen) > 1:
        raise ValueError(
            f"Evidence from different datasets cannot be combined into a single signal. "
            f"Found dataset IDs: {sorted(dataset_ids_seen)}"
        )

    # Determine level: weakest evidence link
    if level is None:
        _level_order = {
            EvidenceLevel.FACT.value: 0,
            EvidenceLevel.INFERENCE.value: 1,
            EvidenceLevel.PREDICTION.value: 2,
            EvidenceLevel.RECOMMENDATION.value: 3,
            EvidenceLevel.LIMITATION.value: 4,
        }
        weakest = 0
        for ev in evidence_items:
            ev_level = ev.level if hasattr(ev, "level") else ev.get("level", "inference")
            weakest = max(weakest, _level_order.get(ev_level, 1))

        # Map back
        _order_to_level = {v: k for k, v in _level_order.items()}
        resolved_level = _order_to_level.get(weakest, EvidenceLevel.INFERENCE.value)
    else:
        resolved_level = level

    resolved_dataset_id = dataset_id or (dataset_ids_seen.pop() if dataset_ids_seen else "")

    return DecisionSignal(
        id=_make_id("sig_"),
        description=description.strip(),
        evidence_ids=evidence_ids,
        level=resolved_level,
        affected_features=affected_features or [],
        limitations=limitations or [],
        dataset_id=resolved_dataset_id,
    )


# ---------------------------------------------------------------------------
# Dataset Isolation Guard
# ---------------------------------------------------------------------------

def validate_dataset_scope(
    evidence_items: list,
    expected_dataset_id: str,
) -> list:
    """Filter evidence items to only those matching the expected dataset.

    Returns a list of evidence items whose dataset_id matches
    expected_dataset_id (or is empty, meaning unscoped).

    This prevents cross-dataset contamination — evidence from Dataset A
    must never be silently attributed to Dataset B.
    """
    if not expected_dataset_id:
        return list(evidence_items)

    result = []
    for ev in evidence_items:
        ev_ds = ev.dataset_id if hasattr(ev, "dataset_id") else ev.get("dataset_id", "")
        if not ev_ds or ev_ds == expected_dataset_id:
            result.append(ev)
    return result
