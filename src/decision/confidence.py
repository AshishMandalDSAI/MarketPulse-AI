"""Phase 7 Part 1 — Deterministic confidence assessment.

All confidence computation is deterministic and explainable. No LLM calls.
Confidence is a qualitative label with structured reduction reasons, NOT
a statistically calibrated probability.

The framework:
1. Assesses evidence sufficiency (count and quality).
2. Identifies specific reasons that reduce confidence.
3. Produces a transparent ConfidenceScore with human-readable explanation.
4. Never fabricates confidence from insufficient evidence.
"""

from __future__ import annotations

from typing import Optional

from src.decision.contracts import (
    ConfidenceLevel,
    ConfidenceScore,
    DecisionEvidence,
    DecisionLimitation,
    DecisionSignal,
    EvidenceLevel,
    LimitationCategory,
)


# ---------------------------------------------------------------------------
# Thresholds (deterministic, no magic numbers without documentation)
# ---------------------------------------------------------------------------

# Minimum evidence items to avoid INSUFFICIENT
MIN_EVIDENCE_FOR_ASSESSMENT = 1

# Thresholds for quality tiers
HIGH_EVIDENCE_THRESHOLD = 3      # >=3 items with at least one FACT
MEDIUM_EVIDENCE_THRESHOLD = 2    # >=2 items


# ---------------------------------------------------------------------------
# Core: Assess Confidence from Evidence
# ---------------------------------------------------------------------------

def assess_confidence(
    evidence_items: list,
    *,
    dataset_id: str = "",
) -> ConfidenceScore:
    """Assess confidence from a collection of evidence items.

    Deterministic rules:
    1. No evidence → INSUFFICIENT
    2. All evidence is LIMITATION → INSUFFICIENT
    3. >=3 items with at least one FACT and no critical limitations → HIGH
    4. >=2 items with at least one FACT → MEDIUM
    5. >=1 item, all PREDICTION or INFERENCE → LOW
    6. Otherwise → LOW

    Confidence is reduced by:
    - Having only prediction-level evidence (no observed facts)
    - Having critical limitations (data quality, sample size)
    - All evidence being from a single source
    - Having conflicting evidence (not implemented in Part 1)
    """
    if not evidence_items:
        return ConfidenceScore(
            level=ConfidenceLevel.INSUFFICIENT.value,
            explanation="No evidence available for assessment.",
            evidence_count=0,
            evidence_quality="none",
            reduction_reasons=["No evidence items provided."],
            dataset_id=dataset_id,
        )

    # Gather evidence stats
    levels = []
    sources = set()
    has_critical_limitation = False
    limitation_categories = set()

    for ev in evidence_items:
        ev_level = ev.level if hasattr(ev, "level") else ev.get("level", "")
        ev_source = ev.source if hasattr(ev, "source") else ev.get("source", "")
        ev_limitations = (
            ev.limitations if hasattr(ev, "limitations")
            else ev.get("limitations", [])
        )

        levels.append(ev_level)
        if ev_source:
            sources.add(ev_source)

        for lim in ev_limitations:
            cat = lim.category if hasattr(lim, "category") else lim.get("category", "")
            limitation_categories.add(cat)
            if cat in {
                LimitationCategory.DATA_QUALITY.value,
                LimitationCategory.SAMPLE_SIZE.value,
            }:
                has_critical_limitation = True

    n_evidence = len(evidence_items)
    n_facts = sum(1 for lv in levels if lv == EvidenceLevel.FACT.value)
    n_inferences = sum(1 for lv in levels if lv == EvidenceLevel.INFERENCE.value)
    n_predictions = sum(1 for lv in levels if lv == EvidenceLevel.PREDICTION.value)
    n_limitations = sum(1 for lv in levels if lv == EvidenceLevel.LIMITATION.value)
    n_recommendations = sum(1 for lv in levels if lv == EvidenceLevel.RECOMMENDATION.value)

    # All limitations → INSUFFICIENT
    if n_limitations == n_evidence:
        return ConfidenceScore(
            level=ConfidenceLevel.INSUFFICIENT.value,
            explanation=(
                "All provided evidence consists of limitations. "
                "Insufficient positive evidence for assessment."
            ),
            evidence_count=n_evidence,
            evidence_quality="limitations_only",
            reduction_reasons=["All evidence items are limitations, not positive findings."],
            dataset_id=dataset_id,
        )

    # Build reduction reasons
    reduction_reasons = []

    if n_facts == 0:
        reduction_reasons.append(
            "No directly observed facts in the evidence; "
            "all findings are inferences, predictions, or recommendations."
        )

    if has_critical_limitation:
        reduction_reasons.append(
            "Critical limitation detected: data quality or sample size concern."
        )

    if len(sources) <= 1 and n_evidence > 1:
        reduction_reasons.append(
            "All evidence comes from a single analysis source."
        )

    if n_predictions > 0 and n_facts == 0:
        reduction_reasons.append(
            "Evidence relies on model predictions without corroborating observed facts."
        )

    if LimitationCategory.OBSERVATIONAL.value in limitation_categories:
        reduction_reasons.append(
            "Observational data only — findings reflect correlation, not causation."
        )

    # Determine level
    positive_evidence = n_evidence - n_limitations

    if positive_evidence == 0:
        level = ConfidenceLevel.INSUFFICIENT.value
        quality = "none"
    elif (
        positive_evidence >= HIGH_EVIDENCE_THRESHOLD
        and n_facts >= 1
        and not has_critical_limitation
        and not reduction_reasons
    ):
        level = ConfidenceLevel.HIGH.value
        quality = "strong"
    elif (
        positive_evidence >= HIGH_EVIDENCE_THRESHOLD
        and n_facts >= 1
        and not has_critical_limitation
    ):
        # Has reduction reasons but still decent evidence
        level = ConfidenceLevel.MEDIUM.value
        quality = "moderate"
    elif positive_evidence >= MEDIUM_EVIDENCE_THRESHOLD and n_facts >= 1:
        level = ConfidenceLevel.MEDIUM.value
        quality = "moderate"
    elif positive_evidence >= MIN_EVIDENCE_FOR_ASSESSMENT:
        level = ConfidenceLevel.LOW.value
        quality = "limited"
    else:
        level = ConfidenceLevel.INSUFFICIENT.value
        quality = "none"

    # Downgrade for critical limitations
    if has_critical_limitation and level == ConfidenceLevel.HIGH.value:
        level = ConfidenceLevel.MEDIUM.value
    if has_critical_limitation and level == ConfidenceLevel.MEDIUM.value and n_facts == 0:
        level = ConfidenceLevel.LOW.value

    # Build explanation
    explanation = _build_explanation(
        level=level,
        n_evidence=n_evidence,
        n_facts=n_facts,
        n_inferences=n_inferences,
        n_predictions=n_predictions,
        n_limitations=n_limitations,
        reduction_reasons=reduction_reasons,
    )

    return ConfidenceScore(
        level=level,
        explanation=explanation,
        evidence_count=n_evidence,
        evidence_quality=quality,
        reduction_reasons=reduction_reasons,
        dataset_id=dataset_id,
    )


def _build_explanation(
    *,
    level: str,
    n_evidence: int,
    n_facts: int,
    n_inferences: int,
    n_predictions: int,
    n_limitations: int,
    reduction_reasons: list,
) -> str:
    """Build a human-readable confidence explanation."""
    parts = []

    parts.append(
        f"Confidence is {level.upper()} based on "
        f"{n_evidence} piece(s) of evidence"
    )

    breakdown = []
    if n_facts:
        breakdown.append(f"{n_facts} observed fact(s)")
    if n_inferences:
        breakdown.append(f"{n_inferences} inference(s)")
    if n_predictions:
        breakdown.append(f"{n_predictions} prediction(s)")
    if n_limitations:
        breakdown.append(f"{n_limitations} limitation(s)")

    if breakdown:
        parts.append(f" ({', '.join(breakdown)}).")
    else:
        parts.append(".")

    if reduction_reasons:
        parts.append(
            " Confidence was reduced because: "
            + "; ".join(reduction_reasons)
        )

    return "".join(parts)


# ---------------------------------------------------------------------------
# Assess Confidence for a Signal
# ---------------------------------------------------------------------------

def assess_signal_confidence(
    signal: DecisionSignal,
    evidence_lookup: dict,
    *,
    dataset_id: str = "",
) -> ConfidenceScore:
    """Assess confidence for a DecisionSignal by looking up its evidence.

    `evidence_lookup` maps evidence IDs to DecisionEvidence objects.
    Missing evidence IDs are treated as missing data (reduces confidence).
    """
    found_evidence = []
    missing_ids = []

    for ev_id in signal.evidence_ids:
        ev = evidence_lookup.get(ev_id)
        if ev is not None:
            found_evidence.append(ev)
        else:
            missing_ids.append(ev_id)

    # Start with base assessment
    score = assess_confidence(found_evidence, dataset_id=dataset_id or signal.dataset_id)

    # Add reduction if evidence was missing
    if missing_ids:
        new_reasons = list(score.reduction_reasons) + [
            f"Could not locate {len(missing_ids)} referenced evidence item(s)."
        ]
        # Downgrade if needed
        new_level = score.level
        if new_level == ConfidenceLevel.HIGH.value:
            new_level = ConfidenceLevel.MEDIUM.value
        elif new_level == ConfidenceLevel.MEDIUM.value and len(missing_ids) > 1:
            new_level = ConfidenceLevel.LOW.value

        score = ConfidenceScore(
            level=new_level,
            explanation=score.explanation + (
                f" Additionally, {len(missing_ids)} referenced evidence "
                f"item(s) could not be located."
            ),
            evidence_count=score.evidence_count,
            evidence_quality=score.evidence_quality,
            reduction_reasons=new_reasons,
            dataset_id=score.dataset_id,
        )

    # Signal-level limitations reduce confidence further
    if signal.limitations:
        has_signal_critical = any(
            (lim.category if hasattr(lim, "category") else lim.get("category", ""))
            in {LimitationCategory.DATA_QUALITY.value, LimitationCategory.SAMPLE_SIZE.value}
            for lim in signal.limitations
        )
        if has_signal_critical and score.level == ConfidenceLevel.HIGH.value:
            new_reasons = list(score.reduction_reasons) + [
                "Signal carries a critical limitation."
            ]
            score = ConfidenceScore(
                level=ConfidenceLevel.MEDIUM.value,
                explanation=score.explanation,
                evidence_count=score.evidence_count,
                evidence_quality=score.evidence_quality,
                reduction_reasons=new_reasons,
                dataset_id=score.dataset_id,
            )

    return score


# ---------------------------------------------------------------------------
# Aggregate Confidence (for future use by later parts)
# ---------------------------------------------------------------------------

def aggregate_confidence(scores: list) -> ConfidenceScore:
    """Aggregate multiple ConfidenceScores into a single overall score.

    Rule: the aggregate is the WEAKEST individual score, because
    a chain of reasoning is only as strong as its weakest link.
    Insufficient scores dominate.
    """
    if not scores:
        return ConfidenceScore(
            level=ConfidenceLevel.INSUFFICIENT.value,
            explanation="No confidence scores to aggregate.",
            evidence_count=0,
            evidence_quality="none",
            reduction_reasons=["No confidence scores provided."],
        )

    _level_order = {
        ConfidenceLevel.HIGH.value: 0,
        ConfidenceLevel.MEDIUM.value: 1,
        ConfidenceLevel.LOW.value: 2,
        ConfidenceLevel.INSUFFICIENT.value: 3,
    }

    weakest_idx = 0
    weakest_order = -1
    all_reasons = []
    total_evidence = 0

    for i, s in enumerate(scores):
        s_level = s.level if hasattr(s, "level") else s.get("level", "insufficient")
        order = _level_order.get(s_level, 3)
        if order > weakest_order:
            weakest_order = order
            weakest_idx = i

        s_reasons = (
            s.reduction_reasons if hasattr(s, "reduction_reasons")
            else s.get("reduction_reasons", [])
        )
        all_reasons.extend(s_reasons)

        s_count = (
            s.evidence_count if hasattr(s, "evidence_count")
            else s.get("evidence_count", 0)
        )
        total_evidence += s_count

    weakest = scores[weakest_idx]
    weakest_level = weakest.level if hasattr(weakest, "level") else weakest.get("level", "insufficient")

    # Deduplicate reasons
    unique_reasons = list(dict.fromkeys(all_reasons))

    return ConfidenceScore(
        level=weakest_level,
        explanation=(
            f"Aggregate confidence is {weakest_level.upper()} "
            f"(weakest of {len(scores)} assessment(s), "
            f"based on {total_evidence} total evidence item(s))."
        ),
        evidence_count=total_evidence,
        evidence_quality=weakest.evidence_quality if hasattr(weakest, "evidence_quality") else "mixed",
        reduction_reasons=unique_reasons,
    )
