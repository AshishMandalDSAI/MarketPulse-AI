"""Phase 7 Part 1 — Decision Intelligence contracts.

Domain-agnostic, immutable data contracts for evidence classification,
decision signals, confidence assessment, and limitation tracking.

Conventions (matching Phase 3-6):
- @dataclass with hand-written to_dict() / from_dict()
- str+Enum for categorical fields
- uuid4 hex[:8] IDs with type-specific prefixes
- _safe_float() for NaN/Inf sanitisation
- No Pydantic, no NamedTuple, no external dependencies
"""

from __future__ import annotations

import enum
import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_id(prefix: str = "ev_") -> str:
    """Generate a short unique ID with the given prefix."""
    return f"{prefix}{uuid.uuid4().hex[:8]}"


def _safe_float(value: Any) -> Optional[float]:
    """Convert to float, replacing NaN/Inf with None."""
    if value is None:
        return None
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _safe_str(value: Any) -> str:
    """Convert to string safely."""
    if value is None:
        return ""
    return str(value)


# ---------------------------------------------------------------------------
# Evidence Level Enum
# ---------------------------------------------------------------------------

class EvidenceLevel(str, enum.Enum):
    """Epistemic status of a piece of evidence.

    Ordered from strongest (directly observed) to weakest (known limitation).
    """
    FACT = "fact"                        # Directly computed from data
    INFERENCE = "inference"              # Statistical relationship found
    PREDICTION = "prediction"            # Model-generated projection
    RECOMMENDATION = "recommendation"    # Suggested action
    LIMITATION = "limitation"            # Known constraint on the analysis


# ---------------------------------------------------------------------------
# Confidence Level Enum
# ---------------------------------------------------------------------------

class ConfidenceLevel(str, enum.Enum):
    """Qualitative confidence assessment.

    This is a transparent, deterministic label — NOT a statistically
    calibrated probability.
    """
    HIGH = "high"                # Strong evidence, multiple corroborating signals
    MEDIUM = "medium"            # Moderate evidence, some uncertainty
    LOW = "low"                  # Weak evidence, significant uncertainty
    INSUFFICIENT = "insufficient"  # Not enough evidence to assess


# ---------------------------------------------------------------------------
# Limitation Category
# ---------------------------------------------------------------------------

class LimitationCategory(str, enum.Enum):
    """Categories of analytical limitations."""
    DATA_QUALITY = "data_quality"              # Missing data, outliers, noise
    SAMPLE_SIZE = "sample_size"                # Too few observations
    OBSERVATIONAL = "observational"            # No causal design (default for most data)
    MODEL_UNCERTAINTY = "model_uncertainty"     # Model fit / prediction interval
    TEMPORAL = "temporal"                      # Time-dependent assumptions
    SCOPE = "scope"                            # Analysis scope limitation
    METHODOLOGY = "methodology"                # Methodological caveat


# ---------------------------------------------------------------------------
# DecisionLimitation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DecisionLimitation:
    """A single, explicit limitation on an analysis or recommendation.

    Immutable to prevent downstream mutation.
    """
    id: str = field(default_factory=lambda: _make_id("lim_"))
    category: str = LimitationCategory.OBSERVATIONAL.value
    description: str = ""
    impact: str = ""              # Potential impact on interpretation
    recommendation: str = ""      # Suggested follow-up to address limitation

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "category": self.category,
            "description": self.description,
            "impact": self.impact,
            "recommendation": self.recommendation,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DecisionLimitation":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# DecisionEvidence
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DecisionEvidence:
    """A single piece of classified evidence from any analysis engine.

    Each evidence item carries its epistemic level, source, and scope.
    Immutable to prevent downstream mutation.
    """
    id: str = field(default_factory=lambda: _make_id("ev_"))
    level: str = EvidenceLevel.FACT.value
    description: str = ""
    source: str = ""              # Originating analysis engine / module
    metric_name: str = ""         # Optional: the metric or column involved
    metric_value: Any = None      # Optional: the observed value
    details: dict = field(default_factory=dict)    # Supporting details
    dataset_id: str = ""          # Dataset/session scope
    limitations: list = field(default_factory=list)  # list[DecisionLimitation]

    def __post_init__(self):
        """Validate fields."""
        if self.level not in {e.value for e in EvidenceLevel}:
            raise ValueError(
                f"Invalid evidence level: {self.level!r}. "
                f"Must be one of: {[e.value for e in EvidenceLevel]}"
            )
        if not self.description:
            raise ValueError("DecisionEvidence requires a non-empty description.")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "level": self.level,
            "description": self.description,
            "source": self.source,
            "metric_name": self.metric_name,
            "metric_value": _safe_float(self.metric_value)
                           if isinstance(self.metric_value, (int, float))
                           else self.metric_value,
            "details": self.details,
            "dataset_id": self.dataset_id,
            "limitations": [
                lim.to_dict() if hasattr(lim, "to_dict") else lim
                for lim in self.limitations
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DecisionEvidence":
        raw = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if "limitations" in raw and isinstance(raw["limitations"], list):
            raw["limitations"] = [
                DecisionLimitation.from_dict(lim) if isinstance(lim, dict) else lim
                for lim in raw["limitations"]
            ]
        return cls(**raw)


# ---------------------------------------------------------------------------
# DecisionSignal
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DecisionSignal:
    """A higher-level signal composed of one or more evidence items.

    Represents a synthesised finding that may span multiple analysis engines.
    """
    id: str = field(default_factory=lambda: _make_id("sig_"))
    description: str = ""
    evidence_ids: list = field(default_factory=list)    # References to DecisionEvidence.id
    level: str = EvidenceLevel.INFERENCE.value
    affected_features: list = field(default_factory=list)  # Column names / entities
    limitations: list = field(default_factory=list)         # list[DecisionLimitation]
    dataset_id: str = ""

    def __post_init__(self):
        if self.level not in {e.value for e in EvidenceLevel}:
            raise ValueError(
                f"Invalid evidence level: {self.level!r}. "
                f"Must be one of: {[e.value for e in EvidenceLevel]}"
            )
        if not self.description:
            raise ValueError("DecisionSignal requires a non-empty description.")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "evidence_ids": list(self.evidence_ids),
            "level": self.level,
            "affected_features": list(self.affected_features),
            "limitations": [
                lim.to_dict() if hasattr(lim, "to_dict") else lim
                for lim in self.limitations
            ],
            "dataset_id": self.dataset_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DecisionSignal":
        raw = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if "limitations" in raw and isinstance(raw["limitations"], list):
            raw["limitations"] = [
                DecisionLimitation.from_dict(lim) if isinstance(lim, dict) else lim
                for lim in raw["limitations"]
            ]
        return cls(**raw)


# ---------------------------------------------------------------------------
# ConfidenceScore
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConfidenceScore:
    """A transparent, deterministic confidence assessment.

    NOT a statistically calibrated probability. This is a qualitative label
    with a structured explanation of why confidence is at the given level.
    """
    level: str = ConfidenceLevel.INSUFFICIENT.value
    explanation: str = ""
    evidence_count: int = 0
    evidence_quality: str = ""       # Brief quality summary
    reduction_reasons: list = field(default_factory=list)  # list[str]
    dataset_id: str = ""

    def __post_init__(self):
        if self.level not in {e.value for e in ConfidenceLevel}:
            raise ValueError(
                f"Invalid confidence level: {self.level!r}. "
                f"Must be one of: {[e.value for e in ConfidenceLevel]}"
            )

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "explanation": self.explanation,
            "evidence_count": self.evidence_count,
            "evidence_quality": self.evidence_quality,
            "reduction_reasons": list(self.reduction_reasons),
            "dataset_id": self.dataset_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ConfidenceScore":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
