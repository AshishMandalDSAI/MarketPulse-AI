"""Phase 6 — Marketing Intelligence Contracts.

Every marketing analysis module produces and consumes these contracts.
No contract references specific column names — all field names come
from the dataset's CanonicalDataset semantic roles at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np


# ---------------------------------------------------------------------------
# 1. Readiness Levels
# ---------------------------------------------------------------------------

class MarketingReadiness(str, Enum):
    """How much marketing intelligence the dataset supports."""
    NOT_READY = "NOT_READY"
    PARTIAL = "PARTIAL"                          # some spend cols but missing time/target
    READY_FOR_MARKETING_ANALYSIS = "READY_FOR_MARKETING_ANALYSIS"  # ROI/ROAS possible
    READY_FOR_MMM = "READY_FOR_MMM"              # full MMM feasible
    READY_FOR_OPTIMIZATION = "READY_FOR_OPTIMIZATION"  # MMM + optimizer feasible


class DataQualityState(str, Enum):
    VALID = "VALID"
    WARNING = "WARNING"
    BLOCKED = "BLOCKED"


# ---------------------------------------------------------------------------
# 2. Per-capability status
# ---------------------------------------------------------------------------

@dataclass
class CapabilityStatus:
    """Status of a single marketing capability."""
    available: bool = False
    confidence: float = 0.0
    reason: str = ""
    required_evidence: list = field(default_factory=list)
    detected_fields: list = field(default_factory=list)
    limitations: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "available": self.available,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
            "required_evidence": list(self.required_evidence),
            "detected_fields": list(self.detected_fields),
            "limitations": list(self.limitations),
        }

    @classmethod
    def from_dict(cls, d: dict) -> CapabilityStatus:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# 3. Marketing Capability Report
# ---------------------------------------------------------------------------

@dataclass
class MarketingCapabilityReport:
    """Per-capability assessment for marketing intelligence."""
    readiness: MarketingReadiness = MarketingReadiness.NOT_READY
    target_field: Optional[str] = None
    time_field: Optional[str] = None
    channel_fields: list = field(default_factory=list)
    control_fields: list = field(default_factory=list)
    conversion_fields: list = field(default_factory=list)

    mmm: CapabilityStatus = field(default_factory=CapabilityStatus)
    adstock: CapabilityStatus = field(default_factory=CapabilityStatus)
    channel_contribution: CapabilityStatus = field(default_factory=CapabilityStatus)
    roi: CapabilityStatus = field(default_factory=CapabilityStatus)
    budget_optimization: CapabilityStatus = field(default_factory=CapabilityStatus)
    what_if: CapabilityStatus = field(default_factory=CapabilityStatus)

    warnings: list = field(default_factory=list)
    unavailable_analyses: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "readiness": self.readiness.value,
            "target_field": self.target_field,
            "time_field": self.time_field,
            "channel_fields": list(self.channel_fields),
            "control_fields": list(self.control_fields),
            "conversion_fields": list(self.conversion_fields),
            "mmm": self.mmm.to_dict(),
            "adstock": self.adstock.to_dict(),
            "channel_contribution": self.channel_contribution.to_dict(),
            "roi": self.roi.to_dict(),
            "budget_optimization": self.budget_optimization.to_dict(),
            "what_if": self.what_if.to_dict(),
            "warnings": list(self.warnings),
            "unavailable_analyses": list(self.unavailable_analyses),
        }

    @classmethod
    def from_dict(cls, d: dict) -> MarketingCapabilityReport:
        report = cls()
        report.readiness = MarketingReadiness(d.get("readiness", "NOT_READY"))
        report.target_field = d.get("target_field")
        report.time_field = d.get("time_field")
        report.channel_fields = d.get("channel_fields", [])
        report.control_fields = d.get("control_fields", [])
        report.conversion_fields = d.get("conversion_fields", [])
        for key in ("mmm", "adstock", "channel_contribution", "roi",
                     "budget_optimization", "what_if"):
            if key in d and isinstance(d[key], dict):
                setattr(report, key, CapabilityStatus.from_dict(d[key]))
        report.warnings = d.get("warnings", [])
        report.unavailable_analyses = d.get("unavailable_analyses", [])
        return report


# ---------------------------------------------------------------------------
# 4. Channel Validation Result
# ---------------------------------------------------------------------------

@dataclass
class ChannelValidation:
    """Result of validating a single channel field."""
    column: str
    is_valid: bool = True
    reasons: list = field(default_factory=list)       # why it qualified
    warnings: list = field(default_factory=list)       # limitations
    rejection_reasons: list = field(default_factory=list)  # why it failed

    def to_dict(self) -> dict:
        return {
            "column": self.column,
            "is_valid": self.is_valid,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
            "rejection_reasons": list(self.rejection_reasons),
        }


# ---------------------------------------------------------------------------
# 5. MMM Result
# ---------------------------------------------------------------------------

@dataclass
class MMMResult:
    """Result of fitting a Marketing Mix Model."""
    selected_model: str = ""          # "baseline" | "adstock" | "adstock_saturation"
    model_type: str = "ridge_linear"
    channels: list = field(default_factory=list)
    control_cols: list = field(default_factory=list)
    target_column: str = ""

    # Per-channel parameters (decay, shape, k, correlation)
    channel_params: dict = field(default_factory=dict)

    # Model diagnostics
    train_metrics: dict = field(default_factory=dict)
    test_metrics: dict = field(default_factory=dict)
    model_comparison: dict = field(default_factory=dict)

    # Train/test split info
    n_train: int = 0
    n_test: int = 0
    split_method: str = "time-aware (chronological), no shuffling"

    # Feature names in the fitted model
    feature_names: list = field(default_factory=list)

    # Fitted objects (not serialized)
    _model: Any = field(default=None, repr=False)
    _scaler: Any = field(default=None, repr=False)

    # Assumptions and limitations
    assumptions: list = field(default_factory=list)
    limitations: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "selected_model": self.selected_model,
            "model_type": self.model_type,
            "channels": list(self.channels),
            "control_cols": list(self.control_cols),
            "target_column": self.target_column,
            "channel_params": _safe_dict(self.channel_params),
            "train_metrics": _safe_dict(self.train_metrics),
            "test_metrics": _safe_dict(self.test_metrics),
            "model_comparison": _safe_dict(self.model_comparison),
            "n_train": self.n_train,
            "n_test": self.n_test,
            "split_method": self.split_method,
            "feature_names": list(self.feature_names),
            "assumptions": list(self.assumptions),
            "limitations": list(self.limitations),
        }

    @classmethod
    def from_dict(cls, d: dict) -> MMMResult:
        result = cls()
        for key in ("selected_model", "model_type", "target_column",
                     "split_method", "n_train", "n_test"):
            if key in d:
                setattr(result, key, d[key])
        for key in ("channels", "control_cols", "feature_names",
                     "assumptions", "limitations"):
            if key in d:
                setattr(result, key, list(d[key]))
        for key in ("channel_params", "train_metrics", "test_metrics",
                     "model_comparison"):
            if key in d:
                setattr(result, key, dict(d[key]))
        return result


# ---------------------------------------------------------------------------
# 6. Channel Contribution
# ---------------------------------------------------------------------------

@dataclass
class ChannelContribution:
    """Contribution of a single channel to the target."""
    channel: str
    spend: float = 0.0
    transformed_spend: float = 0.0
    contribution: float = 0.0
    contribution_share: float = 0.0
    roi: float = 0.0

    def to_dict(self) -> dict:
        return {
            "channel": self.channel,
            "spend": _safe_float(self.spend),
            "transformed_spend": _safe_float(self.transformed_spend),
            "contribution": _safe_float(self.contribution),
            "contribution_share": _safe_float(self.contribution_share),
            "roi": _safe_float(self.roi),
        }

    @classmethod
    def from_dict(cls, d: dict) -> ChannelContribution:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# 7. ROI Result
# ---------------------------------------------------------------------------

@dataclass
class ROIResult:
    """ROI/ROAS calculation for a single channel."""
    channel: str
    total_spend: float = 0.0
    total_target: float = 0.0
    naive_roas: Optional[float] = None
    model_estimated_roi: Optional[float] = None
    method: str = "naive_ratio"           # "naive_ratio" | "model_based"

    def to_dict(self) -> dict:
        return {
            "channel": self.channel,
            "total_spend": _safe_float(self.total_spend),
            "total_target": _safe_float(self.total_target),
            "naive_roas": _safe_float(self.naive_roas),
            "model_estimated_roi": _safe_float(self.model_estimated_roi),
            "method": self.method,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ROIResult:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# 8. Budget Optimization Result
# ---------------------------------------------------------------------------

@dataclass
class BudgetOptimizationResult:
    """Result of budget optimization."""
    success: bool = False
    message: str = ""
    total_budget: float = 0.0
    current_allocation: dict = field(default_factory=dict)
    optimized_allocation: dict = field(default_factory=dict)
    current_predicted_target: float = 0.0
    optimized_predicted_target: float = 0.0
    expected_improvement_pct: float = 0.0
    target_column: str = ""
    channels: list = field(default_factory=list)
    constraints: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "message": self.message,
            "total_budget": _safe_float(self.total_budget),
            "current_allocation": _safe_dict(self.current_allocation),
            "optimized_allocation": _safe_dict(self.optimized_allocation),
            "current_predicted_target": _safe_float(self.current_predicted_target),
            "optimized_predicted_target": _safe_float(self.optimized_predicted_target),
            "expected_improvement_pct": _safe_float(self.expected_improvement_pct),
            "target_column": self.target_column,
            "channels": list(self.channels),
            "constraints": _safe_dict(self.constraints),
        }

    @classmethod
    def from_dict(cls, d: dict) -> BudgetOptimizationResult:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# 9. What-If Result
# ---------------------------------------------------------------------------

@dataclass
class WhatIfScenario:
    """A single what-if scenario."""
    name: str = ""
    allocation: dict = field(default_factory=dict)
    total_budget: float = 0.0
    predicted_target: float = 0.0
    target_change: float = 0.0
    target_change_pct: float = 0.0
    roi: float = 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "allocation": _safe_dict(self.allocation),
            "total_budget": _safe_float(self.total_budget),
            "predicted_target": _safe_float(self.predicted_target),
            "target_change": _safe_float(self.target_change),
            "target_change_pct": _safe_float(self.target_change_pct),
            "roi": _safe_float(self.roi),
        }

    @classmethod
    def from_dict(cls, d: dict) -> WhatIfScenario:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class WhatIfResult:
    """Result of what-if simulation."""
    scenarios: list = field(default_factory=list)   # list[WhatIfScenario]
    baseline_scenario: Optional[WhatIfScenario] = None
    target_column: str = ""
    channels: list = field(default_factory=list)
    disclaimer: str = ("Model-estimated outcomes based on historical patterns. "
                       "Actual results may differ. These are not causal guarantees.")

    def to_dict(self) -> dict:
        return {
            "scenarios": [s.to_dict() for s in self.scenarios],
            "baseline_scenario": self.baseline_scenario.to_dict() if self.baseline_scenario else None,
            "target_column": self.target_column,
            "channels": list(self.channels),
            "disclaimer": self.disclaimer,
        }

    @classmethod
    def from_dict(cls, d: dict) -> WhatIfResult:
        result = cls()
        result.target_column = d.get("target_column", "")
        result.channels = d.get("channels", [])
        result.disclaimer = d.get("disclaimer", result.disclaimer)
        if d.get("baseline_scenario"):
            result.baseline_scenario = WhatIfScenario.from_dict(d["baseline_scenario"])
        result.scenarios = [WhatIfScenario.from_dict(s) for s in d.get("scenarios", [])]
        return result


# ---------------------------------------------------------------------------
# 10. Full Phase 6 Result
# ---------------------------------------------------------------------------

@dataclass
class Phase6Result:
    """Aggregated Phase 6 result for a single dataset."""
    dataset_id: str
    marketing_report: Optional[MarketingCapabilityReport] = None
    mmm_result: Optional[MMMResult] = None
    contributions: list = field(default_factory=list)     # list[ChannelContribution]
    roi_results: list = field(default_factory=list)        # list[ROIResult]
    optimization_result: Optional[BudgetOptimizationResult] = None
    whatif_result: Optional[WhatIfResult] = None
    unavailable_analyses: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "marketing_report": self.marketing_report.to_dict() if self.marketing_report else None,
            "mmm_result": self.mmm_result.to_dict() if self.mmm_result else None,
            "contributions": [c.to_dict() for c in self.contributions],
            "roi_results": [r.to_dict() for r in self.roi_results],
            "optimization_result": self.optimization_result.to_dict() if self.optimization_result else None,
            "whatif_result": self.whatif_result.to_dict() if self.whatif_result else None,
            "unavailable_analyses": list(self.unavailable_analyses),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Phase6Result:
        result = cls(dataset_id=d.get("dataset_id", ""))
        if d.get("marketing_report"):
            result.marketing_report = MarketingCapabilityReport.from_dict(d["marketing_report"])
        if d.get("mmm_result"):
            result.mmm_result = MMMResult.from_dict(d["mmm_result"])
        result.contributions = [ChannelContribution.from_dict(c) for c in d.get("contributions", [])]
        result.roi_results = [ROIResult.from_dict(r) for r in d.get("roi_results", [])]
        if d.get("optimization_result"):
            result.optimization_result = BudgetOptimizationResult.from_dict(d["optimization_result"])
        if d.get("whatif_result"):
            result.whatif_result = WhatIfResult.from_dict(d["whatif_result"])
        result.unavailable_analyses = d.get("unavailable_analyses", [])
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _safe_dict(d: dict) -> dict:
    if not d:
        return {}
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[str(k)] = _safe_dict(v)
        elif isinstance(v, (list, tuple)):
            out[str(k)] = [_safe_float(x) if isinstance(x, (int, float, np.integer, np.floating)) else x for x in v]
        elif isinstance(v, (int, float, np.integer, np.floating)):
            out[str(k)] = _safe_float(v)
        else:
            out[str(k)] = v
    return out
