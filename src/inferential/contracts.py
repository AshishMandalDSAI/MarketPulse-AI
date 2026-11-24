"""Typed, serializable result structures for inferential tests (Phase 8 Part 3)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Optional

import pandas as pd

# How much an assumption problem matters. Ordered from harmless to blocking.
CONSEQUENCE_ORDER = ("none", "usable_with_caution", "unreliable", "invalid")
CONSEQUENCE_TEXT = {
    "none": "no problem found",
    "usable_with_caution": "still usable, with caution",
    "unreliable": "unreliable - do not base a decision on it",
    "invalid": "invalid - the test cannot be run",
}


def _clean(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    return value


@dataclass
class MissingInfo:
    """Listwise-deletion bookkeeping for the columns a test uses."""
    original_n: int = 0          # rows in the dataset (for the selected columns)
    removed_n: int = 0           # rows dropped because a selected column was missing/blank
    final_n: int = 0             # rows used
    missing_pct: float = 0.0     # removed / original * 100
    per_variable: dict = field(default_factory=dict)   # column -> missing count
    not_used_n: int = 0          # complete rows that were not used (e.g. other groups, tiny groups)
    note: str = ""


@dataclass
class AssumptionCheck:
    name: str
    status: str                  # met | violated | caution | not_checked
    detail: str
    why_it_matters: str = ""
    consequence: str = "none"    # one of CONSEQUENCE_ORDER
    recommendation: str = ""


@dataclass
class ConfidenceInterval:
    level: float
    lower: Optional[float] = None
    upper: Optional[float] = None
    parameter: str = ""
    reason: str = ""             # why the interval is undefined (only when lower/upper are None)

    @property
    def defined(self) -> bool:
        return self.lower is not None and self.upper is not None


@dataclass
class EffectSizeResult:
    name: str
    value: Optional[float] = None
    label: str = ""              # negligible / small / medium / large
    practical: str = ""
    limitations: str = ""
    extra: dict = field(default_factory=dict)   # e.g. Hedges' g, omega-squared
    reason: str = ""             # why undefined


@dataclass
class TestResult:
    __test__ = False             # not a pytest class

    test_key: str
    test_name: str
    status: str = "ok"           # ok | refused
    refusal_reason: str = ""
    recommendation: str = ""
    h0: str = ""
    h1: str = ""
    alternative: str = "two-sided"
    statistic_name: str = ""
    statistic: Optional[float] = None
    p_value: Optional[float] = None
    df: Optional[float] = None
    df2: Optional[float] = None
    df_is_approximate: bool = False   # True when ``df`` belongs to an approximating distribution, not an exact property
    df_note: str = ""                 # plain-English basis of ``df`` when it is approximate
    alpha: float = 0.05
    significant: Optional[bool] = None
    ci: Optional[ConfidenceInterval] = None
    effect_size: Optional[EffectSizeResult] = None
    assumptions: list = field(default_factory=list)
    validity: str = "not_run"    # valid | usable_with_caution | unreliable | invalid | not_run
    # Independence of observations is an ASSUMPTION of the test, never a finding of it:
    #   not_verifiable           - the user did not flag time order; independence still cannot be verified from data
    #   time_ordered_unverified  - the user flagged time order; independence is NOT established (no serial dependence screened/detected)
    #   serial_dependence_detected - flagged time order AND a lag-1 screen found serial dependence
    independence_status: str = "not_verifiable"
    sample_sizes: dict = field(default_factory=dict)
    missing: Optional[MissingInfo] = None
    inputs: dict = field(default_factory=dict)
    interpretation: dict = field(default_factory=dict)
    undefined: dict = field(default_factory=dict)          # field -> reason
    warnings: list = field(default_factory=list)
    details: dict = field(default_factory=dict)            # name -> DataFrame
    plot_data: dict = field(default_factory=dict)          # arrays for the page's chart (not exported)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> dict:
        out = {}
        for f in fields(self):
            if f.name == "plot_data":
                continue
            v = getattr(self, f.name)
            if f.name == "details":
                out[f.name] = {k: _clean(d) for k, d in v.items()}
            elif hasattr(v, "__dataclass_fields__"):
                out[f.name] = _clean(asdict(v))
            elif isinstance(v, list):
                out[f.name] = [_clean(asdict(i)) if hasattr(i, "__dataclass_fields__") else _clean(i) for i in v]
            else:
                out[f.name] = _clean(v)
        return out
