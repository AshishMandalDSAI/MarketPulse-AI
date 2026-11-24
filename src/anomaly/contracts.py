"""Phase 5 — Anomaly Detection & Driver Analysis Contracts.

Standardized data contracts for anomaly detection results,
driver/root-cause analysis, and the aggregate Phase 5 result.

Every object is JSON-serializable via its to_dict() method.
No fabricated values. All None rather than invented defaults.
"""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


def _make_id(prefix: str = "anom_") -> str:
    return f"{prefix}{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AnomalySeverity(str, enum.Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AnomalyType(str, enum.Enum):
    NUMERIC_OUTLIER = "numeric_outlier"
    TIME_SERIES_ANOMALY = "time_series_anomaly"
    SUDDEN_CHANGE = "sudden_change"
    DISTRIBUTION_SHIFT = "distribution_shift"
    CATEGORY_ANOMALY = "category_anomaly"
    MISSINGNESS_ANOMALY = "missingness_anomaly"


# ---------------------------------------------------------------------------
# Anomaly
# ---------------------------------------------------------------------------

@dataclass
class Anomaly:
    """A single detected anomaly in the dataset."""

    id: str = field(default_factory=lambda: _make_id("anom_"))
    dataset_id: str = ""
    column: str = ""
    row_index: Optional[int] = None
    anomaly_type: str = ""          # from AnomalyType
    severity: str = "LOW"           # from AnomalySeverity
    score: float = 0.0              # 0-1 normalized anomaly score
    method: str = ""                # e.g. "iqr", "zscore", "mad", "rolling_deviation"
    observed_value: Optional[Any] = None
    expected_value: Optional[Any] = None
    deviation: Optional[float] = None
    explanation: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "dataset_id": self.dataset_id,
            "column": self.column,
            "row_index": self.row_index,
            "anomaly_type": self.anomaly_type,
            "severity": self.severity,
            "score": round(self.score, 4),
            "method": self.method,
            "observed_value": self.observed_value,
            "expected_value": self.expected_value,
            "deviation": round(self.deviation, 4) if self.deviation is not None else None,
            "explanation": self.explanation,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Anomaly:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Anomaly Report
# ---------------------------------------------------------------------------

@dataclass
class AnomalyReport:
    """Aggregated anomaly detection results for one dataset."""

    dataset_id: str = ""
    anomalies: list = field(default_factory=list)               # list[Anomaly]
    summary: dict = field(default_factory=dict)                  # counts by type, severity, affected columns
    unavailable_analyses: list = field(default_factory=list)     # list[dict] with analysis, reason
    metadata: dict = field(default_factory=dict)                 # sampling_used, total_rows_analyzed, etc.

    def by_column(self, col: str) -> list:
        """Return anomalies for a specific column."""
        return [a for a in self.anomalies if a.column == col]

    def by_type(self, anomaly_type: str) -> list:
        """Return anomalies of a specific type."""
        return [a for a in self.anomalies if a.anomaly_type == anomaly_type]

    def by_severity(self, severity: str) -> list:
        """Return anomalies at a specific severity level."""
        return [a for a in self.anomalies if a.severity == severity]

    def critical_anomalies(self) -> list:
        """Return all CRITICAL severity anomalies."""
        return self.by_severity(AnomalySeverity.CRITICAL.value)

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "anomalies": [a.to_dict() for a in self.anomalies],
            "summary": dict(self.summary),
            "unavailable_analyses": list(self.unavailable_analyses),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: dict) -> AnomalyReport:
        return cls(
            dataset_id=d.get("dataset_id", ""),
            anomalies=[Anomaly.from_dict(a) for a in d.get("anomalies", [])],
            summary=d.get("summary", {}),
            unavailable_analyses=d.get("unavailable_analyses", []),
            metadata=d.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Driver Result
# ---------------------------------------------------------------------------

@dataclass
class DriverResult:
    """A single driver/root-cause relationship detected for a target."""

    id: str = field(default_factory=lambda: _make_id("drv_"))
    feature: str = ""
    target: str = ""
    rank: int = 0
    relationship_type: str = ""     # correlation, mutual_information, feature_importance, group_comparison
    strength: float = 0.0
    direction: str = ""             # positive, negative, non_linear, varies
    importance: float = 0.0
    method: str = ""
    confidence_basis: str = ""
    explanation: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "feature": self.feature,
            "target": self.target,
            "rank": self.rank,
            "relationship_type": self.relationship_type,
            "strength": round(self.strength, 4),
            "direction": self.direction,
            "importance": round(self.importance, 4),
            "method": self.method,
            "confidence_basis": self.confidence_basis,
            "explanation": self.explanation,
        }

    @classmethod
    def from_dict(cls, d: dict) -> DriverResult:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Driver Report
# ---------------------------------------------------------------------------

@dataclass
class DriverReport:
    """Aggregated driver/root-cause analysis results for one dataset."""

    dataset_id: str = ""
    target_column: Optional[str] = None
    target_type: str = ""           # numeric, categorical, unavailable
    drivers: list = field(default_factory=list)                 # list[DriverResult]
    relationships: list = field(default_factory=list)           # reuse Phase 3 Relationship objects serialized
    unavailable_analyses: list = field(default_factory=list)    # list[dict]
    metadata: dict = field(default_factory=dict)

    def top_drivers(self, n: int = 5) -> list:
        """Return the top-n drivers sorted by rank."""
        sorted_drivers = sorted(self.drivers, key=lambda d: d.rank)
        return sorted_drivers[:n]

    def by_method(self, method: str) -> list:
        """Return drivers discovered by a specific method."""
        return [d for d in self.drivers if d.method == method]

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "target_column": self.target_column,
            "target_type": self.target_type,
            "drivers": [d.to_dict() for d in self.drivers],
            "relationships": [
                r.to_dict() if hasattr(r, "to_dict") else r
                for r in self.relationships
            ],
            "unavailable_analyses": list(self.unavailable_analyses),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: dict) -> DriverReport:
        return cls(
            dataset_id=d.get("dataset_id", ""),
            target_column=d.get("target_column"),
            target_type=d.get("target_type", ""),
            drivers=[DriverResult.from_dict(dr) for dr in d.get("drivers", [])],
            relationships=d.get("relationships", []),
            unavailable_analyses=d.get("unavailable_analyses", []),
            metadata=d.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Aggregate Phase 5 Result
# ---------------------------------------------------------------------------

@dataclass
class Phase5Result:
    """Aggregate of anomaly detection and driver analysis for one dataset."""

    dataset_id: str = ""
    anomaly_report: Optional[AnomalyReport] = None
    driver_report: Optional[DriverReport] = None

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "anomaly_report": self.anomaly_report.to_dict() if self.anomaly_report else None,
            "driver_report": self.driver_report.to_dict() if self.driver_report else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Phase5Result:
        return cls(
            dataset_id=d.get("dataset_id", ""),
            anomaly_report=AnomalyReport.from_dict(d["anomaly_report"]) if d.get("anomaly_report") else None,
            driver_report=DriverReport.from_dict(d["driver_report"]) if d.get("driver_report") else None,
        )
