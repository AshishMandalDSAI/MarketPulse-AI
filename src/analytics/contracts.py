"""Phase 3 analytics contracts — standardized representations for KPIs,
chart recommendations, insights, and relationships.

These are pure data contracts (dataclasses) with no heavy computation.
Every object is JSON-serializable via its to_dict() method.
"""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# KPI Contract
# ---------------------------------------------------------------------------

class KPICategory(str, enum.Enum):
    SUMMARY = "summary"
    FINANCIAL = "financial"
    OPERATIONAL = "operational"
    PERFORMANCE = "performance"
    QUALITY = "quality"
    DISTRIBUTION = "distribution"
    GROWTH = "growth"
    CUSTOM = "custom"


@dataclass
class KPI:
    """A single discovered KPI."""

    id: str
    label: str
    value: Any
    unit: str = ""
    formatted_value: str = ""
    source_column: str = ""
    calculation: str = ""
    category: str = "summary"
    confidence: float = 0.0
    explanation: str = ""
    priority: int = 0          # lower = more important
    delta: Optional[str] = None
    delta_positive: Optional[bool] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "value": self.value,
            "unit": self.unit,
            "formatted_value": self.formatted_value or str(self.value),
            "source_column": self.source_column,
            "calculation": self.calculation,
            "category": self.category,
            "confidence": round(self.confidence, 3),
            "explanation": self.explanation,
            "priority": self.priority,
            "delta": self.delta,
            "delta_positive": self.delta_positive,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KPI":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Chart / Visualization Recommendation Contract
# ---------------------------------------------------------------------------

class ChartType(str, enum.Enum):
    LINE = "line"
    BAR = "bar"
    HORIZONTAL_BAR = "horizontal_bar"
    AREA = "area"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    BOX = "box"
    PIE = "pie"
    DONUT = "donut"
    HEATMAP = "heatmap"
    TABLE = "table"


@dataclass
class ChartRecommendation:
    """A recommended chart for this dataset."""

    id: str
    chart_type: str
    title: str
    x_column: Optional[str] = None
    y_column: Optional[str] = None
    y_columns: Optional[list] = None
    group_by: Optional[str] = None
    aggregation: str = "none"
    score: float = 0.0
    reason: str = ""
    category: str = ""
    top_n: Optional[int] = None

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "chart_type": self.chart_type,
            "title": self.title,
            "x_column": self.x_column,
            "y_column": self.y_column,
            "aggregation": self.aggregation,
            "score": round(self.score, 3),
            "reason": self.reason,
            "category": self.category,
        }
        if self.y_columns:
            d["y_columns"] = self.y_columns
        if self.group_by:
            d["group_by"] = self.group_by
        if self.top_n is not None:
            d["top_n"] = self.top_n
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ChartRecommendation":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Insight Contract
# ---------------------------------------------------------------------------

class InsightType(str, enum.Enum):
    TREND = "trend"
    GROWTH = "growth"
    DECLINE = "decline"
    CONCENTRATION = "concentration"
    DISTRIBUTION = "distribution"
    CORRELATION = "correlation"
    OUTLIER = "outlier"
    COMPARISON = "comparison"
    DATA_QUALITY = "data_quality"


class InsightSeverity(str, enum.Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class Insight:
    """A data-driven insight with evidence."""

    id: str
    title: str
    description: str
    insight_type: str
    severity: str = "info"
    evidence: str = ""
    source_columns: list = field(default_factory=list)
    confidence: float = 0.0
    metric_value: Optional[Any] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "insight_type": self.insight_type,
            "severity": self.severity,
            "evidence": self.evidence,
            "source_columns": list(self.source_columns),
            "confidence": round(self.confidence, 3),
            "metric_value": self.metric_value,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Insight":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Relationship Contract
# ---------------------------------------------------------------------------

class RelationshipType(str, enum.Enum):
    CORRELATION = "correlation"
    GROUP_COMPARISON = "group_comparison"
    TREND = "trend"


@dataclass
class Relationship:
    """A detected relationship between columns."""

    id: str
    relationship_type: str
    columns: list
    strength: float = 0.0
    direction: str = ""
    description: str = ""
    evidence: str = ""
    p_value: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "relationship_type": self.relationship_type,
            "columns": list(self.columns),
            "strength": round(self.strength, 3),
            "direction": self.direction,
            "description": self.description,
            "evidence": self.evidence,
            "p_value": round(self.p_value, 4) if self.p_value is not None else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Relationship":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Aggregate analytics result
# ---------------------------------------------------------------------------

@dataclass
class AnalyticsResult:
    """Aggregated result from all Phase 3 engines for one dataset."""

    dataset_id: str
    kpis: list = field(default_factory=list)             # list[KPI]
    charts: list = field(default_factory=list)            # list[ChartRecommendation]
    insights: list = field(default_factory=list)          # list[Insight]
    relationships: list = field(default_factory=list)     # list[Relationship]
    unavailable_features: list = field(default_factory=list)  # list[dict]

    def primary_kpis(self, max_count: int = 8) -> list:
        """Return top-priority KPIs for the executive overview."""
        sorted_kpis = sorted(self.kpis, key=lambda k: k.priority)
        return sorted_kpis[:max_count]

    def top_charts(self, max_count: int = 6) -> list:
        """Return highest-scored chart recommendations."""
        sorted_charts = sorted(self.charts, key=lambda c: -c.score)
        return sorted_charts[:max_count]

    def top_insights(self, max_count: int = 8) -> list:
        """Return highest-confidence insights."""
        severity_order = {"high": 0, "medium": 1, "low": 2, "info": 3}
        sorted_insights = sorted(
            self.insights,
            key=lambda i: (severity_order.get(i.severity, 3), -i.confidence),
        )
        return sorted_insights[:max_count]

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "kpis": [k.to_dict() for k in self.kpis],
            "charts": [c.to_dict() for c in self.charts],
            "insights": [i.to_dict() for i in self.insights],
            "relationships": [r.to_dict() for r in self.relationships],
            "unavailable_features": list(self.unavailable_features),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AnalyticsResult":
        return cls(
            dataset_id=d["dataset_id"],
            kpis=[KPI.from_dict(k) for k in d.get("kpis", [])],
            charts=[ChartRecommendation.from_dict(c) for c in d.get("charts", [])],
            insights=[Insight.from_dict(i) for i in d.get("insights", [])],
            relationships=[Relationship.from_dict(r) for r in d.get("relationships", [])],
            unavailable_features=d.get("unavailable_features", []),
        )


def make_id(prefix: str = "") -> str:
    """Generate a short unique id for analytics objects."""
    return f"{prefix}{uuid.uuid4().hex[:8]}"
