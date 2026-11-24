"""Phase 4 — Predictive Analytics Contracts.

Standardized data contracts for all prediction tasks:
forecasting, regression, classification, segmentation.

Every result carries provenance, metrics, limitations, and explainability.
No hardcoded column names. No fabricated confidence.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


def _make_id(prefix: str = "pred_") -> str:
    return f"{prefix}{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Enums-as-strings (lightweight, JSON-safe)
# ---------------------------------------------------------------------------

TASK_TYPES = {"forecasting", "regression", "classification", "segmentation"}
MODEL_STATUS = {"success", "failed", "unavailable", "insufficient_data"}


# ---------------------------------------------------------------------------
# Predictive Task Detection
# ---------------------------------------------------------------------------

@dataclass
class PredictiveTask:
    """A detected predictive task that may be runnable on a dataset."""
    task_type: str  # forecasting | regression | classification | segmentation
    target_column: Optional[str] = None
    feature_columns: list = field(default_factory=list)
    confidence: float = 0.0  # 0.0–1.0, from actual detection metrics
    reason: str = ""
    requirements_met: bool = False
    limitations: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "task_type": self.task_type,
            "target_column": self.target_column,
            "feature_columns": list(self.feature_columns),
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
            "requirements_met": self.requirements_met,
            "limitations": list(self.limitations),
        }

    @classmethod
    def from_dict(cls, d: dict) -> PredictiveTask:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Model Candidate (for model selection)
# ---------------------------------------------------------------------------

@dataclass
class ModelCandidate:
    """A trained model candidate with evaluation metrics."""
    name: str
    model_type: str  # e.g. "Ridge", "RandomForest"
    metrics: dict = field(default_factory=dict)
    score: float = 0.0  # primary evaluation metric
    training_time_seconds: float = 0.0
    selected: bool = False

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "model_type": self.model_type,
            "metrics": dict(self.metrics),
            "score": round(self.score, 6),
            "training_time_seconds": round(self.training_time_seconds, 4),
            "selected": self.selected,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ModelCandidate:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Feature Importance
# ---------------------------------------------------------------------------

@dataclass
class FeatureImportance:
    """Feature importance for a trained model."""
    feature: str
    importance: float
    rank: int = 0

    def to_dict(self) -> dict:
        return {"feature": self.feature, "importance": round(self.importance, 6), "rank": self.rank}

    @classmethod
    def from_dict(cls, d: dict) -> FeatureImportance:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Forecast Result
# ---------------------------------------------------------------------------

@dataclass
class ForecastResult:
    """Result of a forecasting task."""
    id: str = field(default_factory=lambda: _make_id("fcast_"))
    status: str = "success"  # success | failed | unavailable | insufficient_data
    target_column: str = ""
    time_column: str = ""
    horizon: int = 0
    frequency: str = ""  # daily | weekly | monthly | yearly | irregular
    model_name: str = ""
    predictions: list = field(default_factory=list)  # list of float
    lower_bound: list = field(default_factory=list)
    upper_bound: list = field(default_factory=list)
    future_dates: list = field(default_factory=list)  # list of str
    metrics: dict = field(default_factory=dict)  # MAE, RMSE, MAPE, R2
    confidence: Optional[float] = None  # from actual model metrics
    model_candidates: list = field(default_factory=list)  # list[ModelCandidate]
    feature_importance: list = field(default_factory=list)
    limitations: list = field(default_factory=list)
    explanation: str = ""
    # Walk-forward evaluation record: method, folds, data ranges, per-candidate
    # backtest metrics and the comparison with the naive baselines.
    evaluation: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "status": self.status,
            "evaluation": self.evaluation,
            "target_column": self.target_column, "time_column": self.time_column,
            "horizon": self.horizon, "frequency": self.frequency,
            "model_name": self.model_name,
            "predictions": [round(v, 4) if isinstance(v, float) else v for v in self.predictions],
            "lower_bound": [round(v, 4) if isinstance(v, float) else v for v in self.lower_bound],
            "upper_bound": [round(v, 4) if isinstance(v, float) else v for v in self.upper_bound],
            "future_dates": list(self.future_dates),
            "metrics": {k: round(v, 6) if isinstance(v, (int, float)) else v for k, v in self.metrics.items()},
            "confidence": round(self.confidence, 4) if self.confidence is not None else None,
            "model_candidates": [m.to_dict() for m in self.model_candidates],
            "feature_importance": [f.to_dict() for f in self.feature_importance],
            "limitations": list(self.limitations),
            "explanation": self.explanation,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ForecastResult:
        d2 = dict(d)
        d2["model_candidates"] = [ModelCandidate.from_dict(m) for m in d2.get("model_candidates", [])]
        d2["feature_importance"] = [FeatureImportance.from_dict(f) for f in d2.get("feature_importance", [])]
        return cls(**{k: v for k, v in d2.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Regression Result
# ---------------------------------------------------------------------------

@dataclass
class RegressionResult:
    """Result of a regression task."""
    id: str = field(default_factory=lambda: _make_id("reg_"))
    status: str = "success"
    target_column: str = ""
    feature_columns: list = field(default_factory=list)
    model_name: str = ""
    metrics: dict = field(default_factory=dict)  # MAE, RMSE, R2, MAPE
    predictions: list = field(default_factory=list)
    actuals: list = field(default_factory=list)
    residuals: list = field(default_factory=list)
    feature_importance: list = field(default_factory=list)
    model_candidates: list = field(default_factory=list)
    confidence: Optional[float] = None
    limitations: list = field(default_factory=list)
    explanation: str = ""
    n_train: int = 0
    n_test: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id, "status": self.status,
            "target_column": self.target_column,
            "feature_columns": list(self.feature_columns),
            "model_name": self.model_name,
            "metrics": {k: round(v, 6) if isinstance(v, (int, float)) else v for k, v in self.metrics.items()},
            "predictions": [round(v, 4) if isinstance(v, float) else v for v in self.predictions[:200]],
            "actuals": [round(v, 4) if isinstance(v, float) else v for v in self.actuals[:200]],
            "residuals": [round(v, 4) if isinstance(v, float) else v for v in self.residuals[:200]],
            "feature_importance": [f.to_dict() for f in self.feature_importance],
            "model_candidates": [m.to_dict() for m in self.model_candidates],
            "confidence": round(self.confidence, 4) if self.confidence is not None else None,
            "limitations": list(self.limitations),
            "explanation": self.explanation,
            "n_train": self.n_train, "n_test": self.n_test,
        }

    @classmethod
    def from_dict(cls, d: dict) -> RegressionResult:
        d2 = dict(d)
        d2["model_candidates"] = [ModelCandidate.from_dict(m) for m in d2.get("model_candidates", [])]
        d2["feature_importance"] = [FeatureImportance.from_dict(f) for f in d2.get("feature_importance", [])]
        return cls(**{k: v for k, v in d2.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Classification Result
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    """Result of a classification task."""
    id: str = field(default_factory=lambda: _make_id("clf_"))
    status: str = "success"
    target_column: str = ""
    classes: list = field(default_factory=list)
    feature_columns: list = field(default_factory=list)
    model_name: str = ""
    metrics: dict = field(default_factory=dict)  # accuracy, precision, recall, f1, roc_auc
    confusion_matrix: list = field(default_factory=list)  # 2D list
    feature_importance: list = field(default_factory=list)
    model_candidates: list = field(default_factory=list)
    confidence: Optional[float] = None
    limitations: list = field(default_factory=list)
    explanation: str = ""
    n_train: int = 0
    n_test: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id, "status": self.status,
            "target_column": self.target_column,
            "classes": list(self.classes),
            "feature_columns": list(self.feature_columns),
            "model_name": self.model_name,
            "metrics": {k: round(v, 6) if isinstance(v, (int, float)) else v for k, v in self.metrics.items()},
            "confusion_matrix": self.confusion_matrix,
            "feature_importance": [f.to_dict() for f in self.feature_importance],
            "model_candidates": [m.to_dict() for m in self.model_candidates],
            "confidence": round(self.confidence, 4) if self.confidence is not None else None,
            "limitations": list(self.limitations),
            "explanation": self.explanation,
            "n_train": self.n_train, "n_test": self.n_test,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ClassificationResult:
        d2 = dict(d)
        d2["model_candidates"] = [ModelCandidate.from_dict(m) for m in d2.get("model_candidates", [])]
        d2["feature_importance"] = [FeatureImportance.from_dict(f) for f in d2.get("feature_importance", [])]
        return cls(**{k: v for k, v in d2.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Segmentation Result
# ---------------------------------------------------------------------------

@dataclass
class ClusterProfile:
    """Profile of a single cluster."""
    cluster_id: int
    label: str = ""
    size: int = 0
    percentage: float = 0.0
    centroid: dict = field(default_factory=dict)  # feature -> mean value
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id, "label": self.label,
            "size": self.size, "percentage": round(self.percentage, 2),
            "centroid": {k: round(v, 4) if isinstance(v, float) else v for k, v in self.centroid.items()},
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ClusterProfile:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class SegmentationResult:
    """Result of a segmentation/clustering task."""
    id: str = field(default_factory=lambda: _make_id("seg_"))
    status: str = "success"
    cluster_count: int = 0
    labels: list = field(default_factory=list)  # cluster assignment per row
    feature_columns: list = field(default_factory=list)
    cluster_profiles: list = field(default_factory=list)  # list[ClusterProfile]
    quality_metrics: dict = field(default_factory=dict)  # silhouette, inertia
    interpretation: str = ""
    limitations: list = field(default_factory=list)
    explanation: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id, "status": self.status,
            "cluster_count": self.cluster_count,
            "labels": list(self.labels[:500]),  # cap for JSON size
            "feature_columns": list(self.feature_columns),
            "cluster_profiles": [p.to_dict() for p in self.cluster_profiles],
            "quality_metrics": {k: round(v, 6) if isinstance(v, (int, float)) else v
                                for k, v in self.quality_metrics.items()},
            "interpretation": self.interpretation,
            "limitations": list(self.limitations),
            "explanation": self.explanation,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SegmentationResult:
        d2 = dict(d)
        d2["cluster_profiles"] = [ClusterProfile.from_dict(p) for p in d2.get("cluster_profiles", [])]
        return cls(**{k: v for k, v in d2.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Aggregate Prediction Result
# ---------------------------------------------------------------------------

@dataclass
class PredictionReport:
    """Aggregate of all predictive results for one dataset."""
    dataset_id: str
    tasks: list = field(default_factory=list)  # list[PredictiveTask]
    forecast: Optional[ForecastResult] = None
    regression: Optional[RegressionResult] = None
    classification: Optional[ClassificationResult] = None
    segmentation: Optional[SegmentationResult] = None

    def available_tasks(self) -> list[str]:
        return [t.task_type for t in self.tasks if t.requirements_met]

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "tasks": [t.to_dict() for t in self.tasks],
            "forecast": self.forecast.to_dict() if self.forecast else None,
            "regression": self.regression.to_dict() if self.regression else None,
            "classification": self.classification.to_dict() if self.classification else None,
            "segmentation": self.segmentation.to_dict() if self.segmentation else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PredictionReport:
        return cls(
            dataset_id=d["dataset_id"],
            tasks=[PredictiveTask.from_dict(t) for t in d.get("tasks", [])],
            forecast=ForecastResult.from_dict(d["forecast"]) if d.get("forecast") else None,
            regression=RegressionResult.from_dict(d["regression"]) if d.get("regression") else None,
            classification=ClassificationResult.from_dict(d["classification"]) if d.get("classification") else None,
            segmentation=SegmentationResult.from_dict(d["segmentation"]) if d.get("segmentation") else None,
        )
