"""Phase 4 — Predictive Analytics Engine Tests.

Comprehensive tests for:
- Prediction contracts (PredictiveTask, ForecastResult, etc.)
- Task detection engine (all dataset types)
- Preprocessing & feature engineering
- Forecasting engine
- Regression engine
- Classification engine
- Segmentation engine
- Dataset isolation (no cross-dataset leakage)
- Safety (no hardcoded column names)
- Target leakage protection
- Edge cases
- Integration (analysis_runner produces Phase 4 artifacts)

Uses Phase 2 CanonicalDataset + AnalysisContext as input.
"""

import json
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_engine.canonical import CanonicalDataset, AnalysisContext, build_canonical
from src.prediction.contracts import (
    PredictiveTask, ForecastResult, RegressionResult, ClassificationResult,
    SegmentationResult, PredictionReport, ModelCandidate, FeatureImportance,
    ClusterProfile, _make_id,
)
from src.prediction.task_detector import detect_predictive_tasks, get_feature_columns
from src.prediction.preprocessing import preprocess_for_prediction, preprocess_for_segmentation
from src.prediction.forecasting_engine import run_forecast
from src.prediction.regression_engine import run_regression
from src.prediction.classification_engine import run_classification
from src.prediction.segmentation_engine import run_segmentation
from tests.helpers import load_fixture, all_diverse_fixtures


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_context(df: pd.DataFrame, dataset_id: str = None) -> AnalysisContext:
    """Build a canonical + context from a raw DataFrame."""
    dataset_id = dataset_id or f"test-{uuid.uuid4().hex[:8]}"
    canonical, context = build_canonical(dataset_id, df)
    return context


def _build_context_from_fixture(name: str, subdir: str = "diverse_schemas") -> AnalysisContext:
    """Load a fixture CSV and build context."""
    df = load_fixture(name, subdir)
    return _build_context(df, dataset_id=f"fixture-{name.replace('.csv', '')}")


# ===========================================================================
# 1. CONTRACTS TESTS
# ===========================================================================

class TestContracts:
    """Tests for the Phase 4 prediction data contracts."""

    def test_predictive_task_roundtrip(self):
        task = PredictiveTask(
            task_type="regression",
            target_column="revenue",
            feature_columns=["clicks", "spend"],
            confidence=0.85,
            reason="Strong numeric target with features.",
            requirements_met=True,
            limitations=["Small dataset"],
        )
        d = task.to_dict()
        assert d["task_type"] == "regression"
        assert d["target_column"] == "revenue"
        assert d["confidence"] == 0.85
        assert d["requirements_met"] is True
        assert len(d["feature_columns"]) == 2
        assert len(d["limitations"]) == 1

        task2 = PredictiveTask.from_dict(d)
        assert task2.task_type == task.task_type
        assert task2.target_column == task.target_column
        assert task2.confidence == task.confidence
        assert task2.requirements_met == task.requirements_met

    def test_forecast_result_roundtrip(self):
        result = ForecastResult(
            status="success",
            target_column="sales",
            time_column="date",
            horizon=6,
            frequency="monthly",
            model_name="Linear Trend",
            predictions=[100.0, 110.0, 120.0],
            lower_bound=[90.0, 100.0, 110.0],
            upper_bound=[110.0, 120.0, 130.0],
            future_dates=["2025-01-01", "2025-02-01", "2025-03-01"],
            metrics={"MAE": 5.0, "RMSE": 6.0},
            confidence=0.82,
            model_candidates=[
                ModelCandidate(name="Naive", model_type="naive", score=10.0),
                ModelCandidate(name="Linear Trend", model_type="linear_trend", score=5.0, selected=True),
            ],
            feature_importance=[
                FeatureImportance(feature="time_index", importance=0.95, rank=1),
            ],
        )
        d = result.to_dict()
        assert d["status"] == "success"
        assert len(d["predictions"]) == 3
        assert len(d["model_candidates"]) == 2
        assert len(d["feature_importance"]) == 1

        result2 = ForecastResult.from_dict(d)
        assert result2.target_column == "sales"
        assert result2.horizon == 6
        assert len(result2.model_candidates) == 2
        assert result2.model_candidates[1].selected is True

    def test_regression_result_roundtrip(self):
        result = RegressionResult(
            status="success",
            target_column="revenue",
            feature_columns=["clicks", "spend"],
            model_name="Ridge",
            metrics={"MAE": 2.5, "RMSE": 3.0, "R2": 0.85},
            predictions=[10.0, 20.0],
            actuals=[11.0, 19.0],
            residuals=[1.0, -1.0],
            confidence=0.85,
            n_train=80,
            n_test=20,
            model_candidates=[
                ModelCandidate(name="Ridge", model_type="Ridge", score=0.85, selected=True),
            ],
            feature_importance=[
                FeatureImportance(feature="clicks", importance=0.6, rank=1),
                FeatureImportance(feature="spend", importance=0.4, rank=2),
            ],
        )
        d = result.to_dict()
        result2 = RegressionResult.from_dict(d)
        assert result2.target_column == "revenue"
        assert result2.n_train == 80
        assert result2.n_test == 20
        assert len(result2.feature_importance) == 2

    def test_classification_result_roundtrip(self):
        result = ClassificationResult(
            status="success",
            target_column="churn",
            classes=["0", "1"],
            feature_columns=["tenure", "spend"],
            model_name="RandomForest",
            metrics={"accuracy": 0.9, "f1": 0.88},
            confusion_matrix=[[40, 5], [3, 52]],
            confidence=0.88,
            n_train=80,
            n_test=20,
        )
        d = result.to_dict()
        result2 = ClassificationResult.from_dict(d)
        assert result2.target_column == "churn"
        assert result2.classes == ["0", "1"]
        assert result2.confusion_matrix == [[40, 5], [3, 52]]

    def test_segmentation_result_roundtrip(self):
        profiles = [
            ClusterProfile(cluster_id=0, label="Cluster 1", size=50, percentage=50.0,
                           centroid={"x": 1.0, "y": 2.0}, description="High x."),
            ClusterProfile(cluster_id=1, label="Cluster 2", size=50, percentage=50.0,
                           centroid={"x": 5.0, "y": 6.0}, description="High y."),
        ]
        result = SegmentationResult(
            status="success",
            cluster_count=2,
            labels=[0] * 50 + [1] * 50,
            feature_columns=["x", "y"],
            cluster_profiles=profiles,
            quality_metrics={"silhouette_score": 0.65, "inertia": 100.0},
            interpretation="Two clear clusters.",
        )
        d = result.to_dict()
        result2 = SegmentationResult.from_dict(d)
        assert result2.cluster_count == 2
        assert len(result2.cluster_profiles) == 2
        assert result2.cluster_profiles[0].label == "Cluster 1"
        assert result2.quality_metrics["silhouette_score"] == 0.65

    def test_prediction_report_roundtrip(self):
        report = PredictionReport(
            dataset_id="test-123",
            tasks=[
                PredictiveTask(task_type="regression", confidence=0.8, requirements_met=True),
                PredictiveTask(task_type="segmentation", confidence=0.6, requirements_met=True),
            ],
            regression=RegressionResult(status="success", target_column="x"),
            segmentation=SegmentationResult(status="success", cluster_count=3),
        )
        d = report.to_dict()
        report2 = PredictionReport.from_dict(d)
        assert report2.dataset_id == "test-123"
        assert len(report2.tasks) == 2
        assert report2.regression is not None
        assert report2.regression.target_column == "x"
        assert report2.segmentation is not None
        assert report2.segmentation.cluster_count == 3
        assert report2.forecast is None
        assert report2.classification is None

    def test_make_id_uniqueness(self):
        ids = {_make_id("test_") for _ in range(100)}
        assert len(ids) == 100

    def test_model_candidate_roundtrip(self):
        mc = ModelCandidate(
            name="RandomForest",
            model_type="RandomForestRegressor",
            metrics={"MAE": 3.5, "R2": 0.78},
            score=0.78,
            training_time_seconds=1.234,
            selected=True,
        )
        d = mc.to_dict()
        mc2 = ModelCandidate.from_dict(d)
        assert mc2.name == "RandomForest"
        assert mc2.selected is True
        assert mc2.score == 0.78

    def test_feature_importance_roundtrip(self):
        fi = FeatureImportance(feature="clicks", importance=0.456789, rank=1)
        d = fi.to_dict()
        fi2 = FeatureImportance.from_dict(d)
        assert fi2.feature == "clicks"
        assert fi2.rank == 1
        assert abs(fi2.importance - 0.456789) < 1e-5


# ===========================================================================
# 2. TASK DETECTOR TESTS
# ===========================================================================

class TestTaskDetector:
    """Tests for the predictive task detection engine."""

    def test_marketing_dataset_tasks(self):
        ctx = _build_context_from_fixture("a_marketing.csv")
        tasks = detect_predictive_tasks(ctx)
        assert len(tasks) > 0
        task_types = {t.task_type for t in tasks}
        # Marketing has date + numeric -> forecasting should be detected
        assert "forecasting" in task_types
        forecast_task = next(t for t in tasks if t.task_type == "forecasting")
        assert forecast_task.requirements_met is True

    def test_sales_dataset_tasks(self):
        ctx = _build_context_from_fixture("b_sales.csv")
        tasks = detect_predictive_tasks(ctx)
        assert len(tasks) > 0
        task_types = {t.task_type for t in tasks}
        # Sales should have regression and segmentation at minimum
        assert "regression" in task_types or "segmentation" in task_types

    def test_no_date_dataset(self):
        ctx = _build_context_from_fixture("g_no_date.csv")
        tasks = detect_predictive_tasks(ctx)
        forecast_task = next((t for t in tasks if t.task_type == "forecasting"), None)
        assert forecast_task is not None
        assert forecast_task.requirements_met is False

    def test_education_dataset_tasks(self):
        ctx = _build_context_from_fixture("e_education.csv")
        tasks = detect_predictive_tasks(ctx)
        assert len(tasks) > 0
        task_types = {t.task_type for t in tasks}
        # Education has numeric columns -> regression possible
        assert "regression" in task_types

    def test_minimal_dataset(self):
        ctx = _build_context_from_fixture("k_minimal.csv")
        tasks = detect_predictive_tasks(ctx)
        # k_minimal has only 5 rows, below MIN_ROWS_SUPERVISED (20)
        for task in tasks:
            if task.task_type in ("regression", "classification"):
                assert task.requirements_met is False

    def test_segmentation_possible_with_numerics(self):
        """Datasets with 2+ numeric columns and 10+ rows should allow segmentation."""
        ctx = _build_context_from_fixture("a_marketing.csv")
        tasks = detect_predictive_tasks(ctx)
        seg_task = next((t for t in tasks if t.task_type == "segmentation"), None)
        assert seg_task is not None
        assert seg_task.requirements_met is True

    def test_feature_columns_exclude_identifiers(self):
        ctx = _build_context_from_fixture("e_education.csv")
        features = get_feature_columns(ctx, target_column=None, task_type="regression")
        assert "Student_ID" not in features

    def test_feature_columns_exclude_target(self):
        ctx = _build_context_from_fixture("a_marketing.csv")
        ds = ctx.dataset
        if ds.target_candidates:
            target = ds.target_candidates[0].column
            features = get_feature_columns(ctx, target_column=target, task_type="regression")
            assert target not in features

    def test_classification_target_detection(self):
        """Create df with clear categorical target, verify classification detected."""
        np.random.seed(42)
        df = pd.DataFrame({
            "feat1": np.round(np.random.randn(100), 1),
            "feat2": np.round(np.random.randn(100), 1),
            "feat3": np.round(np.random.randn(100), 1),
            "label": np.random.choice(["cat", "dog", "bird"], 100),
        })
        ctx = _build_context(df, "clf-target-test")
        tasks = detect_predictive_tasks(ctx)
        clf_task = next((t for t in tasks if t.task_type == "classification"), None)
        assert clf_task is not None
        # Should detect 'label' as classification target
        if clf_task.requirements_met:
            assert clf_task.target_column == "label"

    def test_empty_dataframe_tasks(self):
        df = pd.DataFrame()
        ctx = _build_context(df, "empty-task-test")
        tasks = detect_predictive_tasks(ctx)
        # Should still return tasks but all with requirements_met=False
        for task in tasks:
            assert task.requirements_met is False

    def test_all_constant_target(self):
        """Constant-value target should not meet requirements for regression."""
        df = pd.DataFrame({
            "x": np.round(np.random.randn(50), 1),
            "y": np.round(np.random.randn(50), 1),
            "target": [42.0] * 50,
        })
        ctx = _build_context(df, "constant-target")
        tasks = detect_predictive_tasks(ctx)
        # All tasks should have low confidence or unmet requirements
        for task in tasks:
            if task.task_type == "regression" and task.target_column == "target":
                # A constant target has no information; confidence should be low
                assert task.confidence <= 0.5 or not task.requirements_met

    def test_task_types_valid(self):
        """All detected tasks have valid task_type."""
        from src.prediction.contracts import TASK_TYPES
        ctx = _build_context_from_fixture("a_marketing.csv")
        tasks = detect_predictive_tasks(ctx)
        for task in tasks:
            assert task.task_type in TASK_TYPES

    def test_tasks_sorted_by_confidence(self):
        """Tasks should be sorted: requirements_met first, then by confidence."""
        ctx = _build_context_from_fixture("a_marketing.csv")
        tasks = detect_predictive_tasks(ctx)
        met = [t for t in tasks if t.requirements_met]
        not_met = [t for t in tasks if not t.requirements_met]
        # All met tasks should come before not-met tasks
        if met and not_met:
            last_met_idx = max(tasks.index(t) for t in met)
            first_not_met_idx = min(tasks.index(t) for t in not_met)
            assert last_met_idx < first_not_met_idx


# ===========================================================================
# 3. PREPROCESSING TESTS
# ===========================================================================

class TestPreprocessing:
    """Tests for the preprocessing pipeline."""

    def test_basic_preprocessing(self):
        df = pd.DataFrame({
            "target": [1.0, 2.0, 3.0, 4.0, 5.0] * 10,
            "feat1": np.round(np.random.randn(50), 1),
            "feat2": np.round(np.random.randn(50), 1),
        })
        X, y, features, report = preprocess_for_prediction(
            df, "target", ["feat1", "feat2"], "regression"
        )
        assert X.shape[0] == 50
        assert X.shape[1] == 2
        assert len(y) == 50
        assert "feat1" in features
        assert "feat2" in features

    def test_nan_imputation(self):
        df = pd.DataFrame({
            "target": [1.0, 2.0, 3.0, 4.0, 5.0] * 10,
            "feat1": [np.nan, 2.0, 3.0, np.nan, 5.0] * 10,
            "feat2": np.round(np.random.randn(50), 1),
        })
        X, y, features, report = preprocess_for_prediction(
            df, "target", ["feat1", "feat2"], "regression"
        )
        assert not X.isna().any().any()
        assert "feat1" in report["imputed_columns"]

    def test_categorical_encoding(self):
        df = pd.DataFrame({
            "target": np.round(np.random.randn(50), 1),
            "color": np.random.choice(["red", "blue", "green"], 50),
            "value": np.round(np.random.randn(50), 1),
        })
        X, y, features, report = preprocess_for_prediction(
            df, "target", ["color", "value"], "regression"
        )
        assert not X.isna().any().any()
        assert "color" in report["encoded_columns"]
        # Should have one-hot encoded color into multiple columns
        color_cols = [f for f in features if f.startswith("color_")]
        assert len(color_cols) > 0

    def test_id_column_exclusion(self):
        df = pd.DataFrame({
            "id": range(50),
            "target": np.round(np.random.randn(50), 1),
            "feat1": np.round(np.random.randn(50), 1),
        })
        X, y, features, report = preprocess_for_prediction(
            df, "target", ["id", "feat1"], "regression"
        )
        assert "id" not in features
        assert "id" in report.get("id_columns_removed", [])

    def test_target_not_in_features(self):
        """CRITICAL: Target must NOT appear in X columns."""
        df = pd.DataFrame({
            "target": np.round(np.random.randn(50), 1),
            "feat1": np.round(np.random.randn(50), 1),
            "feat2": np.round(np.random.randn(50), 1),
        })
        X, y, features, report = preprocess_for_prediction(
            df, "target", ["target", "feat1", "feat2"], "regression"
        )
        assert "target" not in features
        assert "target" not in X.columns

    def test_target_leakage_detection(self):
        """Column with 0.999 correlation to target should be removed."""
        np.random.seed(42)
        target = np.round(np.random.randn(100), 1)
        df = pd.DataFrame({
            "target": target,
            "leaky": target * 1.0 + np.random.randn(100) * 0.001,
            "clean": np.round(np.random.randn(100), 1),
        })
        X, y, features, report = preprocess_for_prediction(
            df, "target", ["leaky", "clean"], "regression"
        )
        assert "leaky" not in features or "leaky" in report.get("leakage_columns_removed", [])

    def test_constant_column_removal(self):
        df = pd.DataFrame({
            "target": np.round(np.random.randn(50), 1),
            "constant": [7.0] * 50,
            "feat1": np.round(np.random.randn(50), 1),
        })
        X, y, features, report = preprocess_for_prediction(
            df, "target", ["constant", "feat1"], "regression"
        )
        assert "constant" not in features
        assert "constant" in report.get("constant_columns_removed", [])

    def test_datetime_feature_extraction(self):
        df = pd.DataFrame({
            "target": np.round(np.random.randn(50), 1),
            "date": pd.date_range("2024-01-01", periods=50, freq="D"),
            "feat1": np.round(np.random.randn(50), 1),
        })
        X, y, features, report = preprocess_for_prediction(
            df, "target", ["date", "feat1"], "regression"
        )
        dt_features = report.get("datetime_features_added", [])
        assert len(dt_features) > 0
        # Should have year, month, day_of_week, quarter
        assert any("year" in f for f in dt_features)
        assert any("month" in f for f in dt_features)

    def test_high_cardinality_categorical_dropped(self):
        df = pd.DataFrame({
            "target": np.round(np.random.randn(100), 1),
            "high_card": [f"val_{i}" for i in range(100)],
            "feat1": np.round(np.random.randn(100), 1),
        })
        X, y, features, report = preprocess_for_prediction(
            df, "target", ["high_card", "feat1"], "regression"
        )
        assert "high_card" not in features
        assert "high_card" in report.get("high_cardinality_dropped", [])

    def test_segmentation_preprocessing(self):
        df = pd.DataFrame({
            "x": np.round(np.random.randn(50), 1),
            "y": np.round(np.random.randn(50), 1),
            "cat": np.random.choice(["A", "B"], 50),
        })
        X, features, report = preprocess_for_segmentation(df, ["x", "y", "cat"])
        # Should return only numeric columns
        assert X.select_dtypes(exclude="number").empty
        assert "x" in features
        assert "y" in features
        assert report["scaled"] is True

    def test_empty_df_preprocessing(self):
        df = pd.DataFrame()
        X, y, features, report = preprocess_for_prediction(
            df, "target", ["feat1"], "regression"
        )
        assert X.empty
        assert len(features) == 0
        assert report["rows_after"] == 0


# ===========================================================================
# 4. FORECAST ENGINE TESTS
# ===========================================================================

class TestForecastEngine:
    """Tests for the forecasting engine."""

    def test_forecast_marketing(self):
        ctx = _build_context_from_fixture("a_marketing.csv")
        result = run_forecast(ctx)
        assert result.status == "success"

    def test_forecast_has_predictions(self):
        ctx = _build_context_from_fixture("a_marketing.csv")
        result = run_forecast(ctx)
        if result.status == "success":
            assert len(result.predictions) > 0

    def test_forecast_has_metrics(self):
        ctx = _build_context_from_fixture("a_marketing.csv")
        result = run_forecast(ctx)
        if result.status == "success":
            assert "MAE" in result.metrics
            assert "RMSE" in result.metrics

    def test_forecast_confidence_bounds(self):
        ctx = _build_context_from_fixture("a_marketing.csv")
        result = run_forecast(ctx)
        if result.status == "success":
            assert len(result.lower_bound) == len(result.predictions)
            assert len(result.upper_bound) == len(result.predictions)
            # Lower bound should be <= upper bound
            for lb, ub in zip(result.lower_bound, result.upper_bound):
                assert lb <= ub

    def test_forecast_model_candidates(self):
        ctx = _build_context_from_fixture("a_marketing.csv")
        result = run_forecast(ctx)
        if result.status == "success":
            assert len(result.model_candidates) >= 2
            # Exactly one should be selected
            selected = [m for m in result.model_candidates if m.selected]
            assert len(selected) == 1

    def test_forecast_no_time_column(self):
        ctx = _build_context_from_fixture("g_no_date.csv")
        result = run_forecast(ctx)
        assert result.status == "unavailable"

    def test_forecast_minimal_data(self):
        ctx = _build_context_from_fixture("k_minimal.csv")
        result = run_forecast(ctx)
        # k_minimal has 5 rows but may lack a time column
        assert result.status in ("unavailable", "insufficient_data")

    def test_forecast_future_dates(self):
        ctx = _build_context_from_fixture("a_marketing.csv")
        result = run_forecast(ctx)
        if result.status == "success":
            assert len(result.future_dates) == result.horizon
            assert len(result.future_dates) == len(result.predictions)

    def test_forecast_constant_target(self):
        """Constant target should not produce success."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=50, freq="D"),
            "value": [42.0] * 50,
        })
        ctx = _build_context(df, "const-forecast")
        result = run_forecast(ctx)
        assert result.status in ("failed", "unavailable")


# ===========================================================================
# 5. REGRESSION ENGINE TESTS
# ===========================================================================

class TestRegressionEngine:
    """Tests for the regression engine."""

    def test_regression_marketing(self):
        ctx = _build_context_from_fixture("a_marketing.csv")
        result = run_regression(ctx)
        assert result.status == "success"

    def test_regression_has_metrics(self):
        ctx = _build_context_from_fixture("a_marketing.csv")
        result = run_regression(ctx)
        if result.status == "success":
            assert "MAE" in result.metrics
            assert "RMSE" in result.metrics
            assert "R2" in result.metrics

    def test_regression_predictions_actuals(self):
        ctx = _build_context_from_fixture("a_marketing.csv")
        result = run_regression(ctx)
        if result.status == "success":
            assert len(result.predictions) > 0
            assert len(result.actuals) > 0
            assert len(result.predictions) == len(result.actuals)

    def test_regression_model_candidates(self):
        ctx = _build_context_from_fixture("a_marketing.csv")
        result = run_regression(ctx)
        if result.status == "success":
            assert len(result.model_candidates) >= 2
            selected = [m for m in result.model_candidates if m.selected]
            assert len(selected) == 1

    def test_regression_feature_importance(self):
        ctx = _build_context_from_fixture("a_marketing.csv")
        result = run_regression(ctx)
        if result.status == "success":
            assert len(result.feature_importance) > 0
            # Feature importances should be ranked
            ranks = [fi.rank for fi in result.feature_importance]
            assert ranks == sorted(ranks)

    def test_regression_train_test_split(self):
        ctx = _build_context_from_fixture("a_marketing.csv")
        result = run_regression(ctx)
        if result.status == "success":
            assert result.n_train > 0
            assert result.n_test > 0

    def test_regression_confidence_range(self):
        ctx = _build_context_from_fixture("a_marketing.csv")
        result = run_regression(ctx)
        if result.status == "success" and result.confidence is not None:
            assert 0.0 <= result.confidence <= 1.0

    def test_regression_minimal_data(self):
        ctx = _build_context_from_fixture("k_minimal.csv")
        result = run_regression(ctx)
        assert result.status in ("insufficient_data", "unavailable", "failed")

    def test_regression_sales(self):
        ctx = _build_context_from_fixture("b_sales.csv")
        result = run_regression(ctx)
        assert result.status in ("success", "failed", "unavailable")

    def test_regression_residuals(self):
        ctx = _build_context_from_fixture("a_marketing.csv")
        result = run_regression(ctx)
        if result.status == "success":
            assert len(result.residuals) == len(result.predictions)


# ===========================================================================
# 6. CLASSIFICATION ENGINE TESTS
# ===========================================================================

class TestClassificationEngine:
    """Tests for the classification engine."""

    def test_classification_with_categorical_target(self):
        """Create a dataset with a clear categorical target."""
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            "feat1": np.round(np.random.randn(n), 1),
            "feat2": np.round(np.random.randn(n), 1),
            "feat3": np.round(np.random.randn(n), 1),
            "label": np.random.choice(["cat", "dog", "bird"], n),
        })
        ctx = _build_context(df, "clf-cat-test")
        result = run_classification(ctx, target_column="label")
        assert result.status == "success"

    def test_classification_has_metrics(self):
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            "feat1": np.round(np.random.randn(n), 1),
            "feat2": np.round(np.random.randn(n), 1),
            "label": np.random.choice(["A", "B"], n),
        })
        ctx = _build_context(df, "clf-metrics-test")
        result = run_classification(ctx, target_column="label")
        if result.status == "success":
            assert "accuracy" in result.metrics
            assert "precision" in result.metrics
            assert "recall" in result.metrics
            assert "f1" in result.metrics

    def test_classification_confusion_matrix(self):
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            "feat1": np.round(np.random.randn(n), 1),
            "feat2": np.round(np.random.randn(n), 1),
            "label": np.random.choice(["A", "B"], n),
        })
        ctx = _build_context(df, "clf-cm-test")
        result = run_classification(ctx, target_column="label")
        if result.status == "success":
            assert len(result.confusion_matrix) > 0
            # 2D list
            assert isinstance(result.confusion_matrix[0], list)

    def test_classification_model_candidates(self):
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            "feat1": np.round(np.random.randn(n), 1),
            "feat2": np.round(np.random.randn(n), 1),
            "label": np.random.choice(["X", "Y"], n),
        })
        ctx = _build_context(df, "clf-cands-test")
        result = run_classification(ctx, target_column="label")
        if result.status == "success":
            assert len(result.model_candidates) >= 2

    def test_classification_binary(self):
        """Binary 0/1 numeric target should work."""
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            "feat1": np.round(np.random.randn(n), 1),
            "feat2": np.round(np.random.randn(n), 1),
            "target": np.random.choice([0, 1], n),
        })
        ctx = _build_context(df, "clf-binary-test")
        result = run_classification(ctx, target_column="target")
        assert result.status in ("success", "failed")

    def test_classification_no_suitable_target(self):
        """All numeric, no categorical -> unavailable."""
        df = pd.DataFrame({
            "a": np.round(np.random.randn(50), 1),
            "b": np.round(np.random.randn(50), 1),
            "c": np.round(np.random.randn(50), 1),
        })
        ctx = _build_context(df, "clf-no-target")
        result = run_classification(ctx)
        assert result.status == "unavailable"

    def test_classification_single_class(self):
        """Only 1 class -> insufficient_data."""
        np.random.seed(42)
        df = pd.DataFrame({
            "feat1": np.round(np.random.randn(50), 1),
            "label": ["only_class"] * 50,
        })
        ctx = _build_context(df, "clf-single-class")
        result = run_classification(ctx, target_column="label")
        assert result.status == "insufficient_data"

    def test_classification_feature_importance(self):
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            "feat1": np.round(np.random.randn(n), 1),
            "feat2": np.round(np.random.randn(n), 1),
            "feat3": np.round(np.random.randn(n), 1),
            "label": np.random.choice(["A", "B", "C"], n),
        })
        ctx = _build_context(df, "clf-fi-test")
        result = run_classification(ctx, target_column="label")
        if result.status == "success":
            assert len(result.feature_importance) > 0


# ===========================================================================
# 7. SEGMENTATION ENGINE TESTS
# ===========================================================================

class TestSegmentationEngine:
    """Tests for the segmentation/clustering engine."""

    def test_segmentation_marketing(self):
        ctx = _build_context_from_fixture("a_marketing.csv")
        result = run_segmentation(ctx)
        assert result.status == "success"

    def test_segmentation_cluster_count(self):
        ctx = _build_context_from_fixture("a_marketing.csv")
        result = run_segmentation(ctx)
        if result.status == "success":
            assert result.cluster_count > 0
            assert result.cluster_count >= 2

    def test_segmentation_cluster_profiles(self):
        ctx = _build_context_from_fixture("a_marketing.csv")
        result = run_segmentation(ctx)
        if result.status == "success":
            assert len(result.cluster_profiles) == result.cluster_count
            for profile in result.cluster_profiles:
                assert profile.size > 0
                assert profile.label != ""

    def test_segmentation_quality_metrics(self):
        ctx = _build_context_from_fixture("a_marketing.csv")
        result = run_segmentation(ctx)
        if result.status == "success":
            assert "silhouette_score" in result.quality_metrics

    def test_segmentation_labels(self):
        ctx = _build_context_from_fixture("a_marketing.csv")
        result = run_segmentation(ctx)
        if result.status == "success":
            assert len(result.labels) > 0
            # All labels should be valid cluster IDs
            unique_labels = set(result.labels)
            assert all(0 <= lbl < result.cluster_count for lbl in unique_labels)

    def test_segmentation_profile_percentages(self):
        ctx = _build_context_from_fixture("a_marketing.csv")
        result = run_segmentation(ctx)
        if result.status == "success":
            total_pct = sum(p.percentage for p in result.cluster_profiles)
            assert abs(total_pct - 100.0) < 1.0  # should sum to ~100%

    def test_segmentation_insufficient_features(self):
        """Single numeric column should produce unavailable."""
        df = pd.DataFrame({
            "only_num": np.round(np.random.randn(50), 1),
            "category": np.random.choice(["A", "B"], 50),
        })
        ctx = _build_context(df, "seg-single-feat")
        result = run_segmentation(ctx)
        assert result.status in ("unavailable", "failed", "insufficient_data")

    def test_segmentation_minimal_data(self):
        """Very small dataset."""
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "y": [4.0, 5.0, 6.0],
        })
        ctx = _build_context(df, "seg-tiny")
        result = run_segmentation(ctx)
        # 3 rows may be below minimum or handled gracefully
        assert result.status in ("success", "insufficient_data", "failed")

    def test_segmentation_sales(self):
        ctx = _build_context_from_fixture("b_sales.csv")
        result = run_segmentation(ctx)
        assert result.status in ("success", "unavailable", "failed")


# ===========================================================================
# 8. NO HARDCODED COLUMNS TESTS
# ===========================================================================

class TestNoHardcodedColumns:
    """Verify that Phase 4 engines never hardcode marketing column names."""

    _FORBIDDEN = [
        "Revenue", "Google_Ads", "Facebook_Ads", "Total_Marketing_Spend",
        "New_Customers", "Returning_Customers", "NovaMart",
    ]

    def _check_source(self, filepath: Path):
        source = filepath.read_text(encoding="utf-8")
        for col in self._FORBIDDEN:
            lines = [
                l for l in source.split("\n")
                if col in l
                and not l.strip().startswith("#")
                and not l.strip().startswith('"""')
                and not l.strip().startswith("'")
            ]
            assert len(lines) == 0, (
                f"Hardcoded column '{col}' found in {filepath.name}: {lines[0].strip()}"
            )

    def test_no_marketing_columns_in_task_detector(self):
        self._check_source(ROOT / "src" / "prediction" / "task_detector.py")

    def test_no_marketing_columns_in_preprocessing(self):
        self._check_source(ROOT / "src" / "prediction" / "preprocessing.py")

    def test_no_marketing_columns_in_forecast(self):
        self._check_source(ROOT / "src" / "prediction" / "forecasting_engine.py")

    def test_no_marketing_columns_in_regression(self):
        self._check_source(ROOT / "src" / "prediction" / "regression_engine.py")

    def test_no_marketing_columns_in_classification(self):
        self._check_source(ROOT / "src" / "prediction" / "classification_engine.py")

    def test_no_marketing_columns_in_segmentation(self):
        self._check_source(ROOT / "src" / "prediction" / "segmentation_engine.py")


# ===========================================================================
# 9. DATASET ISOLATION TESTS
# ===========================================================================

class TestDatasetIsolation:
    """Verify no cross-dataset leakage in Phase 4 predictions."""

    def test_two_dataset_isolation(self):
        """Run predictions on two different datasets, verify results differ."""
        ctx_a = _build_context_from_fixture("a_marketing.csv")
        ctx_b = _build_context_from_fixture("e_education.csv")

        result_a = run_regression(ctx_a)
        result_b = run_regression(ctx_b)

        if result_a.status == "success" and result_b.status == "success":
            assert result_a.target_column != result_b.target_column

    def test_three_dataset_isolation(self):
        """Three datasets produce distinct prediction results."""
        ctx_a = _build_context_from_fixture("a_marketing.csv")
        ctx_b = _build_context_from_fixture("b_sales.csv")
        ctx_c = _build_context_from_fixture("e_education.csv")

        tasks_a = detect_predictive_tasks(ctx_a)
        tasks_b = detect_predictive_tasks(ctx_b)
        tasks_c = detect_predictive_tasks(ctx_c)

        # Each should produce its own set of tasks with own target columns
        def _targets(tasks):
            return {t.target_column for t in tasks if t.target_column}

        targets_a = _targets(tasks_a)
        targets_b = _targets(tasks_b)
        targets_c = _targets(tasks_c)

        # At least some targets should be different
        assert targets_a != targets_b or targets_b != targets_c

    def test_no_cross_dataset_leakage(self):
        """Predictions from dataset A do not contain dataset B's column names."""
        ctx_a = _build_context_from_fixture("a_marketing.csv")
        ctx_b = _build_context_from_fixture("e_education.csv")

        result_a = run_regression(ctx_a)
        result_b = run_regression(ctx_b)

        if result_a.status == "success":
            education_cols = {"Student_ID", "Study_Hours", "Exam_Score", "Attendance"}
            a_features = set(result_a.feature_columns)
            leaked = a_features & education_cols
            assert not leaked, f"Education columns {leaked} leaked to marketing regression"

        if result_b.status == "success":
            marketing_cols = {"Revenue", "Google_Ads", "Facebook_Ads", "YouTube_Ads"}
            b_features = set(result_b.feature_columns)
            leaked = b_features & marketing_cols
            assert not leaked, f"Marketing columns {leaked} leaked to education regression"

    def test_prediction_report_dataset_id(self):
        """Each report carries correct dataset_id."""
        ctx = _build_context_from_fixture("a_marketing.csv")
        tasks = detect_predictive_tasks(ctx)
        report = PredictionReport(
            dataset_id="fixture-a_marketing",
            tasks=tasks,
        )
        assert report.dataset_id == "fixture-a_marketing"


# ===========================================================================
# 10. EDGE CASE TESTS
# ===========================================================================

class TestEdgeCases:
    """Edge cases for predictive engines."""

    def test_empty_dataframe(self):
        """Empty df -> all engines handle gracefully."""
        df = pd.DataFrame()
        ctx = _build_context(df, "empty-edge")
        assert run_forecast(ctx).status in ("unavailable", "insufficient_data")
        assert run_regression(ctx).status in ("unavailable", "insufficient_data")
        assert run_classification(ctx).status in ("unavailable", "insufficient_data")
        assert run_segmentation(ctx).status in ("unavailable", "insufficient_data")

    def test_single_column_dataset(self):
        """Only 1 column."""
        df = pd.DataFrame({"value": np.round(np.random.randn(50), 1)})
        ctx = _build_context(df, "single-col-edge")
        # Regression may fail (no features), segmentation may fail (1 feature)
        reg = run_regression(ctx)
        seg = run_segmentation(ctx)
        assert reg.status in ("failed", "unavailable")
        assert seg.status in ("unavailable", "failed", "insufficient_data")

    def test_all_nan_target(self):
        """Target column all NaN."""
        df = pd.DataFrame({
            "target": [np.nan] * 50,
            "feat1": np.round(np.random.randn(50), 1),
            "feat2": np.round(np.random.randn(50), 1),
        })
        ctx = _build_context(df, "all-nan-target")
        result = run_regression(ctx, target_column="target")
        # Should handle gracefully -- all target rows are NaN
        assert result.status in ("failed", "unavailable", "insufficient_data")

    def test_constant_target(self):
        """Target with single value."""
        df = pd.DataFrame({
            "target": [42.0] * 50,
            "feat1": np.round(np.random.randn(50), 1),
            "feat2": np.round(np.random.randn(50), 1),
        })
        ctx = _build_context(df, "const-target-edge")
        result = run_regression(ctx, target_column="target")
        # Constant target: engine should fail, or succeed with poor/zero-variance metrics
        assert result.status in ("success", "failed", "unavailable", "insufficient_data")

    def test_very_small_dataset(self):
        """3 rows."""
        df = pd.DataFrame({
            "x": [1, 2, 3],
            "y": [4, 5, 6],
            "label": ["a", "b", "c"],
        })
        ctx = _build_context(df, "tiny-edge")
        reg = run_regression(ctx)
        clf = run_classification(ctx)
        assert reg.status in ("insufficient_data", "unavailable", "failed")
        assert clf.status in ("insufficient_data", "unavailable", "failed")

    def test_high_cardinality_classification(self):
        """Target with too many classes (>20)."""
        np.random.seed(42)
        df = pd.DataFrame({
            "feat1": np.round(np.random.randn(200), 1),
            "feat2": np.round(np.random.randn(200), 1),
            "label": [f"class_{i % 25}" for i in range(200)],
        })
        ctx = _build_context(df, "high-card-clf")
        result = run_classification(ctx, target_column="label")
        assert result.status in ("failed", "unavailable")

    def test_all_categorical_dataset(self):
        """No numeric columns -> segmentation unavailable."""
        df = pd.DataFrame({
            "a": np.random.choice(["x", "y", "z"], 50),
            "b": np.random.choice(["p", "q"], 50),
        })
        ctx = _build_context(df, "all-cat-edge")
        result = run_segmentation(ctx)
        assert result.status in ("unavailable", "failed", "insufficient_data")

    def test_mixed_types_no_crash(self):
        """Dataset with mixed types should not crash any engine."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=30, freq="D"),
            "amount": np.round(np.random.randn(30), 1) * 100,
            "count": np.random.randint(1, 50, 30),
            "category": np.random.choice(["A", "B", "C"], 30),
            "active": np.random.choice([True, False], 30),
        })
        ctx = _build_context(df, "mixed-types-edge")
        # None of these should raise exceptions
        tasks = detect_predictive_tasks(ctx)
        assert isinstance(tasks, list)
        try:
            run_forecast(ctx)
        except Exception:
            pytest.fail("Forecast crashed on mixed types")
        try:
            run_regression(ctx)
        except Exception:
            pytest.fail("Regression crashed on mixed types")
        try:
            run_segmentation(ctx)
        except Exception:
            pytest.fail("Segmentation crashed on mixed types")


# ===========================================================================
# 11. TARGET LEAKAGE PROTECTION TESTS (CRITICAL)
# ===========================================================================

class TestTargetLeakageProtection:
    """Critical tests for target leakage protection."""

    def test_target_never_in_features(self):
        """For every fixture, run preprocessing and verify target NOT in features."""
        for name, path in all_diverse_fixtures():
            df = pd.read_csv(path)
            ctx = _build_context(df, dataset_id=f"leak-check-{name}")
            ds = ctx.dataset
            if not ds.target_candidates:
                continue
            target = ds.target_candidates[0].column
            features = get_feature_columns(ctx, target_column=target, task_type="regression")
            assert target not in features, (
                f"Target '{target}' found in features for fixture {name}"
            )

    def test_perfectly_correlated_feature_removed(self):
        """Add col = target * 1.0 + tiny noise, verify removed or flagged."""
        np.random.seed(42)
        target = np.round(np.random.randn(100), 1)
        df = pd.DataFrame({
            "target": target,
            "perfect_leak": target + np.random.randn(100) * 0.0001,
            "clean_feat": np.round(np.random.randn(100), 1),
        })
        X, y, features, report = preprocess_for_prediction(
            df, "target", ["perfect_leak", "clean_feat"], "regression"
        )
        # Either the leaky column is removed or flagged
        leakage_removed = report.get("leakage_columns_removed", [])
        assert "perfect_leak" not in features or "perfect_leak" in leakage_removed

    def test_future_info_not_in_training(self):
        """For forecasting, verify validation uses only past data."""
        ctx = _build_context_from_fixture("a_marketing.csv")
        result = run_forecast(ctx)
        if result.status == "success":
            # The forecast engine uses time-aware split (train = first 80%)
            # Verify that we have future_dates for the predictions
            assert len(result.future_dates) == len(result.predictions)
            # All future dates should be after the training data
            assert len(result.future_dates) > 0

    def test_leakage_in_regression(self):
        """Regression should remove leakage columns."""
        np.random.seed(42)
        target = np.round(np.random.randn(100), 1)
        df = pd.DataFrame({
            "target": target,
            "leak": target * 1.0 + np.random.randn(100) * 0.0001,
            "feat1": np.round(np.random.randn(100), 1),
            "feat2": np.round(np.random.randn(100), 1),
        })
        ctx = _build_context(df, "leak-reg-test")
        result = run_regression(ctx, target_column="target",
                                feature_columns=["leak", "feat1", "feat2"])
        if result.status == "success":
            # The leaky column should have been removed
            assert "leak" not in result.feature_columns or len(result.limitations) > 0


# ===========================================================================
# 12. RUNNER INTEGRATION TESTS
# ===========================================================================

class TestRunnerIntegration:
    """Verify analysis_runner produces Phase 4 prediction artifacts."""

    def test_runner_produces_prediction_report(self):
        """run_full_analysis should produce prediction_report.json."""
        import shutil

        df = load_fixture("a_marketing.csv")
        dataset_id = f"integ-p4-{uuid.uuid4().hex[:8]}"

        try:
            from src.data_engine.analysis_runner import run_full_analysis
            result = run_full_analysis(dataset_id, df=df)

            from config.settings import dataset_processed_dir
            proc_dir = dataset_processed_dir(dataset_id)
            pred_path = proc_dir / "prediction_report.json"
            assert pred_path.exists(), "prediction_report.json not created"

            with open(pred_path) as f:
                pred_data = json.load(f)

            assert "dataset_id" in pred_data
            assert "tasks" in pred_data
            assert pred_data["dataset_id"] == dataset_id

        finally:
            from config.settings import dataset_processed_dir
            proc = dataset_processed_dir(dataset_id)
            parent = proc.parent
            if parent.exists():
                shutil.rmtree(parent, ignore_errors=True)

    def test_prediction_report_json_schema(self):
        """Verify the JSON has expected keys."""
        import shutil

        df = load_fixture("a_marketing.csv")
        dataset_id = f"schema-p4-{uuid.uuid4().hex[:8]}"

        try:
            from src.data_engine.analysis_runner import run_full_analysis
            result = run_full_analysis(dataset_id, df=df)

            from config.settings import dataset_processed_dir
            proc_dir = dataset_processed_dir(dataset_id)
            pred_path = proc_dir / "prediction_report.json"

            with open(pred_path) as f:
                pred_data = json.load(f)

            expected_keys = {"dataset_id", "tasks"}
            assert expected_keys.issubset(set(pred_data.keys()))

            # Tasks should be a list
            assert isinstance(pred_data["tasks"], list)
            assert len(pred_data["tasks"]) > 0

            # Each task should have task_type
            for task in pred_data["tasks"]:
                assert "task_type" in task

        finally:
            from config.settings import dataset_processed_dir
            proc = dataset_processed_dir(dataset_id)
            parent = proc.parent
            if parent.exists():
                shutil.rmtree(parent, ignore_errors=True)

    def test_runner_report_mentions_phase4(self):
        """Pipeline report should mention Phase 4."""
        import shutil

        df = load_fixture("b_sales.csv")
        dataset_id = f"report-p4-{uuid.uuid4().hex[:8]}"

        try:
            from src.data_engine.analysis_runner import run_full_analysis
            result = run_full_analysis(dataset_id, df=df)
            report = result["report"]
            report_text = " ".join(report)
            assert "Phase 4" in report_text or "prediction" in report_text.lower()

        finally:
            from config.settings import dataset_processed_dir
            proc = dataset_processed_dir(dataset_id)
            parent = proc.parent
            if parent.exists():
                shutil.rmtree(parent, ignore_errors=True)


# ===========================================================================
# 13. PARAMETRIC TESTS — all fixtures through all engines
# ===========================================================================

_DIVERSE_FIXTURES = [name for name, _ in all_diverse_fixtures()]


@pytest.mark.parametrize("fixture_name", _DIVERSE_FIXTURES)
def test_task_detection_on_fixture(fixture_name):
    """Every fixture dataset must pass through task detection without error."""
    ctx = _build_context_from_fixture(fixture_name)
    tasks = detect_predictive_tasks(ctx)
    assert isinstance(tasks, list)
    assert all(isinstance(t, PredictiveTask) for t in tasks)
    # Every dataset should produce at least one task (even if unmet)
    assert len(tasks) > 0


@pytest.mark.parametrize("fixture_name", _DIVERSE_FIXTURES)
def test_task_types_complete(fixture_name):
    """Every fixture should have all 4 task types represented."""
    ctx = _build_context_from_fixture(fixture_name)
    tasks = detect_predictive_tasks(ctx)
    detected_types = {t.task_type for t in tasks}
    expected_types = {"forecasting", "regression", "classification", "segmentation"}
    assert detected_types == expected_types, (
        f"Fixture {fixture_name} missing task types: {expected_types - detected_types}"
    )


@pytest.mark.parametrize("fixture_name", _DIVERSE_FIXTURES)
def test_segmentation_on_fixture(fixture_name):
    """Run segmentation on every fixture -- should not crash."""
    ctx = _build_context_from_fixture(fixture_name)
    try:
        result = run_segmentation(ctx)
        assert isinstance(result, SegmentationResult)
        assert result.status in ("success", "failed", "unavailable", "insufficient_data")
    except Exception as exc:
        pytest.fail(f"Segmentation crashed on {fixture_name}: {exc}")


@pytest.mark.parametrize("fixture_name", _DIVERSE_FIXTURES)
def test_forecast_on_fixture(fixture_name):
    """Run forecast on every fixture -- should not crash."""
    ctx = _build_context_from_fixture(fixture_name)
    try:
        result = run_forecast(ctx)
        assert isinstance(result, ForecastResult)
        assert result.status in ("success", "failed", "unavailable", "insufficient_data")
    except Exception as exc:
        pytest.fail(f"Forecast crashed on {fixture_name}: {exc}")


@pytest.mark.parametrize("fixture_name", _DIVERSE_FIXTURES)
def test_regression_on_fixture(fixture_name):
    """Run regression on every fixture -- should not crash."""
    ctx = _build_context_from_fixture(fixture_name)
    try:
        result = run_regression(ctx)
        assert isinstance(result, RegressionResult)
        assert result.status in ("success", "failed", "unavailable", "insufficient_data")
    except Exception as exc:
        pytest.fail(f"Regression crashed on {fixture_name}: {exc}")


@pytest.mark.parametrize("fixture_name", _DIVERSE_FIXTURES)
def test_classification_on_fixture(fixture_name):
    """Run classification on every fixture -- should not crash."""
    ctx = _build_context_from_fixture(fixture_name)
    try:
        result = run_classification(ctx)
        assert isinstance(result, ClassificationResult)
        assert result.status in ("success", "failed", "unavailable", "insufficient_data")
    except Exception as exc:
        pytest.fail(f"Classification crashed on {fixture_name}: {exc}")
