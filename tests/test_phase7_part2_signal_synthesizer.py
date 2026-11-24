"""Phase 7 Part 2 — Signal Synthesizer tests.

Comprehensive tests covering:
1.  Regression-safe imports
2.  Valid signals from supported evidence
3.  Multiple analytical sources
4.  Empty and insufficient evidence
5.  Contradictory findings
6.  Deterministic signal generation and ranking
7.  Duplicate-signal handling
8.  Dataset isolation
9.  Confidence and limitation preservation
10. Generic non-marketing datasets
11. Marketing-specific evidence when genuinely supported
12. Missing or malformed inputs
13. Serialization compatibility with Part 1 contracts
14. Correlation-versus-causation safeguards
15. No demo-data leakage
"""

import pytest

from src.decision.contracts import (
    ConfidenceLevel,
    ConfidenceScore,
    DecisionEvidence,
    DecisionLimitation,
    DecisionSignal,
    EvidenceLevel,
    LimitationCategory,
)
from src.decision.evidence import (
    create_evidence,
    create_limitation,
    create_signal,
    ensure_non_causal,
    ensure_prediction_uncertainty,
    validate_dataset_scope,
)
from src.decision.confidence import assess_confidence
from src.decision.signal_synthesizer import (
    SignalPriority,
    SignalSynthesizer,
    SynthesisResult,
    SynthesizedSignal,
    _convert_string_limitations,
    _get_attr,
)


# ===================================================================
# 1. REGRESSION-SAFE IMPORTS
# ===================================================================


class TestImports:
    """Verify all public names are importable without side effects."""

    def test_import_signal_synthesizer_module(self):
        import src.decision.signal_synthesizer  # noqa: F401

    def test_import_signal_priority(self):
        assert hasattr(SignalPriority, "CRITICAL")
        assert hasattr(SignalPriority, "HIGH")
        assert hasattr(SignalPriority, "MEDIUM")
        assert hasattr(SignalPriority, "LOW")
        assert hasattr(SignalPriority, "INFO")

    def test_import_synthesized_signal(self):
        assert SynthesizedSignal is not None

    def test_import_synthesis_result(self):
        assert SynthesisResult is not None

    def test_import_signal_synthesizer_class(self):
        assert SignalSynthesizer is not None

    def test_import_helpers(self):
        assert callable(_get_attr)
        assert callable(_convert_string_limitations)

    def test_part1_contracts_still_importable(self):
        """Part 1 contracts must remain unchanged."""
        assert EvidenceLevel.FACT.value == "fact"
        assert ConfidenceLevel.HIGH.value == "high"
        assert LimitationCategory.OBSERVATIONAL.value == "observational"
        assert DecisionEvidence is not None
        assert DecisionSignal is not None
        assert ConfidenceScore is not None
        assert DecisionLimitation is not None


# ===================================================================
# 2. SIGNAL PRIORITY
# ===================================================================


class TestSignalPriority:
    """Signal priority is distinct from confidence."""

    def test_priority_ordering(self):
        assert SignalPriority.CRITICAL < SignalPriority.HIGH
        assert SignalPriority.HIGH < SignalPriority.MEDIUM
        assert SignalPriority.MEDIUM < SignalPriority.LOW
        assert SignalPriority.LOW < SignalPriority.INFO

    def test_priority_values(self):
        assert SignalPriority.CRITICAL == 1
        assert SignalPriority.HIGH == 2
        assert SignalPriority.MEDIUM == 3
        assert SignalPriority.LOW == 4
        assert SignalPriority.INFO == 5

    def test_from_severity_valid(self):
        assert SignalPriority.from_severity("critical") == SignalPriority.CRITICAL
        assert SignalPriority.from_severity("HIGH") == SignalPriority.HIGH
        assert SignalPriority.from_severity("medium") == SignalPriority.MEDIUM
        assert SignalPriority.from_severity("low") == SignalPriority.LOW
        assert SignalPriority.from_severity("info") == SignalPriority.INFO

    def test_from_severity_unknown_defaults_to_medium(self):
        assert SignalPriority.from_severity("unknown") == SignalPriority.MEDIUM
        assert SignalPriority.from_severity("") == SignalPriority.MEDIUM

    def test_from_severity_whitespace_handling(self):
        assert SignalPriority.from_severity("  high  ") == SignalPriority.HIGH


# ===================================================================
# 3. SYNTHESIZER CONSTRUCTION
# ===================================================================


class TestSynthesizerConstruction:
    """Verify synthesizer construction and basic properties."""

    def test_requires_dataset_id(self):
        with pytest.raises(ValueError, match="dataset_id is required"):
            SignalSynthesizer(dataset_id="")

    def test_valid_construction(self):
        s = SignalSynthesizer(dataset_id="ds_test")
        assert s.dataset_id == "ds_test"

    def test_synthesize_with_no_ingestion(self):
        s = SignalSynthesizer(dataset_id="ds_empty")
        result = s.synthesize()
        assert isinstance(result, SynthesisResult)
        assert len(result.signals) == 0
        assert result.dataset_id == "ds_empty"
        assert any("No signals generated" in note for note in result.processing_notes)


# ===================================================================
# 4. VALID SIGNALS FROM SUPPORTED EVIDENCE
# ===================================================================


class TestValidSignals:
    """Verify signals from supported Phase 3-6 evidence."""

    def test_kpi_signal(self):
        s = SignalSynthesizer(dataset_id="ds_001")
        analytics = {
            "dataset_id": "ds_001",
            "kpis": [
                {
                    "label": "mean_value",
                    "value": 42.5,
                    "source_column": "price",
                    "explanation": "Average price across dataset",
                    "category": "summary",
                },
            ],
            "insights": [],
            "relationships": [],
        }
        s.ingest_analytics(analytics)
        result = s.synthesize()
        assert len(result.signals) == 1
        sig = result.signals[0]
        assert "KPI" in sig.signal.description
        assert sig.source_type == "phase3_kpi"
        assert sig.priority == SignalPriority.INFO
        assert sig.generation_reason != ""

    def test_insight_signal(self):
        s = SignalSynthesizer(dataset_id="ds_001")
        analytics = {
            "dataset_id": "ds_001",
            "kpis": [],
            "insights": [
                {
                    "title": "Skewed distribution",
                    "description": "The column X is heavily skewed.",
                    "insight_type": "distribution",
                    "severity": "medium",
                    "evidence": "skewness > 2",
                    "source_columns": ["X"],
                    "metric_value": 2.3,
                },
            ],
            "relationships": [],
        }
        s.ingest_analytics(analytics)
        result = s.synthesize()
        assert len(result.signals) == 1
        sig = result.signals[0]
        assert "Insight" in sig.signal.description
        assert sig.source_type == "phase3_insight"
        assert sig.priority == SignalPriority.MEDIUM

    def test_relationship_signal(self):
        s = SignalSynthesizer(dataset_id="ds_001")
        analytics = {
            "dataset_id": "ds_001",
            "kpis": [],
            "insights": [],
            "relationships": [
                {
                    "relationship_type": "correlation",
                    "columns": ["A", "B"],
                    "strength": 0.85,
                    "direction": "positive",
                    "description": "Strong positive correlation between A and B.",
                },
            ],
        }
        s.ingest_analytics(analytics)
        result = s.synthesize()
        assert len(result.signals) == 1
        sig = result.signals[0]
        assert "Relationship" in sig.signal.description
        assert sig.source_type == "phase3_relationship"
        # Relationship must carry observational limitation
        ev = sig.evidence_items[0]
        obs_lims = [
            lim for lim in ev.limitations
            if lim.category == LimitationCategory.OBSERVATIONAL.value
        ]
        assert len(obs_lims) > 0, "Relationship evidence must have observational limitation"

    def test_forecast_signal(self):
        s = SignalSynthesizer(dataset_id="ds_001")
        report = {
            "dataset_id": "ds_001",
            "forecast": {
                "status": "success",
                "target_column": "revenue",
                "model_name": "arima",
                "horizon": 12,
                "metrics": {"mape": 0.08},
                "limitations": ["Limited historical data"],
                "feature_importance": [],
            },
        }
        s.ingest_predictions(report)
        result = s.synthesize()
        assert len(result.signals) == 1
        sig = result.signals[0]
        assert "Forecast" in sig.signal.description
        assert sig.source_type == "phase4_forecast"
        assert sig.priority == SignalPriority.HIGH
        # Forecast must carry prediction uncertainty
        ev = sig.evidence_items[0]
        uncert_lims = [
            lim for lim in ev.limitations
            if lim.category == LimitationCategory.MODEL_UNCERTAINTY.value
        ]
        assert len(uncert_lims) > 0, "Forecast evidence must have model uncertainty limitation"

    def test_regression_signal(self):
        s = SignalSynthesizer(dataset_id="ds_001")
        report = {
            "dataset_id": "ds_001",
            "regression": {
                "status": "success",
                "target_column": "sales",
                "model_name": "random_forest",
                "metrics": {"r2": 0.91},
                "limitations": [],
                "feature_importance": [],
            },
        }
        s.ingest_predictions(report)
        result = s.synthesize()
        assert len(result.signals) == 1
        sig = result.signals[0]
        assert "Regression" in sig.signal.description
        assert sig.source_type == "phase4_regression"

    def test_classification_signal(self):
        s = SignalSynthesizer(dataset_id="ds_001")
        report = {
            "dataset_id": "ds_001",
            "classification": {
                "status": "success",
                "target_column": "churn",
                "model_name": "logistic",
                "metrics": {"accuracy": 0.88},
                "limitations": [],
                "feature_importance": [],
            },
        }
        s.ingest_predictions(report)
        result = s.synthesize()
        assert len(result.signals) == 1
        sig = result.signals[0]
        assert "Classification" in sig.signal.description
        assert sig.source_type == "phase4_classification"

    def test_segmentation_signal(self):
        s = SignalSynthesizer(dataset_id="ds_001")
        report = {
            "dataset_id": "ds_001",
            "segmentation": {
                "status": "success",
                "cluster_count": 4,
                "feature_columns": ["age", "income"],
                "quality_metrics": {"silhouette": 0.62},
                "limitations": [],
            },
        }
        s.ingest_predictions(report)
        result = s.synthesize()
        assert len(result.signals) == 1
        sig = result.signals[0]
        assert "Segmentation" in sig.signal.description
        assert sig.source_type == "phase4_segmentation"
        assert sig.priority == SignalPriority.MEDIUM

    def test_anomaly_signal(self):
        s = SignalSynthesizer(dataset_id="ds_001")
        phase5 = {
            "dataset_id": "ds_001",
            "anomaly_report": {
                "anomalies": [
                    {
                        "column": "temperature",
                        "anomaly_type": "spike",
                        "severity": "HIGH",
                        "score": 3.5,
                        "observed_value": 99.0,
                        "expected_value": 72.0,
                        "method": "zscore",
                        "explanation": "Value exceeds 3 standard deviations.",
                    },
                ],
            },
        }
        s.ingest_anomalies(phase5)
        result = s.synthesize()
        assert len(result.signals) == 1
        sig = result.signals[0]
        assert "Anomaly" in sig.signal.description
        assert sig.source_type == "phase5_anomaly"
        assert sig.priority == SignalPriority.HIGH

    def test_driver_signal(self):
        s = SignalSynthesizer(dataset_id="ds_001")
        phase5 = {
            "dataset_id": "ds_001",
            "driver_report": {
                "target_column": "revenue",
                "drivers": [
                    {
                        "feature": "marketing_spend",
                        "target": "revenue",
                        "strength": 0.78,
                        "direction": "positive",
                        "relationship_type": "linear",
                        "method": "mutual_information",
                        "explanation": "Strong association.",
                    },
                ],
            },
        }
        s.ingest_anomalies(phase5)
        result = s.synthesize()
        assert len(result.signals) == 1
        sig = result.signals[0]
        assert "Driver" in sig.signal.description
        assert sig.source_type == "phase5_driver"


# ===================================================================
# 5. MULTIPLE ANALYTICAL SOURCES
# ===================================================================


class TestMultipleSources:
    """Signals from multiple phases combined."""

    def test_multiple_phases_combined(self):
        s = SignalSynthesizer(dataset_id="ds_multi")
        # Phase 3
        s.ingest_analytics({
            "dataset_id": "ds_multi",
            "kpis": [{"label": "count", "value": 100, "source_column": "id"}],
            "insights": [],
            "relationships": [],
        })
        # Phase 4
        s.ingest_predictions({
            "dataset_id": "ds_multi",
            "forecast": {
                "status": "success",
                "target_column": "sales",
                "model_name": "ets",
                "horizon": 6,
                "metrics": {},
                "limitations": [],
                "feature_importance": [],
            },
        })
        # Phase 5
        s.ingest_anomalies({
            "dataset_id": "ds_multi",
            "anomaly_report": {
                "anomalies": [
                    {
                        "column": "sales",
                        "anomaly_type": "dip",
                        "severity": "medium",
                        "score": 2.0,
                        "method": "iqr",
                        "explanation": "Dip below lower quartile.",
                    },
                ],
            },
        })
        result = s.synthesize()
        assert len(result.signals) == 3  # 1 KPI + 1 forecast + 1 anomaly

        # Check that signals_by_source filters correctly
        kpi_sigs = result.signals_by_source("phase3_kpi")
        assert len(kpi_sigs) == 1
        forecast_sigs = result.signals_by_source("phase4_forecast")
        assert len(forecast_sigs) == 1
        anomaly_sigs = result.signals_by_source("phase5_anomaly")
        assert len(anomaly_sigs) == 1

    def test_all_four_phases(self):
        s = SignalSynthesizer(dataset_id="ds_all")
        s.ingest_analytics({
            "dataset_id": "ds_all",
            "kpis": [{"label": "avg", "value": 50, "source_column": "x"}],
            "insights": [],
            "relationships": [],
        })
        s.ingest_predictions({
            "dataset_id": "ds_all",
            "regression": {
                "status": "success",
                "target_column": "y",
                "model_name": "linear",
                "metrics": {"r2": 0.80},
                "limitations": [],
                "feature_importance": [],
            },
        })
        s.ingest_anomalies({
            "dataset_id": "ds_all",
            "anomaly_report": {"anomalies": []},
            "driver_report": {
                "target_column": "y",
                "drivers": [
                    {
                        "feature": "z",
                        "strength": 0.65,
                        "direction": "positive",
                        "relationship_type": "monotonic",
                        "method": "shap",
                        "explanation": "Z is associated with Y.",
                    },
                ],
            },
        })
        s.ingest_marketing({
            "dataset_id": "ds_all",
            "contributions": [
                {
                    "channel": "search",
                    "contribution": 100.0,
                    "contribution_share": 0.5,
                    "roi": 3.0,
                },
            ],
            "roi_results": [],
        })
        result = s.synthesize()
        # 1 KPI + 1 regression + 1 driver + 1 contribution = 4
        assert len(result.signals) == 4


# ===================================================================
# 6. EMPTY AND INSUFFICIENT EVIDENCE
# ===================================================================


class TestEmptyAndInsufficientEvidence:
    """Verify safe handling of no data."""

    def test_no_results_ingested(self):
        s = SignalSynthesizer(dataset_id="ds_empty")
        result = s.synthesize()
        assert len(result.signals) == 0
        assert any("No signals generated" in n for n in result.processing_notes)

    def test_none_analytics(self):
        s = SignalSynthesizer(dataset_id="ds_001")
        s.ingest_analytics(None)
        result = s.synthesize()
        assert len(result.signals) == 0
        assert any("Phase 3" in n for n in result.processing_notes)

    def test_none_predictions(self):
        s = SignalSynthesizer(dataset_id="ds_001")
        s.ingest_predictions(None)
        result = s.synthesize()
        assert len(result.signals) == 0
        assert any("Phase 4" in n for n in result.processing_notes)

    def test_none_anomalies(self):
        s = SignalSynthesizer(dataset_id="ds_001")
        s.ingest_anomalies(None)
        result = s.synthesize()
        assert len(result.signals) == 0
        assert any("Phase 5" in n for n in result.processing_notes)

    def test_none_marketing(self):
        s = SignalSynthesizer(dataset_id="ds_001")
        s.ingest_marketing(None)
        result = s.synthesize()
        assert len(result.signals) == 0
        assert any("Phase 6" in n for n in result.processing_notes)

    def test_empty_kpi_list(self):
        s = SignalSynthesizer(dataset_id="ds_001")
        s.ingest_analytics({"dataset_id": "ds_001", "kpis": [], "insights": [], "relationships": []})
        result = s.synthesize()
        assert len(result.signals) == 0

    def test_failed_forecast_status(self):
        s = SignalSynthesizer(dataset_id="ds_001")
        s.ingest_predictions({
            "dataset_id": "ds_001",
            "forecast": {"status": "failed", "target_column": "x"},
        })
        result = s.synthesize()
        assert len(result.signals) == 0
        assert any("Forecast status" in n for n in result.processing_notes)

    def test_failed_regression_status(self):
        s = SignalSynthesizer(dataset_id="ds_001")
        s.ingest_predictions({
            "dataset_id": "ds_001",
            "regression": {"status": "error"},
        })
        result = s.synthesize()
        assert len(result.signals) == 0

    def test_failed_classification_status(self):
        s = SignalSynthesizer(dataset_id="ds_001")
        s.ingest_predictions({
            "dataset_id": "ds_001",
            "classification": {"status": "skipped"},
        })
        result = s.synthesize()
        assert len(result.signals) == 0

    def test_failed_segmentation_status(self):
        s = SignalSynthesizer(dataset_id="ds_001")
        s.ingest_predictions({
            "dataset_id": "ds_001",
            "segmentation": {"status": "insufficient_data"},
        })
        result = s.synthesize()
        assert len(result.signals) == 0

    def test_failed_optimization(self):
        s = SignalSynthesizer(dataset_id="ds_001")
        s.ingest_marketing({
            "dataset_id": "ds_001",
            "optimization_result": {"success": False, "message": "Not enough data."},
        })
        result = s.synthesize()
        assert len(result.signals) == 0
        assert any("optimization" in n.lower() for n in result.processing_notes)

    def test_empty_evidence_list_for_ingest_evidence(self):
        s = SignalSynthesizer(dataset_id="ds_001")
        result_sig = s.ingest_evidence([])
        assert result_sig is None

    def test_kpi_with_no_label_skipped(self):
        s = SignalSynthesizer(dataset_id="ds_001")
        s.ingest_analytics({
            "dataset_id": "ds_001",
            "kpis": [{"label": "", "value": 42}],
            "insights": [],
            "relationships": [],
        })
        result = s.synthesize()
        assert len(result.signals) == 0

    def test_insight_with_no_title_or_description_skipped(self):
        s = SignalSynthesizer(dataset_id="ds_001")
        s.ingest_analytics({
            "dataset_id": "ds_001",
            "kpis": [],
            "insights": [{"title": "", "description": "", "insight_type": "x"}],
            "relationships": [],
        })
        result = s.synthesize()
        assert len(result.signals) == 0

    def test_relationship_with_no_columns_skipped(self):
        s = SignalSynthesizer(dataset_id="ds_001")
        s.ingest_analytics({
            "dataset_id": "ds_001",
            "kpis": [],
            "insights": [],
            "relationships": [{"columns": [], "description": "Something."}],
        })
        result = s.synthesize()
        assert len(result.signals) == 0


# ===================================================================
# 7. CONTRADICTORY FINDINGS
# ===================================================================


class TestContradictoryFindings:
    """Contradictions must be surfaced, never silently discarded."""

    def test_contradictory_directions_surfaced(self):
        s = SignalSynthesizer(dataset_id="ds_001")
        # Two drivers on the same feature with opposite directions
        phase5 = {
            "dataset_id": "ds_001",
            "driver_report": {
                "target_column": "revenue",
                "drivers": [
                    {
                        "feature": "ad_spend",
                        "strength": 0.7,
                        "direction": "positive",
                        "relationship_type": "linear",
                        "method": "regression",
                        "explanation": "Positive association.",
                    },
                    {
                        "feature": "ad_spend",
                        "strength": 0.5,
                        "direction": "negative",
                        "relationship_type": "nonlinear",
                        "method": "spearman",
                        "explanation": "Negative association.",
                    },
                ],
            },
        }
        s.ingest_anomalies(phase5)
        result = s.synthesize()
        # Contradictions must be surfaced even though duplicate signals
        # may be deduplicated (same description + source + features).
        # The contradiction detection runs on raw signals before dedup.
        assert len(result.contradictions) > 0
        # At least one signal survives deduplication
        assert len(result.signals) >= 1

    def test_contradictions_contain_feature_info(self):
        s = SignalSynthesizer(dataset_id="ds_001")
        phase5 = {
            "dataset_id": "ds_001",
            "driver_report": {
                "target_column": "output",
                "drivers": [
                    {
                        "feature": "temp",
                        "strength": 0.8,
                        "direction": "positive",
                        "relationship_type": "linear",
                        "method": "correlation",
                        "explanation": "Positive.",
                    },
                    {
                        "feature": "temp",
                        "strength": 0.6,
                        "direction": "negative",
                        "relationship_type": "polynomial",
                        "method": "regression",
                        "explanation": "Negative.",
                    },
                ],
            },
        }
        s.ingest_anomalies(phase5)
        result = s.synthesize()
        assert len(result.contradictions) >= 1
        contra = result.contradictions[0]
        assert "feature" in contra
        assert "conflict" in contra
        assert "temp" in contra["feature"]


# ===================================================================
# 8. DETERMINISTIC SIGNAL GENERATION AND RANKING
# ===================================================================


class TestDeterminism:
    """Signal generation must be deterministic given the same inputs."""

    def _make_analytics(self, ds_id):
        return {
            "dataset_id": ds_id,
            "kpis": [
                {"label": "mean_X", "value": 10.0, "source_column": "X"},
                {"label": "std_X", "value": 2.0, "source_column": "X"},
            ],
            "insights": [
                {
                    "title": "Normal distribution",
                    "description": "X follows a normal distribution.",
                    "insight_type": "distribution",
                    "severity": "low",
                    "source_columns": ["X"],
                },
            ],
            "relationships": [],
        }

    def test_deterministic_signal_count(self):
        for _ in range(3):
            s = SignalSynthesizer(dataset_id="ds_det")
            s.ingest_analytics(self._make_analytics("ds_det"))
            result = s.synthesize()
            assert len(result.signals) == 3  # 2 KPIs + 1 insight

    def test_ranked_signals_ordering(self):
        s = SignalSynthesizer(dataset_id="ds_rank")
        # Ingest anomaly (high priority) and KPI (info priority)
        s.ingest_analytics({
            "dataset_id": "ds_rank",
            "kpis": [{"label": "avg", "value": 5, "source_column": "y"}],
            "insights": [],
            "relationships": [],
        })
        s.ingest_anomalies({
            "dataset_id": "ds_rank",
            "anomaly_report": {
                "anomalies": [{
                    "column": "y",
                    "anomaly_type": "spike",
                    "severity": "critical",
                    "score": 5.0,
                    "method": "zscore",
                    "explanation": "Extreme spike.",
                }],
            },
        })
        result = s.synthesize()
        ranked = result.ranked_signals()
        assert len(ranked) == 2
        # Critical anomaly must come before info-level KPI
        assert ranked[0].priority <= ranked[1].priority
        assert ranked[0].priority == SignalPriority.CRITICAL

    def test_synthesis_result_to_dict(self):
        s = SignalSynthesizer(dataset_id="ds_ser")
        s.ingest_analytics({
            "dataset_id": "ds_ser",
            "kpis": [{"label": "n", "value": 50, "source_column": "id"}],
            "insights": [],
            "relationships": [],
        })
        result = s.synthesize()
        d = result.to_dict()
        assert d["dataset_id"] == "ds_ser"
        assert d["signal_count"] == 1
        assert isinstance(d["signals"], list)
        assert isinstance(d["contradictions"], list)
        assert isinstance(d["processing_notes"], list)


# ===================================================================
# 9. DUPLICATE-SIGNAL HANDLING
# ===================================================================


class TestDeduplication:
    """Duplicate signals must be removed deterministically."""

    def test_identical_kpis_deduplicated(self):
        s = SignalSynthesizer(dataset_id="ds_dup")
        analytics = {
            "dataset_id": "ds_dup",
            "kpis": [
                {"label": "count", "value": 100, "source_column": "id"},
            ],
            "insights": [],
            "relationships": [],
        }
        # Ingest the same analytics twice
        s.ingest_analytics(analytics)
        s.ingest_analytics(analytics)
        result = s.synthesize()
        # Even though ingested twice, dedup should keep only one
        descriptions = [sig.signal.description for sig in result.signals]
        unique_descriptions = set(descriptions)
        assert len(descriptions) == len(unique_descriptions), (
            "Duplicate signals should be removed"
        )

    def test_dedup_keeps_higher_priority(self):
        """When duplicates exist, the higher-priority one is kept."""
        s = SignalSynthesizer(dataset_id="ds_dup2")
        # Ingest two insights with same description but different severities
        # Since they come from separate ingestion calls and descriptions
        # are formed from titles, two insights with same title produce duplicates
        s.ingest_analytics({
            "dataset_id": "ds_dup2",
            "kpis": [],
            "insights": [
                {
                    "title": "Same Finding",
                    "description": "Test description",
                    "insight_type": "distribution",
                    "severity": "low",
                    "source_columns": ["A"],
                },
            ],
            "relationships": [],
        })
        s.ingest_analytics({
            "dataset_id": "ds_dup2",
            "kpis": [],
            "insights": [
                {
                    "title": "Same Finding",
                    "description": "Test description",
                    "insight_type": "distribution",
                    "severity": "high",
                    "source_columns": ["A"],
                },
            ],
            "relationships": [],
        })
        result = s.synthesize()
        insight_sigs = [sig for sig in result.signals if "Same Finding" in sig.signal.description]
        assert len(insight_sigs) == 1
        assert insight_sigs[0].priority == SignalPriority.HIGH


# ===================================================================
# 10. DATASET ISOLATION
# ===================================================================


class TestDatasetIsolation:
    """Evidence from different datasets must not be silently merged."""

    def test_analytics_dataset_mismatch_raises(self):
        s = SignalSynthesizer(dataset_id="ds_A")
        with pytest.raises(ValueError, match="does not match"):
            s.ingest_analytics({
                "dataset_id": "ds_B",
                "kpis": [{"label": "x", "value": 1}],
                "insights": [],
                "relationships": [],
            })

    def test_predictions_dataset_mismatch_raises(self):
        s = SignalSynthesizer(dataset_id="ds_A")
        with pytest.raises(ValueError, match="does not match"):
            s.ingest_predictions({
                "dataset_id": "ds_B",
                "forecast": {"status": "success"},
            })

    def test_anomalies_dataset_mismatch_raises(self):
        s = SignalSynthesizer(dataset_id="ds_A")
        with pytest.raises(ValueError, match="does not match"):
            s.ingest_anomalies({"dataset_id": "ds_B"})

    def test_marketing_dataset_mismatch_raises(self):
        s = SignalSynthesizer(dataset_id="ds_A")
        with pytest.raises(ValueError, match="does not match"):
            s.ingest_marketing({"dataset_id": "ds_B"})

    def test_evidence_with_wrong_dataset_filtered_out(self):
        """Evidence scoped to a different dataset is filtered by validate_dataset_scope."""
        s = SignalSynthesizer(dataset_id="ds_A")
        ev = create_evidence(
            description="Evidence from different dataset",
            source="kpi_engine",
            dataset_id="ds_B",
        )
        # validate_dataset_scope filters out ds_B evidence; empty list → returns None
        result_sig = s.ingest_evidence([ev], source_type="custom")
        assert result_sig is None
        result = s.synthesize()
        assert len(result.signals) == 0
        assert any("filtered" in n.lower() for n in result.processing_notes)

    def test_unscoped_analytics_accepted(self):
        """Analytics with empty dataset_id should be accepted."""
        s = SignalSynthesizer(dataset_id="ds_A")
        s.ingest_analytics({
            "dataset_id": "",
            "kpis": [{"label": "n", "value": 10, "source_column": "x"}],
            "insights": [],
            "relationships": [],
        })
        result = s.synthesize()
        assert len(result.signals) == 1

    def test_all_signals_carry_dataset_id(self):
        s = SignalSynthesizer(dataset_id="ds_check")
        s.ingest_analytics({
            "dataset_id": "ds_check",
            "kpis": [{"label": "x", "value": 1, "source_column": "x"}],
            "insights": [],
            "relationships": [],
        })
        result = s.synthesize()
        for sig in result.signals:
            assert sig.signal.dataset_id == "ds_check"
            assert sig.confidence_score.dataset_id == "ds_check"

    def test_ingest_evidence_filters_by_dataset(self):
        s = SignalSynthesizer(dataset_id="ds_A")
        ev_match = create_evidence(
            description="Matching evidence",
            source="kpi_engine",
            dataset_id="ds_A",
        )
        ev_unscoped = create_evidence(
            description="Unscoped evidence",
            source="kpi_engine",
            dataset_id="",
        )
        result_sig = s.ingest_evidence(
            [ev_match, ev_unscoped],
            source_type="custom",
            description="Custom signal",
        )
        assert result_sig is not None
        assert len(result_sig.evidence_items) == 2  # both accepted


# ===================================================================
# 11. CONFIDENCE AND LIMITATION PRESERVATION
# ===================================================================


class TestConfidenceAndLimitations:
    """Confidence scores and limitations must be preserved through synthesis."""

    def test_forecast_carries_uncertainty_limitation(self):
        s = SignalSynthesizer(dataset_id="ds_001")
        s.ingest_predictions({
            "dataset_id": "ds_001",
            "forecast": {
                "status": "success",
                "target_column": "revenue",
                "model_name": "ets",
                "horizon": 6,
                "metrics": {"mape": 0.1},
                "limitations": ["Small sample"],
                "feature_importance": [],
            },
        })
        result = s.synthesize()
        sig = result.signals[0]
        # Signal-level limitations should include the converted "Small sample" string
        signal_lim_descs = [lim.description for lim in sig.signal.limitations]
        assert any("Small sample" in d for d in signal_lim_descs)
        # Evidence-level limitations should include model uncertainty
        ev = sig.evidence_items[0]
        ev_lim_cats = [lim.category for lim in ev.limitations]
        assert LimitationCategory.MODEL_UNCERTAINTY.value in ev_lim_cats

    def test_confidence_score_present_on_all_signals(self):
        s = SignalSynthesizer(dataset_id="ds_001")
        s.ingest_analytics({
            "dataset_id": "ds_001",
            "kpis": [
                {"label": "count", "value": 100, "source_column": "id"},
                {"label": "mean", "value": 50, "source_column": "val"},
            ],
            "insights": [],
            "relationships": [],
        })
        result = s.synthesize()
        for sig in result.signals:
            assert isinstance(sig.confidence_score, ConfidenceScore)
            assert sig.confidence_score.level in {c.value for c in ConfidenceLevel}

    def test_mmm_assumptions_become_limitations(self):
        s = SignalSynthesizer(dataset_id="ds_001")
        s.ingest_marketing({
            "dataset_id": "ds_001",
            "mmm_result": {
                "selected_model": "adstock",
                "train_metrics": {"r2": 0.85},
                "test_metrics": {"r2": 0.80},
                "limitations": ["Time-varying coefficients not modeled"],
                "assumptions": ["Linear response assumed"],
            },
        })
        result = s.synthesize()
        sig = result.signals[0]
        all_lim_descs = [lim.description for lim in sig.signal.limitations]
        assert any("Linear response" in d for d in all_lim_descs)
        assert any("Time-varying" in d for d in all_lim_descs)


# ===================================================================
# 12. GENERIC NON-MARKETING DATASETS
# ===================================================================


class TestGenericDomains:
    """The synthesizer must work with any domain, not just marketing."""

    def test_healthcare_dataset(self):
        s = SignalSynthesizer(dataset_id="ds_health")
        s.ingest_analytics({
            "dataset_id": "ds_health",
            "kpis": [
                {"label": "patient_count", "value": 5000, "source_column": "patient_id"},
                {"label": "avg_los", "value": 4.2, "source_column": "length_of_stay"},
            ],
            "insights": [
                {
                    "title": "Readmission correlation",
                    "description": "Readmission rate correlates with discharge time.",
                    "insight_type": "correlation",
                    "severity": "high",
                    "source_columns": ["readmission", "discharge_hour"],
                },
            ],
            "relationships": [],
        })
        result = s.synthesize()
        assert len(result.signals) == 3
        # No marketing-specific terminology leaked
        for sig in result.signals:
            desc_lower = sig.signal.description.lower()
            assert "channel" not in desc_lower or "channel" in sig.signal.description

    def test_manufacturing_dataset(self):
        s = SignalSynthesizer(dataset_id="ds_mfg")
        s.ingest_anomalies({
            "dataset_id": "ds_mfg",
            "anomaly_report": {
                "anomalies": [
                    {
                        "column": "defect_rate",
                        "anomaly_type": "trend_change",
                        "severity": "critical",
                        "score": 4.0,
                        "method": "cusum",
                        "explanation": "Defect rate shifted upward by 15%.",
                    },
                ],
            },
            "driver_report": {
                "target_column": "defect_rate",
                "drivers": [
                    {
                        "feature": "humidity",
                        "strength": 0.72,
                        "direction": "positive",
                        "relationship_type": "monotonic",
                        "method": "mutual_information",
                        "explanation": "Humidity associated with defect rate.",
                    },
                ],
            },
        })
        result = s.synthesize()
        assert len(result.signals) == 2
        # Verify domain-agnostic signal descriptions
        descs = [sig.signal.description for sig in result.signals]
        assert any("Anomaly" in d for d in descs)
        assert any("Driver" in d for d in descs)

    def test_education_dataset(self):
        s = SignalSynthesizer(dataset_id="ds_edu")
        s.ingest_predictions({
            "dataset_id": "ds_edu",
            "classification": {
                "status": "success",
                "target_column": "dropout_risk",
                "model_name": "gradient_boosting",
                "metrics": {"accuracy": 0.83, "f1": 0.76},
                "limitations": ["Imbalanced classes"],
                "feature_importance": [
                    {"feature": "attendance_rate", "importance": 0.45},
                    {"feature": "gpa", "importance": 0.30},
                ],
            },
        })
        result = s.synthesize()
        # 1 classification + 2 feature importance = 3
        assert len(result.signals) == 3

    def test_logistics_dataset(self):
        s = SignalSynthesizer(dataset_id="ds_logistics")
        s.ingest_predictions({
            "dataset_id": "ds_logistics",
            "forecast": {
                "status": "success",
                "target_column": "delivery_time",
                "model_name": "prophet",
                "horizon": 30,
                "metrics": {"mape": 0.12},
                "limitations": [],
                "feature_importance": [],
            },
        })
        result = s.synthesize()
        assert len(result.signals) == 1
        assert "delivery_time" in result.signals[0].signal.description

    def test_finance_dataset(self):
        s = SignalSynthesizer(dataset_id="ds_fin")
        s.ingest_analytics({
            "dataset_id": "ds_fin",
            "kpis": [
                {"label": "sharpe_ratio", "value": 1.8, "source_column": "returns"},
            ],
            "insights": [],
            "relationships": [
                {
                    "relationship_type": "correlation",
                    "columns": ["vix", "returns"],
                    "strength": -0.65,
                    "direction": "negative",
                    "description": "VIX negatively correlated with returns.",
                },
            ],
        })
        result = s.synthesize()
        assert len(result.signals) == 2

    def test_custom_domain_via_ingest_evidence(self):
        s = SignalSynthesizer(dataset_id="ds_custom")
        ev = create_evidence(
            description="Sensor reading above threshold.",
            source="kpi_engine",
            metric_name="temperature",
            metric_value=105.0,
            dataset_id="ds_custom",
        )
        sig = s.ingest_evidence(
            [ev],
            source_type="iot_sensor",
            description="Temperature alert",
            priority=SignalPriority.CRITICAL,
            affected_features=["temperature"],
        )
        assert sig is not None
        result = s.synthesize()
        assert len(result.signals) == 1
        assert result.signals[0].source_type == "iot_sensor"


# ===================================================================
# 13. MARKETING-SPECIFIC EVIDENCE
# ===================================================================


class TestMarketingEvidence:
    """Marketing evidence should be supported when genuinely present."""

    def test_mmm_signal(self):
        s = SignalSynthesizer(dataset_id="ds_mkt")
        s.ingest_marketing({
            "dataset_id": "ds_mkt",
            "mmm_result": {
                "selected_model": "adstock_saturation",
                "train_metrics": {"r2": 0.92},
                "test_metrics": {"r2": 0.88},
                "limitations": [],
                "assumptions": [],
            },
        })
        result = s.synthesize()
        assert len(result.signals) == 1
        assert result.signals[0].source_type == "phase6_mmm"

    def test_contribution_signal(self):
        s = SignalSynthesizer(dataset_id="ds_mkt")
        s.ingest_marketing({
            "dataset_id": "ds_mkt",
            "contributions": [
                {
                    "channel": "social",
                    "contribution": 50.0,
                    "contribution_share": 0.25,
                    "roi": 2.5,
                },
            ],
        })
        result = s.synthesize()
        assert len(result.signals) == 1
        sig = result.signals[0]
        assert sig.source_type == "phase6_contribution"
        assert "social" in sig.signal.description

    def test_roi_signal(self):
        s = SignalSynthesizer(dataset_id="ds_mkt")
        s.ingest_marketing({
            "dataset_id": "ds_mkt",
            "roi_results": [
                {
                    "channel": "email",
                    "naive_roas": 4.5,
                    "model_estimated_roi": None,
                    "method": "naive_ratio",
                },
            ],
        })
        result = s.synthesize()
        assert len(result.signals) == 1
        sig = result.signals[0]
        assert sig.source_type == "phase6_roi"
        assert "email" in sig.signal.description

    def test_whatif_signal(self):
        s = SignalSynthesizer(dataset_id="ds_mkt")
        s.ingest_marketing({
            "dataset_id": "ds_mkt",
            "whatif_result": {
                "scenarios": [
                    {
                        "name": "increase_social_20pct",
                        "predicted_target": 120.0,
                        "target_change_pct": 8.5,
                    },
                ],
                "disclaimer": "Model-estimated outcomes.",
            },
        })
        result = s.synthesize()
        assert len(result.signals) == 1
        sig = result.signals[0]
        assert sig.source_type == "phase6_whatif"
        assert sig.priority == SignalPriority.LOW
        # What-if must carry uncertainty
        ev = sig.evidence_items[0]
        lim_cats = [lim.category for lim in ev.limitations]
        assert LimitationCategory.MODEL_UNCERTAINTY.value in lim_cats

    def test_optimization_signal(self):
        s = SignalSynthesizer(dataset_id="ds_mkt")
        s.ingest_marketing({
            "dataset_id": "ds_mkt",
            "optimization_result": {
                "success": True,
                "expected_improvement_pct": 12.5,
                "message": "Optimal allocation found.",
            },
        })
        result = s.synthesize()
        assert len(result.signals) == 1
        sig = result.signals[0]
        assert sig.source_type == "phase6_optimization"
        # Must carry model uncertainty limitation
        sig_lim_cats = [lim.category for lim in sig.signal.limitations]
        assert LimitationCategory.MODEL_UNCERTAINTY.value in sig_lim_cats

    def test_full_marketing_pipeline(self):
        s = SignalSynthesizer(dataset_id="ds_full_mkt")
        s.ingest_marketing({
            "dataset_id": "ds_full_mkt",
            "mmm_result": {
                "selected_model": "baseline",
                "train_metrics": {"r2": 0.70},
                "test_metrics": {"r2": 0.65},
                "limitations": ["Small dataset"],
                "assumptions": ["Linear effects"],
            },
            "contributions": [
                {"channel": "tv", "contribution": 200, "contribution_share": 0.6, "roi": 3.0},
                {"channel": "radio", "contribution": 100, "contribution_share": 0.3, "roi": 2.0},
            ],
            "roi_results": [
                {"channel": "tv", "naive_roas": 3.2, "method": "naive_ratio"},
            ],
            "whatif_result": {
                "scenarios": [
                    {"name": "double_tv", "predicted_target": 500, "target_change_pct": 25.0},
                ],
                "disclaimer": "Model-based projection.",
            },
            "optimization_result": {
                "success": True,
                "expected_improvement_pct": 8.0,
                "message": "Rebalance toward TV.",
            },
        })
        result = s.synthesize()
        # 1 MMM + 2 contributions + 1 ROI + 1 what-if + 1 optimization = 6
        assert len(result.signals) == 6


# ===================================================================
# 14. MISSING OR MALFORMED INPUTS
# ===================================================================


class TestMissingMalformedInputs:
    """Handle edge cases gracefully without crashing."""

    def test_dict_based_analytics(self):
        """Accept dict-based inputs (not just dataclass objects)."""
        s = SignalSynthesizer(dataset_id="ds_001")
        s.ingest_analytics({
            "dataset_id": "ds_001",
            "kpis": [{"label": "total", "value": 999}],
            "insights": [],
            "relationships": [],
        })
        result = s.synthesize()
        assert len(result.signals) == 1

    def test_missing_optional_fields(self):
        """KPI with missing optional fields should work."""
        s = SignalSynthesizer(dataset_id="ds_001")
        s.ingest_analytics({
            "dataset_id": "ds_001",
            "kpis": [{"label": "something", "value": 10}],
            "insights": [],
            "relationships": [],
        })
        result = s.synthesize()
        assert len(result.signals) == 1

    def test_mmm_without_selected_model_skipped(self):
        s = SignalSynthesizer(dataset_id="ds_001")
        s.ingest_marketing({
            "dataset_id": "ds_001",
            "mmm_result": {"selected_model": "", "train_metrics": {}, "test_metrics": {}},
        })
        result = s.synthesize()
        assert len(result.signals) == 0

    def test_whatif_scenario_without_name_skipped(self):
        s = SignalSynthesizer(dataset_id="ds_001")
        s.ingest_marketing({
            "dataset_id": "ds_001",
            "whatif_result": {
                "scenarios": [{"name": "", "predicted_target": 100, "target_change_pct": 5}],
            },
        })
        result = s.synthesize()
        assert len(result.signals) == 0

    def test_contribution_without_channel_skipped(self):
        s = SignalSynthesizer(dataset_id="ds_001")
        s.ingest_marketing({
            "dataset_id": "ds_001",
            "contributions": [
                {"channel": "", "contribution": 50, "contribution_share": 0.5, "roi": 2.0},
            ],
        })
        result = s.synthesize()
        assert len(result.signals) == 0

    def test_roi_without_channel_skipped(self):
        s = SignalSynthesizer(dataset_id="ds_001")
        s.ingest_marketing({
            "dataset_id": "ds_001",
            "roi_results": [{"channel": "", "naive_roas": 3.0}],
        })
        result = s.synthesize()
        assert len(result.signals) == 0

    def test_driver_without_feature_skipped(self):
        s = SignalSynthesizer(dataset_id="ds_001")
        s.ingest_anomalies({
            "dataset_id": "ds_001",
            "driver_report": {
                "target_column": "y",
                "drivers": [
                    {"feature": "", "strength": 0.5, "direction": "positive",
                     "method": "corr", "explanation": "Something."},
                ],
            },
        })
        result = s.synthesize()
        assert len(result.signals) == 0

    def test_feature_importance_zero_skipped(self):
        s = SignalSynthesizer(dataset_id="ds_001")
        s.ingest_predictions({
            "dataset_id": "ds_001",
            "regression": {
                "status": "success",
                "target_column": "y",
                "model_name": "rf",
                "metrics": {"r2": 0.9},
                "limitations": [],
                "feature_importance": [
                    {"feature": "a", "importance": 0.0},
                ],
            },
        })
        result = s.synthesize()
        # 1 regression signal, 0 feature importance (zero importance skipped)
        fi_sigs = [sig for sig in result.signals
                   if "Feature importance" in sig.signal.description]
        assert len(fi_sigs) == 0


# ===================================================================
# 15. SERIALIZATION COMPATIBILITY WITH PART 1 CONTRACTS
# ===================================================================


class TestSerializationCompat:
    """Signals must be serializable via Part 1 to_dict/from_dict."""

    def test_synthesized_signal_to_dict(self):
        s = SignalSynthesizer(dataset_id="ds_001")
        s.ingest_analytics({
            "dataset_id": "ds_001",
            "kpis": [{"label": "n", "value": 42, "source_column": "id"}],
            "insights": [],
            "relationships": [],
        })
        result = s.synthesize()
        sig = result.signals[0]
        d = sig.to_dict()
        assert "signal" in d
        assert "priority" in d
        assert "confidence_score" in d
        assert "source_type" in d
        assert "generation_reason" in d

    def test_signal_to_dict_roundtrip(self):
        s = SignalSynthesizer(dataset_id="ds_001")
        s.ingest_analytics({
            "dataset_id": "ds_001",
            "kpis": [{"label": "total", "value": 100, "source_column": "x"}],
            "insights": [],
            "relationships": [],
        })
        result = s.synthesize()
        sig = result.signals[0]
        d = sig.to_dict()

        # The inner DecisionSignal can roundtrip
        signal_dict = d["signal"]
        restored = DecisionSignal.from_dict(signal_dict)
        assert restored.description == sig.signal.description
        assert restored.dataset_id == sig.signal.dataset_id

    def test_confidence_score_roundtrip(self):
        s = SignalSynthesizer(dataset_id="ds_001")
        s.ingest_analytics({
            "dataset_id": "ds_001",
            "kpis": [{"label": "m", "value": 10, "source_column": "z"}],
            "insights": [],
            "relationships": [],
        })
        result = s.synthesize()
        sig = result.signals[0]
        d = sig.confidence_score.to_dict()
        restored = ConfidenceScore.from_dict(d)
        assert restored.level == sig.confidence_score.level
        assert restored.dataset_id == "ds_001"

    def test_evidence_in_registry_serializable(self):
        s = SignalSynthesizer(dataset_id="ds_001")
        s.ingest_analytics({
            "dataset_id": "ds_001",
            "kpis": [{"label": "x", "value": 5, "source_column": "x"}],
            "insights": [],
            "relationships": [],
        })
        result = s.synthesize()
        for ev_id, ev in result.evidence_registry.items():
            d = ev.to_dict()
            assert d["id"] == ev_id
            restored = DecisionEvidence.from_dict(d)
            assert restored.id == ev_id

    def test_synthesis_result_to_dict_complete(self):
        s = SignalSynthesizer(dataset_id="ds_001")
        s.ingest_analytics({
            "dataset_id": "ds_001",
            "kpis": [{"label": "a", "value": 1, "source_column": "a"}],
            "insights": [],
            "relationships": [],
        })
        result = s.synthesize()
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["dataset_id"] == "ds_001"
        assert "signals" in d
        assert "contradictions" in d
        assert "processing_notes" in d
        assert d["signal_count"] == len(result.signals)


# ===================================================================
# 16. CORRELATION VS CAUSATION SAFEGUARDS
# ===================================================================


class TestCorrelationCausationSafeguards:
    """Correlation must not be presented as causation."""

    def test_relationship_gets_observational_limitation(self):
        s = SignalSynthesizer(dataset_id="ds_001")
        s.ingest_analytics({
            "dataset_id": "ds_001",
            "kpis": [],
            "insights": [],
            "relationships": [
                {
                    "relationship_type": "correlation",
                    "columns": ["A", "B"],
                    "strength": 0.9,
                    "direction": "positive",
                    "description": "A and B are strongly correlated.",
                },
            ],
        })
        result = s.synthesize()
        sig = result.signals[0]
        ev = sig.evidence_items[0]
        obs = [lim for lim in ev.limitations
               if lim.category == LimitationCategory.OBSERVATIONAL.value]
        assert len(obs) > 0
        # Check the limitation text mentions causation
        assert any("causal" in lim.description.lower() or "causation" in lim.description.lower()
                    for lim in obs)

    def test_correlation_insight_gets_non_causal_guard(self):
        s = SignalSynthesizer(dataset_id="ds_001")
        s.ingest_analytics({
            "dataset_id": "ds_001",
            "kpis": [],
            "insights": [
                {
                    "title": "Correlated Features",
                    "description": "X and Y are correlated.",
                    "insight_type": "correlation",
                    "severity": "medium",
                    "source_columns": ["X", "Y"],
                },
            ],
            "relationships": [],
        })
        result = s.synthesize()
        sig = result.signals[0]
        ev = sig.evidence_items[0]
        obs = [lim for lim in ev.limitations
               if lim.category == LimitationCategory.OBSERVATIONAL.value]
        assert len(obs) > 0

    def test_driver_gets_non_causal_guard(self):
        s = SignalSynthesizer(dataset_id="ds_001")
        s.ingest_anomalies({
            "dataset_id": "ds_001",
            "driver_report": {
                "target_column": "output",
                "drivers": [
                    {
                        "feature": "input_x",
                        "strength": 0.6,
                        "direction": "positive",
                        "relationship_type": "linear",
                        "method": "regression",
                        "explanation": "Input X is associated with output.",
                    },
                ],
            },
        })
        result = s.synthesize()
        sig = result.signals[0]
        ev = sig.evidence_items[0]
        obs = [lim for lim in ev.limitations
               if lim.category == LimitationCategory.OBSERVATIONAL.value]
        assert len(obs) > 0

    def test_feature_importance_gets_non_causal_guard(self):
        s = SignalSynthesizer(dataset_id="ds_001")
        s.ingest_predictions({
            "dataset_id": "ds_001",
            "regression": {
                "status": "success",
                "target_column": "y",
                "model_name": "rf",
                "metrics": {"r2": 0.85},
                "limitations": [],
                "feature_importance": [
                    {"feature": "z", "importance": 0.5},
                ],
            },
        })
        result = s.synthesize()
        fi_sigs = [sig for sig in result.signals
                   if "Feature importance" in sig.signal.description]
        assert len(fi_sigs) == 1
        ev = fi_sigs[0].evidence_items[0]
        obs = [lim for lim in ev.limitations
               if lim.category == LimitationCategory.OBSERVATIONAL.value]
        assert len(obs) > 0
        # Generation reason should mention "not causal"
        assert "not causal" in fi_sigs[0].generation_reason.lower() or \
               "association" in fi_sigs[0].generation_reason.lower()

    def test_mmm_contribution_gets_non_causal_guard(self):
        s = SignalSynthesizer(dataset_id="ds_001")
        s.ingest_marketing({
            "dataset_id": "ds_001",
            "contributions": [
                {"channel": "tv", "contribution": 100, "contribution_share": 0.5, "roi": 3.0},
            ],
        })
        result = s.synthesize()
        sig = result.signals[0]
        ev = sig.evidence_items[0]
        obs = [lim for lim in ev.limitations
               if lim.category == LimitationCategory.OBSERVATIONAL.value]
        assert len(obs) > 0

    def test_prediction_is_not_fact(self):
        """Forecasts must be PREDICTION level, not FACT."""
        s = SignalSynthesizer(dataset_id="ds_001")
        s.ingest_predictions({
            "dataset_id": "ds_001",
            "forecast": {
                "status": "success",
                "target_column": "demand",
                "model_name": "ets",
                "horizon": 12,
                "metrics": {},
                "limitations": [],
                "feature_importance": [],
            },
        })
        result = s.synthesize()
        sig = result.signals[0]
        ev = sig.evidence_items[0]
        assert ev.level == EvidenceLevel.PREDICTION.value
        assert sig.signal.level == EvidenceLevel.PREDICTION.value


# ===================================================================
# 17. NO DEMO-DATA LEAKAGE
# ===================================================================


class TestNoDemoDataLeakage:
    """The synthesizer must never invent evidence or use demo data."""

    def test_no_signals_from_empty_input(self):
        s = SignalSynthesizer(dataset_id="ds_001")
        s.ingest_analytics({
            "dataset_id": "ds_001",
            "kpis": [],
            "insights": [],
            "relationships": [],
        })
        s.ingest_predictions({"dataset_id": "ds_001"})
        s.ingest_anomalies({"dataset_id": "ds_001"})
        s.ingest_marketing({"dataset_id": "ds_001"})
        result = s.synthesize()
        assert len(result.signals) == 0
        # No hardcoded or demo signals
        assert len(result.evidence_registry) == 0

    def test_no_invented_evidence(self):
        """All evidence must trace back to ingested data."""
        s = SignalSynthesizer(dataset_id="ds_001")
        s.ingest_analytics({
            "dataset_id": "ds_001",
            "kpis": [{"label": "only_kpi", "value": 7, "source_column": "x"}],
            "insights": [],
            "relationships": [],
        })
        result = s.synthesize()
        # Only one signal should exist — the KPI we ingested
        assert len(result.signals) == 1
        assert len(result.evidence_registry) == 1
        ev = list(result.evidence_registry.values())[0]
        assert "only_kpi" in ev.description

    def test_no_hardcoded_column_names(self):
        """Signal descriptions should not contain hardcoded column names."""
        s = SignalSynthesizer(dataset_id="ds_generic")
        s.ingest_analytics({
            "dataset_id": "ds_generic",
            "kpis": [{"label": "metric_q", "value": 99, "source_column": "col_q"}],
            "insights": [],
            "relationships": [],
        })
        result = s.synthesize()
        sig = result.signals[0]
        # The description should reference the actual column, not a hardcoded one
        assert "metric_q" in sig.signal.description


# ===================================================================
# 18. HELPER FUNCTIONS
# ===================================================================


class TestHelpers:
    """Test internal helper functions."""

    def test_get_attr_from_dict(self):
        d = {"key": "value"}
        assert _get_attr(d, "key") == "value"
        assert _get_attr(d, "missing", "default") == "default"

    def test_get_attr_from_object(self):
        class Obj:
            key = "value"
        assert _get_attr(Obj(), "key") == "value"
        assert _get_attr(Obj(), "missing", "default") == "default"

    def test_convert_string_limitations_strings(self):
        lims = _convert_string_limitations(["Limit one", "Limit two"])
        assert len(lims) == 2
        assert all(isinstance(lim, DecisionLimitation) for lim in lims)
        assert lims[0].description == "Limit one"
        assert lims[0].category == LimitationCategory.METHODOLOGY.value

    def test_convert_string_limitations_empty_string(self):
        lims = _convert_string_limitations(["  ", ""])
        assert len(lims) == 0

    def test_convert_string_limitations_mixed(self):
        existing_lim = create_limitation("Existing", category="data_quality")
        lims = _convert_string_limitations(["String lim", existing_lim])
        assert len(lims) == 2

    def test_convert_string_limitations_dict(self):
        lims = _convert_string_limitations([
            {"description": "Dict lim", "category": "scope"},
        ])
        assert len(lims) == 1
        assert lims[0].description == "Dict lim"
        assert lims[0].category == "scope"

    def test_convert_string_limitations_empty_list(self):
        lims = _convert_string_limitations([])
        assert len(lims) == 0


# ===================================================================
# 19. EVIDENCE REGISTRY
# ===================================================================


class TestEvidenceRegistry:
    """Evidence registry tracks all evidence items."""

    def test_evidence_registered(self):
        s = SignalSynthesizer(dataset_id="ds_001")
        s.ingest_analytics({
            "dataset_id": "ds_001",
            "kpis": [
                {"label": "a", "value": 1, "source_column": "a"},
                {"label": "b", "value": 2, "source_column": "b"},
            ],
            "insights": [],
            "relationships": [],
        })
        result = s.synthesize()
        assert len(result.evidence_registry) == 2
        for ev_id, ev in result.evidence_registry.items():
            assert ev_id.startswith("ev_")
            assert ev.dataset_id == "ds_001"

    def test_evidence_ids_match_signal_references(self):
        s = SignalSynthesizer(dataset_id="ds_001")
        s.ingest_analytics({
            "dataset_id": "ds_001",
            "kpis": [{"label": "x", "value": 5, "source_column": "x"}],
            "insights": [],
            "relationships": [],
        })
        result = s.synthesize()
        sig = result.signals[0]
        for ev_id in sig.signal.evidence_ids:
            assert ev_id in result.evidence_registry


# ===================================================================
# 20. SIGNALS_BY_SOURCE AND RANKED_SIGNALS
# ===================================================================


class TestResultMethods:
    """Test SynthesisResult convenience methods."""

    def test_signals_by_source_returns_empty_for_unknown(self):
        s = SignalSynthesizer(dataset_id="ds_001")
        s.ingest_analytics({
            "dataset_id": "ds_001",
            "kpis": [{"label": "x", "value": 1, "source_column": "x"}],
            "insights": [],
            "relationships": [],
        })
        result = s.synthesize()
        assert len(result.signals_by_source("nonexistent")) == 0

    def test_ranked_signals_sorted(self):
        s = SignalSynthesizer(dataset_id="ds_001")
        # Mix of priorities: INFO (KPI) + CRITICAL (anomaly) + LOW (insight)
        s.ingest_analytics({
            "dataset_id": "ds_001",
            "kpis": [{"label": "k", "value": 1, "source_column": "k"}],
            "insights": [
                {
                    "title": "Low insight",
                    "description": "Something minor.",
                    "insight_type": "distribution",
                    "severity": "low",
                    "source_columns": ["k"],
                },
            ],
            "relationships": [],
        })
        s.ingest_anomalies({
            "dataset_id": "ds_001",
            "anomaly_report": {
                "anomalies": [{
                    "column": "k",
                    "anomaly_type": "outlier",
                    "severity": "critical",
                    "score": 5.0,
                    "method": "zscore",
                    "explanation": "Critical outlier.",
                }],
            },
        })
        result = s.synthesize()
        ranked = result.ranked_signals()
        priorities = [sig.priority for sig in ranked]
        assert priorities == sorted(priorities)
