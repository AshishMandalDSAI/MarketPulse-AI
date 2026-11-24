"""Phase 7 Part 2 — Signal Synthesizer.

Converts analytical results from Phase 3–6 into structured, domain-agnostic
DecisionSignal instances using Part 1 contracts.

Design principles:
1. Deterministic — no LLM calls, no randomness in signal generation.
2. Domain-agnostic — works with any dataset, not hardcoded to marketing.
3. Safe — enforces correlation/causation distinction, prediction uncertainty,
   and evidence-level classification.
4. Transparent — every signal explains why it was generated.
5. Dataset-isolated — signals never silently merge evidence across datasets.
6. No global mutable caches. No demo-data fallback.
"""

from __future__ import annotations

from typing import Any, Optional

from src.decision.contracts import (
    ConfidenceLevel,
    ConfidenceScore,
    DecisionEvidence,
    DecisionLimitation,
    DecisionSignal,
    EvidenceLevel,
    LimitationCategory,
    _make_id,
)
from src.decision.confidence import assess_confidence
from src.decision.evidence import (
    create_evidence,
    create_limitation,
    create_signal,
    ensure_non_causal,
    ensure_prediction_uncertainty,
    validate_dataset_scope,
)


# ---------------------------------------------------------------------------
# Signal Priority — transparent ranking
# ---------------------------------------------------------------------------

class SignalPriority:
    """Transparent, deterministic signal priority levels.

    Priority is distinct from statistical confidence:
    - Priority = how urgently a finding should be reviewed.
    - Confidence = how well-supported the finding is by evidence.

    A high-priority signal may have low confidence (e.g. a critical anomaly
    detected with limited data). A low-priority signal may have high
    confidence (e.g. a well-established but minor KPI observation).
    """
    CRITICAL = 1    # Requires immediate attention (anomalies, data quality)
    HIGH = 2        # Important finding, should be reviewed
    MEDIUM = 3      # Notable finding, useful context
    LOW = 4         # Minor finding, informational
    INFO = 5        # Background context

    @staticmethod
    def from_severity(severity: str) -> int:
        """Map severity strings to priority levels."""
        _map = {
            "critical": SignalPriority.CRITICAL,
            "high": SignalPriority.HIGH,
            "medium": SignalPriority.MEDIUM,
            "low": SignalPriority.LOW,
            "info": SignalPriority.INFO,
        }
        return _map.get(severity.lower().strip(), SignalPriority.MEDIUM)


# ---------------------------------------------------------------------------
# SynthesizedSignal — wraps DecisionSignal with priority + confidence
# ---------------------------------------------------------------------------

class SynthesizedSignal:
    """A signal produced by the synthesizer, with priority and confidence.

    Wraps a DecisionSignal with additional synthesis metadata:
    - priority: how urgently the signal should be reviewed (1=critical..5=info)
    - confidence_score: deterministic ConfidenceScore from Part 1
    - source_type: which Phase produced this signal
    - generation_reason: why this signal was generated
    """

    __slots__ = (
        "signal", "priority", "confidence_score",
        "source_type", "generation_reason", "evidence_items",
    )

    def __init__(
        self,
        signal: DecisionSignal,
        priority: int,
        confidence_score: ConfidenceScore,
        source_type: str,
        generation_reason: str,
        evidence_items: list,
    ):
        self.signal = signal
        self.priority = priority
        self.confidence_score = confidence_score
        self.source_type = source_type
        self.generation_reason = generation_reason
        self.evidence_items = evidence_items

    def to_dict(self) -> dict:
        return {
            "signal": self.signal.to_dict(),
            "priority": self.priority,
            "confidence_score": self.confidence_score.to_dict(),
            "source_type": self.source_type,
            "generation_reason": self.generation_reason,
        }


# ---------------------------------------------------------------------------
# SynthesisResult — aggregate output
# ---------------------------------------------------------------------------

class SynthesisResult:
    """Result of synthesizing signals from analytical results.

    Contains all synthesized signals plus metadata about what was processed.
    """

    __slots__ = (
        "dataset_id", "signals", "evidence_registry",
        "contradictions", "processing_notes",
    )

    def __init__(
        self,
        dataset_id: str,
        signals: list,
        evidence_registry: dict,
        contradictions: list,
        processing_notes: list,
    ):
        self.dataset_id = dataset_id
        self.signals = signals                    # list[SynthesizedSignal]
        self.evidence_registry = evidence_registry  # dict[str, DecisionEvidence]
        self.contradictions = contradictions        # list[dict] — surfaced conflicts
        self.processing_notes = processing_notes    # list[str]

    def ranked_signals(self) -> list:
        """Return signals sorted by priority (1=most urgent first)."""
        return sorted(self.signals, key=lambda s: s.priority)

    def signals_by_source(self, source_type: str) -> list:
        """Return signals from a specific source phase."""
        return [s for s in self.signals if s.source_type == source_type]

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "signals": [s.to_dict() for s in self.signals],
            "contradictions": list(self.contradictions),
            "processing_notes": list(self.processing_notes),
            "signal_count": len(self.signals),
        }


# ---------------------------------------------------------------------------
# Signal Synthesizer
# ---------------------------------------------------------------------------

class SignalSynthesizer:
    """Converts Phase 3–6 analytical results into structured DecisionSignals.

    Usage:
        synthesizer = SignalSynthesizer(dataset_id="ds_001")
        synthesizer.ingest_analytics(analytics_result)
        synthesizer.ingest_predictions(prediction_report)
        synthesizer.ingest_anomalies(phase5_result)
        synthesizer.ingest_marketing(phase6_result)
        result = synthesizer.synthesize()
    """

    def __init__(self, dataset_id: str = ""):
        if not dataset_id:
            raise ValueError(
                "dataset_id is required. Evidence must be scoped to a dataset."
            )
        self._dataset_id = dataset_id
        self._evidence_registry: dict[str, DecisionEvidence] = {}
        self._raw_signals: list[SynthesizedSignal] = []
        self._contradictions: list[dict] = []
        self._processing_notes: list[str] = []
        self._ingested_sources: set[str] = set()

    @property
    def dataset_id(self) -> str:
        return self._dataset_id

    # -------------------------------------------------------------------
    # Evidence registry
    # -------------------------------------------------------------------

    def _register_evidence(self, evidence: DecisionEvidence) -> DecisionEvidence:
        """Register evidence, enforcing dataset isolation."""
        if evidence.dataset_id and evidence.dataset_id != self._dataset_id:
            raise ValueError(
                f"Evidence dataset_id '{evidence.dataset_id}' does not match "
                f"synthesizer dataset_id '{self._dataset_id}'. "
                f"Cross-dataset evidence must not be silently merged."
            )
        self._evidence_registry[evidence.id] = evidence
        return evidence

    def _add_signal(
        self,
        description: str,
        evidence_items: list,
        *,
        priority: int = SignalPriority.MEDIUM,
        source_type: str = "",
        generation_reason: str = "",
        affected_features: list | None = None,
        limitations: list | None = None,
    ) -> SynthesizedSignal:
        """Create and register a synthesized signal."""
        if not evidence_items:
            return None  # type: ignore — caller checks

        # Register all evidence
        registered = []
        for ev in evidence_items:
            registered.append(self._register_evidence(ev))

        # Create the decision signal via Part 1 helpers
        signal = create_signal(
            description,
            registered,
            affected_features=affected_features,
            limitations=limitations,
            dataset_id=self._dataset_id,
        )

        # Assess confidence
        confidence = assess_confidence(
            registered,
            dataset_id=self._dataset_id,
        )

        synth = SynthesizedSignal(
            signal=signal,
            priority=priority,
            confidence_score=confidence,
            source_type=source_type,
            generation_reason=generation_reason,
            evidence_items=registered,
        )
        self._raw_signals.append(synth)
        return synth

    # -------------------------------------------------------------------
    # Phase 3: Analytics (KPIs, Insights, Relationships)
    # -------------------------------------------------------------------

    def ingest_analytics(self, analytics_result) -> None:
        """Ingest Phase 3 AnalyticsResult.

        Accepts an AnalyticsResult object or dict with:
        - dataset_id, kpis, insights, relationships
        """
        if analytics_result is None:
            self._processing_notes.append(
                "Phase 3 analytics: no result provided."
            )
            return

        ds_id = _get_attr(analytics_result, "dataset_id", "")
        if ds_id and ds_id != self._dataset_id:
            raise ValueError(
                f"AnalyticsResult dataset_id '{ds_id}' does not match "
                f"synthesizer dataset_id '{self._dataset_id}'."
            )

        self._ingested_sources.add("phase3_analytics")

        # --- KPIs ---
        kpis = _get_attr(analytics_result, "kpis", [])
        for kpi in kpis:
            label = _get_attr(kpi, "label", "")
            value = _get_attr(kpi, "value", None)
            source_col = _get_attr(kpi, "source_column", "")
            explanation = _get_attr(kpi, "explanation", "")
            category = _get_attr(kpi, "category", "summary")

            if not label:
                continue

            ev = create_evidence(
                description=f"KPI '{label}': {value}" + (f" ({explanation})" if explanation else ""),
                source="kpi_engine",
                metric_name=label,
                metric_value=value,
                dataset_id=self._dataset_id,
                details={"source_column": source_col, "category": category},
            )

            self._add_signal(
                description=f"KPI observation: {label} = {value}",
                evidence_items=[ev],
                priority=SignalPriority.INFO,
                source_type="phase3_kpi",
                generation_reason=f"KPI '{label}' computed from dataset.",
                affected_features=[source_col] if source_col else [],
            )

        # --- Insights ---
        insights = _get_attr(analytics_result, "insights", [])
        for insight in insights:
            title = _get_attr(insight, "title", "")
            desc = _get_attr(insight, "description", "")
            insight_type = _get_attr(insight, "insight_type", "")
            severity = _get_attr(insight, "severity", "info")
            evidence_text = _get_attr(insight, "evidence", "")
            source_cols = _get_attr(insight, "source_columns", [])
            metric_val = _get_attr(insight, "metric_value", None)

            if not title and not desc:
                continue

            # Insights from correlation analysis are INFERENCE, not FACT
            if insight_type in ("correlation", "outlier"):
                ev_source = "insight_engine"
            else:
                ev_source = "insight_engine"

            ev = create_evidence(
                description=desc or title,
                source=ev_source,
                metric_value=metric_val,
                dataset_id=self._dataset_id,
                details={
                    "insight_type": insight_type,
                    "evidence": evidence_text,
                },
            )

            # Correlation insights get the non-causal guard
            if insight_type == "correlation":
                ev = ensure_non_causal(ev)

            priority = SignalPriority.from_severity(severity)

            self._add_signal(
                description=f"Insight: {title}" if title else f"Insight: {desc[:80]}",
                evidence_items=[ev],
                priority=priority,
                source_type="phase3_insight",
                generation_reason=f"Insight of type '{insight_type}' detected.",
                affected_features=list(source_cols),
            )

        # --- Relationships ---
        relationships = _get_attr(analytics_result, "relationships", [])
        for rel in relationships:
            rel_type = _get_attr(rel, "relationship_type", "")
            columns = _get_attr(rel, "columns", [])
            strength = _get_attr(rel, "strength", 0.0)
            direction = _get_attr(rel, "direction", "")
            desc = _get_attr(rel, "description", "")

            if not columns or not desc:
                continue

            ev = create_evidence(
                description=desc,
                source="relationship_engine",
                metric_name="strength",
                metric_value=strength,
                dataset_id=self._dataset_id,
                details={
                    "relationship_type": rel_type,
                    "direction": direction,
                },
            )
            # Relationships are observational — ensure non-causal
            ev = ensure_non_causal(ev)

            self._add_signal(
                description=f"Relationship detected: {desc[:100]}",
                evidence_items=[ev],
                priority=SignalPriority.MEDIUM,
                source_type="phase3_relationship",
                generation_reason=f"Statistical relationship ({rel_type}) found between {columns}.",
                affected_features=list(columns),
            )

    # -------------------------------------------------------------------
    # Phase 4: Predictions (Forecast, Regression, Classification, Segmentation)
    # -------------------------------------------------------------------

    def ingest_predictions(self, prediction_report) -> None:
        """Ingest Phase 4 PredictionReport.

        Accepts a PredictionReport object or dict with:
        - dataset_id, forecast, regression, classification, segmentation
        """
        if prediction_report is None:
            self._processing_notes.append(
                "Phase 4 predictions: no result provided."
            )
            return

        ds_id = _get_attr(prediction_report, "dataset_id", "")
        if ds_id and ds_id != self._dataset_id:
            raise ValueError(
                f"PredictionReport dataset_id '{ds_id}' does not match "
                f"synthesizer dataset_id '{self._dataset_id}'."
            )

        self._ingested_sources.add("phase4_predictions")

        # --- Forecast ---
        forecast = _get_attr(prediction_report, "forecast", None)
        if forecast is not None:
            self._ingest_forecast(forecast)

        # --- Regression ---
        regression = _get_attr(prediction_report, "regression", None)
        if regression is not None:
            self._ingest_regression(regression)

        # --- Classification ---
        classification = _get_attr(prediction_report, "classification", None)
        if classification is not None:
            self._ingest_classification(classification)

        # --- Segmentation ---
        segmentation = _get_attr(prediction_report, "segmentation", None)
        if segmentation is not None:
            self._ingest_segmentation(segmentation)

    def _ingest_forecast(self, forecast) -> None:
        status = _get_attr(forecast, "status", "")
        if status != "success":
            self._processing_notes.append(
                f"Forecast status is '{status}', skipping signal generation."
            )
            return

        target_col = _get_attr(forecast, "target_column", "")
        model_name = _get_attr(forecast, "model_name", "")
        horizon = _get_attr(forecast, "horizon", 0)
        metrics = _get_attr(forecast, "metrics", {})
        limitations = _get_attr(forecast, "limitations", [])
        explanation = _get_attr(forecast, "explanation", "")

        desc = (
            f"Forecast for '{target_col}' using {model_name}, "
            f"horizon={horizon}"
        )
        ev = create_evidence(
            description=desc + (f". {explanation}" if explanation else ""),
            source="forecasting_engine",
            metric_name="forecast_horizon",
            metric_value=horizon,
            dataset_id=self._dataset_id,
            details={"metrics": metrics, "model": model_name},
        )
        ev = ensure_prediction_uncertainty(ev)

        # Build limitations from the forecast's own limitations list
        signal_lims = _convert_string_limitations(limitations)

        self._add_signal(
            description=f"Forecast: {target_col} projected {horizon} periods ahead",
            evidence_items=[ev],
            priority=SignalPriority.HIGH,
            source_type="phase4_forecast",
            generation_reason=f"Forecast model ({model_name}) produced predictions for '{target_col}'.",
            affected_features=[target_col] if target_col else [],
            limitations=signal_lims,
        )

        # Feature importance from forecast (if any)
        self._ingest_feature_importance(
            _get_attr(forecast, "feature_importance", []),
            target_col,
            "phase4_forecast",
        )

    def _ingest_regression(self, regression) -> None:
        status = _get_attr(regression, "status", "")
        if status != "success":
            self._processing_notes.append(
                f"Regression status is '{status}', skipping signal generation."
            )
            return

        target_col = _get_attr(regression, "target_column", "")
        model_name = _get_attr(regression, "model_name", "")
        metrics = _get_attr(regression, "metrics", {})
        limitations = _get_attr(regression, "limitations", [])

        r2 = metrics.get("r2") or metrics.get("R2")
        desc = f"Regression model ({model_name}) for '{target_col}'"
        if r2 is not None:
            desc += f", R²={r2:.4f}"

        ev = create_evidence(
            description=desc,
            source="regression_engine",
            metric_name="r2",
            metric_value=r2,
            dataset_id=self._dataset_id,
            details={"metrics": metrics, "model": model_name},
        )
        ev = ensure_prediction_uncertainty(ev)

        signal_lims = _convert_string_limitations(limitations)

        self._add_signal(
            description=f"Regression: {target_col} modeled with {model_name}",
            evidence_items=[ev],
            priority=SignalPriority.HIGH,
            source_type="phase4_regression",
            generation_reason=f"Regression model fitted for '{target_col}'.",
            affected_features=[target_col] if target_col else [],
            limitations=signal_lims,
        )

        self._ingest_feature_importance(
            _get_attr(regression, "feature_importance", []),
            target_col,
            "phase4_regression",
        )

    def _ingest_classification(self, classification) -> None:
        status = _get_attr(classification, "status", "")
        if status != "success":
            self._processing_notes.append(
                f"Classification status is '{status}', skipping signal generation."
            )
            return

        target_col = _get_attr(classification, "target_column", "")
        model_name = _get_attr(classification, "model_name", "")
        metrics = _get_attr(classification, "metrics", {})
        limitations = _get_attr(classification, "limitations", [])

        accuracy = metrics.get("accuracy")
        desc = f"Classification model ({model_name}) for '{target_col}'"
        if accuracy is not None:
            desc += f", accuracy={accuracy:.4f}"

        ev = create_evidence(
            description=desc,
            source="classification_engine",
            metric_name="accuracy",
            metric_value=accuracy,
            dataset_id=self._dataset_id,
            details={"metrics": metrics, "model": model_name},
        )
        ev = ensure_prediction_uncertainty(ev)

        signal_lims = _convert_string_limitations(limitations)

        self._add_signal(
            description=f"Classification: {target_col} classified with {model_name}",
            evidence_items=[ev],
            priority=SignalPriority.HIGH,
            source_type="phase4_classification",
            generation_reason=f"Classification model fitted for '{target_col}'.",
            affected_features=[target_col] if target_col else [],
            limitations=signal_lims,
        )

        self._ingest_feature_importance(
            _get_attr(classification, "feature_importance", []),
            target_col,
            "phase4_classification",
        )

    def _ingest_segmentation(self, segmentation) -> None:
        status = _get_attr(segmentation, "status", "")
        if status != "success":
            self._processing_notes.append(
                f"Segmentation status is '{status}', skipping signal generation."
            )
            return

        cluster_count = _get_attr(segmentation, "cluster_count", 0)
        feature_cols = _get_attr(segmentation, "feature_columns", [])
        quality = _get_attr(segmentation, "quality_metrics", {})
        interpretation = _get_attr(segmentation, "interpretation", "")
        limitations = _get_attr(segmentation, "limitations", [])

        silhouette = quality.get("silhouette")
        desc = f"{cluster_count} segments identified"
        if silhouette is not None:
            desc += f", silhouette={silhouette:.4f}"
        if interpretation:
            desc += f". {interpretation}"

        ev = create_evidence(
            description=desc,
            source="segmentation_engine",
            metric_name="cluster_count",
            metric_value=cluster_count,
            dataset_id=self._dataset_id,
            details={"quality_metrics": quality},
        )
        ev = ensure_prediction_uncertainty(ev)

        signal_lims = _convert_string_limitations(limitations)

        self._add_signal(
            description=f"Segmentation: {cluster_count} segments found",
            evidence_items=[ev],
            priority=SignalPriority.MEDIUM,
            source_type="phase4_segmentation",
            generation_reason=f"Segmentation produced {cluster_count} clusters.",
            affected_features=list(feature_cols),
            limitations=signal_lims,
        )

    def _ingest_feature_importance(
        self, importances: list, target: str, source_type: str,
    ) -> None:
        """Convert feature importance items into inference signals."""
        for fi in importances:
            feature = _get_attr(fi, "feature", "")
            importance = _get_attr(fi, "importance", 0.0)
            if not feature or importance == 0.0:
                continue

            ev = create_evidence(
                description=(
                    f"Feature '{feature}' has importance {importance:.4f} "
                    f"for predicting '{target}'"
                ),
                source="driver_engine",
                metric_name="importance",
                metric_value=importance,
                dataset_id=self._dataset_id,
                details={"target": target, "feature": feature},
            )
            # Feature importance is NOT causal
            ev = ensure_non_causal(ev)

            self._add_signal(
                description=(
                    f"Feature importance: '{feature}' contributes to "
                    f"predicting '{target}' (importance={importance:.4f})"
                ),
                evidence_items=[ev],
                priority=SignalPriority.LOW,
                source_type=source_type,
                generation_reason=(
                    f"Feature importance computed for '{feature}'. "
                    "This reflects predictive association, not causal effect."
                ),
                affected_features=[feature, target],
            )

    # -------------------------------------------------------------------
    # Phase 5: Anomaly Detection + Driver Analysis
    # -------------------------------------------------------------------

    def ingest_anomalies(self, phase5_result) -> None:
        """Ingest Phase 5 Phase5Result.

        Accepts a Phase5Result object or dict with:
        - dataset_id, anomaly_report, driver_report
        """
        if phase5_result is None:
            self._processing_notes.append(
                "Phase 5 anomalies/drivers: no result provided."
            )
            return

        ds_id = _get_attr(phase5_result, "dataset_id", "")
        if ds_id and ds_id != self._dataset_id:
            raise ValueError(
                f"Phase5Result dataset_id '{ds_id}' does not match "
                f"synthesizer dataset_id '{self._dataset_id}'."
            )

        self._ingested_sources.add("phase5_anomalies")

        # --- Anomaly Report ---
        anomaly_report = _get_attr(phase5_result, "anomaly_report", None)
        if anomaly_report is not None:
            self._ingest_anomaly_report(anomaly_report)

        # --- Driver Report ---
        driver_report = _get_attr(phase5_result, "driver_report", None)
        if driver_report is not None:
            self._ingest_driver_report(driver_report)

    def _ingest_anomaly_report(self, report) -> None:
        anomalies = _get_attr(report, "anomalies", [])
        for anom in anomalies:
            column = _get_attr(anom, "column", "")
            anom_type = _get_attr(anom, "anomaly_type", "")
            severity = _get_attr(anom, "severity", "LOW")
            score = _get_attr(anom, "score", 0.0)
            observed = _get_attr(anom, "observed_value", None)
            expected = _get_attr(anom, "expected_value", None)
            method = _get_attr(anom, "method", "")
            explanation = _get_attr(anom, "explanation", "")

            desc = f"Anomaly in '{column}': {anom_type}"
            if explanation:
                desc += f". {explanation}"
            elif observed is not None and expected is not None:
                desc += f" (observed={observed}, expected={expected})"

            ev = create_evidence(
                description=desc,
                source="anomaly_detector",
                metric_name="anomaly_score",
                metric_value=score,
                dataset_id=self._dataset_id,
                details={
                    "anomaly_type": anom_type,
                    "method": method,
                    "observed": observed,
                    "expected": expected,
                },
            )

            priority = SignalPriority.from_severity(severity)

            self._add_signal(
                description=f"Anomaly detected: {column} ({anom_type}, severity={severity})",
                evidence_items=[ev],
                priority=priority,
                source_type="phase5_anomaly",
                generation_reason=(
                    f"Anomaly of type '{anom_type}' detected in '{column}' "
                    f"with severity {severity} using method '{method}'."
                ),
                affected_features=[column] if column else [],
            )

    def _ingest_driver_report(self, report) -> None:
        drivers = _get_attr(report, "drivers", [])
        target_col = _get_attr(report, "target_column", "")

        for driver in drivers:
            feature = _get_attr(driver, "feature", "")
            target = _get_attr(driver, "target", "") or target_col
            strength = _get_attr(driver, "strength", 0.0)
            direction = _get_attr(driver, "direction", "")
            rel_type = _get_attr(driver, "relationship_type", "")
            method = _get_attr(driver, "method", "")
            explanation = _get_attr(driver, "explanation", "")

            if not feature:
                continue

            desc = (
                f"Driver: '{feature}' associated with '{target}' "
                f"(strength={strength:.4f}, direction={direction})"
            )
            if explanation:
                desc += f". {explanation}"

            ev = create_evidence(
                description=desc,
                source="driver_engine",
                metric_name="strength",
                metric_value=strength,
                dataset_id=self._dataset_id,
                details={
                    "relationship_type": rel_type,
                    "method": method,
                    "direction": direction,
                },
            )
            # Drivers are observational — association, not causation
            ev = ensure_non_causal(ev)

            self._add_signal(
                description=(
                    f"Driver analysis: '{feature}' is associated with '{target}'"
                ),
                evidence_items=[ev],
                priority=SignalPriority.MEDIUM,
                source_type="phase5_driver",
                generation_reason=(
                    f"Driver analysis found '{feature}' associated with '{target}' "
                    f"via {method}. This is an observational association, not a causal finding."
                ),
                affected_features=[feature, target],
            )

    # -------------------------------------------------------------------
    # Phase 6: Marketing Intelligence
    # -------------------------------------------------------------------

    def ingest_marketing(self, phase6_result) -> None:
        """Ingest Phase 6 Phase6Result.

        Accepts a Phase6Result object or dict with:
        - dataset_id, contributions, roi_results, optimization_result,
          whatif_result, mmm_result
        """
        if phase6_result is None:
            self._processing_notes.append(
                "Phase 6 marketing: no result provided."
            )
            return

        ds_id = _get_attr(phase6_result, "dataset_id", "")
        if ds_id and ds_id != self._dataset_id:
            raise ValueError(
                f"Phase6Result dataset_id '{ds_id}' does not match "
                f"synthesizer dataset_id '{self._dataset_id}'."
            )

        self._ingested_sources.add("phase6_marketing")

        # --- MMM Result ---
        mmm = _get_attr(phase6_result, "mmm_result", None)
        if mmm is not None:
            self._ingest_mmm(mmm)

        # --- Channel Contributions ---
        contributions = _get_attr(phase6_result, "contributions", [])
        for contrib in contributions:
            channel = _get_attr(contrib, "channel", "")
            contrib_val = _get_attr(contrib, "contribution", 0.0)
            share = _get_attr(contrib, "contribution_share", 0.0)
            roi_val = _get_attr(contrib, "roi", 0.0)

            if not channel:
                continue

            ev = create_evidence(
                description=(
                    f"Channel '{channel}': contribution={contrib_val:.4f}, "
                    f"share={share:.4f}, ROI={roi_val:.4f}"
                ),
                source="contribution",
                metric_name="contribution_share",
                metric_value=share,
                dataset_id=self._dataset_id,
                details={"roi": roi_val, "contribution": contrib_val},
            )
            # Model-derived contribution is inference, not fact
            ev = ensure_non_causal(ev)

            self._add_signal(
                description=(
                    f"Channel contribution: '{channel}' accounts for "
                    f"{share:.1%} of modeled effect"
                ),
                evidence_items=[ev],
                priority=SignalPriority.MEDIUM,
                source_type="phase6_contribution",
                generation_reason=(
                    f"Model-estimated contribution for channel '{channel}'. "
                    "This is a model estimate, not a directly measured causal effect."
                ),
                affected_features=[channel],
            )

        # --- ROI Results ---
        roi_results = _get_attr(phase6_result, "roi_results", [])
        for roi in roi_results:
            channel = _get_attr(roi, "channel", "")
            roas = _get_attr(roi, "naive_roas", None)
            model_roi = _get_attr(roi, "model_estimated_roi", None)
            method = _get_attr(roi, "method", "")

            if not channel:
                continue

            metric_val = model_roi if model_roi is not None else roas
            ev = create_evidence(
                description=(
                    f"ROI for '{channel}': "
                    f"{'model-estimated ROI=' + str(model_roi) if model_roi is not None else 'naive ROAS=' + str(roas)}"
                ),
                source="roi",
                metric_name="roi",
                metric_value=metric_val,
                dataset_id=self._dataset_id,
                details={"method": method},
            )
            ev = ensure_non_causal(ev)

            self._add_signal(
                description=f"ROI estimate for '{channel}': {metric_val}",
                evidence_items=[ev],
                priority=SignalPriority.MEDIUM,
                source_type="phase6_roi",
                generation_reason=(
                    f"ROI computed for '{channel}' using {method}. "
                    "ROI estimates depend on model assumptions."
                ),
                affected_features=[channel],
            )

        # --- What-If ---
        whatif = _get_attr(phase6_result, "whatif_result", None)
        if whatif is not None:
            self._ingest_whatif(whatif)

        # --- Optimization ---
        optimization = _get_attr(phase6_result, "optimization_result", None)
        if optimization is not None:
            self._ingest_optimization(optimization)

    def _ingest_mmm(self, mmm) -> None:
        selected = _get_attr(mmm, "selected_model", "")
        train_metrics = _get_attr(mmm, "train_metrics", {})
        test_metrics = _get_attr(mmm, "test_metrics", {})
        limitations = _get_attr(mmm, "limitations", [])
        assumptions = _get_attr(mmm, "assumptions", [])

        if not selected:
            return

        r2 = test_metrics.get("r2") or test_metrics.get("R2")
        desc = f"Marketing Mix Model ({selected})"
        if r2 is not None:
            desc += f", test R²={r2:.4f}"

        ev = create_evidence(
            description=desc,
            source="mmm_engine",
            metric_name="test_r2",
            metric_value=r2,
            dataset_id=self._dataset_id,
            details={
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "assumptions": assumptions,
            },
        )
        ev = ensure_non_causal(ev)

        signal_lims = _convert_string_limitations(limitations)
        for assumption in assumptions:
            if assumption:
                signal_lims.append(
                    create_limitation(
                        str(assumption),
                        category=LimitationCategory.METHODOLOGY.value,
                    )
                )

        self._add_signal(
            description=f"Marketing Mix Model fitted ({selected})",
            evidence_items=[ev],
            priority=SignalPriority.HIGH,
            source_type="phase6_mmm",
            generation_reason=(
                f"MMM ({selected}) fitted. Results are model estimates "
                "based on observational data, not controlled experiments."
            ),
            limitations=signal_lims,
        )

    def _ingest_whatif(self, whatif) -> None:
        scenarios = _get_attr(whatif, "scenarios", [])
        disclaimer = _get_attr(whatif, "disclaimer", "")

        for scenario in scenarios:
            name = _get_attr(scenario, "name", "")
            predicted = _get_attr(scenario, "predicted_target", 0.0)
            change_pct = _get_attr(scenario, "target_change_pct", 0.0)

            if not name:
                continue

            ev = create_evidence(
                description=(
                    f"Scenario '{name}': predicted target={predicted:.2f}, "
                    f"change={change_pct:.2f}%"
                ),
                source="scenario_engine",
                metric_name="target_change_pct",
                metric_value=change_pct,
                dataset_id=self._dataset_id,
                details={"disclaimer": disclaimer},
            )
            ev = ensure_prediction_uncertainty(ev)

            lims = []
            if disclaimer:
                lims.append(create_limitation(
                    disclaimer,
                    category=LimitationCategory.MODEL_UNCERTAINTY.value,
                ))

            self._add_signal(
                description=f"What-if scenario: '{name}' projects {change_pct:.1f}% change",
                evidence_items=[ev],
                priority=SignalPriority.LOW,
                source_type="phase6_whatif",
                generation_reason=(
                    f"What-if scenario '{name}' simulated. "
                    "Scenarios are projections, not guaranteed outcomes."
                ),
                limitations=lims,
            )

    def _ingest_optimization(self, optimization) -> None:
        success = _get_attr(optimization, "success", False)
        if not success:
            self._processing_notes.append(
                "Budget optimization was not successful, skipping signal."
            )
            return

        improvement = _get_attr(optimization, "expected_improvement_pct", 0.0)
        message = _get_attr(optimization, "message", "")

        ev = create_evidence(
            description=(
                f"Budget optimization suggests {improvement:.2f}% improvement. "
                f"{message}"
            ),
            source="optimizer",
            metric_name="expected_improvement_pct",
            metric_value=improvement,
            dataset_id=self._dataset_id,
        )
        ev = ensure_prediction_uncertainty(ev)

        self._add_signal(
            description=f"Budget optimization: {improvement:.1f}% projected improvement",
            evidence_items=[ev],
            priority=SignalPriority.MEDIUM,
            source_type="phase6_optimization",
            generation_reason=(
                "Optimization result suggests a reallocation. "
                "This is a model recommendation, not a guaranteed outcome."
            ),
            limitations=[
                create_limitation(
                    "Optimization results are model-based recommendations. "
                    "Actual outcomes depend on market conditions and execution.",
                    category=LimitationCategory.MODEL_UNCERTAINTY.value,
                ),
            ],
        )

    # -------------------------------------------------------------------
    # Ingest generic evidence (for non-marketing / custom domains)
    # -------------------------------------------------------------------

    def ingest_evidence(
        self,
        evidence_items: list,
        *,
        source_type: str = "custom",
        description: str = "",
        priority: int = SignalPriority.MEDIUM,
        affected_features: list | None = None,
    ) -> Optional[SynthesizedSignal]:
        """Ingest pre-built DecisionEvidence items directly.

        For domain-specific usage where Phase 3–6 result types don't apply.
        All evidence must belong to this synthesizer's dataset.
        """
        if not evidence_items:
            return None

        # Validate dataset scope
        scoped = validate_dataset_scope(evidence_items, self._dataset_id)
        if not scoped:
            self._processing_notes.append(
                f"All evidence items filtered out for dataset '{self._dataset_id}'."
            )
            return None

        self._ingested_sources.add(source_type)

        return self._add_signal(
            description=description or "Custom evidence signal",
            evidence_items=scoped,
            priority=priority,
            source_type=source_type,
            generation_reason="Custom evidence ingested directly.",
            affected_features=affected_features,
        )

    # -------------------------------------------------------------------
    # Contradiction detection
    # -------------------------------------------------------------------

    def _detect_contradictions(self) -> None:
        """Detect contradictory signals and surface them.

        A contradiction occurs when two signals about the same feature
        have opposing evidence (e.g. one says X is increasing, another
        says X is decreasing).

        Contradictions are surfaced, never silently discarded.
        """
        # Group signals by affected features
        feature_signals: dict[str, list[SynthesizedSignal]] = {}
        for synth in self._raw_signals:
            for feat in synth.signal.affected_features:
                feature_signals.setdefault(feat, []).append(synth)

        # Look for conflicting evidence directions
        for feature, signals in feature_signals.items():
            if len(signals) < 2:
                continue

            # Check for directional conflicts in evidence details
            directions_seen = {}
            for synth in signals:
                for ev in synth.evidence_items:
                    details = ev.details if hasattr(ev, "details") else {}
                    direction = details.get("direction", "")
                    if direction and direction in ("positive", "negative"):
                        key = (synth.source_type, feature)
                        if key in directions_seen and directions_seen[key] != direction:
                            self._contradictions.append({
                                "feature": feature,
                                "conflict": (
                                    f"Contradictory directions for '{feature}': "
                                    f"'{directions_seen[key]}' vs '{direction}'"
                                ),
                                "signal_ids": [s.signal.id for s in signals],
                            })
                        directions_seen[key] = direction

    # -------------------------------------------------------------------
    # Deduplication
    # -------------------------------------------------------------------

    def _deduplicate_signals(self) -> list:
        """Remove duplicate signals deterministically.

        Two signals are considered duplicates if they have:
        - The same description
        - The same source_type
        - The same affected features (as a sorted set)

        When duplicates are found, the one with higher priority (lower
        number) is kept. If tied, the first encountered is kept.
        """
        seen = {}
        deduped = []

        for synth in self._raw_signals:
            key = (
                synth.signal.description,
                synth.source_type,
                tuple(sorted(synth.signal.affected_features)),
            )

            if key in seen:
                existing = seen[key]
                if synth.priority < existing.priority:
                    # Replace with higher-priority version
                    deduped = [s for s in deduped if s is not existing]
                    deduped.append(synth)
                    seen[key] = synth
                # else skip the duplicate
            else:
                seen[key] = synth
                deduped.append(synth)

        return deduped

    # -------------------------------------------------------------------
    # Synthesize
    # -------------------------------------------------------------------

    def synthesize(self) -> SynthesisResult:
        """Produce the final SynthesisResult.

        Steps:
        1. Detect contradictions (surface them, don't discard).
        2. Deduplicate signals.
        3. Rank by priority.
        4. Return result.
        """
        if not self._raw_signals:
            self._processing_notes.append(
                "No signals generated — no analytical results ingested "
                "or all results were empty/failed."
            )

        # Step 1: Detect contradictions
        self._detect_contradictions()

        # Step 2: Deduplicate
        deduped = self._deduplicate_signals()

        # Step 3: Already ranked by priority in SynthesisResult.ranked_signals()

        return SynthesisResult(
            dataset_id=self._dataset_id,
            signals=deduped,
            evidence_registry=dict(self._evidence_registry),
            contradictions=list(self._contradictions),
            processing_notes=list(self._processing_notes),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    """Get attribute from an object or dict, with a default."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _convert_string_limitations(raw_limitations: list) -> list:
    """Convert a list of limitation strings into DecisionLimitation objects.

    Phase 3–6 store limitations as plain strings; this converts them
    into structured Part 1 DecisionLimitation objects.
    """
    result = []
    for lim in raw_limitations:
        if isinstance(lim, str) and lim.strip():
            result.append(
                create_limitation(
                    lim.strip(),
                    category=LimitationCategory.METHODOLOGY.value,
                )
            )
        elif hasattr(lim, "category"):
            # Already a DecisionLimitation
            result.append(lim)
        elif isinstance(lim, dict) and lim.get("description"):
            result.append(
                create_limitation(
                    lim["description"],
                    category=lim.get("category", LimitationCategory.METHODOLOGY.value),
                )
            )
    return result
