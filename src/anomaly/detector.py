"""Phase 5 — Anomaly Detection Orchestrator.

Merges numeric, time-series, and categorical anomaly detection into a
single AnomalyReport.  Each sub-detector is called independently; a
failure in one does not block the others.
"""

from __future__ import annotations

import logging
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.anomaly.contracts import AnomalyReport, AnomalySeverity
from src.anomaly.numeric import detect_numeric_anomalies
from src.anomaly.time_series import detect_time_series_anomalies
from src.anomaly.categorical import detect_categorical_anomalies
from src.data_engine.canonical import AnalysisContext

logger = logging.getLogger(__name__)

# Severity ordering for sorting (higher value = more severe).
_SEVERITY_RANK = {
    AnomalySeverity.LOW.value: 0,
    AnomalySeverity.MEDIUM.value: 1,
    AnomalySeverity.HIGH.value: 2,
    AnomalySeverity.CRITICAL.value: 3,
}


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def detect_anomalies(
    context: AnalysisContext,
    dataset_id: str,
    max_anomalies: int = 100,
) -> AnomalyReport:
    """Run all anomaly sub-detectors and merge results into one report.

    Parameters
    ----------
    context : AnalysisContext
        Phase 2 analysis context wrapping a CanonicalDataset.
    dataset_id : str
        Identifier stamped on the report and every anomaly.
    max_anomalies : int
        Cap on total anomalies returned (most severe first).

    Returns
    -------
    AnomalyReport
    """
    canonical = context.dataset
    df = context.get_dataframe()

    # -- Empty / unavailable dataset ------------------------------------------
    if df is None or df.empty:
        return AnomalyReport(
            dataset_id=dataset_id,
            unavailable_analyses=[
                {
                    "analysis": "anomaly_detection",
                    "reason": "Dataset is empty or unavailable.",
                }
            ],
        )

    # -- Call each sub-detector -----------------------------------------------
    all_anomalies = []
    unavailable: list[dict] = []

    # Numeric
    try:
        all_anomalies.extend(detect_numeric_anomalies(context, dataset_id))
    except Exception as exc:
        logger.exception("Numeric anomaly detection failed.")
        unavailable.append({
            "analysis": "numeric_anomaly_detection",
            "reason": str(exc),
        })

    # Time-series
    try:
        all_anomalies.extend(detect_time_series_anomalies(context, dataset_id))
    except Exception as exc:
        logger.exception("Time-series anomaly detection failed.")
        unavailable.append({
            "analysis": "time_series_anomaly_detection",
            "reason": str(exc),
        })

    # Categorical
    try:
        all_anomalies.extend(detect_categorical_anomalies(context, dataset_id))
    except Exception as exc:
        logger.exception("Categorical anomaly detection failed.")
        unavailable.append({
            "analysis": "categorical_anomaly_detection",
            "reason": str(exc),
        })

    # -- Sort: severity descending, then score descending ---------------------
    all_anomalies.sort(
        key=lambda a: (_SEVERITY_RANK.get(a.severity, 0), a.score),
        reverse=True,
    )

    # -- Cap ------------------------------------------------------------------
    if len(all_anomalies) > max_anomalies:
        all_anomalies = all_anomalies[:max_anomalies]

    # -- Build summary --------------------------------------------------------
    type_counts: Counter = Counter(a.anomaly_type for a in all_anomalies)
    severity_counts: Counter = Counter(a.severity for a in all_anomalies)
    column_counts: Counter = Counter(a.column for a in all_anomalies)

    affected_columns = sorted(column_counts.keys())
    most_anomalous_columns = [
        col for col, _ in column_counts.most_common(5)
    ]

    summary = {
        "total_anomalies": len(all_anomalies),
        "by_type": dict(type_counts),
        "by_severity": {
            sev.value: severity_counts.get(sev.value, 0)
            for sev in AnomalySeverity
        },
        "affected_columns": affected_columns,
        "most_anomalous_columns": most_anomalous_columns,
    }

    # -- Build metadata -------------------------------------------------------
    time_info = canonical.time_info if hasattr(canonical, "time_info") else None
    time_available = time_info is not None
    if time_available:
        time_col = (
            time_info.time_column
            if hasattr(time_info, "time_column")
            else (time_info.get("time_column") if isinstance(time_info, dict) else None)
        )
        time_available = time_col is not None

    cat_cols = canonical.categorical_columns if hasattr(canonical, "categorical_columns") else []

    metadata = {
        "total_rows_analyzed": len(df),
        "numeric_columns_analyzed": len(canonical.numeric_columns) if canonical.numeric_columns else 0,
        "time_series_available": time_available,
        "categorical_columns_analyzed": len(cat_cols) if cat_cols else 0,
    }

    return AnomalyReport(
        dataset_id=dataset_id,
        anomalies=all_anomalies,
        summary=summary,
        unavailable_analyses=unavailable,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def get_anomaly_summary(context: AnalysisContext, dataset_id: str) -> dict:
    """Quick summary without full anomaly details."""
    report = detect_anomalies(context, dataset_id)
    return report.summary
