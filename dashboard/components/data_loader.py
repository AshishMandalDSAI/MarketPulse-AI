"""Cached data-loading helpers shared across all dashboard pages.

Centralizing loads here means expensive operations (reading CSVs, unpickling
the trained model) happen once per session via st.cache_data / st.cache_resource
rather than on every page render (Phase 15 — Performance requirement).

Dataset-first update: these loaders now resolve DATA_DIR/MODELS_DIR against
the *active* dataset (built-in demo, or an uploaded dataset) via
dashboard.components.session_paths, and tolerate EmptyDataError since some
pipeline stages legitimately write empty CSVs (e.g. a dataset with 0
marketing channels writes an empty channel_roi.csv).

Phase 1: st.cache_data is cleared on every dataset switch (via
session_paths._clear_page_caches), so cached results cannot leak between
datasets. has_capability() now returns False for uploaded datasets with
no capabilities.json, preventing silent fallback to demo behavior.

Legacy cache isolation fix: Streamlit's cache is shared by every browser
session of the server process and is keyed only by the function and its
arguments. The public loaders below take no arguments and used to read the
active dataset from st.session_state *inside* the cached function, so the
dataset was not part of the cache key and one user could be served another
user's data. Each public loader is now a thin, uncached wrapper that resolves
the caller's context (dataset id, directories, date column) and the
fingerprints (absolute path, mtime_ns, size) of the files it reads, then calls
a private ``*_worker`` that is cached on those explicit arguments. Workers
never read st.session_state or session_paths. The public names, zero-argument
signatures, return types and ``.clear()`` methods are unchanged.
"""

import json
import os
import sys
from pathlib import Path
from typing import NamedTuple, Optional

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dashboard.components import session_paths

# Phase 1: these module-level constants are ONLY for backward compat
# in demo mode. They must NOT be used for uploaded datasets.
DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"


def _safe_read_csv(path: Path, parse_dates=None) -> pd.DataFrame:
    """Reads a CSV, gracefully handling a missing file or a legitimately
    empty one (pandas.errors.EmptyDataError) instead of crashing."""
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, parse_dates=parse_dates) if parse_dates else pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _safe_read_json(path: Path):
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


# ---------------------------------------------------------------------------
# Cache-key helpers (legacy cache isolation fix)
# ---------------------------------------------------------------------------

class _Context(NamedTuple):
    """The calling session's active-dataset context, resolved once per call."""
    dataset_id: str
    processed_dir: Path
    models_dir: Path


def _context() -> _Context:
    """Resolve the active dataset context for THIS session (never cached)."""
    return _Context(
        session_paths.get_dataset_id(),
        session_paths.get_processed_dir(),
        session_paths.get_models_dir(),
    )


def _fingerprint(path) -> tuple:
    """File-specific cache fingerprint: (absolute path, mtime_ns, size).

    A missing or unreadable file gets the deterministic fingerprint
    (absolute path, -1, -1) and never raises, so a file that appears later
    changes the key. Only os.stat is used - file contents are never hashed.
    """
    try:
        resolved = Path(path).resolve()
    except (OSError, RuntimeError, ValueError):
        resolved = Path(path).absolute()
    try:
        info = os.stat(resolved)
    except (OSError, ValueError):
        return (str(resolved), -1, -1)
    return (str(resolved), info.st_mtime_ns, info.st_size)


def _marketing_data_fingerprints(processed_dir: Path) -> tuple:
    """Fingerprints of both files load_marketing_data may read (it falls back
    from marketing_data_clean.csv to standardized_data.csv)."""
    return (
        _fingerprint(processed_dir / "marketing_data_clean.csv"),
        _fingerprint(processed_dir / "standardized_data.csv"),
    )


def _read_marketing_df(processed_dir: Path, date_col: Optional[str]) -> pd.DataFrame:
    """Uncached read of the analysis table. All context is passed explicitly."""
    # dataset-first: standardized_data.csv for an uploaded dataset,
    # marketing_data_clean.csv for the built-in demo dataset.
    path = processed_dir / "marketing_data_clean.csv"
    if not path.exists():
        path = processed_dir / "standardized_data.csv"
    df = _safe_read_csv(path)
    date_col = date_col or "Date"
    if date_col in df.columns:
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        except Exception:
            pass
    return df


# ---------------------------------------------------------------------------
# Loaders. Each public loader is an uncached wrapper around a private keyed
# worker. Worker arguments are part of the cache key; the ones not read in the
# body (dataset id, fingerprints) exist only to make the key correct.
# ---------------------------------------------------------------------------

@st.cache_data
def _marketing_data_worker(dataset_id: str, processed_dir: str, date_col: Optional[str],
                           clean_fp: tuple, standardized_fp: tuple) -> pd.DataFrame:
    return _read_marketing_df(Path(processed_dir), date_col)


def load_marketing_data() -> pd.DataFrame:
    c = _context()
    clean_fp, standardized_fp = _marketing_data_fingerprints(c.processed_dir)
    return _marketing_data_worker(c.dataset_id, str(c.processed_dir), session_paths.get_date_column(),
                                  clean_fp, standardized_fp)


load_marketing_data.clear = _marketing_data_worker.clear


@st.cache_data
def _kpis_worker(dataset_id: str, processed_dir: str, fp: tuple) -> dict:
    return _safe_read_json(Path(processed_dir) / "marketing_kpis.json")


def load_kpis() -> dict:
    c = _context()
    return _kpis_worker(c.dataset_id, str(c.processed_dir),
                        _fingerprint(c.processed_dir / "marketing_kpis.json"))


load_kpis.clear = _kpis_worker.clear


@st.cache_data
def _generic_kpis_worker(dataset_id: str, processed_dir: str, fp: tuple) -> list:
    return _safe_read_json(Path(processed_dir) / "generic_kpis.json") or []


def load_generic_kpis() -> list:
    c = _context()
    return _generic_kpis_worker(c.dataset_id, str(c.processed_dir),
                                _fingerprint(c.processed_dir / "generic_kpis.json"))


load_generic_kpis.clear = _generic_kpis_worker.clear


@st.cache_data
def _capabilities_worker(dataset_id: str, processed_dir: str, fp: tuple) -> dict:
    return _safe_read_json(Path(processed_dir) / "capabilities.json")


def load_capabilities() -> dict:
    c = _context()
    return _capabilities_worker(c.dataset_id, str(c.processed_dir),
                                _fingerprint(c.processed_dir / "capabilities.json"))


load_capabilities.clear = _capabilities_worker.clear


def has_capability(key: str) -> bool:
    """Check if the active dataset supports a given capability.

    Phase 1 fix: For uploaded (non-demo) datasets, a missing capabilities.json
    means the pipeline hasn't run, so capabilities are UNKNOWN = unavailable.
    Only the demo dataset assumes 'available' when capabilities.json is missing
    (backward compatibility with the pre-pipeline demo data).
    """
    caps = load_capabilities()
    if not caps:
        # No capabilities.json found
        if session_paths.is_user_dataset_mode():
            # Uploaded dataset: pipeline didn't produce capabilities -> unavailable
            return False
        else:
            # Demo dataset: backward compat, assume available
            return True
    entry = caps.get(key)
    if entry is None:
        if session_paths.is_user_dataset_mode():
            return False
        return True
    return bool(entry.get("available", True))


def capability_reason(key: str) -> str:
    caps = load_capabilities()
    entry = caps.get(key) if caps else None
    if entry:
        return entry.get("reason", "")
    if session_paths.is_user_dataset_mode():
        return "Capability information not available for this dataset."
    return ""


@st.cache_data
def _channel_performance_worker(dataset_id: str, processed_dir: str, fp: tuple) -> pd.DataFrame:
    return _safe_read_csv(Path(processed_dir) / "channel_performance.csv")


def load_channel_performance() -> pd.DataFrame:
    c = _context()
    return _channel_performance_worker(c.dataset_id, str(c.processed_dir),
                                       _fingerprint(c.processed_dir / "channel_performance.csv"))


load_channel_performance.clear = _channel_performance_worker.clear


@st.cache_data
def _channel_attribution_worker(dataset_id: str, processed_dir: str, models_dir: str,
                                processed_fp: tuple, models_fp: tuple) -> pd.DataFrame:
    df = _safe_read_csv(Path(processed_dir) / "channel_attribution.csv")
    if df.empty:
        df = _safe_read_csv(Path(models_dir) / "channel_attribution.csv")
    return df


def load_channel_attribution() -> pd.DataFrame:
    c = _context()
    return _channel_attribution_worker(c.dataset_id, str(c.processed_dir), str(c.models_dir),
                                       _fingerprint(c.processed_dir / "channel_attribution.csv"),
                                       _fingerprint(c.models_dir / "channel_attribution.csv"))


load_channel_attribution.clear = _channel_attribution_worker.clear


@st.cache_data
def _model_metrics_worker(dataset_id: str, models_dir: str, fp: tuple) -> dict:
    return _safe_read_json(Path(models_dir) / "model_metrics.json")


def load_model_metrics() -> dict:
    c = _context()
    return _model_metrics_worker(c.dataset_id, str(c.models_dir),
                                 _fingerprint(c.models_dir / "model_metrics.json"))


load_model_metrics.clear = _model_metrics_worker.clear


@st.cache_data
def _model_diagnostics_worker(dataset_id: str, models_dir: str, fp: tuple) -> dict:
    return _safe_read_json(Path(models_dir) / "model_diagnostics.json")


def load_model_diagnostics() -> dict:
    c = _context()
    return _model_diagnostics_worker(c.dataset_id, str(c.models_dir),
                                     _fingerprint(c.models_dir / "model_diagnostics.json"))


load_model_diagnostics.clear = _model_diagnostics_worker.clear


@st.cache_data
def _actual_vs_predicted_worker(dataset_id: str, processed_dir: str, date_col: Optional[str],
                                fp: tuple) -> pd.DataFrame:
    # Phase 1: use dynamic date column instead of hardcoded ["Date"]
    parse_dates = [date_col] if date_col else None
    return _safe_read_csv(Path(processed_dir) / "actual_vs_predicted.csv", parse_dates=parse_dates)


def load_actual_vs_predicted() -> pd.DataFrame:
    c = _context()
    return _actual_vs_predicted_worker(c.dataset_id, str(c.processed_dir), session_paths.get_date_column(),
                                       _fingerprint(c.processed_dir / "actual_vs_predicted.csv"))


load_actual_vs_predicted.clear = _actual_vs_predicted_worker.clear


@st.cache_data
def _budget_optimization_worker(dataset_id: str, processed_dir: str, fp: tuple) -> dict:
    return _safe_read_json(Path(processed_dir) / "budget_optimization_summary.json")


def load_budget_optimization() -> dict:
    c = _context()
    return _budget_optimization_worker(c.dataset_id, str(c.processed_dir),
                                       _fingerprint(c.processed_dir / "budget_optimization_summary.json"))


load_budget_optimization.clear = _budget_optimization_worker.clear


@st.cache_data
def _budget_table_worker(dataset_id: str, processed_dir: str, fp: tuple) -> pd.DataFrame:
    return _safe_read_csv(Path(processed_dir) / "budget_optimization_result.csv")


def load_budget_table() -> pd.DataFrame:
    c = _context()
    return _budget_table_worker(c.dataset_id, str(c.processed_dir),
                                _fingerprint(c.processed_dir / "budget_optimization_result.csv"))


load_budget_table.clear = _budget_table_worker.clear


@st.cache_data
def _scenario_comparison_worker(dataset_id: str, processed_dir: str, fp: tuple) -> pd.DataFrame:
    return _safe_read_csv(Path(processed_dir) / "scenario_comparison.csv")


def load_scenario_comparison() -> pd.DataFrame:
    c = _context()
    return _scenario_comparison_worker(c.dataset_id, str(c.processed_dir),
                                       _fingerprint(c.processed_dir / "scenario_comparison.csv"))


load_scenario_comparison.clear = _scenario_comparison_worker.clear


@st.cache_data
def _forecast_worker(dataset_id: str, processed_dir: str, models_dir: str, date_col: Optional[str],
                     processed_fp: tuple, models_fp: tuple) -> pd.DataFrame:
    # The forecast file's date column is "Date" for generated forecasts, but may
    # carry the dataset's own date column name. Parse whichever is present —
    # passing a column that does not exist to parse_dates makes pandas raise,
    # which previously turned every uploaded forecast into an empty frame.
    df = _safe_read_csv(Path(processed_dir) / "revenue_forecast.csv")
    if df.empty:
        df = _safe_read_csv(Path(models_dir) / "revenue_forecast.csv")
    for col in dict.fromkeys(c for c in (date_col, "Date") if c):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def load_forecast() -> pd.DataFrame:
    c = _context()
    return _forecast_worker(c.dataset_id, str(c.processed_dir), str(c.models_dir),
                            session_paths.get_date_column(),
                            _fingerprint(c.processed_dir / "revenue_forecast.csv"),
                            _fingerprint(c.models_dir / "revenue_forecast.csv"))


load_forecast.clear = _forecast_worker.clear


@st.cache_data
def _monthly_performance_worker(dataset_id: str, processed_dir: str, fp: tuple) -> pd.DataFrame:
    return _safe_read_csv(Path(processed_dir) / "monthly_performance.csv")


def load_monthly_performance() -> pd.DataFrame:
    c = _context()
    return _monthly_performance_worker(c.dataset_id, str(c.processed_dir),
                                       _fingerprint(c.processed_dir / "monthly_performance.csv"))


load_monthly_performance.clear = _monthly_performance_worker.clear


@st.cache_data
def _quarterly_performance_worker(dataset_id: str, processed_dir: str, fp: tuple) -> pd.DataFrame:
    return _safe_read_csv(Path(processed_dir) / "quarterly_performance.csv")


def load_quarterly_performance() -> pd.DataFrame:
    c = _context()
    return _quarterly_performance_worker(c.dataset_id, str(c.processed_dir),
                                         _fingerprint(c.processed_dir / "quarterly_performance.csv"))


load_quarterly_performance.clear = _quarterly_performance_worker.clear


@st.cache_data
def _canonical_metadata_worker(dataset_id: str, processed_dir: str, fp: tuple) -> dict:
    return _safe_read_json(Path(processed_dir) / "canonical_metadata.json")


def load_canonical_metadata() -> dict:
    """Load the Phase 2 canonical metadata for the active dataset."""
    c = _context()
    return _canonical_metadata_worker(c.dataset_id, str(c.processed_dir),
                                      _fingerprint(c.processed_dir / "canonical_metadata.json"))


load_canonical_metadata.clear = _canonical_metadata_worker.clear


def load_analysis_context():
    """Reconstruct a Phase 2 AnalysisContext from persisted canonical metadata + dataframe.

    Returns None if canonical metadata is not available (e.g. pre-Phase 2 dataset).
    """
    meta = load_canonical_metadata()
    if not meta:
        return None

    try:
        from src.data_engine.canonical import CanonicalDataset, AnalysisContext

        # Load the dataframe
        df = load_marketing_data()
        if df is None or df.empty:
            return None

        canonical = CanonicalDataset.from_metadata_dict(meta, df=df)
        return AnalysisContext.from_canonical(canonical)
    except Exception:
        return None


@st.cache_data
def _dynamic_analytics_worker(dataset_id: str, processed_dir: str, date_col: Optional[str],
                              metadata_fp: tuple, clean_fp: tuple, standardized_fp: tuple) -> dict:
    # One explicit dataset context for the whole computation: the metadata and
    # the data frame are both read from the SAME processed_dir. This worker
    # never calls a public loader, so it can never mix two datasets' inputs.
    meta = _safe_read_json(Path(processed_dir) / "canonical_metadata.json")
    if not meta:
        return {}

    try:
        from src.data_engine.canonical import CanonicalDataset, AnalysisContext
        from src.analytics.contracts import AnalyticsResult
        from src.analytics.kpi_engine import discover_kpis
        from src.analytics.visualization_engine import recommend_charts
        from src.analytics.insight_engine import discover_insights
        from src.analytics.relationship_engine import analyze_relationships

        df = _read_marketing_df(Path(processed_dir), date_col)
        if df is None or df.empty:
            return {}

        canonical = CanonicalDataset.from_metadata_dict(meta, df=df)
        context = AnalysisContext.from_canonical(canonical)

        result = AnalyticsResult(
            dataset_id=canonical.dataset_id,
            kpis=discover_kpis(context),
            charts=recommend_charts(context),
            insights=discover_insights(context),
            relationships=analyze_relationships(context),
        )
        return result.to_dict()
    except Exception:
        return {}


def load_dynamic_analytics() -> dict:
    """Run Phase 3 dynamic analytics engines and return the AnalyticsResult as a dict.

    Cached per dataset and per input-file version. Returns an empty dict if
    canonical metadata is unavailable.
    """
    c = _context()
    clean_fp, standardized_fp = _marketing_data_fingerprints(c.processed_dir)
    return _dynamic_analytics_worker(c.dataset_id, str(c.processed_dir), session_paths.get_date_column(),
                                     _fingerprint(c.processed_dir / "canonical_metadata.json"),
                                     clean_fp, standardized_fp)


load_dynamic_analytics.clear = _dynamic_analytics_worker.clear


@st.cache_data
def _prediction_report_worker(dataset_id: str, processed_dir: str, fp: tuple) -> dict:
    return _safe_read_json(Path(processed_dir) / "prediction_report.json")


def load_prediction_report() -> dict:
    """Load the Phase 4 prediction report for the active dataset.

    Returns an empty dict if the prediction report is unavailable.
    """
    c = _context()
    return _prediction_report_worker(c.dataset_id, str(c.processed_dir),
                                     _fingerprint(c.processed_dir / "prediction_report.json"))


load_prediction_report.clear = _prediction_report_worker.clear


@st.cache_data
def _phase5_analysis_worker(dataset_id: str, processed_dir: str, fp: tuple) -> dict:
    return _safe_read_json(Path(processed_dir) / "phase5_analysis.json")


def load_phase5_analysis() -> dict:
    """Load the Phase 5 anomaly + driver analysis for the active dataset.

    Returns an empty dict if the Phase 5 analysis is unavailable.
    """
    c = _context()
    return _phase5_analysis_worker(c.dataset_id, str(c.processed_dir),
                                   _fingerprint(c.processed_dir / "phase5_analysis.json"))


load_phase5_analysis.clear = _phase5_analysis_worker.clear


def pipeline_ready() -> bool:
    """Whether the active dataset has a standardized/cleaned data file -
    the only thing every page needs at minimum. Individual pages handle
    their own missing MMM-specific files (channel_attribution.csv,
    model_metrics.json) gracefully instead of gating the whole app on them."""
    data_dir = session_paths.get_processed_dir()
    candidates = [data_dir / "marketing_data_clean.csv", data_dir / "standardized_data.csv"]
    return any(p.exists() for p in candidates)
