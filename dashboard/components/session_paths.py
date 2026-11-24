"""Tracks the active dataset (built-in demo, or an uploaded dataset) for the
Streamlit session, and resolves its processed/models directories + column
mapping. Both dataset-switch functions clear all page-level cached
session_state keys and st.cache_data so no stale results leak between
datasets.

Phase 1: enhanced to be the ONE authoritative active-dataset resolver.
All dataset-dependent modules must use this module's functions.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import (
    DEMO_DATASET_ID, PROCESSED_DATA_DIR, MODELS_DIR,
    dataset_processed_dir, dataset_models_dir,
)

_SESSION_KEY_DATASET_ID = "mp_active_dataset_id"
_SESSION_KEY_DATASET_FILENAME = "mp_active_dataset_filename"
_SESSION_KEY_DEMO_CHOSEN = "mp_demo_explicitly_chosen"

# All session_state keys that hold dataset-dependent data and must be
# invalidated when switching datasets. mp_page_* prefixed keys are also
# cleared (see _clear_page_caches).
_DATASET_DEPENDENT_KEYS = [
    # Dataset identity
    "mp_pending_dataset_id",
    # Copilot / AI analyst
    "chat_history",
    # Budget optimizer
    "budget_opt_result",
    "budget_opt_table",
    # What-if simulator
    "simulator_instance",
    "simulator_result",
    # Forecast
    "forecast_result",
    # MMM
    "mmm_result",
    "mmm_model",
    # ROI
    "roi_result",
    # KPI
    "kpi_result",
    # Scenario
    "scenario_result",
    # Recommendations
    "recommendations_result",
    # Decision Intelligence (Phase 7 Part 7)
    "decision_result",
    "decision_copilot_context",
    # Model references
    "model_bundle",
    "model_metrics",
    # Generic cached results
    "analysis_result",
    "cached_df",
]


def _st():
    import streamlit as st
    return st


def get_dataset_id() -> str:
    """Return the active dataset ID, or DEMO_DATASET_ID if none set."""
    st = _st()
    return st.session_state.get(_SESSION_KEY_DATASET_ID, DEMO_DATASET_ID)


def get_active_dataset():
    """Return a dict with the active dataset's metadata, or None if no
    dataset is active (after a reset).

    This is the ONE authoritative resolver. All pages must use this
    instead of independently guessing the active dataset.
    """
    st = _st()
    dataset_id = st.session_state.get(_SESSION_KEY_DATASET_ID)
    if dataset_id is None:
        return None

    result = {
        "dataset_id": dataset_id,
        "filename": st.session_state.get(_SESSION_KEY_DATASET_FILENAME, ""),
        "is_demo": dataset_id == DEMO_DATASET_ID,
        "processed_dir": str(get_processed_dir()),
        "models_dir": str(get_models_dir()),
    }

    # Add row/column counts from universal_kpis.json if available
    info = _load_dataset_info()
    if info:
        result["n_rows"] = info.get("n_rows")
        result["n_cols"] = info.get("n_cols")
        result["date_column"] = info.get("date_column")
        result["revenue_column"] = info.get("revenue_column")

    return result


def is_user_dataset_mode() -> bool:
    """True when an uploaded (non-demo) dataset is active."""
    st = _st()
    dataset_id = st.session_state.get(_SESSION_KEY_DATASET_ID)
    return dataset_id is not None and dataset_id != DEMO_DATASET_ID


def _clear_page_caches() -> None:
    """Clear ALL dataset-dependent caches and session state.

    Phase 1: expanded to clear all known dataset-dependent keys,
    not just mp_page_* prefixed ones.
    """
    st = _st()
    try:
        st.cache_data.clear()
    except Exception:
        pass

    # Clear all known dataset-dependent session state keys
    for key in _DATASET_DEPENDENT_KEYS:
        if key in st.session_state:
            del st.session_state[key]

    # Drop any page-level cached values a page may have stashed in
    # session_state (keys prefixed mp_page_) so nothing leaks across datasets.
    for key in list(st.session_state.keys()):
        if key.startswith("mp_page_"):
            del st.session_state[key]


def set_active_dataset(dataset_id: str, filename: str = "") -> None:
    """Set the active dataset. Clears all caches and dataset-dependent state."""
    st = _st()
    st.session_state[_SESSION_KEY_DATASET_ID] = dataset_id
    st.session_state[_SESSION_KEY_DATASET_FILENAME] = filename
    st.session_state[_SESSION_KEY_DEMO_CHOSEN] = (dataset_id == DEMO_DATASET_ID)
    _clear_page_caches()


def reset_to_demo_dataset() -> None:
    """Switch back to the built-in demo dataset."""
    st = _st()
    st.session_state[_SESSION_KEY_DATASET_ID] = DEMO_DATASET_ID
    st.session_state[_SESSION_KEY_DATASET_FILENAME] = "NovaMart (demo)"
    st.session_state[_SESSION_KEY_DEMO_CHOSEN] = True
    _clear_page_caches()


def reset_analysis() -> None:
    """Clear the active dataset entirely. After this, no dataset is active.

    The UI should show: 'No dataset is currently active. Upload a CSV/XLSX
    file to begin.'

    Does NOT automatically load demo data.
    """
    st = _st()

    # Remove the dataset identity keys
    for key in [_SESSION_KEY_DATASET_ID, _SESSION_KEY_DATASET_FILENAME, _SESSION_KEY_DEMO_CHOSEN]:
        if key in st.session_state:
            del st.session_state[key]

    _clear_page_caches()


def demo_mode_explicitly_chosen() -> bool:
    st = _st()
    return bool(st.session_state.get(_SESSION_KEY_DEMO_CHOSEN, False))


def dataset_chosen() -> bool:
    """True once the user has explicitly picked demo or uploaded a dataset
    this session (used to gate the app before rendering analysis pages)."""
    st = _st()
    return _SESSION_KEY_DATASET_ID in st.session_state


def get_processed_dir() -> Path:
    dataset_id = get_dataset_id()
    if dataset_id == DEMO_DATASET_ID:
        return PROCESSED_DATA_DIR
    return dataset_processed_dir(dataset_id)


def get_models_dir() -> Path:
    dataset_id = get_dataset_id()
    if dataset_id == DEMO_DATASET_ID:
        return MODELS_DIR
    return dataset_models_dir(dataset_id)


def _load_dataset_info() -> dict:
    """Load universal_kpis.json for the active dataset."""
    info_path = get_processed_dir() / "universal_kpis.json"
    if not info_path.exists():
        return {}
    try:
        with open(info_path) as f:
            return json.load(f)
    except Exception:
        return {}


def get_date_column():
    """Returns the active dataset's confirmed date column name, or None."""
    dataset_id = get_dataset_id()
    if dataset_id == DEMO_DATASET_ID:
        return "Date"
    info = _load_dataset_info()
    return info.get("date_column") if info else None
