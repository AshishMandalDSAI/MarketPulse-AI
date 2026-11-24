"""Reusable test helpers for MarketPulse AI Phase 0+ testing.

Provides utilities for:
- Creating temporary dataset directories with proper isolation
- Loading fixture datasets
- Asserting dataset isolation (no cross-dataset leakage)
- Asserting no stale artifacts remain from prior runs
- Asserting canonical result schemas
- Cleaning test artifacts safely
"""

import json
import os
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = ROOT / "tests" / "fixtures"
DIVERSE_FIXTURES_DIR = FIXTURES_DIR / "diverse_schemas"
GENERIC_FIXTURES_DIR = FIXTURES_DIR / "generic_schemas"


# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------

def load_fixture(name: str, subdir: str = "diverse_schemas") -> pd.DataFrame:
    """Load a fixture CSV by name from the given subdirectory."""
    path = FIXTURES_DIR / subdir / name
    if not path.exists():
        raise FileNotFoundError(f"Fixture not found: {path}")
    return pd.read_csv(path)


def fixture_path(name: str, subdir: str = "diverse_schemas") -> Path:
    """Return the Path to a fixture CSV."""
    return FIXTURES_DIR / subdir / name


def all_diverse_fixtures() -> list:
    """Return (name, path) pairs for all diverse fixture CSVs."""
    if not DIVERSE_FIXTURES_DIR.exists():
        return []
    return sorted(
        [(f.name, f) for f in DIVERSE_FIXTURES_DIR.glob("*.csv")],
        key=lambda x: x[0],
    )


# ---------------------------------------------------------------------------
# Temporary dataset directories (isolated per test)
# ---------------------------------------------------------------------------

@contextmanager
def temp_dataset_dir(dataset_id: Optional[str] = None):
    """Create a temporary dataset directory mimicking the real upload structure.

    Yields (dataset_id, base_dir, processed_dir) and cleans up after.
    The directory structure mirrors: <base>/uploads/<dataset_id>/processed/
    """
    dataset_id = dataset_id or f"test-{uuid.uuid4().hex[:12]}"
    base_dir = Path(tempfile.mkdtemp(prefix="mp_test_"))
    processed_dir = base_dir / "uploads" / dataset_id / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    try:
        yield dataset_id, base_dir, processed_dir
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


@contextmanager
def isolated_dataset_pair():
    """Create TWO isolated dataset directories for cross-dataset testing.

    Yields (id_a, proc_a, id_b, proc_b) and cleans up both after.
    """
    base_dir = Path(tempfile.mkdtemp(prefix="mp_test_pair_"))
    id_a = f"test-a-{uuid.uuid4().hex[:8]}"
    id_b = f"test-b-{uuid.uuid4().hex[:8]}"
    proc_a = base_dir / "uploads" / id_a / "processed"
    proc_b = base_dir / "uploads" / id_b / "processed"
    proc_a.mkdir(parents=True, exist_ok=True)
    proc_b.mkdir(parents=True, exist_ok=True)
    try:
        yield id_a, proc_a, id_b, proc_b
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Schema / contract assertions
# ---------------------------------------------------------------------------

def assert_has_keys(data: dict, required_keys: set, label: str = "object"):
    """Assert that a dict contains all required keys."""
    missing = required_keys - set(data.keys())
    assert not missing, f"{label} missing keys: {missing}"


def assert_json_schema(filepath: Path, required_keys: set, label: str = ""):
    """Load a JSON file and assert it contains the required top-level keys."""
    assert filepath.exists(), f"Expected file not found: {filepath}"
    with open(filepath) as f:
        data = json.load(f)
    label = label or filepath.name
    assert_has_keys(data, required_keys, label)
    return data


def assert_csv_columns(filepath: Path, required_columns: set, label: str = ""):
    """Load a CSV and assert it contains the required columns."""
    assert filepath.exists(), f"Expected file not found: {filepath}"
    df = pd.read_csv(filepath)
    label = label or filepath.name
    missing = required_columns - set(df.columns)
    assert not missing, f"{label} missing columns: {missing}"
    return df


# ---------------------------------------------------------------------------
# Dataset isolation assertions
# ---------------------------------------------------------------------------

def assert_datasets_isolated(proc_dir_a: Path, proc_dir_b: Path):
    """Assert that two processed directories have no shared file content.

    Checks that JSON and CSV files in one directory do not contain data
    from the other (by comparing file contents byte-for-byte).
    """
    files_a = set(f.name for f in proc_dir_a.iterdir() if f.is_file())
    files_b = set(f.name for f in proc_dir_b.iterdir() if f.is_file())

    # Same-named files should have different content (different datasets)
    shared_names = files_a & files_b
    for name in shared_names:
        content_a = (proc_dir_a / name).read_bytes()
        content_b = (proc_dir_b / name).read_bytes()
        assert content_a != content_b, (
            f"File '{name}' has identical content in both dataset directories — "
            f"possible data leakage between datasets."
        )


def assert_no_demo_data_in_dir(proc_dir: Path):
    """Assert that a processed directory does not contain NovaMart demo data artifacts.

    Checks for the specific hardcoded column names and file references that
    would indicate demo data leaked into an uploaded dataset's directory.
    """
    novamara_markers = [
        "marketing_data_clean.csv",
        "New_Customers",
        "Promotion_Spend",
        "Competitor_Index",
        "Economic_Index",
    ]
    for f in proc_dir.iterdir():
        if not f.is_file():
            continue
        try:
            content = f.read_text(errors="ignore")
        except Exception:
            continue
        for marker in novamara_markers:
            # Only flag if marker appears as a column name or path, not as
            # a substring of a longer word
            if marker in content:
                # Allow it in documentation/metadata that mentions the demo
                if f.suffix in (".md", ".txt"):
                    continue
                assert False, (
                    f"NovaMart demo data marker '{marker}' found in "
                    f"uploaded dataset file: {f.name}"
                )


# ---------------------------------------------------------------------------
# Stale artifact detection
# ---------------------------------------------------------------------------

def assert_no_stale_artifacts(proc_dir: Path, expected_files: set):
    """Assert that a processed directory contains only expected output files.

    Stale artifacts from a previous pipeline run for a DIFFERENT dataset
    should not remain.
    """
    actual_files = set(f.name for f in proc_dir.iterdir() if f.is_file())
    unexpected = actual_files - expected_files
    if unexpected:
        # Allow common metadata files
        metadata = {"capabilities.json", "generic_kpis.json", "universal_kpis.json",
                     "standardized_data.csv", "dataset_profile.json", "canonical_metadata.json",
                     "dynamic_analytics.json", "prediction_report.json", "phase5_analysis.json"}
        truly_unexpected = unexpected - metadata
        if truly_unexpected:
            assert False, (
                f"Stale artifacts found in processed directory: {truly_unexpected}"
            )


# ---------------------------------------------------------------------------
# Pipeline runner helper (wraps analysis_runner for testing)
# ---------------------------------------------------------------------------

def run_pipeline_on_fixture(fixture_name: str, fixture_subdir: str = "diverse_schemas",
                             confirmed_roles: Optional[set] = None):
    """Run the data engine pipeline on a fixture dataset inside a temp dir.

    Returns (dataset_id, processed_dir, profile, mapping, capabilities).
    The temp dir is NOT cleaned up — caller must handle cleanup or use
    temp_dataset_dir context manager.
    """
    import sys
    sys.path.insert(0, str(ROOT))

    from src.data_engine.profiler import profile_dataset
    from src.data_engine.column_classifier import classify_columns
    from src.data_engine.data_standardizer import standardize
    from src.data_engine.capability_engine import detect_capabilities

    df = load_fixture(fixture_name, fixture_subdir)
    profile = profile_dataset(df)
    mapping = classify_columns(df)
    dataset = standardize(df, mapping, confirmed_roles=confirmed_roles)
    capabilities = detect_capabilities(dataset)

    return df, profile, mapping, dataset, capabilities


# ---------------------------------------------------------------------------
# Dashboard navigation (upgrade PART 01)
# ---------------------------------------------------------------------------

def goto_page(at, label, stats_tab=None):
    """Select a sidebar menu entry in an AppTest and run. ``stats_tab`` (a tab label
    from dashboard.navigation.STATISTICS_TABS) also selects a Statistics tab."""
    from dashboard import navigation
    if stats_tab is not None:
        at.session_state[navigation.STATS_TAB_FORCE_KEY] = stats_tab
    return at.sidebar.radio[0].set_value(label).run()


def goto_module(at, module_key):
    """Open the place a Phase 8 module lives (a Statistics tab or its own page)."""
    from dashboard import navigation
    label, tab = navigation.page_for_module(module_key)
    return goto_page(at, label, tab)
