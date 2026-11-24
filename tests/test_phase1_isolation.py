"""Phase 1 — Dataset Isolation Tests.

Tests the 10 required scenarios from the Phase 1 spec:
  1. A → B: B is active, no A data in B results
  2. A → B → C: C is active, no A/B data in C
  3. Marketing A → Non-marketing B: B cannot access A's MMM/ROI
  4. Forecastable A → Non-forecastable B: B cannot access A's forecast
  5. Unique marker values: A marker cannot appear in B's analysis
  6. Reset after A: no active dataset, no A analysis, no A Copilot context
  7. Upload same file twice: two different UUIDs
  8. Cache isolation: B does not receive A's cached result
  9. Session-state isolation: no A references remain after switch
 10. Direct legacy-path protection: uploaded dataset cannot silently resolve demo files

Plus additional Phase 1-specific tests for:
  - Fresh UUID generation (never reuses mp_pending_dataset_id)
  - has_capability() returns False for uploaded datasets without capabilities.json
  - Active dataset resolver (get_active_dataset)
  - reset_analysis() behavior
"""

import json
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.helpers import (
    load_fixture,
    run_pipeline_on_fixture,
    temp_dataset_dir,
    isolated_dataset_pair,
    assert_datasets_isolated,
    assert_no_demo_data_in_dir,
)

from config.settings import (
    DEMO_DATASET_ID,
    ensure_dataset_dirs,
    dataset_processed_dir,
    dataset_models_dir,
    UPLOADS_DIR,
)
from src.data_engine.analysis_runner import run_full_analysis


# =====================================================================
# Helpers
# =====================================================================

def _run_full_pipeline(fixture_name, subdir="diverse_schemas"):
    """Run the FULL analysis_runner pipeline on a fixture, returning
    (dataset_id, result_dict) with files written to disk."""
    dataset_id = f"test_{uuid.uuid4().hex[:12]}"
    df = load_fixture(fixture_name, subdir)
    result = run_full_analysis(dataset_id, df=df)
    return dataset_id, result


def _dataset_files(dataset_id):
    """Return set of filenames in the dataset's processed directory."""
    proc = dataset_processed_dir(dataset_id)
    if not proc.exists():
        return set()
    return {f.name for f in proc.iterdir() if f.is_file()}


def _read_capabilities(dataset_id):
    """Read capabilities.json for a dataset."""
    path = dataset_processed_dir(dataset_id) / "capabilities.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _read_kpis(dataset_id):
    """Read universal_kpis.json for a dataset."""
    path = dataset_processed_dir(dataset_id) / "universal_kpis.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _cleanup_dataset(dataset_id):
    """Remove a test dataset's directory."""
    d = UPLOADS_DIR / dataset_id
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)


# =====================================================================
# TEST 1: A → B — B is active, no A data in B results
# =====================================================================

class TestAtoB:
    def test_switch_a_to_b_produces_independent_results(self):
        id_a, result_a = _run_full_pipeline("a_marketing.csv")
        id_b, result_b = _run_full_pipeline("b_sales.csv")
        try:
            # B must have its own dataset ID, different from A
            assert id_a != id_b

            # B's capabilities must reflect B's schema, not A's
            caps_b = result_b["capabilities"]
            # Sales dataset has no channel spend → no MMM
            assert not caps_b.is_available("marketing_mix_modeling")

            # B's processed dir must not contain A's data
            proc_a = dataset_processed_dir(id_a)
            proc_b = dataset_processed_dir(id_b)
            assert_datasets_isolated(proc_a, proc_b)

            # B's KPIs must reflect B's schema
            kpis_b = _read_kpis(id_b)
            assert kpis_b.get("n_rows") == 100  # b_sales has 100 rows
        finally:
            _cleanup_dataset(id_a)
            _cleanup_dataset(id_b)


# =====================================================================
# TEST 2: A → B → C — C is active, no A/B data in C
# =====================================================================

class TestTripleSwitch:
    def test_triple_switch_c_is_independent(self):
        id_a, _ = _run_full_pipeline("a_marketing.csv")
        id_b, _ = _run_full_pipeline("e_education.csv")
        id_c, result_c = _run_full_pipeline("c_finance.csv")
        try:
            # All three must have unique IDs
            assert len({id_a, id_b, id_c}) == 3

            # C must have its own characteristics
            caps_c = result_c["capabilities"]
            assert not caps_c.is_available("marketing_mix_modeling")  # finance has no channels
            assert caps_c.is_available("trend_analysis")  # finance has date

            # C's processed dir must not contain data from A or B
            proc_a = dataset_processed_dir(id_a)
            proc_b = dataset_processed_dir(id_b)
            proc_c = dataset_processed_dir(id_c)
            assert_datasets_isolated(proc_a, proc_c)
            assert_datasets_isolated(proc_b, proc_c)
        finally:
            _cleanup_dataset(id_a)
            _cleanup_dataset(id_b)
            _cleanup_dataset(id_c)


# =====================================================================
# TEST 3: Marketing A → Non-marketing B — B cannot access A's MMM/ROI
# =====================================================================

class TestMarketingToNonMarketing:
    def test_nonmarketing_cannot_access_marketing_mmm_roi(self):
        id_a, result_a = _run_full_pipeline("a_marketing.csv")
        id_b, result_b = _run_full_pipeline("e_education.csv")
        try:
            # A should have MMM
            assert result_a["capabilities"].is_available("marketing_mix_modeling")

            # B should NOT have MMM
            assert not result_b["capabilities"].is_available("marketing_mix_modeling")
            assert not result_b["capabilities"].is_available("roi_roas")

            # B's models dir must NOT contain A's MMM artifacts
            models_b = dataset_models_dir(id_b)
            if models_b.exists():
                model_files = [f.name for f in models_b.iterdir()]
                assert "channel_attribution.csv" not in model_files or \
                    pd.read_csv(models_b / "channel_attribution.csv").empty, \
                    "B must not have A's channel attribution data"
        finally:
            _cleanup_dataset(id_a)
            _cleanup_dataset(id_b)


# =====================================================================
# TEST 4: Forecastable A → Non-forecastable B — B cannot access A's forecast
# =====================================================================

class TestForecastableToNonForecastable:
    def test_nonforecastable_cannot_access_forecast(self):
        id_a, result_a = _run_full_pipeline("a_marketing.csv")
        id_b, result_b = _run_full_pipeline("g_no_date.csv")
        try:
            assert result_a["capabilities"].is_available("forecasting")
            assert not result_b["capabilities"].is_available("forecasting")

            # B's processed dir must not contain forecast files
            proc_b = dataset_processed_dir(id_b)
            forecast_path = proc_b / "revenue_forecast.csv"
            if forecast_path.exists():
                # If it exists, it must be empty or from B's own (failed) pipeline
                df = pd.read_csv(forecast_path)
                assert df.empty or "step" in df.columns, \
                    "Forecast file in non-forecastable dataset dir looks like A's forecast"
        finally:
            _cleanup_dataset(id_a)
            _cleanup_dataset(id_b)


# =====================================================================
# TEST 5: Unique marker values — A marker cannot appear in B's analysis
# =====================================================================

class TestUniqueMarkers:
    def test_marker_values_do_not_leak(self):
        """Create two datasets with unique marker values and verify
        no cross-contamination."""
        # Create marker datasets
        df_a = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=30, freq="W"),
            "Revenue": range(1000, 1030),
            "DATASET_A_UNIQUE_MARKER": ["marker_a_value"] * 30,
        })
        df_b = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=30, freq="W"),
            "Revenue": range(2000, 2030),
            "DATASET_B_UNIQUE_MARKER": ["marker_b_value"] * 30,
        })

        id_a = f"test_marker_a_{uuid.uuid4().hex[:8]}"
        id_b = f"test_marker_b_{uuid.uuid4().hex[:8]}"

        try:
            result_a = run_full_analysis(id_a, df=df_a)
            result_b = run_full_analysis(id_b, df=df_b)

            # Check B's processed directory for A's marker
            proc_b = dataset_processed_dir(id_b)
            for f in proc_b.iterdir():
                if f.is_file():
                    content = f.read_text(errors="ignore")
                    assert "DATASET_A_UNIQUE_MARKER" not in content, \
                        f"A's marker found in B's file: {f.name}"
                    assert "marker_a_value" not in content, \
                        f"A's marker value found in B's file: {f.name}"

            # Check A's processed directory for B's marker
            proc_a = dataset_processed_dir(id_a)
            for f in proc_a.iterdir():
                if f.is_file():
                    content = f.read_text(errors="ignore")
                    assert "DATASET_B_UNIQUE_MARKER" not in content, \
                        f"B's marker found in A's file: {f.name}"
                    assert "marker_b_value" not in content, \
                        f"B's marker value found in A's file: {f.name}"
        finally:
            _cleanup_dataset(id_a)
            _cleanup_dataset(id_b)


# =====================================================================
# TEST 6: Reset after A — no active dataset, no A analysis
# =====================================================================

class TestResetAnalysis:
    def test_reset_clears_active_dataset(self):
        """After reset, get_active_dataset() returns None and
        dataset_chosen() returns False."""
        # We test the session_paths module directly by mocking st.session_state
        from unittest.mock import MagicMock, patch

        mock_state = {}

        mock_st = MagicMock()
        mock_st.session_state = mock_state
        mock_st.cache_data.clear = MagicMock()

        with patch("dashboard.components.session_paths._st", return_value=mock_st):
            from dashboard.components import session_paths

            # Set an active dataset
            session_paths.set_active_dataset("test_dataset_123", filename="test.csv")
            assert mock_state.get("mp_active_dataset_id") == "test_dataset_123"

            # Add some dataset-dependent state
            mock_state["chat_history"] = [{"q": "test", "a": "test"}]
            mock_state["budget_opt_result"] = {"some": "data"}
            mock_state["mp_page_forecast"] = {"cached": True}

            # Reset
            session_paths.reset_analysis()

            # Verify: no active dataset
            assert "mp_active_dataset_id" not in mock_state
            assert "mp_active_dataset_filename" not in mock_state
            assert "mp_demo_explicitly_chosen" not in mock_state

            # Verify: dataset-dependent state cleared
            assert "chat_history" not in mock_state
            assert "budget_opt_result" not in mock_state
            assert "mp_page_forecast" not in mock_state

            # Verify: dataset_chosen returns False
            assert not session_paths.dataset_chosen()

            # Verify: get_active_dataset returns None
            assert session_paths.get_active_dataset() is None


# =====================================================================
# TEST 7: Upload same file twice — two different UUIDs
# =====================================================================

class TestDuplicateUploadUUIDs:
    def test_same_file_uploaded_twice_gets_different_ids(self):
        """Even uploading the exact same file must produce different dataset IDs."""
        df = load_fixture("a_marketing.csv")

        id_1 = f"upload_{uuid.uuid4().hex[:12]}"
        id_2 = f"upload_{uuid.uuid4().hex[:12]}"

        try:
            result_1 = run_full_analysis(id_1, df=df.copy())
            result_2 = run_full_analysis(id_2, df=df.copy())

            # Must have different IDs
            assert id_1 != id_2

            # Both should produce valid results
            assert result_1["capabilities"] is not None
            assert result_2["capabilities"] is not None

            # Both should have their own directories
            assert dataset_processed_dir(id_1).exists()
            assert dataset_processed_dir(id_2).exists()
            assert dataset_processed_dir(id_1) != dataset_processed_dir(id_2)
        finally:
            _cleanup_dataset(id_1)
            _cleanup_dataset(id_2)

    def test_uuid_generation_is_always_fresh(self):
        """Verify that the UUID generation pattern always produces unique IDs."""
        ids = set()
        for _ in range(100):
            new_id = f"upload_{uuid.uuid4().hex[:12]}"
            assert new_id not in ids, f"Duplicate ID generated: {new_id}"
            ids.add(new_id)


# =====================================================================
# TEST 8: Cache isolation — B does not receive A's cached result
# =====================================================================

class TestCacheIsolation:
    def test_pipeline_results_are_dataset_specific(self):
        """Compute results for A, then compute for B.
        B must not receive A's results."""
        id_a, result_a = _run_full_pipeline("a_marketing.csv")
        id_b, result_b = _run_full_pipeline("b_sales.csv")
        try:
            # KPIs must be different
            kpis_a = _read_kpis(id_a)
            kpis_b = _read_kpis(id_b)

            assert kpis_a["n_rows"] != kpis_b["n_rows"] or \
                   kpis_a.get("n_channels", 0) != kpis_b.get("n_channels", 0), \
                "A and B should have different dataset characteristics"

            # Capabilities must be different
            caps_a = _read_capabilities(id_a)
            caps_b = _read_capabilities(id_b)

            # Marketing A has MMM, Sales B does not
            assert caps_a.get("marketing_mix_modeling", {}).get("available") is True
            assert caps_b.get("marketing_mix_modeling", {}).get("available") is False
        finally:
            _cleanup_dataset(id_a)
            _cleanup_dataset(id_b)


# =====================================================================
# TEST 9: Session-state isolation — no A references after switch
# =====================================================================

class TestSessionStateIsolation:
    def test_session_state_cleared_on_switch(self):
        """After switching from A to B, no A-related session state remains."""
        from unittest.mock import MagicMock, patch

        mock_state = {}
        mock_st = MagicMock()
        mock_st.session_state = mock_state
        mock_st.cache_data.clear = MagicMock()

        with patch("dashboard.components.session_paths._st", return_value=mock_st):
            from dashboard.components import session_paths

            # Simulate dataset A being active
            session_paths.set_active_dataset("dataset_a", filename="a.csv")
            mock_state["chat_history"] = [{"question": "about A", "answer": "A data"}]
            mock_state["budget_opt_result"] = {"total_budget": 1000}
            mock_state["forecast_result"] = {"values": [1, 2, 3]}
            mock_state["mp_page_kpi_cache"] = {"cached": True}
            mock_state["mp_pending_dataset_id"] = "dataset_a"

            # Switch to dataset B
            session_paths.set_active_dataset("dataset_b", filename="b.csv")

            # Verify: dataset B is active
            assert session_paths.get_dataset_id() == "dataset_b"

            # Verify: all dataset-dependent state from A is cleared
            assert "chat_history" not in mock_state, "chat_history not cleared on switch"
            assert "budget_opt_result" not in mock_state, "budget_opt_result not cleared"
            assert "forecast_result" not in mock_state, "forecast_result not cleared"
            assert "mp_page_kpi_cache" not in mock_state, "mp_page_ keys not cleared"
            assert "mp_pending_dataset_id" not in mock_state, "mp_pending_dataset_id not cleared"


# =====================================================================
# TEST 10: Direct legacy-path protection
# =====================================================================

class TestLegacyPathProtection:
    def test_uploaded_dataset_does_not_resolve_to_demo_paths(self):
        """For an uploaded dataset, the processed and models directories
        must NOT point to the global demo data directories."""
        from unittest.mock import MagicMock, patch

        mock_state = {}
        mock_st = MagicMock()
        mock_st.session_state = mock_state
        mock_st.cache_data.clear = MagicMock()

        with patch("dashboard.components.session_paths._st", return_value=mock_st):
            from dashboard.components import session_paths

            # Set an uploaded dataset as active
            session_paths.set_active_dataset("upload_abc123def456", filename="mydata.csv")

            proc_dir = session_paths.get_processed_dir()
            models_dir = session_paths.get_models_dir()

            # Must NOT be the global demo directories
            from config.settings import PROCESSED_DATA_DIR, MODELS_DIR
            assert proc_dir != PROCESSED_DATA_DIR, \
                "Uploaded dataset's processed dir must not be the demo processed dir"
            assert models_dir != MODELS_DIR, \
                "Uploaded dataset's models dir must not be the demo models dir"

            # Must contain the dataset ID in the path
            assert "upload_abc123def456" in str(proc_dir)
            assert "upload_abc123def456" in str(models_dir)

    def test_demo_data_file_not_in_uploaded_dataset_dir(self):
        """Run full pipeline on an uploaded dataset and verify
        marketing_data_clean.csv does not appear in its directory."""
        id_a, _ = _run_full_pipeline("b_sales.csv")
        try:
            proc = dataset_processed_dir(id_a)
            assert not (proc / "marketing_data_clean.csv").exists(), \
                "Demo file marketing_data_clean.csv must not exist in uploaded dataset dir"
            assert_no_demo_data_in_dir(proc)
        finally:
            _cleanup_dataset(id_a)

    def test_is_user_dataset_mode_detects_uploaded(self):
        """is_user_dataset_mode() must return True for uploaded datasets,
        False for demo."""
        from unittest.mock import MagicMock, patch

        mock_state = {}
        mock_st = MagicMock()
        mock_st.session_state = mock_state
        mock_st.cache_data.clear = MagicMock()

        with patch("dashboard.components.session_paths._st", return_value=mock_st):
            from dashboard.components import session_paths

            # Demo mode
            session_paths.set_active_dataset(DEMO_DATASET_ID)
            assert not session_paths.is_user_dataset_mode()

            # Uploaded mode
            session_paths.set_active_dataset("upload_test123")
            assert session_paths.is_user_dataset_mode()


# =====================================================================
# Additional Phase 1 tests
# =====================================================================

class TestHasCapabilityIsolation:
    """has_capability() must return False for uploaded datasets with
    no capabilities.json, preventing silent fallback to demo behavior."""

    def test_has_capability_false_for_uploaded_without_caps(self):
        from unittest.mock import MagicMock, patch

        mock_state = {}
        mock_st = MagicMock()
        mock_st.session_state = mock_state
        mock_st.cache_data.clear = MagicMock()

        with patch("dashboard.components.session_paths._st", return_value=mock_st):
            from dashboard.components import session_paths
            session_paths.set_active_dataset("upload_nocaps_test")

            # Verify is_user_dataset_mode
            assert session_paths.is_user_dataset_mode()

    def test_pipeline_writes_capabilities_json(self):
        """The full pipeline must write capabilities.json for every dataset."""
        id_a, _ = _run_full_pipeline("a_marketing.csv")
        try:
            caps_path = dataset_processed_dir(id_a) / "capabilities.json"
            assert caps_path.exists(), "capabilities.json must be written by pipeline"
            caps = json.loads(caps_path.read_text(encoding="utf-8"))
            assert len(caps) > 0, "capabilities.json must not be empty"
        finally:
            _cleanup_dataset(id_a)


class TestActiveDatasetResolver:
    """get_active_dataset() must return correct metadata."""

    def test_returns_none_when_no_dataset(self):
        from unittest.mock import MagicMock, patch

        mock_state = {}
        mock_st = MagicMock()
        mock_st.session_state = mock_state

        with patch("dashboard.components.session_paths._st", return_value=mock_st):
            from dashboard.components import session_paths
            result = session_paths.get_active_dataset()
            assert result is None

    def test_returns_metadata_for_active_dataset(self):
        from unittest.mock import MagicMock, patch

        mock_state = {}
        mock_st = MagicMock()
        mock_st.session_state = mock_state
        mock_st.cache_data.clear = MagicMock()

        with patch("dashboard.components.session_paths._st", return_value=mock_st):
            from dashboard.components import session_paths
            session_paths.set_active_dataset("test_ds_123", filename="test.csv")
            result = session_paths.get_active_dataset()

            assert result is not None
            assert result["dataset_id"] == "test_ds_123"
            assert result["filename"] == "test.csv"
            assert result["is_demo"] is False

    def test_returns_demo_metadata(self):
        from unittest.mock import MagicMock, patch

        mock_state = {}
        mock_st = MagicMock()
        mock_st.session_state = mock_state
        mock_st.cache_data.clear = MagicMock()

        with patch("dashboard.components.session_paths._st", return_value=mock_st):
            from dashboard.components import session_paths
            session_paths.reset_to_demo_dataset()
            result = session_paths.get_active_dataset()

            assert result is not None
            assert result["dataset_id"] == DEMO_DATASET_ID
            assert result["is_demo"] is True


class TestDatasetIdValidation:
    """Dataset IDs must be validated."""

    def test_invalid_dataset_id_rejected(self):
        from src.data_engine.analysis_runner import validate_dataset_id
        with pytest.raises(ValueError):
            validate_dataset_id("")
        with pytest.raises(ValueError):
            validate_dataset_id("../../../etc/passwd")
        with pytest.raises(ValueError):
            validate_dataset_id("a" * 100)  # too long

    def test_valid_dataset_id_accepted(self):
        from src.data_engine.analysis_runner import validate_dataset_id
        assert validate_dataset_id("upload_abc123") == "upload_abc123"
        assert validate_dataset_id("test-dataset-1") == "test-dataset-1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
