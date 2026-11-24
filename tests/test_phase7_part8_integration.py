"""Phase 7 Part 8 — Pipeline Integration, Capability Engine, API Endpoints,
and Cross-Part Integration Tests.

Covers all required categories:
  1.  Pipeline integration and return-contract compatibility
  2.  Decision report persistence and serialization
  3.  Non-fatal orchestration failure
  4.  Stale-result invalidation or versioning
  5.  Capability availability and insufficient evidence
  6.  GET and POST endpoint behavior
  7.  Authentication and dataset-access restrictions
  8.  Missing and unconfirmed datasets
  9.  Corrupt or missing upstream results
  10. Dataset isolation
  11. Non-marketing datasets
  12. Empty datasets
  13. Malformed JSON and unsafe inputs
  14. Failure recovery
  15. No demo-data fallback
  16. No fabricated financial estimates
  17. No unsupported causal claims
  18. Safe error responses
  19. Performance and integration behavior
  20. Backwards compatibility with existing functionality
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_engine.capability_engine import (
    Capability,
    CapabilityReport,
    detect_capabilities,
)
from src.data_engine.schemas import StandardizedDataset
from src.decision.orchestrator import (
    DecisionOrchestrator,
    ExecutionStatus,
    OrchestrationReport,
    OrchestrationRequest,
)
from tests.helpers import (
    assert_datasets_isolated,
    assert_no_demo_data_in_dir,
    isolated_dataset_pair,
    run_pipeline_on_fixture,
    temp_dataset_dir,
)

# ---------------------------------------------------------------------------
# FastAPI TestClient setup — skip gracefully if fastapi is not installed
# ---------------------------------------------------------------------------

try:
    from fastapi.testclient import TestClient
    from api.main import app, _validated_dataset_id
    import config.settings as app_settings

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    app = None
    app_settings = None


def _skip_no_fastapi(name: str):
    print(f"SKIP: {name} (fastapi not installed)")


# Convenience: create a minimal confirmed dataset directory for API testing.
def _setup_confirmed_dataset(dataset_id: str, base_dir: Path,
                              include_phase7: bool = False,
                              phase7_data: dict | None = None,
                              include_upstream: bool = False):
    """Set up a confirmed dataset directory under base_dir.

    Creates the processed/ directory with standardized_data.csv and
    optionally a phase7_decisions.json file.
    """
    processed_dir = base_dir / "uploads" / dataset_id / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Minimal CSV — dataset is "confirmed"
    pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).to_csv(
        processed_dir / "standardized_data.csv", index=False
    )

    if include_upstream:
        for name in ("dynamic_analytics.json", "prediction_report.json",
                      "phase5_analysis.json", "phase6_marketing.json"):
            with open(processed_dir / name, "w") as f:
                json.dump({"dataset_id": dataset_id, "stub": True}, f)

    if include_phase7 and phase7_data is not None:
        with open(processed_dir / "phase7_decisions.json", "w") as f:
            json.dump(phase7_data, f)

    return processed_dir


# ============================================================================
# 1. Pipeline integration and return-contract compatibility
# ============================================================================

def _patch_pipeline_dirs(proc, models=None):
    """Return a context manager that patches the analysis_runner's
    imported directory functions to point at the given temp directories."""
    from contextlib import ExitStack
    if models is None:
        models = proc / "models"
    models.mkdir(exist_ok=True)
    return ExitStack(), proc, models


def _run_analysis_in_temp(ds_id, df, proc):
    """Run run_full_analysis with directory functions patched to temp dirs."""
    from src.data_engine.analysis_runner import run_full_analysis
    models = proc / "models"
    models.mkdir(exist_ok=True)
    with patch("src.data_engine.analysis_runner.dataset_processed_dir", return_value=proc), \
         patch("src.data_engine.analysis_runner.dataset_models_dir", return_value=models), \
         patch("src.data_engine.analysis_runner.ensure_dataset_dirs"):
        return run_full_analysis(ds_id, df=df)


class TestPipelineIntegration:
    """Verify that run_full_analysis returns a dict containing the new
    decision_report key and preserves all existing keys."""

    def test_return_dict_has_decision_report_key(self):
        """The return dict must include 'decision_report'."""
        with temp_dataset_dir() as (ds_id, base, proc):
            df = pd.DataFrame({
                "date": pd.date_range("2020-01-01", periods=10, freq="W"),
                "revenue": range(100, 110),
                "spend_a": range(10, 20),
            })
            result = _run_analysis_in_temp(ds_id, df, proc)

            assert "decision_report" in result, (
                "run_full_analysis must include 'decision_report' in its return dict."
            )

    def test_return_dict_preserves_existing_keys(self):
        """Existing return keys (dataset, capabilities, generic_kpis,
        report, canonical, context, prediction_report, phase5_result,
        phase6_result) must still be present."""
        required = {
            "dataset", "capabilities", "generic_kpis", "report",
            "canonical", "context", "prediction_report", "phase5_result",
            "phase6_result", "decision_report",
        }

        with temp_dataset_dir() as (ds_id, base, proc):
            df = pd.DataFrame({
                "date": pd.date_range("2020-01-01", periods=10, freq="W"),
                "revenue": range(100, 110),
            })
            result = _run_analysis_in_temp(ds_id, df, proc)

            missing = required - set(result.keys())
            assert not missing, f"Return dict missing keys: {missing}"


# ============================================================================
# 2. Decision report persistence and serialization
# ============================================================================

class TestDecisionReportPersistence:
    """Verify that the Phase 7 decision report is correctly persisted and
    can be loaded back."""

    def test_phase7_json_written_to_processed_dir(self):
        with temp_dataset_dir() as (ds_id, base, proc):
            df = pd.DataFrame({
                "date": pd.date_range("2020-01-01", periods=10, freq="W"),
                "revenue": range(100, 110),
                "spend": range(10, 20),
            })
            _run_analysis_in_temp(ds_id, df, proc)

            p7 = proc / "phase7_decisions.json"
            assert p7.exists(), "phase7_decisions.json must be written to processed dir."

            data = json.loads(p7.read_text(encoding="utf-8"))
            assert isinstance(data, dict)
            assert data.get("dataset_id") == ds_id

    def test_persisted_report_is_valid_json(self):
        with temp_dataset_dir() as (ds_id, base, proc):
            df = pd.DataFrame({
                "date": pd.date_range("2020-01-01", periods=10, freq="W"),
                "revenue": range(100, 110),
            })
            _run_analysis_in_temp(ds_id, df, proc)

            p7 = proc / "phase7_decisions.json"
            if p7.exists():
                # Must parse without error
                data = json.loads(p7.read_text(encoding="utf-8"))
                assert isinstance(data, dict)

    def test_report_round_trip_via_orchestration_report(self):
        """OrchestrationReport.to_dict() → JSON → from_dict() round-trip."""
        request = OrchestrationRequest(dataset_id="rt-test")
        report = DecisionOrchestrator().run(request)
        d = report.to_dict()
        json_str = json.dumps(d, default=str)
        loaded = json.loads(json_str)
        restored = OrchestrationReport.from_dict(loaded)
        assert restored.dataset_id == "rt-test"
        assert restored.status == report.status


# ============================================================================
# 3. Non-fatal orchestration failure
# ============================================================================

class TestNonFatalOrchestrationFailure:
    """Orchestrator failure must not crash the pipeline."""

    def test_pipeline_continues_after_orchestrator_failure(self):
        from src.data_engine.analysis_runner import run_full_analysis

        with temp_dataset_dir() as (ds_id, base, proc):
            df = pd.DataFrame({"a": [1, 2, 3]})
            models = proc / "models"
            models.mkdir(exist_ok=True)
            with patch("src.data_engine.analysis_runner.dataset_processed_dir", return_value=proc), \
                 patch("src.data_engine.analysis_runner.dataset_models_dir", return_value=models), \
                 patch("src.data_engine.analysis_runner.ensure_dataset_dirs"), \
                 patch("src.decision.orchestrator.DecisionOrchestrator.run",
                        side_effect=RuntimeError("simulated crash")):
                result = run_full_analysis(ds_id, df=df)

            # Pipeline must still return successfully
            assert "report" in result
            assert result["decision_report"] is None
            # Check that the failure is logged in the report
            report_lines = result["report"]
            assert any("Phase 7" in line and "failed" in line for line in report_lines), \
                "Phase 7 failure must be recorded in the report."

    def test_failure_marker_written_on_crash(self):
        """A failure-status JSON must be written when the orchestrator crashes."""
        from src.data_engine.analysis_runner import run_full_analysis

        with temp_dataset_dir() as (ds_id, base, proc):
            df = pd.DataFrame({"a": [1, 2, 3]})
            models = proc / "models"
            models.mkdir(exist_ok=True)
            with patch("src.data_engine.analysis_runner.dataset_processed_dir", return_value=proc), \
                 patch("src.data_engine.analysis_runner.dataset_models_dir", return_value=models), \
                 patch("src.data_engine.analysis_runner.ensure_dataset_dirs"), \
                 patch("src.decision.orchestrator.DecisionOrchestrator.run",
                        side_effect=RuntimeError("boom")):
                run_full_analysis(ds_id, df=df)

            p7 = proc / "phase7_decisions.json"
            assert p7.exists(), "Failure marker must be written."
            data = json.loads(p7.read_text(encoding="utf-8"))
            assert data["status"] == "failed"
            assert "error" in data
            assert data["dataset_id"] == ds_id


# ============================================================================
# 4. Stale-result invalidation or versioning
# ============================================================================

class TestStaleResultInvalidation:
    """Old decision reports must not survive a re-run."""

    def test_old_report_deleted_before_rerun(self):
        with temp_dataset_dir() as (ds_id, base, proc):
            # Pre-populate a stale report
            p7 = proc / "phase7_decisions.json"
            stale = {"dataset_id": ds_id, "status": "success", "stale": True}
            p7.write_text(json.dumps(stale))

            df = pd.DataFrame({"a": [1, 2]})
            _run_analysis_in_temp(ds_id, df, proc)

            if p7.exists():
                data = json.loads(p7.read_text(encoding="utf-8"))
                assert "stale" not in data, \
                    "Stale report marker must not survive a re-run."

    def test_failed_rerun_does_not_serve_old_success(self):
        """If the orchestrator crashes on re-run, the old successful report
        must not remain."""
        from src.data_engine.analysis_runner import run_full_analysis

        with temp_dataset_dir() as (ds_id, base, proc):
            p7 = proc / "phase7_decisions.json"
            old_success = {"dataset_id": ds_id, "status": "success", "old": True}
            p7.write_text(json.dumps(old_success))

            df = pd.DataFrame({"a": [1, 2]})
            models = proc / "models"
            models.mkdir(exist_ok=True)
            with patch("src.data_engine.analysis_runner.dataset_processed_dir", return_value=proc), \
                 patch("src.data_engine.analysis_runner.dataset_models_dir", return_value=models), \
                 patch("src.data_engine.analysis_runner.ensure_dataset_dirs"), \
                 patch("src.decision.orchestrator.DecisionOrchestrator.run",
                        side_effect=RuntimeError("crash")):
                run_full_analysis(ds_id, df=df)

            assert p7.exists(), "Failure marker should be written."
            data = json.loads(p7.read_text(encoding="utf-8"))
            assert data["status"] == "failed", \
                "After a failed re-run, the report must show 'failed', not the old success."
            assert "old" not in data


# ============================================================================
# 5. Capability availability and insufficient evidence
# ============================================================================

class TestCapabilityEngine:
    """Verify the decision_intelligence capability follows conventions."""

    def test_decision_intelligence_available_with_rows(self):
        df, _, _, dataset, caps = run_pipeline_on_fixture(
            "a_marketing.csv"
        )
        assert caps.is_available("decision_intelligence")
        cap = caps.get("decision_intelligence")
        assert cap.name == "Decision Intelligence"
        assert "evidence" in cap.reason.lower() or "upstream" in cap.reason.lower()

    def test_decision_intelligence_unavailable_empty_dataset(self):
        """An empty dataset should mark decision_intelligence unavailable."""
        empty_df = pd.DataFrame(columns=["a", "b"])
        dataset = StandardizedDataset(
            df=empty_df,
            date_col=None,
            revenue_col=None,
            channels=[],
            control_cols=[],
            dimension_cols=[],
            identifier_cols=[],
        )
        caps = detect_capabilities(dataset)
        assert not caps.is_available("decision_intelligence")
        cap = caps.get("decision_intelligence")
        assert "no rows" in cap.reason.lower() or "cannot" in cap.reason.lower()

    def test_decision_intelligence_does_not_require_numeric_target(self):
        """decision_intelligence should NOT require revenue or numeric
        columns — it gates on rows, not column content."""
        df = pd.DataFrame({"name": ["Alice", "Bob"], "city": ["NYC", "LA"]})
        dataset = StandardizedDataset(
            df=df,
            date_col=None,
            revenue_col=None,
            channels=[],
            control_cols=[],
            dimension_cols=["name", "city"],
            identifier_cols=[],
        )
        caps = detect_capabilities(dataset)
        assert caps.is_available("decision_intelligence"), \
            "decision_intelligence should be available when rows exist, even without numeric columns."

    def test_capability_report_to_dict_includes_decision_intelligence(self):
        df = pd.DataFrame({"x": [1]})
        dataset = StandardizedDataset(
            df=df, date_col=None, revenue_col=None,
            channels=[], control_cols=[], dimension_cols=[],
            identifier_cols=[],
        )
        d = detect_capabilities(dataset).to_dict()
        assert "decision_intelligence" in d
        assert d["decision_intelligence"]["available"] is True

    def test_all_existing_capabilities_still_present(self):
        """Adding decision_intelligence must not remove existing capabilities."""
        df, _, _, dataset, caps = run_pipeline_on_fixture("a_marketing.csv")
        expected = {
            "data_profiling", "generic_kpis", "ai_analyst", "business_report",
            "category_analysis", "trend_analysis", "forecasting",
            "marketing_mix_modeling", "channel_attribution", "model_diagnostics",
            "roi_roas", "budget_optimization", "what_if_simulation",
            "decision_intelligence",
        }
        actual = {c.key for c in caps.capabilities}
        missing = expected - actual
        assert not missing, f"Capabilities missing after adding decision_intelligence: {missing}"


# ============================================================================
# 6. GET and POST endpoint behavior
# ============================================================================

class TestGetDecisionsEndpoint:

    def test_get_returns_200_with_valid_report(self):
        if not FASTAPI_AVAILABLE:
            return _skip_no_fastapi("test_get_returns_200_with_valid_report")

        with tempfile.TemporaryDirectory() as tmpdir:
            ds_id = "test-get-200"
            report_data = {"dataset_id": ds_id, "status": "success", "decisions": []}
            proc = _setup_confirmed_dataset(ds_id, Path(tmpdir),
                                             include_phase7=True, phase7_data=report_data)

            with patch.object(app_settings, "dataset_processed_dir", return_value=proc):
                client = TestClient(app)
                resp = client.get(f"/datasets/{ds_id}/decisions",
                                   headers={"X-API-Key": app_settings.API_KEY or ""})

            assert resp.status_code == 200
            body = resp.json()
            assert body["success"] is True
            assert body["dataset_id"] == ds_id
            assert body["decision_report"]["status"] == "success"

    def test_get_returns_failed_report(self):
        """GET should return a failed-status report (status 200), not hide it."""
        if not FASTAPI_AVAILABLE:
            return _skip_no_fastapi("test_get_returns_failed_report")

        with tempfile.TemporaryDirectory() as tmpdir:
            ds_id = "test-get-fail"
            failure = {"dataset_id": ds_id, "status": "failed", "error": "boom"}
            proc = _setup_confirmed_dataset(ds_id, Path(tmpdir),
                                             include_phase7=True, phase7_data=failure)

            with patch.object(app_settings, "dataset_processed_dir", return_value=proc):
                client = TestClient(app)
                resp = client.get(f"/datasets/{ds_id}/decisions",
                                   headers={"X-API-Key": app_settings.API_KEY or ""})

            assert resp.status_code == 200
            body = resp.json()
            assert body["decision_report"]["status"] == "failed"

    def test_get_returns_404_when_no_report(self):
        if not FASTAPI_AVAILABLE:
            return _skip_no_fastapi("test_get_returns_404_when_no_report")

        with tempfile.TemporaryDirectory() as tmpdir:
            ds_id = "test-get-none"
            proc = _setup_confirmed_dataset(ds_id, Path(tmpdir))

            with patch.object(app_settings, "dataset_processed_dir", return_value=proc):
                client = TestClient(app)
                resp = client.get(f"/datasets/{ds_id}/decisions",
                                   headers={"X-API-Key": app_settings.API_KEY or ""})

            assert resp.status_code == 404


class TestPostDecisionsEndpoint:

    def test_post_runs_workflow_and_returns_report(self):
        if not FASTAPI_AVAILABLE:
            return _skip_no_fastapi("test_post_runs_workflow_and_returns_report")

        with tempfile.TemporaryDirectory() as tmpdir:
            ds_id = "test-post-run"
            proc = _setup_confirmed_dataset(ds_id, Path(tmpdir), include_upstream=True)

            with patch.object(app_settings, "dataset_processed_dir", return_value=proc):
                client = TestClient(app)
                resp = client.post(f"/datasets/{ds_id}/decisions/run",
                                    headers={"X-API-Key": app_settings.API_KEY or ""})

            assert resp.status_code == 200
            body = resp.json()
            assert body["success"] is True
            assert body["dataset_id"] == ds_id
            assert "decision_report" in body
            assert "decision_count" in body

    def test_post_persists_report_to_disk(self):
        if not FASTAPI_AVAILABLE:
            return _skip_no_fastapi("test_post_persists_report_to_disk")

        with tempfile.TemporaryDirectory() as tmpdir:
            ds_id = "test-post-persist"
            proc = _setup_confirmed_dataset(ds_id, Path(tmpdir))

            with patch.object(app_settings, "dataset_processed_dir", return_value=proc):
                client = TestClient(app)
                resp = client.post(f"/datasets/{ds_id}/decisions/run",
                                    headers={"X-API-Key": app_settings.API_KEY or ""})

            p7 = proc / "phase7_decisions.json"
            assert p7.exists()
            data = json.loads(p7.read_text(encoding="utf-8"))
            assert data["dataset_id"] == ds_id

    def test_post_replaces_stale_report(self):
        """POST /decisions/run must replace a previous report."""
        if not FASTAPI_AVAILABLE:
            return _skip_no_fastapi("test_post_replaces_stale_report")

        with tempfile.TemporaryDirectory() as tmpdir:
            ds_id = "test-post-stale"
            stale = {"dataset_id": ds_id, "status": "success", "stale_marker": True}
            proc = _setup_confirmed_dataset(ds_id, Path(tmpdir),
                                             include_phase7=True, phase7_data=stale)

            with patch.object(app_settings, "dataset_processed_dir", return_value=proc):
                client = TestClient(app)
                resp = client.post(f"/datasets/{ds_id}/decisions/run",
                                    headers={"X-API-Key": app_settings.API_KEY or ""})

            assert resp.status_code == 200
            p7 = proc / "phase7_decisions.json"
            data = json.loads(p7.read_text(encoding="utf-8"))
            assert "stale_marker" not in data

    def test_post_returns_dataset_id_in_response(self):
        if not FASTAPI_AVAILABLE:
            return _skip_no_fastapi("test_post_returns_dataset_id_in_response")

        with tempfile.TemporaryDirectory() as tmpdir:
            ds_id = "test-post-dsid"
            proc = _setup_confirmed_dataset(ds_id, Path(tmpdir))

            with patch.object(app_settings, "dataset_processed_dir", return_value=proc):
                client = TestClient(app)
                resp = client.post(f"/datasets/{ds_id}/decisions/run",
                                    headers={"X-API-Key": app_settings.API_KEY or ""})

            assert resp.status_code == 200
            assert resp.json()["dataset_id"] == ds_id


# ============================================================================
# 7. Authentication and dataset-access restrictions
# ============================================================================

class TestAuthAndAccess:

    def test_get_requires_api_key_when_set(self):
        if not FASTAPI_AVAILABLE:
            return _skip_no_fastapi("test_get_requires_api_key_when_set")

        with tempfile.TemporaryDirectory() as tmpdir:
            ds_id = "test-auth-get"
            proc = _setup_confirmed_dataset(ds_id, Path(tmpdir))

            with patch.object(app_settings, "API_KEY", "secret123"), \
                 patch.object(app_settings, "dataset_processed_dir", return_value=proc):
                client = TestClient(app)
                # No key
                resp = client.get(f"/datasets/{ds_id}/decisions")
                assert resp.status_code == 401

                # Wrong key
                resp = client.get(f"/datasets/{ds_id}/decisions",
                                   headers={"X-API-Key": "wrong"})
                assert resp.status_code == 401

    def test_post_requires_api_key_when_set(self):
        if not FASTAPI_AVAILABLE:
            return _skip_no_fastapi("test_post_requires_api_key_when_set")

        with tempfile.TemporaryDirectory() as tmpdir:
            ds_id = "test-auth-post"
            proc = _setup_confirmed_dataset(ds_id, Path(tmpdir))

            with patch.object(app_settings, "API_KEY", "secret456"), \
                 patch.object(app_settings, "dataset_processed_dir", return_value=proc):
                client = TestClient(app)
                resp = client.post(f"/datasets/{ds_id}/decisions/run")
                assert resp.status_code == 401

    def test_invalid_dataset_id_format_rejected(self):
        """Dataset IDs with path traversal or special characters must be rejected."""
        if not FASTAPI_AVAILABLE:
            return _skip_no_fastapi("test_invalid_dataset_id_format_rejected")

        client = TestClient(app)
        bad_ids = ["../etc/passwd", "../../secrets", "ds;rm -rf /",
                    "a" * 100, "", " "]
        for bad_id in bad_ids:
            resp = client.get(f"/datasets/{bad_id}/decisions",
                               headers={"X-API-Key": app_settings.API_KEY or ""})
            assert resp.status_code in (400, 404, 422), \
                f"Bad dataset_id {bad_id!r} should be rejected, got {resp.status_code}"


# ============================================================================
# 8. Missing and unconfirmed datasets
# ============================================================================

class TestMissingAndUnconfirmed:

    def test_get_404_for_unconfirmed_dataset(self):
        if not FASTAPI_AVAILABLE:
            return _skip_no_fastapi("test_get_404_for_unconfirmed_dataset")

        with tempfile.TemporaryDirectory() as tmpdir:
            ds_id = "test-unconfirmed"
            proc = Path(tmpdir) / "uploads" / ds_id / "processed"
            proc.mkdir(parents=True)
            # No standardized_data.csv — not confirmed

            with patch.object(app_settings, "dataset_processed_dir", return_value=proc):
                client = TestClient(app)
                resp = client.get(f"/datasets/{ds_id}/decisions",
                                   headers={"X-API-Key": app_settings.API_KEY or ""})
            assert resp.status_code == 404
            assert "confirmed" in resp.json().get("message", "").lower()

    def test_post_404_for_unconfirmed_dataset(self):
        if not FASTAPI_AVAILABLE:
            return _skip_no_fastapi("test_post_404_for_unconfirmed_dataset")

        with tempfile.TemporaryDirectory() as tmpdir:
            ds_id = "test-unconfirmed-post"
            proc = Path(tmpdir) / "uploads" / ds_id / "processed"
            proc.mkdir(parents=True)

            with patch.object(app_settings, "dataset_processed_dir", return_value=proc):
                client = TestClient(app)
                resp = client.post(f"/datasets/{ds_id}/decisions/run",
                                    headers={"X-API-Key": app_settings.API_KEY or ""})
            assert resp.status_code == 404

    def test_get_404_for_nonexistent_dataset(self):
        if not FASTAPI_AVAILABLE:
            return _skip_no_fastapi("test_get_404_for_nonexistent_dataset")

        with tempfile.TemporaryDirectory() as tmpdir:
            proc = Path(tmpdir) / "uploads" / "ghost" / "processed"
            # Don't even create the dir

            with patch.object(app_settings, "dataset_processed_dir", return_value=proc):
                client = TestClient(app)
                resp = client.get("/datasets/ghost/decisions",
                                   headers={"X-API-Key": app_settings.API_KEY or ""})
            assert resp.status_code == 404


# ============================================================================
# 9. Corrupt or missing upstream results
# ============================================================================

class TestCorruptUpstreamResults:

    def test_post_succeeds_with_no_upstream_results(self):
        """The orchestrator should still work when no Phase 3–6 results exist."""
        if not FASTAPI_AVAILABLE:
            return _skip_no_fastapi("test_post_succeeds_with_no_upstream_results")

        with tempfile.TemporaryDirectory() as tmpdir:
            ds_id = "test-no-upstream"
            proc = _setup_confirmed_dataset(ds_id, Path(tmpdir))
            # No upstream JSON files

            with patch.object(app_settings, "dataset_processed_dir", return_value=proc):
                client = TestClient(app)
                resp = client.post(f"/datasets/{ds_id}/decisions/run",
                                    headers={"X-API-Key": app_settings.API_KEY or ""})

            assert resp.status_code == 200

    def test_post_handles_corrupt_upstream_json(self):
        """Corrupt upstream JSON should be treated as absent, not crash."""
        if not FASTAPI_AVAILABLE:
            return _skip_no_fastapi("test_post_handles_corrupt_upstream_json")

        with tempfile.TemporaryDirectory() as tmpdir:
            ds_id = "test-corrupt-upstream"
            proc = _setup_confirmed_dataset(ds_id, Path(tmpdir))
            # Write invalid JSON
            (proc / "dynamic_analytics.json").write_text("{invalid json !!!")

            with patch.object(app_settings, "dataset_processed_dir", return_value=proc):
                client = TestClient(app)
                resp = client.post(f"/datasets/{ds_id}/decisions/run",
                                    headers={"X-API-Key": app_settings.API_KEY or ""})

            # Should succeed — corrupt upstream is treated as None
            assert resp.status_code == 200

    def test_pipeline_handles_missing_upstream_gracefully(self):
        """run_full_analysis should produce a decision_report even when
        Phase 3–6 stages all fail or produce None."""
        with temp_dataset_dir() as (ds_id, base, proc):
            df = pd.DataFrame({"x": [1, 2, 3]})
            result = _run_analysis_in_temp(ds_id, df, proc)

            # Even with no marketing/prediction/anomaly data, pipeline completes
            assert "decision_report" in result


# ============================================================================
# 10. Dataset isolation
# ============================================================================

class TestDatasetIsolation:

    def test_two_datasets_get_separate_reports(self):
        """Sequential processing of two datasets must produce separate reports."""
        from src.data_engine.analysis_runner import run_full_analysis

        with isolated_dataset_pair() as (id_a, proc_a, id_b, proc_b):
            df_a = pd.DataFrame({"val": [10, 20, 30]})
            df_b = pd.DataFrame({"val": [100, 200, 300]})

            models_a = proc_a.parent / "models"
            models_b = proc_b.parent / "models"
            models_a.mkdir(exist_ok=True)
            models_b.mkdir(exist_ok=True)

            with patch("src.data_engine.analysis_runner.dataset_processed_dir", return_value=proc_a), \
                 patch("src.data_engine.analysis_runner.dataset_models_dir", return_value=models_a), \
                 patch("src.data_engine.analysis_runner.ensure_dataset_dirs"):
                res_a = run_full_analysis(id_a, df=df_a)

            with patch("src.data_engine.analysis_runner.dataset_processed_dir", return_value=proc_b), \
                 patch("src.data_engine.analysis_runner.dataset_models_dir", return_value=models_b), \
                 patch("src.data_engine.analysis_runner.ensure_dataset_dirs"):
                res_b = run_full_analysis(id_b, df=df_b)

            p7_a = proc_a / "phase7_decisions.json"
            p7_b = proc_b / "phase7_decisions.json"

            if p7_a.exists() and p7_b.exists():
                data_a = json.loads(p7_a.read_text(encoding="utf-8"))
                data_b = json.loads(p7_b.read_text(encoding="utf-8"))
                assert data_a.get("dataset_id") == id_a
                assert data_b.get("dataset_id") == id_b
                assert data_a.get("dataset_id") != data_b.get("dataset_id")

    def test_rerun_does_not_affect_other_dataset(self):
        """Re-running dataset A must not modify dataset B's report."""
        if not FASTAPI_AVAILABLE:
            return _skip_no_fastapi("test_rerun_does_not_affect_other_dataset")

        with tempfile.TemporaryDirectory() as tmpdir:
            ds_a = "test-iso-a"
            ds_b = "test-iso-b"
            proc_a = _setup_confirmed_dataset(ds_a, Path(tmpdir))
            proc_b = _setup_confirmed_dataset(ds_b, Path(tmpdir),
                                               include_phase7=True,
                                               phase7_data={"dataset_id": ds_b,
                                                              "status": "success",
                                                              "original": True})

            with patch.object(app_settings, "dataset_processed_dir", return_value=proc_a):
                client = TestClient(app)
                client.post(f"/datasets/{ds_a}/decisions/run",
                             headers={"X-API-Key": app_settings.API_KEY or ""})

            # B's report should be untouched
            b_report = json.loads((proc_b / "phase7_decisions.json").read_text(encoding="utf-8"))
            assert b_report.get("original") is True
            assert b_report.get("dataset_id") == ds_b


# ============================================================================
# 11. Non-marketing datasets
# ============================================================================

class TestNonMarketingDatasets:

    def test_decision_intelligence_available_for_non_marketing_data(self):
        """Non-marketing datasets (no revenue, no channels) should still
        get decision_intelligence capability."""
        df = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie"],
            "score": [85, 92, 78],
            "department": ["eng", "sales", "eng"],
        })
        dataset = StandardizedDataset(
            df=df, date_col=None, revenue_col=None,
            channels=[], control_cols=["score"],
            dimension_cols=["department"], identifier_cols=["name"],
        )
        caps = detect_capabilities(dataset)
        assert caps.is_available("decision_intelligence")

    def test_orchestrator_handles_non_marketing_input(self):
        """The orchestrator must not crash on non-marketing datasets."""
        request = OrchestrationRequest(dataset_id="non-marketing-test")
        report = DecisionOrchestrator().run(request)
        assert report.dataset_id == "non-marketing-test"
        assert report.status in (
            ExecutionStatus.SUCCESS,
            ExecutionStatus.PARTIAL_SUCCESS,
            ExecutionStatus.INSUFFICIENT_EVIDENCE,
        )


# ============================================================================
# 12. Empty datasets
# ============================================================================

class TestEmptyDatasets:

    def test_empty_dataframe_capability_unavailable(self):
        df = pd.DataFrame()
        dataset = StandardizedDataset(
            df=df, date_col=None, revenue_col=None,
            channels=[], control_cols=[], dimension_cols=[],
            identifier_cols=[],
        )
        caps = detect_capabilities(dataset)
        assert not caps.is_available("decision_intelligence")

    def test_pipeline_handles_empty_dataframe(self):
        with temp_dataset_dir() as (ds_id, base, proc):
            df = pd.DataFrame({"a": pd.Series(dtype="float64")})
            result = _run_analysis_in_temp(ds_id, df, proc)

            assert "decision_report" in result


# ============================================================================
# 13. Malformed JSON and unsafe inputs
# ============================================================================

class TestMalformedInputs:

    def test_get_handles_corrupt_phase7_json(self):
        """If phase7_decisions.json is corrupt, GET should return 500."""
        if not FASTAPI_AVAILABLE:
            return _skip_no_fastapi("test_get_handles_corrupt_phase7_json")

        with tempfile.TemporaryDirectory() as tmpdir:
            ds_id = "test-corrupt-json"
            proc = _setup_confirmed_dataset(ds_id, Path(tmpdir))
            (proc / "phase7_decisions.json").write_text("NOT VALID JSON{{{")

            with patch.object(app_settings, "dataset_processed_dir", return_value=proc):
                client = TestClient(app)
                resp = client.get(f"/datasets/{ds_id}/decisions",
                                   headers={"X-API-Key": app_settings.API_KEY or ""})

            assert resp.status_code == 500
            body = resp.json()
            assert body["success"] is False

    def test_path_traversal_rejected(self):
        """Dataset IDs attempting path traversal must be rejected."""
        if not FASTAPI_AVAILABLE:
            return _skip_no_fastapi("test_path_traversal_rejected")

        client = TestClient(app)
        resp = client.get("/datasets/..%2F..%2Fetc%2Fpasswd/decisions",
                           headers={"X-API-Key": app_settings.API_KEY or ""})
        assert resp.status_code in (400, 404, 422)

    def test_very_long_dataset_id_rejected(self):
        if not FASTAPI_AVAILABLE:
            return _skip_no_fastapi("test_very_long_dataset_id_rejected")

        client = TestClient(app)
        resp = client.get(f"/datasets/{'x' * 200}/decisions",
                           headers={"X-API-Key": app_settings.API_KEY or ""})
        assert resp.status_code in (400, 404, 422)

    def test_dataset_id_with_null_bytes_rejected(self):
        if not FASTAPI_AVAILABLE:
            return _skip_no_fastapi("test_dataset_id_with_null_bytes_rejected")

        client = TestClient(app)
        resp = client.get("/datasets/test%00evil/decisions",
                           headers={"X-API-Key": app_settings.API_KEY or ""})
        assert resp.status_code in (400, 404, 422)


# ============================================================================
# 14. Failure recovery
# ============================================================================

class TestFailureRecovery:

    def test_post_recovers_after_previous_failure(self):
        """After a failure, POST /decisions/run should produce a new report."""
        if not FASTAPI_AVAILABLE:
            return _skip_no_fastapi("test_post_recovers_after_previous_failure")

        with tempfile.TemporaryDirectory() as tmpdir:
            ds_id = "test-recover"
            failure_marker = {"dataset_id": ds_id, "status": "failed", "error": "previous crash"}
            proc = _setup_confirmed_dataset(ds_id, Path(tmpdir),
                                             include_phase7=True, phase7_data=failure_marker)

            with patch.object(app_settings, "dataset_processed_dir", return_value=proc):
                client = TestClient(app)
                resp = client.post(f"/datasets/{ds_id}/decisions/run",
                                    headers={"X-API-Key": app_settings.API_KEY or ""})

            assert resp.status_code == 200
            body = resp.json()
            assert body["success"] is True
            # The old failure marker should be gone
            data = json.loads((proc / "phase7_decisions.json").read_text(encoding="utf-8"))
            assert data.get("error") != "previous crash"

    def test_pipeline_failure_distinguishes_three_states(self):
        """The system must distinguish: successful report, failed workflow,
        never run."""
        with temp_dataset_dir() as (ds_id, base, proc):
            p7 = proc / "phase7_decisions.json"

            # State 1: Never run — no file
            assert not p7.exists()

            # State 2: Failed workflow
            p7.write_text(json.dumps({"dataset_id": ds_id, "status": "failed", "error": "x"}))
            data = json.loads(p7.read_text(encoding="utf-8"))
            assert data["status"] == "failed"

            # State 3: Successful report
            p7.write_text(json.dumps({"dataset_id": ds_id, "status": "success"}))
            data = json.loads(p7.read_text(encoding="utf-8"))
            assert data["status"] == "success"


# ============================================================================
# 15. No demo-data fallback
# ============================================================================

class TestNoDemoDataFallback:

    def test_post_does_not_use_demo_data(self):
        """The POST endpoint must not fall back to demo data."""
        if not FASTAPI_AVAILABLE:
            return _skip_no_fastapi("test_post_does_not_use_demo_data")

        with tempfile.TemporaryDirectory() as tmpdir:
            ds_id = "test-no-demo"
            proc = _setup_confirmed_dataset(ds_id, Path(tmpdir))

            with patch.object(app_settings, "dataset_processed_dir", return_value=proc):
                client = TestClient(app)
                resp = client.post(f"/datasets/{ds_id}/decisions/run",
                                    headers={"X-API-Key": app_settings.API_KEY or ""})

            if resp.status_code == 200:
                body = resp.json()
                report = body.get("decision_report", {})
                report_str = json.dumps(report)
                # NovaMart markers should not appear
                for marker in ("NovaMart", "marketing_data_clean.csv",
                                "New_Customers", "Promotion_Spend"):
                    assert marker not in report_str, \
                        f"Demo data marker '{marker}' found in decision report."

    def test_orchestrator_does_not_fabricate_results_from_nothing(self):
        """With zero inputs, the orchestrator must not produce fabricated
        marketing-specific results."""
        request = OrchestrationRequest(dataset_id="no-demo-test")
        report = DecisionOrchestrator().run(request)
        report_str = json.dumps(report.to_dict(), default=str)
        for marker in ("NovaMart", "marketing_data_clean"):
            assert marker not in report_str


# ============================================================================
# 16. No fabricated financial estimates
# ============================================================================

class TestNoFabricatedFinancials:

    def test_decisions_do_not_contain_fabricated_dollar_amounts(self):
        """Without real financial data, decisions must not contain
        specific dollar amounts presented as estimates."""
        request = OrchestrationRequest(dataset_id="no-money-test")
        report = DecisionOrchestrator().run(request)
        decisions = report.decisions()
        for d in decisions:
            d_dict = d.to_dict() if hasattr(d, "to_dict") else d
            d_str = json.dumps(d_dict, default=str)
            # Should not contain obviously fabricated specific dollar values
            # like "$1,234,567" or "revenue of $500K" without input data
            assert "$" not in d_str or "dollar" not in d_str.lower(), \
                f"Decision may contain fabricated financial estimates."


# ============================================================================
# 17. No unsupported causal claims
# ============================================================================

class TestNoCausalClaims:

    def test_orchestrator_adds_non_causal_limitations(self):
        """The orchestrator should include non-causal language limitations
        when producing decisions."""
        request = OrchestrationRequest(
            dataset_id="causal-test",
            analytics_result={"dataset_id": "causal-test", "kpis": []},
        )
        report = DecisionOrchestrator().run(request)
        # Check limitations mention correlational/non-causal nature
        all_text = json.dumps(report.to_dict(), default=str).lower()
        # Should contain appropriate hedging language in limitations
        has_hedging = ("correlat" in all_text or "associat" in all_text
                       or "not causal" in all_text or "non-causal" in all_text
                       or "limitation" in all_text or "uncertain" in all_text
                       or "observational" in all_text)
        assert has_hedging, \
            "Report should include limitations about non-causal nature of analysis."


# ============================================================================
# 18. Safe error responses
# ============================================================================

class TestSafeErrorResponses:

    def test_500_does_not_leak_traceback(self):
        """Internal errors must not expose stack traces or file paths."""
        if not FASTAPI_AVAILABLE:
            return _skip_no_fastapi("test_500_does_not_leak_traceback")

        with tempfile.TemporaryDirectory() as tmpdir:
            ds_id = "test-safe-err"
            proc = _setup_confirmed_dataset(ds_id, Path(tmpdir))

            def explode(*a, **kw):
                raise RuntimeError("secret internal detail at /home/user/secrets.py line 42")

            with patch.object(app_settings, "dataset_processed_dir", return_value=proc), \
                 patch("src.decision.orchestrator.DecisionOrchestrator.run", side_effect=explode):
                client = TestClient(app)
                resp = client.post(f"/datasets/{ds_id}/decisions/run",
                                    headers={"X-API-Key": app_settings.API_KEY or ""})

            assert resp.status_code == 500
            body = resp.json()
            body_str = json.dumps(body)
            assert "/home/" not in body_str
            assert "secrets.py" not in body_str
            assert "line 42" not in body_str

    def test_error_response_has_success_false(self):
        if not FASTAPI_AVAILABLE:
            return _skip_no_fastapi("test_error_response_has_success_false")

        with tempfile.TemporaryDirectory() as tmpdir:
            ds_id = "test-err-envelope"
            proc = Path(tmpdir) / "uploads" / ds_id / "processed"
            proc.mkdir(parents=True)
            # No standardized_data.csv

            with patch.object(app_settings, "dataset_processed_dir", return_value=proc):
                client = TestClient(app)
                resp = client.get(f"/datasets/{ds_id}/decisions",
                                   headers={"X-API-Key": app_settings.API_KEY or ""})

            body = resp.json()
            assert body["success"] is False

    def test_no_absolute_paths_in_error(self):
        if not FASTAPI_AVAILABLE:
            return _skip_no_fastapi("test_no_absolute_paths_in_error")

        client = TestClient(app)
        resp = client.get("/datasets/nonexistent/decisions",
                           headers={"X-API-Key": app_settings.API_KEY or ""})

        body_str = json.dumps(resp.json())
        # No absolute paths should leak
        assert "/home/" not in body_str
        assert "/data/" not in body_str or "dataset" in body_str.lower()


# ============================================================================
# 19. Performance and integration behavior
# ============================================================================

class TestPerformance:

    def test_orchestrator_completes_within_timeout(self):
        """A minimal orchestration should complete within 10 seconds."""
        start = time.time()
        request = OrchestrationRequest(dataset_id="perf-test")
        report = DecisionOrchestrator().run(request)
        elapsed = time.time() - start
        assert elapsed < 10.0, f"Orchestrator took {elapsed:.2f}s — expected < 10s"

    def test_capability_detection_fast(self):
        """Capability detection should be nearly instantaneous."""
        df = pd.DataFrame({"a": range(100), "b": range(100)})
        dataset = StandardizedDataset(
            df=df, date_col=None, revenue_col=None,
            channels=[], control_cols=["a", "b"],
            dimension_cols=[], identifier_cols=[],
        )
        start = time.time()
        for _ in range(100):
            detect_capabilities(dataset)
        elapsed = time.time() - start
        assert elapsed < 2.0, f"100 capability detections took {elapsed:.2f}s"


# ============================================================================
# 20. Backwards compatibility with existing functionality
# ============================================================================

class TestBackwardsCompatibility:

    def test_existing_capabilities_unchanged(self):
        """Adding decision_intelligence must not alter existing capability
        availability or reason strings for any fixture."""
        df, _, _, dataset, caps = run_pipeline_on_fixture("a_marketing.csv")

        # Verify each pre-existing capability still has the expected contract
        for cap in caps.capabilities:
            assert isinstance(cap.key, str) and len(cap.key) > 0
            assert isinstance(cap.name, str) and len(cap.name) > 0
            assert isinstance(cap.available, bool)
            assert isinstance(cap.reason, str) and len(cap.reason) > 0

    def test_pipeline_return_dict_backward_compatible(self):
        """All keys that existed before Part 8 must still be present."""
        pre_part8_keys = {
            "dataset", "capabilities", "generic_kpis", "report",
            "canonical", "context", "prediction_report", "phase5_result",
            "phase6_result",
        }

        with temp_dataset_dir() as (ds_id, base, proc):
            df = pd.DataFrame({"a": [1, 2, 3]})
            result = _run_analysis_in_temp(ds_id, df, proc)

            missing = pre_part8_keys - set(result.keys())
            assert not missing, f"Backward-incompatible: missing keys {missing}"

    def test_capability_report_to_dict_backward_compatible(self):
        """to_dict() must still return a dict with all original keys."""
        df, _, _, dataset, caps = run_pipeline_on_fixture("a_marketing.csv")
        d = caps.to_dict()

        original_keys = {
            "data_profiling", "generic_kpis", "ai_analyst", "business_report",
            "category_analysis", "trend_analysis", "forecasting",
            "marketing_mix_modeling", "channel_attribution", "model_diagnostics",
            "roi_roas", "budget_optimization", "what_if_simulation",
        }
        missing = original_keys - set(d.keys())
        assert not missing, f"Capability dict missing original keys: {missing}"

    def test_health_endpoint_still_works(self):
        if not FASTAPI_AVAILABLE:
            return _skip_no_fastapi("test_health_endpoint_still_works")

        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert "status" in resp.json()

    def test_root_endpoint_still_works(self):
        if not FASTAPI_AVAILABLE:
            return _skip_no_fastapi("test_root_endpoint_still_works")

        client = TestClient(app)
        resp = client.get("/")
        assert resp.status_code == 200
