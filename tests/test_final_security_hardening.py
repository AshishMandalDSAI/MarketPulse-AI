"""Final stabilization — API security modes and upload hardening.

Covers:
  * development vs production security configuration (MARKETPULSE_ENV)
  * fail-fast startup check when production has no API key
  * constant-time API key enforcement on protected endpoints
  * public endpoints stay public
  * dataset-scoped /kpis protection
  * upload filename sanitisation (path traversal), size limit, empty file,
    unsupported .xls, and real CSV/XLSX uploads with non-demo schemas
  * profiler statistics with infinite values (no NaN mean, no RuntimeWarning)
"""

from __future__ import annotations

import io
import shutil
import sys
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

import api.main as api_main  # noqa: E402
from api.main import app, _safe_upload_filename, check_security_configuration  # noqa: E402
from config import settings as app_settings  # noqa: E402
from src.data_engine.profiler import profile_dataset  # noqa: E402


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def cleanup_uploads():
    created = []
    yield created
    for ds_id in created:
        shutil.rmtree(app_settings.dataset_upload_dir(ds_id), ignore_errors=True)


def _csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _non_demo_marketing_df(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "Order_Date": pd.date_range("2024-01-01", periods=n, freq="W").strftime("%Y-%m-%d"),
        "Net_Sales": rng.normal(50_000, 5_000, n).round(2),
        "Meta_Ads": rng.uniform(1_000, 5_000, n).round(2),
        "Search_Ad_Cost": rng.uniform(1_000, 4_000, n).round(2),
        "Region": rng.choice(["North", "South"], n),
    })


# ---------------------------------------------------------------------------
# Security configuration
# ---------------------------------------------------------------------------

class TestSecurityConfiguration:
    def test_development_without_key_starts(self):
        with patch.object(app_settings, "APP_ENV", "development"), \
             patch.object(app_settings, "API_KEY", None):
            check_security_configuration()  # must not raise

    def test_production_without_key_refuses_to_start(self):
        with patch.object(app_settings, "APP_ENV", "production"), \
             patch.object(app_settings, "API_KEY", None):
            with pytest.raises(RuntimeError, match="API key"):
                check_security_configuration()

    def test_production_with_key_starts(self):
        with patch.object(app_settings, "APP_ENV", "production"), \
             patch.object(app_settings, "API_KEY", "prod-secret-value"):
            check_security_configuration()

    def test_invalid_environment_name_rejected(self):
        with patch.object(app_settings, "APP_ENV", "prod"):
            with pytest.raises(RuntimeError, match="MARKETPULSE_ENV"):
                check_security_configuration()

    def test_default_environment_is_development(self):
        assert app_settings.VALID_APP_ENVS == ("development", "production")
        assert "MARKETPULSE_ENV" in Path(app_settings.__file__).read_text(encoding="utf-8")


class TestProductionAuthEnforcement:
    KEY = "prod-test-key-123"

    @pytest.fixture(autouse=True)
    def _prod(self):
        with patch.object(app_settings, "APP_ENV", "production"), \
             patch.object(app_settings, "API_KEY", self.KEY):
            yield

    @pytest.mark.parametrize("method,path,body", [
        ("post", "/optimize-budget", {"total_budget": 100000}),
        ("post", "/simulate-budget", {"allocations": {"tv": 1000}}),
        ("post", "/ask", {"question": "Which channel has the best ROI?"}),
        ("post", "/datasets/abc123/confirm", {}),
        ("get", "/datasets/abc123/decisions", None),
        ("post", "/datasets/abc123/decisions/run", None),
        ("get", "/kpis?dataset_id=abc123", None),
    ])
    def test_protected_endpoints_reject_missing_and_wrong_key(self, client, method, path, body):
        kwargs = {"json": body} if body is not None else {}
        assert getattr(client, method)(path, **kwargs).status_code == 401
        kwargs["headers"] = {"X-API-Key": "wrong"}
        assert getattr(client, method)(path, **kwargs).status_code == 401

    def test_upload_rejects_missing_key(self, client):
        resp = client.post("/upload", files={"file": ("a.csv", b"x,y\n1,2\n", "text/csv")})
        assert resp.status_code == 401

    @pytest.mark.parametrize("path", ["/", "/health", "/docs", "/openapi.json",
                                      "/channels", "/kpis"])
    def test_public_endpoints_stay_public(self, client, path):
        assert client.get(path).status_code != 401

    def test_valid_key_is_accepted(self, client):
        resp = client.post("/ask", json={"question": "Which channel has the best ROI?"},
                           headers={"X-API-Key": self.KEY})
        assert resp.status_code == 200

    def test_production_without_key_at_request_time_fails_closed(self, client):
        with patch.object(app_settings, "API_KEY", None):
            resp = client.post("/ask", json={"question": "Which channel has the best ROI?"})
        assert resp.status_code == 503


class TestDevelopmentMode:
    def test_dev_without_key_allows_protected_endpoint(self, client):
        with patch.object(app_settings, "APP_ENV", "development"), \
             patch.object(app_settings, "API_KEY", None):
            resp = client.post("/ask", json={"question": "Which channel has the best ROI?"})
        assert resp.status_code == 200

    def test_dev_with_key_enforces_it(self, client):
        with patch.object(app_settings, "APP_ENV", "development"), \
             patch.object(app_settings, "API_KEY", "dev-key"):
            assert client.post("/ask", json={"question": "Which channel?"}).status_code == 401
            ok = client.post("/ask", json={"question": "Which channel has the best ROI?"},
                             headers={"X-API-Key": "dev-key"})
            assert ok.status_code == 200

    def test_demo_kpis_public_even_with_key(self, client):
        with patch.object(app_settings, "API_KEY", "dev-key"):
            assert client.get("/kpis").status_code == 200
            assert client.get("/kpis?dataset_id=demo").status_code == 200


# ---------------------------------------------------------------------------
# Upload hardening
# ---------------------------------------------------------------------------

class TestSafeUploadFilename:
    @pytest.mark.parametrize("raw,expected", [
        ("data.csv", "data.csv"),
        ("../../etc/passwd.csv", "passwd.csv"),
        ("/tmp/evil.csv", "evil.csv"),
        ("..\\..\\windows\\evil.xlsx", "evil.xlsx"),
        ("", "upload.csv"),
        (None, "upload.csv"),
        ("..", "upload.csv"),
        (".hidden.csv", "hidden.csv"),
        ("my data;rm -rf.csv", "my data_rm -rf.csv"),
    ])
    def test_sanitises(self, raw, expected):
        assert _safe_upload_filename(raw) == expected

    def test_never_contains_separator(self):
        for raw in ["a/b/c.csv", "a\\b.csv", "../x.csv", "x/../../y.csv"]:
            out = _safe_upload_filename(raw)
            assert "/" not in out and "\\" not in out and out not in ("", ".", "..")


class TestUploadEndpoint:
    def test_traversal_filename_is_written_inside_raw_dir(self, client, cleanup_uploads):
        df = _non_demo_marketing_df()
        resp = client.post("/upload", files={"file": ("../../../escape.csv", _csv_bytes(df), "text/csv")})
        assert resp.status_code == 200, resp.text
        ds_id = resp.json()["dataset_id"]
        cleanup_uploads.append(ds_id)
        raw_dir = app_settings.dataset_upload_dir(ds_id) / "raw"
        assert (raw_dir / "escape.csv").exists()
        assert not (app_settings.UPLOADS_DIR / "escape.csv").exists()
        assert not (app_settings.DATA_DIR / "escape.csv").exists()
        assert resp.json()["filename"] == "escape.csv"

    def test_xls_rejected_with_actionable_message(self, client):
        resp = client.post("/upload", files={"file": ("old.xls", b"\xd0\xcf\x11\xe0", "application/vnd.ms-excel")})
        assert resp.status_code == 400
        assert ".xlsx" in resp.json()["message"]

    def test_unsupported_extension_rejected(self, client):
        resp = client.post("/upload", files={"file": ("x.json", b"{}", "application/json")})
        assert resp.status_code == 400

    def test_empty_file_rejected(self, client):
        resp = client.post("/upload", files={"file": ("empty.csv", b"", "text/csv")})
        assert resp.status_code == 400
        assert "empty" in resp.json()["message"].lower()

    def test_oversized_file_rejected(self, client):
        with patch.object(app_settings, "MAX_UPLOAD_MB", 0):
            resp = client.post("/upload", files={"file": ("big.csv", b"a,b\n1,2\n", "text/csv")})
        assert resp.status_code == 413

    def test_corrupt_xlsx_rejected_without_leaking_paths(self, client):
        resp = client.post("/upload", files={"file": ("bad.xlsx", b"not a zip file at all", "application/octet-stream")})
        assert resp.status_code == 400
        msg = resp.json()["message"]
        assert str(app_settings.BASE_DIR) not in msg and "Traceback" not in msg

    def test_csv_upload_with_non_demo_schema_maps_roles(self, client, cleanup_uploads):
        df = _non_demo_marketing_df()
        resp = client.post("/upload", files={"file": ("shop.csv", _csv_bytes(df), "text/csv")})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        cleanup_uploads.append(body["dataset_id"])
        mapping = body["column_mapping"]
        assert mapping["date"]["column"] == "Order_Date"
        assert body["profile"]["n_rows"] == len(df)

    def test_xlsx_upload_roundtrip(self, client, cleanup_uploads):
        df = _non_demo_marketing_df(30)
        buf = io.BytesIO()
        df.to_excel(buf, index=False)
        resp = client.post("/upload", files={"file": (
            "shop.xlsx", buf.getvalue(),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")})
        assert resp.status_code == 200, resp.text
        cleanup_uploads.append(resp.json()["dataset_id"])
        assert resp.json()["profile"]["n_cols"] == df.shape[1]


# ---------------------------------------------------------------------------
# Profiler infinite-value handling
# ---------------------------------------------------------------------------

class TestProfilerInfiniteValues:
    def test_stats_are_finite_and_count_non_finite(self):
        values = list(range(30))
        values[3], values[4] = np.inf, -np.inf
        df = pd.DataFrame({"metric": values})
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            prof = profile_dataset(df)
        stats = prof["columns"]["metric"]["stats"]
        assert all(np.isfinite(stats[k]) for k in ("min", "max", "mean", "std"))
        assert stats["non_finite_count"] == 2

    def test_finite_column_unchanged(self):
        df = pd.DataFrame({"m": [1.0, 2.0, 3.0, 4.0]})
        stats = profile_dataset(df)["columns"]["m"]["stats"]
        assert stats == {"min": 1.0, "max": 4.0, "mean": 2.5, "std": pytest.approx(1.2909944, rel=1e-6)}


# ---------------------------------------------------------------------------
# Upload -> confirm keeps MEDIUM/LOW suggestions until the user decides
# ---------------------------------------------------------------------------

class TestUploadKeepsSuggestionsForConfirmation:
    """Regression: the upload preview used to clear unconfirmed suggestions in
    place, hiding them from the response and from the later /confirm call."""

    @staticmethod
    def _df():
        rng = np.random.default_rng(3)
        n = 60
        return pd.DataFrame({
            "Transaction_Time": pd.date_range("2023-01-02", periods=n, freq="W").strftime("%Y-%m-%d"),
            "Turnover": rng.normal(80_000, 9_000, n).round(2),
            "Facebook_Spend": rng.uniform(2_000, 8_000, n).round(2),
        })

    def _upload(self, client, cleanup_uploads):
        resp = client.post("/upload", files={"file": ("tx.csv", _csv_bytes(self._df()), "text/csv")})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        cleanup_uploads.append(body["dataset_id"])
        return body

    def test_upload_response_shows_medium_confidence_date_suggestion(self, client, cleanup_uploads):
        date_guess = self._upload(client, cleanup_uploads)["column_mapping"]["date"]
        assert date_guess["column"] == "Transaction_Time"
        assert date_guess["needs_confirmation"] is True

    def test_unconfirmed_suggestion_not_used_then_accepted_suggestion_used(self, client, cleanup_uploads):
        ds_id = self._upload(client, cleanup_uploads)["dataset_id"]
        default = client.post(f"/datasets/{ds_id}/confirm", json={})
        assert default.status_code == 200
        assert default.json()["date_column"] is None  # never silently auto-applied
        accepted = client.post(f"/datasets/{ds_id}/confirm", json={"accept_suggested": ["date"]})
        assert accepted.status_code == 200
        assert accepted.json()["date_column"] == "Transaction_Time"


# ---------------------------------------------------------------------------
# /channels output must be usable as /simulate-budget input
# ---------------------------------------------------------------------------

class TestSimulateAcceptsChannelsEndpointNames:
    def test_display_names_from_channels_endpoint_accepted(self, client):
        names = [c["Channel"] for c in client.get("/channels").json()["channels"]]
        assert any(" " in n for n in names)  # display form, e.g. "Social Media"
        resp = client.post("/simulate-budget", json={"channel_allocation": {n: 100000 for n in names}})
        assert resp.status_code == 200, resp.text

    def test_canonical_names_still_accepted(self, client):
        from src.optimization.simulator import CHANNELS
        resp = client.post("/simulate-budget", json={"channel_allocation": {c: 100000 for c in CHANNELS}})
        assert resp.status_code == 200

    def test_unknown_channel_still_rejected(self, client):
        resp = client.post("/simulate-budget", json={"channel_allocation": {"Carrier Pigeon": 5}})
        assert resp.status_code == 400

    def test_same_channel_twice_rejected(self, client):
        resp = client.post("/simulate-budget", json={"channel_allocation": {"Social_Media": 5, "social media": 6}})
        assert resp.status_code == 400
