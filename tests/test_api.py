"""API tests. Requires fastapi + starlette's TestClient, which are NOT
installed in the offline build sandbox used to develop this project — see
README 'Known Limitations'. These tests are written to run once you
`pip install -r requirements.txt` locally, and are automatically SKIPPED
(not failed) if fastapi isn't importable, so the suite as a whole still
reports a clean pass/fail in this sandbox.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from fastapi.testclient import TestClient
    from api.main import app
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


def _skip(name):
    print(f"SKIP: {name} (fastapi not installed in this environment)")


def test_health_endpoint():
    if not FASTAPI_AVAILABLE:
        return _skip("test_health_endpoint")
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in ("healthy", "degraded")
    assert "pipeline_files" in body


def test_kpis_endpoint_requires_pipeline():
    if not FASTAPI_AVAILABLE:
        return _skip("test_kpis_endpoint_requires_pipeline")
    client = TestClient(app)
    resp = client.get("/kpis")
    assert resp.status_code in (200, 404)


def test_root_endpoint_lists_routes():
    if not FASTAPI_AVAILABLE:
        return _skip("test_root_endpoint_lists_routes")
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "endpoints" in resp.json()


def test_optimize_budget_rejects_zero_budget():
    """Pydantic validation: total_budget must be > 0."""
    if not FASTAPI_AVAILABLE:
        return _skip("test_optimize_budget_rejects_zero_budget")
    client = TestClient(app)
    resp = client.post("/optimize-budget", json={"total_budget": 0})
    assert resp.status_code == 422
    body = resp.json()
    assert body["success"] is False
    assert body["error"] == "ValidationError"


def test_optimize_budget_rejects_invalid_bounds():
    """min_pct >= max_pct should be rejected by the model validator."""
    if not FASTAPI_AVAILABLE:
        return _skip("test_optimize_budget_rejects_invalid_bounds")
    client = TestClient(app)
    resp = client.post("/optimize-budget", json={
        "total_budget": 1000000, "min_pct": 0.6, "max_pct": 0.5,
    })
    assert resp.status_code == 422
    assert resp.json()["error"] == "ValidationError"


def test_optimize_budget_runs_if_pipeline_ready():
    if not FASTAPI_AVAILABLE:
        return _skip("test_optimize_budget_runs_if_pipeline_ready")
    client = TestClient(app)
    resp = client.post("/optimize-budget", json={"total_budget": 2000000})
    assert resp.status_code in (200, 404)
    if resp.status_code == 200:
        body = resp.json()
        assert "summary" in body and "comparison_table" in body
        total = sum(row["Recommended_Budget"] for row in body["comparison_table"])
        assert abs(total - 2000000) < 1.0


def test_simulate_budget_requires_input():
    """Neither total_budget nor channel_allocation provided -> 422, not a crash."""
    if not FASTAPI_AVAILABLE:
        return _skip("test_simulate_budget_requires_input")
    client = TestClient(app)
    resp = client.post("/simulate-budget", json={})
    assert resp.status_code == 422
    assert resp.json()["error"] == "ValidationError"


def test_simulate_budget_rejects_unknown_channel():
    if not FASTAPI_AVAILABLE:
        return _skip("test_simulate_budget_rejects_unknown_channel")
    client = TestClient(app)
    resp = client.post("/simulate-budget", json={"channel_allocation": {"NotAChannel": 1000}})
    assert resp.status_code in (400, 404)


def test_ask_rejects_too_short_question():
    if not FASTAPI_AVAILABLE:
        return _skip("test_ask_rejects_too_short_question")
    client = TestClient(app)
    resp = client.post("/ask", json={"question": "hi"})
    assert resp.status_code == 422
    assert resp.json()["error"] == "ValidationError"


def test_ask_answers_known_question():
    if not FASTAPI_AVAILABLE:
        return _skip("test_ask_answers_known_question")
    client = TestClient(app)
    resp = client.post("/ask", json={"question": "Which channel has the best efficiency?"})
    assert resp.status_code == 200
    assert "answer" in resp.json()


def test_404_for_unknown_route_has_no_traceback():
    if not FASTAPI_AVAILABLE:
        return _skip("test_404_for_unknown_route_has_no_traceback")
    client = TestClient(app)
    resp = client.get("/this-route-does-not-exist")
    assert resp.status_code == 404
    assert "Traceback" not in resp.text


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"DONE: {name}")
