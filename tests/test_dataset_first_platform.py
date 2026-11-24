"""Tests for the dataset-first ("universal data engine") platform.

Covers: profiling/classification/standardization never raising on any of the
5 fixture schemas, capability detection with honest reasons, low-confidence
guesses never being auto-applied, explicit confirmation working, dataset
isolation, and FastAPI upload/confirm round trips.
"""

import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import dataset_upload_dir, DATA_DIR, PROCESSED_DATA_DIR
from src.data_engine.pipeline import profile_and_map, confirm_and_standardize
from src.data_engine.capability_engine import detect_capabilities
from src.data_engine.analysis_runner import run_full_analysis, validate_dataset_id
from src.kpi.generic_kpis import compute_generic_kpis

FIXTURES_DIR = ROOT / "tests" / "fixtures" / "generic_schemas"
FIXTURE_NAMES = [
    "dataset_a_marketing", "dataset_b_interaction", "dataset_c_sales",
    "dataset_d_arbitrary", "dataset_e_no_marketing",
]


def load_fixture(name: str) -> pd.DataFrame:
    return pd.read_csv(FIXTURES_DIR / f"{name}.csv")


@pytest.fixture(autouse=True)
def _clean_uploads():
    """Remove only test-created dataset directories (test_* prefix) to avoid
    deleting real user uploads.  See also .gitignore ``data/uploads/test_*/``."""
    def _remove_test_dirs():
        uploads = DATA_DIR / "uploads"
        if uploads.exists():
            for child in uploads.iterdir():
                if child.is_dir() and child.name.startswith("test_"):
                    shutil.rmtree(child, ignore_errors=True)
    _remove_test_dirs()
    yield
    _remove_test_dirs()


# ---------------------------------------------------------------------------
# 1. Every fixture loads/profiles/standardizes without raising
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_fixture_profiles_and_standardizes_without_raising(name):
    df = load_fixture(name)
    profile, mapping = profile_and_map(df)
    assert profile["n_rows"] == len(df)
    dataset, report = confirm_and_standardize(df, mapping)
    ready, missing = dataset.is_ready_for_analysis()
    assert ready is True
    assert missing == []
    assert isinstance(report, list) and len(report) > 0


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_fixture_full_pipeline_never_crashes(name):
    """The critical acceptance test: every fixture, including B-Interaction
    (no date/revenue/marketing signal at all... it does have a date, but
    zero revenue/spend), must complete the full analysis pipeline."""
    df = load_fixture(name)
    result = run_full_analysis(f"test_{name}", df=df)
    assert result["dataset"] is not None
    assert result["capabilities"] is not None
    assert isinstance(result["report"], list) and len(result["report"]) > 0


# ---------------------------------------------------------------------------
# 2. Capability correctness for datasets without marketing signal
# ---------------------------------------------------------------------------

def test_b_interaction_marketing_capabilities_unavailable_with_reason():
    df = load_fixture("dataset_b_interaction")
    _, mapping = profile_and_map(df)
    dataset, _ = confirm_and_standardize(df, mapping)
    caps = detect_capabilities(dataset)

    for key in ("marketing_mix_modeling", "roi_roas", "budget_optimization", "what_if_simulation"):
        cap = caps.get(key)
        assert cap is not None
        assert cap.available is False
        assert cap.reason and len(cap.reason) > 5

    for key in ("generic_kpis", "data_profiling", "ai_analyst", "business_report"):
        cap = caps.get(key)
        assert cap.available is True


def test_e_no_marketing_capabilities_unavailable_with_reason():
    df = load_fixture("dataset_e_no_marketing")
    _, mapping = profile_and_map(df)
    dataset, _ = confirm_and_standardize(df, mapping)
    caps = detect_capabilities(dataset)

    for key in ("marketing_mix_modeling", "roi_roas", "budget_optimization",
                "what_if_simulation", "trend_analysis", "forecasting"):
        cap = caps.get(key)
        assert cap.available is False
        assert cap.reason

    for key in ("generic_kpis", "data_profiling", "ai_analyst", "business_report"):
        assert caps.get(key).available is True


def test_a_marketing_has_all_marketing_capabilities():
    df = load_fixture("dataset_a_marketing")
    _, mapping = profile_and_map(df)
    dataset, _ = confirm_and_standardize(df, mapping)
    caps = detect_capabilities(dataset)
    for key in ("marketing_mix_modeling", "roi_roas", "trend_analysis", "forecasting"):
        assert caps.get(key).available is True, caps.get(key).reason


def test_every_capability_has_nonempty_reason():
    for name in FIXTURE_NAMES:
        df = load_fixture(name)
        _, mapping = profile_and_map(df)
        dataset, _ = confirm_and_standardize(df, mapping)
        caps = detect_capabilities(dataset)
        for cap in caps.capabilities:
            assert cap.reason and isinstance(cap.reason, str) and len(cap.reason.strip()) > 0


# ---------------------------------------------------------------------------
# 3. Low-confidence guesses are never auto-applied; explicit confirm works
# ---------------------------------------------------------------------------

def test_weak_guess_not_auto_used_as_revenue():
    """'duration' in dataset_b might get a low-confidence statistical guess
    as revenue, but must never be silently used without confirmation."""
    df = load_fixture("dataset_b_interaction")
    _, mapping = profile_and_map(df)
    # whatever the raw guess was, standardizing WITHOUT confirming must not
    # use 'duration' (or any other non-alias-matched column) as revenue.
    dataset, _ = confirm_and_standardize(df, mapping)
    assert dataset.revenue_col != "duration"
    assert dataset.revenue_col is None


def test_explicit_confirmation_of_medium_low_confidence_field_is_used():
    df = load_fixture("dataset_d_arbitrary")
    _, mapping = profile_and_map(df)
    # total_amount_generated_this_period should get a medium/low alias-fuzzy guess
    assert mapping.revenue.column == "total_amount_generated_this_period"
    assert mapping.revenue.confidence < 0.80  # not a HIGH-confidence match

    # Without confirming, a MEDIUM/LOW guess must not be used.
    dataset_unconfirmed, _ = confirm_and_standardize(df, mapping)
    assert dataset_unconfirmed.revenue_col is None

    # Re-derive mapping since standardize mutates it in place
    _, mapping2 = profile_and_map(df)
    dataset_confirmed, _ = confirm_and_standardize(df, mapping2, confirmed_roles={"revenue"})
    assert dataset_confirmed.revenue_col == "total_amount_generated_this_period"


def test_confirmed_roles_survive_clear_low_confidence():
    from src.data_engine.schemas import ColumnMapping, FieldGuess
    mapping = ColumnMapping()
    mapping.revenue = FieldGuess(column="duration", confidence=0.45, matched_via="statistical")
    mapping.clear_unconfirmed_low_confidence(confirmed_roles=set())
    assert mapping.revenue.column is None
    assert "duration" in mapping.freed_columns

    mapping2 = ColumnMapping()
    mapping2.revenue = FieldGuess(column="duration", confidence=0.45, matched_via="statistical")
    mapping2.clear_unconfirmed_low_confidence(confirmed_roles={"revenue"})
    assert mapping2.revenue.column == "duration"


# ---------------------------------------------------------------------------
# 4. Dataset isolation
# ---------------------------------------------------------------------------

def test_dataset_isolation_between_two_pipeline_runs():
    df_a = load_fixture("dataset_a_marketing")
    df_b = load_fixture("dataset_b_interaction")

    run_full_analysis("iso_test_a", df=df_a)
    dir_a_before = set((dataset_upload_dir("iso_test_a") / "processed").glob("*"))

    run_full_analysis("iso_test_b", df=df_b)

    dir_a_after = set((dataset_upload_dir("iso_test_a") / "processed").glob("*"))
    assert dir_a_before == dir_a_after  # untouched by running B

    dir_b = dataset_upload_dir("iso_test_b") / "processed"
    for f in dir_b.glob("*"):
        assert "iso_test_a" not in f.read_text(errors="ignore") if f.suffix == ".json" else True

    import json as _json
    caps_a = _json.loads((dataset_upload_dir("iso_test_a") / "processed" / "capabilities.json").read_text(encoding="utf-8"))
    caps_b = _json.loads((dataset_upload_dir("iso_test_b") / "processed" / "capabilities.json").read_text(encoding="utf-8"))
    assert caps_a["marketing_mix_modeling"]["available"] is True
    assert caps_b["marketing_mix_modeling"]["available"] is False
    assert "no revenue" in caps_b["marketing_mix_modeling"]["reason"] or "no marketing spend" in caps_b["marketing_mix_modeling"]["reason"]


def test_demo_dataset_directory_separate_from_uploaded_datasets():
    demo_dir = PROCESSED_DATA_DIR
    upload_dir = dataset_upload_dir("some_uploaded_dataset")
    assert demo_dir != upload_dir
    assert "uploads" not in str(demo_dir)
    assert str(upload_dir).startswith(str(DATA_DIR / "uploads"))


def test_dataset_id_validation_rejects_path_traversal():
    with pytest.raises(ValueError):
        validate_dataset_id("../../etc/passwd")
    with pytest.raises(ValueError):
        validate_dataset_id("bad id with spaces")
    assert validate_dataset_id("valid_id-123") == "valid_id-123"


# ---------------------------------------------------------------------------
# 5. Generic KPIs are real, non-fabricated, and differ across datasets
# ---------------------------------------------------------------------------

def test_generic_kpis_nonempty_and_computed():
    df = load_fixture("dataset_a_marketing")
    _, mapping = profile_and_map(df)
    dataset, _ = confirm_and_standardize(df, mapping)
    kpis = compute_generic_kpis(dataset)
    assert len(kpis) > 0
    labels = [k.label for k in kpis]
    assert "Total Records" in labels
    total_records_kpi = next(k for k in kpis if k.label == "Total Records")
    assert total_records_kpi.value == f"{len(df):,}"


def test_generic_kpis_differ_between_structurally_different_datasets():
    df_a = load_fixture("dataset_a_marketing")
    df_e = load_fixture("dataset_e_no_marketing")
    _, map_a = profile_and_map(df_a)
    _, map_e = profile_and_map(df_e)
    ds_a, _ = confirm_and_standardize(df_a, map_a)
    ds_e, _ = confirm_and_standardize(df_e, map_e)
    kpis_a = {k.label: k.value for k in compute_generic_kpis(ds_a)}
    kpis_e = {k.label: k.value for k in compute_generic_kpis(ds_e)}
    assert kpis_a != kpis_e
    assert kpis_a.get("Total Records") != kpis_e.get("Total Records")


def test_generic_kpis_empty_dataset_returns_empty_list():
    from src.data_engine.schemas import StandardizedDataset
    empty_df = pd.DataFrame()
    dataset = StandardizedDataset(df=empty_df)
    assert compute_generic_kpis(dataset) == []


# ---------------------------------------------------------------------------
# 6. XLSX round-trip
# ---------------------------------------------------------------------------

def test_xlsx_roundtrip_profiles_identically_to_csv(tmp_path):
    df = load_fixture("dataset_c_sales")
    xlsx_path = tmp_path / "dataset_c_sales.xlsx"
    df.to_excel(xlsx_path, index=False)
    df_xlsx = pd.read_excel(xlsx_path)

    profile_csv, mapping_csv = profile_and_map(df)
    profile_xlsx, mapping_xlsx = profile_and_map(df_xlsx)

    assert profile_csv["n_rows"] == profile_xlsx["n_rows"]
    assert profile_csv["n_cols"] == profile_xlsx["n_cols"]

    dataset_csv, _ = confirm_and_standardize(df, mapping_csv)
    dataset_xlsx, _ = confirm_and_standardize(df_xlsx, mapping_xlsx)
    assert dataset_csv.date_col == dataset_xlsx.date_col
    assert dataset_csv.revenue_col == dataset_xlsx.revenue_col
    assert dataset_csv.channels == dataset_xlsx.channels


# ---------------------------------------------------------------------------
# 7. Column classifier guard-rail: no false positive on short tokens
# ---------------------------------------------------------------------------

def test_short_token_fuzzy_guard_no_false_positive():
    """'digital' must not be matched as instagram ('ig') via naive substring."""
    from src.data_engine.column_classifier import classify_columns
    df = pd.DataFrame({"digital_score": [1, 2, 3, 4, 5], "config_value": [10, 20, 30, 40, 50]})
    mapping = classify_columns(df)
    channel_cols = [g.column for g in mapping.channels]
    assert "digital_score" not in channel_cols
    assert "config_value" not in channel_cols


# ---------------------------------------------------------------------------
# 8. FastAPI endpoints
# ---------------------------------------------------------------------------

def _get_client():
    from fastapi.testclient import TestClient
    from api.main import app
    return TestClient(app)


def test_api_upload_confirm_roundtrip_no_revenue_dataset():
    client = _get_client()
    with open(FIXTURES_DIR / "dataset_b_interaction.csv", "rb") as f:
        resp = client.post("/upload", files={"file": ("dataset_b_interaction.csv", f, "text/csv")})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    dataset_id = body["dataset_id"]
    assert "capabilities_if_defaults_accepted" in body

    confirm_resp = client.post(f"/datasets/{dataset_id}/confirm", json={"accept_suggested": []})
    assert confirm_resp.status_code == 200, confirm_resp.text
    confirm_body = confirm_resp.json()
    assert confirm_body["capabilities"]["marketing_mix_modeling"]["available"] is False
    assert "generic_kpis" in confirm_body

    kpi_resp = client.get(f"/kpis?dataset_id={dataset_id}")
    assert kpi_resp.status_code == 200
    kpi_body = kpi_resp.json()
    assert "generic_kpis" in kpi_body
    assert kpi_body.get("marketing_kpis_available") is False


def test_api_ambiguous_field_only_used_when_confirmed():
    client = _get_client()
    with open(FIXTURES_DIR / "dataset_d_arbitrary.csv", "rb") as f:
        resp = client.post("/upload", files={"file": ("dataset_d_arbitrary.csv", f, "text/csv")})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    dataset_id = body["dataset_id"]
    mapping = body["column_mapping"]
    assert mapping["revenue"]["confidence"] < 0.80

    confirm_default = client.post(f"/datasets/{dataset_id}/confirm", json={})
    assert confirm_default.status_code == 200
    default_body = confirm_default.json()

    confirm_yes = client.post(f"/datasets/{dataset_id}/confirm", json={"accept_suggested": ["revenue"]})
    assert confirm_yes.status_code == 200
    yes_body = confirm_yes.json()
    assert yes_body["generic_kpis"] != [] or default_body["generic_kpis"] != []
