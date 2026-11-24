"""Upgrade PART 06: final completion checks.

A. The user's REAL Kaggle file (synthetic_mmm_weekly_india_SAT.csv): detection, roles, targets,
   mode, no fabricated money, models, optimiser bounds, page-level target switching.
   Skipped (with the reason) only when the file is absent.
B. Creative / content quality scores are a business variable, never channel spend (the defect the
   real file exposed in PART 06), without changing earlier role behaviour.
C. One-sample Z-test checked against independently hand-computed textbook values (math.erfc,
   no scipy), plus the confidence interval shown in the UI.
"""
from __future__ import annotations

import html
import io
import json
import math
import shutil
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from config.settings import DEMO_DATASET_ID, dataset_processed_dir, dataset_upload_dir
from dashboard import navigation
from src.data_engine import role_engine as RE

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "tests" / "fixtures" / "part02" / "real" / "synthetic_mmm_weekly_india_SAT.csv"
EXPOSURE_SIX = ["TV_Impressions", "YouTube_Impressions", "Facebook_Impressions", "Instagram_Impressions",
                "Print_Readership", "Radio_Listenership"]
needs_real = pytest.mark.skipif(not REAL.exists(), reason=(
    "The user's real file synthetic_mmm_weekly_india_SAT.csv is not present; place it at "
    "tests/fixtures/part02/real/ to run these checks on it."))


def _texts(at) -> str:
    return " ".join(html.unescape(str(e.value)) for coll in (at.markdown, at.info, at.warning, at.error,
                                                               at.success, at.caption) for e in coll)


def _titles(at) -> list[str]:
    out = []
    for c in at.get("plotly_chart"):
        t = json.loads(c.proto.spec)["layout"].get("title")
        out.append(((t.get("text") if isinstance(t, dict) else t) or ""))
    return out


# =========================================================================== A. real file
@pytest.fixture(scope="module")
def real_df():
    return pd.read_csv(REAL)


@needs_real
class TestRealFileDetection:
    def test_shape_and_quality(self, real_df):
        assert real_df.shape == (11232, 28)
        assert int(real_df.isna().sum().sum()) == 0 and int(real_df.duplicated().sum()) == 0
        assert int(real_df.duplicated(["Week", "Geo", "Brand", "SKU"]).sum()) == 0
        wk = pd.to_datetime(real_df["Week"], errors="coerce")
        assert wk.notna().all() and wk.nunique() == 156
        assert set(wk.drop_duplicates().sort_values().diff().dropna().dt.days) == {7}

    @pytest.mark.parametrize("col,role", [
        ("Week", "date"), ("Sales_Units", "sales_units"), ("Sales_Value", "revenue"), ("Trade_Spend", "trade_spend"),
        ("Brand", "brand"), ("SKU", "sku"), ("Geo", "geography"), ("MRP", "price"), ("Net_Price", "price"),
        ("Feature_Flag", "promotion"), ("Display_Flag", "promotion"), ("TPR_Flag", "promotion"),
        ("Weighted_Distribution", "distribution"), ("Numeric_Distribution", "distribution"),
        ("TDP", "distribution"), ("NOS", "distribution"),
        ("FB_Banner_Content_Score", "quality_score"), ("IG_Banner_Content_Score", "quality_score"),
        ("CPI", "external"), ("GDP_Growth", "external"), ("Festival_Index", "external"), ("Rainfall_Index", "external")])
    def test_column_roles(self, real_df, col, role):
        assert RE.suggest_roles(real_df).columns[col].role == role

    def test_six_exposures_and_no_channel_spend(self, real_df):
        m = RE.suggest_roles(real_df)
        assert sorted(m.group_cols("exposure")) == sorted(EXPOSURE_SIX)
        assert m.cols("ad_spend") == [] and m.cols("total_marketing_spend") == []

    def test_mode_is_exposure_with_trade_spend_explained(self, real_df):
        mode = RE.detect_mode(RE.suggest_roles(real_df))
        assert mode.mode == "exposure" and mode.spend_channels == []
        ev = " ".join(mode.evidence)
        assert "Trade_Spend" in ev and "not channel advertising spend" in ev
        assert "ROI / ROAS" in ev and "not available" in ev

    def test_target_never_silently_chosen(self, real_df):
        m = RE.suggest_roles(real_df)
        assert set(m.target_candidates) == {"Sales_Value", "Sales_Units"}
        assert m.needs_target_confirmation and list(m.targets) == []

    def test_legacy_pipeline_gets_no_spend_channels(self, real_df):
        mm = RE.apply_user_roles(RE.suggest_roles(real_df), {}, targets=["Sales_Value"])
        _, o = RE.legacy_overrides(mm)
        assert o["channel_columns"] == [] and o["revenue_column"] == "Sales_Value"


@needs_real
class TestRealFileModels:
    @pytest.mark.parametrize("target", ["Sales_Value", "Sales_Units"])
    def test_no_monetary_capability(self, real_df, target):
        from src.marketing_intel.capabilities import model_capabilities
        caps = model_capabilities(RE.apply_user_roles(RE.suggest_roles(real_df), {}, targets=[target]), target)
        for items in caps.values():
            for c in items:
                if any(w in c.key for w in ("roi", "roas", "monetary", "budget")):
                    assert not c.available, c.key
                    assert "channel-wise advertising spend" in (c.needs + c.reason)

    @pytest.mark.parametrize("target", ["Sales_Value", "Sales_Units"])
    def test_models_run_and_optimiser_respects_bounds(self, real_df, target):
        from src.marketing_intel import optimizer as O, scenarios as SC
        from src.marketing_intel.channels import run_channel_performance
        from src.marketing_intel.context import build_context
        from src.marketing_intel.effectiveness import run_effectiveness
        ctx = build_context(real_df, RE.apply_user_roles(RE.suggest_roles(real_df), {}, targets=[target]), target)
        eff = run_effectiveness(ctx)
        assert eff.status == "ok" and not eff.mmm
        assert sorted(eff.effects[eff.effects.kind == "channel"].column) == sorted(EXPOSURE_SIX)
        assert {"FB_Banner_Content_Score", "Trade_Spend"} <= set(eff.effects[eff.effects.kind != "channel"].column)
        ch = run_channel_performance(ctx, eff.effects, None)
        assert ch.status == "ok"
        assert not [c for c in ch.table.columns if any(w in c.lower() for w in ("roi", "roas", "revenue", "cost"))]
        for g in O.exposure_groups(ctx, eff.model):
            cur = O.current_levels(ctx, g.eligible) if g.eligible else {}
            tot = float(sum(cur.values()))
            mins = {c: 0.5 * v for c, v in cur.items()}
            maxs = {c: 1.5 * v for c, v in cur.items()}
            r = O.optimize_exposure(ctx, eff.model, g, tot, mins, maxs)
            if len(g.channels) < 2:
                assert r.solution_status == "not_supported" and "at least two channels" in r.message
            if r.status == "ok":
                assert math.isclose(sum(r.recommended.values()), tot, rel_tol=1e-6)
                for c, v in r.recommended.items():
                    assert mins[c] - 1e-6 <= v <= maxs[c] + 1e-6, c
        assert SC.simulate(ctx, eff.model, SC.preset_scenarios(eff.model, ctx)).status == "ok"
        assert len(SC.sensitivity(ctx, eff.model)) > 0


@pytest.fixture(scope="module")
def real_upload():
    if not REAL.exists():
        pytest.skip("real file absent")
    from src.data_engine.analysis_runner import run_full_analysis
    from src.data_engine.currency import choice_to_info, detect_currency, save_currency
    from src.data_engine.pipeline import profile_and_map
    df = pd.read_csv(REAL)
    did = f"upload_p06r{uuid.uuid4().hex[:8]}"
    targets = ["Sales_Value", "Sales_Units"]
    _, m = profile_and_map(df)
    mm = RE.apply_user_roles(RE.suggest_roles(df), {}, targets=targets)
    c, o = RE.legacy_overrides(mm)
    run_full_analysis(did, df=df, mapping=m, confirmed_roles=c, overrides=o)
    std = pd.read_csv(dataset_processed_dir(did) / "standardized_data.csv")
    RE.save_mapping(dataset_processed_dir(did), RE.apply_user_roles(RE.suggest_roles(std), {}, targets=targets))
    save_currency(dataset_processed_dir(did), choice_to_info("INR", detect_currency(df), RE.monetary_columns(mm)))
    yield {"mp_active_dataset_id": did, "mp_active_dataset_filename": REAL.name, "mp_demo_explicitly_chosen": False}
    shutil.rmtree(dataset_upload_dir(did), ignore_errors=True)
    shutil.rmtree(dataset_processed_dir(did), ignore_errors=True)


def _page(state, fragment):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=600)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    at.sidebar.radio[0].set_value(next(p for p in navigation.PAGES if fragment.lower() in p.lower())).run()
    return at


@needs_real
@pytest.mark.parametrize("fragment", ["Effectiveness", "Channel Performance", "Resource Optimizer", "Scenario"])
def test_real_upload_target_switch_and_no_roi(real_upload, fragment):
    at = _page(real_upload, fragment)
    assert not at.exception
    sel = [s for s in at.selectbox if s.label == "Target analysed on this page"]
    assert sel and set(sel[0].options) == {"Sales_Value", "Sales_Units"}
    first = sel[0].value
    other = "Sales_Units" if first == "Sales_Value" else "Sales_Value"
    sel[0].set_value(other).run()
    assert not at.exception
    text = _texts(at)
    assert "Exposure-Based Marketing Analytics" in text
    assert "ROI" in text and ("not available" in text or "not calculated" in text or "not computed" in text)
    if fragment in ("Effectiveness", "Scenario"):
        assert any(other in t for t in _titles(at)), _titles(at)
    for t in _titles(at):           # no money unit on exposure-only charts
        if any(u in t for u in ("impressions", "readers", "listeners")):
            assert "₹" not in t and "INR" not in t, t


# =========================================================================== B. quality scores
@pytest.mark.parametrize("name", ["FB_Banner_Content_Score", "IG_Banner_Content_Score", "TV_Ad_Quality_Rating",
                                  "Facebook_Sentiment", "YouTube_Creative_Score", "NPS_Score"])
def test_quality_scores_are_not_spend_or_exposure(name):
    df = pd.DataFrame({"Week": pd.date_range("2024-01-01", periods=40, freq="W").astype(str),
                       name: np.linspace(40, 90, 40), "Sales": np.linspace(100, 200, 40)})
    r = RE.suggest_roles(df).columns[name]
    assert r.role == "quality_score" and r.group == "business"


@pytest.mark.parametrize("name,role", [("TV_Spend", "ad_spend"), ("Facebook_Impressions", "impressions"),
                                       ("Trade_Spend", "trade_spend"), ("Facebook_Engagements", "engagement")])
def test_existing_roles_unchanged(name, role):
    df = pd.DataFrame({"Week": pd.date_range("2024-01-01", periods=40, freq="W").astype(str),
                       name: np.linspace(1000, 9000, 40), "Sales": np.linspace(100, 200, 40)})
    assert RE.suggest_roles(df).columns[name].role == role


# =========================================================================== C. Z-test
def _phi_sf(z: float) -> float:          # standard normal upper tail, independent of scipy
    return 0.5 * math.erfc(z / math.sqrt(2))


def _sample(n: int, mean: float, spread: float = 5.0) -> np.ndarray:
    base = np.linspace(-1, 1, n)
    base = base - base.mean()
    return mean + spread * base / base.std(ddof=1)


@pytest.mark.parametrize("alt,p_expected", [("two-sided", 0.0455003), ("greater", 0.0227501), ("less", 0.9772499)])
def test_z_textbook_example(alt, p_expected):
    """n = 36, x̄ = 102, μ0 = 100, σ = 6  →  SE = 1, z = 2.000 (classic textbook case)."""
    from src.inferential import tests as T
    x = _sample(36, 102.0)
    r = T.z_test(x, 100.0, 6.0, alternative=alt, alpha=0.05, confidence_level=0.95)
    assert r.status == "ok" and math.isclose(r.statistic, 2.0, abs_tol=1e-9)
    hand = {"two-sided": 2 * _phi_sf(2.0), "greater": _phi_sf(2.0), "less": 1 - _phi_sf(2.0)}[alt]
    assert math.isclose(r.p_value, hand, abs_tol=1e-12) and math.isclose(r.p_value, p_expected, abs_tol=1e-6)
    assert math.isclose(r.ci.lower, 102 - 1.959963985, abs_tol=1e-6)
    assert math.isclose(r.ci.upper, 102 + 1.959963985, abs_tol=1e-6)


def test_z_second_example_99pct():
    """n = 50, x̄ = 48.2, μ0 = 50, σ = 4: z = −1.8·√50/4 = −3.18198; 99% CI = 48.2 ± 2.575829·4/√50."""
    from src.inferential import tests as T
    x = _sample(50, 48.2, 3.0)
    r = T.z_test(x, 50.0, 4.0, alternative="less", confidence_level=0.99, alpha=0.01)
    z = -1.8 * math.sqrt(50) / 4
    assert math.isclose(r.statistic, z, abs_tol=1e-9)
    assert math.isclose(r.p_value, 1 - _phi_sf(z), abs_tol=1e-12)
    half = 2.5758293035 * 4 / math.sqrt(50)
    assert math.isclose(r.ci.lower, 48.2 - half, abs_tol=1e-6) and math.isclose(r.ci.upper, 48.2 + half, abs_tol=1e-6)
    assert r.p_value < 0.01


@pytest.mark.parametrize("sigma", [None, 0.0, -2.0, float("nan"), float("inf")])
def test_z_refuses_without_valid_sigma(sigma):
    from src.inferential import tests as T
    r = T.z_test(_sample(40, 10.0), 9.0, sigma)
    assert r.status != "ok" and r.statistic is None and r.p_value is None


def test_z_sample_sd_is_never_substituted():
    from src.inferential import tests as T
    x = _sample(40, 10.0, 3.0)
    a = T.z_test(x, 9.0, 1.0)
    b = T.z_test(x, 9.0, 3.0)                   # = the sample SD
    assert not math.isclose(a.statistic, b.statistic)
    assert math.isclose(a.statistic, (10 - 9) / (1 / math.sqrt(40)), rel_tol=1e-12)


def test_z_ui_shows_statistic_p_ci_and_conclusion():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=600)
    at.session_state["mp_active_dataset_id"] = DEMO_DATASET_ID
    at.session_state["mp_demo_explicitly_chosen"] = True
    at.run()
    at.session_state[navigation.STATS_TAB_FORCE_KEY] = "🔬 Inferential Statistics"
    at.sidebar.radio[0].set_value(navigation.STATISTICS).run()
    [s for s in at.selectbox if s.label == "Which test do you want to run?"][0].set_value("z").run()
    [s for s in at.selectbox if s.label == "Confidence level"][0].set_value(0.99).run()
    col = [s for s in at.selectbox if s.label == "Numeric column"][0].value
    [x for x in at.text_input if x.label.startswith("Hypothesised mean")][0].set_value("500000").run()
    [x for x in at.text_input if x.label.startswith("Known population")][0].set_value("150000").run()
    assert not at.exception
    x = pd.read_csv(ROOT / "data/processed/marketing_data_clean.csv")[col].dropna().to_numpy(float)
    se = 150000 / math.sqrt(len(x))
    z = (x.mean() - 500000) / se
    p = 2 * _phi_sf(abs(z))
    half = 2.5758293035 * se
    text = _texts(at)
    assert f"**Z-statistic:** {z:.4f}" in text
    assert f"99% confidence interval for the population mean of '{col}':** {x.mean() - half:.4g} to {x.mean() + half:.4g}" in text
    assert ("do not reject H0" in text) == (p >= 0.05) and "reject H0" in text


# =========================================================================== D. security fixes (PART 06)
def _client():
    from fastapi.testclient import TestClient
    from api.main import app
    return TestClient(app)


def test_api_security_headers_on_every_response():
    c = _client()
    for path in ("/health", "/channels", "/this-does-not-exist"):
        r = c.get(path)
        assert r.headers.get("x-content-type-options") == "nosniff", path
        assert r.headers.get("x-frame-options") == "DENY", path
        assert r.headers.get("referrer-policy") == "no-referrer", path
        assert r.headers.get("cache-control") == "no-store", path


def test_api_docs_disabled_in_production_only():
    import subprocess
    import sys as _sys
    code = ("import api.main as m; print(m.app.docs_url, m.app.openapi_url, m.app.redoc_url)")
    env = {**__import__("os").environ, "MARKETPULSE_ENV": "production", "MARKETPULSE_API_KEY": "k" * 40}
    out = subprocess.run([_sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True,
                         encoding="utf-8", timeout=120)
    assert out.stdout.strip().splitlines()[-1] == "None None None", out.stderr[-500:]
    env["MARKETPULSE_ENV"] = "development"
    out = subprocess.run([_sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True,
                         encoding="utf-8", timeout=120)
    assert out.stdout.strip().splitlines()[-1] == "/docs /openapi.json /redoc"


def test_api_row_limit_is_reported_not_silent(monkeypatch):
    from config import settings as app_settings
    monkeypatch.setattr(app_settings, "MAX_ROWS", 50)
    csv = "Week,Revenue\n" + "".join(f"2024-01-{1 + i % 28:02d},{i}\n" for i in range(80))
    r = _client().post("/upload", files={"file": ("rows.csv", csv.encode())})
    assert r.status_code == 200
    body = r.json()
    assert body["rows_received"] == 80 and body["rows_used"] == 50 and body["rows_truncated"] is True
    assert body["profile"]["n_rows"] == 50 and "only the first 50" in body["warnings"][0]
    small = _client().post("/upload", files={"file": ("ok.csv", "a,b\n1,2\n3,4\n".encode())}).json()
    assert small["rows_truncated"] is False and small["warnings"] == []


def test_api_unparseable_upload_leaves_nothing_on_disk():
    from config.settings import UPLOADS_DIR
    before = {p.name for p in UPLOADS_DIR.iterdir()}
    r = _client().post("/upload", files={"file": ("bad.xlsx", b"this is not a workbook")})
    assert r.status_code == 400
    assert {p.name for p in UPLOADS_DIR.iterdir()} == before


def test_page_csv_exports_neutralise_formula_cells():
    """A column named like a formula must not become a live formula in the 04/05/07 CSV downloads."""
    from src.common.export import dataframe_to_csv_bytes
    from src.common.headers import neutralize_headers
    from src.marketing_intel.channels import run_channel_performance
    from src.marketing_intel.context import build_context
    from src.marketing_intel.effectiveness import run_effectiveness
    evil = '=HYPERLINK("http://evil","x") TV_Spend'
    df = pd.read_csv(ROOT / "tests" / "fixtures" / "part02" / "spend_weekly.csv").rename(columns={"TV_Spend": evil})
    mm = RE.apply_user_roles(RE.suggest_roles(df), {}, targets=["Revenue"])
    ctx = build_context(df, mm, "Revenue")
    eff = run_effectiveness(ctx)
    ch = run_channel_performance(ctx, eff.effects, eff.mmm)
    for t in (eff.effects, ch.table):
        out = pd.read_csv(io.BytesIO(dataframe_to_csv_bytes(neutralize_headers(t))), dtype=str)
        cells = [v for col in out.columns for v in out[col].dropna()] + list(out.columns)
        assert not [v for v in cells if v.startswith(("=", "+", "@"))], "live formula in export"
        assert any(v == "'" + evil for v in cells)
    for page in ("marketing_effectiveness.py", "channel_performance.py", "scenario_simulator.py"):
        src = (ROOT / "dashboard" / "pages" / page).read_text(encoding="utf-8")
        assert "dataframe_to_csv_bytes(neutralize_headers(" in src and ".to_csv(index=False)" not in src, page


# =========================================================================== E. optimiser regression matrix (old 55% bug)
def _fitted(n):
    import importlib.util
    spec = importlib.util.spec_from_file_location("p04_helpers", ROOT / "tests" / "test_upgrade_part04.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.fitted({1: "one_channel.csv", 2: "two_channel.csv", 4: "spend_weekly.csv"}[n])


@pytest.mark.parametrize("n", [1, 2, 4])
@pytest.mark.parametrize("factor", [0.5, 1.0, 2.0])
def test_default_limits_never_block_or_violate(n, factor):
    """For 1, 2 and 4 channels and three budgets, the default limits are feasible (the old fixed 55%
    maximum made one or two channels infeasible), and the allocation respects every bound and the total."""
    from src.prescriptive import engine as E, limits as LIM
    per, mmm, ch, tot = _fitted(n)
    total = tot * factor
    mins, maxs, _ = LIM.default_limits(total, ch, {c: float(per[c].tail(8).mean()) for c in ch})
    assert sum(maxs.values()) >= total - 1e-6 and sum(mins.values()) <= total + 1e-6
    if n <= 2:
        assert max(maxs.values()) > 0.55 * total            # the old default could not reach the total
    r = E.optimize_allocation(per, mmm, total, mins, maxs)
    assert r.status == "ok" and r.solution_status in ("optimal", "feasible_not_proven")
    assert math.isclose(sum(r.recommended.values()), total, rel_tol=1e-9)
    for c in ch:
        assert mins[c] - 1e-6 <= r.recommended[c] <= maxs[c] + 1e-6, c


@pytest.mark.parametrize("n", [1, 2, 4])
def test_infeasible_limits_are_reported_never_violated(n):
    from src.prescriptive import engine as E
    per, mmm, ch, tot = _fitted(n)
    maxs = {c: 0.4 * tot / len(ch) for c in ch}                # maximums cover only 40% of the total
    mins = {c: 0.0 for c in ch}
    r = E.optimize_allocation(per, mmm, tot, mins, maxs)
    assert r.status != "ok" and r.solution_status == "infeasible"
    assert not r.recommended or all(r.recommended[c] <= maxs[c] + 1e-6 for c in r.recommended)
    rr = E.optimize_allocation(per, mmm, tot, mins, maxs, allow_reserve=True)   # explicit opt-in only
    assert rr.status == "ok" and rr.allow_reserve
    assert all(rr.recommended[c] <= maxs[c] + 1e-6 for c in ch)
    assert math.isclose(rr.budget_used + rr.unallocated, tot, rel_tol=1e-9) and rr.unallocated > 0
