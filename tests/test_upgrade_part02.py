"""PART 02 -- universal dataset engine, three analytics modes, four new marketing models,
optimizer default-limit fix.

Datasets: ``tests/fixtures/part02`` (seeded synthetic files from ``make_datasets.py``). The
``kaggle_structure_replica.csv`` file reproduces the column structure of the user's Kaggle file
``synthetic_mmm_weekly_india_SAT.csv``; it is NOT that file. When the real file is placed at
``tests/fixtures/part02/real/synthetic_mmm_weekly_india_SAT.csv`` the ``TestRealKaggleFile`` class runs the
same checks on it; otherwise those tests are skipped and say why.
"""
from __future__ import annotations

import html
import json
import math
import re
import shutil
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import dataset_processed_dir, dataset_upload_dir  # noqa: E402
from src.data_engine import role_engine as RE  # noqa: E402
from src.data_engine.role_lexicon import is_trade_or_promotion_spend_name, tokens  # noqa: E402
from src.marketing_intel import optimizer as O  # noqa: E402
from src.marketing_intel import scenarios as SC  # noqa: E402
from src.marketing_intel import stats as S  # noqa: E402
from src.marketing_intel.capabilities import model_capabilities  # noqa: E402
from src.marketing_intel.channels import run_channel_performance  # noqa: E402
from src.marketing_intel.context import build_context, period_totals, value_grain  # noqa: E402
from src.marketing_intel.effectiveness import run_effectiveness  # noqa: E402
from src.marketing_intel.response import fit_response  # noqa: E402
from src.prescriptive import limits as LIM  # noqa: E402

FIX = ROOT / "tests" / "fixtures" / "part02"
APP = str(ROOT / "app.py")
REAL_KAGGLE = FIX / "real" / "synthetic_mmm_weekly_india_SAT.csv"
EXPOSURE_SIX = ["TV_Impressions", "YouTube_Impressions", "Facebook_Impressions", "Instagram_Impressions",
                "Print_Readership", "Radio_Listenership"]
PAGES = {"marketing_effectiveness": "04 · 🧠 Marketing Effectiveness Model",
         "channel_performance": "05 · 📡 Channel Performance & Contribution",
         "resource_optimizer": "06 · ⚖️ Marketing Resource Optimizer",
         "scenario_simulator": "07 · 🧪 Marketing Scenario Simulator"}
MONEY_WORDS = re.compile(r"\b(ROI|ROAS)\b")


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(FIX / name)


def mapped(df: pd.DataFrame, targets=None) -> RE.RoleMapping:
    m = RE.suggest_roles(df)
    return RE.apply_user_roles(m, {}, targets=targets) if targets is not None else m


def ctx_for(name: str, target: str):
    df = load(name)
    return build_context(df, mapped(df, [target]), target)


# ============================================================================ I. role engine
EXPECTED = {
    # file: (mode, {column: role})
    "kaggle_structure_replica.csv": ("exposure", {
        "Week": "date", "Geo": "geography", "Brand": "brand", "SKU": "sku", "Sales_Units": "sales_units",
        "Sales_Value": "revenue", "MRP": "price", "Net_Price": "price", "Feature_Flag": "promotion",
        "Display_Flag": "promotion", "TPR_Flag": "promotion", "Trade_Spend": "trade_spend",
        "TV_Impressions": "impressions", "YouTube_Impressions": "impressions", "Facebook_Impressions": "impressions",
        "Instagram_Impressions": "impressions", "Print_Readership": "readership",
        "Radio_Listenership": "listenership", "Weighted_Distribution": "distribution",
        "Numeric_Distribution": "distribution", "TDP": "distribution", "NOS": "distribution", "CPI": "external",
        "GDP_Growth": "external", "Festival_Index": "external", "Rainfall_Index": "external"}),
    "spend_weekly.csv": ("spend", {"Week": "date", "TV_Spend": "ad_spend", "Search_Spend": "ad_spend",
                                   "Social_Spend": "ad_spend", "Radio_Spend": "ad_spend", "Revenue": "revenue"}),
    "exposure_daily.csv": ("exposure", {"Date": "date", "FB_Reach": "reach", "Meta_Views": "views",
                                        "Google_Clicks": "clicks", "YouTube_Impr": "impressions",
                                        "Conversions": "conversions"}),
    "general_sales.csv": ("general", {"Order_ID": "identifier", "Order_Date": "date", "Region": "geography",
                                      "Product": "product", "Quantity": "sales_units", "Unit_Price": "price",
                                      "Discount": "discount", "Net_Sales": "revenue"}),
    "renamed_columns.csv": ("spend", {"Period": "date", "Television_Cost": "ad_spend",
                                      "Online_Ads_Investment": "ad_spend", "Meta_Ad_Cost": "ad_spend",
                                      "Sales_Turnover": "revenue"}),
    "multi_revenue.csv": ("spend", {"Gross_Revenue": "revenue", "Net_Revenue": "revenue", "Total_Sales": "revenue",
                                    "TV_Spend": "ad_spend", "Digital_Spend": "ad_spend"}),
    "missing_marketing.csv": ("general", {"Month": "date", "Store": "geography", "Customers": "customers",
                                          "Revenue": "revenue", "Avg_Basket": "derived"}),
    "one_channel.csv": ("spend", {"Digital_Spend": "ad_spend", "Revenue": "revenue"}),
    "two_channel.csv": ("spend", {"TV_Spend": "ad_spend", "Search_Spend": "ad_spend"}),
    "brand_geo_spend.csv": ("spend", {"Week_Start": "date", "Brand": "brand", "Region": "geography",
                                      "Promo_Flag": "promotion", "Avg_Price": "price", "Units_Sold": "sales_units",
                                      "Revenue": "revenue"}),
    "mixed_types.csv": ("exposure", {"Campaign_Type": "channel_dimension", "Website_Visits": "visits",
                                     "Is_Holiday": "external", "Leads": "leads", "Notes": "text"}),
}


class TestRoleEngine:
    @pytest.mark.parametrize("name", list(EXPECTED))
    def test_roles_and_mode_per_dataset(self, name):
        mode, roles = EXPECTED[name]
        m = mapped(load(name))
        for col, role in roles.items():
            assert m.columns[col].role == role, (name, col, m.columns[col].role, m.columns[col].evidence)
        assert RE.detect_mode(m).mode == mode

    @pytest.mark.parametrize("name", list(EXPECTED))
    def test_every_column_has_confidence_evidence_and_group(self, name):
        m = mapped(load(name))
        for c, r in m.columns.items():
            assert 0.05 <= r.confidence <= 0.99
            assert r.evidence and all(isinstance(e, str) and e for e in r.evidence)
            assert r.group == RE.ROLE_INFO[r.role][0]
            assert all({"role", "label", "confidence", "why"} <= set(a) for a in r.alternatives)

    def test_spend_and_exposure_columns_carry_channels_and_units(self):
        m = mapped(load("kaggle_structure_replica.csv"))
        ch = {c: m.columns[c].channel for c in EXPOSURE_SIX}
        assert ch == {"TV_Impressions": "TV", "YouTube_Impressions": "YouTube", "Facebook_Impressions": "Facebook",
                      "Instagram_Impressions": "Instagram", "Print_Readership": "Print",
                      "Radio_Listenership": "Radio"}
        assert m.columns["Print_Readership"].unit == "readers"
        assert m.columns["Radio_Listenership"].unit == "listeners"
        assert m.columns["TV_Impressions"].unit == "impressions"

    def test_alternative_names_for_the_same_concept(self):
        df = pd.DataFrame({"Week": pd.date_range("2024-01-01", periods=30, freq="W-MON").astype(str),
                           "Sales_Value": np.linspace(1000.5, 2000.5, 30), "TV_Investment": np.linspace(10.5, 90.5, 30),
                           "Television_Cost": np.linspace(5.5, 9.5, 30), "FB_Reach": np.arange(30) * 10 + 5,
                           "Meta_Views": np.arange(30) * 7 + 3, "Units_Sold": np.arange(30) + 1,
                           "Quantity": np.arange(30) + 2, "Net_Sales": np.linspace(900.5, 1900.5, 30),
                           "Total_Revenue": np.linspace(1100.5, 2100.5, 30)})
        m = mapped(df)
        for c in ("Sales_Value", "Net_Sales", "Total_Revenue"):
            assert m.columns[c].role == "revenue", c
        for c in ("TV_Investment", "Television_Cost"):
            assert m.columns[c].role == "ad_spend" and m.columns[c].channel == "TV", c
        assert m.columns["FB_Reach"].role == "reach" and m.columns["Meta_Views"].role == "views"
        for c in ("Units_Sold", "Quantity"):
            assert m.columns[c].role == "sales_units", c

    def test_first_numeric_column_is_never_assumed_to_be_revenue(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"X1": rng.uniform(100, 900, 50), "Value_A": rng.uniform(1, 5, 50),
                           "col3": rng.normal(size=50)})
        m = mapped(df)
        assert m.target_candidates == []
        assert all(r.group != "outcome" for r in m.columns.values())
        assert m.needs_target_confirmation
        assert RE.detect_mode(m).mode == "general" and not RE.detect_mode(m).has_outcome

    def test_multiple_target_candidates_require_confirmation(self):
        m = mapped(load("multi_revenue.csv"))
        assert set(m.target_candidates) == {"Gross_Revenue", "Net_Revenue", "Total_Sales"}
        assert m.targets == [] and m.needs_target_confirmation and m.primary_target is None
        assert any("Several columns could be the analysis target" in w for w in m.warnings)

    def test_ambiguous_column_gets_a_warning_and_alternatives(self):
        df = pd.DataFrame({"Week": pd.date_range("2024-01-01", periods=20, freq="W-MON").astype(str),
                           "Sales": np.arange(20) + 100})
        r = mapped(df).columns["Sales"]
        assert {r.role} | {a["role"] for a in r.alternatives} >= {"revenue", "sales_units"}
        assert any(w.startswith("Ambiguous") for w in r.warnings)   # never a silent pick between the two

    def test_aggregate_spend_columns_are_not_channels(self):
        m = mapped(pd.read_csv(ROOT / "data" / "processed" / "marketing_data_clean.csv"))
        for c in ("Total_Marketing_Spend", "Digital_Spend", "Traditional_Spend"):
            assert m.columns[c].role == "total_marketing_spend", c
            assert any("equals the sum of" in e for e in m.columns[c].evidence), c
        for c in ("TV_Spend_Share", "Naive_ROAS", "AOV", "Revenue_Growth_WoW"):
            assert m.columns[c].role == "derived", c
        assert len(m.cols("ad_spend")) == 8
        assert m.columns["Promotion_Spend"].role == "trade_spend"

    def test_value_relationship_evidence(self):
        m = mapped(load("kaggle_structure_replica.csv"))
        assert any("Sales_Units × Net_Price (median difference 0.0%)" in e for e in m.columns["Sales_Value"].evidence)

    def test_tokenizer_rejoins_camel_case_brand_words(self):
        assert tokens("YouTube_Impressions") == ["youtube", "impressions"]
        assert tokens("netSalesValue") == ["net", "sales", "value"]

    def test_excel_input_gives_the_same_roles(self, tmp_path):
        df = load("brand_geo_spend.csv")
        p = tmp_path / "x.xlsx"
        df.to_excel(p, index=False)
        back = pd.read_excel(p)
        a, b = mapped(df), mapped(back)
        assert {c: r.role for c, r in a.columns.items()} == {c: r.role for c, r in b.columns.items()}


class TestTradeSpendRule:
    def test_trade_spend_is_a_business_variable(self):
        m = mapped(load("kaggle_structure_replica.csv"))
        r = m.columns["Trade_Spend"]
        assert r.role == "trade_spend" and r.group == "business" and r.channel is None
        assert "not advertising" in " ".join(r.evidence)
        assert not m.cols("ad_spend")

    @pytest.mark.parametrize("name,expected", [("Trade_Spend", True), ("Promo_Budget", True),
                                               ("Coupon_Cost", True), ("TPR_Spend", True), ("TV_Spend", False),
                                               ("Search_Cost", False), ("Display_Ads_Spend", False)])
    def test_trade_name_rule(self, name, expected):
        assert is_trade_or_promotion_spend_name(name) is expected

    def test_manual_correction_can_make_it_a_channel(self):
        m = mapped(load("kaggle_structure_replica.csv"))
        m2 = RE.apply_user_roles(m, {"Trade_Spend": "ad_spend"}, channels={"Trade_Spend": "Trade"})
        assert m2.columns["Trade_Spend"].role == "ad_spend" and m2.columns["Trade_Spend"].user_set
        assert m2.columns["Trade_Spend"].confidence == 1.0
        assert RE.detect_mode(m2).mode == "spend"
        assert m.columns["Trade_Spend"].role == "trade_spend"     # original untouched

    def test_legacy_classifier_and_field_discovery_skip_trade_spend(self):
        from src.data_engine.pipeline import profile_and_map
        df = load("kaggle_structure_replica.csv")
        _, mapping = profile_and_map(df)
        assert "Trade_Spend" not in [g.column for g in mapping.channels]
        from src.marketing.field_discovery import _has_spend_semantic_evidence  # noqa: F401
        from src.marketing import field_discovery as FD
        from src.data_engine.canonical import build_canonical
        sig = FD.detect_channel_candidates.__code__.co_varnames
        assert "canonical" in sig
        # legacy detectors still find real advertising channels
        _, m2 = profile_and_map(load("spend_weekly.csv"))
        assert [g.column for g in m2.channels] == ["TV_Spend", "Search_Spend", "Social_Spend", "Radio_Spend"]

    def test_confirmed_channel_override_still_reaches_the_legacy_pipeline(self):
        df = load("kaggle_structure_replica.csv")
        m = RE.apply_user_roles(mapped(df), {"Trade_Spend": "ad_spend"}, targets=["Sales_Value"])
        conf, ov = RE.legacy_overrides(m)
        assert ov["channel_columns"] == ["Trade_Spend"] and "channels" in conf


class TestMappingPersistence:
    def test_save_load_and_versioning(self, tmp_path):
        df = load("multi_revenue.csv")
        m = mapped(df, ["Net_Revenue"])
        s1 = RE.save_mapping(tmp_path, m, note="first")
        assert s1.version == 1 and s1.confirmed and s1.confirmed_at
        m2 = RE.apply_user_roles(s1, {"Total_Sales": "measure"}, targets=["Gross_Revenue"])
        s2 = RE.save_mapping(tmp_path, m2, note="second")
        assert s2.version == 2 and len(s2.history) == 2
        assert s2.history[-1]["changed"] == {"Total_Sales": ["revenue", "measure"]}
        back = RE.load_mapping(tmp_path, df)
        assert back.version == 2 and back.targets == ["Gross_Revenue"]
        assert back.columns["Total_Sales"].role == "measure"
        data = json.loads((tmp_path / RE.MAPPING_FILE).read_text(encoding="utf-8"))
        assert data["schema_version"] == RE.SCHEMA_VERSION and data["fingerprint"]

    def test_mapping_for_another_table_or_schema_is_not_reused(self, tmp_path):
        df = load("multi_revenue.csv")
        RE.save_mapping(tmp_path, mapped(df, ["Net_Revenue"]))
        assert RE.load_mapping(tmp_path, df.drop(columns=["Total_Sales"])) is None
        assert RE.load_mapping(tmp_path, df.iloc[:10]) is None
        data = json.loads((tmp_path / RE.MAPPING_FILE).read_text(encoding="utf-8"))
        data["schema_version"] = 99
        (tmp_path / RE.MAPPING_FILE).write_text(json.dumps(data))
        assert RE.load_mapping(tmp_path, df) is None
        (tmp_path / RE.MAPPING_FILE).write_text("{not json")
        assert RE.load_mapping(tmp_path, df) is None

    def test_fingerprint_ignores_loader_dtype_differences(self):
        df = load("spend_weekly.csv")
        parsed = df.copy()
        parsed["Week"] = pd.to_datetime(parsed["Week"])
        assert RE.fingerprint(df) == RE.fingerprint(parsed)

    def test_targets_must_be_outcomes(self):
        df = load("spend_weekly.csv")
        m = RE.apply_user_roles(mapped(df), {}, targets=["TV_Spend", "Revenue"])
        assert m.targets == ["Revenue"]
        assert RE.validate_targets(m, df, ["TV_Spend"])

    def test_legacy_bridge(self):
        df = load("kaggle_structure_replica.csv")
        conf, ov = RE.legacy_overrides(mapped(df, ["Sales_Value", "Sales_Units"]))
        assert ov == {"date_column": "Week", "revenue_column": "Sales_Value", "channel_columns": []}
        conf, ov = RE.legacy_overrides(mapped(df, ["Sales_Units"]))
        assert ov["revenue_column"] is None       # a unit target is never passed on as revenue


# ============================================================================ II. capabilities / modes
class TestCapabilities:
    def _avail(self, name, target):
        caps = model_capabilities(mapped(load(name), [target]), target)
        return {(k, c.key): c for k, items in caps.items() for c in items}

    def test_exposure_mode_never_offers_roi_or_money(self):
        c = self._avail("kaggle_structure_replica.csv", "Sales_Value")
        for key in [("marketing_effectiveness", "roi_roas"), ("channel_performance", "roi_roas"),
                    ("channel_performance", "monetary_contribution"), ("resource_optimizer", "budget_allocation"),
                    ("marketing_effectiveness", "spend_effectiveness"), ("marketing_effectiveness", "mmm"),
                    ("scenario_simulator", "spend_changes")]:
            assert not c[key].available, key
            assert c[key].reason and c[key].needs, key
        for key in [("marketing_effectiveness", "exposure_effectiveness"), ("marketing_effectiveness", "association"),
                    ("marketing_effectiveness", "by_segment"), ("marketing_effectiveness", "promotion"),
                    ("channel_performance", "comparison"), ("channel_performance", "contribution_share"),
                    ("channel_performance", "breakdown"), ("resource_optimizer", "exposure_allocation"),
                    ("scenario_simulator", "exposure_changes"), ("scenario_simulator", "price_changes"),
                    ("scenario_simulator", "promotion_changes")]:
            assert c[key].available, key
        assert c[("resource_optimizer", "exposure_allocation")].conditional

    def test_spend_mode_with_money_target(self):
        c = self._avail("spend_weekly.csv", "Revenue")
        for key in [("marketing_effectiveness", "roi_roas"), ("marketing_effectiveness", "mmm"),
                    ("channel_performance", "roi_roas"), ("resource_optimizer", "budget_allocation"),
                    ("scenario_simulator", "spend_changes")]:
            assert c[key].available, key
        assert not c[("resource_optimizer", "exposure_allocation")].available

    def test_spend_mode_with_unit_target_has_no_roi(self):
        c = self._avail("brand_geo_spend.csv", "Units_Sold")
        assert not c[("marketing_effectiveness", "roi_roas")].available
        assert "not money" in c[("marketing_effectiveness", "roi_roas")].reason
        assert c[("resource_optimizer", "budget_allocation")].available

    def test_general_mode_explains_missing_marketing_inputs(self):
        c = self._avail("general_sales.csv", "Net_Sales")
        for key in [("channel_performance", "comparison"), ("resource_optimizer", "budget_allocation"),
                    ("resource_optimizer", "exposure_allocation"), ("marketing_effectiveness", "exposure_effectiveness"),
                    ("scenario_simulator", "spend_changes")]:
            assert not c[key].available and c[key].needs, key
        assert c[("scenario_simulator", "price_changes")].available
        assert c[("marketing_effectiveness", "business_drivers")].available


# ============================================================================ III. statistics
class TestStats:
    def test_hc3_matches_the_textbook_sandwich(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(60, 2))
        y = 1 + X @ [2.0, -1.0] + rng.normal(size=60) * (1 + np.abs(X[:, 0]))
        f = S.ols_hc3(X, y, ["a", "b"])
        Xc = np.column_stack([np.ones(60), X])
        b = np.linalg.solve(Xc.T @ Xc, Xc.T @ y)
        e = y - Xc @ b
        H = Xc @ np.linalg.inv(Xc.T @ Xc) @ Xc.T
        omega = np.diag(e ** 2 / (1 - np.diag(H)) ** 2)
        inv = np.linalg.inv(Xc.T @ Xc)
        se = np.sqrt(np.diag(inv @ Xc.T @ omega @ Xc @ inv))
        assert np.allclose(f.coef, b) and np.allclose(f.se, se)
        assert 0 < f.r2 < 1 and f.p[1] < 0.01

    def test_vif(self):
        rng = np.random.default_rng(1)
        a, b = rng.normal(size=200), rng.normal(size=200)
        v = S.vif(pd.DataFrame({"a": a, "b": b}))
        assert all(abs(x - 1) < 0.1 for x in v.values())
        assert math.isinf(S.vif(pd.DataFrame({"a": a, "b": b, "c": a + b}))["c"])

    def test_permutation_importance_is_deterministic(self):
        rng = np.random.default_rng(2)
        X = pd.DataFrame({"x": rng.normal(size=100), "noise": rng.normal(size=100)})
        y = 3 * X.x.to_numpy() + rng.normal(size=100) * 0.1
        f = S.ols_hc3(X.to_numpy(), y, list(X.columns))
        pred = lambda Z: f.predict(Z.to_numpy())  # noqa: E731
        a = S.permutation_importance(pred, X, y, {"x": ["x"], "noise": ["noise"]}, seed=5)
        b = S.permutation_importance(pred, X, y, {"x": ["x"], "noise": ["noise"]}, seed=5)
        assert a.equals(b) and a.iloc[0].driver == "x" and a.iloc[0].importance > 1


# ============================================================================ IV. models
class TestContextAndGrain:
    def test_exposure_grain_detected_and_counted_once(self):
        ctx = ctx_for("kaggle_structure_replica.csv", "Sales_Value")
        assert ctx.mode.mode == "exposure" and ctx.channels == EXPOSURE_SIX
        assert set(value_grain(ctx.df, "TV_Impressions", "Week", ctx.dim_cols)) == {"Geo", "Brand"}
        per = period_totals(ctx, ["TV_Impressions"])
        manual = ctx.df.drop_duplicates(["Week", "Geo", "Brand"]).groupby("Week")["TV_Impressions"].sum()
        assert np.allclose(per["TV_Impressions"].to_numpy(), manual.to_numpy())
        assert per["TV_Impressions"].sum() < ctx.df["TV_Impressions"].sum()    # naive sum would double-count SKUs

    def test_other_outcomes_and_derived_columns_never_drive_the_target(self):
        ctx = ctx_for("kaggle_structure_replica.csv", "Sales_Value")
        assert "Sales_Units" in ctx.excluded and "Sales_Units" not in ctx.default_drivers()
        ctx2 = build_context(pd.read_csv(ROOT / "data/processed/marketing_data_clean.csv"),
                             mapped(pd.read_csv(ROOT / "data/processed/marketing_data_clean.csv"), ["Revenue"]), "Revenue")
        for c in ("Orders", "Naive_ROAS", "Total_Marketing_Spend", "AOV"):
            assert c in ctx2.excluded and c not in ctx2.default_drivers()

    def test_build_context_rejects_non_outcome_target(self):
        df = load("spend_weekly.csv")
        with pytest.raises(ValueError, match="not mapped as a business outcome"):
            build_context(df, mapped(df, ["Revenue"]), "TV_Spend")


class TestEffectiveness:
    @pytest.mark.parametrize("target", ["Sales_Value", "Sales_Units"])
    def test_kaggle_structure_both_targets_independently(self, target):
        res = run_effectiveness(ctx_for("kaggle_structure_replica.csv", target))
        assert res.status == "ok" and res.target == target and res.mode == "exposure"
        ch = res.effects[res.effects.kind == "channel"]
        assert sorted(ch.column) == sorted(EXPOSURE_SIX)
        assert abs(ch.association_share_pct.sum() - 100) < 1e-6
        assert not res.mmm
        assert not any(MONEY_WORDS.search(c) or "roi" in c.lower() or "roas" in c.lower() for c in res.effects.columns)
        assert res.model.quality in ("good", "moderate", "weak")
        assert res.model.split.startswith("chronological")
        assert any("no ROI, ROAS or monetary contribution" in a for a in res.assumptions)

    def test_spend_mode_roi_only_for_money_targets(self):
        res = run_effectiveness(ctx_for("spend_weekly.csv", "Revenue"))
        assert res.mmm["status"] == "ok" and {"model_roi", "model_roas"} <= set(res.mmm["table"].columns)
        res2 = run_effectiveness(ctx_for("brand_geo_spend.csv", "Units_Sold"))
        assert res2.mmm["status"] == "ok"
        assert "model_roas" not in res2.mmm["table"] and "outcome_per_unit_spend" in res2.mmm["table"]

    def test_general_mode_runs_business_driver_analysis(self):
        res = run_effectiveness(ctx_for("general_sales.csv", "Net_Sales"))
        assert res.status == "ok" and set(res.effects.kind) <= {"business", "other"}
        assert {"Unit_Price", "Discount"} <= set(res.effects.column)

    def test_promotion_effects_and_segments(self):
        res = run_effectiveness(ctx_for("kaggle_structure_replica.csv", "Sales_Units"))
        assert set(res.promotions.promotion) == {"Feature_Flag", "Display_Flag", "TPR_Flag"}
        assert set(res.by_segment) >= {"Brand", "Geo"}
        assert res.by_time is not None and len(res.by_time) >= 2

    def test_unusable_driver_set_is_reported_not_hidden(self):
        df = load("spend_weekly.csv").head(8)
        res = run_effectiveness(build_context(df, mapped(df, ["Revenue"]), "Revenue"))
        assert res.status == "unavailable" and "usable rows" in res.message


class TestChannelPerformance:
    def test_exposure_shares_within_unit_and_no_money(self):
        ctx = ctx_for("kaggle_structure_replica.csv", "Sales_Value")
        eff = run_effectiveness(ctx)
        res = run_channel_performance(ctx, eff.effects, eff.mmm)
        t = res.table
        assert set(t.unit) == {"impressions", "readers", "listeners"}
        for unit, g in t.groupby("unit"):
            assert abs(g.share_within_unit_pct.sum() - 100) < 1e-6, unit
        assert not {"model_roi", "model_roas", "model_contribution"} & set(t.columns)
        assert res.ranking_basis.startswith("statistical association")
        assert set(res.breakdowns) >= {"Brand", "Geo"}
        assert any("not revenue attributed" in w for w in res.warnings)

    def test_spend_mode_ranks_by_model_roas(self):
        ctx = ctx_for("spend_weekly.csv", "Revenue")
        eff = run_effectiveness(ctx)
        res = run_channel_performance(ctx, eff.effects, eff.mmm)
        assert "ROAS" in res.ranking_basis and res.table["rank"].min() == 1

    def test_no_channels_is_explained(self):
        ctx = ctx_for("general_sales.csv", "Net_Sales")
        res = run_channel_performance(ctx)
        assert res.status == "unavailable" and "cannot be analysed" in res.message


class TestExposureOptimizer:
    def _run(self, total_factor=1.0):
        ctx = ctx_for("kaggle_structure_replica.csv", "Sales_Value")
        model = run_effectiveness(ctx).model
        groups = {g.unit: g for g in O.exposure_groups(ctx, model)}
        g = groups["impressions"]
        cur = O.current_levels(ctx, g.eligible)
        total = sum(cur.values()) * total_factor
        mins, maxs, _ = LIM.default_limits(total, g.eligible, cur)
        return ctx, groups, O.optimize_exposure(ctx, model, g, total, mins, maxs), total, mins, maxs

    def test_only_same_unit_groups_with_two_channels_are_optimisable(self):
        _, groups, *_ = self._run()
        assert groups["impressions"].available and len(groups["impressions"].eligible) >= 2
        assert not groups["readers"].available and not groups["listeners"].available
        assert "at least two channels in the same unit" in groups["readers"].reasons[0]

    def test_result_respects_total_and_limits_and_is_not_worse(self):
        ctx, _, r, total, mins, maxs = self._run()
        assert r.status == "ok" and r.solver["converged"]
        assert abs(sum(r.recommended.values()) - total) <= 1e-6 * total
        for c, v in r.recommended.items():
            assert mins[c] - 1e-6 * total <= v <= maxs[c] + 1e-6 * total
        assert r.recommended_outcome >= r.current_outcome - 1e-6
        assert r.unit == "impressions" and "not a monetary budget" in " ".join(r.assumptions)
        assert "impressions" in r.objective

    def test_infeasible_limits_are_refused_without_a_fake_solution(self):
        ctx = ctx_for("kaggle_structure_replica.csv", "Sales_Value")
        model = run_effectiveness(ctx).model
        g = [x for x in O.exposure_groups(ctx, model) if x.unit == "impressions"][0]
        r = O.optimize_exposure(ctx, model, g, 1000.0, {c: 0.0 for c in g.eligible}, {c: 1.0 for c in g.eligible})
        assert r.status == "infeasible" and not r.recommended and "maximums add up to only" in r.message

    def test_weak_model_blocks_optimisation(self):
        df = load("kaggle_structure_replica.csv").copy()
        df["Sales_Value"] = np.random.default_rng(0).permutation(df["Sales_Value"].to_numpy())
        ctx = build_context(df, mapped(df, ["Sales_Value"]), "Sales_Value")
        model = fit_response(ctx, ctx.channels, fixed_effects=[], trend=False)
        groups = O.exposure_groups(ctx, model)
        assert not any(g.available for g in groups)
        assert any("too weak" in " ".join(g.reasons) or "statistically clear" in " ".join(g.reasons) for g in groups)


class TestScenarios:
    def test_baseline_separates_actual_and_model(self):
        ctx = ctx_for("kaggle_structure_replica.csv", "Sales_Units")
        model = run_effectiveness(ctx).model
        res = SC.simulate(ctx, model, SC.preset_scenarios(model, ctx))
        t = res.table()
        assert list(t.kind[:2]) == ["actual", "model estimate"]
        assert t.scenario[0].startswith("Historical actual") and t.scenario[1].startswith("Model baseline")
        names = set(t.scenario)
        assert {"All channels +10% exposure", "All channels -10% exposure", "Price +5%", "No promotions"} <= names
        up = t[t.scenario == "All channels +10% exposure"].iloc[0]
        assert up.low_90 < up.estimate < up.high_90

    def test_extrapolation_and_reliability_flags(self):
        ctx = ctx_for("kaggle_structure_replica.csv", "Sales_Value")
        model = run_effectiveness(ctx).model
        res = SC.simulate(ctx, model, [SC.Scenario("Huge TV", {"TV_Impressions": {"type": "pct", "value": 300}})])
        o = res.outcomes[0]
        assert any("beyond the observed range" in e for e in o.extrapolation)
        assert any("large relative" in e for e in o.extrapolation)
        assert res.reliability in ("good", "moderate", "weak") and res.reliability_text
        assert any("exposure units" in a for a in res.assumptions)

    def test_promotion_share_scenario(self):
        ctx = ctx_for("brand_geo_spend.csv", "Units_Sold")
        model = run_effectiveness(ctx).model
        res = SC.simulate(ctx, model, [SC.Scenario("All promo", {"Promo_Flag": {"type": "share", "value": 100}}),
                                       SC.Scenario("No promo", {"Promo_Flag": {"type": "share", "value": 0}})])
        assert res.outcomes[0].estimate > res.outcomes[1].estimate

    def test_spend_mode_mmm_cross_check(self):
        ctx = ctx_for("spend_weekly.csv", "Revenue")
        eff = run_effectiveness(ctx)
        res = SC.simulate(ctx, eff.model, SC.preset_scenarios(eff.model, ctx), mmm=eff.mmm)
        assert len(res.mmm_check) == 2 and res.mmm_check[0]["mmm_change_pct"] > 0

    def test_weak_model_is_flagged_unreliable(self):
        df = load("spend_weekly.csv").copy()
        df["Revenue"] = np.random.default_rng(1).permutation(df["Revenue"].to_numpy())
        ctx = build_context(df, mapped(df, ["Revenue"]), "Revenue")
        model = fit_response(ctx)
        res = SC.simulate(ctx, model, SC.preset_scenarios(model, ctx))
        assert res.reliability == "weak" and "Do not use these estimates as forecasts" in res.reliability_text


# ============================================================================ V. optimizer default limits
class TestDefaultLimits:
    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 10, 25, 60])
    @pytest.mark.parametrize("total", [1.0, 225_984.30, 1e9])
    def test_defaults_always_feasible(self, n, total):
        ch = [f"c{i}" for i in range(n)]
        mins, maxs, rule = LIM.default_limits(total, ch)
        assert sum(mins.values()) <= total <= sum(maxs.values()) + 1e-9
        assert all(mins[c] <= maxs[c] for c in ch)
        assert LIM.feasibility(total, mins, maxs, ch)["feasible"]
        assert rule

    def test_reported_bug_case_one_channel(self):
        from src.prescriptive.engine import validate_constraints
        total = 225_984.30
        old = validate_constraints(["Only"], total, {"Only": total * 0.05}, {"Only": total * 0.55})
        assert any("maximums add up to only" in i for i in old)          # the old 55% default
        mins, maxs, _ = LIM.default_limits(total, ["Only"])
        assert maxs["Only"] >= total and mins["Only"] == 0.0
        assert validate_constraints(["Only"], total, mins, maxs) == []

    def test_no_fixed_55_percent_for_small_channel_counts(self):
        for n in (1, 2):
            _, maxs, _ = LIM.default_limits(100.0, [f"c{i}" for i in range(n)])
            assert all(v == 100.0 for v in maxs.values())

    def test_defaults_never_force_cutting_a_dominant_channel(self):
        cur = {"a": 700.0, "b": 100.0, "c": 100.0, "d": 100.0}
        mins, maxs, _ = LIM.default_limits(1000.0, list(cur), cur)
        assert maxs["a"] >= 0.875 * 1000 - 0.01
        assert mins["b"] <= 50.0

    def test_invalid_total(self):
        mins, maxs, rule = LIM.default_limits(float("nan"), ["a"])
        assert maxs == {"a": 0.0} and "positive" in rule
        assert not LIM.feasibility(0, {"a": 0}, {"a": 0}, ["a"])["feasible"]

    def test_reconcile_refreshes_defaults_but_keeps_manual_values(self):
        state: dict = {}
        pairs = {"a": ("min_a", "max_a"), "b": ("min_b", "max_b")}
        m1, x1, _ = LIM.default_limits(1000.0, ["a", "b"])
        st1 = LIM.reconcile_limits(state, pairs, m1, x1, "reg")
        assert state["max_a"] == 1000.0 and st1["a"] == {"min": "auto", "max": "auto"}
        state["max_b"] = 333.0                                   # the user types a value
        m2, x2, _ = LIM.default_limits(5000.0, ["a", "b"])        # the total changes
        st2 = LIM.reconcile_limits(state, pairs, m2, x2, "reg")
        assert state["max_a"] == 5000.0 and st2["a"]["max"] == "auto"
        assert state["max_b"] == 333.0 and st2["b"]["max"] == "manual"
        assert state["min_b"] == m2["b"] and st2["b"]["min"] == "auto"
        st3 = LIM.reconcile_limits(state, pairs, m1, x1, "reg")   # and back: still never overwritten
        assert state["max_b"] == 333.0 and st3["b"]["max"] == "manual"
        LIM.reset_limits(state, pairs, m1, x1, "reg")            # explicit reset only
        assert state["max_b"] == 1000.0
        st4 = LIM.reconcile_limits(state, pairs, m2, x2, "reg")
        assert state["max_b"] == 5000.0 and st4["b"]["max"] == "auto"

    def test_feasibility_messages(self):
        f = LIM.feasibility(100.0, {"a": 80, "b": 40}, {"a": 90, "b": 90}, ["a", "b"])
        assert not f["feasible"] and "minimums add up to 120.00" in f["messages"][0]
        f = LIM.feasibility(100.0, {"a": 0, "b": 0}, {"a": 30, "b": 30}, ["a", "b"])
        assert not f["feasible"] and "maximums add up to only 60.00" in f["messages"][0]
        f = LIM.feasibility(100.0, {"a": 0, "b": 0}, {"a": 60, "b": 60}, ["a", "b"])
        assert f["feasible"] and f["messages"][0].startswith("Feasible")

    def test_prescriptive_ui_has_no_fixed_55_percent(self):
        src = (ROOT / "dashboard/components/prescriptive_ui.py").read_text(encoding="utf-8")
        assert "0.55" not in src and "default_limits" in src and "reconcile_limits" in src


# ============================================================================ VI. Streamlit pages
def _app(state):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=300)
    for k, v in state.items():
        at.session_state[k] = v
    return at


def _demo():
    from config.settings import DEMO_DATASET_ID
    return {"mp_active_dataset_id": DEMO_DATASET_ID, "mp_demo_explicitly_chosen": True}


def _upload_state(did, name):
    return {"mp_active_dataset_id": did, "mp_active_dataset_filename": name, "mp_demo_explicitly_chosen": False}


def _goto(at, label):
    at.run()
    at.sidebar.radio[0].set_value(label).run()
    return at


def _texts(at):
    out = []
    for coll in (at.markdown, at.info, at.warning, at.error, at.success, at.caption, at.subheader, at.title):
        out.extend(html.unescape(str(e.value)) for e in coll)
    return out


def _figures(at):
    return [json.loads(c.proto.spec) for c in at.get("plotly_chart")]


def _title(t):
    return (t.get("text") if isinstance(t, dict) else t) or ""


def _assert_labelled(at, min_figs=1):
    figs = _figures(at)
    assert len(figs) >= min_figs, f"{len(figs)} figures"
    for f in figs:
        lay = f["layout"]
        assert _title(lay.get("title")).strip()
        assert _title((lay.get("xaxis") or {}).get("title")).strip(), _title(lay.get("title"))
        assert _title((lay.get("yaxis") or {}).get("title")).strip(), _title(lay.get("title"))
    return figs


def _no_failure(at):
    assert not at.exception, at.exception
    bad = [t for t in _texts(at) if "Unexpected error" in t or " failed: " in t or "is not available for this dataset" in t]
    assert not bad, bad


@pytest.fixture(scope="module")
def uploads():
    """Real uploads through the platform pipeline (same calls as the Upload page)."""
    from src.data_engine.analysis_runner import run_full_analysis
    from src.data_engine.currency import choice_to_info, detect_currency, save_currency
    from src.data_engine.pipeline import profile_and_map
    from dashboard.components import role_mapping_ui as RMU  # noqa: F401  (import check)
    made = {}

    def make(key, name, targets, currency="auto", save_targets=True):
        did = f"upload_p2{uuid.uuid4().hex[:10]}"
        df = load(name)
        _, mapping = profile_and_map(df)
        m = mapped(df, targets)
        conf, ov = RE.legacy_overrides(m)
        run_full_analysis(did, df=df, mapping=mapping, confirmed_roles=conf, overrides=ov)
        std = pd.read_csv(dataset_processed_dir(did) / "standardized_data.csv")
        m2 = mapped(std, targets if save_targets else [])
        RE.save_mapping(dataset_processed_dir(did), m2)
        save_currency(dataset_processed_dir(did), choice_to_info(currency, detect_currency(df), RE.monetary_columns(m2)))
        made[key] = (did, name)
        return did

    make("kaggle", "kaggle_structure_replica.csv", ["Sales_Value", "Sales_Units"], "INR")
    make("kaggle_unconfirmed", "kaggle_structure_replica.csv", [], "INR", save_targets=False)
    make("general", "general_sales.csv", ["Net_Sales"], "USD")
    make("spend1", "one_channel.csv", ["Revenue"], "USD")
    make("spend2", "two_channel.csv", ["Revenue"], "EUR")
    make("spend4", "spend_weekly.csv", ["Revenue"], "none")
    make("daily", "exposure_daily.csv", ["Conversions"])
    yield made
    for did, _ in made.values():
        shutil.rmtree(dataset_upload_dir(did), ignore_errors=True)


def _state(uploads, key):
    did, name = uploads[key]
    return _upload_state(did, name)


class TestModelPages:
    @pytest.mark.parametrize("page", list(PAGES))
    def test_demo_spend_mode(self, page):
        at = _goto(_app(_demo()), PAGES[page])
        _no_failure(at)
        text = " ".join(_texts(at))
        assert "Spend-Based Marketing Analytics" in text and "Demo dataset" in text
        assert "Available for this dataset" in text and "Not available, and why" in text
        if page != "resource_optimizer":
            _assert_labelled(at)

    @pytest.mark.parametrize("page", list(PAGES))
    def test_kaggle_structure_exposure_mode(self, uploads, page):
        at = _goto(_app(_state(uploads, "kaggle")), PAGES[page])
        _no_failure(at)
        text = " ".join(_texts(at))
        assert "Exposure-Based Marketing Analytics" in text and "kaggle_structure_replica.csv" in text
        assert "ROI / ROAS by channel" not in [m.value for m in at.markdown if "✅" in m.value]
        figs = _assert_labelled(at) if page != "resource_optimizer" else _figures(at)
        for f in figs:
            lay = f["layout"]
            words = " ".join([_title(lay.get("title")), _title(lay["xaxis"].get("title")), _title(lay["yaxis"].get("title"))])
            assert not MONEY_WORDS.search(words), words
        for df in at.dataframe:
            cols = " ".join(map(str, df.value.columns))
            assert not re.search(r"roi|roas|model_contribution", cols, re.I), cols
        # every visible sentence that mentions ROI / ROAS says it is NOT available (never a value)
        for t in _texts(at):
            plain = re.sub(r"<[^>]+>", " ", t)
            for sent in re.split(r"(?<=[.;])\s+|\n|- ", plain):
                if MONEY_WORDS.search(sent):
                    assert re.search(r"\b(not|no|unavailable|never|need|needs|⛔)\b|⛔", sent, re.I), sent

    def test_kaggle_targets_selectable_independently(self, uploads):
        at = _goto(_app(_state(uploads, "kaggle")), PAGES["marketing_effectiveness"])
        sel = [s for s in at.selectbox if s.label == "Target analysed on this page"][0]
        assert list(sel.options) == ["Sales_Value", "Sales_Units"]
        t1 = _figures(at)[0]["layout"]["title"]["text"]
        sel.set_value("Sales_Units").run()
        _no_failure(at)
        t2 = _figures(at)[0]["layout"]["title"]["text"]
        assert "Sales_Value" in t1 and "Sales_Units" in t2

    def test_unconfirmed_target_blocks_models_and_asks(self, uploads):
        at = _goto(_app(_state(uploads, "kaggle_unconfirmed")), PAGES["marketing_effectiveness"])
        _no_failure(at)
        text = " ".join(_texts(at))
        assert "Please confirm the target" in text and "Target needed" in text
        assert not _figures(at)
        ms = [m for m in at.multiselect if m.label == "Analysis target(s)"][0]
        assert ms.value == [] and set(ms.options) == {"Sales_Value", "Sales_Units"}
        ms.set_value(["Sales_Units"]).run()
        [b for b in at.button if b.label == "Save mapping"][0].click().run()
        _no_failure(at)
        assert _figures(at)
        did, _ = uploads["kaggle_unconfirmed"]
        saved = RE.load_mapping(dataset_processed_dir(did))
        assert saved.targets == ["Sales_Units"] and saved.version == 2

    @pytest.mark.parametrize("page", list(PAGES))
    def test_general_mode(self, uploads, page):
        at = _goto(_app(_state(uploads, "general")), PAGES[page])
        _no_failure(at)
        text = " ".join(_texts(at))
        assert "General Business Analytics" in text
        if page == "resource_optimizer":
            assert "Missing inputs" in text and "channel-wise advertising spend" in text
        elif page == "channel_performance":
            assert "No channels" in text and "segment performance" in text.lower()
        elif page == "marketing_effectiveness":
            assert "Marketing model unavailable" in text
            _assert_labelled(at)
        else:
            _assert_labelled(at)

    def test_previous_module_toggle_kept(self):
        at = _goto(_app(_demo()), PAGES["resource_optimizer"])
        t = [x for x in at.toggle if "previous" in x.label.lower()][0]
        assert "Budget Optimizer" in t.label and not t.value

    def test_currency_follows_dataset(self, uploads):
        at = _goto(_app(_state(uploads, "spend2")), PAGES["marketing_effectiveness"])
        ys = " ".join(_title(f["layout"]["yaxis"].get("title")) for f in _figures(at))
        assert "(EUR)" in ys and "INR" not in ys and "USD" not in ys
        at = _goto(_app(_state(uploads, "spend4")), PAGES["marketing_effectiveness"])
        axes = " ".join(_title(f["layout"][a].get("title")) for f in _figures(at) for a in ("xaxis", "yaxis"))
        assert "INR" not in axes and "USD" not in axes and "₹" not in axes

    def test_daily_exposure_dataset(self, uploads):
        at = _goto(_app(_state(uploads, "daily")), PAGES["channel_performance"])
        _no_failure(at)
        titles = [_title(f["layout"]["title"]) for f in _figures(at)]
        assert any("per day" in t for t in titles)


class TestOptimizerPages:
    def _opt(self, uploads, key, label=PAGES["resource_optimizer"]):
        at = _goto(_app(_state(uploads, key)), label)
        _no_failure(at)
        return at

    @pytest.mark.parametrize("key,n", [("spend1", 1), ("spend2", 2), ("spend4", 4)])
    def test_defaults_feasible_and_solved(self, uploads, key, n):
        at = self._opt(uploads, key)
        ni = {x.label: x.value for x in at.number_input}
        total = [v for k, v in ni.items() if k.startswith("Total budget")][0]
        maxs = [v for k, v in ni.items() if k.endswith("maximum")]
        mins = [v for k, v in ni.items() if k.endswith("minimum")]
        assert len(maxs) == n and sum(mins) <= total <= sum(maxs) + 1e-6
        assert any("Feasible:" in s.value for s in at.success)
        b = [x for x in at.button if x.label == "Find the best allocation"][0]
        assert not b.disabled
        b.click().run()
        _no_failure(at)
        text = " ".join(_texts(at))
        assert "Solver status:** Solved" in text and "Objective:**" in text
        assert "maximums add up to only" not in text

    def test_budget_change_refreshes_defaults_but_not_manual_limits(self, uploads):
        at = self._opt(uploads, "spend2")
        mx = sorted(x.label for x in at.number_input if x.label.endswith("maximum"))
        first, second = mx
        [x for x in at.number_input if x.label == second][0].set_value(1234.0).run()
        tot = [x for x in at.number_input if x.label.startswith("Total budget")][0]
        tot.set_value(tot.value * 2).run()
        _no_failure(at)
        ni = {x.label: x.value for x in at.number_input}
        new_total = [v for k, v in ni.items() if k.startswith("Total budget")][0]
        assert ni[first] == pytest.approx(new_total)                    # auto default followed the total
        manual = [k for k in ni if k.startswith(second.split(" - ")[0]) and "maximum" in k][0]
        assert ni[manual] == 1234.0 and "set by you" in manual          # the typed value is kept, and marked
        assert [x for x in at.button if x.label == "Reset all limits to the defaults"]

    def test_infeasible_manual_limits_are_explained_not_changed(self, uploads):
        at = self._opt(uploads, "spend4")
        for lab in sorted(x.label for x in at.number_input if x.label.endswith("maximum")):
            [x for x in at.number_input if x.label == lab][0].set_value(1.0).run()
        _no_failure(at)
        ni = {x.label: x.value for x in at.number_input}
        assert all(v == 1.0 for k, v in ni.items() if "maximum" in k)   # nothing silently changed
        assert any("add up to only 4.00" in e.value or "is below the minimum" in e.value for e in at.error)
        assert [x for x in at.button if x.label == "Find the best allocation"][0].disabled
        assert not any("Solver status:** Solved" in t for t in _texts(at))
        [x for x in at.button if x.label == "Reset all limits to the defaults"][0].click().run()
        assert any("Feasible:" in s.value for s in at.success)

    def test_page15_prescriptive_one_channel_regression(self, uploads):
        at = self._opt(uploads, "spend1", label="15 · 🎯 Prescriptive Analytics")
        ni = {x.label: x.value for x in at.number_input}
        tot_label = [k for k in ni if k.startswith("Total budget per period")][0]
        assert ni["Digital_Spend - maximum"] >= ni[tot_label]
        [x for x in at.button if x.label == "Find the best allocation"][0].click().run()
        _no_failure(at)
        assert "Solver status:** Solved" in " ".join(_texts(at))

    def test_exposure_optimizer_page(self, uploads):
        at = self._opt(uploads, "kaggle")
        text = " ".join(_texts(at))
        assert "mix of exposure units" in text and "(not money)" in " ".join(x.label for x in at.number_input)
        assert not [x for x in at.button if x.label == "Find the best allocation"]
        [x for x in at.button if x.label == "Find the best exposure mix"][0].click().run()
        _no_failure(at)
        text = " ".join(_texts(at))
        assert "Solver status:** Solved" in text and "impressions" in text
        f = _assert_labelled(at)
        assert any("impressions" in _title(x["layout"]["yaxis"].get("title")).lower() for x in f)


class TestScenarioPage:
    def test_add_and_remove_scenario(self, uploads):
        at = _goto(_app(_state(uploads, "kaggle")), PAGES["scenario_simulator"])
        n0 = len([b for b in at.button if b.label == "Remove"])
        assert n0 >= 3
        inp = [x for x in at.number_input if x.label.startswith("TV (impressions)")][0]
        inp.set_value(25.0)
        at.text_input[0].set_value("TV +25%")
        [b for b in at.button if b.label == "Add scenario"][0].click().run()
        _no_failure(at)
        assert len([b for b in at.button if b.label == "Remove"]) == n0 + 1
        assert "TV +25%" in " ".join(_texts(at))
        tbl = [d.value for d in at.dataframe if "scenario" in d.value.columns][0]
        assert "TV +25%" in set(tbl.scenario) and tbl.kind.iloc[0] == "actual"
        [b for b in at.button if b.label == "Remove"][-1].click().run()
        assert len([b for b in at.button if b.label == "Remove"]) == n0


# ============================================================================ VII. legacy pipeline / API / analyst
class TestLegacyIntegration:
    def test_kaggle_structure_through_full_pipeline_has_no_channel_roi(self, uploads):
        did, _ = uploads["kaggle"]
        d = dataset_processed_dir(did)
        p6 = json.loads((d / "phase6_marketing.json").read_text(encoding="utf-8"))
        rep = p6.get("marketing_report") or {}
        assert not (rep.get("channel_fields") or [])
        assert p6.get("roi_results") in ([], None)
        caps = json.loads((d / "capabilities.json").read_text(encoding="utf-8"))
        assert caps["roi_roas"]["available"] is False

    def test_ai_analyst_does_not_invent_channel_roi(self, uploads):
        from src.ai.analyst import MarketingAnalyst
        did, _ = uploads["kaggle"]
        a = MarketingAnalyst(data_dir=dataset_processed_dir(did), dataset_id=did)
        for q in ("Which channel has the best ROI?", "What is the ROAS of TV?", "How should I reallocate budget?"):
            ans = a.answer(q)
            text = json.dumps(ans, default=str)
            assert "Trade_Spend ROI" not in text
            assert not re.search(r"(TV|YouTube|Facebook|Instagram|Print|Radio)[_ A-Za-z]*\s*(ROI|ROAS)\s*(of|=|:|is)\s*[0-9]", text), text[:500]

    def test_api_upload_confirm_kaggle_structure(self):
        from fastapi.testclient import TestClient
        from api.main import app, app_settings
        client = TestClient(app)
        headers = {"X-API-Key": app_settings.API_KEY or ""}
        with open(FIX / "kaggle_structure_replica.csv", "rb") as fh:
            r = client.post("/upload", files={"file": ("k.csv", fh, "text/csv")}, headers=headers)
        assert r.status_code == 200, r.text
        body = r.json()
        did = body["dataset_id"]
        try:
            channels = [g.get("column") for g in body["column_mapping"].get("channels", [])]
            assert "Trade_Spend" not in channels
            r2 = client.post(f"/datasets/{did}/confirm", json={"date_column": "Week", "revenue_column": "Sales_Value",
                                                             "accept_suggested": ["date", "revenue"]}, headers=headers)
            assert r2.status_code == 200, r2.text
            assert r2.json().get("channel_columns") == []
        finally:
            shutil.rmtree(dataset_upload_dir(did), ignore_errors=True)
            shutil.rmtree(dataset_processed_dir(did), ignore_errors=True)


# ============================================================================ VIII. real Kaggle file
@pytest.mark.skipif(not REAL_KAGGLE.exists(), reason=(
    "The user's real file synthetic_mmm_weekly_india_SAT.csv has not been provided; place it at "
    "tests/fixtures/part02/real/ to run these checks on it."))
class TestRealKaggleFile:
    def test_mode_trade_spend_and_exposures(self):
        df = pd.read_csv(REAL_KAGGLE)
        m = mapped(df)
        assert RE.detect_mode(m).mode == "exposure"
        assert m.columns["Trade_Spend"].role == "trade_spend" and not m.cols("ad_spend")
        for c in EXPOSURE_SIX:
            assert m.columns[c].group == "exposure", c
        assert {"Sales_Value", "Sales_Units"} <= set(m.target_candidates) and m.needs_target_confirmation

    @pytest.mark.parametrize("target", ["Sales_Value", "Sales_Units"])
    def test_models_on_real_file(self, target):
        df = pd.read_csv(REAL_KAGGLE)
        ctx = build_context(df, mapped(df, [target]), target)
        eff = run_effectiveness(ctx)
        assert eff.status == "ok" and not eff.mmm
        assert sorted(eff.effects[eff.effects.kind == "channel"].column) == sorted(EXPOSURE_SIX)
        caps = model_capabilities(ctx.mapping, target)
        assert not [c for items in caps.values() for c in items if c.key == "roi_roas" and c.available]


# ============================================================================ IX. robustness (plan tests H, I, J, Y)
class TestRobustness:
    def test_missing_values_are_dropped_and_reported(self):
        df = load("spend_weekly.csv").copy()
        df.loc[[3, 10, 50], "TV_Spend"] = np.nan
        df.loc[[7], "Revenue"] = np.nan
        res = run_effectiveness(build_context(df, mapped(df, ["Revenue"]), "Revenue"))
        assert res.status == "ok" and res.model.n_dropped == 4 and res.model.n_rows == len(df) - 4
        assert any("4 row(s) with missing values" in n for n in res.model.notes)

    def test_duplicate_rows_do_not_break_models(self):
        df = pd.concat([load("brand_geo_spend.csv"), load("brand_geo_spend.csv").head(30)], ignore_index=True)
        res = run_effectiveness(build_context(df, mapped(df, ["Revenue"]), "Revenue"))
        assert res.status == "ok"

    @pytest.mark.parametrize("fmt", ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%b-%Y", "%Y%m%d"])
    def test_date_formats(self, fmt):
        df = load("spend_weekly.csv").copy()
        df["Week"] = pd.to_datetime(df["Week"]).dt.strftime(fmt)
        if fmt == "%Y%m%d":
            df["Week"] = df["Week"].astype(int)
        m = mapped(df, ["Revenue"])
        assert m.date_col == "Week"
        ctx = build_context(df, m, "Revenue")
        assert ctx.frequency == "weekly" and ctx.n_periods == 120

    def test_bh_adjustment_matches_scipy(self):
        from scipy.stats import false_discovery_control
        p = np.array([0.001, 0.02, 0.03, 0.04, 0.2, 0.7])
        assert np.allclose(S.bh_adjust(p), false_discovery_control(p))
        res = run_effectiveness(ctx_for("spend_weekly.csv", "Revenue"))
        assert {"pearson_p_bh", "spearman_p_bh"} <= set(res.correlations.columns)
        assert (res.correlations.pearson_p_bh >= res.correlations.pearson_p - 1e-12).all()

    def test_segment_sample_gate(self):
        df = load("brand_geo_spend.csv").copy()
        keep = df[df.Region != "Mumbai"]
        few = df[df.Region == "Mumbai"].groupby("Brand").head(5)          # 5 periods only for Mumbai
        df2 = pd.concat([keep, few], ignore_index=True)
        res = run_effectiveness(build_context(df2, mapped(df2, ["Units_Sold"]), "Units_Sold"))
        seg = res.by_segment["Region"].set_index("Region")
        assert seg.loc["Mumbai", "periods"] < 8 and seg.loc["Mumbai"][["TV_Spend", "Digital_Spend"]].isna().all()
        assert seg.loc["Delhi NCR"][["TV_Spend", "Digital_Spend"]].notna().all()

    def test_click_through_rate_only_when_both_measures_exist(self):
        rng = np.random.default_rng(4)
        n = 60
        imp = rng.integers(10_000, 50_000, n)
        df = pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=n, freq="D").astype(str),
                           "Google_Impressions": imp, "Google_Clicks": (imp * 0.03).astype(int),
                           "Meta_Reach": rng.integers(5_000, 9_000, n), "Conversions": rng.integers(50, 90, n)})
        ctx = build_context(df, mapped(df, ["Conversions"]), "Conversions")
        res = run_channel_performance(ctx)
        r = res.rates
        ctr = r[r.metric.str.startswith("Click-through")]
        assert len(ctr) == 1 and ctr.iloc[0].channel == "Google" and abs(ctr.iloc[0].rate_pct - 3.0) < 0.1
        assert not (r.channel == "Meta").any() or not r[r.channel == "Meta"].metric.str.startswith("Click").any()
        assert run_channel_performance(ctx_for("spend_weekly.csv", "Revenue")).rates is None

    def test_no_target_leakage_from_other_outcomes(self):
        ctx = ctx_for("brand_geo_spend.csv", "Revenue")
        res = run_effectiveness(ctx)
        assert "Units_Sold" not in set(res.effects.column)
        assert "another outcome column" in ctx.excluded["Units_Sold"]
