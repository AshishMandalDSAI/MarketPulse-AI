"""PART 06: validation of the user's real Kaggle file (synthetic_mmm_weekly_india_SAT.csv).

Usage:  python tools/part06/real_dataset_validation.py PATH_TO_CSV OUT.json
Runs every applicable engine on the real file and records what actually happened.
Nothing here is mocked; failures are recorded, not hidden.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import os
import time
import traceback
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CSV, OUT = Path(sys.argv[1]), Path(sys.argv[2])
R: dict = {"file": CSV.name, "sha256": hashlib.sha256(CSV.read_bytes()).hexdigest(), "steps": {}}


def step(name):
    def deco(fn):
        t0 = time.perf_counter()
        try:
            R["steps"][name] = {"status": "ok", **(fn() or {})}
        except Exception as e:  # recorded, never hidden
            R["steps"][name] = {"status": "error", "error": f"{type(e).__name__}: {e}",
                                "trace": traceback.format_exc()[-1500:]}
        R["steps"][name]["seconds"] = round(time.perf_counter() - t0, 2)
        print(name, R["steps"][name]["status"], R["steps"][name]["seconds"], flush=True)
        return fn
    return deco


df = pd.read_csv(CSV)
EXPO = ["TV_Impressions", "YouTube_Impressions", "Facebook_Impressions", "Instagram_Impressions",
        "Print_Readership", "Radio_Listenership"]

from src.data_engine import role_engine as RE  # noqa: E402
from src.data_engine.currency import choice_to_info, detect_currency, save_currency  # noqa: E402


@step("H_quality")
def _():
    keys = [c for c in ("Week", "Geo", "Brand", "SKU") if c in df]
    num = df.select_dtypes("number")
    q1, q3 = num.quantile(0.25), num.quantile(0.75)
    iqr = q3 - q1
    out_cnt = ((num < q1 - 3 * iqr) | (num > q3 + 3 * iqr)).sum()
    wk = pd.to_datetime(df["Week"], errors="coerce")
    gaps = wk.drop_duplicates().sort_values().diff().dt.days.value_counts().to_dict()
    neg = {c: int((num[c] < 0).sum()) for c in num if (num[c] < 0).any()}
    return {"rows": int(len(df)), "columns": int(df.shape[1]),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "missing_cells": int(df.isna().sum().sum()), "missing_by_column": {c: int(v) for c, v in df.isna().sum().items() if v},
            "duplicate_rows": int(df.duplicated().sum()), "duplicate_keys_week_geo_brand_sku": int(df.duplicated(keys).sum()),
            "date_parse_share": float(wk.notna().mean()), "date_min": str(wk.min().date()), "date_max": str(wk.max().date()),
            "distinct_weeks": int(wk.nunique()), "week_gap_days_counts": {str(k): int(v) for k, v in gaps.items()},
            "distinct": {c: int(df[c].nunique()) for c in ("Geo", "Brand", "SKU")},
            "rows_per_week": sorted({int(v) for v in df.groupby("Week").size()}),
            "extreme_outliers_3iqr": {c: int(v) for c, v in out_cnt.items() if v},
            "negative_values": neg,
            "constant_columns": [c for c in df if df[c].nunique() <= 1]}


m = RE.suggest_roles(df)


@step("A_B_C_detection_mapping")
def _():
    roles = {c: {"role": r.role, "group": r.group, "confidence": r.confidence, "evidence": r.evidence[:2],
                 "alternatives": [a["role"] for a in (r.alternatives or [])][:3]} for c, r in m.columns.items()}
    want = {"Week": "date", "Sales_Units": "sales_units", "Sales_Value": "revenue", "Trade_Spend": "trade_spend",
            "Brand": "brand", "SKU": "sku", "Geo": "geography", "MRP": "price", "Net_Price": "price",
            "Feature_Flag": "promotion", "Display_Flag": "promotion", "TPR_Flag": "promotion",
            "Weighted_Distribution": "distribution", "Numeric_Distribution": "distribution",
            "TDP": "distribution", "NOS": "distribution",
            "FB_Banner_Content_Score": "quality_score", "IG_Banner_Content_Score": "quality_score"}
    want.update({c: "exposure-group" for c in EXPO})
    check = {}
    for c, w in want.items():
        got = m.columns[c].group if w == "exposure-group" else m.columns[c].role
        check[c] = {"expected": "exposure" if w == "exposure-group" else w, "got": got,
                    "ok": got == ("exposure" if w == "exposure-group" else w)}
    return {"roles": roles, "checks": check, "all_ok": all(v["ok"] for v in check.values()),
            "ad_spend_columns": m.cols("ad_spend"), "total_marketing_spend": m.cols("total_marketing_spend")}


@step("D_target_selection")
def _():
    return {"target_candidates": m.target_candidates, "needs_target_confirmation": m.needs_target_confirmation,
            "suggested_targets_before_choice": list(m.targets)}


@step("E_mode")
def _():
    mode = RE.detect_mode(m)
    return {"mode": mode.mode, "label": mode.label, "evidence": mode.evidence, "spend_channels": mode.spend_channels,
            "exposure_channels": mode.exposure_channels}


@step("currency_detection")
def _():
    cr = detect_currency(df)
    return {"detected": getattr(cr, "code", None), "confidence": getattr(cr, "confidence", None),
            "evidence": str(getattr(cr, "evidence", ""))[:300],
            "note": "No currency symbol or code exists in the file, so the user must choose (INR for this Indian dataset)."}


from src.marketing_intel.capabilities import model_capabilities  # noqa: E402
from src.marketing_intel.channels import run_channel_performance  # noqa: E402
from src.marketing_intel.context import build_context  # noqa: E402
from src.marketing_intel.effectiveness import run_effectiveness  # noqa: E402
from src.marketing_intel import optimizer as O, scenarios as SC  # noqa: E402

for tgt in ("Sales_Value", "Sales_Units"):
    mm = RE.apply_user_roles(RE.suggest_roles(df), {}, targets=[tgt])

    @step(f"F_capabilities[{tgt}]")
    def _(mm=mm, tgt=tgt):
        caps = model_capabilities(mm, tgt)
        flat = {f"{k}:{c.key}": {"available": c.available, "reason": c.reason} for k, items in caps.items() for c in items}
        money_keys = [k for k, v in flat.items() if any(w in k for w in ("roi", "roas", "budget", "monetary")) and v["available"]]
        return {"capabilities": flat, "monetary_capabilities_available": money_keys}

    @step(f"G_models[{tgt}]")
    def _(mm=mm, tgt=tgt):
        ctx = build_context(df, mm, tgt)
        eff = run_effectiveness(ctx)
        model = eff.model
        fx = eff.effects
        ch = run_channel_performance(ctx, fx, None)
        groups = O.exposure_groups(ctx, model)
        opt = []
        for g in groups:
            cur = O.current_levels(ctx, g.eligible) if g.eligible else {}
            total = float(sum(cur.values())) if cur else 0.0
            mins = {c: 0.5 * v for c, v in cur.items()}
            maxs = {c: 1.5 * v for c, v in cur.items()}
            r = O.optimize_exposure(ctx, model, g, total, mins, maxs)
            alloc = r.recommended or {}
            viol = [c for c, v in alloc.items() if v < mins.get(c, -np.inf) - 1e-6 or v > maxs.get(c, np.inf) + 1e-6]
            opt.append({"unit": g.unit, "channels": g.channels, "eligible": g.eligible, "available": g.available,
                        "reasons": g.reasons, "status": r.status, "solution_status": r.solution_status,
                        "message": r.message, "total_in": total,
                        "total_out": float(sum(alloc.values())) if alloc else None, "bound_violations": viol,
                        "change_pct": r.change_pct, "range": [r.low, r.high]})
        sims = SC.simulate(ctx, model, SC.preset_scenarios(model, ctx))
        sens = SC.sensitivity(ctx, model)
        tbl = ch.table
        money_cols = [c for c in (tbl.columns if tbl is not None else []) if any(w in c.lower() for w in ("roi", "roas", "cost per", "₹", "inr"))]
        q = model.quality if hasattr(model, "quality") else None
        return {"grain": ctx.grain if hasattr(ctx, "grain") else None, "channels": ctx.channels,
                "effectiveness_status": eff.status, "mmm_present": bool(eff.mmm),
                "model_quality": str(getattr(model, "quality_grade", q)), "quality_reason": getattr(model, "quality_reason", None),
                "holdout_r2": getattr(model, "holdout_r2", None), "usable_for_decisions": getattr(model, "usable_for_decisions", None),
                "n_obs": getattr(getattr(model, "fit", None), "n", None),
                "channel_effects": fx[fx.kind == "channel"][["column"] + [c for c in ("coef", "p", "elasticity", "direction") if c in fx]].to_dict("records") if fx is not None else None,
                "business_effects": fx[fx.kind != "channel"]["column"].tolist() if fx is not None else None,
                "warnings": eff.warnings[:6],
                "channel_status": ch.status, "channel_table_columns": list(tbl.columns) if tbl is not None else None,
                "money_columns_in_channel_table": money_cols,
                "optimizer_groups": opt, "scenario_status": sims.status, "scenario_outcomes": len(sims.outcomes), "scenario_reliability": sims.reliability,
                "sensitivity_rows": int(len(sens)) if sens is not None else 0}


SLOW = not os.environ.get("P06_SKIP_SLOW")


@step("legacy_pipeline_run_full_analysis") if SLOW else (lambda f: f)
def _():
    from config.settings import dataset_processed_dir, dataset_upload_dir
    from src.data_engine.analysis_runner import run_full_analysis
    from src.data_engine.pipeline import profile_and_map
    _, mp = profile_and_map(df)
    mm = RE.apply_user_roles(RE.suggest_roles(df), {}, targets=["Sales_Value"])
    c, o = RE.legacy_overrides(mm)
    did = f"upload_p06v{uuid.uuid4().hex[:8]}"
    t0 = time.perf_counter()
    res = run_full_analysis(did, df=df, mapping=mp, confirmed_roles=c, overrides=o)
    dt = time.perf_counter() - t0
    pdir = dataset_processed_dir(did)
    files = sorted(p.name for p in pdir.iterdir())
    p6 = json.loads((pdir / "phase6_marketing.json").read_text()) if (pdir / "phase6_marketing.json").exists() else {}
    txt = json.dumps(p6).lower()
    fc = (pdir / "revenue_forecast.csv").exists()
    shutil.rmtree(pdir, ignore_errors=True)
    shutil.rmtree(dataset_upload_dir(did), ignore_errors=True)
    return {"seconds_run_full_analysis": round(dt, 1), "legacy_channel_columns": o["channel_columns"],
            "outputs": files, "forecast_written": fc, "result_keys": list(res)[:12],
            "phase6_mentions_roi": "roi" in txt}


@step("predictive_statistics") if SLOW else (lambda f: f)
def _():
    from src.predictive.contracts import ALL_CLASSIFICATION_MODELS, Settings
    from src.predictive.evaluate import run_supervised
    import dataclasses
    X = df.drop(columns=["Week"])
    out = {}
    r = run_supervised(X, "Sales_Value")
    out["regression_Sales_Value"] = {"status": r.status, "task": getattr(r, "task", None),
                                     "selected": getattr(r, "selected", None)}
    Xc = df.drop(columns=["Week"]).copy()
    names = {f.name for f in dataclasses.fields(Settings)}
    s = Settings(models=list(ALL_CLASSIFICATION_MODELS)) if "models" in names else Settings()
    r2 = run_supervised(Xc, "TPR_Flag", settings=s)
    ho = r2.holdout_all
    out["classification_TPR_Flag_all12"] = {
        "status": r2.status, "task": getattr(r2, "task", None), "selected": getattr(r2, "selected", None),
        "holdout_rows": ho[["model", "status"]].to_dict("records") if ho is not None else None}
    return out


@step("inferential_statistics")
def _():
    from src.inferential import tests as T
    v = df["Sales_Value"].to_numpy(float)
    mu0 = float(np.mean(v)) * 0.98
    t = T.one_sample_t(v, mu0)
    z_refused = T.z_test(v, mu0, None)
    z = T.z_test(v, mu0, float(np.std(v, ddof=1)))
    return {"t1": {k: getattr(t, k, None) for k in ("statistic", "p_value", "ci_low", "ci_high", "conclusion")},
            "z_without_sigma": {k: getattr(z_refused, k, None) for k in ("status", "message", "statistic")},
            "z_with_sigma": {k: getattr(z, k, None) for k in ("statistic", "p_value", "ci_low", "ci_high")}}


R["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
OUT.write_text(json.dumps(R, indent=1, default=str, ensure_ascii=False))
print("WROTE", OUT)
