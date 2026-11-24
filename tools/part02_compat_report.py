"""Dataset compatibility report for the PART 02 engine (Markdown).

Usage:
    python tools/part02_compat_report.py FILE.csv|xlsx [FILE ...] [--targets A,B] [--out report.md]

For each file: size, inferred types, date range and frequency, missing values,
duplicates, brand / geography dimensions, the suggested role of every column
(confidence, evidence, alternatives, warnings), the analytics mode with its
evidence, each model's supported / unsupported capabilities, and the results
of actually running the four models for each target (all outcome candidates
unless --targets is given). Nothing is estimated by hand: every number comes
from running the engine on the file.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_engine import role_engine as RE  # noqa: E402
from src.marketing_intel import optimizer as O  # noqa: E402
from src.marketing_intel import scenarios as SC  # noqa: E402
from src.marketing_intel.capabilities import MODELS, model_capabilities  # noqa: E402
from src.marketing_intel.channels import run_channel_performance  # noqa: E402
from src.marketing_intel.context import build_context, infer_frequency, parse_date_column  # noqa: E402
from src.marketing_intel.effectiveness import run_effectiveness  # noqa: E402
from src.prescriptive import limits as LIM  # noqa: E402


def _read(path: Path) -> pd.DataFrame:
    return pd.read_excel(path) if path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(path)


def _f(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:,.{nd}f}"


def report_for(path: Path, targets=None) -> str:
    df = _read(path)
    m = RE.suggest_roles(df, dataset_name=path.name)
    mode = RE.detect_mode(m)
    out = [f"## {path.name}", ""]
    out.append(f"- Rows × columns: **{len(df):,} × {df.shape[1]}**")
    if m.date_col:
        d = parse_date_column(df[m.date_col])
        out.append(f"- Date column: `{m.date_col}` — {d.min().date()} to {d.max().date()}, "
                   f"{d.nunique()} distinct periods, frequency **{infer_frequency(d)}**")
    else:
        out.append("- Date column: none recognised")
    miss = df.isna().sum()
    out.append(f"- Missing values: {int(miss.sum()):,} cells"
               + (" (" + ", ".join(f"{c}: {int(v)}" for c, v in miss[miss > 0].items()) + ")" if miss.sum() else ""))
    out.append(f"- Exact duplicate rows: {int(df.duplicated().sum()):,}")
    for role in ("brand", "geography", "product", "sku"):
        for c in m.cols(role):
            out.append(f"- {RE.ROLE_INFO[role][1]}: `{c}` — {df[c].nunique()} levels "
                       f"({', '.join(map(str, sorted(df[c].dropna().unique())[:8]))}{'…' if df[c].nunique() > 8 else ''})")
    out += ["", f"**Analytics mode: {mode.label}**", ""] + [f"- {e}" for e in mode.evidence] + [""]
    out += ["| Column | Type | Suggested role | Channel | Confidence | Evidence | Alternatives | Warnings |",
            "|---|---|---|---|---|---|---|---|"]
    for c, r in m.columns.items():
        alts = "; ".join(f"{a['label']} ({a['confidence']:.2f})" for a in r.alternatives[:2])
        out.append(f"| `{c}` | {r.dtype} | {r.label} | {r.channel or ''} | {r.confidence:.2f} | "
                   f"{'; '.join(r.evidence).replace('|', '/')} | {alts} | {' '.join(r.warnings)} |")
    out += ["", "Dataset-level notes:"] + [f"- {w}" for w in m.warnings] + [""]
    tlist = targets or m.target_candidates
    if not tlist:
        out.append("_No outcome column recognised, so no model was run._")
        return "\n".join(out)
    for t in tlist:
        mm = RE.apply_user_roles(m, {}, targets=[t])
        caps = model_capabilities(mm, t)
        out += [f"### Target `{t}`", ""]
        for k, items in caps.items():
            out.append(f"**{MODELS[k]}**")
            for c in items:
                out.append(f"- {'✅' if c.available else '⛔'} {c.label} — {c.reason}"
                           + (f" (needs: {c.needs})" if c.needs and not c.available else "")
                           + (f" [{c.conditional}]" if c.conditional else ""))
            out.append("")
        t0 = time.time()
        ctx = build_context(df, mm, t)
        eff = run_effectiveness(ctx)
        out.append(f"Results (computed in {time.time() - t0:.1f}s):")
        if eff.status != "ok":
            out.append(f"- Effectiveness model: not available — {eff.message}")
            continue
        h = eff.model.holdout
        out.append(f"- Driver model: {len(eff.model.drivers)} drivers, {eff.model.n_rows:,} rows, quality "
                   f"**{eff.model.quality}** ({eff.model.quality_reason}); validation: {eff.model.split}")
        ch = eff.effects[eff.effects.kind == "channel"]
        if len(ch):
            out.append("- Channel effects: " + "; ".join(
                f"{r.driver}: elasticity {r.elasticity_at_mean:.4f}, p={r.p_value:.3g}"
                + (f", association share {r.association_share_pct:.1f}%" if "association_share_pct" in ch and np.isfinite(r.association_share_pct) else "")
                for r in ch.itertuples()))
        if eff.mmm:
            if eff.mmm.get("status") == "ok":
                tb = eff.mmm["table"]
                cols = [c for c in ("model_roi", "model_roas", "outcome_per_unit_spend") if c in tb]
                out.append(f"- Marketing-mix model ({eff.mmm['selected_model']}): " + "; ".join(
                    f"{r['channel']}: " + ", ".join(f"{c}={_f(r[c], 3)}" for c in cols) for _, r in tb.iterrows()))
            else:
                out.append(f"- Marketing-mix model: {eff.mmm.get('message')}")
        else:
            out.append("- Marketing-mix model / ROI / ROAS: not computed (no channel-wise advertising spend).")
        chan = run_channel_performance(ctx, eff.effects, eff.mmm)
        out.append(f"- Channel performance: {chan.status}"
                   + (f"; ranking basis: {chan.ranking_basis}; units: {', '.join(chan.unit_groups)}" if chan.status == "ok" else f" — {chan.message}"))
        if ctx.mode.mode == "exposure":
            for g in O.exposure_groups(ctx, eff.model):
                if g.available:
                    cur = O.current_levels(ctx, g.eligible)
                    tot = sum(cur.values())
                    mins, maxs, _ = LIM.default_limits(tot, g.eligible, cur)
                    r = O.optimize_exposure(ctx, eff.model, g, tot, mins, maxs)
                    out.append(f"- Exposure optimizer ({g.unit}): {r.status}, solver converged={r.solver.get('converged')}, "
                               f"estimated change {_f(r.change_pct)}% (range ±{_f((r.high - r.low) / 2 if r.low is not None else None, 0)})")
                else:
                    out.append(f"- Exposure optimizer ({g.unit}): not available — {' '.join(g.reasons)}")
        sim = SC.simulate(ctx, eff.model, SC.preset_scenarios(eff.model, ctx))
        out.append(f"- Scenario simulator: reliability {sim.reliability}; baseline actual {_f(sim.actual_baseline)} vs "
                   f"model {_f(sim.model_baseline)} per {'period' if ctx.date_col else 'record'}; scenarios: " + "; ".join(
                       f"{o.name}: {_f(o.change_pct)}%" + (" [extrapolation]" if o.extrapolation else "") for o in sim.outcomes))
        out.append("")
    return "\n".join(out)


def main(argv: list) -> None:
    targets = None
    out_path = None
    files = []
    i = 0
    while i < len(argv):
        if argv[i] == "--targets":
            targets = argv[i + 1].split(","); i += 2
        elif argv[i] == "--out":
            out_path = Path(argv[i + 1]); i += 2
        else:
            files.append(Path(argv[i])); i += 1
    text = "\n\n".join(report_for(f, targets) for f in files)
    if out_path:
        out_path.write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main(sys.argv[1:])
