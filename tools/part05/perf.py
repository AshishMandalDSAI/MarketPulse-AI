"""PART 05 performance check: timings and peak memory on large synthetic datasets.
Usage: python perf.py ROOT OUT.json ROWS
"""
import json, resource, sys, time, uuid, shutil
ROOT, OUT, ROWS = sys.argv[1], sys.argv[2], int(sys.argv[3])
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd

def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

res = {"rows": ROWS, "steps": []}
def step(name, fn):
    t0 = time.perf_counter(); out = fn(); dt = time.perf_counter() - t0
    res["steps"].append({"step": name, "seconds": round(dt, 2), "peak_rss_mb": round(rss_mb(), 1)})
    print(name, round(dt, 2), "s", round(rss_mb()), "MB", flush=True); return out

rng = np.random.default_rng(7)
weeks = 208; stores = max(1, ROWS // weeks)
d = pd.date_range("2022-01-03", periods=weeks, freq="W-MON")
n = weeks * stores
df = pd.DataFrame({"Week": np.tile(d, stores), "Store": np.repeat([f"S{i:04d}" for i in range(stores)], weeks),
                   "Region": np.repeat(rng.choice(["North", "South", "East", "West"], stores), weeks)})
for c, s in [("TV_Spend", 3000), ("Search_Spend", 1200), ("Social_Spend", 800), ("Radio_Spend", 600)]:
    df[c] = rng.gamma(2.0, s / 2, n).round(2)
df["Price"] = rng.normal(100, 5, n).round(2)
df["Revenue"] = (20000 + 3 * np.log1p(df.TV_Spend) * 900 + 2 * np.log1p(df.Search_Spend) * 700
                 + np.log1p(df.Social_Spend) * 500 - 50 * df.Price + rng.normal(0, 800, n)).round(2)
res["shape"] = list(df.shape)

from src.data_engine.pipeline import profile_and_map
from src.data_engine import role_engine as RE
from src.data_engine.analysis_runner import run_full_analysis
from config.settings import dataset_upload_dir, dataset_processed_dir
_, m = step("profile_and_map (upload profiling)", lambda: profile_and_map(df))
roles = step("role engine suggest_roles", lambda: RE.suggest_roles(df))
mm = RE.apply_user_roles(roles, {}, targets=["Revenue"])
c, o = RE.legacy_overrides(mm)
did = f"upload_perf{uuid.uuid4().hex[:8]}"
step("run_full_analysis (Confirm & Analyze)", lambda: run_full_analysis(did, df=df, mapping=m, confirmed_roles=c, overrides=o))

from src.marketing_intel.context import build_context
from src.marketing_intel.response import fit_response
from src.marketing_intel.effectiveness import spend_mmm
from src.marketing_intel import scenarios as SC
from src.prescriptive import engine as E, limits as LIM
ctx = step("build_context", lambda: build_context(df, mm, "Revenue"))
model = step("driver response model (OLS HC3, fixed effects)", lambda: fit_response(ctx))
mmr = step("spend MMM", lambda: spend_mmm(ctx))
per, mmm = mmr["period_data"], mmr["mmm"]
ch = list(mmm.channels); tot = float(per[ch].tail(8).mean().sum())
mins, maxs, _ = LIM.default_limits(tot, ch, {k: float(per[k].tail(8).mean()) for k in ch})
r = step("optimize_allocation (multi-start)", lambda: E.optimize_allocation(per, mmm, tot, mins, maxs))
res["optimizer_status"] = getattr(r, "solution_status", None)
step("scenario sensitivity", lambda: SC.sensitivity(ctx, model))

from src.predictive.evaluate import run_supervised
from src.predictive.contracts import Settings
X = df.drop(columns=["Week", "Store"])
sup = step("predictive run_supervised regression (default models)", lambda: run_supervised(X.sample(min(len(X), 20000), random_state=1), "Revenue"))
res["predictive_status"] = getattr(sup, "status", None)
shutil.rmtree(dataset_upload_dir(did), ignore_errors=True); shutil.rmtree(dataset_processed_dir(did), ignore_errors=True)
res["peak_rss_mb"] = round(rss_mb(), 1)
json.dump(res, open(OUT, "w"), indent=1, default=str)
print("DONE", json.dumps(res, default=str)[:300])
