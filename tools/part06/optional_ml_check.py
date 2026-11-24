import sys, time, json, dataclasses
sys.path.insert(0, "/app")
import numpy as np, pandas as pd
from src.predictive import classifiers as C
from src.predictive.contracts import Settings, ALL_CLASSIFICATION_MODELS
from src.predictive.evaluate import run_supervised
out = {"availability": {k: C.availability(k) for k in ("xgboost", "lightgbm", "catboost")}}
rng = np.random.default_rng(3); n = 1500
X = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"x{i}" for i in range(6)]); X["seg"] = rng.choice(list("abc"), n)
X["y"] = np.where(X.x0 + 0.6 * X.x1 + rng.normal(0, 1, n) > 0, "yes", "no")
s = Settings(models=list(ALL_CLASSIFICATION_MODELS)) if "models" in {f.name for f in dataclasses.fields(Settings)} else Settings()
t0 = time.time(); r = run_supervised(X, "y", settings=s)
ho = r.holdout_all
out["run_status"] = r.status; out["seconds"] = round(time.time() - t0, 1); out["selected"] = r.selected
out["holdout"] = ho[["model", "status"] + [c for c in ("note", "reason", "message") if c in ho]].to_dict("records")
print(json.dumps(out, default=str))
