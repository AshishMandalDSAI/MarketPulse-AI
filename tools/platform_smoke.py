"""Cross-platform smoke checks (used by scripts/windows/validate_windows.ps1; also runs on Linux/macOS).

Usage:  python tools/platform_smoke.py OUT.json
Each check is recorded as PASS / FAIL with details; nothing is assumed. Exit code 1 if any check fails.
"""
from __future__ import annotations

import io
import json
import locale
import os
import platform
import shutil
import sys
import tempfile
import time
import traceback
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "platform_smoke.json"
results: list[dict] = []


def check(name):
    def deco(fn):
        t0 = time.perf_counter()
        try:
            detail = fn()
            results.append({"check": name, "result": "PASS", "detail": detail})
        except Exception as e:  # recorded, never hidden
            results.append({"check": name, "result": "FAIL", "detail": f"{type(e).__name__}: {e}",
                            "trace": traceback.format_exc()[-1200:]})
        results[-1]["seconds"] = round(time.perf_counter() - t0, 2)
        print(f"[{results[-1]['result']}] {name} ({results[-1]['seconds']} s)", flush=True)
        return fn
    return deco


@check("environment")
def _():
    return {"os": platform.platform(), "python": sys.version.split()[0], "executable": sys.executable,
            "preferred_encoding": locale.getpreferredencoding(False), "utf8_mode": sys.flags.utf8_mode,
            "project_path": str(ROOT), "path_has_spaces": " " in str(ROOT)}


@check("python_3_13")
def _():
    assert sys.version_info[:2] == (3, 13), f"expected Python 3.13, found {sys.version.split()[0]}"
    return sys.version.split()[0]


@check("paths_and_write_permissions")
def _():
    from config.settings import DATA_DIR, dataset_processed_dir, dataset_upload_dir
    from src.data_engine.analysis_runner import validate_dataset_id
    did = f"upload_smoke{uuid.uuid4().hex[:8]}"
    up, pr = dataset_upload_dir(did), dataset_processed_dir(did)
    up.mkdir(parents=True, exist_ok=True)
    pr.mkdir(parents=True, exist_ok=True)
    f = pr / "smoke ₹ test.json"                    # space + non-ASCII in a file name
    f.write_text(json.dumps({"currency": "₹"}, ensure_ascii=False), encoding="utf-8")
    ok = json.loads(f.read_text(encoding="utf-8"))["currency"] == "₹"
    shutil.rmtree(up, ignore_errors=True)
    shutil.rmtree(pr, ignore_errors=True)
    assert ok
    for bad in ("..\\..\\Windows", "../etc", "C:\\Windows", "a/b"):
        try:
            validate_dataset_id(bad)
        except Exception:
            continue
        raise AssertionError(f"unsafe dataset id was accepted: {bad!r}")
    return {"data_dir": str(DATA_DIR), "upload_dir_example": str(up), "separator": os.sep}


def _upload_frame(name: str, data: bytes):
    """The same parsing path the Upload page uses for CSV / XLSX."""
    import pandas as pd
    if name.lower().endswith(".xlsx"):
        return pd.read_excel(io.BytesIO(data))
    return pd.read_csv(io.BytesIO(data), encoding="utf-8", sep=None, engine="python")


@check("csv_and_excel_upload_parsing")
def _():
    import pandas as pd
    src = ROOT / "tests" / "fixtures" / "part02" / "kaggle_structure_replica.csv"
    df_csv = _upload_frame(src.name, src.read_bytes())
    with tempfile.TemporaryDirectory() as td:
        x = Path(td) / "replica with space.xlsx"
        df_csv.to_excel(x, index=False)
        df_x = _upload_frame(x.name, x.read_bytes())
    assert df_csv.shape == df_x.shape, (df_csv.shape, df_x.shape)
    pd.testing.assert_frame_equal(df_csv.reset_index(drop=True), df_x.reset_index(drop=True), check_dtype=False)
    return {"csv_shape": list(df_csv.shape), "xlsx_shape": list(df_x.shape)}


@check("role_engine_and_marketing_models")
def _():
    import pandas as pd
    from src.data_engine import role_engine as RE
    from src.marketing_intel.context import build_context
    from src.marketing_intel.effectiveness import run_effectiveness
    real = ROOT / "tests" / "fixtures" / "part02" / "real" / "synthetic_mmm_weekly_india_SAT.csv"
    src = real if real.exists() else ROOT / "tests" / "fixtures" / "part02" / "kaggle_structure_replica.csv"
    df = pd.read_csv(src)
    m = RE.suggest_roles(df)
    mode = RE.detect_mode(m).mode
    ctx = build_context(df, RE.apply_user_roles(m, {}, targets=["Sales_Value"]), "Sales_Value")
    eff = run_effectiveness(ctx)
    assert mode == "exposure" and eff.status == "ok" and not m.cols("ad_spend")
    return {"file": src.name, "mode": mode, "quality": eff.model.quality_reason}


@check("predictive_models_and_optional_libraries")
def _():
    import dataclasses
    import numpy as np
    import pandas as pd
    from src.predictive import classifiers as C
    from src.predictive.contracts import ALL_CLASSIFICATION_MODELS, Settings
    from src.predictive.evaluate import run_supervised
    avail = {k: C.availability(k) for k in ("xgboost", "lightgbm", "catboost")}
    rng = np.random.default_rng(1)
    n = 600
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"))
    X["y"] = np.where(X.a + rng.normal(0, 1, n) > 0, "yes", "no")
    names = {f.name for f in dataclasses.fields(Settings)}
    s = Settings(models=list(ALL_CLASSIFICATION_MODELS)) if "models" in names else Settings()
    r = run_supervised(X, "y", settings=s)
    rows = r.holdout_all[["model", "status"]].to_dict("records")
    assert r.status == "ok"
    for k, (ok, *_rest) in avail.items():
        label = {"xgboost": "XGBoost", "lightgbm": "LightGBM", "catboost": "CatBoost"}[k]
        st = next(x["status"] for x in rows if x["model"] == label)
        assert (st == "ok") == bool(ok), (label, st, ok)
    return {"optional": {k: v[0] for k, v in avail.items()}, "holdout": rows}


@check("report_generation_page")
def _():
    from streamlit.testing.v1 import AppTest
    from config.settings import DEMO_DATASET_ID
    from dashboard import navigation
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
    at.session_state["mp_active_dataset_id"] = DEMO_DATASET_ID
    at.session_state["mp_demo_explicitly_chosen"] = True
    at.run()
    at.sidebar.radio[0].set_value(next(p for p in navigation.PAGES if "Reports" in p)).run()
    assert not at.exception, at.exception
    labels = [b.proto.label for b in at.get("download_button")]
    assert any("Executive Summary" in x for x in labels), labels
    return {"download_buttons": labels}


@check("all_pages_render_demo")
def _():
    from streamlit.testing.v1 import AppTest
    from config.settings import DEMO_DATASET_ID
    from dashboard import navigation
    bad = []
    for label in navigation.PAGES:
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
        at.session_state["mp_active_dataset_id"] = DEMO_DATASET_ID
        at.session_state["mp_demo_explicitly_chosen"] = True
        at.run()
        at.sidebar.radio[0].set_value(label).run()
        if at.exception:
            bad.append((label, str(at.exception[0].value)[:150]))
    assert not bad, bad
    return {"pages": len(navigation.PAGES)}


OUT.write_text(json.dumps({"results": results, "all_passed": all(r["result"] == "PASS" for r in results)},
                          indent=1, default=str, ensure_ascii=False), encoding="utf-8")
print("WROTE", OUT)
sys.exit(0 if all(r["result"] == "PASS" for r in results) else 1)
