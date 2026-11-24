"""App-wide page and chart audit (AppTest): every menu page, several datasets."""
import sys, json, html, re, uuid, shutil
ROOT = sys.argv[1]; sys.path.insert(0, ROOT)
import pandas as pd
from streamlit.testing.v1 import AppTest
from dashboard import navigation
from config.settings import dataset_processed_dir, dataset_upload_dir, DEMO_DATASET_ID
from src.data_engine.analysis_runner import run_full_analysis
from src.data_engine.pipeline import profile_and_map
from src.data_engine import role_engine as RE
from src.data_engine.currency import save_currency, choice_to_info, detect_currency
FIX = f"{ROOT}/tests/fixtures/part02"
def upload(name, targets, cur):
    df = pd.read_csv(f"{FIX}/{name}"); did = f"upload_aud{uuid.uuid4().hex[:8]}"
    _, m = profile_and_map(df); mm = RE.apply_user_roles(RE.suggest_roles(df), {}, targets=targets)
    c, o = RE.legacy_overrides(mm); run_full_analysis(did, df=df, mapping=m, confirmed_roles=c, overrides=o)
    std = pd.read_csv(dataset_processed_dir(did) / "standardized_data.csv")
    RE.save_mapping(dataset_processed_dir(did), RE.apply_user_roles(RE.suggest_roles(std), {}, targets=targets))
    save_currency(dataset_processed_dir(did), choice_to_info(cur, detect_currency(df), RE.monetary_columns(mm)))
    return {"mp_active_dataset_id": did, "mp_active_dataset_filename": name, "mp_demo_explicitly_chosen": False}, did
sets = {"demo(INR)": ({"mp_active_dataset_id": DEMO_DATASET_ID, "mp_demo_explicitly_chosen": True}, None)}
sets["spend(USD)"] = upload("spend_weekly.csv", ["Revenue"], "USD")
sets["exposure-kaggle(INR)"] = upload("kaggle_structure_replica.csv", ["Sales_Value", "Sales_Units"], "INR")
sets["general(none)"] = upload("general_sales.csv", ["Net_Sales"], "none")
def t(lay, ax=None):
    node = lay.get(ax, {}) if ax else lay; x = node.get("title")
    return ((x.get("text") if isinstance(x, dict) else x) or "").strip()
report = []
for sname, (state, did) in sets.items():
    for label in navigation.PAGES:
        tabs = [None] if label != navigation.STATISTICS else [x[1] for x in navigation.STATISTICS_TABS]
        for tab in tabs:
            at = AppTest.from_file(f"{ROOT}/app.py", default_timeout=600)
            for k, v in state.items(): at.session_state[k] = v
            at.run()
            if tab: at.session_state[navigation.STATS_TAB_FORCE_KEY] = tab
            try:
                at.sidebar.radio[0].set_value(label).run()
            except Exception as e:
                report.append({"set": sname, "page": label, "tab": tab, "problem": f"run error {type(e).__name__}: {e}"}); continue
            texts = [html.unescape(str(e.value)) for coll in (at.markdown, at.info, at.warning, at.error, at.caption) for e in coll]
            probs = []
            if at.exception: probs.append(f"EXCEPTION {at.exception[0].value[:200]}")
            for e in at.error: probs.append("ERRORBOX " + html.unescape(e.value)[:160])
            for w in texts:
                if "is not available for this dataset" in w or "Unexpected error" in w: probs.append("SAFEPAGE " + w[:160])
            charts = [json.loads(c.proto.spec) for c in at.get("plotly_chart")]
            for f in charts:
                lay = f.get("layout", {}); kinds = {d.get("type", "scatter") for d in f.get("data", [])}
                if not t(lay): probs.append(f"NO TITLE ({kinds})")
                if not kinds <= {"pie", "indicator", "table", "sunburst"}:
                    if not t(lay, "xaxis"): probs.append(f"NO X '{t(lay)}'")
                    if not t(lay, "yaxis"): probs.append(f"NO Y '{t(lay)}'")
                words = " ".join([t(lay), t(lay, "xaxis"), t(lay, "yaxis")])
                cur = sname[sname.index("(")+1:-1]
                if cur != "INR" and ("₹" in words or "INR" in words): probs.append(f"WRONG CURRENCY INR '{t(lay)}' | {words[:120]}")
                if cur != "USD" and ("USD" in words or "$" in words): probs.append(f"WRONG CURRENCY USD '{t(lay)}'")
            body = " ".join(texts)
            if sname.startswith("general") or sname.startswith("spend"):
                if "₹" in body: probs.append("₹ in page text")
            report.append({"set": sname, "page": label, "tab": tab, "charts": len(charts), "problems": probs})
            print(sname, "|", label, "|", tab or "", "| charts", len(charts), "| problems", len(probs), flush=True)
json.dump(report, open(sys.argv[2], "w"), indent=1, ensure_ascii=False)
for s in sets.values():
    if s[1]: shutil.rmtree(dataset_upload_dir(s[1]), ignore_errors=True)
