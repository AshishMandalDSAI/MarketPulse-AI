"""PART 02 browser pass: real Chrome, real Streamlit server.
Usage: python part02_browser.py BASE OUT_DIR TAG [--upload FILE --targets A,B] [--click-opt]
"""
import json, re, sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright

BASE, OUT, TAG = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
OUT.mkdir(parents=True, exist_ok=True)
arg = lambda k: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else None
UPLOAD, TARGETS = arg("--upload"), (arg("--targets") or "").split(",") if arg("--targets") else []
CLICK_OPT = "--click-opt" in sys.argv
PAGES = ["04 ·", "05 ·", "06 ·", "07 ·"]

def idle(page, timeout=300):
    time.sleep(0.8); t0 = time.time()
    while time.time() - t0 < timeout:
        if page.locator('[data-testid="stStatusWidget"]').count() == 0:
            time.sleep(0.7)
            if page.locator('[data-testid="stStatusWidget"]').count() == 0 and page.locator('[data-stale="true"]').count() == 0:
                return True
        time.sleep(0.4)
    return False

def charts_ready(page, timeout=60):
    """Wait until every Plotly chart has drawn its SVG (Streamlit being idle is not enough)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        n = page.locator('[data-testid="stPlotlyChart"]').count()
        drawn = page.locator('[data-testid="stPlotlyChart"] .main-svg').count()
        if n == 0 or drawn >= n:
            time.sleep(1.0); return True
        page.mouse.wheel(0, 400); time.sleep(0.5)
    return False


def issues(page):
    return {"exceptions": [e[:300] for e in page.locator('[data-testid="stException"]').all_inner_texts()],
            "errors": [e[:220] for e in page.locator('[data-testid="stAlertContentError"]').all_inner_texts()]}

report = {"tag": TAG, "steps": []}
with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/usr/bin/google-chrome", headless=True, args=["--no-sandbox"])
    page = b.new_context(viewport={"width": 1440, "height": int(arg("--height") or 1800)}).new_page()
    errs = []; page.on("pageerror", lambda e: errs.append(str(e)[:200]))
    page.goto(BASE, wait_until="networkidle"); idle(page)
    if UPLOAD:
        go = page.get_by_role("button", name=re.compile("Go to Upload", re.I))
        if go.count(): go.first.click(); idle(page)
        page.locator('input[type="file"]').set_input_files(UPLOAD); idle(page)
        page.get_by_text("2 · Column roles and target").first.scroll_into_view_if_needed(); time.sleep(1)
        page.screenshot(path=str(OUT / f"{TAG}_upload_roles.png"), full_page=False)
        btn = page.get_by_role("button", name=re.compile("Confirm & Analyze"))
        report["confirm_disabled_before_target"] = btn.first.is_disabled()
        for t in TARGETS:
            ms = page.locator('[data-testid="stMultiSelect"]').filter(has_text="Analysis target(s)").first
            ms.locator("input").first.click(); time.sleep(0.5)
            page.get_by_role("option", name=t, exact=True).first.click(); idle(page)
            page.keyboard.press("Escape"); time.sleep(0.3)
        page.get_by_text("2 · Column roles and target").first.scroll_into_view_if_needed(); time.sleep(0.5)
        page.screenshot(path=str(OUT / f"{TAG}_upload_roles_confirmed.png"), full_page=False)
        btn = page.get_by_role("button", name=re.compile("Confirm & Analyze"))
        report["confirm_disabled_after_target"] = btn.first.is_disabled()
        btn.first.click(); t0 = time.time()
        while time.time() - t0 < 900:
            if page.get_by_text(re.compile("Analysis complete")).count(): break
            time.sleep(2)
        idle(page); report["analysis_s"] = round(time.time() - t0, 1)
        report["upload_issues"] = issues(page)
    else:
        d = page.get_by_role("button", name=re.compile("demo dataset", re.I))
        if d.count(): d.first.click(); idle(page)
    labels = [t.strip() for t in page.locator('section[data-testid="stSidebar"] [role="radiogroup"] label').all_inner_texts()]
    for i, lab in enumerate(labels):
        pass
        page.locator('section[data-testid="stSidebar"] [role="radiogroup"] label').nth(i).click(); idle(page)
        charts_ready(page)
        name = f"{TAG}_{lab[:2]}"
        page.screenshot(path=str(OUT / f"{name}.png"), full_page=True)
        main = page.locator('[data-testid="stMainBlockContainer"]').first.inner_text()
        step = {"page": lab, **issues(page), "roi_mentions": len(re.findall(r"\bROI\b|\bROAS\b", main)),
                "mode_line": next((l for l in main.split("\n") if "Analytics" in l and "—" in l), "")[:160],
                "charts": page.locator('[data-testid="stPlotlyChart"]').count()}
        if CLICK_OPT and lab.startswith("06"):
            for bn in ("Find the best allocation", "Find the best exposure mix"):
                bt = page.get_by_role("button", name=bn)
                if bt.count() and not bt.first.is_disabled():
                    bt.first.click(); idle(page); charts_ready(page)
                    page.screenshot(path=str(OUT / f"{name}_result.png"), full_page=True)
                    m2 = page.locator('[data-testid="stMainBlockContainer"]').first.inner_text()
                    step["solver_line"] = next((l for l in m2.split("\n") if "Solver status" in l), "")
                    step["after_click"] = issues(page)
        report["steps"].append(step)
        print(json.dumps(step, ensure_ascii=False)[:600], flush=True)
    report["console_errors"] = errs
    b.close()
(OUT / f"{TAG}_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
print("REPORT", json.dumps({k: v for k, v in report.items() if k != "steps"}, ensure_ascii=False))
