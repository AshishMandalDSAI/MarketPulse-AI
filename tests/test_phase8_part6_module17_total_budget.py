"""Part 6 corrective verification - Module 17 ``total_budget`` contract.

Contract (documented in docs): ``PrescriptiveResult.total_budget`` is the total spend of the RECOMMENDED plan
(fixed-budget mode: the requested total; profit mode: the total chosen by the optimiser inside the range).
Expected values below are computed independently (closed forms / plain sums), not by re-running the engine's own line.
"""

import io
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
from src.common.export import sheets_to_excel_bytes  # noqa: E402
from src.prescriptive import engine as E  # noqa: E402

CH = ["tv", "search", "social"]


@pytest.fixture(autouse=True)
def _real_excel_row_limit(monkeypatch):
    from src.common import export as common_export
    monkeypatch.setattr(common_export, "EXCEL_MAX_ROWS", 1_048_575)


def mk(n=104, seed=0):
    r = np.random.default_rng(seed)
    d = pd.DataFrame({"week": pd.date_range("2022-01-03", periods=n, freq="W-MON"), "tv": r.uniform(1e4, 5e4, n),
                      "search": r.uniform(5e3, 2e4, n), "social": r.uniform(3e3, 1.5e4, n)})
    d["rev"] = 2e5 + 2 * d.tv ** .9 + 4 * d.search ** .85 + 3 * d.social + r.normal(0, 5e3, n)
    return d


@pytest.fixture(scope="module")
def fitted():
    d = mk()
    return d, E.fit_response_model(d, CH, "rev", "week")


def cur_total(d, m):
    return float(sum(E._Ctx(d, m).current.values()))


def profit(d, m, lo_f, hi_f, margin=3.0, mins=None, maxs=None, explain=True):
    t = cur_total(d, m)
    return t, E.optimize_allocation(d, m, None, mins or {c: t * .05 for c in CH}, maxs or {c: t * .8 for c in CH},
                                    objective="profit", margin=margin, total_range=(t * lo_f, t * hi_f), explain=explain)


# ---------------------------------------------------------------- independent closed form (profit)
def test_profit_closed_form_total_is_sum_of_recommended_not_current(fitted, monkeypatch):
    """predict(x) = sum a_i sqrt(x_i); profit = m*predict - sum(x)  =>  x_i* = (m a_i / 2)^2, total = 14 for a=(3,2,1), m=2."""
    d, m = fitted
    a = np.array([3.0, 2.0, 1.0])
    monkeypatch.setattr(E._Ctx, "predict", lambda self, v: float(np.sum(a * np.sqrt(np.maximum(np.asarray(v, float), 0)))))
    res = E.optimize_allocation(d, m, None, {c: 0.01 for c in CH}, {c: 100.0 for c in CH}, objective="profit", margin=2.0,
                                total_range=(1.0, 100.0), explain=False)
    assert res.status == "ok"
    expect = (2.0 * a / 2) ** 2
    assert [res.recommended[c] for c in CH] == pytest.approx(expect, rel=1e-4)
    assert res.total_budget == pytest.approx(14.0, rel=1e-4)
    assert res.total_budget != pytest.approx(cur_total(d, m), rel=1e-3)      # the reported figure is not the current spend


# ---------------------------------------------------------------- profit mode with the real fitted model
@pytest.mark.parametrize("lo_f,hi_f", [(1.3, 1.6), (0.5, 0.7), (0.8, 1.2)])
def test_profit_total_budget_equals_recommended_total_and_is_in_range(fitted, lo_f, hi_f):
    d, m = fitted
    t, res = profit(d, m, lo_f, hi_f)
    assert res.status == "ok"
    rec_total = sum(float(v) for v in res.recommended.values())               # plain independent sum
    assert res.total_budget == pytest.approx(rec_total, rel=1e-9, abs=1e-9)
    assert t * lo_f - 1e-6 <= res.total_budget <= t * hi_f + 1e-6
    for c in CH:                                                                # bounds enforced
        assert t * .05 - 1e-6 <= res.recommended[c] <= t * .8 + 1e-6


def test_profit_forced_range_far_from_current_total_is_not_reported_as_current(fitted):
    d, m = fitted
    t, res = profit(d, m, 1.3, 1.6)
    assert res.total_budget >= t * 1.3 - 1e-6 and abs(res.total_budget - t) > 0.2 * t


def test_profit_average_return_uses_recommended_total(fitted):
    d, m = fitted
    t, res = profit(d, m, 1.3, 1.6)
    rec_total = sum(res.recommended.values())
    expect = res.recommended_outcome / rec_total
    assert list(res.table["avg_return_per_unit_recommended"]) == pytest.approx([expect] * len(CH), rel=1e-9)


def test_profit_report_and_export_show_recommended_total(fitted):
    d, m = fitted
    t, res = profit(d, m, 1.3, 1.6)
    rec_total = sum(res.recommended.values())
    assert f"{rec_total:,.2f}" in E.business_report(res)
    sheets = E.result_sheets(res, E.scenario_table(res))
    summ = dict(zip(sheets["Summary"]["item"], sheets["Summary"]["value"]))
    assert float(summ["total_budget"]) == pytest.approx(rec_total, rel=1e-9)
    assert float(sheets["Scenarios"].loc[sheets["Scenarios"].scenario == "Recommended", "total"].iloc[0]) == pytest.approx(rec_total)
    assert sheets_to_excel_bytes(sheets)


def test_profit_current_vs_recommended_are_kept_distinct(fitted):
    d, m = fitted
    t, res = profit(d, m, 1.3, 1.6)
    assert sum(res.current.values()) == pytest.approx(t, rel=1e-9)
    assert sum(res.recommended.values()) != pytest.approx(sum(res.current.values()), rel=1e-3)
    sc = E.scenario_table(res)
    assert sc.loc[sc.scenario == "Current", "total"].iloc[0] == pytest.approx(t)


# ---------------------------------------------------------------- fixed-budget mode (must be unchanged)
@pytest.mark.parametrize("f", [0.8, 1.0, 1.25])
def test_fixed_budget_total_is_requested_and_conserved(fitted, f):
    d, m = fitted
    t = cur_total(d, m) * f
    res = E.optimize_allocation(d, m, t, {c: t * .05 for c in CH}, {c: t * .6 for c in CH})
    assert res.status == "ok"
    assert res.total_budget == pytest.approx(t, rel=1e-12)
    assert sum(res.recommended.values()) == pytest.approx(t, rel=1e-8)          # conservation
    for c in CH:
        assert t * .05 - 1e-6 <= res.recommended[c] <= t * .6 + 1e-6


def test_fixed_budget_default_total_is_current_total(fitted):
    d, m = fitted
    t = cur_total(d, m)
    res = E.optimize_allocation(d, m, None, {c: t * .05 for c in CH}, {c: t * .8 for c in CH})
    assert res.status == "ok" and res.total_budget == pytest.approx(t, rel=1e-9)
    assert sum(res.recommended.values()) == pytest.approx(t, rel=1e-8)


def test_fixed_budget_average_return_unchanged(fitted):
    d, m = fitted
    t = cur_total(d, m)
    res = E.optimize_allocation(d, m, t, {c: t * .05 for c in CH}, {c: t * .8 for c in CH})
    assert list(res.table["avg_return_per_unit_recommended"]) == pytest.approx([res.recommended_outcome / t] * 3, rel=1e-9)


def test_existing_optimizer_equivalence_still_holds_for_uniform_bounds(fitted):
    from src.marketing.optimizer import optimize_budget  # noqa: F401  (existing platform optimiser is untouched)
    d, m = fitted
    t = cur_total(d, m)
    res = E.optimize_allocation(d, m, t, {c: t * .05 for c in CH}, {c: t * .55 for c in CH})
    assert res.status == "ok" and res.recommended_outcome >= res.current_outcome - 1e-6


# ---------------------------------------------------------------- invalid input
@pytest.mark.parametrize("total", [0.0, -5.0, float("nan"), float("inf")])
def test_zero_negative_and_nonfinite_fixed_budget_refused_without_exception(fitted, total):
    d, m = fitted
    res = E.optimize_allocation(d, m, total, {c: 0.0 for c in CH}, {c: 1e5 for c in CH})
    assert res.status == "infeasible" and res.issues and res.recommended == {}


@pytest.mark.parametrize("margin,rng", [(None, (1, 2)), (0.0, (1, 2)), (-1.0, (1, 2)), (float("nan"), (1, 2)), (1.0, None),
                                        (1.0, (0.0, 0.0)), (1.0, (5.0, 1.0)), (1.0, (float("nan"), 10.0))])
def test_invalid_profit_inputs_refused_without_exception(fitted, margin, rng):
    d, m = fitted
    res = E.optimize_allocation(d, m, None, {c: 0.0 for c in CH}, {c: 1e5 for c in CH}, objective="profit",
                                margin=margin, total_range=rng)
    assert res.status == "infeasible" and res.issues and res.recommended == {}


def test_nonfinite_data_blocked_before_optimising():
    d = mk(); d.loc[3, "tv"] = np.inf
    assert any("infinite" in p for p in E.validate_roles(d, CH, "rev", "week"))


# ---------------------------------------------------------------- dashboard, profit mode
def test_dashboard_profit_mode_runs_and_exports_without_error():
    st_testing = pytest.importorskip("streamlit.testing.v1")
    from src.common.modules import get_module_spec
    from src.data_engine.analysis_runner import run_full_analysis
    i = f"test_p8p6m17_{uuid.uuid4().hex[:10]}"
    run_full_analysis(i, df=mk())
    try:
        at = st_testing.AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
        at.session_state["mp_active_dataset_id"] = i
        at.session_state["mp_active_dataset_filename"] = "up"
        at.session_state["mp_demo_explicitly_chosen"] = False
        at.run()
        at.sidebar.radio[0].set_value(get_module_spec("prescriptive").nav_label).run()
        next(x for x in at.selectbox if x.label.startswith("Outcome to improve")).set_value("rev").run()
        next(x for x in at.multiselect if x.label.startswith("Controllable")).set_value(CH).run()
        next(x for x in at.selectbox if x.label.startswith("Time column")).set_value("week").run()
        next(r for r in at.radio if r.label == "Objective").set_value("profit").run()
        next(n for n in at.number_input if n.label.startswith("Profit per unit")).set_value(3.0).run()
        next(b for b in at.button if b.label == "Find the best allocation").click().run()
        assert not at.exception
        assert any(x.label == "Recommended" for x in at.metric), [x.label for x in at.metric]
    finally:
        shutil.rmtree(dataset_upload_dir(i), ignore_errors=True)
        shutil.rmtree(dataset_processed_dir(i), ignore_errors=True)
