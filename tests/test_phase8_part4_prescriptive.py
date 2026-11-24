"""Phase 8 Part 4 - Prescriptive engine and page: constraints, equivalence with the platform optimizer, scenarios, export."""

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
from src.common.export import ExportError  # noqa: E402
from src.marketing.optimizer import optimize_budget  # noqa: E402
from src.prescriptive import engine as E  # noqa: E402

CH = ["tv", "search", "social"]

@pytest.fixture(autouse=True)
def _real_excel_row_limit(monkeypatch):
    from src.common import export as common_export
    monkeypatch.setattr(common_export, "EXCEL_MAX_ROWS", 1_048_575)



def mk(n=104, seed=0, names=("tv", "search", "social"), target="rev"):
    r = np.random.default_rng(seed)
    d = pd.DataFrame({"week": pd.date_range("2022-01-03", periods=n, freq="W-MON"), names[0]: r.uniform(1e4, 5e4, n),
                      names[1]: r.uniform(5e3, 2e4, n), names[2]: r.uniform(3e3, 1.5e4, n)})
    d[target] = 2e5 + 2 * d[names[0]] ** .9 + 4 * d[names[1]] ** .85 + 3 * d[names[2]] + r.normal(0, 5e3, n)
    return d


@pytest.fixture(scope="module")
def fitted():
    d = mk()
    return d, E.fit_response_model(d, CH, "rev", "week")


def bounds(mmm, d, lo=.05, hi=.55):
    tot = sum(E._Ctx(d, mmm).current.values())
    return tot, {c: tot * lo for c in CH}, {c: tot * hi for c in CH}


# ------------------------------------------------------------ constraint validation
@pytest.mark.parametrize("total,mins,maxs,frag", [
    (-5, {}, {"a": 1}, "positive"), (0, {}, {"a": 1}, "positive"), (float("nan"), {}, {"a": 1}, "positive"),
    (100, {"a": -1}, {"a": 50}, "non-negative"), (100, {"a": 60}, {"a": 50}, "below minimum"),
    (100, {"a": 80}, {"a": 90, "b": 90}, "minimums add up"), (100, {"a": 0}, {"a": 40}, "less than the total"),
    (100, {"a": 0}, {"a": None}, "number")])
def test_constraint_validation_rejects(total, mins, maxs, frag):
    chans = list(maxs) or ["a"]
    if "b" in maxs:
        mins = {**mins, "b": 30}
    issues = E.validate_constraints(chans, total, mins, maxs)
    assert issues and any(frag in i for i in issues)


def test_constraint_validation_accepts_feasible():
    assert E.validate_constraints(["a", "b"], 100, {"a": 10, "b": 10}, {"a": 90, "b": 90}) == []


def test_range_validation():
    assert E.validate_constraints(["a"], None, {"a": 0}, {"a": 10}, (5, 3))
    assert E.validate_constraints(["a"], None, {"a": 0}, {"a": 10}, (5, 20)) == []    # range overlaps what the caps allow
    assert E.validate_constraints(["a"], None, {"a": 0}, {"a": 10}, (20, 30))         # caps below minimum total


def test_infeasible_request_returns_no_recommendation(fitted):
    d, m = fitted
    r = E.optimize_allocation(d, m, 1000.0, {c: 600.0 for c in CH}, {c: 700.0 for c in CH})
    assert r.status == "infeasible" and not r.recommended and r.issues


# ------------------------------------------------------------ correctness
def test_solution_satisfies_every_constraint_independently(fitted):
    d, m = fitted
    tot, mins, maxs = bounds(m, d)
    mins["tv"], maxs["search"] = tot * .30, tot * .20
    r = E.optimize_allocation(d, m, tot, mins, maxs)
    assert r.status == "ok"
    assert sum(r.recommended.values()) == pytest.approx(tot, rel=1e-6)
    for c in CH:
        assert mins[c] - 1e-6 <= r.recommended[c] <= maxs[c] + 1e-6
    assert E.check_solution(r.recommended, mins, maxs, tot) == []


def test_equivalent_to_platform_optimizer_with_uniform_bounds(fitted):
    d, m = fitted
    tot, mins, maxs = bounds(m, d)
    ref = optimize_budget(d, m, tot, .05, .55)
    r = E.optimize_allocation(d, m, tot, mins, maxs)
    assert r.recommended_outcome == pytest.approx(ref.optimized_predicted_target, rel=1e-5)
    assert r.current_outcome == pytest.approx(ref.current_predicted_target, rel=1e-9)
    assert r.recommended_outcome >= ref.optimized_predicted_target - 1e-6 * abs(ref.optimized_predicted_target)


def test_recommendation_never_worse_than_feasible_current(fitted):
    d, m = fitted
    tot, mins, maxs = bounds(m, d, 0.0, 1.0)
    r = E.optimize_allocation(d, m, tot, mins, maxs)
    assert r.current_feasible and r.recommended_outcome >= r.current_outcome - 1e-6


def test_tighter_constraints_cannot_improve_outcome(fitted):
    d, m = fitted
    tot, _, _ = bounds(m, d)
    loose = E.optimize_allocation(d, m, tot, {c: 0.0 for c in CH}, {c: tot for c in CH})
    tight = E.optimize_allocation(d, m, tot, {c: tot * .30 for c in CH}, {c: tot * .36 for c in CH})
    assert tight.status == "ok" and loose.recommended_outcome >= tight.recommended_outcome - 1e-6


def test_deterministic(fitted):
    d, m = fitted
    tot, mins, maxs = bounds(m, d)
    a, b = E.optimize_allocation(d, m, tot, mins, maxs), E.optimize_allocation(d, m, tot, mins, maxs)
    assert a.recommended == b.recommended


def test_explanations_traceable_and_risk_present(fitted):
    d, m = fitted
    tot, mins, maxs = bounds(m, d)
    r = E.optimize_allocation(d, m, tot, mins, maxs)
    assert set(r.table.channel) == set(CH) and r.table.explanation.str.len().gt(10).all()
    assert r.risk["level"] in ("low", "medium", "high") and r.risk["reasons"]
    assert r.risk["model_test_rmse"] == pytest.approx(m.test_metrics["rmse"])
    assert r.budget_response is not None and (r.budget_response.status == "ok").all()
    assert "not causal guarantees" in r.disclaimer


def test_extrapolation_flagged_as_high_risk(fitted):
    d, m = fitted
    tot, _, _ = bounds(m, d)
    r = E.optimize_allocation(d, m, tot, {c: 0.0 for c in CH}, {c: tot for c in CH})
    if r.table.outside_observed_range.any():
        assert r.risk["level"] == "high"


def test_profit_objective_needs_margin_and_range(fitted):
    d, m = fitted
    tot, mins, maxs = bounds(m, d, 0, 1)
    assert E.optimize_allocation(d, m, None, mins, maxs, objective="profit", total_range=(tot * .8, tot * 1.2)).status == "infeasible"
    assert E.optimize_allocation(d, m, None, mins, maxs, objective="profit", margin=.3).status == "infeasible"
    r = E.optimize_allocation(d, m, None, mins, maxs, objective="profit", margin=.3, total_range=(tot * .8, tot * 1.2))
    assert r.status == "ok" and tot * .8 - 1 <= sum(r.recommended.values()) <= tot * 1.2 + 1


def test_profit_with_tiny_margin_spends_the_minimum(fitted):
    d, m = fitted
    tot, mins, maxs = bounds(m, d, 0, 1)
    r = E.optimize_allocation(d, m, None, mins, maxs, objective="profit", margin=1e-6, total_range=(tot * .5, tot * 1.5))
    assert sum(r.recommended.values()) == pytest.approx(tot * .5, rel=1e-3)


def test_binding_bounds_solution_beats_random_feasible_allocations(fitted):
    d, m = fitted
    tot, mins, maxs = bounds(m, d)
    mins["social"], maxs["tv"], maxs["search"] = tot * .10, tot * .25, tot * .30
    r = E.optimize_allocation(d, m, tot, mins, maxs)
    assert r.status == "ok" and r.recommended["tv"] <= maxs["tv"] + 1e-6 and r.recommended["search"] <= maxs["search"] + 1e-6
    ctx = E._Ctx(d, m)
    rng = np.random.default_rng(0)
    lo = np.array([mins[c] for c in CH]); hi = np.array([maxs[c] for c in CH])
    best_random = -np.inf
    for _ in range(3000):
        x = E._clip_project(rng.dirichlet(np.ones(3)) * tot, lo, hi, tot)
        if abs(x.sum() - tot) < 1e-6 * tot and (x >= lo - 1e-9).all() and (x <= hi + 1e-9).all():
            best_random = max(best_random, ctx.predict(x))
    assert best_random > -np.inf and r.recommended_outcome >= best_random - 1e-6 * abs(best_random)
    # active constraint is reported
    assert set(r.table.binding_constraint) & {"max", "min"}


def test_check_solution_flags_each_violation_type():
    mins, maxs = {"a": 10., "b": 10.}, {"a": 60., "b": 60.}
    assert E.check_solution({"a": 50., "b": 50.}, mins, maxs, 100.) == []
    assert E.check_solution({"a": 5., "b": 95.}, mins, maxs, 100.)
    assert E.check_solution({"a": 70., "b": 30.}, mins, maxs, 100.)
    assert E.check_solution({"a": 40., "b": 40.}, mins, maxs, 100.)
    assert E.check_solution({"a": 40., "b": 40.}, mins, maxs, None, (100., 200.))
    assert E.check_solution({"a": 40., "b": 40.}, mins, maxs, None, (50., 90.)) == []


def test_violating_solver_output_is_discarded_not_shown(fitted, monkeypatch):
    d, m = fitted
    tot, mins, maxs = bounds(m, d)
    bad = np.array([tot, 0.0, 0.0])
    monkeypatch.setattr(E, "_solve", lambda *a, **k: ((1.0, bad, True, "forced"), 1))
    r = E.optimize_allocation(d, m, tot, mins, maxs)
    assert r.status == "failed" and r.recommended == {} and "violated" in r.message


# ------------------------------------------------------------ independent reference (audit): exact optimum of a known problem
class _Stub:
    """Concave separable response sum(a_i * sqrt(x_i)); its constrained optimum has a closed KKT form."""
    def __init__(self, a):
        self.a = np.asarray(a, float); self.channels = [f"c{i}" for i in range(len(a))]

    def predict(self, x):
        return float(np.sum(self.a * np.sqrt(np.maximum(np.asarray(x, float), 0))))


def _reference(a, lo, hi, B):
    """x_i = clip((a_i / (2 lam))^2, lo_i, hi_i) with sum(x) = B; lam found by bisection (independent of the engine)."""
    a, lo, hi = (np.asarray(v, float) for v in (a, lo, hi))
    g = lambda lam: np.clip((a / (2 * lam)) ** 2, lo, hi).sum() - B
    l, h = 1e-9, 1e9
    for _ in range(300):
        m = np.sqrt(l * h)
        l, h = (m, h) if g(m) > 0 else (l, m)
    return np.clip((a / (2 * np.sqrt(l * h))) ** 2, lo, hi)


def _engine_solution(a, lo, hi, B):
    ctx, lo, hi = _Stub(a), np.array(lo, float), np.array(hi, float)
    n = len(a)
    starts = [E._clip_project(np.full(n, B / n), lo, hi, B), E._clip_project((lo + hi) / 2, lo, hi, B)]
    best, _ = E._solve(ctx, lo, hi, B, None, "outcome", None, starts)
    return best[1], ctx


_REF_CASES = [([1, 2, 3], [0, 0, 0], [100, 100, 30], 100), ([1, 2, 3], [5, 5, 5], [100, 100, 100], 100),
              ([1, 2, 3], [20, 0, 0], [100, 100, 25], 100), ([1, 1, 4, 9], [10, 0, 0, 0], [50, 50, 20, 60], 120),
              ([3, 1], [0, 30], [60, 60], 70), ([2, 2, 2], [0, 0, 0], [100, 100, 100], 90)]


@pytest.mark.parametrize("a,lo,hi,B", _REF_CASES)
def test_optimizer_matches_independent_kkt_reference(a, lo, hi, B):
    x, ctx = _engine_solution(a, lo, hi, B)
    ref = _reference(a, lo, hi, B)
    assert x.sum() == pytest.approx(B, rel=1e-10)
    assert (x >= np.array(lo) - 1e-9).all() and (x <= np.array(hi) + 1e-9).all()
    assert ctx.predict(x) >= ctx.predict(ref) - 1e-8                       # never materially worse than the exact optimum
    np.testing.assert_allclose(x, ref, atol=5e-3)                          # and at the same allocation


def test_known_closed_form_value():
    # a = [1, 2, 3], budget 100, channel 3 capped at 30: x3 = 30, rest split 1:4 -> [14, 56, 30]
    x, ctx = _engine_solution([1, 2, 3], [0, 0, 0], [100, 100, 30], 100)
    np.testing.assert_allclose(x, [14, 56, 30], atol=5e-3)
    assert ctx.predict(x) == pytest.approx(np.sqrt(14) + 2 * np.sqrt(56) + 3 * np.sqrt(30), rel=1e-9)


def test_projection_alone_is_not_the_optimum_documenting_why_solver_bounds_matter():
    # clip-and-project of the unbounded optimum is feasible but strictly worse here (this is why the mutant is not equivalent)
    a, lo, hi, B = [1, 2, 3], np.zeros(3), np.array([100., 100., 30.]), 100.
    unbounded = (np.array(a, float) ** 2) / (np.array(a, float) ** 2).sum() * B
    proj = E._clip_project(unbounded, lo, hi, B)
    ctx = _Stub(a)
    assert ctx.predict(_reference(a, lo, hi, B)) - ctx.predict(proj) > 0.1


def test_forced_allocation_when_minimums_equal_total_and_maximums_equal_total():
    x, _ = _engine_solution([1, 2, 3], [10, 20, 70], [10, 20, 70], 100)
    np.testing.assert_allclose(x, [10, 20, 70], atol=1e-9)
    x, _ = _engine_solution([1, 2, 3], [0, 0, 0], [20, 30, 50], 100)
    np.testing.assert_allclose(x, [20, 30, 50], atol=1e-9)


def test_sum_to_budget_is_tight_on_the_real_model(fitted):
    d, m = fitted
    tot, mins, maxs = bounds(m, d)
    mins["tv"], maxs["search"] = tot * .30, tot * .20
    r = E.optimize_allocation(d, m, tot, mins, maxs)
    assert abs(sum(r.recommended.values()) - tot) <= 1e-9 * tot
    for c in CH:
        assert mins[c] <= r.recommended[c] <= maxs[c]


class _FakeOpt:
    def __init__(self, x, success=False):
        self.x, self.success, self.message = np.asarray(x, float), success, "forced"


def test_solver_returning_out_of_bounds_and_wrong_sum_is_repaired(monkeypatch):
    lo, hi, B = np.array([0., 0., 0.]), np.array([100., 100., 30.]), 100.
    monkeypatch.setattr(E, "minimize", lambda *a, **k: _FakeOpt([10.0, 70.0, 35.0]))   # channel 3 above its cap, sum 115
    x, _ = _engine_solution([1, 2, 3], lo, hi, B)
    assert (x >= lo - 1e-12).all() and (x <= hi + 1e-12).all()
    assert abs(x.sum() - B) <= 1e-9 * B


def test_solver_returning_slightly_off_sum_is_projected_to_exact_budget(monkeypatch):
    lo, hi, B = np.zeros(3), np.full(3, 100.), 100.
    monkeypatch.setattr(E, "minimize", lambda *a, **k: _FakeOpt([33.0, 33.0, 33.0 + 5e-7]))
    x, _ = _engine_solution([1, 1, 1], lo, hi, B)
    assert abs(x.sum() - B) <= 1e-9 * B


def test_never_worse_than_best_feasible_start_even_if_solver_fails(monkeypatch):
    lo, hi, B = np.zeros(3), np.full(3, 100.), 90.
    monkeypatch.setattr(E, "minimize", lambda *a, **k: _FakeOpt([90.0, 0.0, 0.0]))     # a poor but feasible point
    x, ctx = _engine_solution([2, 2, 2], lo, hi, B)
    assert ctx.predict(x) >= ctx.predict(np.array([30., 30., 30.])) - 1e-9              # the equal-split start is optimal here


def test_solver_returning_correct_sum_but_out_of_bounds_is_clipped_and_rebalanced(monkeypatch):
    lo, hi, B = np.zeros(3), np.array([60., 100., 100.]), 100.
    monkeypatch.setattr(E, "minimize", lambda *a, **k: _FakeOpt([110.0, -10.0, 0.0]))  # sums to 100 but breaks two bounds
    x, _ = _engine_solution([10, 1, 1], lo, hi, B)       # heavy weight on channel 1 makes the bad point look attractive
    assert (x >= lo - 1e-12).all() and (x <= hi + 1e-12).all() and abs(x.sum() - B) <= 1e-9 * B


# ------------------------------------------------------------ what-if reuse, export, unavailable states
def test_what_if_matches_existing_simulator(fitted):
    d, m = fitted
    from src.marketing.simulator import simulate_custom_allocation
    a = {"tv": 20000., "search": 15000., "social": 12000.}
    w = E.what_if_custom(d, m, a)
    ref = simulate_custom_allocation(d, m, a).scenarios[0]
    assert w["scenario_outcome"] == ref.predicted_target
    t = E.what_if_total(d, m, 70000.)
    assert t["scenario_total"] == pytest.approx(70000.)


def test_export_roundtrip_and_report(fitted):
    d, m = fitted
    tot, mins, maxs = bounds(m, d)
    r = E.optimize_allocation(d, m, tot, mins, maxs)
    from src.common.export import sheets_to_excel_bytes
    x = pd.ExcelFile(io.BytesIO(sheets_to_excel_bytes(E.result_sheets(r, E.scenario_table(r)))))
    assert {"Summary", "Channel recommendations", "Constraints", "Risk", "Assumptions", "Scenarios"} <= set(x.sheet_names)
    back = pd.read_excel(x, "Channel recommendations")
    assert back.recommended.sum() == pytest.approx(tot, rel=1e-6)
    rep = E.business_report(r)
    assert "Channel recommendations" in rep and f"{r.recommended_outcome:,.2f}" in rep
    for c in CH:
        assert c in rep


def test_export_of_unavailable_result_refused():
    with pytest.raises(ExportError):
        E.result_sheets(E.PrescriptiveResult())
    with pytest.raises(ValueError):
        E.business_report(E.PrescriptiveResult())


def test_unfitted_model_gives_unavailable_not_invented_numbers():
    d = mk(8)
    m = E.fit_response_model(d, CH, "rev", "week")
    r = E.optimize_allocation(d, m, 1000., {}, {c: 1000. for c in CH})
    assert r.status == "unavailable" and not r.recommended and r.recommended_outcome == 0.0


def test_role_validation():
    d = mk(); d["flat"] = 5.0; d["txt"] = "a"
    assert E.validate_roles(d, CH, "rev") == []
    assert E.validate_roles(d, [], "rev")
    assert any("vary" in i for i in E.validate_roles(d, ["flat"], "rev"))
    assert any("numeric" in i for i in E.validate_roles(d, ["txt"], "rev"))
    assert any("both" in i for i in E.validate_roles(d, ["rev"], "rev"))
    assert E.validate_roles(d, CH, "missing") and E.validate_roles(pd.DataFrame(), CH, "rev")
    d["neg"] = -d.tv
    assert any("negative" in i for i in E.validate_roles(d, ["neg"], "rev"))


def test_roles_from_report_missing_or_corrupt(tmp_path):
    assert E.roles_from_report(tmp_path / "none.json").source == "none"
    p = tmp_path / "bad.json"; p.write_text("{not json")
    assert E.roles_from_report(p).source == "unreadable"


def test_dynamic_column_names_no_hardcoding():
    d = mk(names=("ads_alpha", "ads_beta", "ads_gamma"), target="units_sold")
    chs = ["ads_alpha", "ads_beta", "ads_gamma"]
    m = E.fit_response_model(d, chs, "units_sold", "week")
    tot = sum(E._Ctx(d, m).current.values())
    r = E.optimize_allocation(d, m, tot, {c: tot * .05 for c in chs}, {c: tot * .6 for c in chs})
    assert r.status == "ok" and set(r.recommended) == set(chs)


# ------------------------------------------------------------ dashboard
st_testing = pytest.importorskip("streamlit.testing.v1")
from src.common.modules import get_module_spec  # noqa: E402
LABEL = get_module_spec("prescriptive").nav_label


@pytest.fixture
def make_dataset():
    from src.data_engine.analysis_runner import run_full_analysis
    ids = []

    def make(df):
        i = f"test_p8p4p_{uuid.uuid4().hex[:10]}"
        run_full_analysis(i, df=df)
        ids.append(i)
        return i
    yield make
    for i in ids:
        shutil.rmtree(dataset_upload_dir(i), ignore_errors=True)
        shutil.rmtree(dataset_processed_dir(i), ignore_errors=True)


def _open(i):
    at = st_testing.AppTest.from_file(str(ROOT / "app.py"), default_timeout=240)
    at.session_state["mp_active_dataset_id"] = i
    at.session_state["mp_active_dataset_filename"] = "d.csv"
    at.session_state["mp_demo_explicitly_chosen"] = False
    at.run(); at.sidebar.radio[0].set_value(LABEL).run()
    return at


def _text(at):
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption] + [i.value for i in at.info]
    parts += [w.value for w in at.warning] + [e.value for e in at.error]
    return "\n".join(parts)


MAPPED = dict(names=("tv_spend", "search_spend", "social_spend"), target="revenue")


def test_page_runs_optimisation_and_offers_downloads(make_dataset):
    at = _open(make_dataset(mk(**MAPPED)))
    assert not at.exception
    next(b for b in at.button if b.label == "Find the best allocation").click().run()
    assert not at.exception
    assert any("recommended" in m.label.lower() for m in at.metric)
    assert len(at.get("download_button")) >= 3


def test_page_manual_selection_when_no_mapping(make_dataset):
    at = _open(make_dataset(mk()))                 # names carry no spend evidence: nothing is guessed
    assert not at.exception and "Cannot optimise yet" in _text(at)
    assert not any(b.label == "Find the best allocation" for b in at.button)
    next(s for s in at.selectbox if s.label.startswith("Outcome")).set_value("rev").run()
    next(m for m in at.multiselect if m.label.startswith("Controllable")).set_value(CH).run()
    next(b for b in at.button if b.label == "Find the best allocation").click().run()
    assert not at.exception and any("recommended" in m.label.lower() for m in at.metric)


def test_page_blocks_infeasible_constraints(make_dataset):
    at = _open(make_dataset(mk(**MAPPED)))
    first_min = next(n for n in at.number_input if n.label.endswith("- minimum"))
    first_min.set_value(1e9).run()
    assert at.error
    assert next(b for b in at.button if b.label == "Find the best allocation").disabled


def test_dataset_without_channels_shows_guidance_not_crash(make_dataset):
    r = np.random.default_rng(0)
    d = pd.DataFrame({"x": r.normal(size=60), "y": r.normal(size=60)})
    at = _open(make_dataset(d))
    assert not at.exception


def test_switch_does_not_show_stale_recommendation(make_dataset):
    a = make_dataset(mk(seed=1, **MAPPED))
    b = make_dataset(mk(seed=2, names=("p1_spend", "p2_spend", "p3_spend"), target="sales"))
    at = _open(a)
    next(x for x in at.button if x.label == "Find the best allocation").click().run()
    assert any("recommended" in m.label.lower() for m in at.metric)
    at.session_state["mp_active_dataset_id"] = b
    at.run()
    assert not any("recommended" in m.label.lower() for m in at.metric)
