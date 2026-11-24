"""PART 04 -- budget feasibility (remedies A-D, explicit reserve), optimizer result reporting and honest solution
statuses, Scenario Simulator completion (budget scenarios, sensitivity), Prescriptive Analytics across the three
analytics modes (no exposure-as-money, no ROAS without spend)."""
from __future__ import annotations

import html
import json
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
from src.marketing_intel import optimizer as O  # noqa: E402
from src.marketing_intel import scenarios as SC  # noqa: E402
from src.marketing_intel.context import build_context  # noqa: E402
from src.marketing_intel.effectiveness import run_effectiveness, spend_mmm  # noqa: E402
from src.marketing_intel.response import fit_response  # noqa: E402
from src.prescriptive import engine as E  # noqa: E402
from src.prescriptive import feasibility as FZ  # noqa: E402
from src.prescriptive import limits as LIM  # noqa: E402

FIX = ROOT / "tests" / "fixtures" / "part02"
APP = str(ROOT / "app.py")
P06 = "06 · ⚖️ Marketing Resource Optimizer"
P07 = "07 · 🧪 Marketing Scenario Simulator"
P15 = "15 · 🎯 Prescriptive Analytics"


def load(name):
    return pd.read_csv(FIX / name)


def mapped(df, targets):
    return RE.apply_user_roles(RE.suggest_roles(df), {}, targets=targets)


def fitted(name="spend_weekly.csv", target="Revenue", channels=None):
    df = load(name)
    ctx = build_context(df, mapped(df, [target]), target)
    mm = spend_mmm(ctx)
    per, mmm = mm["period_data"], mm["mmm"]
    if channels is not None:
        mmm = E.fit_response_model(per, channels, target, ctx.date_col, [])
    ch = list(mmm.channels)
    return per, mmm, ch, float(per[ch].tail(8).mean().sum())


@pytest.fixture(scope="module")
def four():
    return fitted()


# ============================================================================ I. feasibility diagnostics
class TestDiagnose:
    CH = ["a", "b", "c"]

    def test_feasible(self):
        r = FZ.diagnose(100, {"a": 0, "b": 0, "c": 0}, {"a": 50, "b": 50, "c": 50}, self.CH)
        assert r.feasible and not r.remedies

    def test_maximums_too_low_message_and_four_remedies(self):
        r = FZ.diagnose(100, {c: 0 for c in self.CH}, {c: 20 for c in self.CH}, self.CH, eligible_extra=("d",))
        assert not r.feasible and r.side == "max" and r.gap == pytest.approx(40)
        assert "The entered channel maximums cannot accommodate the full budget" in r.problems[0]
        codes = {m.code: m for m in r.remedies}
        assert set(codes) == {"A", "B", "C", "D"} and all(m.applicable for m in codes.values())
        assert codes["B"].action == {"set_total": 60}
        assert sum(codes["A"].action["set_max"].values()) >= 100 - 1e-9
        assert codes["C"].action == {"allow_reserve": True}
        assert codes["D"].action == {"add_channels": ["d"]}

    def test_no_extra_channels_means_remedy_d_not_applicable(self):
        r = FZ.diagnose(100, {c: 0 for c in self.CH}, {c: 20 for c in self.CH}, self.CH)
        d = [m for m in r.remedies if m.code == "D"][0]
        assert not d.applicable and "No further eligible channels" in d.detail

    def test_minimums_too_high(self):
        r = FZ.diagnose(100, {c: 50 for c in self.CH}, {c: 90 for c in self.CH}, self.CH)
        assert not r.feasible and r.side == "min" and r.gap == pytest.approx(50)
        codes = {m.code: m for m in r.remedies}
        assert codes["A"].applicable and sum(codes["A"].action["set_min"].values()) <= 100 + 1e-9
        assert codes["B"].action == {"set_total": 150}
        assert not codes["C"].applicable and not codes["D"].applicable

    def test_channel_max_below_min(self):
        r = FZ.diagnose(100, {"a": 30, "b": 0, "c": 0}, {"a": 10, "b": 90, "c": 90}, self.CH)
        assert not r.feasible and r.side == "channel" and "'a'" in r.problems[0]

    def test_reserve_makes_maximum_side_feasible_and_says_so(self):
        r = FZ.diagnose(100, {c: 0 for c in self.CH}, {c: 20 for c in self.CH}, self.CH, allow_reserve=True)
        assert r.feasible and "at least 40.00 will stay unallocated" in r.problems[0]

    def test_inputs_never_modified(self):
        mins, maxs = {c: 0.0 for c in self.CH}, {c: 20.0 for c in self.CH}
        FZ.diagnose(100, mins, maxs, self.CH)
        assert maxs == {c: 20.0 for c in self.CH} and mins == {c: 0.0 for c in self.CH}

    @pytest.mark.parametrize("status,solver,expected", [
        ("infeasible", None, "infeasible"), ("failed", None, "failed"), ("unavailable", None, "not_supported"),
        ("ok", {"converged": True, "starts_converged": 3, "starts_agreeing": 3}, "optimal"),
        ("ok", {"converged": True, "starts_converged": 3, "starts_agreeing": 2}, "feasible_not_proven"),
        ("ok", {"converged": False}, "feasible_not_proven"),
        ("ok", {"converged": True, "kept_start": True}, "feasible_not_proven")])
    def test_status_classification(self, status, solver, expected):
        assert FZ.classify(status, solver)[0] == expected


# ============================================================================ II. engine
class TestEngine:
    @pytest.mark.parametrize("n", [1, 2, 4])
    def test_one_two_and_multi_channel_optimal(self, n):
        name = {1: "one_channel.csv", 2: "two_channel.csv", 4: "spend_weekly.csv"}[n]
        per, mmm, ch, tot = fitted(name)
        mins, maxs, _ = LIM.default_limits(tot, ch, {c: float(per[c].tail(8).mean()) for c in ch})
        r = E.optimize_allocation(per, mmm, tot, mins, maxs)
        assert r.status == "ok" and r.solution_status == "optimal", r.solution_status_text
        assert r.budget_used == pytest.approx(tot, rel=1e-9) and r.unallocated == pytest.approx(0, abs=1e-6)
        assert list(r.utilization.channel) == ch and r.requested_total == pytest.approx(tot)
        assert r.solver["starts_converged"] >= 1 and r.solver["starts_agreeing"] >= 1

    def test_infeasible_max_no_fabricated_solution(self, four):
        per, mmm, ch, tot = four
        r = E.optimize_allocation(per, mmm, tot, {c: 0 for c in ch}, {c: 0.2 * tot for c in ch})
        assert r.status == "infeasible" and r.solution_status == "infeasible" and not r.recommended
        assert r.utilization is None and r.budget_used == 0

    def test_infeasible_min(self, four):
        per, mmm, ch, tot = four
        r = E.optimize_allocation(per, mmm, tot, {c: 0.3 * tot for c in ch}, {c: tot for c in ch})
        assert r.status == "infeasible" and "minimums add up to" in r.issues[0] and not r.recommended

    def test_reserve_when_maximums_are_too_low(self, four):
        per, mmm, ch, tot = four
        maxs = {c: 0.2 * tot for c in ch}
        r = E.optimize_allocation(per, mmm, tot, {c: 0.05 * tot for c in ch}, maxs, allow_reserve=True)
        assert r.status == "ok" and r.allow_reserve
        assert r.budget_used <= sum(maxs.values()) + 1e-6
        assert r.unallocated == pytest.approx(tot - r.budget_used) and r.unallocated >= 0.2 * tot - 1e-6
        assert all(r.recommended[c] <= maxs[c] + 1e-6 for c in ch)
        assert any("unallocated reserve" in a for a in r.assumptions)

    def test_reserve_never_exceeds_total(self, four):
        per, mmm, ch, tot = four
        r = E.optimize_allocation(per, mmm, tot, {c: 0 for c in ch}, {c: tot for c in ch}, allow_reserve=True)
        assert r.status == "ok" and r.budget_used <= tot + 1e-6 and r.unallocated >= -1e-9

    def test_reserve_cannot_fix_minimums_too_high(self, four):
        per, mmm, ch, tot = four
        r = E.optimize_allocation(per, mmm, tot, {c: 0.4 * tot for c in ch}, {c: tot for c in ch}, allow_reserve=True)
        assert r.status == "infeasible" and not r.recommended

    @pytest.mark.parametrize("factor", [0.7, 1.0, 1.3])
    def test_changed_total_budgets(self, four, factor):
        per, mmm, ch, tot = four
        t = tot * factor
        mins, maxs, _ = LIM.default_limits(t, ch)
        r = E.optimize_allocation(per, mmm, t, mins, maxs)
        assert r.status == "ok" and r.budget_used == pytest.approx(t, rel=1e-9)

    def test_profit_objective(self, four):
        per, mmm, ch, tot = four
        r = E.optimize_allocation(per, mmm, None, {c: 0 for c in ch}, {c: tot for c in ch}, objective="profit",
                                  margin=0.5, total_range=(0.5 * tot, 1.5 * tot))
        assert r.status == "ok" and r.solution_status in ("optimal", "feasible_not_proven")
        assert 0.5 * tot * (1 - 1e-9) <= r.budget_used <= 1.5 * tot * (1 + 1e-9)

    def test_solver_exception_gives_feasible_not_proven_never_optimal(self, four, monkeypatch):
        per, mmm, ch, tot = four
        monkeypatch.setattr(E, "minimize", lambda *a, **k: (_ for _ in ()).throw(ValueError("solver crashed")))
        r = E.optimize_allocation(per, mmm, tot, {c: 0 for c in ch}, {c: tot for c in ch})
        assert r.status == "ok" and r.solution_status == "feasible_not_proven"
        assert "not proven optimal" in r.solution_status_text
        assert r.budget_used == pytest.approx(tot)

    def test_solver_returning_nothing_is_failed(self, four, monkeypatch):
        per, mmm, ch, tot = four
        monkeypatch.setattr(E, "_solve", lambda *a, **k: (None, 0))
        r = E.optimize_allocation(per, mmm, tot, {c: 0 for c in ch}, {c: tot for c in ch})
        assert r.status == "failed" and r.solution_status == "failed" and not r.recommended

    def test_non_convergence_is_not_optimal(self, four, monkeypatch):
        per, mmm, ch, tot = four
        from scipy.optimize import minimize as real

        def fake(*a, **k):
            out = real(*a, **k)
            out.success = False
            return out
        monkeypatch.setattr(E, "minimize", fake)
        r = E.optimize_allocation(per, mmm, tot, {c: 0 for c in ch}, {c: tot for c in ch})
        assert r.solution_status != "optimal"

    def test_unfitted_model_not_supported(self, four):
        per, mmm, ch, tot = four
        import copy
        m2 = copy.copy(mmm)
        m2._model = None
        r = E.optimize_allocation(per, m2, tot, {c: 0 for c in ch}, {c: tot for c in ch})
        assert r.solution_status == "not_supported" and not r.recommended

    def test_missing_spend_values(self):
        df = load("spend_weekly.csv").copy()
        df.loc[[5, 40, 77], "TV_Spend"] = np.nan
        ctx = build_context(df, mapped(df, ["Revenue"]), "Revenue")
        mm = spend_mmm(ctx)
        assert mm["status"] == "ok"
        per = mm["period_data"]
        ch = list(mm["mmm"].channels)
        tot = float(per[ch].tail(8).mean().sum())
        r = E.optimize_allocation(per, mm["mmm"], tot, {c: 0 for c in ch}, {c: tot for c in ch})
        assert r.status == "ok"

    def test_no_roas_when_channels_are_not_money(self, four):
        per, mmm, ch, tot = four
        r = E.optimize_allocation(per, mmm, tot, {c: 0 for c in ch}, {c: tot for c in ch}, monetary=False)
        assert "historical_style_roas_current" not in r.table.columns and not r.monetary
        r2 = E.optimize_allocation(per, mmm, tot, {c: 0 for c in ch}, {c: tot for c in ch})
        assert "historical_style_roas_current" in r2.table.columns

    def test_utilization_binding_flags(self, four):
        per, mmm, ch, tot = four
        maxs = {c: 0.25 * tot for c in ch}
        r = E.optimize_allocation(per, mmm, tot, {c: 0 for c in ch}, maxs)
        assert r.status == "ok" and (r.utilization.binding == "max").all()
        assert (r.utilization.share_of_max_pct.round(6) == 100).all()


class TestExposureOptimizerStatus:
    def test_statuses(self):
        df = load("kaggle_structure_replica.csv")
        ctx = build_context(df, mapped(df, ["Sales_Value"]), "Sales_Value")
        model = run_effectiveness(ctx).model
        groups = {g.unit: g for g in O.exposure_groups(ctx, model)}
        g = groups["impressions"]
        cur = O.current_levels(ctx, g.eligible)
        tot = sum(cur.values())
        mins, maxs, _ = LIM.default_limits(tot, g.eligible, cur)
        r = O.optimize_exposure(ctx, model, g, tot, mins, maxs)
        assert r.solution_status in ("optimal", "feasible_not_proven") and r.solver["starts_converged"] >= 1
        r2 = O.optimize_exposure(ctx, model, groups["readers"], 1.0, {}, {})
        assert r2.solution_status == "not_supported"
        r3 = O.optimize_exposure(ctx, model, g, tot, {c: 0 for c in g.eligible}, {c: 1 for c in g.eligible})
        assert r3.solution_status == "infeasible" and not r3.recommended


# ============================================================================ III. scenarios
class TestScenarios:
    def test_budget_change_and_combination(self):
        df = load("spend_weekly.csv")
        ctx = build_context(df, mapped(df, ["Revenue"]), "Revenue")
        m = fit_response(ctx)
        b = SC.budget_change(m, 10)
        assert set(b) == set(ctx.channels) and all(v == {"type": "pct", "value": 10.0} for v in b.values())
        c = SC.combine(b, {"TV_Spend": {"type": "pct", "value": 10}})
        assert c["TV_Spend"]["value"] == pytest.approx(21.0) and c["Radio_Spend"]["value"] == pytest.approx(10.0)
        res = SC.simulate(ctx, m, [SC.Scenario("Budget +10%", b), SC.Scenario("Budget -10%", SC.budget_change(m, -10))])
        t = res.table()
        assert t.estimate.iloc[2] > t.estimate.iloc[1] > t.estimate.iloc[3]

    def test_sensitivity_one_at_a_time(self):
        df = load("brand_geo_spend.csv")
        ctx = build_context(df, mapped(df, ["Units_Sold"]), "Units_Sold")
        m = fit_response(ctx)
        s = SC.sensitivity(ctx, m, step=10)
        assert set(s.column) == {d.column for d in m.drivers}
        assert (s.range.diff().dropna() <= 1e-9).all()                      # sorted by impact
        promo = s.set_index("column").loc["Promo_Flag"]
        assert "share of records on" in promo.change_tested
        price = s.set_index("column").loc["Avg_Price"]
        assert price.change_high < 0 < price.change_low                     # higher price, lower units here

    def test_sensitivity_without_model(self):
        df = load("spend_weekly.csv")
        ctx = build_context(df, mapped(df, ["Revenue"]), "Revenue")
        assert SC.sensitivity(ctx, None).empty


# ============================================================================ IV. pages
def _app(state):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=600)
    for k, v in state.items():
        at.session_state[k] = v
    return at


def _goto(at, label):
    at.run()
    at.sidebar.radio[0].set_value(label).run()
    return at


def _texts(at):
    out = []
    for coll in (at.markdown, at.info, at.warning, at.error, at.success, at.caption):
        out.extend(html.unescape(str(e.value)) for e in coll)
    return out


def _layouts(at):
    return [json.loads(c.proto.spec)["layout"] for c in at.get("plotly_chart")]


def _t(lay, ax=None):
    node = lay.get(ax, {}) if ax else lay
    t = node.get("title")
    return ((t.get("text") if isinstance(t, dict) else t) or "").strip()


@pytest.fixture(scope="module")
def ups():
    from src.data_engine.analysis_runner import run_full_analysis
    from src.data_engine.currency import choice_to_info, detect_currency, save_currency
    from src.data_engine.pipeline import profile_and_map
    made = {}
    for key, name, targets, cur in [("spend", "spend_weekly.csv", ["Revenue"], "USD"),
                                    ("kaggle", "kaggle_structure_replica.csv", ["Sales_Value"], "INR"),
                                    ("general", "general_sales.csv", ["Net_Sales"], "USD")]:
        did = f"upload_p4{uuid.uuid4().hex[:10]}"
        df = load(name)
        _, m = profile_and_map(df)
        mm = mapped(df, targets)
        c, o = RE.legacy_overrides(mm)
        run_full_analysis(did, df=df, mapping=m, confirmed_roles=c, overrides=o)
        std = pd.read_csv(dataset_processed_dir(did) / "standardized_data.csv")
        RE.save_mapping(dataset_processed_dir(did), mapped(std, targets))
        save_currency(dataset_processed_dir(did), choice_to_info(cur, detect_currency(df), RE.monetary_columns(mm)))
        made[key] = {"mp_active_dataset_id": did, "mp_active_dataset_filename": name, "mp_demo_explicitly_chosen": False}
    yield made
    for s in made.values():
        shutil.rmtree(dataset_upload_dir(s["mp_active_dataset_id"]), ignore_errors=True)


def _set_maxima(at, value):
    for lab in sorted(x.label for x in at.number_input if "maximum" in x.label):
        [x for x in at.number_input if x.label == lab][0].set_value(value).run()


class TestOptimizerPage:
    def test_result_panel_reports_everything(self, ups):
        at = _goto(_app(ups["spend"]), P06)
        [b for b in at.button if b.label == "Find the best allocation"][0].click().run()
        assert not at.exception
        text = " ".join(_texts(at))
        assert "Solution status" in text and "Optimal." in text
        assert "Solver status:** Solved" in text and "Objective:**" in text
        labels = [m.label for m in at.metric]
        for need in ("Total budget (USD)", "Budget used (USD)", "Unallocated (USD)", "Feasibility"):
            assert need in labels, labels
        assert "Recommended allocation within each channel's limits" in [_t(l) for l in _layouts(at)]
        assert any("binding" in d.value.columns for d in at.dataframe)
        for l in _layouts(at):
            assert _t(l) and _t(l, "xaxis") and _t(l, "yaxis")

    def test_infeasible_shows_remedies_and_apply_b(self, ups):
        at = _goto(_app(ups["spend"]), P06)
        tot = [x for x in at.number_input if x.label.startswith("Total budget")][0].value
        _set_maxima(at, round(tot * 0.1, 2))
        text = " ".join(_texts(at))
        assert "Feasibility status: infeasible" in text and "cannot accommodate the full budget" in text
        assert [b for b in at.button if b.label == "Find the best allocation"][0].disabled
        for code in ("A", "B", "C"):
            assert [b for b in at.button if b.label == f"Apply {code}"], code
        before = {x.label: x.value for x in at.number_input}
        assert before[[k for k in before if k.startswith("Total budget")][0]] == tot     # nothing changed silently
        [b for b in at.button if b.label == "Apply B"][0].click().run()
        after = {x.label: x.value for x in at.number_input}
        assert after[[k for k in after if k.startswith("Total budget")][0]] == pytest.approx(round(tot * 0.1, 2) * 4, rel=1e-6)
        assert any("Feasibility status: feasible" in s.value for s in at.success)

    def test_apply_a_raises_maximums_and_marks_them_as_yours(self, ups):
        at = _goto(_app(ups["spend"]), P06)
        tot = [x for x in at.number_input if x.label.startswith("Total budget")][0].value
        _set_maxima(at, round(tot * 0.1, 2))
        [b for b in at.button if b.label == "Apply A"][0].click().run()
        ni = {x.label: x.value for x in at.number_input}
        assert sum(v for k, v in ni.items() if "maximum" in k) >= tot
        assert all("set by you" in k for k in ni if "maximum" in k)
        assert any("Feasibility status: feasible" in s.value for s in at.success)

    def test_apply_c_reserve_then_run_reports_unallocated(self, ups):
        at = _goto(_app(ups["spend"]), P06)
        tot = [x for x in at.number_input if x.label.startswith("Total budget")][0].value
        _set_maxima(at, round(tot * 0.1, 2))
        [b for b in at.button if b.label == "Apply C"][0].click().run()
        assert [c for c in at.checkbox if c.label.startswith("Allow an unallocated reserve")][0].value
        [b for b in at.button if b.label == "Find the best allocation"][0].click().run()
        un = [m for m in at.metric if m.label.startswith("Unallocated")][0]
        assert float(un.value.replace(",", "")) == pytest.approx(tot - round(tot * 0.1, 2) * 4, rel=1e-4)

    def test_remedy_d_adds_a_deselected_channel(self, ups):
        at = _goto(_app(ups["spend"]), P06)
        ms = [m for m in at.multiselect if m.label == "Channels to allocate across"][0]
        ms.set_value(["TV_Spend", "Search_Spend"]).run()
        tot = [x for x in at.number_input if x.label.startswith("Total budget")][0].value
        _set_maxima(at, round(tot * 0.2, 2))
        assert [b for b in at.button if b.label == "Apply D"]
        [b for b in at.button if b.label == "Apply D"][0].click().run()
        assert len([m for m in at.multiselect if m.label == "Channels to allocate across"][0].value) == 3

    def test_general_dataset_not_supported(self, ups):
        at = _goto(_app(ups["general"]), P06)
        assert not at.exception and "Missing inputs" in " ".join(_texts(at))


class TestPrescriptivePageModes:
    def test_exposure_columns_never_money_and_units_not_mixed(self, ups):
        at = _goto(_app(ups["kaggle"]), P15)
        [s for s in at.selectbox if s.label.startswith("Outcome")][0].set_value("Sales_Value").run()
        ch = [m for m in at.multiselect if m.label.startswith("Controllable")][0]
        ch.set_value(["TV_Impressions", "Print_Readership"]).run()
        text = " ".join(_texts(at))
        assert "different units" in text and "Not supported" in text
        assert not [b for b in at.button if b.label == "Find the best allocation"]
        [m for m in at.multiselect if m.label.startswith("Controllable")][0].set_value(
            ["TV_Impressions", "YouTube_Impressions"]).run()
        text = " ".join(_texts(at))
        assert "not money" in text
        obj = [r for r in at.radio if r.label == "Objective"][0]
        assert list(obj.options) == ["Maximise the outcome for a fixed total budget"]     # no profit objective
        assert any("(impressions)" in x.label for x in at.number_input if x.label.startswith("Total budget"))
        [b for b in at.button if b.label == "Find the best allocation"][0].click().run()
        assert not at.exception
        cols = " ".join(" ".join(map(str, d.value.columns)) for d in at.dataframe)
        assert "roas" not in cols.lower()

    def test_spend_dataset_keeps_profit_objective(self, ups):
        at = _goto(_app(ups["spend"]), P15)
        [s for s in at.selectbox if s.label.startswith("Outcome")][0].set_value("Revenue").run()
        [m for m in at.multiselect if m.label.startswith("Controllable")][0].set_value(["TV_Spend", "Search_Spend"]).run()
        obj = [r for r in at.radio if r.label == "Objective"][0]
        assert len(obj.options) == 2


class TestScenarioPage:
    def test_budget_input_and_sensitivity_chart(self, ups):
        at = _goto(_app(ups["spend"]), P07)
        assert not at.exception
        assert [x for x in at.number_input if x.label.startswith("Total marketing budget")]
        titles = [_t(l) for l in _layouts(at)]
        assert any(t.startswith("Sensitivity of Revenue") for t in titles)
        for l in _layouts(at):
            assert _t(l) and _t(l, "xaxis") and _t(l, "yaxis")
        [x for x in at.number_input if x.label.startswith("Total marketing budget")][0].set_value(20.0)
        at.text_input[0].set_value("Budget +20%")
        [b for b in at.button if b.label == "Add scenario"][0].click().run()
        tbl = [d.value for d in at.dataframe if "scenario" in d.value.columns][0]
        assert "Budget +20%" in set(tbl.scenario)

    def test_exposure_dataset_has_no_budget_input(self, ups):
        at = _goto(_app(ups["kaggle"]), P07)
        assert not at.exception
        assert not [x for x in at.number_input if x.label.startswith("Total marketing budget")]
        assert any(_t(l).startswith("Sensitivity of Sales_Value") for l in _layouts(at))
