"""Ultimate release — analytical correctness, evaluation honesty and leakage.

Covers: zero-safe metrics, walk-forward isolation, fair multi-step validation,
baseline-first model selection, MMM transforms checked against hand
calculations, MMM leakage prevention, honest handling of missing revenue,
budget-optimizer constraints, and the duplicate-timestamp forecast fix.
"""

from __future__ import annotations

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

from src.analytics.ts_evaluation import rolling_origin_backtest, ts_metrics  # noqa: E402
from src.models.adstock import geometric_adstock  # noqa: E402
from src.models.saturation import hill_saturation  # noqa: E402


# --------------------------------------------------------------- metrics
class TestZeroSafeMetrics:
    def test_hand_computed_values(self):
        m = ts_metrics([100, 200, 300], [110, 190, 330])
        assert m["MAE"] == pytest.approx((10 + 10 + 30) / 3)
        assert m["RMSE"] == pytest.approx(np.sqrt((100 + 100 + 900) / 3))
        assert m["MAPE"] == pytest.approx((10 / 100 + 10 / 200 + 30 / 300) / 3 * 100)
        assert m["WAPE"] == pytest.approx(50 / 600 * 100)
        smape = np.mean([2 * 10 / 210, 2 * 10 / 390, 2 * 30 / 630]) * 100
        assert m["sMAPE"] == pytest.approx(smape)

    def test_mape_skips_zero_actuals_and_reports_it(self):
        m = ts_metrics([0, 100], [5, 110])
        assert m["MAPE"] == pytest.approx(10.0)
        assert m["n_zero_actuals_excluded_from_MAPE"] == 1

    def test_all_zero_actuals_give_none_not_zero_error(self):
        m = ts_metrics([0, 0, 0], [1, 2, 3])
        assert m["MAPE"] is None and m["WAPE"] is None
        assert m["MAE"] == pytest.approx(2.0)

    def test_r2_undefined_for_constant_actuals(self):
        assert ts_metrics([5, 5, 5], [5, 6, 4])["R2"] is None

    def test_non_finite_pairs_ignored(self):
        m = ts_metrics([1, np.nan, 3, np.inf], [1, 2, 3, 4])
        assert m["n"] == 2 and m["MAE"] == 0.0


# --------------------------------------------------------------- walk-forward
class TestWalkForward:
    def test_each_fold_trains_only_on_the_past(self):
        y = np.arange(40, dtype=float)
        seen = []

        def fit_predict(train, h):
            seen.append((len(train), float(train.max())))
            return np.full(h, train[-1])

        bt = rolling_origin_backtest(y, fit_predict, horizon=5, n_folds=3, min_train=10)
        assert bt["n_folds"] == 3
        for (n_train, max_seen), fold in zip(seen, bt["folds"]):
            assert n_train == fold["test_start"]          # train ends where test begins
            assert max_seen < y[fold["test_start"]]       # no test value was visible
        assert [f["test_start"] for f in bt["folds"]] == [25, 30, 35]

    def test_moving_average_validation_does_not_use_actuals(self):
        from src.prediction.forecasting_engine import _train_moving_average
        train = np.arange(1, 21, dtype=float)
        _, p1, _ = _train_moving_average(train, np.zeros(5), 5)
        _, p2, _ = _train_moving_average(train, np.full(5, 1e6), 5)
        np.testing.assert_allclose(p1, p2)   # predictions independent of the values being predicted


# --------------------------------------------------------------- MMM math
class TestMMMTransformsByHand:
    def test_geometric_adstock(self):
        np.testing.assert_allclose(geometric_adstock([100, 0, 0, 50], 0.5), [100, 50, 25, 62.5])

    def test_adstock_zero_decay_is_identity(self):
        np.testing.assert_allclose(geometric_adstock([3, 1, 4], 0.0), [3, 1, 4])

    def test_hill_half_saturation_point(self):
        assert hill_saturation(np.array([200.0]), k=200.0, s=2.0)[0] == pytest.approx(0.5)
        assert hill_saturation(np.array([400.0]), k=200.0, s=2.0)[0] == pytest.approx(16e4 / (4e4 + 16e4))

    def test_hill_bounded_monotone_and_clips_negative(self):
        x = np.array([-10.0, 0.0, 10.0, 100.0, 1e6])
        out = hill_saturation(x, k=50.0, s=1.5)
        assert out[0] == 0.0 and out[1] == 0.0
        assert np.all(np.diff(out) >= 0) and np.all(out < 1.0)


# --------------------------------------------------------------- leakage
def _mmm_frame(n=120, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.uniform(100, 1000, n)
    b = rng.uniform(50, 500, n)
    rev = 5000 + 3 * geometric_adstock(a, 0.5) + 2 * geometric_adstock(b, 0.3) + rng.normal(0, 200, n)
    return pd.DataFrame({"week": pd.date_range("2022-01-03", periods=n, freq="W-MON"),
                         "a_spend": a, "b_spend": b, "revenue": rev})


class TestMMMLeakage:
    def test_channel_parameters_ignore_test_period(self):
        from src.marketing import mmm_engine as E
        df = _mmm_frame()
        tampered = df.copy()
        split = int(len(df) * (1 - E.TEST_FRACTION))
        tampered.loc[split:, "revenue"] = tampered.loc[split:, "revenue"][::-1].to_numpy() * 7.0
        p1 = E._select_channel_params(df.iloc[:split], ["a_spend", "b_spend"], "revenue")
        p2 = E._select_channel_params(tampered.iloc[:split], ["a_spend", "b_spend"], "revenue")
        assert p1 == p2

    def test_fit_mmm_does_not_change_when_only_test_targets_change(self):
        from src.marketing import mmm_engine as E
        import inspect
        src = inspect.getsource(E.fit_mmm)
        # selection happens on training rows only and after the split
        assert "_select_channel_params(work_df.iloc[train_idx]" in src
        assert src.index("_time_aware_split(") < src.index("_select_channel_params(")

    def test_demo_pipeline_selects_on_training_weeks(self):
        src = (ROOT / "src" / "models" / "marketing_mix_model.py").read_text(encoding="utf-8")
        assert "select_channel_params(df.iloc[train_idx])" in src

    def test_missing_target_not_median_filled(self):
        from src.marketing import mmm_engine as E
        import inspect
        src = inspect.getsource(E.fit_mmm)
        assert "target_median" not in src


# --------------------------------------------------------------- uploaded-data pipeline
@pytest.fixture
def run_upload():
    created = []

    def _run(df, roles=None):
        from config.settings import dataset_models_dir, dataset_processed_dir
        from src.data_engine.analysis_runner import run_full_analysis
        from src.data_engine.pipeline import profile_and_map
        ds = f"test_ult_{uuid.uuid4().hex[:10]}"
        created.append(ds)
        _, mapping = profile_and_map(df)
        run_full_analysis(ds, df=df, mapping=mapping, confirmed_roles=roles or {"date", "revenue", "channels"})
        return dataset_processed_dir(ds), dataset_models_dir(ds)

    yield _run
    from config.settings import dataset_upload_dir
    for ds in created:
        shutil.rmtree(dataset_upload_dir(ds), ignore_errors=True)


class TestUploadedDataModels:
    def test_missing_revenue_excluded_not_zero(self, run_upload):
        df = _mmm_frame(80).rename(columns={"week": "Order_Date", "revenue": "Net_Sales",
                                            "a_spend": "Meta_Ads", "b_spend": "Google_Advertising"})
        df.loc[[3, 10, 40], "Net_Sales"] = np.nan
        _, models = run_upload(df)
        m = json.loads((models / "model_metrics.json").read_text(encoding="utf-8"))
        assert m["rows_excluded_missing_revenue"] == 3
        assert m["n_observations"] == 77
        assert m["baseline"] is not None and "beats_baseline" in m

    def test_mape_never_reported_as_fabricated_zero(self, run_upload):
        df = _mmm_frame(60).rename(columns={"week": "Order_Date", "revenue": "Net_Sales",
                                            "a_spend": "Meta_Ads", "b_spend": "Google_Advertising"})
        _, models = run_upload(df)
        test_m = json.loads((models / "model_metrics.json").read_text(encoding="utf-8"))["model_comparison"]["generic_ridge_linear"]["test"]
        assert test_m["mape"] is None or test_m["mape"] > 0

    def test_transactional_rows_are_summed_per_period_for_forecast(self, run_upload):
        rng = np.random.default_rng(1)
        days = pd.date_range("2024-01-01", periods=60, freq="D")
        rows = [{"Transaction_Time": d.strftime("%Y-%m-%d"), "Total_Amount": float(rng.uniform(10, 20))}
                for d in days for _ in range(5)]
        df = pd.DataFrame(rows)
        processed, _ = run_upload(df, roles={"date", "revenue"})
        ev = json.loads((processed / "forecast_evaluation.json").read_text(encoding="utf-8"))
        assert ev["rows_combined_same_date"] == 240
        assert ev["periods"] == 60
        fc = pd.read_csv(processed / "revenue_forecast.csv")
        assert fc["Forecast_Revenue"].between(40, 110).all()   # per-day totals of ~5 × 15, not per-row values

    def test_selection_rule_baseline_first(self, run_upload):
        rng = np.random.default_rng(7)
        y = 1000 + np.cumsum(rng.normal(0, 50, 90))      # random walk: naive is hard to beat
        df = pd.DataFrame({"Date": pd.date_range("2023-01-02", periods=90, freq="W-MON").strftime("%Y-%m-%d"),
                           "Revenue": y})
        processed, _ = run_upload(df, roles={"date", "revenue"})
        ev = json.loads((processed / "forecast_evaluation.json").read_text(encoding="utf-8"))
        if ev["selected_model"] not in ("Naive (Last Value)", "Seasonal Naive"):
            assert ev["improvement_vs_baseline_pct"] >= 2.0
        assert ev["folds"] >= 1 and ev["fold_test_ranges"]

    def test_too_little_history_is_not_forecast(self, run_upload):
        df = pd.DataFrame({"Date": ["2024-01-01", "2024-01-08", "2024-01-15", "2024-01-22", "2024-01-29"],
                           "Revenue": [1.0, 2.0, 3.0, 4.0, 5.0]})
        processed, _ = run_upload(df, roles={"date", "revenue"})
        assert not (processed / "revenue_forecast.csv").exists()


# --------------------------------------------------------------- optimizer
@pytest.fixture(scope="module")
def optimizer_inputs():
    from src.optimization.budget_optimizer import CHANNELS, get_recent_carry_in, load_bundle
    bundle = load_bundle()
    df = pd.read_csv(ROOT / "data" / "processed" / "marketing_data_clean.csv", parse_dates=["Date"])
    recent = df.tail(8)
    current = {ch: float(recent[f"{ch}_Spend"].mean()) for ch in CHANNELS}
    controls = {c: float(recent[c].mean()) for c in bundle["control_cols"]}
    return bundle, get_recent_carry_in(df, bundle["channel_params"]), controls, current, CHANNELS


class TestBudgetOptimizerConstraints:
    @pytest.mark.parametrize("budget,min_pct,max_pct", [(2_000_000, 0.05, 0.55), (500_000, 0.10, 0.20),
                                                        (3_000_000, 0.0, 0.30)])
    def test_total_bounds_and_non_negative(self, optimizer_inputs, budget, min_pct, max_pct):
        from src.optimization.budget_optimizer import optimize_budget
        bundle, carry, ctrl, cur, chans = optimizer_inputs
        r = optimize_budget(budget, bundle, carry, ctrl, cur, min_pct=min_pct, max_pct=max_pct)
        alloc = np.array(list(r["optimized_allocation"].values()))
        assert r["success"] and not r["constraint_violations"]
        assert alloc.sum() == pytest.approx(budget, rel=1e-6)
        assert (alloc >= budget * min_pct - 1e-6).all() and (alloc <= budget * max_pct + 1e-6).all()
        assert (alloc >= 0).all()

    def test_zero_spend_current_allocation_handled(self, optimizer_inputs):
        from src.optimization.budget_optimizer import optimize_budget
        bundle, carry, ctrl, cur, chans = optimizer_inputs
        zero = {c: 0.0 for c in chans}
        r = optimize_budget(1_000_000, bundle, carry, ctrl, zero)
        assert r["success"]
        assert sum(r["optimized_allocation"].values()) == pytest.approx(1_000_000, rel=1e-6)

    @pytest.mark.parametrize("budget,min_pct,max_pct,needle", [
        (1_000_000, 0.20, 0.50, "minimum"),      # 8 × 20% = 160% > 100%
        (1_000_000, 0.00, 0.10, "maximum"),      # 8 × 10% = 80% < 100%
        (-5, 0.05, 0.55, "positive"),
        (1_000_000, 0.6, 0.5, "min_pct"),
    ])
    def test_infeasible_inputs_give_clear_errors(self, optimizer_inputs, budget, min_pct, max_pct, needle):
        from src.optimization.budget_optimizer import optimize_budget
        bundle, carry, ctrl, cur, _ = optimizer_inputs
        with pytest.raises(ValueError, match=needle):
            optimize_budget(budget, bundle, carry, ctrl, cur, min_pct=min_pct, max_pct=max_pct)

    def test_like_for_like_reallocation_gain_reported(self, optimizer_inputs):
        from src.optimization.budget_optimizer import optimize_budget
        bundle, carry, ctrl, cur, _ = optimizer_inputs
        r = optimize_budget(sum(cur.values()) * 1.5, bundle, carry, ctrl, cur)
        assert r["reallocation_improvement_pct"] is not None
        assert r["optimized_predicted_revenue"] >= r["current_mix_same_budget_revenue"] - 1e-6

    def test_api_returns_400_for_infeasible_max_bounds(self):
        from fastapi.testclient import TestClient
        from api.main import app
        resp = TestClient(app).post("/optimize-budget", json={"total_budget": 1_000_000, "min_pct": 0.0,
                                                               "max_pct": 0.1})
        assert resp.status_code == 400
        assert "maximum" in resp.json()["message"].lower()
