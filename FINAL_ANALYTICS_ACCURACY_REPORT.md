# MarketPulse AI: Analytics Accuracy Report

**Release:** MarketPulse_AI_ULTIMATE_FINAL_RELEASE · **Date:** 2026-11-22 · **Python:** 3.13.18

The evidence labels used throughout are:

- **[EXECUTED]**: run in this release environment, with the output recorded.
- **[TEST]**: covered by an automated test that passes in the final suite.
- **[STATIC]**: established by reading the code.
- **[NOT VERIFIED]**: not checked.

All numbers below come from the demo dataset (synthetic, 209 weekly rows) or from synthetic test data. They describe model behaviour on that data, not accuracy on any real business.

---

## 1. Defects found and fixed

| # | Defect (before) | Effect | Fix | Evidence |
|---|---|---|---|---|
| A1 | MMM adstock/saturation parameters were chosen using **all** weeks, including the test weeks. | Test metrics were optimistic (information leaked from the test period). | Parameters are selected on training rows only, in both `src/marketing/mmm_engine.py` and `src/models/marketing_mix_model.py`. | [TEST] `TestMMMLeakage` (4 tests) |
| A2 | Missing target values were median-filled before the split, using test-period values. | Leakage, plus invented revenue. | Rows with missing revenue keep their position but are excluded from fitting and evaluation. Control medians come from training rows only. | [TEST] `test_missing_target_not_median_filled`, `test_missing_revenue_excluded_not_zero` |
| A3 | Moving-average validation used actual values for multi-step horizons. | The moving average looked better than it is (unfair advantage over the other models). | Validation is now recursive over multiple steps. | [TEST] `test_moving_average_validation_does_not_use_actuals` |
| A4 | Forecast model selection used one holdout; a complex model could win by noise. | Fragile model choice. | Rolling-origin (walk-forward) backtest over 3 folds, scored against naive and seasonal-naive baselines. A non-baseline model is used only if it beats the best baseline by at least 2% MAE. | [TEST] `test_selection_rule_baseline_first`, `test_each_fold_trains_only_on_the_past` |
| A5 | MAPE divided by zero actuals, and undefined metrics were reported as 0. | A fabricated "0% error". | `ts_metrics` skips zero actuals for MAPE and reports how many were skipped. It returns `None` ("—" in the UI) when MAPE or R² is undefined. | [TEST] `TestZeroSafeMetrics` (5 tests) |
| A6 | Uploaded row-level data (several rows per date) was forecast row by row. | Meaningless forecasts for transactional data. | Values are summed per period before forecasting, and the number of combined rows is reported. | [TEST] `test_transactional_rows_are_summed_per_period_for_forecast` |
| A7 | Prediction intervals came from in-sample residuals. | Intervals could be too narrow. | Intervals for uploaded data come from walk-forward out-of-sample errors. For the demo, the published interval is compared with out-of-sample error and the comparison is reported (see §3). | [EXECUTED] |
| A8 | **Day-first dates** (`dd/mm/yyyy`) were parsed month-first by pandas' default inference. | For 104 weekly dates, 64 became missing and 40 were silently wrong (03/01/2022 read as 1 March). | `src/data_engine/date_parsing.py` decides the day/month order once for the whole column. Ambiguous or inconsistent columns are reported in the processing log, never silently. | [TEST] `test_ultimate_dates_aliases.py` (12 tests, including an end-to-end MMM run on dd/mm data) |
| A9 | The budget optimizer accepted infeasible bounds and returned `success=True` with violations. | Silent constraint breaches. | Inputs are validated (`ValueError`, or HTTP 400 from the API). The allocation is clipped to the bounds and re-checked, and `success` is false if any violation remains. | [TEST] `TestBudgetOptimizerConstraints` (5 tests) |
| A10 | The "forecasting available" capability counted rows as time points. | A misleading reason, and a capability shown as available that the forecaster could not support. | Distinct dates are counted. | [TEST] `test_forecast_capability_counts_distinct_dates_not_rows` |

## 2. Marketing Mix Model (demo dataset) [EXECUTED]

The split is chronological with no shuffling: 166 training weeks and 42 test weeks. The artifacts were regenerated after fix A1; the demo data itself is unchanged.

| Model | Train R² | Test R² | Test adj. R² | Test MAE | Test MAPE |
|---|---|---|---|---|---|
| Baseline (linear on raw spend) | 0.539 | −0.229 | −0.800 | 349,060 | 6.11% |
| **Adstock (selected)** | 0.609 | **0.348** | 0.045 | 248,061 | 4.38% |
| Adstock + saturation | 0.590 | 0.172 | −0.212 | 290,403 | 5.07% |

Effect of removing the leakage. The previous, leaky numbers are shown first:

- adstock test R²: 0.362 → 0.348
- adstock + saturation test R²: 0.207 → 0.172
- optimizer uplift: 21.7% → 16.6%

The selected model did not change.

**Interpretation.** A test R² of 0.35 means the model explains about a third of week-to-week revenue variance on unseen weeks. The adjusted R² of 0.045 shows that most of that is used up by the number of parameters. The coefficients are statistical associations, not causal effects, and the app says so on the MMM and ROI pages.

**Hand-verified transforms [TEST]:**

- Geometric adstock reproduces a hand calculation (`[100, 0, 0, 50]` with decay 0.5 gives `[100, 50, 25, 62.5]`).
- Decay 0 is the identity.
- The Hill function equals 0.5 at its half-saturation point (x = k = 200, s = 2). At x = 400 it equals 16e4/(4e4+16e4) = 0.8. It is bounded, monotone, and clips negative input.

## 3. Forecast evaluation

### Demo (published Ridge trend + seasonal model) [EXECUTED]

The method is a walk-forward backtest: 4 folds × 12-week horizon, 48 out-of-sample points. The data range is 2022-01-03 to 2025-12-22.

| Model | MAE | RMSE | MAPE | sMAPE | WAPE | R² |
|---|---|---|---|---|---|---|
| **Ridge trend + seasonal** | 242,745 | 284,372 | 4.34% | 4.29% | 4.24% | 0.511 |
| Naive (last value) | 363,185 | 434,260 | 6.50% | 6.33% | 6.35% | −0.141 |
| Seasonal naive (same week last year) | 425,843 | 556,058 | 7.31% | 7.80% | 7.45% | −0.871 |

- The published model improves MAE by **33.2%** over the best baseline (naive).
- The 90% interval uses the in-sample residual standard deviation (336,225). The out-of-sample error standard deviation is 283,064, so the published interval is not narrower than the out-of-sample error.
- The published forecast numbers are unchanged; only the evaluation was added.

### Uploaded data (browser test file: 2,000 order rows over 30 days) [EXECUTED]

- 1,939 rows sharing a date were summed into 30 daily values. 31 rows without a valid revenue value were left out and reported.
- Candidates were scored by 3-fold walk-forward over 5-day horizons. **Moving Average** was selected.
- MAE was 7,178 and WAPE 10.1%, a 22.7% MAE improvement over the best naive baseline.
- These numbers describe random synthetic data; they show the method works, not a level of accuracy to expect.

## 4. Budget optimizer [EXECUTED + TEST]

On the demo, with the total budget fixed at ₹2,224,461.85:

- **Sum of the optimized allocation:** 2,224,461.85; the difference from the target is 4.7 × 10⁻¹⁰.
- **Bounds:** every channel lies within [5%, 55%] of the total, i.e. [111,223.09, 1,223,454.02].
- **Violations:** `constraint_violations = []` and `success = true`.
- **Revenue:** predicted revenue goes from 5,267,836 to 6,142,168 (+16.6%), compared like-for-like at the same total budget. This is a model estimate, labelled "not a guarantee".

Tests cover:

- non-negativity;
- total and bounds;
- an all-zero current allocation;
- infeasible minimum/maximum inputs (clear `ValueError` / HTTP 400);
- like-for-like gain when the budget changes.

## 5. Statistical significance vs. business importance [STATIC + TEST]

The Inferential Statistics module (page 15) reports, alongside every p-value:

- an effect size (Cohen's d, η², Cramér's V / Cohen's w, r) with Cohen's labels;
- a plain-language "practical" sentence;
- the confidence interval.

The export states that effect-size labels "describe size, not business importance". Tests refuse to run when their assumptions fail; for example, the Z-test without a known population SD is refused with an explanation. That refusal is now shown as a warning instead of a red error, because it is expected behaviour. The 71 existing Phase 8 dashboard tests for this page pass.

## 6. Dataset compatibility sweep [EXECUTED]

These are real pipeline runs (`tools/compat_sweep.py`, synthetic data). Medium- and high-confidence role suggestions were confirmed, as a user would on the Upload page; LOW-confidence (statistical) guesses were not.

| Dataset | Rows | Quality | Roles confirmed | Capabilities available | Notes |
|---|---|---|---|---|---|
| Weekly MMM, `dd/mm/yyyy` dates, `facebook_cost` | 104 | 100.0 | date, revenue, channels | 13 of 14 (all except category analysis) | Date order detected as day-first. Before fixes A8 and the alias fix, MMM, forecast and trend were unavailable. |
| E-commerce orders | 5,000 | 99.7 | date, revenue | 8 | No spend columns, so MMM/ROI/optimizer are correctly unavailable. Daily forecast. |
| Retail monthly (`Store Revenue (INR)`, `Promo Spend`) | 36 | 100.0 | date, revenue, channels | 8 | Monthly forecast; MMM needs more channels or rows. |
| Non-marketing HR table | 800 | 100.0 | none | 6 | No date and no revenue: only profiling, KPIs, categories, analyst and report. The statistical guess `tenure_years` as revenue (LOW, 0.45) is offered for confirmation, never applied silently. |
| Messy values (`N/A`, blanks, `1,234.50`, invalid date) | 300 | 93.1 | date, revenue, channels | 9 | 1 unreadable date excluded and reported. Numeric text converted. |

## 7. Not verified

- Accuracy on real (non-synthetic) business data.
- Causal validity of MMM coefficients. This would need experiments or a causal design.
- Forecast behaviour on series with structural breaks, or on very short series beyond the minimum-history checks.
