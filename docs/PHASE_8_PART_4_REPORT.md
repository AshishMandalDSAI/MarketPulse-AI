# Phase 8 Part 4 report - Modules 14-17

Status: implemented and delivered for independent review. **Not marked accepted.**

## Scope delivered
| Module | State |
|---|---|
| 14 Descriptive | Inspected, unchanged (already complete; Part 2 tests pass). |
| 15 Inferential | Inspected, unchanged (already complete; Part 3 tests pass). |
| 16 Predictive | New: `src/predictive/`, `dashboard/components/predictive_ui.py`, page 16 wired, registry flag set. |
| 17 Prescriptive | New: `src/prescriptive/`, `dashboard/components/prescriptive_ui.py`, page 17 wired, registry flag set. |

## Module 16 - how leakage is prevented
* P0 (before any fitting) uses only dtypes, counts and the target-free split: task detection, candidate columns, duplicate handling,
  the 20,000-row cap, the train/test split.
* P1 screening (target copies / one-to-one recodings = Tier 1a, never usable; identifier-like = Tier 1b, excluded by default, user may
  re-include; suspiciously strong = Tier 2, kept with warning plus CV-only sensitivity) is computed **per CV fold on that fold's
  training rows `T_f` only**, and for the final model on the full training portion `T` only. Evidence is partition-local and recorded
  (`Fold screening` table). No global summary decides any exclusion.
* Preprocessing (imputation, scaling, one-hot) is fitted inside each fold's `T_f`, and for the final model on `T`.
* Model choice uses cross-validated RMSE (regression) / macro-F1 (classification) on the training portion only; a baseline is always
  scored. The held-out test set is evaluated once, after the model and its settings are fixed.
* Permutation importance uses CV validation folds of the training portion, for the selected model only.
* Random and chronological validation (TimeSeriesSplit; later rows are the test set); deterministic seeds.
* Exact duplicate rows are removed; identical features with different targets are kept and reported.
* Bootstrap 95% intervals on the test set (1,000 seeded resamples, no refit), omitted below 20 test rows; ROC AUC only when valid; MAPE
  undefined when any actual is zero.
* Segmentation: K-Means with standardised numeric columns, silhouette-based k (2-8) or user k, profiles with plain descriptions, 2-D PCA view.
* Invariance tests: changing validation/test **features** changes no screen, eligibility, fitted preprocessing, fitted model parameters,
  CV scores or selection (predictions are recomputed and do differ); changing test **targets** changes no selection.

## Module 17 - how it reuses the platform
* Roles (channels, outcome, time, controls) are suggested from the platform's own saved mapping for the dataset; the user confirms or
  overrides. Nothing is guessed when no mapping exists.
* Response model: `fit_mmm` (unchanged). Objective: the optimiser's own `_predict_for_allocation` / `_get_recent_carry_in`.
  Only addition: per-channel absolute min/max bounds (the existing `optimize_budget` supports only uniform percentage bounds). With uniform
  bounds the result equals `optimize_budget` (tested: same predicted outcome to 1e-5 relative; the solver finds a flat optimum so the exact
  allocation can differ by well under 1%).
* Objectives: maximise outcome for a fixed total, or maximise `margin x outcome - spend` over a budget range (margin supplied by the user).
* Constraint validation before solving and an independent check after solving; an invalid solution is discarded, never shown.
* Explanations per channel (marginal return before/after, binding constraint, historical range), risk level with reasons, an approximate
  band from the response model's test RMSE, budget sensitivity (80-120%), a what-if panel that reuses the existing simulator, Excel / CSV /
  Markdown business-report downloads.
* Existing What-if page, Budget Optimizer page and all `src/marketing`, `src/optimization` files are untouched.

## Files changed
New: `src/predictive/*`, `src/prescriptive/*`, `dashboard/components/predictive_ui.py`, `dashboard/components/prescriptive_ui.py`,
`tests/test_phase8_part4_{engine,dashboard,prescriptive,workflow}.py`, this report.
Modified: `dashboard/pages/16_predictive_analytics.py`, `dashboard/pages/17_prescriptive_analytics.py`, `src/common/modules.py`
(two `implemented=True`), `docs/PART_STATUS.md` (one appended row), `tests/test_phase8_part2_dashboard.py` and
`tests/test_phase8_part3_dashboard.py` (approved compatibility edits: registry list now names all four modules; the placeholder test
now only requires the page to render and not show the placeholder notice).

## Testing (actual runs, this environment)
* Full suite (final audit): `python -m pytest tests -q -p no:cacheprovider` -> **2749 passed**, 0 failed, 0 skipped, 16 warnings (all from pre-existing legacy tests: an old pickled model and a numpy divide), 7 min 39 s. 2,637 are pre-existing tests (all passing, none deleted or weakened) and 112 are Part 4 tests.
* Part 4 tests: engine 35, dashboard 6, audit 20, prescriptive 49, workflow 2.
* Audit corrections: the earlier "near-equivalent" classification of the mutant *SLSQP with channel bounds ignored* was **wrong**. Against an independent closed-form (KKT) reference the mutant returns feasible but strictly suboptimal allocations (e.g. objective 34.77 vs 35.14), because clip-and-project can move away from the true optimum. Deterministic reference tests were added; the mutant is now killed by assertions. Numerical tolerances were tightened (sum-to-budget now exact to 1e-10 relative before the post-solve check, which now allows 1e-6 on the sum and 1e-8 on bounds; previously 1e-6 / 1e-4).
* Mutation check (assertion kills only), 5 optimiser mutants: solver bounds ignored, final projection removed, clip-to-bounds removed, start candidates dropped, post-solve check removed -> all 5 killed. Earlier leakage/holdout mutants (6) were killed in the previous run.
* Module 14/15 numbers were re-verified against independent scipy / numpy / pandas calculations (`tests/test_phase8_part4_audit.py`).

## Known limitations
* Leakage that is not visible in the data (a field recorded after the outcome) cannot be detected; the page says so.
* The optional "same identifiers on train and test" warning is not implemented, because it would read test features.
* Chronological validation is supervised validation on later rows, not forecasting; forecasting engines are untouched.
* Prescriptive outcomes are model estimates for one representative period (carry-over held at its recent average); the response model
  measures association, not causation. Profit optimisation needs a user-supplied margin. Requires a dataset where at least one spend
  column and a numeric outcome vary and the MMM can be fitted (at least 12 rows).
* Fixed model settings (no hyper-parameter search); 20,000-row cap for Module 16.
* Global optimality is guaranteed only for concave responses; the marketing model's saturation curves can be non-concave, so the optimiser uses three feasible starts and keeps the best (never worse than any start). Coverage percentages were not measured. Streamlit AppTest simulates sessions in one process; no live multi-browser test was run.
* Entity aggregation (many rows per customer) for segmentation is not offered as a separate mode; identifier columns can be carried into the export.

## Run locally
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
Upload a CSV or Excel file on page 00, then open Modules 14-17 from the sidebar. Tests: `python -m pytest tests -q`.
