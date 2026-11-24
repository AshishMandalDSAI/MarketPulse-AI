# Phase 8 – Part 6: Final Integration, Testing and Release Preparation

Status: release candidate, **self-verified by the implementer only**. It has not been independently accepted.

## 1. What Phase 8 adds
Four dataset-driven analytics pages on top of the existing MarketPulse AI platform (Modules 00–13 are unchanged in behaviour):

| Module | Page | Purpose |
|---|---|---|
| 14 | Descriptive Statistics | Dataset profile, per-column statistics, missing values and duplicates, distributions/charts, Excel/CSV export |
| 15 | Inferential Statistics | Hypothesis tests, confidence intervals, p-values, assumption checks, plain-language interpretation, export |
| 16 | Predictive Analytics | Regression, classification and K-Means segmentation; cross-validated model comparison; leakage protection; predictions export |
| 17 | Prescriptive Analytics | Budget optimisation under per-channel and total constraints; profit objective; what-if; explainable recommendations; export |

## 2. Architecture and workflow
Descriptive → Inferential → Predictive → Prescriptive all read the **same active dataset** held in session state (`mp_active_dataset_id`). Each page is rendered through `render_module_page(SPEC, body_fn)`; results and widget keys are scoped by dataset id, so switching datasets never shows stale results. Demo data is used only when the user explicitly chooses it.

* Module 16 follows a staged protocol: target-independent preparation before the split; per-fold, training-only feature screening; cross-validated selection (RMSE for regression, macro-F1 for classification); one single evaluation on the held-out set. Leakage tiers: 1a (target copy / one-to-one recode, never usable), 1b (identifier-like, excluded by default), 2 (suspiciously strong, kept with a warning and a CV-only sensitivity check).
* Module 17 reuses the platform MMM (`fit_mmm`) and simulation code, adds per-channel absolute min/max bounds, solves with SLSQP from several feasible starts, projects onto the constraints and independently re-checks the result. Infeasible constraints produce a refusal instead of a recommendation.
* Exports neutralise spreadsheet-formula prefixes in cell values (`src/common/export.py`) and in column headers / index labels (`src/common/headers.py`).

## 3. Supported data
CSV (`.csv`) and Excel (`.xlsx`) uploads, up to the limits in `config/settings.py` (`MARKETPULSE_MAX_ROWS`, default 100000; `MARKETPULSE_MAX_UPLOAD_MB`, default 50). Column names are free-form; roles are mapped dynamically.

Requirements per module:
* 14 / 15: at least one column with usable values (15 additionally needs enough numeric/categorical observations for the chosen test; otherwise it refuses with a reason).
* 16: a target with an eligible type and enough labelled rows; segmentation needs at least two usable numeric features.
* 17: a numeric outcome and at least one numeric spend column, ideally a date column; enough rows for the MMM fit. Infinite values in channels or outcome are blocked with a message.

Empty, one-row, one-column, all-text, all-missing and constant datasets are handled with readable messages, not exceptions.

## 4. Installation, run, test
```
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
python -m pytest -q          # full suite
```
Docker: `docker compose up` (see `Dockerfile`, `docker-compose.yml`). `ANTHROPIC_API_KEY` is optional and read only from the environment.
Tested here with Python 3.13 and Streamlit 1.67.0.

## 5. Part 6 corrections (regression tests: `tests/test_phase8_part6_integration.py`)
1. Module 16 raised an error when a numeric feature contained ±infinity. Infinite feature values are now treated as missing before the split, counted in `data_prep` and reported as a warning.
2. Column headers beginning with `= + - @` could survive as formulas in the segmentation Assignments export and the prescriptive Scenarios sheet. All Module 16/17 exports and on-screen download tables now neutralise headers and index labels. Modules 14/15 were checked and were already safe.
3. Module 17 now validates infinite values in channels and outcome up front with a clear message instead of failing inside the model fit.

4. (Corrective review) In profit mode `PrescriptiveResult.total_budget` reported the *current* spend instead of the recommended plan's total, which also fed the report, the Summary sheet, `avg_return_per_unit_recommended` and the what-if hint. It now equals the recommended total (fixed-budget mode is unchanged). Tests: `tests/test_phase8_part6_module17_total_budget.py`.

**Dependency change:** `pytest` was constrained to `>=7.4.0,<9.2` in `requirements.txt`. Reason: pytest 9.2.0 rejects `pytest.approx` on nested data structures used in existing tests (specifically `tests/test_phase8_part2_engine.py`). The upper bound `<9.2` ensures the test suite runs without modification on current pytest releases (tested with 9.1.1).

## 6. Known limitations
* `use_container_width` is deprecated in current Streamlit and emits warnings project-wide (legacy pages included); it still works and was not changed.
* Loading the bundled MMM pickle can emit a scikit-learn `InconsistentVersionWarning` if the installed scikit-learn differs from the version that saved it; behaviour is unaffected.
* Module 16 caps training at 20,000 rows (a documented, user-visible cap); model families are classical (linear, tree-based); no hyperparameter search beyond the documented defaults.
* Module 17 optimises the fitted MMM response; recommendations are only as good as that model. Allocations outside the observed spend range are flagged as extrapolation / high risk. It cannot guarantee real-world uplift.
* Module 15 assumption checks are screening aids, not proofs; independence of observations cannot be verified from data alone.
* Large Excel exports are limited by Excel's row limit; use the CSV download for bigger tables.
* Verified by automated tests and AppTest simulation only. No real-browser visual regression, load, or multi-user concurrency testing was performed. No deployment was made.

## 7. Troubleshooting
* *"Upload a dataset"* prompt: no active dataset; upload a file or explicitly choose the demo dataset.
* *Refusal messages*: they name the cause (too few rows, constant target, infeasible budget bounds); adjust the selection and rerun.
* *Excel export refused*: sheet exceeds Excel's row limit; use CSV.
* *Slow first run*: first MMM/CV fit is computed once per dataset and cached per dataset id.
