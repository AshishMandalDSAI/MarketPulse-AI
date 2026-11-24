# Model validation report

This report covers the predictive models in Statistics → Predictive Analysis (PART 03) and the marketing models on pages 04–07 (PART 02). Every number here comes from running the code; the generating commands are listed in each section.

## 1. Predictive Analysis: protocol

| Step | What happens | Where |
|---|---|---|
| Target check | **Continuous number:** regression. **2–20 categories:** classification. **Integer-coded:** the user chooses. Continuous targets are never cut into classes, and the 12 classifiers are not offered for them; the page explains why. | `src/predictive/task.py`, `predictive_ui.py` |
| Cleaning | Infinite values become missing; rows with a missing target are set aside; exact duplicate rows are removed; rows with conflicting labels are kept and reported; above 20,000 rows, a seeded sample is used (stratified for classification) | `evaluate.run_supervised` |
| Holdout | **10–40%** (default 20%) of rows are set aside **before anything is fitted**. The split is random and stratified, or chronological (the last rows) | same |
| Leakage screening | Identifier-like columns, copies of the target and suspiciously strong columns are screened **inside every training fold** and again on the full training portion. Test rows are never screened | `src/predictive/leakage.py` |
| Preprocessing | Fitted on training rows only: median imputation and standard scaling for numeric columns; most-frequent imputation and one-hot encoding for categorical columns | `src/predictive/procedure.py` |
| Class imbalance | Optional (default **on** in the UI when the largest class is at least 3× the smallest). How each model handles it is in the table below | `src/predictive/classifiers.py` |
| **Model selection** | **k-fold cross-validation on the training portion only** (stratified, or time-ordered). The metric is macro-F1 (classification) or RMSE (regression). The best CV mean wins, ties going to the earlier-listed model | `evaluate.run_supervised` |
| Selected-model holdout | The selected model is refitted on the full training portion and scored once on the held-out rows, with bootstrap 95% intervals | same (unchanged) |
| **Every model's holdout (PART 03)** | Every model that passed CV is refitted on the full training portion and scored **once** on the same held-out rows, with training and prediction times. **Reporting only:** the choice is never revisited. The selected model's single evaluation is reused, not repeated | `evaluate._holdout_all` |
| Failures | A model that fails or is unavailable is shown with its status and reason. It is never hidden, and no other model is put in its place | same; `predictive_ui.py` |

### Proof that the holdout cannot influence selection

`tests/test_upgrade_part03.py::TestEvaluation::test_holdout_labels_cannot_change_the_choice` uses a chronological split, so the test rows are fixed as the last 20%. It flips every test-row label, then reruns.

- The cross-validation scores of all models and the selected model are **identical**.
- Each model's holdout accuracy becomes exactly 1 − its original value. That confirms the test rows really changed.

Two older tests in `test_phase8_part4_*` now assert that each model is scored on the test rows **exactly once** and that nothing else is.

## 2. The 12 classifiers

| Model | Implementation | Class imbalance handling | Probabilities | Built-in importance |
|---|---|---|---|---|
| Logistic Regression | scikit-learn `LogisticRegression` (lbfgs) | `class_weight='balanced'` | yes | standardized coefficients |
| Decision Tree | `DecisionTreeClassifier(max_depth=5)` | `class_weight='balanced'` | yes | impurity |
| Random Forest | `RandomForestClassifier(200 trees)` | `class_weight='balanced'` | yes | impurity |
| Support Vector Machine | `SVC(rbf)` in `CalibratedClassifierCV(sigmoid, cv=3)`, scikit-learn's replacement for the deprecated `probability=True` | `class_weight='balanced'` | yes (Platt) | none (RBF) |
| K-Nearest Neighbors | `KNeighborsClassifier(15, distance)` | none available | yes | none |
| Gradient Boosting | `GradientBoostingClassifier(150)` | none available | yes | impurity |
| Extra Trees | `ExtraTreesClassifier(300)` | `class_weight='balanced'` | yes | impurity |
| XGBoost | `xgboost.XGBClassifier` (hist, CPU), with labels encoded internally | none (no multiclass class weights) | yes | gain-based |
| LightGBM | `lightgbm.LGBMClassifier` | `class_weight='balanced'` | yes | split-based |
| CatBoost | `catboost.CatBoostClassifier`, with predictions flattened to 1-D | `auto_class_weights='Balanced'` | yes | CatBoost importance |
| Naive Bayes | `GaussianNB` | equal class priors | yes | none |
| AdaBoost | `AdaBoostClassifier(150)` | none available | yes | impurity |

**Defaults:** Logistic Regression, Decision Tree and Random Forest. They are built exactly as before PART 03 when balancing is off (a test checks this).

**Selection:** the user can pick any subset, or click **Select all available models**. Only the selected models are trained.

**Optional packages:**

- **Packages:** `xgboost-cpu` 3.4.1, `lightgbm` 4.7.0 and `catboost` 1.2.10, in `requirements-optional.txt`, separate from `requirements.txt`.
- **Detection:** they are detected at run time. A missing package makes that model *unavailable*, with an install hint, and no other model is used in its place.
- **Tested:** the missing-package path is exercised by monkeypatching, in both the engine and the page.
- **Installation:** installed and run in the development environment and in a fresh clean virtual environment (Linux, Python 3.13). The Docker image check is in `DEPLOYMENT_READINESS_REPORT.md`.

### Metrics

| | Metrics | Validity rules |
|---|---|---|
| Classification | accuracy, balanced accuracy, precision / recall / F1 (macro and weighted), ROC-AUC, log loss, confusion matrix, per-class table, train and predict seconds | **ROC-AUC:** binary uses the second sorted class as positive; multiclass uses one-vs-rest macro, only when every class is present in the test rows. **Log loss:** only when probabilities cover every training class and every test label was seen in training. Otherwise each is shown as "—" and is never imputed |
| Regression | R², MAE, MSE, RMSE, MAPE, train and predict seconds | MAPE is undefined (shown as "—") when any actual value is 0; R² is not clipped |

**Cross-checks against scikit-learn:** accuracy, F1 (macro and weighted), weighted precision, ROC-AUC and log loss are checked against `sklearn.metrics` on the real holdout predictions. Weighted metrics are also checked on a hand-made confusion matrix.

### Charts (Predictive Analysis page, all labelled; checked by tests)

The original charts stay: the CV comparison, permutation importance, and the actual vs predicted plot or confusion matrix.

New charts:

1. model performance comparison (grouped metrics);
2. accuracy;
3. precision (macro);
4. recall (macro);
5. F1 (macro);
6. ROC curves: all models for a binary target, one-vs-rest per class for multiclass;
7. confusion matrix for any model, with its per-class table;
8. training time;
9. prediction time;
10. feature importance for any model that has a built-in measure.

**Regression targets** get R², MAE, RMSE, MSE, MAPE, time and importance charts.

## 3. Validation runs (synthetic data from the PART 03 tests; seed 42)

Generated with `run_supervised(..., Settings(models=ALL_CLASSIFICATION_MODELS))` on `binary_df()`, `multi_df()` and `regression_df()` from `tests/test_upgrade_part03.py`. The data were made with a known signal, so a good model should clearly beat the baseline.

Note in the tables that the **holdout** ranking can differ from the **CV** ranking. The model is chosen by CV only, as required.

### Binary (500 rows, 2 classes)

Selected by 5-fold cross-validation (macro-F1): **Logistic Regression**. Test rows: 100.

| model | status | CV macro-F1 | CV SD | holdout accuracy | precision (macro) | recall (macro) | F1 (macro) | F1 (weighted) | ROC-AUC | log loss | train s | predict s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | ok | 0.810 | 0.041 | 0.820 | 0.819 | 0.809 | 0.812 | 0.819 | 0.894 | 0.406 | 0.013 | 0.007 |
| Decision Tree | ok | 0.732 | 0.045 | 0.770 | 0.765 | 0.759 | 0.761 | 0.769 | 0.819 | 3.056 | 0.012 | 0.006 |
| Random Forest | ok | 0.761 | 0.019 | 0.780 | 0.775 | 0.771 | 0.773 | 0.779 | 0.860 | 0.466 | 0.226 | 0.030 |
| Support Vector Machine | ok | 0.797 | 0.023 | 0.790 | 0.797 | 0.770 | 0.776 | 0.785 | 0.858 | 0.472 | 0.026 | 0.008 |
| K-Nearest Neighbors | ok | 0.781 | 0.019 | 0.760 | 0.756 | 0.747 | 0.750 | 0.758 | 0.861 | 0.454 | 0.013 | 0.011 |
| Gradient Boosting | ok | 0.747 | 0.045 | 0.820 | 0.823 | 0.805 | 0.811 | 0.818 | 0.857 | 0.526 | 0.111 | 0.011 |
| Extra Trees | ok | 0.755 | 0.012 | 0.770 | 0.765 | 0.759 | 0.761 | 0.769 | 0.862 | 0.487 | 0.210 | 0.036 |
| XGBoost | ok | 0.727 | 0.038 | 0.800 | 0.796 | 0.791 | 0.793 | 0.799 | 0.845 | 0.590 | 0.076 | 0.010 |
| LightGBM | ok | 0.748 | 0.029 | 0.790 | 0.785 | 0.783 | 0.784 | 0.790 | 0.846 | 0.627 | 0.043 | 0.012 |
| CatBoost | ok | 0.750 | 0.035 | 0.810 | 0.814 | 0.794 | 0.799 | 0.807 | 0.867 | 0.613 | 0.210 | 0.013 |
| Naive Bayes | ok | 0.787 | 0.036 | 0.830 | 0.826 | 0.824 | 0.825 | 0.830 | 0.880 | 0.428 | 0.023 | 0.012 |
| AdaBoost | ok | 0.796 | 0.030 | 0.810 | 0.810 | 0.797 | 0.801 | 0.808 | 0.875 | 0.535 | 0.211 | 0.035 |
| Baseline (mean / most frequent) | ok | 0.365 | 0.000 | 0.580 | 0.290 | 0.500 | 0.367 | 0.426 | 0.500 | 14.506 | 0.001 | 0.000 |

### Multiclass (600 rows, 3 classes)

Selected by 5-fold cross-validation (macro-F1): **Logistic Regression**. Test rows: 120.

| model | status | CV macro-F1 | CV SD | holdout accuracy | precision (macro) | recall (macro) | F1 (macro) | F1 (weighted) | ROC-AUC | log loss | train s | predict s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | ok | 0.974 | 0.013 | 0.992 | 0.992 | 0.991 | 0.991 | 0.992 | 1.000 | 0.172 | 0.017 | 0.008 |
| Decision Tree | ok | 0.871 | 0.012 | 0.892 | 0.900 | 0.891 | 0.894 | 0.893 | 0.968 | 0.616 | 0.014 | 0.007 |
| Random Forest | ok | 0.924 | 0.023 | 0.983 | 0.984 | 0.983 | 0.983 | 0.983 | 0.999 | 0.300 | 0.226 | 0.026 |
| Support Vector Machine | ok | 0.941 | 0.025 | 0.967 | 0.969 | 0.965 | 0.966 | 0.967 | 0.996 | 0.129 | 0.034 | 0.010 |
| K-Nearest Neighbors | ok | 0.850 | 0.040 | 0.883 | 0.906 | 0.881 | 0.884 | 0.886 | 0.993 | 0.343 | 0.011 | 0.008 |
| Gradient Boosting | ok | 0.920 | 0.011 | 0.933 | 0.937 | 0.933 | 0.935 | 0.934 | 0.995 | 0.105 | 0.311 | 0.009 |
| Extra Trees | ok | 0.891 | 0.020 | 0.958 | 0.959 | 0.958 | 0.958 | 0.958 | 0.996 | 0.514 | 0.253 | 0.041 |
| XGBoost | ok | 0.931 | 0.035 | 0.958 | 0.961 | 0.957 | 0.958 | 0.959 | 0.999 | 0.085 | 0.137 | 0.015 |
| LightGBM | ok | 0.940 | 0.026 | 0.958 | 0.960 | 0.958 | 0.959 | 0.959 | 0.997 | 0.136 | 0.137 | 0.017 |
| CatBoost | ok | 0.950 | 0.014 | 0.992 | 0.992 | 0.991 | 0.991 | 0.992 | 1.000 | 0.076 | 0.393 | 0.009 |
| Naive Bayes | ok | 0.816 | 0.029 | 0.858 | 0.882 | 0.859 | 0.863 | 0.862 | 0.965 | 0.486 | 0.012 | 0.007 |
| AdaBoost | ok | 0.773 | 0.075 | 0.817 | 0.875 | 0.810 | 0.813 | 0.817 | 0.914 | 1.014 | 0.240 | 0.037 |
| Baseline (mean / most frequent) | ok | 0.177 | 0.002 | 0.358 | 0.119 | 0.333 | 0.176 | 0.189 | 0.500 | 22.162 | 0.001 | 0.000 |

### Regression (300 rows, continuous target)

Selected by 5-fold cross-validation (RMSE): **Multiple Linear Regression**.

| model | status | CV RMSE | r2 | mae | mse | rmse | mape | train_seconds | predict_seconds |
|---|---|---|---|---|---|---|---|---|---|
| Simple Linear Regression | ok | 7.709 | 0.864 | 5.689 | 50.896 | 7.134 | 5.931 | 0.007 | 0.003 |
| Multiple Linear Regression | ok | 3.117 | 0.974 | 2.594 | 9.585 | 3.096 | 2.670 | 0.009 | 0.006 |
| Decision Tree | ok | 6.117 | 0.893 | 5.149 | 39.995 | 6.324 | 5.161 | 0.010 | 0.006 |
| Random Forest | ok | 4.400 | 0.959 | 3.007 | 15.461 | 3.932 | 3.050 | 0.124 | 0.014 |
| Baseline (mean / most frequent) | ok | 22.313 | -0.008 | 16.036 | 376.509 | 19.404 | 16.449 | 0.001 | 0.000 |

## 4. Marketing models (PART 02)

The driver response model (OLS with HC3 standard errors), chronological holdout quality grades, the marketing-mix model and the exposure optimizer are documented, with their validation results, in:

- `PART02_MODEL_IMPLEMENTATION_REPORT.md`;
- `DATASET_COMPATIBILITY_REPORT.md`;
- `docs/part02/compatibility_details.md`.

PART 04 additions are in section 5.

## Limitations

- **Hyperparameters:** fixed, documented defaults, with no hyperparameter search. That avoids optimistic selection, but no model is tuned.
- **Model-internal importance** is not comparable across model families, and it isn't causal.
- **SVM:** probability calibration uses 3 internal folds, so very small classes (under 3 rows per fold) make it fail. That failure is shown.
- **XGBoost class weights:** none for multiclass targets.

## 5. Budget optimizer (PART 04)

| Case | Result | Test |
|---|---|---|
| 1, 2 and 4 channels with default limits | Optimal; budget used = total; unallocated 0; utilisation for each channel | `TestEngine::test_one_two_and_multi_channel_optimal[*]` |
| Maximums below the total (no reserve) | Infeasible; no allocation shown; remedies A–D | `test_infeasible_max_no_fabricated_solution`, `TestDiagnose` |
| Minimums above the total | Infeasible; remedies A (lower minimums) and B (raise total); C and D not applicable | `test_infeasible_min`, `TestDiagnose::test_minimums_too_high` |
| Reserve with maximums too low | Optimal; budget used ≤ sum of maximums; unallocated = total − used | `test_reserve_when_maximums_are_too_low` |
| Reserve can't fix minimums that are too high | Infeasible | `test_reserve_cannot_fix_minimums_too_high` |
| Totals at 70%, 100% and 130% | Optimal at each; budget used = total | `test_changed_total_budgets[*]` |
| Profit objective | Solved within the budget range | `test_profit_objective` |
| Solver raises an exception | Feasible but not proven optimal (best feasible start kept) | `test_solver_exception_gives_feasible_not_proven_never_optimal` |
| Solver reports no convergence | Not "Optimal" | `test_non_convergence_is_not_optimal` |
| Solver returns nothing | Failed; no allocation | `test_solver_returning_nothing_is_failed` |
| No fitted model | Not supported | `test_unfitted_model_not_supported` |
| Missing spend values | Solved (rows with missing values are excluded by the MMM) | `test_missing_spend_values` |
| Non-monetary channels | No ROAS-style column | `test_no_roas_when_channels_are_not_money` |
| Manual limits / budget changes | Typed limits are kept; defaults follow the total; remedies apply only on click and are then marked "set by you" | PART 02 `TestDefaultLimits`, PART 04 `TestOptimizerPage` |

## 6. PART 05: final checks

- **Regression.** The full suite was re-run in a clean environment on the final package
  (`FINAL_TEST_REPORT.md`). All PART 02–04 model tests pass.
- **12 classifiers at scale** (synthetic binary, 9 features, all models selected): 29.3 s for
  2,000 rows and 84.9 s for 10,000 rows. At both sizes all 12 models plus the baseline were
  scored on the holdout (`PERFORMANCE_REPORT.md` §3).
- **Driver response model at 99,840 rows:** fitted in 0.36 s. The optimiser status was
  *feasible, not proven* (starts disagreed on noisy data), and it is reported as such.
- **Optional libraries in Docker.** The images install only `requirements.txt`, so XGBoost,
  LightGBM and CatBoost show as *unavailable* there. The tests cover both the available and
  the unavailable paths.
- **No model, metric or selection logic changed in PART 05.**

## 7. PART 06: real data, Z-test verification, and performance changes that leave results identical

**Real Kaggle file** (`synthetic_mmm_weekly_india_SAT.csv`, 11,232 rows; details in `DATASET_COMPATIBILITY_REPORT.md`):

- **Driver response model:** holdout R² on weekly totals is 0.91 for Sales_Value and 0.94 for Sales_Units (graded
  *good*, usable for decisions).
- **Collinearity:** VIF above 10 among the exposure, price and TPR drivers. The page states that individual effects
  are uncertain and the combined effect is reliable.
- **Negative coefficients:** YouTube (and Facebook for units) are explained as timing/targeting associations, not
  harm.
- **Classifiers:** all 12 ran on the real file (target TPR_Flag), and each got its own holdout row (12 + baseline,
  all *ok*).
- **Regression:** Sales_Value ran; Random Forest was selected by cross-validation.

**Z-test** (independent check):

- The engine was checked against hand-computed textbook values using `math.erfc`, with no scipy:
  - n = 36, x̄ = 102, μ0 = 100, σ = 6 gives z = 2.000. p = 0.0455003 two-sided, 0.0227501 for *greater*, and
    0.9772499 for *less*. The 95% CI is 102 ± 1.959964.
  - n = 50, x̄ = 48.2, μ0 = 50, σ = 4 gives z = −3.18198 (*less*), with a 99% CI of 48.2 ± 2.575829·4/√50.
- σ = None, 0, −2, NaN and ∞ are all refused; no statistic is computed.
- The sample SD is never used as σ (proved by changing σ and comparing).
- **UI:** the user enters μ0, the known σ, the alternative, α and the confidence level. The page shows the
  Z-statistic, p-value, conclusion and confidence interval, matching hand computation (`tests/test_upgrade_part06.py`
  §C). The interval is always two-sided, and the page says so.

**Performance changes** (PART 06) give **bit-identical results**. All 17 output files of *Confirm & Analyze* were
compared before and after at 10k, 50k and 100k rows; after normalising run IDs and elapsed-time text, there were
**no differences** (`PERFORMANCE_REPORT.md` §PART 06).

- Random forests: trees are built in parallel and predictions are made sequentially. That gives identical trees
  (fixed `random_state`) and identical predictions; parallel prediction alone changed only the last floating-point
  digits, so it was not used.
- Silhouette: still exact on all rows (no sampling); distance chunks are computed in parallel with a 256 MB working
  chunk.
- Spike/drop detection: positional array access in place of per-row label lookups; same arithmetic, same order.
