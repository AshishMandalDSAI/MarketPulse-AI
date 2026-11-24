# MarketPulse AI -- Phase 4 Completion Report

## Dataset-Agnostic Predictive Analytics Engine

---

## 1. Executive Summary

Phase 4 adds a fully dataset-agnostic predictive analytics engine to MarketPulse AI. The engine supports four task types -- **forecasting**, **regression**, **classification**, and **segmentation** -- and works with any dataset that has passed through the canonical data pipeline.

**Key deliverables:**
- 4 predictive task types, automatically detected from dataset metadata
- 6 new source modules in `src/prediction/`
- 1 new Streamlit dashboard page (`11_predictions.py`)
- 160 new tests (all passing)
- **507 total tests pass** (347 existing + 160 new)
- Zero hardcoded column names across all prediction modules

All existing functionality remains intact. No marketing-specific models (MMM, Adstock, ROI) were modified.

---

## 2. Scope of Changes

Phase 4 is strictly a **prediction layer addition**. It does NOT touch:

- Media Mix Modeling (MMM)
- Adstock transformation logic
- ROI/ROAS calculation pipelines
- Existing marketing analytics models
- Phase 1 (data loading), Phase 2 (canonical contracts), or Phase 3 (dynamic analytics) logic

The prediction modules sit alongside the existing `src/data_engine/` package as a new `src/prediction/` package and integrate only through the established `AnalysisContext` and `CanonicalDataset` interfaces.

---

## 3. New Files Created

| File | Lines | Purpose |
|---|---|---|
| `src/prediction/__init__.py` | 0 | Package marker |
| `src/prediction/contracts.py` | 355 | Data contracts for all prediction tasks |
| `src/prediction/task_detector.py` | 653 | Detects feasible predictive tasks from dataset metadata |
| `src/prediction/preprocessing.py` | 451 | Generic feature engineering pipeline |
| `src/prediction/forecasting_engine.py` | 648 | Time-series forecasting with multiple model candidates |
| `src/prediction/regression_engine.py` | 387 | General regression with Ridge, RF, GBR |
| `src/prediction/classification_engine.py` | 402 | Classification with LogReg, RF, GBC |
| `src/prediction/segmentation_engine.py` | 457 | K-Means clustering with optimal K selection |
| `dashboard/pages/11_predictions.py` | 612 | Streamlit prediction results dashboard page |
| `tests/test_phase4_prediction.py` | 1,332 | 160 tests for all Phase 4 modules |
| `docs/PREDICTION_ENGINE.md` | ~250 | Technical documentation for the prediction engine |

**Total new code: ~5,547 lines** (including tests and documentation).

---

## 4. Files Modified

| File | Change Description |
|---|---|
| `src/data_engine/analysis_runner.py` | Added Phase 4 execution block: detects tasks, dispatches to engines, assembles PredictionReport, saves `prediction_report.json` |
| `dashboard/components/data_loader.py` | Added `load_prediction_report()` function to deserialize `prediction_report.json` |
| `tests/helpers.py` | Added `prediction_report.json` to the allowed metadata files list |

---

## 5. Prediction Contracts Design

All contracts are defined in `src/prediction/contracts.py` as Python `@dataclass` classes with `to_dict()` / `from_dict()` for round-trip JSON serialization.

### PredictiveTask

```
task_type: str           # forecasting | regression | classification | segmentation
target_column: str?      # None for segmentation
feature_columns: list    # discovered feature column names
confidence: float        # 0.0-1.0, from actual detection metrics
reason: str              # human-readable explanation
requirements_met: bool   # whether the task can actually run
limitations: list        # known limitations for this task
```

### ModelCandidate

```
name: str                     # display name
model_type: str               # e.g. "Ridge", "RandomForest"
metrics: dict                 # evaluation metrics
score: float                  # primary evaluation metric value
training_time_seconds: float  # wall-clock training duration
selected: bool                # True for the winning model
```

### FeatureImportance

```
feature: str       # feature column name
importance: float  # importance value
rank: int          # importance rank (1 = most important)
```

### ForecastResult

```
id: str                  # unique ID (prefix: fcast_)
status: str              # success | failed | unavailable | insufficient_data
target_column: str       # the column being forecasted
time_column: str         # the datetime column used
horizon: int             # number of future periods
frequency: str           # daily | weekly | monthly | yearly | irregular
model_name: str          # name of selected model
predictions: list        # point forecasts
lower_bound: list        # lower prediction interval bound
upper_bound: list        # upper prediction interval bound
future_dates: list       # ISO date strings for each forecast step
metrics: dict            # MAE, RMSE, MAPE, R2
confidence: float?       # from actual model metrics
model_candidates: list   # all trained ModelCandidate objects
feature_importance: list # FeatureImportance entries
limitations: list        # known limitations
explanation: str         # human-readable summary
```

### RegressionResult

```
id: str                  # unique ID (prefix: reg_)
status: str              # success | failed | unavailable | insufficient_data
target_column: str       # predicted column
feature_columns: list    # feature names used
model_name: str          # selected model name
metrics: dict            # MAE, RMSE, R2, MAPE
predictions: list        # predicted values (capped at 200)
actuals: list            # actual values (capped at 200)
residuals: list          # residuals (capped at 200)
feature_importance: list # ranked FeatureImportance
model_candidates: list   # all ModelCandidate objects
confidence: float?       # from R-squared
limitations: list        # known limitations
explanation: str         # human-readable summary
n_train: int             # training set size
n_test: int              # test set size
```

### ClassificationResult

```
id: str                  # unique ID (prefix: clf_)
status: str              # success | failed | unavailable | insufficient_data
target_column: str       # predicted column
classes: list            # class labels
feature_columns: list    # feature names used
model_name: str          # selected model name
metrics: dict            # accuracy, precision, recall, f1, roc_auc
confusion_matrix: list   # 2D list
feature_importance: list # ranked FeatureImportance
model_candidates: list   # all ModelCandidate objects
confidence: float?       # from accuracy/F1
limitations: list        # known limitations
explanation: str         # human-readable summary
n_train: int             # training set size
n_test: int              # test set size
```

### ClusterProfile

```
cluster_id: int     # cluster index
label: str          # auto-generated label
size: int           # number of rows in cluster
percentage: float   # percentage of total rows
centroid: dict      # feature -> mean value
description: str    # human-readable profile
```

### SegmentationResult

```
id: str                  # unique ID (prefix: seg_)
status: str              # success | failed | unavailable | insufficient_data
cluster_count: int       # number of clusters (K)
labels: list             # cluster assignment per row (capped at 500)
feature_columns: list    # feature names used
cluster_profiles: list   # ClusterProfile objects
quality_metrics: dict    # silhouette, calinski_harabasz, inertia
interpretation: str      # overall interpretation
limitations: list        # known limitations
explanation: str         # human-readable summary
```

### PredictionReport

```
dataset_id: str               # links to the canonical dataset
tasks: list[PredictiveTask]   # all detected tasks (met and unmet)
forecast: ForecastResult?     # None if not feasible
regression: RegressionResult? # None if not feasible
classification: ClassificationResult?
segmentation: SegmentationResult?
```

---

## 6. Task Detection Design

### How Tasks Are Detected

The `detect_predictive_tasks(context)` function inspects the `CanonicalDataset` wrapped by an `AnalysisContext` and produces one `PredictiveTask` per task type. Each task is returned regardless of whether requirements are met, so the dashboard can explain why a task is unavailable.

### Detection Rules

**Forecasting:**
1. Check for a time column in `ds.time_info`.
2. Check for at least one numeric target candidate in `ds.target_candidates`.
3. Verify `row_count >= 10`.
4. Confidence from: regularity (+0.30), observation count ratio (+0.30), target quality score (+0.25), known granularity (+0.10).

**Regression:**
1. Check for numeric target candidates.
2. Check for at least one numeric feature column (via `get_feature_columns()`).
3. Verify `row_count >= 20`.
4. Confidence from: target quality (+0.30), feature count ratio (+0.25), row count ratio (+0.25), data quality score (+0.15).

**Classification:**
1. Scan categorical columns for 2-20 unique values (excluding identifiers).
2. Scan binary numeric columns (only 0/1 values).
3. Scan boolean columns.
4. Pick best candidate (prefer lower cardinality, prefer binary_numeric source).
5. Verify features available and `row_count >= 20`.
6. Confidence from: cardinality bonus (+0.10 to +0.30), feature count (+0.20), row count (+0.25), class balance (+0.15), data quality (+0.10).

**Segmentation:**
1. Count numeric non-identifier features via `get_feature_columns()` in segmentation mode.
2. Verify `>= 2` numeric features and `row_count >= 10`.
3. Confidence from: feature count ratio (+0.30), row count ratio (+0.30), data quality (+0.20), feature variance ratio (+0.20).

### Resource Limits Enforced at Detection

| Constant | Value |
|---|---|
| `MAX_ROWS` | 100,000 |
| `MAX_FEATURES` | 200 |
| `MAX_CARDINALITY_FOR_ENCODING` | 50 |
| `MIN_ROWS_FORECAST` | 10 |
| `MIN_ROWS_SUPERVISED` | 20 |
| `MIN_ROWS_SEGMENTATION` | 10 |
| `MIN_FEATURES_SEGMENTATION` | 2 |
| `MAX_CLASSES` | 20 |

---

## 7. Preprocessing Pipeline Design

### Steps (in order)

1. **Row sampling** -- Datasets exceeding `MAX_ROWS` (100,000) are randomly sampled with a fixed seed for reproducibility.
2. **Target NaN removal** -- Rows with missing target values are dropped (supervised tasks only).
3. **Identifier column removal** -- Columns matching ID patterns (regex: `_id$`, `^id$`, `^uuid`, etc.) or with uniqueness ratio > 0.95 are removed from features.
4. **Constant column removal** -- Columns with zero variance are dropped.
5. **Target leakage protection** -- Features with Pearson correlation > 0.95 to the target are removed (supervised tasks only).
6. **Datetime feature extraction** -- Datetime columns are expanded into year, month, day-of-week, hour components and the original column is dropped.
7. **Numeric imputation** -- Missing numeric values filled with column median.
8. **One-hot encoding** -- Categorical features with <= `MAX_CARDINALITY_FOR_ENCODING` unique values are one-hot encoded; others are dropped.
9. **Standard scaling** -- `sklearn.preprocessing.StandardScaler` applied to all numeric features.

### Target Leakage Protection

- The target column is **never** included in the feature matrix `X`.
- After feature assembly, Pearson correlation between each feature and the target is computed.
- Features with |correlation| > 0.95 are removed and logged.
- This prevents features that are trivially derived from the target (e.g., `revenue_per_unit` when predicting `revenue`) from inflating model accuracy.

### Two Entry Points

| Function | Use Case |
|---|---|
| `preprocess_for_prediction(df, target, features)` | Supervised tasks (regression, classification) |
| `preprocess_for_segmentation(df, features)` | Unsupervised tasks (no target column, no leakage check) |

---

## 8. Forecasting Engine Design

### Models Trained

| Model | Description | When It Wins |
|---|---|---|
| Naive (Last Value) | Repeats the last observed value | Very noisy or random-walk series |
| Seasonal Naive | Repeats the last complete seasonal cycle | Strong seasonality, stable level |
| Moving Average | Rolling mean over a configurable window | Smooth, slowly trending series |
| Linear Trend | Linear regression on the integer time index | Clear upward or downward trend |
| Exponential Smoothing | Weighted average with exponential decay | Combination of trend and noise |

### Validation Strategy

- **Time-aware split:** First 80% for training, last 20% for validation. No random shuffling.
- **Metrics on validation set:** MAE, RMSE, MAPE, R-squared.
- **Model selection:** Lowest RMSE on the validation set wins.
- All model candidates are recorded with their individual metrics for transparency.

### Prediction Intervals

- 95% prediction intervals computed using `z = 1.96` times the standard deviation of validation residuals.
- Intervals widen for longer horizons (linearly scaled by sqrt of step).

### Output

- Point forecasts for the requested horizon.
- Lower and upper bounds for each forecast step.
- Future dates computed from the detected time frequency.
- All `ModelCandidate` objects with comparative metrics.

---

## 9. Regression Engine Design

### Models Trained

| Model | Class | Key Properties |
|---|---|---|
| Ridge Regression | `sklearn.linear_model.Ridge` | L2 regularization, stable with collinear features |
| Random Forest Regressor | `sklearn.ensemble.RandomForestRegressor` | Ensemble of decision trees, built-in feature importance |
| Gradient Boosting Regressor | `sklearn.ensemble.GradientBoostingRegressor` | Sequential boosting, strong on tabular data |

### Metrics Computed

- **MAE** -- Mean Absolute Error
- **RMSE** -- Root Mean Squared Error
- **R-squared** -- Coefficient of determination
- **MAPE** -- Mean Absolute Percentage Error (safe against zero actuals)

### Model Selection

Best model chosen by **highest R-squared** on the test set. Confidence score is derived directly from the winning model's R-squared value.

### Feature Importance

- Ridge: Absolute coefficient values, normalized.
- Random Forest / Gradient Boosting: `feature_importances_` attribute.
- Top features ranked and returned as `FeatureImportance` objects.

---

## 10. Classification Engine Design

### Target Detection

Automatic target column discovery:
1. Scan categorical columns for 2-20 unique values, excluding identifier columns.
2. Scan binary numeric columns (values in {0, 1}).
3. Scan boolean columns.
4. Select candidate with lowest cardinality (binary preferred).

### Models Trained

| Model | Class | Key Properties |
|---|---|---|
| Logistic Regression | `sklearn.linear_model.LogisticRegression` | Linear baseline, probability outputs |
| Random Forest Classifier | `sklearn.ensemble.RandomForestClassifier` | Ensemble, handles non-linear boundaries |
| Gradient Boosting Classifier | `sklearn.ensemble.GradientBoostingClassifier` | Sequential boosting, strong performance |

### Metrics Computed

- **Accuracy** -- Overall correct predictions / total
- **Precision** -- Per-class and weighted average
- **Recall** -- Per-class and weighted average
- **F1-score** -- Harmonic mean of precision and recall
- **ROC-AUC** -- Area under the ROC curve (binary classification)

### Confusion Matrix

Full confusion matrix (2D list, `classes x classes`) is included in the result for detailed error analysis.

---

## 11. Segmentation Engine Design

### Optimal K Selection

1. Test K values from 2 to min(10, n_samples / 2).
2. Fit K-Means for each K.
3. Compute silhouette score for each K.
4. Select K with **highest silhouette score**.

### Cluster Profiling

For each cluster:
- **Size** and **percentage** of total rows.
- **Centroid** -- Mean value of each feature within the cluster.
- **Description** -- Auto-generated text comparing cluster centroid to overall mean (e.g., "Higher than average spend, lower than average frequency").

### Quality Metrics

| Metric | Source | Interpretation |
|---|---|---|
| Silhouette Score | `sklearn.metrics.silhouette_score` | -1 to 1; higher = better separation |
| Calinski-Harabasz Index | `sklearn.metrics.calinski_harabasz_score` | Higher = denser, well-separated clusters |
| Inertia | KMeans attribute | Sum of squared distances to centroids |

---

## 12. Dashboard Integration

### Prediction Page (`dashboard/pages/11_predictions.py`)

Page 11 in the Streamlit dashboard, accessible via the sidebar navigation.

**Layout sections:**

1. **Header** -- Page title and dataset context.
2. **Task Overview** -- Card showing which tasks are feasible and which are not, with reasons.
3. **Forecasting Section** -- Time-series chart with predictions and confidence intervals (if forecasting ran).
4. **Regression Section** -- Predicted vs. actual scatter plot, residual histogram, feature importance bar chart, model comparison table.
5. **Classification Section** -- Confusion matrix heatmap, per-class metrics table, feature importance, model comparison.
6. **Segmentation Section** -- Cluster size bar chart, cluster profiles table, quality metrics display.

### Routing

The page reads `prediction_report.json` via `load_prediction_report()` from `data_loader.py`. If no prediction report exists (Phase 4 has not run), the page displays an informational message.

---

## 13. Runner Integration

### Phase 4 Block in `analysis_runner.py`

After Phases 1-3 complete successfully, the runner executes Phase 4:

```
Phase 4 block (pseudocode):
1. from src.prediction.task_detector import detect_predictive_tasks
2. tasks = detect_predictive_tasks(context)
3. report = PredictionReport(dataset_id=context.dataset.id, tasks=tasks)
4. For each task where requirements_met == True:
   a. forecasting  -> run_forecast(context) -> report.forecast
   b. regression   -> run_regression(context) -> report.regression
   c. classification -> run_classification(context) -> report.classification
   d. segmentation -> run_segmentation(context) -> report.segmentation
5. Serialize report.to_dict() -> prediction_report.json
```

Each engine call is wrapped in a try/except. If an engine fails, its result is set to a failed status object with the error message -- it does not halt other engines or the pipeline.

---

## 14. No-Hardcoded-Columns Guarantee

### Forbidden Patterns

The following patterns are **never** present in any `src/prediction/` module:

- No string literals like `"revenue"`, `"spend"`, `"clicks"`, `"impressions"`, `"ctr"`, `"roas"`, `"channel"`, `"campaign"`, etc.
- No `df["specific_column_name"]` with marketing-specific names.
- No assumptions about column count, order, or naming conventions.

### How Column Discovery Works

1. **Numeric columns:** Read from `ds.numeric_columns` (set by Phase 2 canonical pipeline).
2. **Categorical columns:** Read from `ds.categorical_columns`.
3. **Boolean columns:** Read from `ds.boolean_columns`.
4. **Identifier columns:** Read from `ds.identifier_columns`.
5. **Time column:** Read from `ds.time_info.time_column`.
6. **Target candidates:** Read from `ds.target_candidates` (ranked list from Phase 2).
7. **Empty/constant columns:** Read from `ds.empty_columns` and `ds.data_quality.constant_columns`.

Every column reference in the prediction modules derives from these metadata attributes. The modules will work identically on a marketing dataset, a healthcare dataset, a financial dataset, or any other tabular data.

---

## 15. Dataset Isolation

### Per-Dataset Output

Each dataset's prediction results are stored in their own `prediction_report.json` file within the dataset's output directory. Prediction results from one dataset never leak into another.

### Cache Clearing

When a new dataset is loaded or re-analyzed:
- The previous `prediction_report.json` is overwritten with the new results.
- In-memory caches (if any) are cleared when the `AnalysisContext` is rebuilt.
- The dashboard reads fresh data on each page load.

---

## 16. Target Leakage Protection

### Principle

The target column must never appear in the feature matrix. Features that are trivially derived from the target (near-perfect correlation) must also be excluded.

### Implementation

1. **Explicit exclusion:** `get_feature_columns()` always removes the target column from the feature list.
2. **High-correlation removal:** `preprocess_for_prediction()` computes Pearson correlation between each numeric feature and the target. Features with |r| > 0.95 are dropped.
3. **Logging:** Removed features are logged with their correlation values for auditability.
4. **Segmentation exempt:** `preprocess_for_segmentation()` has no target column, so no leakage check applies.

---

## 17. Edge Case Handling

| Edge Case | Behavior |
|---|---|
| Empty DataFrame | Task detection returns `requirements_met=False` with clear reason |
| Single column dataset | No features available; all supervised tasks report "no feature columns" |
| All-null target column | Target NaN removal drops all rows; engine returns `insufficient_data` status |
| Constant features | Removed during preprocessing; if all features are constant, engine returns failed status |
| Single unique class | Classification target rejected (needs 2-20 classes) |
| Too few rows (< minimum) | Task detection sets `requirements_met=False` with row count details |
| Very high cardinality categoricals | Excluded from features (> `MAX_CARDINALITY_FOR_ENCODING`) |
| No time column | Forecasting returns `requirements_met=False`; other tasks unaffected |
| Irregular time intervals | Forecasting still attempted with limitation noted; resampling may be applied |
| More rows than `MAX_ROWS` | Data is sampled down with a fixed seed before training |
| No numeric columns | Regression/forecasting infeasible; segmentation requires >= 2 |
| Binary numeric columns (0/1) | Detected as classification candidates |
| Mixed type columns | Handled by Phase 2 canonical pipeline before reaching prediction |

---

## 18. Model Safety / Resource Limits

### Training Safeguards

| Limit | Value | Purpose |
|---|---|---|
| `MAX_ROWS` | 100,000 | Prevents OOM on large datasets; sampling with fixed seed |
| `MAX_FEATURES` | 200 | Caps dimensionality; features beyond this are dropped |
| `MAX_CARDINALITY_FOR_ENCODING` | 50 | Prevents one-hot explosion on high-cardinality categoricals |
| `MAX_CLASSES` | 20 | Caps classification complexity |
| Prediction output cap | 200 rows | Keeps serialized JSON manageable |
| Cluster label cap | 500 rows | Keeps segmentation output bounded |
| K-range for clustering | 2 to min(10, n/2) | Prevents degenerate clusterings |

### No External Model Downloads

All models are from scikit-learn and require no network access, no GPU, and no external model downloads. Training runs entirely in-process on CPU.

---

## 19. Test Coverage

### Test Statistics

| Metric | Value |
|---|---|
| New tests added | 160 |
| Test file | `tests/test_phase4_prediction.py` |
| Test file lines | ~1,332 |
| Pre-Phase 4 tests | 347 |
| **Total tests** | **507** |
| All passing | Yes |

### Test Categories

| Category | Coverage |
|---|---|
| Contract serialization | `to_dict()` / `from_dict()` round-trips for all 9 dataclasses |
| Task detection | All four task types, edge cases (no time column, insufficient rows, no target, etc.) |
| `get_feature_columns()` | Target exclusion, ID exclusion, constant exclusion, segmentation mode |
| Preprocessing | Leakage removal, one-hot encoding, scaling, NaN handling, ID heuristics |
| Forecasting engine | Model training, validation split, horizon generation, prediction intervals |
| Regression engine | Model selection, metric computation, feature importance extraction |
| Classification engine | Target auto-detection, confusion matrix, multi-class handling |
| Segmentation engine | Optimal K selection, cluster profiling, quality metrics |
| PredictionReport | Assembly, serialization, `available_tasks()` method |
| Edge cases | Empty data, single-column, all-null, constant features, tiny datasets |
| Resource limits | Row sampling, feature capping, cardinality limits |

---

## 20. Existing Test Preservation

All **347 pre-Phase 4 tests remain green**. Phase 4 changes are purely additive:

- No existing test files were modified (except `tests/helpers.py` for the metadata allowlist).
- No existing module behavior was changed.
- No existing imports were broken.
- The `analysis_runner.py` Phase 4 block executes only after Phases 1-3 succeed and does not alter their outputs.

---

## 21. Performance Considerations

### Caching

- Prediction results are persisted to `prediction_report.json` so the dashboard does not re-run engines on every page load.
- The dashboard's `load_prediction_report()` reads from disk once per session.

### Output Limits

- Regression predictions, actuals, and residuals capped at 200 rows in serialized output.
- Segmentation labels capped at 500 rows.
- Prevents multi-megabyte JSON files for large datasets.

### Sampling

- Datasets exceeding `MAX_ROWS` (100,000) are sampled with a fixed random seed before training.
- Ensures reproducible results while keeping training time bounded.

### Training Time

- All models are scikit-learn estimators with default or conservative hyperparameters.
- No hyperparameter search (grid search / random search) -- keeps training fast.
- Training time per model is recorded in `ModelCandidate.training_time_seconds` for monitoring.

---

## 22. Backward Compatibility

### Existing Features Unchanged

- All Phase 1, 2, and 3 functionality operates identically.
- No existing API signatures were modified.
- No existing data contracts were altered.

### New Keys Are Additive

- `prediction_report.json` is a new file alongside existing metadata files.
- The `PredictionReport` key in the analysis output is new; existing keys are untouched.
- Dashboard page 11 is a new page; existing pages 1-10 are unchanged.

### Import Safety

- `src/prediction/` is a new package with its own `__init__.py`.
- It imports from `src/data_engine/canonical` (read-only) and `sklearn` (external dependency already in requirements).
- No circular imports. No monkey-patching.

---

## 23. Known Limitations

1. **No hyperparameter tuning** -- Models use default or conservative parameters. Cross-validated grid search could improve accuracy but would increase training time.

2. **Forecasting models are simple** -- No ARIMA, Prophet, or deep learning models. Suitable for trend detection and baseline forecasts; not for production-grade time-series forecasting.

3. **Single target per task** -- Each task type selects one target column (the best candidate). Multi-target regression/classification is not supported.

4. **No ensemble stacking** -- Model candidates are compared individually. Stacking or blending could improve accuracy.

5. **Classification assumes balanced evaluation** -- Weighted metrics are used, but no resampling (SMOTE, oversampling) is applied to imbalanced classes.

6. **Segmentation is K-Means only** -- DBSCAN, hierarchical clustering, or Gaussian Mixture Models could handle non-spherical clusters better.

7. **No feature selection beyond correlation** -- Mutual information, recursive feature elimination, or L1-based selection could improve feature sets.

8. **Prediction intervals are Gaussian** -- Forecasting assumes normally distributed residuals for confidence intervals.

---

## 24. Phase 4 Completion Checklist

- [x] Prediction contracts defined (`contracts.py`)
- [x] Task detector implemented (`task_detector.py`)
- [x] Preprocessing pipeline implemented (`preprocessing.py`)
- [x] Forecasting engine implemented (`forecasting_engine.py`)
- [x] Regression engine implemented (`regression_engine.py`)
- [x] Classification engine implemented (`classification_engine.py`)
- [x] Segmentation engine implemented (`segmentation_engine.py`)
- [x] Dashboard page 11 created (`11_predictions.py`)
- [x] Runner integration added to `analysis_runner.py`
- [x] Data loader updated for prediction report
- [x] Test helpers updated for new metadata file
- [x] 160 new tests written and passing
- [x] All 347 existing tests still pass
- [x] 507 total tests pass
- [x] No hardcoded column names in any prediction module
- [x] Target leakage protection implemented
- [x] Resource limits enforced (MAX_ROWS, MAX_FEATURES, etc.)
- [x] Edge cases handled gracefully
- [x] Technical documentation written (`docs/PREDICTION_ENGINE.md`)
- [x] Backward compatibility verified

---

**Phase 4 is COMPLETE. Phase 5 has NOT been started.**
