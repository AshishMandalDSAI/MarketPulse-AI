# Prediction Engine -- Technical Documentation

## Phase 4: Dataset-Agnostic Predictive Analytics

### Overview

Phase 4 adds a fully dataset-agnostic predictive analytics engine to MarketPulse AI. The engine supports four task types -- **forecasting**, **regression**, **classification**, and **segmentation** -- and works with any dataset that has passed through the Phase 2 canonical pipeline.

Every module operates without hardcoded column names. Column discovery, target selection, and feature engineering are driven entirely by the structural metadata embedded in `CanonicalDataset` and `AnalysisContext`. Confidence scores derive from actual model evaluation metrics, never from fabricated values.

---

### Architecture

```
                         +-----------------+
                         | CanonicalDataset|
                         +--------+--------+
                                  |
                         +--------v--------+
                         | AnalysisContext  |
                         +--------+--------+
                                  |
                         +--------v--------+
                         |  Task Detector   |
                         | (task_detector)  |
                         +--------+--------+
                                  |
                    +-------------+-------------+
                    |                           |
           +-------v-------+          +--------v--------+
           | Preprocessing |          | Preprocessing   |
           | (supervised)  |          | (segmentation)  |
           +-------+-------+          +--------+--------+
                   |                           |
       +-----------+-----------+               |
       |           |           |               |
  +----v----+ +----v----+ +---v-----+   +-----v------+
  |Forecast | |Regress. | |Classif. |   |Segmentation|
  | Engine  | | Engine  | | Engine  |   |  Engine    |
  +---------+ +---------+ +---------+   +------------+
       |           |           |               |
       +-----------+-----------+---------------+
                        |
               +--------v--------+
               | PredictionReport|
               +-----------------+
```

**Data flow:**

1. `CanonicalDataset` (from Phase 2) is wrapped in an `AnalysisContext`.
2. `TaskDetector` inspects the dataset's metadata (column types, cardinality, row count, time info) and returns a ranked list of feasible `PredictiveTask` objects.
3. For each task where `requirements_met == True`, the corresponding engine runs:
   - Preprocessing prepares features (imputation, encoding, scaling, leakage protection).
   - The engine trains multiple model candidates, selects the best, and produces a typed result.
4. All results are aggregated into a single `PredictionReport` and persisted as `prediction_report.json`.

---

### Module Descriptions

#### `src/prediction/contracts.py` (~355 lines)

Standardized data contracts for all prediction tasks. Every result carries provenance, metrics, limitations, and explainability. All contracts support round-trip serialization via `to_dict()` / `from_dict()`.

**Data contracts defined:**

| Contract | Purpose |
|---|---|
| `PredictiveTask` | A detected task with type, target, features, confidence, and requirements status |
| `ModelCandidate` | A trained model with name, type, metrics, score, training time, and selection flag |
| `FeatureImportance` | Single feature importance entry with feature name, importance value, and rank |
| `ForecastResult` | Forecasting output: predictions, bounds, future dates, model candidates, metrics |
| `RegressionResult` | Regression output: predictions, actuals, residuals, feature importance, train/test split info |
| `ClassificationResult` | Classification output: classes, confusion matrix, feature importance, metrics |
| `ClusterProfile` | Single cluster profile: ID, label, size, percentage, centroid, description |
| `SegmentationResult` | Clustering output: labels, cluster profiles, quality metrics, interpretation |
| `PredictionReport` | Aggregate container linking dataset ID to all task results |

**Constants:**
- `TASK_TYPES`: `{"forecasting", "regression", "classification", "segmentation"}`
- `MODEL_STATUS`: `{"success", "failed", "unavailable", "insufficient_data"}`

---

#### `src/prediction/task_detector.py` (~653 lines)

Inspects a `CanonicalDataset` (via `AnalysisContext`) and determines which predictive tasks are feasible. Returns a ranked list of `PredictiveTask` objects with confidence scores, feature lists, and human-readable explanations.

**Detection rules:**

| Task | Key Requirements | Confidence Factors |
|---|---|---|
| Forecasting | Time column present, numeric target candidate, >= 10 rows | Regularity (+0.30), observation count (+0.30), target quality (+0.25), known granularity (+0.10) |
| Regression | Numeric target candidate, numeric features present, >= 20 rows | Target quality (+0.30), feature count (+0.25), row count (+0.25), data quality (+0.15) |
| Classification | Categorical/binary target with 2-20 classes, features present, >= 20 rows | Cardinality bonus (+0.10-0.30), feature count (+0.20), row count (+0.25), class balance (+0.15), data quality (+0.10) |
| Segmentation | >= 2 numeric non-identifier features, >= 10 rows | Feature count (+0.30), row count (+0.30), data quality (+0.20), feature variance (+0.20) |

**Resource limits:**

| Constant | Value | Purpose |
|---|---|---|
| `MAX_ROWS` | 100,000 | Maximum rows before sampling |
| `MAX_FEATURES` | 200 | Maximum feature columns retained |
| `MAX_CARDINALITY_FOR_ENCODING` | 50 | Maximum unique values for one-hot encoding |
| `MIN_ROWS_FORECAST` | 10 | Minimum rows for forecasting |
| `MIN_ROWS_SUPERVISED` | 20 | Minimum rows for regression/classification |
| `MIN_ROWS_SEGMENTATION` | 10 | Minimum rows for clustering |
| `MIN_FEATURES_SEGMENTATION` | 2 | Minimum numeric features for clustering |
| `MAX_CLASSES` | 20 | Maximum class count for classification targets |

**Key helper -- `get_feature_columns()`:**

Returns valid feature columns by excluding: the target column, identifier columns, constant columns, empty columns, and the time column (for non-forecasting tasks). Segmentation mode returns only numeric columns. Supervised mode includes numeric + low-cardinality categoricals. Feature list is capped at `MAX_FEATURES`.

---

#### `src/prediction/preprocessing.py` (~451 lines)

Generic feature engineering pipeline for all ML tasks.

**Pipeline steps:**

1. **Row sampling** -- If `row_count > MAX_ROWS`, sample down with reproducible seed.
2. **Target NaN removal** -- Drop rows where the target column has missing values.
3. **Identifier/constant removal** -- Heuristic detection of ID columns (regex patterns + uniqueness ratio) and constant columns.
4. **Target leakage protection** -- Remove features with correlation > 0.95 to the target. The target column itself is never included in the feature matrix.
5. **Datetime feature extraction** -- Extract year, month, day-of-week, hour from datetime columns.
6. **Numeric imputation** -- Fill missing numeric values with column medians.
7. **One-hot encoding** -- Encode low-cardinality categoricals (up to `MAX_CARDINALITY_FOR_ENCODING` unique values).
8. **Standard scaling** -- Apply `StandardScaler` to all numeric features.

**Separate entry points:**
- `preprocess_for_prediction(df, target, features)` -- For supervised tasks (regression, classification).
- `preprocess_for_segmentation(df, features)` -- For unsupervised tasks (no target, no leakage check).

Returns: `(X, y, final_feature_names, preprocessing_report)`.

---

#### `src/prediction/forecasting_engine.py` (~648 lines)

Time-series forecasting with multiple model candidates and time-aware validation.

**Models trained:**

| Model | Description |
|---|---|
| Naive (Last Value) | Repeats the last observed value for all horizon steps |
| Seasonal Naive | Repeats the last observed seasonal cycle |
| Moving Average | Rolling mean over a configurable window |
| Linear Trend | Linear regression on the time index |
| Exponential Smoothing | Weighted moving average with decay |

**Validation strategy:**
- Time-aware train/test split (80/20 by default, `_TRAIN_FRACTION = 0.80`).
- No random shuffling -- preserves temporal order.
- Models evaluated on MAE, RMSE, MAPE, R-squared.
- Best model selected by lowest RMSE on the validation set.

**Output:**
- Point forecasts for the specified horizon.
- 95% prediction intervals (`_CONFIDENCE_Z = 1.96`).
- Future dates computed from detected frequency.
- All model candidates with individual metrics for comparison.

---

#### `src/prediction/regression_engine.py` (~387 lines)

General regression with multiple model candidates and feature importance.

**Models trained:**

| Model | Library | Key Properties |
|---|---|---|
| Ridge Regression | `sklearn.linear_model.Ridge` | L2 regularization, handles collinearity |
| Random Forest | `sklearn.ensemble.RandomForestRegressor` | Ensemble of trees, built-in feature importance |
| Gradient Boosting | `sklearn.ensemble.GradientBoostingRegressor` | Sequential boosting, strong on tabular data |

**Metrics computed:** MAE, RMSE, R-squared, MAPE.

**Model selection:** Best model chosen by highest R-squared on test set. Confidence score derived from actual R-squared (capped at 1.0).

**Feature importance:** Extracted from the best model. Ridge uses absolute coefficient values; tree-based models use `feature_importances_`.

---

#### `src/prediction/classification_engine.py` (~402 lines)

Classification with automatic target detection, multiple classifiers, and confusion matrix output.

**Target auto-detection:** Scans for categorical columns with 2-20 unique values (excluding identifiers), then binary numeric columns (0/1). Selects lowest cardinality as best candidate.

**Models trained:**

| Model | Library | Key Properties |
|---|---|---|
| Logistic Regression | `sklearn.linear_model.LogisticRegression` | Linear baseline, probability calibration |
| Random Forest | `sklearn.ensemble.RandomForestClassifier` | Ensemble, handles non-linear boundaries |
| Gradient Boosting | `sklearn.ensemble.GradientBoostingClassifier` | Sequential boosting, strong on tabular data |

**Metrics computed:** Accuracy, precision, recall, F1-score, ROC-AUC (where applicable).

**Output includes:** Full confusion matrix, per-class metrics, feature importance ranked by the best model, all model candidates with comparative metrics.

---

#### `src/prediction/segmentation_engine.py` (~457 lines)

K-Means clustering with automatic optimal-K selection and cluster profiling.

**Optimal K selection:**
- Tests K = 2 through K = min(10, n_samples/2).
- Evaluates each K using silhouette score.
- Selects K with highest silhouette score.

**Cluster profiling:**
- Computes centroid values per feature per cluster.
- Calculates cluster sizes and percentages.
- Generates human-readable descriptions comparing each cluster to overall means.

**Quality metrics:** Silhouette score, Calinski-Harabasz index, inertia.

**Libraries used:** `sklearn.cluster.KMeans`, `sklearn.metrics.silhouette_score`, `sklearn.metrics.calinski_harabasz_score`.

---

### Integration Points

#### `src/data_engine/analysis_runner.py`

Phase 4 execution block added to the main analysis pipeline. After Phases 1-3 complete, the runner:

1. Calls `detect_predictive_tasks(context)` to find feasible tasks.
2. For each task with `requirements_met == True`, dispatches to the appropriate engine.
3. Assembles results into a `PredictionReport`.
4. Serializes to `prediction_report.json` alongside existing dataset metadata.

#### `dashboard/components/data_loader.py`

New function `load_prediction_report()` reads `prediction_report.json` from the dataset's output directory and deserializes it into a `PredictionReport` object.

#### `dashboard/pages/11_predictions.py` (~612 lines)

Streamlit dashboard page displaying prediction results:
- Task feasibility overview (which tasks ran, which were skipped and why).
- Forecast charts with prediction intervals.
- Regression scatter plots (predicted vs. actual) and residual distributions.
- Classification confusion matrices and per-class metrics.
- Segmentation cluster profiles and quality metrics.

---

### Safety Guarantees

1. **No hardcoded columns** -- All column references derive from `CanonicalDataset` metadata. Forbidden patterns: no string literals matching specific marketing column names in any prediction module.

2. **Target leakage protection** -- The target column is never included in the feature matrix. Features with correlation > 0.95 to the target are automatically removed during preprocessing.

3. **ID column exclusion** -- Identifier columns (detected via regex heuristics and uniqueness ratio) are excluded from feature sets.

4. **Confidence from metrics** -- All confidence scores derive from actual model evaluation metrics (R-squared, silhouette score, validation MAE). No fabricated or assumed confidence values.

5. **Graceful degradation** -- If a task's requirements are not met, it appears in the report with `requirements_met=False` and a human-readable explanation. No task silently fails.

6. **Output size limits** -- Predictions capped at 200 rows in serialized output. Cluster labels capped at 500. Prevents JSON bloat.

---

### Resource Limits

| Limit | Value | Enforced In |
|---|---|---|
| `MAX_ROWS` | 100,000 | `task_detector.py`, all engines |
| `MAX_FEATURES` | 200 | `task_detector.py`, `preprocessing.py` |
| `MAX_CARDINALITY_FOR_ENCODING` | 50 | `task_detector.py`, `preprocessing.py` |
| `MAX_CLASSES` | 20 | `task_detector.py`, `classification_engine.py` |
| `MIN_ROWS_FORECAST` | 10 | `task_detector.py` |
| `MIN_ROWS_SUPERVISED` | 20 | `task_detector.py` |
| `MIN_ROWS_SEGMENTATION` | 10 | `task_detector.py` |
| `MIN_FEATURES_SEGMENTATION` | 2 | `task_detector.py` |

---

### Testing Overview

Phase 4 adds **160 new tests** in `tests/test_phase4_prediction.py` (~1,332 lines).

**Test categories:**
- Contract serialization round-trips (all dataclasses)
- Task detection (all four types, edge cases, insufficient data)
- Preprocessing (leakage protection, encoding, scaling, ID removal)
- Forecasting engine (model training, validation, horizon generation)
- Regression engine (model selection, metrics, feature importance)
- Classification engine (target detection, confusion matrix, multi-class)
- Segmentation engine (optimal K, cluster profiling, quality metrics)
- Integration (PredictionReport assembly, runner dispatch)
- Edge cases (empty datasets, single-column, all-null columns, constant features)

All 347 pre-Phase 4 tests remain green. Total test count: **507**.
