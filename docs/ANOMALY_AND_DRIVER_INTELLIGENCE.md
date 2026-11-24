# Anomaly Detection & Driver Intelligence -- Technical Documentation

## Phase 5: Dataset-Agnostic Anomaly Detection and Driver/Relationship Intelligence

### Overview

Phase 5 adds anomaly detection and driver/root-cause analysis to MarketPulse AI. The engine detects numeric outliers, time-series anomalies, and categorical anomalies across any dataset that has passed through the Phase 2 canonical pipeline, then identifies which features are most strongly associated with a target variable using multiple statistical and ML methods.

Every module operates without hardcoded column names. Column discovery, anomaly thresholds, and driver ranking are driven entirely by the structural metadata in `CanonicalDataset` and `AnalysisContext`. All explanations use associative, non-causal language.

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
              +-------------------+-------------------+
              |                                       |
     +--------v--------+                    +---------v---------+
     | Anomaly Detector |                    | Driver Engine     |
     | (detector.py)    |                    | (driver_engine.py)|
     +--------+---------+                    +---------+---------+
              |                                        |
    +---------+---------+              +---------------+----------------+
    |         |         |              |         |           |          |
+---v---+ +--v---+ +---v----+   +-----v----+ +-v------+ +--v-------+ +v----------+
|Numeric| |Time  | |Categor.|   |Correlat. | |Mutual  | |Gradient  | |Group      |
|Detect | |Series| |Detect  |   |Ranking   | |Info    | |Boosting  | |Comparison |
+-------+ +------+ +--------+   +----------+ +--------+ +----------+ +-----------+
    |         |         |              |         |           |          |
    +---------+---------+              +---------+-----------+----------+
              |                                  |
     +--------v--------+               +--------v--------+
     | AnomalyReport   |               | DriverReport    |
     +--------+--------+               +--------+--------+
              |                                  |
              +----------------+-----------------+
                               |
                      +--------v--------+
                      | Phase5Result    |
                      +-----------------+
```

**Data flow:**

1. `CanonicalDataset` (Phase 2) is wrapped in an `AnalysisContext`.
2. The **Anomaly Detector** orchestrator calls three independent sub-detectors (numeric, time-series, categorical), merges results by severity, and produces an `AnomalyReport`.
3. The **Driver Engine** auto-detects (or accepts) a target column, runs four analysis methods, merges and ranks results, and produces a `DriverReport`. It extends (not duplicates) the Phase 3 `relationship_engine`.
4. Both reports are aggregated into a `Phase5Result` and persisted as `phase5_analysis.json`.

---

### Package Structure

```
src/anomaly/
    __init__.py           # Package init
    contracts.py          # Data contracts (enums, dataclasses)
    numeric.py            # IQR, Z-score, MAD outlier detection
    time_series.py        # Rolling deviation, spike/drop, trend residual
    categorical.py        # Rare categories, unusual metrics, missingness
    detector.py           # Orchestrator merging all sub-detectors
    driver_engine.py      # Target-aware driver/relationship analysis
```

---

### Contracts (`contracts.py`)

All contracts are JSON-serializable via `to_dict()` and reconstructable via `from_dict()`.

#### Enums

| Enum | Values |
|------|--------|
| `AnomalySeverity` | `LOW`, `MEDIUM`, `HIGH`, `CRITICAL` |
| `AnomalyType` | `numeric_outlier`, `time_series_anomaly`, `sudden_change`, `distribution_shift`, `category_anomaly`, `missingness_anomaly` |

#### Dataclasses

| Class | Key Fields | Purpose |
|-------|-----------|---------|
| `Anomaly` | `id`, `dataset_id`, `column`, `row_index`, `anomaly_type`, `severity`, `score` (0-1), `method`, `observed_value`, `expected_value`, `deviation`, `explanation` | A single detected anomaly |
| `AnomalyReport` | `dataset_id`, `anomalies`, `summary`, `unavailable_analyses`, `metadata` | Aggregated anomaly results for one dataset |
| `DriverResult` | `id`, `feature`, `target`, `rank`, `relationship_type`, `strength`, `direction`, `importance`, `method`, `confidence_basis`, `explanation` | A single driver/feature relationship |
| `DriverReport` | `dataset_id`, `target_column`, `target_type`, `drivers`, `relationships`, `unavailable_analyses`, `metadata` | Aggregated driver analysis for one dataset |
| `Phase5Result` | `dataset_id`, `anomaly_report`, `driver_report` | Top-level aggregate of both reports |

**AnomalyReport** provides query helpers: `by_column(col)`, `by_type(type)`, `by_severity(sev)`, `critical_anomalies()`.

**DriverReport** provides: `top_drivers(n=5)`, `by_method(method)`.

---

### Numeric Anomaly Detection (`numeric.py`)

Detects outliers in numeric columns using three complementary methods:

| Method | Min Observations | When Used | Severity Mapping |
|--------|-----------------|-----------|------------------|
| **IQR** (Interquartile Range) | 10 | Always (primary method) | 1.5x IQR = MEDIUM, 2x = HIGH, 3x = CRITICAL |
| **Z-score** | 30 | Skewness <= 2.0 (approximately normal) | \|z\| > 2.5 = MEDIUM, > 3 = HIGH, > 4 = CRITICAL |
| **MAD** (Median Absolute Deviation) | 20 | Skewed data (\|skew\| > 2) | Modified z > 3 = MEDIUM, > 4 = HIGH, > 5 = CRITICAL |

**Key behaviors:**

- Columns with zero variance (constant) are skipped.
- Identifier columns are excluded.
- Infinite values are replaced with NaN before analysis.
- Anomaly `score` is normalized 0-1 based on deviation magnitude.

**Resource limits:**
- `MAX_ROWS_ANALYZE = 50,000` (rows sampled if exceeded)
- `MAX_NUMERIC_COLS = 50`

**Public API:**

```python
detect_numeric_anomalies(
    context: AnalysisContext,
    dataset_id: str,
    max_anomalies: int = 50,
) -> list[Anomaly]
```

---

### Time-Series Anomaly Detection (`time_series.py`)

Detects anomalies in time-ordered numeric data using three methods:

| Method | Min Observations | Threshold | Anomaly Types Produced |
|--------|-----------------|-----------|----------------------|
| **Rolling deviation** | 10 | 2.5 std from rolling mean | `time_series_anomaly` |
| **Spike/drop detection** | 10 | 50% period-over-period change | `sudden_change` |
| **Trend residual** | 20 | 2.5 std from linear trend | `time_series_anomaly` |

**Key behaviors:**

- Requires a `time_column` in `CanonicalDataset.time_info`; graceful no-op if absent.
- Data is sorted by time column before analysis.
- Each numeric column is analyzed independently against the time axis.
- Constant series are skipped.

**Resource limits:**
- `MAX_ROWS = 50,000`
- `MAX_TIME_COLS = 10`

**Public API:**

```python
detect_time_series_anomalies(
    context: AnalysisContext,
    dataset_id: str,
    max_anomalies: int = 50,
) -> list[Anomaly]
```

---

### Categorical Anomaly Detection (`categorical.py`)

Detects anomalies in categorical columns:

| Detection Type | Description | Anomaly Type |
|---------------|-------------|--------------|
| **Rare categories** | Categories appearing in < 1% of rows (min 5 total obs) | `category_anomaly` |
| **Unusual metrics** | Category groups where a numeric measure deviates > 2 std from the grand mean | `distribution_shift` |
| **Missingness** | Columns with > 30% missing values | `missingness_anomaly` |

**Key behaviors:**

- Prefers `dimension_candidates` from canonical metadata; falls back to `is_categorical` flag.
- High-cardinality columns (> 100 unique values) are skipped.
- Identifier columns are excluded.
- At most 3 measure columns are checked per categorical column.

**Resource limits:**
- `MAX_CATEGORICAL_COLS = 20`
- `MAX_CATEGORIES_PER_COL = 100`

**Public API:**

```python
detect_categorical_anomalies(
    context: AnalysisContext,
    dataset_id: str,
    max_anomalies: int = 50,
) -> list[Anomaly]
```

---

### Anomaly Detector Orchestrator (`detector.py`)

Merges numeric, time-series, and categorical sub-detectors into a single `AnomalyReport`.

**Behavior:**

1. Calls each sub-detector independently; a failure in one does not block the others.
2. Failed sub-detectors are recorded in `unavailable_analyses`.
3. All anomalies are sorted by severity (CRITICAL first), then by score descending.
4. Results are capped at `max_anomalies` (default 100).
5. Summary includes counts by type, by severity, affected columns, and most anomalous columns (top 5).
6. Metadata records `total_rows_analyzed`, columns analyzed per type, and time-series availability.

**Public API:**

```python
detect_anomalies(
    context: AnalysisContext,
    dataset_id: str,
    max_anomalies: int = 100,
) -> AnomalyReport

get_anomaly_summary(
    context: AnalysisContext,
    dataset_id: str,
) -> dict
```

---

### Driver Analysis Engine (`driver_engine.py`)

Target-aware driver analysis that extends (not duplicates) the Phase 3 `relationship_engine`. Reuses `analyze_relationships()` and `get_correlation_matrix()` from Phase 3; adds four driver methods oriented toward a detected or user-specified target column.

#### Target Resolution

The engine auto-detects the target from `CanonicalDataset.target_candidates` (highest-scoring candidate) or accepts a user-specified `target_column`. Target type is classified as `numeric`, `categorical`, or `unavailable`.

#### Driver Methods

| Method | Applies When | Strength Metric | Direction |
|--------|-------------|-----------------|-----------|
| **Correlation ranking** | Numeric target | max(\|Pearson r\|, \|Spearman rho\|) | `positive` / `negative` |
| **Mutual information** | Any target type | Normalized MI (0-1) | `non_linear` |
| **GradientBoosting importance** | >= 30 rows, any target | Normalized feature importance (0-1) | `non_linear` |
| **Group comparison** | Numeric target, categorical features | Spread of group means as % of overall mean | `varies` |

- **Correlation** reuses the Phase 3 correlation matrix when the target is present; computes pairwise Pearson as fallback. Spearman is always computed independently.
- **Mutual information** uses `sklearn.feature_selection.mutual_info_regression` or `mutual_info_classif` depending on target type.
- **Model importance** fits a `GradientBoostingRegressor` or `GradientBoostingClassifier` (50 estimators, depth 3).
- **Group comparison** checks each categorical dimension (cardinality 2-20, min 3 obs per group) against the numeric target.

#### Merge and Rank

When multiple methods detect the same feature, results are merged:
- Importance is averaged across methods.
- The best single-method result provides direction, explanation, and relationship type.
- Methods and confidence bases are concatenated.
- Final ranking is by merged importance descending.

#### Phase 3 Integration

The driver report always includes Phase 3 relationships (serialized via `to_dict()`). Phase 3's `analyze_relationships()` is called with error handling -- a failure does not block driver results.

**Resource limits:**
- `MAX_FEATURES = 50` (by variance)
- `MAX_ROWS = 50,000` (sampled if exceeded)

**Public API:**

```python
analyze_drivers(
    context: AnalysisContext,
    dataset_id: str,
    target_column: str | None = None,
) -> DriverReport
```

---

### Non-Causal Language Policy

All generated explanations use associative, non-causal language:

| Never | Always |
|-------|--------|
| "causes" | "associated with" |
| "leads to" | "predictive of" |
| "results in" | "linked to variation in" |
| "because" (causal) | "This indicates predictive importance, not causation." |

Every driver explanation ends with a non-causation disclaimer. This policy is enforced in tests.

---

### Resource Limits Summary

| Limit | Value | Module |
|-------|-------|--------|
| `MAX_ROWS` / `MAX_ROWS_ANALYZE` | 50,000 | numeric, time_series, driver_engine |
| `MAX_NUMERIC_COLS` | 50 | numeric |
| `MAX_TIME_COLS` | 10 | time_series |
| `MAX_CATEGORICAL_COLS` | 20 | categorical |
| `MAX_CATEGORIES_PER_COL` | 100 | categorical |
| `MAX_FEATURES` | 50 | driver_engine |
| `max_anomalies` (per sub-detector) | 50 (default) | numeric, time_series, categorical |
| `max_anomalies` (orchestrator) | 100 (default) | detector |

---

### Dataset Isolation

Each analysis run is stamped with a `dataset_id`. Guarantees:

- Every `Anomaly.dataset_id` matches the report's `dataset_id`.
- Every `DriverResult` is scoped to one target within one dataset.
- `Phase5Result.dataset_id` propagates to both sub-reports.
- No state is shared between analysis runs for different datasets.

---

### Pipeline Integration (`analysis_runner.py`)

Phase 5 runs after Phase 4 in `run_full_analysis()`:

```python
from src.anomaly.detector import detect_anomalies
from src.anomaly.driver_engine import analyze_drivers
from src.anomaly.contracts import Phase5Result

anomaly_report = detect_anomalies(context, dataset_id)
driver_report = analyze_drivers(context, dataset_id)

phase5_result = Phase5Result(
    dataset_id=dataset_id,
    anomaly_report=anomaly_report,
    driver_report=driver_report,
)
# Persisted to: <processed_dir>/phase5_analysis.json
```

Phase 5 is non-fatal: if it fails, earlier phases' results are preserved and the error is appended to the run report.

---

### Dashboard Page (`dashboard/pages/12_anomalies_drivers.py`)

Streamlit page rendering Phase 5 results. Loads `phase5_analysis.json` via `load_phase5_analysis()`. All labels, metrics, and chart axes are derived from the report JSON -- no hardcoded column names.

Features:
- Anomaly summary cards (counts by severity and type)
- Anomaly detail table with filtering by column, type, and severity
- Top drivers ranked list with strength/importance bars
- Driver detail view with method breakdown
- Phase 3 relationship integration display
