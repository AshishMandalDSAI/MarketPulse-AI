# Canonical Data Contract

## Purpose

The canonical data contract provides a **universal internal representation** for any CSV/XLSX dataset uploaded to MarketPulse AI. It eliminates implicit assumptions between modules by providing a standardized interface that every analysis module can consume.

```
ANY CSV/XLSX
     |
Universal Data Engine
     |
CanonicalDataset + AnalysisContext
     |
Capability-Aware Modules
```

No module may assume that a particular business schema (marketing columns, revenue, date) exists. The contract works for marketing, sales, finance, operations, education, HR, customer, healthcare-like, time-series, cross-sectional, no-date, unusual-column, missing-value, and minimal datasets.

---

## Object Structure

### CanonicalDataset

The primary contract object. Always associated with a `dataset_id` — never a global singleton.

```python
from src.data_engine.canonical import CanonicalDataset, build_canonical

canonical, context = build_canonical(
    dataset_id="upload_abc123",
    df=my_dataframe,
    filename="quarterly_sales.csv"
)
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `dataset_id` | `str` | Unique dataset identifier |
| `filename` | `str` | Original upload filename |
| `row_count` | `int` | Number of rows |
| `column_count` | `int` | Number of columns |
| `dataframe` | `DataFrame` | In-memory data (not serialized) |
| `columns` | `list[str]` | Column names in order |
| `column_metadata` | `dict[str, ColumnMetadata]` | Per-column metadata |
| `date_columns` | `list[str]` | Datetime-type columns |
| `numeric_columns` | `list[str]` | Numeric columns |
| `categorical_columns` | `list[str]` | Categorical columns |
| `boolean_columns` | `list[str]` | Boolean columns |
| `text_columns` | `list[str]` | Free-text columns |
| `identifier_columns` | `list[str]` | ID/key columns |
| `semantic_roles` | `dict[str, list[str]]` | Column → semantic role(s) |
| `target_candidates` | `list[TargetCandidate]` | Ranked target columns |
| `measure_candidates` | `list[str]` | Numeric non-ID columns |
| `dimension_candidates` | `list[str]` | Categorical grouping columns |
| `time_candidates` | `list[str]` | Potential date columns |
| `data_quality` | `DataQuality` | Quality report |
| `time_info` | `TimeInfo` | Time series characteristics |
| `capabilities` | `dict` | Available analysis capabilities |
| `overrides` | `UserOverrides` | User-specified role overrides |

---

## Semantic Roles

Every column is assigned one or more semantic roles from `SemanticRole`:

| Role | Description |
|------|-------------|
| `TIME` | Date/datetime column |
| `TARGET` | Primary outcome variable |
| `MEASURE` | Numeric measurement |
| `DIMENSION` | Categorical grouping |
| `CATEGORY` | Categorical values |
| `IDENTIFIER` | Unique ID/key |
| `TEXT` | Free-text content |
| `BOOLEAN` | True/false values |
| `UNKNOWN` | Unclassified |
| `MARKETING_SPEND` | Marketing channel spend (optional) |
| `MARKETING_CHANNEL` | Marketing channel name (optional) |

Marketing roles are **optional extensions** — never mandatory. A dataset with no marketing data still works fully.

---

## Capability Contract

Capabilities are determined from the **active dataset**, not globally:

```python
# Marketing time-series dataset:
context.has_capability("marketing_mix_modeling")  # True
context.has_capability("forecasting")             # True

# Education cross-sectional dataset:
context.has_capability("marketing_mix_modeling")  # False
context.has_capability("forecasting")             # False
context.has_capability("data_profiling")          # True
```

Formula: `Application capability + Dataset capability = Available feature`

---

## Target Detection

Target candidates are detected using structural and semantic signals, **never hardcoded** to Revenue/Sales/Profit:

- Semantic role analysis (TARGET, MEASURE roles)
- Name-based hints (generic terms: revenue, score, salary, outcome, etc.)
- Statistical variance (coefficient of variation)
- Non-negativity
- Cardinality
- Missing rate

```python
canonical.target_candidates
# [TargetCandidate(column="Exam_Score", score=0.85, reasons=[...]),
#  TargetCandidate(column="Assignment_Score", score=0.65, reasons=[...])]
```

---

## Time Detection

Supports datetime columns, date-like strings, and timestamps:

```python
canonical.time_info
# TimeInfo(
#     time_column="Date",
#     time_granularity="weekly",      # daily | weekly | monthly | quarterly | yearly | irregular
#     is_regular_time_series=True,
#     time_range={"start": "2024-01-07", "end": "2024-12-29", "periods": 52}
# )
```

For datasets without time: `time_column = None`, and downstream features understand that.

---

## Quality Contract

```python
canonical.data_quality
# DataQuality(
#     row_count=100,
#     column_count=5,
#     missing_cells=75,
#     missing_cells_pct=15.0,
#     duplicate_rows=0,
#     constant_columns=[],
#     empty_columns=[],
#     high_cardinality_columns=[],
#     mixed_type_columns=[],
#     infinite_value_columns=[],
#     score=0.925,
#     warnings=["3 column(s) with >50% missing: [...]"]
# )
```

Quality score (0.0–1.0) is computed from actual issues, not fabricated.

---

## Dataset Isolation

Every canonical object is bound to a `dataset_id`. There are no global singletons:

```python
canonical_a = build_canonical("upload_abc", df_a)[0]
canonical_b = build_canonical("upload_xyz", df_b)[0]
# canonical_a.dataset_id != canonical_b.dataset_id — always
```

Cache keys remain dataset-specific. Phase 1 isolation rules are fully preserved.

---

## User Overrides

The contract supports future user overrides:

```python
from src.data_engine.canonical import UserOverrides

overrides = UserOverrides(
    target_column="Net_Sales",
    date_column="Transaction_Date",
    ignore_columns=["internal_id"],
    force_categorical=["status_code"],
    force_measure=["custom_metric"],
    force_marketing_spend=["search_spend", "social_spend"]
)

canonical, ctx = build_canonical("my-dataset", df, overrides=overrides)
```

---

## Examples

### Marketing Dataset

```
Date,Revenue,Google_Ads,Facebook_Ads,YouTube_Ads,Orders,Customers
```

- `Date` → TIME
- `Revenue` → TARGET, MEASURE
- `Google_Ads`, `Facebook_Ads`, `YouTube_Ads` → MEASURE, MARKETING_SPEND
- `Orders`, `Customers` → MEASURE
- Capabilities: profiling, forecasting, MMM, ROI, budget optimization

### Education Dataset

```
Student_ID,Attendance,Assignment_Score,Study_Hours,Exam_Score,Department
```

- `Student_ID` → IDENTIFIER
- `Attendance`, `Assignment_Score`, `Study_Hours`, `Exam_Score` → MEASURE
- `Department` → DIMENSION, CATEGORY
- Target candidates: `Exam_Score` (highest), `Assignment_Score`
- Capabilities: profiling, generic KPIs, AI analyst — no MMM, no ROI

### No-Date Dataset

```
product_id,weight_kg,price,rating,category,warehouse
```

- `product_id` → IDENTIFIER
- `weight_kg`, `price`, `rating` → MEASURE
- `category`, `warehouse` → DIMENSION, CATEGORY
- `time_column` = None, `time_granularity` = None
- Capabilities: profiling, generic KPIs — no forecasting, no trend analysis

---

## Serialization

All metadata (everything except the DataFrame) is JSON-serializable:

```python
# Serialize
metadata = canonical.to_metadata_dict()
json.dumps(metadata)  # safe

# Deserialize
restored = CanonicalDataset.from_metadata_dict(metadata, df=my_dataframe)
```

Safe for: Streamlit session state, caching, API responses, testing, reports, AI Copilot context.

---

## Integration Path

```
Upload → profile_and_map → standardize → CanonicalDataset.from_standardized()
                                              |
                                    AnalysisContext.from_canonical()
                                              |
                                    capability-aware modules
```

The canonical contract builds **on top of** the existing `StandardizedDataset` — it does not replace it. Backward compatibility is preserved through `canonical.get_standardized()`.
