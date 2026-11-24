# Dynamic Analytics Engine (Phase 3)

## Overview

Phase 3 introduces a fully dynamic, dataset-agnostic analytics layer that automatically discovers KPIs, recommends visualizations, detects insights, and analyzes relationships from **any** uploaded dataset. It builds on Phase 2's `CanonicalDataset` and `AnalysisContext` contracts without replacing them.

## Architecture

```
CanonicalDataset (Phase 2)
    |
    v
AnalysisContext (Phase 2)
    |
    +---> KPI Engine        --> list[KPI]
    +---> Visualization Engine --> list[ChartRecommendation]
    +---> Insight Engine     --> list[Insight]
    +---> Relationship Engine --> list[Relationship]
    |
    v
AnalyticsResult (Phase 3 aggregate)
```

All four engines consume `AnalysisContext` and produce typed dataclass results. No engine hardcodes column names. Every output carries provenance (source columns, calculations, evidence, confidence scores).

## Modules

### `src/analytics/contracts.py`

Phase 3 data contracts:

- **`KPI`** — label, value, formatted_value, source_column, calculation, category, confidence, explanation, priority, delta
- **`ChartRecommendation`** — chart_type, title, x/y columns, group_by, aggregation, score, reason, category
- **`Insight`** — title, description, insight_type, severity, evidence, source_columns, confidence, metric_value
- **`Relationship`** — relationship_type, columns, strength, direction, description, evidence
- **`AnalyticsResult`** — aggregate container with `primary_kpis()`, `top_charts()`, `top_insights()` convenience methods

All have `to_dict()` / `from_dict()` for JSON round-tripping.

### `src/analytics/kpi_engine.py`

**`discover_kpis(context, max_primary=8) -> list[KPI]`**

Discovers KPIs dynamically:
1. **Total Records** — always present (row count)
2. **Date Range** — if time column detected (start, end, granularity)
3. **Target totals/averages** — for top target candidates (sum, mean)
4. **Growth rates** — period-over-period change for targets with time
5. **Non-target measure averages** — for other numeric columns
6. **Identifier uniqueness** — unique count for ID columns
7. **Dimension top categories** — most common value in each dimension
8. **Data quality score** — from the data quality contract
9. **Derived boolean rates** — percentage for 0/1 binary columns

### `src/analytics/visualization_engine.py`

**`recommend_charts(context, max_charts=12) -> list[ChartRecommendation]`**

Scored recommendation rules:
| Condition | Chart Type | Base Score |
|-----------|-----------|------------|
| Time + numeric target | Line | 0.95 |
| Time + multiple measures | Area | 0.80 |
| Category + numeric | Bar | 0.82 |
| Category + numeric (many cats) | Horizontal Bar | 0.75 |
| Two numeric columns | Scatter | 0.60 + |r|*0.3 |
| Single numeric | Histogram | 0.55 |
| Low-cardinality category | Donut | 0.65 |
| Multiple numeric columns | Heatmap | 0.70 |
| Numeric by category | Box | 0.60 |

### `src/analytics/insight_engine.py`

**`discover_insights(context, max_insights=15) -> list[Insight]`**

Detectors:
- **Trends** — growth/decline >2% over time (linear correlation)
- **Concentration** — top-3 categories hold >50% of a measure
- **Distribution** — skewness analysis for numeric columns
- **Outliers** — IQR method, flagged when >1% of values are outliers
- **Correlations** — Pearson |r| >= 0.5 between non-ID numeric columns
- **Data Quality** — missing >5%, duplicates >1%, constant columns

### `src/analytics/relationship_engine.py`

**`analyze_relationships(context, max_relationships=15) -> list[Relationship]`**

- **Correlations** — Pearson r between numeric columns (|r| >= 0.3, excludes IDs/constants)
- **Group Comparisons** — measure variance across categorical dimensions (>5% spread)
- **Trend Relationships** — linear correlation between time index and numeric columns

**`get_correlation_matrix(context) -> pd.DataFrame | None`**

Returns the full Pearson correlation matrix for non-ID, non-constant numeric columns.

## Integration Points

### Analysis Runner (`src/data_engine/analysis_runner.py`)

After Phase 2 canonical metadata is persisted, the runner executes all four engines and saves results to `processed/dynamic_analytics.json`.

### Dashboard Data Loader (`dashboard/components/data_loader.py`)

Three cached loaders:
- `load_canonical_metadata()` — reads canonical_metadata.json
- `load_analysis_context()` — reconstructs AnalysisContext from metadata + dataframe
- `load_dynamic_analytics()` — runs engines on-demand, cached per dataset

### Executive Overview (`dashboard/pages/1_executive_overview.py`)

Smart routing:
1. Try dynamic analytics first (if KPIs available)
2. Fall back to legacy marketing overview (if marketing capabilities present)
3. Final fallback: dynamic overview with whatever analytics are available

Dynamic sections:
- Top KPIs with expandable methodology
- Recommended charts rendered as Plotly figures
- Severity-coded insight cards with evidence
- Relationship strength bars
- Data quality summary
- Capability summary

## Safety Guarantees

1. **No hardcoded columns** — engines never assume Revenue, Date, Google Ads, etc.
2. **ID exclusion** — identifier columns excluded from correlations, charts, insights
3. **Constant column handling** — excluded from correlations, flagged in data quality
4. **Dataset isolation** — KPIs/charts/insights from dataset A never leak to dataset B
5. **No fabrication** — every metric computed from actual data with source provenance
6. **No demo fallback** — dynamic analytics never falls back to demo data
7. **Edge cases** — empty DataFrames, single-column, all-categorical, high-cardinality all handled gracefully

## Testing

90 tests in `tests/test_phase3_dynamic_analytics.py` covering:
- Contract serialization round-trips
- KPI discovery across all fixture schemas
- Visualization recommendations (line, bar, scatter, histogram, heatmap, donut)
- Insight detection (trends, correlations, quality, outliers, concentration)
- Relationship analysis (correlations, group comparisons, trends)
- No-hardcoded-columns safety checks
- Dataset isolation (2-way and 3-way)
- Edge cases (constant, single numeric, all categorical, high cardinality, very small, mixed types)
- Analysis runner integration
- Parametric tests across all 11 diverse fixture schemas
