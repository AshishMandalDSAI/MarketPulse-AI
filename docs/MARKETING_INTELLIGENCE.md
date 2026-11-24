# Phase 6: Universal Marketing Intelligence

## Overview

Phase 6 adds **dataset-agnostic marketing intelligence** to MarketPulse AI.
Rather than assuming every dataset is a marketing dataset, the system
dynamically discovers whether a dataset contains marketing-relevant fields
and, only when it does, runs a full marketing analysis pipeline:

1. **Field Discovery & Capability Assessment** — detect channels, target,
   time, and controls from any schema
2. **Marketing Mix Modeling (MMM)** — Ridge regression with adstock and
   saturation transforms
3. **Channel Contribution** — leave-one-channel-out ablation
4. **ROI / ROAS** — naive and model-based computation
5. **Budget Optimisation** — SLSQP-based allocation maximising predicted target
6. **What-If Simulation** — scenario comparison (current, optimised, +20%, -20%)

Every step is capability-gated: if the dataset lacks the required fields or
rows, the step is skipped with a clear reason.  No step ever falls back to
demo data.

---

## Architecture

```
src/marketing/
  __init__.py
  contracts.py          # All Phase 6 data contracts
  field_discovery.py    # 3-signal channel detection + 8-rule validation
  mmm_engine.py         # Dataset-agnostic MMM (Ridge + adstock + saturation)
  contribution.py       # Channel contribution via ablation
  roi.py                # ROI/ROAS (naive + model-based)
  optimizer.py          # Budget optimisation (scipy SLSQP)
  simulator.py          # What-if scenario simulation
  orchestrator.py       # Pipeline orchestrator (single entry point)
```

The orchestrator is called by `src/data_engine/analysis_runner.py` at the
end of `run_full_analysis()`, receiving a `CanonicalDataset` and
`AnalysisContext` from Phase 2.

### Dependency Flow

```
CanonicalDataset + AnalysisContext
        |
   field_discovery.assess_marketing_capabilities()
        |
   MarketingCapabilityReport (channels, target, readiness)
        |
   +-- roi.compute_roi_naive()              [if channels + target]
   +-- mmm_engine.fit_mmm()                 [if channels + target + time + rows >= 12]
        |
        +-- contribution.compute_contributions()
        +-- roi.compute_roi_model_based()
        +-- optimizer.optimize_budget()      [if run_optimization]
        +-- simulator.simulate_standard_scenarios()  [if run_whatif]
```

---

## Field Discovery

Channel detection uses a **three-signal approach** (no giant hardcoded keyword list):

1. **Semantic roles** — columns already tagged `MARKETING_SPEND` or
   `MARKETING_CHANNEL` by Phase 2's profiler
2. **Alias vocabulary** — matching against `CHANNEL_SPEND_ALIASES` from
   `config/column_aliases.py` (platform names like `google`, `facebook`,
   `tv`, `radio` combined with spend indicators like `_spend`, `_cost`)
3. **Statistical profile** — continuous, non-negative, variable columns that
   correlate with the target

Each candidate channel passes through **8 validation rules**:

| Rule | Description |
|------|-------------|
| Numeric type | Must be numeric (int/float) |
| Non-negative | No negative values (spend can't be negative) |
| Non-identifier | Not an ID column (unique ratio < 0.9) |
| Sufficient observations | At least max(10, 10% of rows) non-null |
| Variance | Not constant or near-constant |
| Not the target | Can't be the revenue/target column itself |
| Correlation sign | Non-negative correlation with target (if available) |
| Multicollinearity | Warning (not rejection) if r > 0.95 with another channel |

### Readiness Levels

- **FULL** — channels + target + time column + >= 20 rows
- **PARTIAL** — channels + target but missing time or insufficient rows
- **MINIMAL** — some channels detected but no clear target
- **NOT_READY** — no marketing-relevant fields detected

---

## Marketing Mix Modeling

The MMM engine preserves the proven algorithms from the original codebase
but generalises inputs:

### Algorithm

1. **Per-channel parameter search**: grid search over `DECAY_GRID` x
   `SHAPE_GRID` to find optimal adstock decay and Hill saturation parameters
   for each channel independently (minimising negative correlation with target)

2. **Three model variants**:
   - `baseline` — raw spend features
   - `adstock` — geometric adstock-transformed features
   - `adstock_saturation` — adstock + Hill saturation

3. **Time-aware train/test split**: 80/20 chronological split when a time
   column is available; random split otherwise

4. **Model selection**: prefers the simplest model unless a more complex
   variant improves OOS RMSE by > 0.1% AND MAPE by > 2%

5. **Ridge regression** with `alpha=5.0` for regularisation

### Feature Naming

Features use `__` (double underscore) as separator:
- `{channel}__raw` — baseline model
- `{channel}__adstock` — adstock model
- `{channel}__adstock_sat` — adstock + saturation model

Control columns are included as-is.

### Resource Limits

- `MIN_ROWS_FOR_MMM = 12`
- `MAX_ROWS_FOR_MMM = 50,000` (tail-sampled if exceeded)
- `TEST_FRACTION = 0.20`

---

## Channel Contribution

Uses **leave-one-channel-out ablation**:

1. Predict target with all channels at actual values → `full_pred`
2. For each channel, zero out that channel's feature and predict → `ablated_pred`
3. Contribution = `full_pred - ablated_pred` (clipped to non-negative)
4. Normalise to percentage shares that sum to 100%

This measures each channel's marginal contribution to predicted target,
accounting for adstock and saturation effects when those models are selected.

---

## ROI / ROAS

Two computation modes:

- **Naive**: `ROAS = total_target / total_channel_spend` (simple ratio)
- **Model-based**: uses channel contribution from ablation to attribute
  target to each channel, then `ROAS = attributed_target / spend`

Model-based ROI replaces naive ROI when MMM fitting succeeds.

---

## Budget Optimisation

Preserves the SLSQP algorithm from the original codebase:

- **Objective**: maximise predicted target (via fitted MMM)
- **Constraint**: total allocation = total budget (equality)
- **Bounds**: each channel gets between `min_pct` (5%) and `max_pct` (55%)
  of total budget
- **Carry-in**: adstock carry-in from recent periods held fixed at recent
  average levels

Returns current vs. optimised allocation with expected improvement percentage.

---

## What-If Simulation

Generates four standard scenarios:

| Scenario | Description |
|----------|-------------|
| Current | Recent average allocation |
| Optimised | Output of budget optimiser |
| Aggressive (+20%) | Current mix scaled up by 20% |
| Conservative (-20%) | Current mix scaled down by 20% |

Also supports custom allocation and total-budget scaling via
`simulate_custom_allocation()` and `simulate_total_budget()`.

### Non-Causal Language

All what-if results use correlational language:

> "These results reflect correlations in the fitted model, not
> established causal effects.  The model estimates predicted outcomes
> based on historical patterns."

The system never claims "increasing X causes Y" or "leads to".

---

## Contracts

All data flows through typed contracts in `src/marketing/contracts.py`:

| Contract | Purpose |
|----------|---------|
| `ChannelEvidence` | Evidence for why a column was identified as a channel |
| `ChannelValidation` | Validation result for a candidate channel |
| `AnalysisCapability` | Whether a specific analysis is available + reason |
| `MarketingCapabilityReport` | Full field discovery result |
| `MMMResult` | Fitted model + metrics + parameters |
| `ChannelContribution` | Per-channel contribution share |
| `ROIResult` | Per-channel ROI/ROAS |
| `BudgetOptimizationResult` | Optimiser output |
| `WhatIfScenario` | Single scenario prediction |
| `WhatIfResult` | Collection of scenarios + baseline |
| `Phase6Result` | Top-level result aggregating all analyses |

All contracts implement `to_dict()` for JSON serialization.  `MMMResult`
excludes private model objects (`_model`, `_scaler`) from serialization.

---

## Integration with Analysis Runner

Phase 6 is integrated at the end of `run_full_analysis()` in
`src/data_engine/analysis_runner.py`:

```python
from src.marketing.orchestrator import run_marketing_analysis

phase6_result = run_marketing_analysis(
    canonical=canonical,
    context=context,
    dataset_id=dataset_id,
)
```

Results are persisted to `data/uploads/<dataset_id>/processed/phase6_marketing.json`
and included in the returned dict under the `phase6_result` key.

---

## Non-Marketing Datasets

When a dataset has no marketing-relevant fields (e.g., HR, education,
operations data), the system:

1. Sets readiness to `NOT_READY`
2. Returns an empty `Phase6Result` with clear `unavailable_analyses` messages
3. Does **not** fall back to demo data
4. Does **not** fabricate results
5. Adds a single report line: "Phase 6 marketing intelligence: readiness=NOT_READY, 0 channel(s), MMM=not fitted."
