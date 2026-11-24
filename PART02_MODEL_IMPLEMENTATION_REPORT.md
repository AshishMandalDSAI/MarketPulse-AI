# PART 02 model implementation report

## Architecture

| Layer | Module | Role |
|---|---|---|
| Vocabulary | `src/data_engine/role_lexicon.py` | Generic business words, the tokenizer, and the trade/promotion-spend rule shared with the legacy detectors |
| Dataset engine | `src/data_engine/role_engine.py` | Column roles, confidence, evidence, alternatives, warnings; relationship checks; mode detection; versioned mapping JSON; bridge to the legacy pipeline |
| Model context | `src/marketing_intel/context.py` | Target and channels for the active mode, business drivers, dimensions, excluded columns (with reasons), value grain, per-period totals |
| Statistics | `src/marketing_intel/stats.py` | OLS with HC3 robust SEs, VIF, permutation importance, Welch t-test, correlations, Benjamini-Hochberg (numpy/scipy only; no new dependency) |
| Response model | `src/marketing_intel/response.py` | Shared driver model with chronological holdout validation and a quality grade |
| Models 04–07 | `effectiveness.py`, `channels.py`, `optimizer.py`, `scenarios.py` | Business logic, no Streamlit |
| Capabilities | `src/marketing_intel/capabilities.py` | Each model's supported and unsupported capabilities, with reasons |
| Optimizer limits | `src/prescriptive/limits.py` | The 55% default fix (see `PART02_OPTIMIZER_FIX_REPORT.md`) |
| UI | `dashboard/components/role_mapping_ui.py`, `model_page.py`; `dashboard/pages/{marketing_effectiveness,channel_performance,resource_optimizer,scenario_simulator}.py` | Shared page frame: dataset, mode, roles/target, capabilities, results, methodology, assumptions, warnings |

Existing engines are **reused, not duplicated**:

- **Spend mode:**
  - the marketing-mix model (`src.marketing.mmm_engine.fit_mmm`, via `prescriptive.engine.fit_response_model`);
  - contributions (`src.marketing.contribution`);
  - the allocation optimiser and what-if simulator (`src.prescriptive.engine`).
- **Kept and unchanged:** the old pages (Marketing Mix Model, Channel ROI, Budget Optimizer, What-If Simulator), the `src/models`, `src/optimization` and `src/marketing` engines, the API endpoints and the AI analyst.
- **Old pages:** each new page keeps the previous module reachable behind its labelled switch, as in PART 01.

## Universal dataset engine

### Signals

- **Column names:** split into words, matched against generic vocabulary, never against fixed column names.
- **Type and values:** numeric share, whole numbers, 0/1 flags, 0–1 and 0–100 ranges, negatives, uniqueness, dates that parse (ISO, day-first, mixed, `YYYYMMDD`).
- **Relationships between columns:**
  - value ≈ units × price (the best-fitting price column is chosen);
  - a spend column that equals the sum of other spend columns is flagged as an aggregate, not a channel;
  - outcome totals made of other outcomes;
  - identical columns.

### Roles

Each group below is listed with its roles:

- **Outcomes:** revenue, sales units, orders, conversions, leads, profit, customers.
- **Spend:**
  - advertising spend for one channel;
  - total marketing spend, an aggregate that is not channel-wise.
- **Exposure:** impressions, reach, clicks, views, visits, engagement, readership, listenership, GRPs, audience.
- **Business variables:** price, discount, promotion, **trade / promotion spend**, distribution, competitor, external factors, non-marketing cost.
- **Dimensions:** brand, product, SKU, geography, channel name.
- **Time:** date, calendar part.
- **Other:** identifier, derived ratio, other measure, text, ignore.

### What each column shows

The suggested role, confidence (0.05–0.99), evidence sentences, up to four alternatives, and warnings:

- "Ambiguous…" when the top two scores are within 0.12;
- low confidence;
- a constant column;
- more than 20% missing;
- a column identical to another.

### Target confirmation

All outcome columns are candidates, and **none is selected when there's more than one**. The Upload page's "Confirm & Analyze" button and the model pages are blocked until the user chooses.

A single, clearly named outcome (confidence ≥ 0.75) is preselected but can be changed. More than one target can be confirmed, and each one is analysed independently through a selector on every model page.

### Corrections and storage

- **Corrections:** the user edits roles and channel names in an editable table. A user choice gets confidence 1.0 and `user_set`.
- **Storage:** `role_mapping.json` in the dataset's processed folder (`schema_version`, dataset fingerprint, `version`, `confirmed_at`, history of up to 50 entries with the roles changed). A mapping for a different table structure or schema version is ignored, never misapplied.
- **The demo:** it uses engine suggestions with its documented target, Revenue, and edits apply to the session only.

### Modes

- **Spend-based:** at least one channel advertising-spend column.
- **Exposure-based:** exposure columns and no channel spend.
- **General:** neither.

The banner shows the evidence for the mode.

### Legacy pipeline

`legacy_overrides()` turns the confirmed mapping into the overrides the existing `run_full_analysis` already accepted:

- the date;
- a monetary target as revenue (a unit target is never passed as revenue);
- channel columns = advertising spend only.

`column_classifier` and `field_discovery` also skip trade/promotion spend when guessing, but a channel the user confirms still goes through.

## 04 · Marketing Effectiveness Model

### Driver model

- **Form:** OLS of the target on the drivers. Channels enter as `log(1+x)` (diminishing returns); business drivers enter linearly.
- **Segments:** optional fixed effects for brand, geography, product and SKU. Nested dimensions are reduced automatically (SKU within Brand keeps SKU).
- **Trend:** an optional linear time trend.
- **Driver defaults:**
  - the active mode's channels plus the business variables;
  - near-duplicates (|r| > 0.98) are left out by default, with a note, and can be added back;
  - other outcomes, derived ratios, aggregates and identifiers are always excluded (no leakage).
- **Validation:**
  - the last 20% of periods are held out (chronological split);
  - quality comes from holdout R² on **per-period totals** (panel data) or on rows: good ≥ 0.5, moderate ≥ 0.3, weak < 0.3;
  - coefficients come from a refit on all rows with HC3 robust SEs, 95% CIs and p-values.

### Outputs

- the effect table: coefficient, CI, p-value, elasticity at average levels, standardized effect, VIF, and the channel association share (%);
- elasticity and association-share charts;
- actual vs model chart with the holdout periods shaded;
- permutation importance (10 repeats, fixed seed);
- Pearson / Spearman correlations, with Benjamini-Hochberg adjusted p-values;
- effectiveness over time and by brand / geography / product (minimum 8 periods per cell);
- promotion effectiveness (raw on/off difference, Welch test, controlled coefficient).

### Effectiveness by mode

- **Spend mode:** adds the existing marketing-mix model's contribution.
  - With a monetary target: **ROI = (contribution − spend)/spend** and **ROAS = contribution/spend**.
  - With a unit target: "outcome per unit of spend" instead.
- **Exposure mode:** a banner explains that effects are per exposure unit, with no ROI, ROAS or monetary contribution.
- **General mode:** explains why the marketing model can't run and what's needed; it runs the same analysis on the business drivers and shows segment performance.

## 05 · Channel Performance & Contribution

- **Channels:** spend columns (spend mode) or exposure columns (exposure mode), each in its own unit.
- **Value grain:** detected per channel. An exposure column recorded per week × brand × geography and repeated on every SKU row is counted once.
- **Channel table:**
  - total, average per period, share **within the same unit**, active periods, change between halves;
  - correlation with the target per period;
  - elasticity and association share.
  - Spend mode also shows MMM contribution and ROI/ROAS, or outcome per unit of spend.
- **Ranking basis:** model ROAS, then outcome per unit of spend, then association share, then correlation, using the first one available.
- **Charts:** ranking, average per period per unit, trends per unit, breakdowns by brand / geography / product, and MMM contribution (spend mode only).
- **Rates:** click-through and other rates only where both measures exist for the same channel.
- **No channels:** an explanation plus segment performance.

## 06 · Marketing Resource Optimizer

- **Spend mode:** the existing prescriptive engine on per-period totals, with the MMM response, feasible default limits, a feasibility summary, solver status and the objective.
- **Exposure mode:**
  - **What it does:** reallocates exposure units among channels in the **same unit**, holding the total fixed. The objective is the model-estimated target, using the fitted log curves over the recent baseline window (SLSQP, three starts, share-scaled variables).
  - **When it's allowed:** only when the driver model's quality is at least moderate and at least two channels in that unit have positive, significant coefficients (p < 0.10). Otherwise it states why and recommends nothing.
  - **What it reports:** the result is labelled in units and "not money", with an extrapolation warning, the held channels, an approximate range and assumptions, including that buying exposure has costs not in the data.
- **General mode:** an explanation of the missing inputs.

## 07 · Marketing Scenario Simulator

- **Baseline window:** last 8, 13 or 26 periods, or all history.
- **Baseline values:** the **historical actual** average per period and the **model baseline** are shown as separate rows and as separate bar types.
- **Scenarios:** several at once, as % changes to channels, prices, distribution and other numeric drivers, or as a promotion share (% of records on), applied as its expected value. Presets are built only from drivers that exist. Scenarios can be added and removed.
- **What each result shows:**
  - the estimate and its change vs the model baseline;
  - an approximate 90% range (± 1.645 × holdout RMSE of per-period totals);
  - extrapolation flags (values outside the observed range, or changes larger than ±50%);
  - a reliability grade. With a weak model the page says "must not be used as forecasts".
- **Spend mode:** adds the marketing-mix model cross-check for spend-only scenarios.

## Visualisation

Every chart on pages 04–07 has a title, x and y axis titles, and units:

- the dataset currency for money (none when unknown, never INR or USD by default);
- the exposure unit for exposure;
- per day / week / month from the detected frequency.

Tests assert this on the real pages.

## Known limitations

- **Simple driver model:** the shared driver model has no adstock (carry-over). On the bundled demo, its holdout R² is about 0.0, so the pages grade it **weak** and say so. In spend mode the existing adstock / saturation MMM is shown alongside it.
- **Long-format channel data:** a channel name stored in one column, with values in another, is recognised as a channel *dimension* but isn't pivoted into per-channel measures.
- **Scenario ranges:** these use the holdout error only. Full prediction intervals, multi-scenario completion and sensitivity analysis are PART 04.
- **Exposure optimizer:** reallocates units, not money. Without costs per unit it can't say whether a new mix is affordable, and the page says so.
- **Demo mapping edits:** these apply for the session only, because the demo is bundled read-only data.
