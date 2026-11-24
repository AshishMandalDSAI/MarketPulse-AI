# Phase 7 — Decision Intelligence

## Architecture Overview

Phase 7 introduces a Decision Intelligence layer that provides structured evidence classification, confidence assessment, and limitation tracking for all analysis outputs. The module is domain-agnostic and deterministic — it works across business domains without hardcoded assumptions about any particular industry, dataset schema, or column names.

The Decision Intelligence module sits downstream of all analysis engines (Phases 1–6) and provides a uniform framework for:

1. Classifying analytical findings by epistemic status
2. Tracking limitations and uncertainty explicitly
3. Assessing confidence deterministically
4. Enforcing evidence safety rules (correlation vs causation, predictions vs facts)
5. Maintaining dataset isolation

## Part 1 Scope: Evidence + Confidence Foundation

Part 1 establishes the foundational contracts and helpers. It does **not** integrate with existing analysis engines — that is deferred to later parts.

### Files Created

| File | Purpose |
|------|---------|
| `src/decision/__init__.py` | Package marker |
| `src/decision/contracts.py` | Immutable data contracts and enums |
| `src/decision/evidence.py` | Evidence classification and safety helpers |
| `src/decision/confidence.py` | Deterministic confidence assessment |
| `tests/test_phase7_part1_evidence_confidence.py` | 130 comprehensive tests |

## Contract Descriptions

### EvidenceLevel

Epistemic status of a piece of evidence, ordered from strongest to weakest:

| Level | Value | Meaning |
|-------|-------|---------|
| FACT | `"fact"` | Directly computed from data (aggregations, row counts, data quality metrics) |
| INFERENCE | `"inference"` | Statistical relationship found (correlations, anomalies, drivers) |
| PREDICTION | `"prediction"` | Model-generated projection (forecasts, scenarios, simulations) |
| RECOMMENDATION | `"recommendation"` | Suggested action based on analysis |
| LIMITATION | `"limitation"` | Known constraint on the analysis |

### ConfidenceLevel

Qualitative confidence assessment. This is a transparent, deterministic label — **not** a statistically calibrated probability.

| Level | Value | Meaning |
|-------|-------|---------|
| HIGH | `"high"` | Strong evidence, multiple corroborating signals, no critical limitations |
| MEDIUM | `"medium"` | Moderate evidence, some uncertainty |
| LOW | `"low"` | Weak evidence, significant uncertainty |
| INSUFFICIENT | `"insufficient"` | Not enough evidence to assess |

### LimitationCategory

Categories of analytical limitations:

| Category | Value | Meaning |
|----------|-------|---------|
| DATA_QUALITY | `"data_quality"` | Missing data, outliers, noise |
| SAMPLE_SIZE | `"sample_size"` | Too few observations |
| OBSERVATIONAL | `"observational"` | No causal design (default for most data) |
| MODEL_UNCERTAINTY | `"model_uncertainty"` | Model fit / prediction interval |
| TEMPORAL | `"temporal"` | Time-dependent assumptions |
| SCOPE | `"scope"` | Analysis scope limitation |
| METHODOLOGY | `"methodology"` | Methodological caveat |

### DecisionLimitation

A single, explicit limitation on an analysis or recommendation. Frozen (immutable) dataclass with:
- `id` — Unique identifier (prefix `lim_`)
- `category` — One of `LimitationCategory` values
- `description` — Clear explanation
- `impact` — Potential impact on interpretation
- `recommendation` — Suggested follow-up to address limitation

### DecisionEvidence

A single piece of classified evidence from any analysis engine. Frozen dataclass with:
- `id` — Unique identifier (prefix `ev_`)
- `level` — One of `EvidenceLevel` values
- `description` — Clear, human-readable description
- `source` — Originating analysis engine or module
- `metric_name` — Optional metric or column involved
- `metric_value` — Optional observed value (NaN/Inf sanitised to None)
- `details` — Supporting details dictionary
- `dataset_id` — Dataset/session scope
- `limitations` — List of `DecisionLimitation` objects

### DecisionSignal

A higher-level signal composed of one or more evidence items. Frozen dataclass with:
- `id` — Unique identifier (prefix `sig_`)
- `description` — Human-readable description
- `evidence_ids` — References to supporting `DecisionEvidence` IDs
- `level` — Evidence level (defaults to weakest evidence link)
- `affected_features` — Column names or entities involved
- `limitations` — Signal-level limitations
- `dataset_id` — Dataset/session scope

### ConfidenceScore

A transparent, deterministic confidence assessment. Frozen dataclass with:
- `level` — One of `ConfidenceLevel` values
- `explanation` — Human-readable explanation of why confidence is at this level
- `evidence_count` — Number of evidence items considered
- `evidence_quality` — Brief quality summary
- `reduction_reasons` — List of reasons confidence was reduced
- `dataset_id` — Dataset/session scope

## Evidence Safety Principles

The following rules are enforced deterministically:

1. **Correlation does not establish causation.** Inference-level evidence derived from observational data automatically receives an observational limitation via `ensure_non_causal()`.

2. **Association does not prove a cause.** Same enforcement as rule 1.

3. **A feature's importance is not automatically a causal effect.** Driver and relationship engine outputs are classified as INFERENCE, not FACT.

4. **A prediction is not an observed fact.** Predictions receive explicit uncertainty limitations via `ensure_prediction_uncertainty()`. Forecast, scenario, and simulation outputs are classified as PREDICTION.

5. **A recommendation is not a guaranteed outcome.** Recommendations are classified at their own evidence level, distinct from facts and inferences.

6. **A scenario simulation is not the actual future.** Scenario engine and simulator outputs are classified as PREDICTION with model uncertainty limitations.

7. **A forecast is not certain.** All forecasting engine outputs are classified as PREDICTION.

8. **Missing evidence must not be presented as evidence of absence.** `create_signal()` raises `ValueError` when given empty evidence. `assess_confidence()` returns INSUFFICIENT for empty evidence.

9. **Insufficient data produces INSUFFICIENT, not invented confidence.** The confidence assessment is deterministic and never fabricates confidence from insufficient evidence.

10. **Evidence from one dataset must not be silently attributed to another.** `create_signal()` raises `ValueError` when evidence items have conflicting `dataset_id` values. `validate_dataset_scope()` filters evidence to a specific dataset.

## Dataset Isolation

- Every evidence item, signal, and confidence score carries an optional `dataset_id`.
- `create_signal()` rejects evidence from different datasets.
- `validate_dataset_scope()` filters evidence to a specific dataset, allowing unscoped evidence.
- No global mutable caches. No demo-data fallback behaviour.

## Confidence Assessment Rules

Confidence is assessed deterministically using these rules (in order):

1. No evidence → INSUFFICIENT
2. All evidence is LIMITATION → INSUFFICIENT
3. >=3 positive items with >=1 FACT and no critical limitations and no reduction reasons → HIGH
4. >=3 positive items with >=1 FACT and no critical limitations → MEDIUM
5. >=2 positive items with >=1 FACT → MEDIUM
6. >=1 positive item → LOW
7. Otherwise → INSUFFICIENT

Confidence is reduced by:
- No directly observed facts (all inferences/predictions)
- Critical limitations (data quality, sample size)
- All evidence from a single source
- Predictions without corroborating facts
- Observational data only (correlation, not causation)

Aggregate confidence uses the **weakest-link rule**: the aggregate score equals the weakest individual score.

## Part 2 Scope: Signal Synthesizer

Part 2 converts structured analytical results from Phase 3–6 into domain-agnostic `DecisionSignal` instances using Part 1 contracts.

### Files Created

| File | Purpose |
|------|---------|
| `src/decision/signal_synthesizer.py` | Signal synthesis engine |
| `tests/test_phase7_part2_signal_synthesizer.py` | 104 comprehensive tests |

### Signal Synthesizer Design

The `SignalSynthesizer` class accepts a `dataset_id` and provides ingestion methods for each upstream Phase:

| Method | Phase | Input Types |
|--------|-------|-------------|
| `ingest_analytics()` | Phase 3 | KPIs, Insights, Relationships |
| `ingest_predictions()` | Phase 4 | Forecast, Regression, Classification, Segmentation, Feature Importance |
| `ingest_anomalies()` | Phase 5 | Anomalies, Drivers |
| `ingest_marketing()` | Phase 6 | MMM, Contributions, ROI, What-If, Optimization |
| `ingest_evidence()` | Any | Custom `DecisionEvidence` items for non-standard domains |

After ingestion, `synthesize()` produces a `SynthesisResult` containing:
- **Deduplicated signals** — duplicate signals (same description, source, features) are collapsed deterministically, keeping higher priority
- **Contradiction detection** — conflicting evidence directions on the same feature are surfaced, never silently discarded
- **Evidence registry** — all evidence items indexed by ID
- **Processing notes** — what was skipped and why

### Signal Priority

Priority is distinct from confidence:

| Priority | Value | Meaning |
|----------|-------|---------|
| CRITICAL | 1 | Requires immediate attention (critical anomalies, data quality) |
| HIGH | 2 | Important finding (forecasts, regressions, MMM) |
| MEDIUM | 3 | Notable finding (drivers, contributions, ROI, segmentation) |
| LOW | 4 | Minor finding (what-if scenarios, low-severity insights) |
| INFO | 5 | Background context (KPI observations) |

A high-priority signal may have low confidence (e.g. a critical anomaly with limited data). A low-priority signal may have high confidence (e.g. a well-established KPI).

### Evidence Level Mapping

| Upstream Output | Evidence Level | Safety Guard |
|-----------------|----------------|--------------|
| KPIs | FACT | None |
| Insights (correlation) | INFERENCE | `ensure_non_causal()` |
| Insights (other) | INFERENCE | None |
| Relationships | INFERENCE | `ensure_non_causal()` |
| Forecasts | PREDICTION | `ensure_prediction_uncertainty()` |
| Regression models | PREDICTION | `ensure_prediction_uncertainty()` |
| Classification models | PREDICTION | `ensure_prediction_uncertainty()` |
| Segmentation | PREDICTION | `ensure_prediction_uncertainty()` |
| Feature importance | INFERENCE | `ensure_non_causal()` |
| Anomalies | INFERENCE | None |
| Drivers | INFERENCE | `ensure_non_causal()` |
| MMM | INFERENCE | `ensure_non_causal()` |
| Channel contributions | INFERENCE | `ensure_non_causal()` |
| ROI estimates | INFERENCE | `ensure_non_causal()` |
| What-if scenarios | PREDICTION | `ensure_prediction_uncertainty()` |
| Budget optimization | PREDICTION | `ensure_prediction_uncertainty()` |

### SynthesizedSignal

Wraps a `DecisionSignal` with synthesis metadata:
- `signal` — The underlying `DecisionSignal`
- `priority` — Urgency of review (1=critical to 5=info)
- `confidence_score` — Deterministic `ConfidenceScore` from Part 1
- `source_type` — Which Phase/engine produced this signal
- `generation_reason` — Why this signal was generated
- `evidence_items` — The supporting `DecisionEvidence` items

### SynthesisResult

Aggregate output with:
- `dataset_id` — Dataset scope
- `signals` — Deduplicated `SynthesizedSignal` list
- `evidence_registry` — All evidence items by ID
- `contradictions` — Detected conflicts (surfaced, not discarded)
- `processing_notes` — Skipped/failed items and reasons
- `ranked_signals()` — Signals sorted by priority (most urgent first)
- `signals_by_source(source_type)` — Filter by originating phase
- `to_dict()` — Serialization compatible with Part 1 contracts

### Dataset Isolation in Part 2

- `SignalSynthesizer` requires a `dataset_id` at construction
- Each `ingest_*()` method validates the incoming result's `dataset_id` matches
- `ingest_evidence()` filters evidence by `validate_dataset_scope()`
- Cross-dataset evidence raises `ValueError`
- All generated signals and confidence scores carry the synthesizer's `dataset_id`

## Tests Executed

**Part 1 tests:** 130 passed, 0 failed, 0 skipped

Test categories covered:
- Regression-safe imports (all modules and public names)
- Every required enum value (EvidenceLevel, ConfidenceLevel, LimitationCategory)
- Contract validation and serialization (to_dict/from_dict roundtrips)
- Missing and malformed evidence handling
- Evidence-level distinctions (fact vs inference vs prediction vs recommendation vs limitation)
- Explicit limitations (creation, attachment, categories)
- Confidence levels including INSUFFICIENT
- Confidence explanations and reduction reasons
- Deterministic results verification
- Empty and insufficient evidence
- Dataset/session isolation and conflicting identifiers
- Correlation-versus-causation safeguards
- Prediction-versus-fact safeguards
- Recommendation-versus-guarantee safeguards
- Scenario-versus-future safeguards
- Generic non-marketing examples (healthcare, manufacturing, education, logistics, finance)
- Edge cases (dict-based evidence, mixed types, NaN/Inf sanitisation)
- ID generation and safe float helpers

**Part 2 tests:** 104 passed, 0 failed, 0 skipped

Test categories covered:
- Regression-safe imports (all modules and public names from Part 1 and Part 2)
- Signal priority ordering and from_severity mapping
- Valid signals from all supported evidence types (KPIs, insights, relationships, forecasts, regression, classification, segmentation, anomalies, drivers)
- Multiple analytical sources combined (all four phases)
- Empty and insufficient evidence (None inputs, empty lists, failed statuses)
- Contradictory findings detected and surfaced
- Deterministic signal generation, ranking, and to_dict serialization
- Duplicate-signal handling (deduplication keeps higher priority)
- Dataset isolation (mismatch rejection, unscoped acceptance, evidence filtering)
- Confidence and limitation preservation through synthesis
- Generic non-marketing datasets (healthcare, manufacturing, education, logistics, finance, IoT)
- Marketing-specific evidence (MMM, contributions, ROI, what-if, optimization, full pipeline)
- Missing or malformed inputs (missing fields, zero importance, empty channels)
- Serialization compatibility with Part 1 contracts (to_dict/from_dict roundtrips)
- Correlation-versus-causation safeguards on all observational evidence
- No demo-data leakage (no invented evidence, no hardcoded columns)
- Helper function tests (_get_attr, _convert_string_limitations)
- Evidence registry tracking and ID cross-references

**Full regression:** 977 passed, 0 failed, 0 skipped (873 baseline + 104 new)

## Part 3 Scope: Impact Quantifier

Part 3 provides a deterministic, domain-agnostic engine that assesses the potential impact of `DecisionSignal` objects produced by Part 2's `SignalSynthesizer`.

### Files Created

| File | Purpose |
|------|---------|
| `src/decision/impact_quantifier.py` | Impact quantification engine |
| `tests/test_phase7_part3_impact_quantifier.py` | 98 comprehensive tests |

### Impact Type

How the impact was determined:

| Type | Value | Meaning |
|------|-------|---------|
| MEASURED | `"measured"` | Directly computed from observed data (KPIs, aggregations) |
| MODEL_ESTIMATED | `"model_estimated"` | Estimated via statistical model (regression, classification, drivers) |
| SCENARIO_BASED | `"scenario_based"` | Derived from hypothetical scenario or optimization |
| QUALITATIVE | `"qualitative"` | Qualitative assessment without numeric estimate |
| UNKNOWN | `"unknown"` | Insufficient evidence to determine impact |

### Impact Category

Domain of the impact:

| Category | Value | Meaning |
|----------|-------|---------|
| PERFORMANCE | `"performance"` | Performance metrics (accuracy, throughput, KPIs) |
| EFFICIENCY | `"efficiency"` | Resource usage or optimization |
| RISK | `"risk"` | Risk exposure or anomaly severity |
| QUALITY | `"quality"` | Data or process quality |
| VOLUME | `"volume"` | Volume, count, or scale metrics |
| RELATIONSHIP | `"relationship"` | Feature relationships and associations |
| ANOMALY | `"anomaly"` | Anomalous behavior or outliers |
| GENERAL | `"general"` | General or cross-cutting impact |

### ImpactAssessment

A single impact assessment for one signal. Frozen (immutable) dataclass with:
- `id` — Unique identifier (prefix `imp_`)
- `signal_id` — ID of the assessed `DecisionSignal`
- `dataset_id` — Dataset scope
- `impact_type` — One of `ImpactType` values
- `impact_category` — One of `ImpactCategory` values
- `estimation_method` — How the estimate was derived (e.g. "direct_measurement", "model_coefficient")
- `estimated_value` — Numeric estimate if justified (NaN/Inf sanitised to None)
- `unit` — Unit of the estimate (original units preserved; never currency unless evidence provides it)
- `time_horizon` — Time scope if known
- `lower_bound` — Lower uncertainty bound, if available
- `upper_bound` — Upper uncertainty bound, if available
- `description` — Human-readable impact description
- `explanation` — How the impact was assessed
- `evidence_ids` — References to supporting evidence
- `confidence` — `ConfidenceScore` from Part 1
- `limitations` — Limitations on the assessment
- `assumptions` — Assumptions made during assessment

### ImpactReport

Aggregate output from quantifying multiple signals:
- `dataset_id` — Dataset scope
- `assessments` — List of `ImpactAssessment` objects
- `processing_notes` — Skipped/failed items and reasons
- `assessed_signals()` — Set of signal IDs that were assessed
- `by_type(impact_type)` — Filter assessments by impact type
- `by_category(category)` — Filter by impact category
- `quantified()` — Only assessments with a numeric `estimated_value`
- `to_dict()` / `from_dict()` — Serialization

### ImpactQuantifier Engine

The `ImpactQuantifier` class accepts a `dataset_id` and provides:
- `assess_signals(synthesized_signals, evidence_registry)` → `ImpactReport` — Main entry point, processes all signals
- `assess_single_signal(synthesized_signal, evidence_registry)` → `Optional[ImpactAssessment]` — Single signal assessment

Assessment is dispatched by `source_type` keyword matching:

| Source Type Keyword | Impact Type | Impact Category | Key Behaviour |
|---------------------|-------------|-----------------|---------------|
| kpi | MEASURED | PERFORMANCE | Direct measurement, uses observed metric value |
| anomaly | MEASURED | ANOMALY | Uses deviation magnitude if available |
| driver | MODEL_ESTIMATED | RELATIONSHIP | Adds non-causal limitation; no monetary inference |
| relationship | MODEL_ESTIMATED | RELATIONSHIP | Adds non-causal limitation |
| forecast | MODEL_ESTIMATED | PERFORMANCE | Uses error metric (RMSE/MAE), NOT the forecast value |
| regression, classification | MODEL_ESTIMATED | PERFORMANCE | Uses model fit metrics (R², accuracy) |
| segmentation | QUALITATIVE | GENERAL | Qualitative only — no numeric cluster-to-impact |
| whatif, scenario | SCENARIO_BASED | GENERAL | Adds hypothetical limitation |
| optimization | SCENARIO_BASED | EFFICIENCY | Adds hypothetical limitation |
| mmm | MODEL_ESTIMATED | RELATIONSHIP | Adds non-causal limitation; uses R² if available |
| contribution | MODEL_ESTIMATED | PERFORMANCE | Adds non-causal limitation |
| recommendation | QUALITATIVE | GENERAL | Qualitative only |
| (unmatched) | Mapped from evidence level | GENERAL | FACT→MEASURED, PREDICTION→SCENARIO_BASED, INFERENCE→MODEL_ESTIMATED, else QUALITATIVE |

### Quantification Rules

1. Impact is quantified **only** when evidence supports the calculation.
2. Original units are preserved; incompatible units are never combined.
3. Correlation is never presented as causation — all observational evidence receives an OBSERVATIONAL limitation.
4. Feature importance and model coefficients are not described as causal impact.
5. Forecast values are not presented as observed outcomes — error metrics (RMSE/MAE) are used instead.
6. Scenario results are not presented as actual future outcomes — explanations explicitly state "hypothetical".
7. No currency, percentages, savings, revenue, costs, or business benefits are invented.
8. No monetary impact is inferred when no monetary evidence exists.
9. Qualitative severity is never converted into arbitrary numeric financial estimates.
10. Insufficient evidence returns an explicit UNKNOWN type with INSUFFICIENT confidence.
11. Impact magnitude is kept separate from confidence and urgency.
12. Dataset isolation is enforced — evidence from another dataset is filtered out.
13. Conflicting evidence is handled by assessing each signal independently.
14. No global mutable caches or demo-data fallbacks.

### Dataset Isolation in Part 3

- `ImpactQuantifier` requires a `dataset_id` at construction
- `assess_single_signal()` validates the synthesized signal's dataset scope
- Evidence items are filtered by `validate_dataset_scope()` before assessment
- Mismatched-dataset signals are skipped with a processing note
- All generated assessments carry the quantifier's `dataset_id`

### Part 3 Tests

**Part 3 tests:** 98 passed, 0 failed, 0 skipped

Test categories covered:
- Regression-safe imports (all modules and public names from Parts 1–3)
- Valid quantitative impact (KPI measured, anomaly deviation, model R², report quantified filter)
- Qualitative-only impact (segmentation, recommendation)
- Insufficient evidence (no evidence, empty signals, dataset-filtered-out, no underlying signal)
- Missing values (None metric, empty details, missing source_type)
- NaN and infinity (NaN/Inf/neg-Inf metric values, NaN in anomaly details, NaN/Inf in serialization)
- Zero and negative values (zero metric, negative metric, zero expected anomaly)
- Incompatible units (unit preservation, driver generic unit without monetary inference)
- Dataset isolation (empty dataset_id raises, mismatch skipped, evidence filtered, report/assessment carry dataset_id, unscoped accepted)
- Conflicting evidence (conflicting signals still assessed, multiple signals independently assessed)
- Confidence separated from impact magnitude (high impact/low confidence, small impact/high confidence, separate fields)
- Forecast-based impact labeling (MODEL_ESTIMATED type, model_uncertainty limitation, explanation says not fact, uses error metric)
- Scenario-based impact labeling (what-if is SCENARIO_BASED, hypothetical limitation, optimization is SCENARIO_BASED/EFFICIENCY, explanation says not guaranteed)
- Correlation-versus-causation safeguards (driver/relationship/mmm/contribution all have OBSERVATIONAL limitation, driver explanation mentions not causal)
- No invented financial estimates (no currency in KPI/driver/scenario/unknown outputs)
- Generic non-marketing datasets (healthcare, manufacturing, education, logistics, finance, IoT)
- Marketing examples when supported by actual evidence (MMM with R², contribution with ROI, optimization scenario)
- Serialization (ImpactAssessment roundtrip, ImpactReport roundtrip, counts in to_dict, extra keys ignored, None confidence)
- Determinism (same inputs same output, report order deterministic)
- Malformed inputs (invalid type/category/description raise, exception captured in notes, non-numeric metric, dict-based _get_attr, nested _get_attr, from_dict with limitation dicts, evidence lookup from registry)
- Impact report helpers (by_type, by_category, assessed_signals, dataset_id property)
- Classification dispatch (classification model, insight→relationship, feature_importance→driver)
- Generic impact fallback (FACT→MEASURED, PREDICTION→SCENARIO_BASED, INFERENCE→MODEL_ESTIMATED, LIMITATION→QUALITATIVE)

**Full regression:** 1075 passed, 0 failed, 0 skipped (977 baseline + 98 new)

## Part 4 Scope: Domain-Agnostic Scenario Engine

Part 4 provides a deterministic, domain-agnostic engine for defining, validating, and comparing hypothetical scenarios using available evidence and supported analytical capabilities.

### Files Created

| File | Purpose |
|------|---------|
| `src/decision/scenario_engine.py` | Scenario definition, evaluation, and comparison engine |
| `tests/test_phase7_part4_scenario_engine.py` | 86 comprehensive tests |

### Scenario Status

How the scenario evaluation was resolved:

| Status | Value | Meaning |
|--------|-------|---------|
| SIMULATED | `"simulated"` | Fully evaluated using a defensible model with all parameters |
| PARTIALLY_SIMULATED | `"partially_simulated"` | Some parameters evaluated, others lacked model support |
| NOT_SIMULATABLE | `"not_simulatable"` | Evidence exists but no defensible calculation method available |
| INSUFFICIENT_EVIDENCE | `"insufficient_evidence"` | Not enough evidence or parameters to attempt evaluation |

### Core Contracts

| Contract | Purpose |
|----------|---------|
| `ScenarioParameter` | Captures baseline/alternative values, unit, support status, and change percentage |
| `ScenarioDefinition` | Scenario specification with parameters, assumptions, evidence IDs, and limitations |
| `ScenarioResult` | Evaluation outcome with status, estimated/baseline outcomes, delta, confidence, and missing requirements |
| `ScenarioComparison` | Compares two or more scenario results with notes, unit compatibility checks, and limitations |

All contracts are frozen dataclasses with `to_dict()` / `from_dict()` serialization.

### ScenarioEngine API

- `ScenarioEngine(dataset_id)` — Requires a non-empty dataset_id; all operations are scoped to this dataset
- `define_scenario(name, description, parameters, evidence_items, assumptions)` → `ScenarioDefinition` — Validates parameters, filters evidence by dataset, creates limitations for unsupported parameters
- `evaluate_scenario(scenario, evidence_items, model_info)` → `ScenarioResult` — Evaluates a scenario against available evidence and model information
- `compare_scenarios(results)` → `ScenarioComparison` — Compares two or more scenario results

### Evaluation Paths

| Condition | Result Status | Behaviour |
|-----------|---------------|-----------|
| Model with coefficients for all supported params | SIMULATED | Computes baseline and alternative outcomes using model |
| Model with coefficients for some params only | PARTIALLY_SIMULATED | Evaluates known params, lists unevaluable ones |
| Evidence exists but no model provided | NOT_SIMULATABLE | Returns explanation with missing requirements |
| Model provided but empty coefficients | NOT_SIMULATABLE | Returns explanation that coefficients are missing |
| No evidence and no model | INSUFFICIENT_EVIDENCE | Returns explanation of what is needed |
| No parameters defined | INSUFFICIENT_EVIDENCE | Returns explanation that parameters are required |
| All parameters unsupported | NOT_SIMULATABLE | Returns list of unsupported parameters |
| Dataset mismatch | INSUFFICIENT_EVIDENCE | Rejects cross-dataset evaluation |

### Supported Scenario Types

The engine is domain-agnostic and supports any scenario where:

1. Parameters have explicit baseline and alternative values
2. A defensible model (with coefficients) is externally provided
3. Evidence is scoped to the correct dataset

No formulas are invented. If no `model_info` is provided or coefficients are missing, the engine returns NOT_SIMULATABLE or INSUFFICIENT_EVIDENCE with a clear explanation of what is needed.

### Validation Rules

1. Parameter names must be non-empty strings
2. NaN/Inf values in baselines or alternatives are sanitized to None (parameter marked unsupported)
3. NaN/Inf coefficients in model_info cause the parameter to be unevaluable
4. Change percentage is computed only when both baseline and alternative are finite and baseline is non-zero
5. Out-of-range parameter values generate SCOPE limitations but evaluation still proceeds (with warning)
6. Unit compatibility is checked during scenario comparison
7. Cross-dataset evidence is filtered out during definition and evaluation
8. Comparison requires at least two results

### Safety Guarantees

1. A scenario is not an observed fact — all results explicitly state this
2. A simulation is not a guaranteed future outcome — no result uses "guaranteed"
3. Correlation does not establish causation
4. Model coefficients are not automatically causal effects
5. Unsupported assumptions are disclosed in the result
6. No financial benefits, savings, ROI, or currency are invented
7. Incompatible units are never silently combined
8. Out-of-range extrapolation generates explicit warnings
9. Confidence is kept separate from impact magnitude
10. Contradictory evidence is preserved
11. Dataset isolation is enforced at every step
12. No global mutable caches or demo-data fallbacks
13. No LLM calls for scenario calculations

### Domain-Agnostic Design

The engine contains no hardcoded references to:
- Revenue, Sales, Date
- Google Ads, Facebook Ads
- NovaMart or any specific company
- Any particular currency
- Any particular industry or dataset schema

Generic examples tested: healthcare (patient outcomes), manufacturing (defect rates), education (completion rates), logistics (delivery times), IoT (sensor readings).

### Dataset Isolation in Part 4

- `ScenarioEngine` requires a `dataset_id` at construction
- `define_scenario()` filters evidence by `validate_dataset_scope()`
- `evaluate_scenario()` validates scenario dataset matches engine dataset
- `ScenarioResult` carries the engine's `dataset_id`
- `ScenarioComparison` rejects results from different datasets

### Part 4 Tests

**Part 4 tests:** 86 passed, 0 failed, 0 skipped

Test categories covered:
- Regression-safe imports (all modules and public names from Parts 1–4)
- Scenario contract validation (parameter, definition, result, comparison, engine construction)
- Valid supported numerical simulations (single param, multi-param, delta_pct)
- Missing assumptions (assumptions carried, model assumptions added)
- Insufficient evidence (no evidence/no model, no parameters, all evidence filtered)
- Unsupported calculations (evidence no model, no matching coefficients, empty coefficients, all params unsupported)
- Baseline-versus-alternative comparisons (separate outcomes, change_pct verification)
- Multiple scenarios (compare two, compare mixed status)
- Deterministic results (same inputs same output, comparison deterministic)
- Invalid and non-finite values (NaN baseline, Inf alternative, NaN coefficient, serialization NaN/Inf)
- Unit compatibility (unit preserved, incompatible units flagged, parameter unit preserved)
- Out-of-range changes (out-of-range warning generated, within-range no warning)
- Contradictory evidence (both evidence items preserved)
- Dataset isolation (engine requires dataset_id, definition requires dataset_id, mismatched scenario, cross-dataset filtered, result carries dataset_id, comparison rejects cross-dataset, unscoped accepted)
- Serialization (parameter/definition/result/comparison roundtrips, extra keys ignored, None confidence)
- Forecast and model-estimate labeling (estimation method mentions model, R² in explanation, model_uncertainty limitation)
- Scenario-versus-fact safeguards (explanation says not observed, limitations mention not observed, NOT_SIMULATABLE explains missing)
- No guaranteed-outcome claims (no "guaranteed" in simulated description, comparison disclaims guarantees)
- No invented financial estimates (no currency in result, no fabricated business benefit)
- Generic non-marketing examples (healthcare, manufacturing, education, logistics, IoT)

**Full regression:** 1161 passed, 0 failed, 0 skipped (1075 baseline + 86 new)

## Part 5: Decision Generator

Part 5 implements a deterministic, domain-agnostic Decision Generator that converts evidence, synthesized signals, impact assessments, and scenario results into structured decision options and actionable recommendations.

### Files Created

| File | Purpose |
|------|---------|
| `src/decision/decision_generator.py` | Decision generation engine and contracts |
| `tests/test_phase7_part5_decision_generator.py` | 97 comprehensive tests |

### Decision Contracts

**Enums:**

| Enum | Values | Purpose |
|------|--------|---------|
| `DecisionCategory` | RECOMMEND, INVESTIGATE, MONITOR, MITIGATE_RISK, BLOCKED | Decision action type |
| `PriorityLevel` | CRITICAL, HIGH, MEDIUM, LOW | Composite priority ranking |
| `UrgencyLevel` | IMMEDIATE, SHORT_TERM, MEDIUM_TERM, LONG_TERM, NOT_ASSESSED | Time-sensitivity |
| `Reversibility` | EASILY_REVERSIBLE, PARTIALLY_REVERSIBLE, IRREVERSIBLE, UNKNOWN | Risk of action |

**DecisionOption** — frozen dataclass with 28 fields:
- Identity: `id` (prefixed `dec_`), `dataset_id`, `title`, `description`
- Category and action: `category`, `recommended_action`
- Traceability: `signal_ids`, `evidence_ids`, `impact_ids`, `scenario_ids`
- Impact: `estimated_impact`, `impact_type`, `impact_value`, `impact_unit`
- Assessment: `confidence`, `priority`, `priority_rationale`, `urgency`
- Context: `assumptions`, `limitations`, `risks`, `reversibility`
- Follow-up: `required_follow_up`, `invalidation_conditions`
- Explanation: `explanation`, `evidence_basis`, `contradictions`
- Serialization: `to_dict()` / `from_dict()` with NaN/Inf-safe floats

**DecisionReport** — frozen dataclass aggregating decisions with:
- Filtering: `by_category()`, `by_priority()`, `recommended()`, `investigations()`, `monitoring()`, `risk_mitigations()`, `blocked()`
- Ranking: `ranked()` returns decisions in deterministic priority order
- Serialization: `to_dict()` / `from_dict()`

### Decision Generation Rules

1. Decisions generated only from available, relevant, dataset-scoped evidence
2. No invented evidence, numerical benefits, financial returns, or business facts
3. Measured outcomes distinguished from model estimates and hypothetical scenarios
4. Recommendations never presented as guaranteed results
5. Correlation never converted into causation
6. Model coefficients or feature importance never interpreted as causal impact
7. Confidence, impact magnitude, urgency, and priority kept conceptually separate
8. Basis explained for every recommendation
9. Contradictory evidence and material limitations surfaced
10. Insufficient evidence produces investigation or monitoring, not unsupported action
11. Irreversible or high-impact recommendations avoided when evidence is weak
12. Actions never claimed to solve problems unless evidence supports it
13. All recommendations domain-agnostic
14. Evidence from different datasets never silently mixed
15. No LLM used to invent or rank recommendations
16. No global mutable caches or demo-data fallback behavior

### Category Determination

| Condition | Category |
|-----------|----------|
| No evidence or insufficient confidence | BLOCKED |
| Anomaly source type | INVESTIGATE |
| Forecast or scenario source type | MONITOR |
| Risk source type | MITIGATE_RISK |
| Fact-level evidence + high/medium confidence | RECOMMEND |
| Model-based source type | MONITOR |
| Low confidence | INVESTIGATE |
| Default (err on side of caution) | INVESTIGATE |

### Prioritization Logic

Priority is a composite of four independent dimensions, each scored separately:

1. **Evidence score** — based on evidence level (fact > observation > inference > assumption)
2. **Confidence score** — from confidence assessment level
3. **Signal priority score** — from Part 2 signal priority (CRITICAL through INFO)
4. **Category score** — category-specific weight (RECOMMEND > MONITOR > INVESTIGATE > MITIGATE_RISK > BLOCKED)

The composite score determines the `PriorityLevel`. Decisions are ranked by: priority level → category weight → confidence level → evidence count → ID (deterministic tie-break).

The prioritization rationale is exposed in every decision's `priority_rationale` field.

### Limitations

- Decisions are only as strong as the upstream evidence quality
- No feasibility scores are manufactured when feasibility data is unavailable
- When evidence cannot support a defensible ranking, the limitation is reported rather than invented precision
- Model-based evidence produces MONITOR, not RECOMMEND decisions

### Test Coverage (97 tests)

Tests cover all 19 required categories:
- Decisions from valid evidence
- Investigation and monitoring recommendations
- Insufficient evidence handling
- Contradictory evidence surfacing
- Traceability to supporting signals and evidence
- Impact and confidence preservation
- Transparent prioritization with exposed rationale
- Deterministic generation and ordering
- Dataset isolation enforcement
- Malformed and incomplete input handling
- Full serialization round-trip
- Forecast-based recommendations
- Scenario-based recommendations
- Correlation-versus-causation safeguards
- No guaranteed-outcome claims
- No invented financial benefits
- Risk and reversibility handling
- Generic non-marketing datasets (healthcare, education, manufacturing)
- Regression-safe imports

**Full regression:** 1258 passed, 0 failed, 0 skipped (1161 baseline + 97 new)

## Part 6: Decision Orchestrator

Part 6 introduces a stateless **Decision Orchestrator** that coordinates all Parts 1–5 components into a single, deterministic end-to-end workflow.

### Files Created / Modified

| File | Purpose |
|------|---------|
| `src/decision/orchestrator.py` | Orchestrator implementation |
| `tests/test_phase7_part6_orchestrator.py` | 79 comprehensive tests |
| `src/decision/__init__.py` | Updated docstring |

### Request and Result Contracts

**`OrchestrationRequest`** (frozen dataclass):
- `dataset_id` — required scope for the entire workflow
- `evidence_items` — pre-built `DecisionEvidence` objects
- `analytics_result`, `prediction_report`, `phase5_result`, `marketing_result` — optional upstream results
- `scenario_definitions`, `scenario_model_info` — optional scenario inputs
- `evaluate_scenarios` — boolean flag (default `False`)

**`OrchestrationReport`** (frozen dataclass):
- `workflow_id` — unique identifier (prefixed `wf_`)
- `dataset_id` — echoes the request scope
- `status` — `ExecutionStatus` enum: SUCCESS, PARTIAL_SUCCESS, INSUFFICIENT_EVIDENCE, FAILED
- `evidence_summary` — counts by level, sources, confidence
- `synthesis_result`, `impact_report`, `scenario_results`, `decision_report` — stage outputs
- `stages` — ordered list of `StageResult` with per-stage status
- `warnings`, `limitations`, `errors`, `processing_notes` — traceability
- `to_dict()` / `from_dict()` — safe serialization with NaN/Infinity handling

### Orchestration Workflow (7 Stages)

1. **Validate Request** — checks dataset_id validity, rejects cross-dataset component results
2. **Classify Evidence** — filters evidence by dataset scope, applies safety guards (non-causal, prediction uncertainty)
3. **Assess Confidence** — calls `assess_confidence()` on validated evidence
4. **Synthesize Signals** — creates `SignalSynthesizer`, ingests all available upstream results, calls `synthesize()`
5. **Quantify Impacts** — creates `ImpactQuantifier`, calls `assess_signals()` with synthesis output
6. **Evaluate Scenarios** — optional; only runs when `evaluate_scenarios=True` and definitions are provided
7. **Generate Decisions** — creates `DecisionGenerator`, calls `generate()` with all available stage outputs

### Safety and Reliability

- **Dataset isolation**: every workflow scoped to a single dataset; cross-dataset evidence filtered with warnings; cross-dataset component results rejected immediately
- **Stateless**: no instance or class state persists between `run()` calls
- **Deterministic**: no LLM, no randomness, no network, no global mutable cache
- **Error handling**: all stage exceptions caught and recorded in errors list; stage status set to FAILED; no silently swallowed exceptions
- **No fabrication**: failed stages produce `None` outputs, never fake success
- **No demo-data fallback**: empty evidence yields INSUFFICIENT_EVIDENCE, not synthetic data

### Test Coverage (79 tests)

All 25 required test categories covered:

- Successful end-to-end orchestration
- Empty workflow and empty evidence
- Insufficient evidence
- Correct stage execution order
- Optional scenario evaluation
- Unsupported scenario evaluation
- Dataset-scope validation
- Cross-dataset evidence rejection
- Cross-dataset component-result rejection
- Two datasets processed sequentially without leakage
- Recovery after a failed workflow
- Stage-specific failures
- Partial-success reporting
- Complete-failure reporting
- Warning and limitation propagation
- Evidence provenance and decision traceability
- Deterministic decision ordering
- Safe serialization and deserialization
- NaN/Infinity handling
- No demo-data fallback
- No fabricated successful outputs
- No silently swallowed exceptions
- Compatibility with Parts 1–5 public contracts
- No global mutable state
- Invalid request handling

**Full regression:** 1337 passed, 0 failed, 0 skipped (1258 baseline + 79 new)

## Part 7: Dashboard + AI Copilot Integration

Part 7 makes the Decision Intelligence engine accessible through the existing Streamlit dashboard and AI Copilot.

### Files Created

| File | Purpose |
|------|---------|
| `src/decision/dashboard_service.py` | Stateless service layer bridging orchestrator to dashboard/copilot |
| `dashboard/pages/13_decision_intelligence.py` | Streamlit dashboard page for decision results |
| `tests/test_phase7_part7_dashboard_copilot.py` | 75 tests covering all 24 required categories |

### Files Modified

| File | Change |
|------|--------|
| `app.py` | Added Decision Intelligence page to PAGES navigation dict |
| `dashboard/components/session_paths.py` | Added `decision_result`, `decision_copilot_context` to `_DATASET_DEPENDENT_KEYS` |
| `src/ai/analyst.py` | Added `decision_context`/`dataset_id` parameters and decision question routing |
| `dashboard/pages/8_ai_marketing_analyst.py` | Passes decision context and dataset ID to `MarketingAnalyst` |
| `src/decision/__init__.py` | Updated docstring to include Part 7 |

### Dashboard Service Layer (`dashboard_service.py`)

A pure-Python module (no Streamlit dependency) that provides:

- **`run_decision_workflow(dataset_id, ...)`** — Creates a fresh `DecisionOrchestrator` per call, runs the full pipeline, returns a `DashboardResult`.
- **`format_dashboard_result(report)`** — Converts `OrchestrationReport` into structured display data with status labels, confidence info, decision cards, stage displays, and empty-state explanations.
- **`build_copilot_context(result, dataset_id)`** — Builds a sanitized context dict for the AI Copilot, validating dataset match. Returns empty context on mismatch.
- **`format_copilot_answer_context(context)`** — Renders context into natural-language text for the analyst.

### Decision Intelligence Dashboard Page

The page follows existing patterns:
- Uses `@safe_page()` decorator for graceful degradation
- Results cached in `st.session_state["decision_result"]`
- Dataset-scoped: verifies dataset match before displaying cached results
- Explicit "Run Analysis" button — no auto-run on every rerender
- Clear separation of confidence, impact, urgency, priority, and evidence strength
- Honest empty states explaining why no decisions are available
- Pipeline stage details in expandable section

### AI Copilot Integration

- `MarketingAnalyst` accepts optional `decision_context` and `dataset_id`
- Context is validated for dataset match at construction time (mismatched context silently discarded)
- Decision questions detected via keyword matching and routed to `decision_intelligence_answer()`
- Non-decision questions continue to use existing handlers unchanged
- When no decision context is available, the copilot says so honestly

### Dataset Isolation

- `decision_result` and `decision_copilot_context` added to `_DATASET_DEPENDENT_KEYS`
- Cleared automatically on `set_active_dataset()`, `reset_to_demo_dataset()`, `reset_analysis()`
- Dashboard page validates `result.dataset_id` against current `get_dataset_id()` before display
- `build_copilot_context()` rejects results from a different dataset
- `MarketingAnalyst` rejects context where `dataset_id` doesn't match
- Each `run_decision_workflow()` call creates a fresh orchestrator — no state leaks

### Test Coverage (75 tests)

All 24 specification categories are covered:
1. Dashboard rendering (3 tests)
2. Orchestrator integration (2 tests)
3. Empty results (2 tests)
4. Insufficient evidence (2 tests)
5. Partial success (2 tests)
6. Failed workflow (2 tests)
7. Evidence provenance (4 tests)
8. Dimension separation (3 tests)
9. Epistemic distinction (5 tests)
10. Dataset switching (3 tests)
11. Copilot isolation (3 tests)
12. Copilot active dataset (2 tests)
13. Missing copilot capability (3 tests)
14. Unsupported questions (2 tests)
15. Missing scenarios (2 tests)
16. Warnings and errors (5 tests)
17. No demo fallback (2 tests)
18. No fabricated recommendations (3 tests)
19. No secrets exposed (2 tests)
20. Dashboard compatibility (3 tests)
21. Invalid dataset handling (4 tests)
22. Cache behavior (2 tests)
23. Deterministic presentation (3 tests)
24. Regression boundaries (12 tests)

### Known Limitations

- The dashboard page requires Streamlit and cannot be tested with a live UI in automated tests. Tests cover the service layer, context construction, and module integration.
- Copilot decision routing uses keyword matching — complex natural-language questions may not route correctly.
- Scenario results display is minimal; a future part could expand it.
- The dashboard page does not auto-refresh — users must click "Run Analysis" or "Re-run Analysis".

**Full regression:** 1412 passed, 0 failed, 0 skipped (1337 baseline + 75 new)

## Part 8: Pipeline Integration, Capability Engine and API Endpoints

**Status: implemented.** (This section previously read "Not Started"; that
statement was outdated and was corrected during final release stabilization
after checking the code and tests below.)

### What Part 8 delivers

- **Pipeline integration** — `src/data_engine/analysis_runner.py::run_full_analysis`
  runs the `DecisionOrchestrator` after the Phase 3–6 stages and persists the
  report to `data/uploads/<dataset_id>/processed/phase7_decisions.json`.
- **Stale-result protection** — any previous `phase7_decisions.json` is deleted
  before a run, so a failure can never leave an old report looking current.
- **Non-fatal failure** — an orchestration error is recorded in the pipeline
  report and a `{"status": "failed"}` marker is persisted, so consumers can tell
  "failed" from "never run". Upstream results are unaffected.
- **Capability engine** — `decision_intelligence` capability in
  `src/data_engine/capability_engine.py`, available only when upstream stages can
  supply evidence.
- **API endpoints** (`api/main.py`), both protected by the API key:
  - `GET /datasets/{dataset_id}/decisions` — returns the persisted report;
    404 if the dataset is unconfirmed or no report exists.
  - `POST /datasets/{dataset_id}/decisions/run` — re-runs the orchestrator from
    persisted Phase 3–6 results without recomputing them.
  - Dataset IDs are validated against `DATASET_ID_PATTERN` (path-traversal safe).

### Tests

`tests/test_phase7_part8_integration.py` — 56 test functions across 21 classes:
pipeline integration, report persistence, non-fatal failure, stale-result
invalidation, capability engine, GET/POST endpoint behaviour, authentication and
dataset access, missing/unconfirmed datasets, corrupt upstream results, dataset
isolation, non-marketing and empty datasets, malformed inputs, failure recovery,
no demo-data fallback, no fabricated financials, no causal claims, safe error
responses, performance and backwards compatibility. All pass in the final
release run (see `FINAL_TESTING_REPORT.md`).

### Known limitations

- Decision reports are stored as JSON files per dataset; there is no history of
  earlier runs (each run replaces the previous report).
- The endpoints read persisted upstream results; if those are missing, the
  orchestrator reports insufficient evidence rather than recomputing them.
