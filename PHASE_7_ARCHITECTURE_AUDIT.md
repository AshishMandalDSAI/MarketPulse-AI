# Phase 7 Architecture Audit Report
## MarketPulse AI → Universal Decision Intelligence Platform

**Date:** 2026-10-06
**Scope:** Read-only pre-implementation audit of Phase 0–6
**Phase 7 Code Written:** NONE — this is analysis only

---

## 1. Current Architecture

### Project Structure

```
MarketPulse-AI/
├── src/
│   ├── data_engine/        # Phase 1-2: Universal Data Engine
│   │   ├── schemas.py          FieldGuess, ColumnMapping, StandardizedDataset
│   │   ├── canonical.py        CanonicalDataset, AnalysisContext, SemanticRole
│   │   ├── pipeline.py         profile_and_map → confirm_and_standardize
│   │   ├── capability_engine.py  12 capability gates
│   │   ├── column_classifier.py  3-signal semantic classification
│   │   ├── data_standardizer.py  Role-to-column standardization
│   │   ├── profiler.py          Per-column type/stats profiling
│   │   └── analysis_runner.py   Master orchestrator (Phases 1-6)
│   ├── analytics/          # Phase 3: Dynamic Analytics
│   │   ├── contracts.py        KPI, ChartRecommendation, Insight, Relationship
│   │   ├── kpi_engine.py       Dynamic KPI discovery
│   │   ├── insight_engine.py   6 insight detectors
│   │   ├── relationship_engine.py  3 relationship detectors
│   │   └── visualization_engine.py  8 chart recommendation rules
│   ├── prediction/         # Phase 4: Predictive Intelligence
│   │   ├── contracts.py        PredictiveTask, ForecastResult, RegressionResult, etc.
│   │   ├── task_detector.py    4 task detectors with confidence scoring
│   │   ├── forecasting_engine.py  5 model candidates (Naive→ExpSmooth)
│   │   ├── regression_engine.py   3 models (Ridge/RF/GBM)
│   │   ├── classification_engine.py  3 models (LogReg/RF/GBM)
│   │   └── segmentation_engine.py   KMeans with auto-K
│   ├── anomaly/            # Phase 5: Anomaly & Driver Intelligence
│   │   ├── contracts.py        Anomaly, AnomalyReport, DriverResult, DriverReport
│   │   ├── detector.py         Orchestrator for 3 sub-detectors
│   │   ├── numeric.py          IQR + Z-score + MAD
│   │   ├── categorical.py      Rare category + unusual metric + missingness
│   │   ├── time_series.py      Rolling deviation + spike/drop + trend residual
│   │   └── driver_engine.py    4 driver methods (correlation/MI/model/group)
│   ├── marketing/          # Phase 6: Marketing Intelligence
│   │   ├── contracts.py        MarketingReadiness, MMMResult, Phase6Result, etc.
│   │   ├── field_discovery.py  3-signal channel detection + 8-rule validation
│   │   ├── mmm_engine.py       Ridge MMM with adstock + Hill saturation
│   │   ├── contribution.py     Leave-one-out ablation
│   │   ├── roi.py              Naive + model-based ROI
│   │   ├── optimizer.py        SLSQP budget optimizer
│   │   ├── simulator.py        What-if scenario engine
│   │   └── orchestrator.py     Phase 6 pipeline orchestrator
│   ├── data/               # Legacy: synthetic data generation + cleaning
│   ├── features/           # Legacy: feature engineering
│   ├── models/             # Legacy: MMM models (adstock, saturation, Ridge)
│   ├── optimization/       # Legacy: budget optimizer + simulator
│   ├── forecasting/        # Legacy: trend forecasting
│   ├── ai/                 # AI analyst (rule-based + optional LLM)
│   ├── kpi/                # Generic KPIs from StandardizedDataset
│   └── utils/              # Database utility
├── dashboard/              # Streamlit UI
│   ├── pages/              # 13 dashboard pages (0-12)
│   ├── components/         # data_loader, session_paths, graceful, kpi_card
│   └── styles/             # theme
├── api/                    # FastAPI REST API
├── tests/                  # 743 tests across 17 test files
├── config/                 # settings, constants, column_aliases
├── data/                   # raw, processed, uploads (per-dataset isolation)
├── docs/                   # Phase reports, architecture, contracts
└── models/                 # Fitted model artifacts
```

### Architecture Layers (Bottom-Up)

| Layer | Phase | Module | Purpose |
|-------|-------|--------|---------|
| Data Ingestion | 0 | `src/data/` | Legacy synthetic data + cleaning |
| Universal Data Engine | 1-2 | `src/data_engine/` | Dataset-agnostic profiling, classification, canonicalization |
| Dynamic Analytics | 3 | `src/analytics/` | KPI discovery, insights, relationships, chart recommendations |
| Predictive Intelligence | 4 | `src/prediction/` | Forecasting, regression, classification, segmentation |
| Anomaly & Driver | 5 | `src/anomaly/` | Anomaly detection, root-cause driver analysis |
| Marketing Intelligence | 6 | `src/marketing/` | MMM, contribution, ROI, optimization, simulation |
| Presentation | — | `dashboard/`, `api/` | Streamlit + FastAPI consumers |

---

## 2. Current Pipeline

### Execution Flow (analysis_runner.py)

```
Upload (CSV/XLSX)
  │
  ▼
profile_and_map(df)
  ├── profiler.profile_dataset() → per-column type, stats, cardinality
  └── column_classifier.classify_columns() → ColumnMapping with confidence scores
  │
  ▼
confirm_and_standardize(df, mapping, confirmed_roles, overrides)
  ├── Apply user overrides (force date, revenue, channels)
  ├── Clear unconfirmed low-confidence guesses
  └── Build StandardizedDataset (date/revenue/channels/controls/dimensions/identifiers)
  │
  ▼
detect_capabilities(standardized)
  └── 12 capability gates → CapabilityReport
  │
  ▼
Persist: standardized_data.csv, generic_kpis.json, capabilities.json
  │
  ▼
Conditional: Generic MMM (Ridge) if marketing_mix_modeling available
Conditional: Generic Forecast (Ridge) if forecasting available
Conditional: ROI/ROAS if roi_roas available
  │
  ▼
Phase 2: build_canonical() → CanonicalDataset + AnalysisContext
  ├── ColumnMetadata per column
  ├── SemanticRole assignment
  ├── TargetCandidate detection
  ├── TimeInfo detection
  └── DataQuality computation
  │
  ▼
Phase 3: Dynamic Analytics
  ├── KPI discovery (7 stages, max 8 primary)
  ├── Chart recommendations (8 rules, max 12)
  ├── Insight discovery (6 detectors, max 15)
  └── Relationship analysis (3 detectors, max 15)
  │
  ▼
Phase 4: Predictive Intelligence
  ├── Task detection → list[PredictiveTask] with confidence
  ├── Forecasting (if detected + time + numeric target)
  ├── Regression (if detected + numeric target + features)
  ├── Classification (if detected + categorical target)
  └── Segmentation (if detected + numeric features, no target needed)
  │
  ▼
Phase 5: Anomaly & Driver Intelligence
  ├── Anomaly detection: numeric (IQR/Z/MAD) + time-series + categorical
  └── Driver analysis: correlation + MI + model-based + group comparison
  │
  ▼
Phase 6: Marketing Intelligence
  ├── Field discovery → MarketingCapabilityReport
  ├── MMM fitting (adstock + saturation + Ridge, 3 model variants)
  ├── Channel contributions (ablation)
  ├── ROI/ROAS (naive + model-based)
  ├── Budget optimization (SLSQP)
  └── What-if simulation (4 standard scenarios)
  │
  ▼
Persist all results to data/uploads/<dataset_id>/processed/
```

### Entry Points

1. **Streamlit upload page** → `run_full_analysis()` → full pipeline
2. **API `/upload` + `/datasets/{id}/confirm`** → `run_full_analysis()` → full pipeline
3. **Legacy demo** → pre-computed `data/processed/` files → legacy endpoints

---

## 3. Existing Reusable Engines

| Engine | Location | Reusable? | Notes |
|--------|----------|-----------|-------|
| KPI Engine | `src/analytics/kpi_engine.py` | ✅ Directly | Dataset-agnostic, accepts AnalysisContext |
| Insight Engine | `src/analytics/insight_engine.py` | ✅ Directly | 6 generic detectors |
| Relationship Engine | `src/analytics/relationship_engine.py` | ✅ Directly | Correlation/group/trend |
| Visualization Engine | `src/analytics/visualization_engine.py` | ✅ Directly | Rule-based chart recommendations |
| Forecasting Engine | `src/prediction/forecasting_engine.py` | ✅ Directly | 5-model competition |
| Regression Engine | `src/prediction/regression_engine.py` | ✅ Directly | 3-model competition |
| Classification Engine | `src/prediction/classification_engine.py` | ✅ Directly | 3-model competition |
| Segmentation Engine | `src/prediction/segmentation_engine.py` | ✅ Directly | KMeans auto-K |
| Task Detector | `src/prediction/task_detector.py` | ✅ Directly | Confidence-scored task detection |
| Anomaly Detector | `src/anomaly/detector.py` | ✅ Directly | 3 sub-detector orchestrator |
| Driver Engine | `src/anomaly/driver_engine.py` | ✅ Directly | 4-method driver analysis |
| Marketing Orchestrator | `src/marketing/orchestrator.py` | ✅ Directly | Capability-gated pipeline |
| MMM Engine | `src/marketing/mmm_engine.py` | ✅ Directly | Dataset-agnostic MMM |
| Budget Optimizer | `src/marketing/optimizer.py` | ✅ Directly | SLSQP on MMM surface |
| What-If Simulator | `src/marketing/simulator.py` | ✅ Directly | Scenario comparison |
| CanonicalDataset | `src/data_engine/canonical.py` | ✅ Directly | Universal data contract |
| Capability Engine | `src/data_engine/capability_engine.py` | ✅ Directly | 12-gate detection |
| Dataset Isolation | `config/settings.py` + `session_paths.py` | ✅ Directly | Per-dataset directory isolation |

**All Phase 1-6 engines are reusable.** They accept `CanonicalDataset` / `AnalysisContext` as input and return typed contract objects. No rewrite is needed.

---

## 4. Existing Contracts

### Data Contracts (Phase 1-2)

| Contract | Module | Purpose |
|----------|--------|---------|
| `FieldGuess` | schemas.py | Single column→role mapping with confidence |
| `ColumnMapping` | schemas.py | Full column-to-role map |
| `StandardizedDataset` | schemas.py | Post-mapping dataset with typed columns |
| `SemanticRole` | canonical.py | Enum: TIME, TARGET, MEASURE, DIMENSION, etc. |
| `ColumnMetadata` | canonical.py | Per-column stats + semantic type |
| `DataQuality` | canonical.py | Composite quality score + warnings |
| `TargetCandidate` | canonical.py | Scored target column candidate |
| `TimeInfo` | canonical.py | Time column + granularity + regularity |
| `CanonicalDataset` | canonical.py | **Universal internal representation** |
| `AnalysisContext` | canonical.py | CanonicalDataset + capabilities + roles |
| `Capability` | capability_engine.py | Per-analysis gate: available + reason |
| `CapabilityReport` | capability_engine.py | All capability gates for a dataset |

### Analytics Contracts (Phase 3)

| Contract | Purpose |
|----------|---------|
| `KPI` | Discovered metric with value, unit, category, priority |
| `ChartRecommendation` | Recommended visualization with score |
| `Insight` | Detected pattern with type, severity, confidence |
| `Relationship` | Detected correlation/trend/group comparison |
| `AnalyticsResult` | Aggregate container |

### Prediction Contracts (Phase 4)

| Contract | Purpose |
|----------|---------|
| `PredictiveTask` | Detected task type with confidence + requirements |
| `ForecastResult` | Time-series forecast with intervals |
| `RegressionResult` | Regression model results + feature importance |
| `ClassificationResult` | Classification + confusion matrix |
| `SegmentationResult` | Cluster profiles + quality metrics |
| `PredictionReport` | Aggregate container |

### Anomaly/Driver Contracts (Phase 5)

| Contract | Purpose |
|----------|---------|
| `Anomaly` | Single anomaly with type, severity, score, explanation |
| `AnomalyReport` | All anomalies + summary by type/severity/column |
| `DriverResult` | Single driver with method, strength, importance |
| `DriverReport` | All drivers + relationships |
| `Phase5Result` | Aggregate container |

### Marketing Contracts (Phase 6)

| Contract | Purpose |
|----------|---------|
| `MarketingReadiness` | 5-level readiness enum |
| `MarketingCapabilityReport` | Marketing-specific capability gates |
| `MMMResult` | Fitted MMM with channel parameters + metrics |
| `ChannelContribution` | Per-channel contribution + share + ROI |
| `ROIResult` | Naive + model-based ROI/ROAS |
| `BudgetOptimizationResult` | Current vs optimized allocation |
| `WhatIfScenario/Result` | Scenario comparison with disclaimer |
| `Phase6Result` | Aggregate container |

### Contract Flow

```
ColumnMapping → StandardizedDataset → CanonicalDataset → AnalysisContext
                                                              │
                    ┌─────────────────────┬──────────────────┬─┴──────────────┐
                    ▼                     ▼                  ▼                ▼
              AnalyticsResult    PredictionReport    Phase5Result    Phase6Result
              (KPIs, Insights,   (Forecast, Regr,    (Anomalies,    (MMM, ROI,
               Charts, Rels)     Class, Segment)      Drivers)       Optimizer)
```

---

## 5. Existing Orchestration

### analysis_runner.run_full_analysis()

The master orchestrator runs all phases sequentially with try/except isolation:
1. Profile → Standardize → Capabilities → Persist base artifacts
2. Generic MMM/Forecast/ROI (conditional, lightweight Ridge-based)
3. Phase 2: Build CanonicalDataset + AnalysisContext
4. Phase 3: Dynamic Analytics (KPI, Chart, Insight, Relationship)
5. Phase 4: Prediction (task detection → engine dispatch)
6. Phase 5: Anomaly + Driver
7. Phase 6: Marketing Intelligence

Each phase failure is non-fatal — captured in the report dict, subsequent phases still run.

### Session State Management

`session_paths.py` manages active dataset, resolves file paths, and clears all caches on dataset switch. `data_loader.py` provides `@st.cache_data`-decorated loaders that reconstruct contracts from persisted JSON/CSV.

---

## 6. Existing UI/API Structure

### Streamlit Dashboard (13 pages)

| Page | Content | Dataset-Agnostic? |
|------|---------|-------------------|
| 0: Upload Data | Upload + profile + confirm + analyze | ✅ Yes |
| 1: Executive Overview | Dynamic KPIs/charts/insights OR legacy marketing | ✅ Dynamic path |
| 2: Marketing Performance | Channel metrics + trends | ❌ Marketing-only |
| 3: MMM | Model results + channel parameters | ❌ Marketing-only |
| 4: Channel ROI | ROI/ROAS table | ❌ Marketing-only |
| 5: Budget Optimizer | Optimized allocation | ❌ Marketing-only |
| 6: What-If Simulator | Scenario comparison | ❌ Marketing-only |
| 7: Forecast | Time-series forecast chart | ❌ Marketing-oriented |
| 8: AI Marketing Analyst | Q&A interface | ❌ Marketing-only |
| 9: Recommendations | Business recommendations | ❌ Marketing-only |
| 10: Reports | Export reports | ❌ Marketing-only |
| 11: Predictions | Phase 4 prediction results | ✅ Yes |
| 12: Anomalies/Drivers | Phase 5 anomaly + driver results | ✅ Yes |

### FastAPI API

| Endpoint | Dataset-Agnostic? |
|----------|-------------------|
| `POST /upload` + `POST /datasets/{id}/confirm` | ✅ Yes |
| `GET /kpis` | ✅ Yes (generic for uploaded) |
| `GET /health` | ✅ Yes |
| `GET /channels`, `GET /attribution`, `GET /model-performance` | ❌ Legacy |
| `POST /optimize-budget`, `POST /simulate-budget` | ❌ Legacy (hardcoded channels) |
| `POST /ask` | ❌ Legacy |

---

## 7. Phase 7 Architectural Gaps

### The Decision Intelligence Framework

Phase 7 must bridge from "here are your analytics" to "here is what you should consider doing." This requires four intelligence tiers:

| Tier | Question | Existing Coverage | Gap |
|------|----------|-------------------|-----|
| **Descriptive** | What happened? | ✅ Full: KPIs, insights, charts, data quality | None |
| **Diagnostic** | Why did it happen? | ✅ Partial: Drivers, correlations, anomalies | Missing: causal reasoning, cross-signal synthesis |
| **Predictive** | What will happen? | ✅ Full: Forecast, regression, classification | Missing: impact quantification for decisions |
| **Prescriptive** | What should we do? | ⚠️ Marketing-only: optimizer, simulator | Missing: domain-agnostic recommendations |

### Specific Missing Components

**1. Evidence Classification System**
The system currently has no unified way to classify a finding as FACT, INFERENCE, PREDICTION, or RECOMMENDATION. Each engine outputs its own contract, but there's no meta-layer that labels the epistemic status of each claim.

**2. Decision Chain Orchestrator**
No module currently links: KPI anomaly → driver explanation → predictive impact → scenario → recommendation. Each engine runs independently in `analysis_runner.py`. The missing piece is a chain that connects outputs across phases into a coherent decision narrative.

**3. Business Impact Quantifier**
The prediction engines produce forecasts and regression results, but don't answer: "If metric X drops 10%, what is the expected dollar/unit impact on the business?" The optimizer does this for marketing budgets only.

**4. Domain-Agnostic Scenario Engine**
`src/marketing/simulator.py` runs what-if scenarios on the MMM's prediction surface. For non-marketing datasets (HR, operations, finance), there's no equivalent. A general-purpose "change feature X by Y%, predict impact on target Z" simulator is missing.

**5. Recommendation Generator**
No module currently generates structured, evidence-based recommendations with:
- What to do
- Why (linked evidence chain)
- Expected impact (quantified)
- Confidence level
- Limitations and assumptions
- Alternative options

The AI analyst (`src/ai/analyst.py`) generates narrative-style insights but doesn't produce structured decision recommendations.

**6. Confidence Framework**
The system has per-engine confidence scores (field discovery confidence, prediction confidence, anomaly scores), but no unified confidence framework that propagates uncertainty through the decision chain.

**7. Cross-Signal Synthesis**
No module currently synthesizes across engines: "KPI X dropped (insight) because of driver Y (driver engine) which correlates with time-series anomaly Z (anomaly engine), and the regression model shows this trend will continue (prediction)." Each phase persists independently.

---

## 8. Proposed Decision Intelligence Architecture

### Design Principle: Orchestrate, Don't Recreate

Phase 7 should be a thin orchestration + synthesis layer on top of existing Phases 3-6, NOT a reimplementation of their mathematics.

```
                    CanonicalDataset + AnalysisContext
                              │
                ┌─────────────┼──────────────────────┐
                ▼             ▼                      ▼
          Phase 3-4       Phase 5              Phase 6
          (Analytics +    (Anomaly +           (Marketing
           Prediction)     Driver)              Intelligence)
                │             │                      │
                └──────┬──────┘──────────────────────┘
                       ▼
              ┌─────────────────┐
              │  PHASE 7:       │
              │  Decision       │
              │  Intelligence   │
              │                 │
              │  ┌────────────┐ │
              │  │ Evidence   │ │  ← Classify all findings by epistemic status
              │  │ Classifier │ │
              │  └─────┬──────┘ │
              │        ▼        │
              │  ┌────────────┐ │
              │  │ Signal     │ │  ← Connect KPIs → Anomalies → Drivers → Predictions
              │  │ Synthesizer│ │
              │  └─────┬──────┘ │
              │        ▼        │
              │  ┌────────────┐ │
              │  │ Impact     │ │  ← Quantify business impact of each signal
              │  │ Quantifier │ │
              │  └─────┬──────┘ │
              │        ▼        │
              │  ┌────────────┐ │
              │  │ Scenario   │ │  ← Domain-agnostic what-if on regression surface
              │  │ Engine     │ │
              │  └─────┬──────┘ │
              │        ▼        │
              │  ┌────────────┐ │
              │  │ Decision   │ │  ← Generate structured recommendations
              │  │ Generator  │ │
              │  └─────┬──────┘ │
              │        ▼        │
              │  ┌────────────┐ │
              │  │ Confidence │ │  ← Propagate and aggregate uncertainty
              │  │ Framework  │ │
              │  └────────────┘ │
              └─────────────────┘
                       │
                       ▼
              DecisionReport contract
                       │
              ┌────────┼────────┐
              ▼                 ▼
         Dashboard          API
```

---

## 9. Recommended Phase 7 Modules

### New Modules (under `src/decision/`)

| Module | Purpose | Inputs | Output |
|--------|---------|--------|--------|
| `contracts.py` | All Phase 7 dataclasses | — | EvidenceItem, SignalChain, ImpactEstimate, Scenario, Decision, DecisionReport |
| `evidence_classifier.py` | Label every finding | KPIs, Insights, Anomalies, Drivers, Predictions | list[EvidenceItem] with epistemic tag |
| `signal_synthesizer.py` | Connect related findings | list[EvidenceItem] | list[SignalChain] — linked causal/associative chains |
| `impact_quantifier.py` | Quantify business impact | SignalChains + regression/forecast models | list[ImpactEstimate] |
| `scenario_engine.py` | Domain-agnostic what-if | AnalysisContext + fitted models | list[Scenario] |
| `decision_generator.py` | Structured recommendations | SignalChains + ImpactEstimates + Scenarios | list[Decision] |
| `confidence.py` | Uncertainty propagation | All upstream confidence scores | Aggregated confidence per decision |
| `orchestrator.py` | Phase 7 pipeline | CanonicalDataset + Phase 3-6 results | DecisionReport |

### Modified Modules

| Module | Change |
|--------|--------|
| `analysis_runner.py` | Add Phase 7 block (~20 lines), same pattern as Phase 6 |
| `capability_engine.py` | Add `decision_intelligence` capability gate |
| Dashboard pages | Add Phase 7 decision page |

---

## 10. Data Flow

### Phase 7 Internal Flow

```
Phase 3 AnalyticsResult ──┐
Phase 4 PredictionReport ─┤
Phase 5 Phase5Result ──────┼──► evidence_classifier ──► list[EvidenceItem]
Phase 6 Phase6Result ──────┘              │
                                          ▼
                               signal_synthesizer ──► list[SignalChain]
                                          │
                            ┌─────────────┤
                            ▼             ▼
                    impact_quantifier  scenario_engine
                            │             │
                            └──────┬──────┘
                                   ▼
                          decision_generator ──► list[Decision]
                                   │
                                   ▼
                          confidence.propagate() ──► DecisionReport
```

### Evidence Flow Example (Marketing Dataset)

```
KPI: Revenue dropped 15% last quarter [FACT]
  ↓
Anomaly: Sudden drop in period 8 [FACT]
  ↓
Driver: Channel_A spend reduction (r=0.82) is strongest driver [INFERENCE]
  ↓
Prediction: If Channel_A stays at current level, revenue will be $X ± Y [PREDICTION]
  ↓
Scenario: Restoring Channel_A spend to previous level → predicted +12% revenue [PREDICTION]
  ↓
Decision: "Consider restoring Channel_A spend. Based on observed relationship 
           (not proven causation), the model estimates +12% revenue improvement.
           Confidence: MEDIUM. Limitation: correlation-based, not causal experiment." [RECOMMENDATION]
```

### Evidence Flow Example (HR Dataset)

```
KPI: Attrition rate is 18% (above 15% threshold) [FACT]
  ↓
Anomaly: Spike in departures in Q3 [FACT]
  ↓
Driver: Overtime_Hours (importance=0.45) is strongest driver [INFERENCE]
  ↓
Prediction: If overtime stays constant, predicted attrition = 20% ± 3% [PREDICTION]
  ↓
Scenario: Reducing overtime by 20% → predicted attrition drops to 14% ± 4% [PREDICTION]
  ↓
Decision: "Consider reviewing overtime policies for high-overtime departments.
           Based on model association, reducing overtime is associated with lower attrition.
           Confidence: MEDIUM. Limitation: observational data, no randomized trial." [RECOMMENDATION]
```

---

## 11. Dataset-Agnostic Strategy

### Domain Detection (NOT Domain Assumption)

Phase 7 must never assume a domain. Instead, detect evidence patterns:

| Evidence Pattern | Detected Domain | Available Analyses |
|------------------|----------------|-------------------|
| Spend columns + target revenue/sales | Marketing | Full MMM + optimization |
| Employee + salary + performance | HR/Workforce | Attrition drivers, compensation analysis |
| Student + grades + attendance | Education | Performance predictors |
| Transaction + customer + product | Retail/Commerce | Customer segmentation, product analysis |
| Revenue + cost + margin | Finance | Profitability analysis, risk factors |
| Process + defect + duration | Operations | Quality drivers, bottleneck analysis |
| Generic numeric + categorical | General | Statistical analysis, prediction |

### Implementation Approach

1. **Don't classify domains.** Instead, detect available capabilities using the existing Phase 2 capability system.
2. **Extend capabilities**, not domain-specific pipelines. Add: `has_drivers`, `has_predictive_model`, `has_anomalies`, `has_scenarios` — capability gates for Phase 7 steps.
3. **Use the existing TargetCandidate system** to identify what to optimize / predict / explain.
4. **Let the evidence speak**: the recommendation generator should produce domain-appropriate language based on column names and semantic roles, not hardcoded domain templates.

---

## 12. Evidence / Recommendation Safety

### Current State

The system already has some safety mechanisms:
- MMM uses correlational language ("associated with", not "caused")
- What-if results include disclaimer text
- Driver engine explicitly states "predictive importance, not causation"
- ROI distinguishes naive vs. model-based
- Classification uses safe language for imprecise predictions

### Required Phase 7 Safety System

Every output must carry an `EvidenceLevel` tag:

| Level | Meaning | Example |
|-------|---------|---------|
| FACT | Directly computed from data | "Revenue was $1.2M last quarter" |
| INFERENCE | Statistical relationship found | "Revenue is correlated with Channel_A (r=0.82)" |
| PREDICTION | Model-generated projection | "Predicted revenue next quarter: $1.1M ± $100K" |
| RECOMMENDATION | Suggested action | "Consider increasing Channel_A budget by 15%" |
| LIMITATION | Known constraint on the analysis | "Model explains 72% of variance; other factors exist" |

### Safety Rules

1. **Never use causal language** unless an experimental design is detected (pre/post with control group — essentially never in observational data).
2. **Always pair recommendations with limitations.**
3. **Propagate uncertainty**: a recommendation built on a low-confidence driver + low-R² prediction should inherit low confidence.
4. **Distinguish correlation strength**: r=0.3 ("weak association") vs r=0.8 ("strong association").
5. **Flag conflicting signals**: if one driver suggests increasing spend while an anomaly suggests budget waste, surface both.
6. **Never fabricate evidence**: if no driver is found, say "no clear driver identified" — don't invent one.

---

## 13. Performance Considerations

### Current Bottlenecks

| Operation | Current | At 100K Rows | Recommendation |
|-----------|---------|-------------|----------------|
| Column classification | O(cols × aliases) | Fast | No change needed |
| Profiling | O(rows × cols) | ~1-2s | No change needed |
| Correlation matrix | O(cols²) | Grows with columns | Cap at 50 numeric columns (already done in anomaly) |
| KMeans segmentation | O(rows × K × iters) | ~5-10s | Already capped at 8 clusters |
| Ridge regression | O(rows × features²) | ~1s | No change needed |
| Anomaly detection (IQR/Z) | O(rows × cols) | ~2-3s | Already samples at 50K |
| MMM grid search | O(channels × decay_grid × shape_grid) | ~5-10s | Already small grids |
| Phase 3 insight detection | O(rows × cols²) | ~3-5s | No change needed |

### Phase 7 Performance

| New Operation | Expected Cost | Mitigation |
|---------------|-------------|------------|
| Evidence classification | O(n_findings) — small | Negligible |
| Signal synthesis | O(findings²) for pairwise matching | Cap at top 50 findings |
| Impact quantification | Reuses existing models — O(1) per signal | Already fitted |
| Scenario engine | O(n_scenarios × model_predict) | Cap scenarios at 10 |
| Decision generation | O(n_signal_chains) — small | Negligible |

### Caching Strategy

1. **Phase 3-6 results are already persisted** to JSON/CSV. Phase 7 should read these, not recompute.
2. **Lazy execution**: only run impact quantification for the top-K signal chains, not all.
3. **Persist Phase 7 results** to `phase7_decisions.json` alongside other artifacts.

---

## 14. Testing Strategy

### Proposed Phase 7 Test Structure

| Test Class | Tests | Description |
|------------|-------|-------------|
| `TestDecisionContracts` | ~15 | Serialization, roundtrip, evidence levels |
| `TestEvidenceClassifier` | ~12 | Correct labeling of KPI/Insight/Anomaly/Driver/Prediction findings |
| `TestSignalSynthesizer` | ~15 | Chain construction, cross-phase linking, no fabrication |
| `TestImpactQuantifier` | ~10 | Business impact estimation, uncertainty propagation |
| `TestScenarioEngine` | ~12 | Domain-agnostic scenarios, edge cases (no model, no target) |
| `TestDecisionGenerator` | ~15 | Recommendation structure, safety language, limitations |
| `TestConfidenceFramework` | ~8 | Uncertainty propagation, low-confidence flagging |
| `TestOrchestrator` | ~5 | E2E: marketing dataset, HR dataset, partial dataset |
| `TestDatasetAgnostic` | ~10 | Multiple domain datasets with different column schemas |
| `TestNonMarketingDecisions` | ~8 | HR, education, finance, operations datasets |
| `TestPartialData` | ~6 | Missing target, missing time, insufficient rows |
| `TestConflictingSignals` | ~5 | Contradictory drivers, uncertain predictions |
| `TestLowConfidence` | ~5 | Correct handling of low-confidence chains |
| `TestDatasetSwitching` | ~5 | No stale decisions from previous dataset |
| `TestDemoLeakage` | ~5 | No NovaMart-specific recommendations |
| `TestSafetyLanguage` | ~8 | Non-causal phrasing, uncertainty in every recommendation |
| `TestSecurity` | ~3 | No path traversal, no eval/exec |
| `TestPerformance` | ~3 | 10K+ rows complete in <30s |

**Estimated: ~150 new tests, total ~893**

---

## 15. Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Over-confident recommendations | HIGH | Mandatory confidence propagation + limitation pairing |
| Causal language creeping in | HIGH | Automated safety language tests + review |
| Domain-specific assumptions leaking in | MEDIUM | Dataset-agnostic tests with 5+ domain datasets |
| Signal synthesis fabricating connections | MEDIUM | Only link findings that share columns or time indices |
| Performance regression at scale | LOW | Phase 7 is thin orchestration, not computation |
| Breaking existing 743 tests | LOW | Phase 7 adds modules, doesn't modify existing ones |
| Stale decisions after dataset switch | MEDIUM | Reuse existing cache-clearing in session_paths |
| Demo data leakage | LOW | Same patterns as Phase 6 testing |

---

## 16. Dependencies Between Phase 7 Components

```
contracts.py                    ← No dependencies (define first)
       │
confidence.py                  ← contracts only
       │
evidence_classifier.py         ← contracts + Phase 3-6 contracts
       │
signal_synthesizer.py          ← contracts + evidence_classifier
       │
impact_quantifier.py           ← contracts + Phase 4 prediction models
       │
scenario_engine.py             ← contracts + Phase 4 models + Phase 6 optimizer
       │
decision_generator.py          ← contracts + all above
       │
orchestrator.py                ← all above + analysis_runner integration
```

---

## 17. Recommended Implementation Order

| Step | Module | Dependencies | Est. Tests |
|------|--------|-------------|------------|
| 1 | `src/decision/contracts.py` | None | ~15 |
| 2 | `src/decision/confidence.py` | contracts | ~8 |
| 3 | `src/decision/evidence_classifier.py` | contracts, Phase 3-6 contracts | ~12 |
| 4 | `src/decision/signal_synthesizer.py` | contracts, evidence_classifier | ~15 |
| 5 | `src/decision/impact_quantifier.py` | contracts, Phase 4 models | ~10 |
| 6 | `src/decision/scenario_engine.py` | contracts, Phase 4 models | ~12 |
| 7 | `src/decision/decision_generator.py` | all above | ~15 |
| 8 | `src/decision/orchestrator.py` | all above | ~5 |
| 9 | `analysis_runner.py` integration | orchestrator | ~3 |
| 10 | `capability_engine.py` update | contracts | ~2 |
| 11 | Dashboard page | orchestrator + data_loader | ~5 |
| 12 | API endpoints | orchestrator | ~3 |
| 13 | Dataset-agnostic tests | orchestrator | ~10 |
| 14 | Safety + edge case tests | all | ~20 |
| 15 | Performance + integration tests | all | ~8 |

**Estimated total: ~150 new tests**
**Phase 7 should NOT modify any existing Phase 0-6 source files** except for adding integration hooks in `analysis_runner.py` (~20 lines) and `capability_engine.py` (~5 lines).

---

## PHASE 7 ARCHITECTURE AUDIT STATUS

| Item | Status |
|------|--------|
| Architecture understood | **YES** |
| Phase 0–6 preserved | **YES** |
| Phase 7 implementation started | **NO** |
| Files modified | **0** |
| Ready for Phase 7 implementation | **YES** |

---

**END OF AUDIT — Awaiting instruction to begin Phase 7 implementation.**
