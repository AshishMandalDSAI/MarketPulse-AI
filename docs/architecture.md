# MarketPulse AI — Architecture

## System Overview

```
                    MARKETPULSE AI
                          |
                          v
   generate_dataset.py -> RAW DATA (data/raw/)
                          |
                          v
   data_cleaning.py -> VALIDATED + CLEANED (stage 1)
                          |
                          v
   feature_engineering.py -> ANALYSIS-READY DATASET (data/processed/marketing_data_clean.csv)
                          |
             +------------+------------+
             v                         v
   marketing_analytics.py     src/models/*
   (KPIs, channel perf,       (adstock, saturation,
    monthly/quarterly,         marketing_mix_model,
    holiday/promo)             model_evaluation,
             |                 channel_contribution)
             |                         |
             +------------+------------+
                          v
              src/optimization/*
     (budget_optimizer.py — scipy SLSQP,
      simulator.py — what-if scenarios)
                          |
             +------------+------------+
             v                         v
   src/forecasting/forecast.py   src/ai/analyst.py
   (trend + seasonal Ridge)      (rule-based / optional LLM)
             |                         |
             +------------+------------+
                          v
              +-----------+-----------+
              v                       v
        app.py (Streamlit)      api/main.py (FastAPI)
        dashboard/pages/*             |
                          v           v
                 src/utils/database.py (SQLite,
                 marketpulse.db — read-model store)
```

## Layer Responsibilities

- **Data layer** (`src/data/`): synthetic dataset generation, validation, and cleaning.
- **Feature layer** (`src/features/`): time features, spend aggregates/shares, growth rates.
- **Analytics layer** (`src/analytics/`): descriptive/observational KPIs and breakdowns — no modeling.
- **Model layer** (`src/models/`): the Marketing Mix Model itself — adstock, saturation, regression,
  evaluation, and attribution. This is the only layer that produces causal-flavored (model-estimated)
  claims, and it consistently labels them as such.
- **Optimization layer** (`src/optimization/`): budget optimization (scipy.optimize) and what-if simulation,
  both built directly on top of the fitted model from the model layer.
- **Forecasting layer** (`src/forecasting/`): a separate, deliberately simple time-series forecast — not
  part of the MMM causal story, just a near-term revenue projection.
- **AI layer** (`src/ai/`): a rule-based analyst grounded in the above layers' outputs, with an optional
  LLM narrative layer that never touches the underlying numbers.
- **Presentation layer**: `app.py` + `dashboard/` (Streamlit) and `api/main.py` (FastAPI) are two
  independent consumers of the same underlying pipeline outputs and fitted model — they do not duplicate
  business logic.
- **Persistence**: `data/processed/marketpulse.db` (SQLite) is a convenience read-model built from the
  same CSV/JSON pipeline outputs, for anyone who wants to query the project's results with SQL.

## Why this layering

Each layer only depends on layers "above" it in the diagram and writes its outputs to `data/processed/` or
`models/` as CSV/JSON/pickle files. This means:

- Every layer can be re-run independently (`python src/models/marketing_mix_model.py`, etc.) without
  needing the dashboard or API running.
- The dashboard and API never compute business logic themselves — they only read pipeline outputs or call
  the same `src/` functions the pipeline scripts use, so the two presentation layers can never disagree
  with each other or with the CLI pipeline output.
