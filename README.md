# MarketPulse AI

### AI-Driven Marketing Mix Modeling & Budget Optimization Platform

**Turn Marketing Spend into Smarter Business Decisions.**

MarketPulse AI is an end-to-end Marketing Mix Modeling (MMM) and budget-optimization platform, built as an
MBA Data Science & AI final-year project for a fictional e-commerce company, **NovaMart**. It quantifies
how each marketing channel drives revenue, accounts for carryover and diminishing-returns effects, and
recommends how to reallocate a fixed marketing budget to maximize predicted revenue.

> **The dataset is entirely synthetic**, generated for academic/project demonstration purposes. It does not
> represent a real company, and no figure in this project should be read as describing an actual business.

---

## Quick Start

**Required (dashboard + all analytics):**
```bash
pip install -r requirements.txt

python src/data/generate_dataset.py
python src/data/data_cleaning.py
python src/features/feature_engineering.py

python src/analytics/marketing_analytics.py

python src/models/marketing_mix_model.py
python src/models/model_evaluation.py
python src/models/channel_contribution.py

python src/optimization/budget_optimizer.py
python src/optimization/simulator.py

python src/forecasting/forecast.py
python src/utils/database.py

streamlit run app.py
```
This opens directly into **MarketPulse AI — AI-Driven Marketing Mix Modeling & Budget Optimization** at
`http://localhost:8501`, with no further setup. If you skip the pipeline steps above, the dashboard detects
this and offers a one-click **"Initialize Demo Data"** button on first launch that runs the same steps for
you (see `app.py`).

**Optional — FastAPI backend** (only needed if you want the JSON API, not required for the dashboard):
```bash
uvicorn api.main:app --reload   # http://localhost:8000/docs
```

**Optional — LLM-assisted AI Analyst** (the AI Marketing Analyst works fully offline in "Demo AI Mode" by
default; this only enables an optional fluent-language layer on top of the same computed numbers):
```bash
cp .env.example .env
# then set ANTHROPIC_API_KEY=... inside .env
```

---

## Overview

| | |
|---|---|
| **Business problem** | "Where should we spend our next marketing dollar?" |
| **Channels modeled** | TV, Radio, Print, Search, Social Media, Display Ads, Email Marketing, Influencer Marketing |
| **Core technique** | Geometric adstock → Hill saturation → Ridge regression, time-aware validation |
| **Deliverables** | Streamlit dashboard, FastAPI backend, rule-based/optional-LLM AI analyst, SQLite persistence |

## Objectives

1. Quantify each channel's estimated contribution to revenue.
2. Model marketing carryover (adstock) and diminishing returns (saturation).
3. Recommend an optimal weekly budget allocation under real-world constraints.
4. Provide an interactive "what-if" simulator for hypothetical budget scenarios.
5. Provide a natural-language AI Marketing Analyst grounded in the project's own analytics.

## Key Features

- 📊 **Executive Overview** — KPI cards, revenue/spend trends, channel contribution, business summary.
- 📈 **Marketing Performance** — filterable revenue/spend/channel/holiday/promotion analytics.
- 🧠 **Marketing Mix Model** — model comparison, actual-vs-predicted, residuals, adstock/saturation curves.
- 💰 **Channel ROI** — attribution table, top/bottom performer, spend-vs-contribution.
- 🎯 **Budget Optimizer** — live scipy.optimize run against the fitted model, current vs. recommended.
- 🔮 **What-If Simulator** — interactive per-channel sliders with live model predictions.
- 📅 **Forecast** — 12-week revenue forecast with a 90% prediction interval.
- 🤖 **AI Marketing Analyst** — works fully offline ("Demo AI Mode"); optional LLM narrative layer.
- 💡 **Recommendations** — dynamically generated from attribution + optimization outputs.
- 📋 **Reports** — CSV/JSON downloads and an auto-generated executive summary.

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for the full data-flow diagram. In brief:

```
raw data -> clean -> feature-engineer -> analytics
                                       -> MMM (adstock, saturation, regression, attribution)
                                            -> budget optimizer -> what-if simulator
                                            -> forecast
                                            -> AI analyst
                                       -> Streamlit dashboard + FastAPI (two independent consumers
                                          of the same pipeline outputs / fitted model)
```

## Technology Stack

Python · Pandas · NumPy · Scikit-learn · SciPy · Plotly · Streamlit · FastAPI · Pydantic · SQLite · Pytest · Docker

> `statsmodels` was unavailable in the offline sandbox used to build this project, so the regression layer
> uses scikit-learn's `Ridge` instead — a standard, defensible substitute at this data scale (see
> `docs/methodology.md`).

## Data Pipeline

```bash
python src/data/generate_dataset.py        # synthetic NovaMart dataset -> data/raw/
python src/data/data_cleaning.py           # validation + cleaning -> data/processed/ (stage 1)
python src/features/feature_engineering.py # -> data/processed/marketing_data_clean.csv
python src/analytics/marketing_analytics.py
```

## Marketing Mix Modeling, Adstock, Saturation, Attribution

```bash
python src/models/marketing_mix_model.py    # trains + compares baseline / adstock / adstock+saturation
python src/models/model_evaluation.py       # residual diagnostics + curve data for the dashboard
python src/models/channel_contribution.py   # leave-one-channel-out attribution -> channel_attribution.csv
```

**Selected model:** `adstock` — test R² 36.2%, RMSE ₹316,323, MAE ₹239,478, MAPE 4.2%.
Full comparison and selection rationale in [`docs/model_documentation.md`](docs/model_documentation.md).

## Budget Optimization

```bash
python src/optimization/budget_optimizer.py   # scipy.optimize SLSQP, real numbers, no hard-coding
python src/optimization/simulator.py          # what-if scenario comparison
```

Last verified run: reallocating ₹2,224,462/week is projected to improve predicted
revenue by **+21.7%**. See
[`docs/optimization_methodology.md`](docs/optimization_methodology.md).

## What-If Simulator

`src/optimization/simulator.py::BudgetSimulator` powers both the dashboard's What-If page and the API's
`/simulate-budget` endpoint, so results are always consistent across surfaces.

## AI Marketing Analyst

`src/ai/analyst.py` answers questions like *"Which channel should get more budget?"* using this project's
own computed analytics. **Works fully offline** ("Demo AI Mode") — no API key required, and the app never
crashes without one. If `ANTHROPIC_API_KEY` is set (see `.env.example`), an LLM can optionally be layered on
for more fluent phrasing — the underlying numbers always come from this project's analytics, never the LLM.

## Dashboard

```bash
pip install -r requirements.txt
streamlit run app.py
```

10-page navigation (Executive Overview → Reports), professional dark-navy/white-card SaaS styling
(`dashboard/styles/theme.py`), Plotly charts throughout, interactive filters, and `st.cache_data` used for
all expensive loads (Phase 15 — Performance).

## API

```bash
uvicorn api.main:app --reload
```
Interactive docs at `http://127.0.0.1:8000/docs`. Endpoints: `/health`, `/kpis`, `/channels`,
`/model-performance`, `/attribution`, `/optimize-budget` (POST), `/simulate-budget` (POST), `/ask` (POST).
Full spec in [`docs/api_documentation.md`](docs/api_documentation.md).

## Installation & Running (detailed)

The **Quick Start** section above covers the fast path. For a virtual-environment setup or more detail on
each step:

```bash
git clone <this-repo>
cd marketpulse-ai
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

Run the full pipeline once (regenerates all `data/processed/` and `models/` artifacts — each step consumes
the previous step's output, so run them in order):

```bash
python src/data/generate_dataset.py
python src/data/data_cleaning.py
python src/features/feature_engineering.py
python src/analytics/marketing_analytics.py
python src/models/marketing_mix_model.py
python src/models/model_evaluation.py
python src/models/channel_contribution.py
python src/optimization/budget_optimizer.py
python src/optimization/simulator.py
python src/forecasting/forecast.py
python src/utils/database.py
```

Then launch either or both:
```bash
streamlit run app.py                 # dashboard on http://localhost:8501
uvicorn api.main:app --reload        # API on http://localhost:8000
```

## Testing

```bash
pytest                        # if pytest is installed (recommended)
python tests/run_all_tests.py # manual runner, no pytest required
```

**Last verified run (this environment, no pytest/fastapi installed): 40/40 tests passed** — covers dataset
generation, validation/cleaning, KPI calculation, adstock, saturation, MMM training/evaluation, attribution,
budget optimization (allocation sums to budget, no negative spend), the offline AI Marketing Analyst (all 5
required demo questions plus increase/reduce routing), and FastAPI request-validation/error-handling tests
(auto-skip if `fastapi` isn't installed, rather than failing).

## Docker

```bash
docker build -t marketpulse-ai .
docker run -p 8501:8501 marketpulse-ai
```
or both services via Compose:
```bash
docker compose up --build
```
> **Not verified in this environment** — the development sandbox used to build this project has no Docker
> daemon and no internet access, so the Dockerfiles are written and reviewed but not build-tested here.
> Please verify `docker build` locally before relying on it.

## Project Structure

```
marketpulse-ai/
├── app.py                    # Streamlit entry point
├── api/main.py                # FastAPI backend
├── src/
│   ├── data/                  # generation, cleaning, validation
│   ├── features/               # feature engineering
│   ├── analytics/              # KPIs, channel performance
│   ├── models/                 # adstock, saturation, MMM, evaluation, attribution
│   ├── optimization/           # budget optimizer, what-if simulator
│   ├── forecasting/             # revenue forecast
│   ├── ai/                     # rule-based / optional-LLM analyst
│   └── utils/                  # SQLite database
├── dashboard/
│   ├── pages/                  # 10 dashboard pages
│   ├── components/              # KPI cards, cached data loaders
│   └── styles/                  # theme CSS
├── data/{raw,processed}/       # dataset + all pipeline outputs
├── models/                     # trained model + metrics + diagnostics
├── notebooks/                  # EDA notebook
├── docs/                       # architecture, methodology, model/API/optimization docs, glossary
├── reports/                    # final project report support (academic write-up)
├── tests/                      # pytest-compatible test suite + manual runner
├── requirements.txt
├── Dockerfile / Dockerfile.api / docker-compose.yml
├── .env.example
└── .gitignore
```

## Limitations

- Synthetic dataset — not real company data.
- `statsmodels` unavailable in the build sandbox; Ridge (scikit-learn) used instead.
- Per-channel adstock/saturation parameters use a univariate grid-search heuristic, not a full joint
  optimization across channels.
- Budget optimizer treats a single representative week, not a full multi-week dynamic spend path.
- Streamlit/FastAPI/Plotly could not be installed or runtime-tested in the offline build sandbox (no
  internet access) — all such code was syntax-checked (`py_compile`) and statically audited (route
  inspection, unused-import scan, circular-import check), but not executed end-to-end. Please run
  `pip install -r requirements.txt` and verify locally before presenting. The FastAPI test suite
  (`tests/test_api.py`) is written and will run automatically once `fastapi`/`starlette` are installed.
- Docker build was not tested in this environment (no Docker daemon, no internet access).

## Future Improvements

- Bayesian MMM (e.g. PyMC-Marketing) for posterior uncertainty on channel contributions.
- Joint, not per-channel-univariate, adstock/saturation parameter optimization.
- Multi-week dynamic budget-pacing optimization.
- Swap in real transactional data if adapted beyond an academic setting.

## Academic Contribution

Demonstrates the standard end-to-end Marketing Mix Modeling workflow (carryover + saturation transforms,
regularized regression, time-aware validation, model-based attribution, constrained optimization) at MBA
capstone scope, packaged as a working software product rather than only a notebook.

## Business Impact

Under the assumptions documented in `docs/methodology.md` and `docs/optimization_methodology.md`, this
project's optimizer projects a **+21.7% revenue improvement** from
reallocating (not increasing) the current weekly marketing budget — illustrating the kind of decision
support this class of tool can offer a real marketing team, while being explicit that this figure is a
model-based projection on synthetic data, not a guarantee.
