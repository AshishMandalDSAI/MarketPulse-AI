<div align="center">

# 📊 MarketPulse AI

### AI-Powered Marketing Mix Modeling & Budget Optimization

**Transform Marketing Data into Smarter Business Decisions.**

<p>
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Machine-Learning-orange?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="Machine Learning">
</p>

<p>
  <a href="https://marketpulseaifinalrelease-iuhclpukvwxpdqj7hmhjy8.streamlit.app/">
    <img src="https://img.shields.io/badge/🚀_LIVE_DEMO-Try_MarketPulse_AI-FF4B4B?style=for-the-badge" alt="Live Demo">
  </a>
  <a href="https://github.com/AshishMandalDSAI/MarketPulse_AI_FINAL_RELEASE">
    <img src="https://img.shields.io/badge/GitHub-View_Repository-181717?style=for-the-badge&logo=github" alt="GitHub Repository">
  </a>
</p>

</div>

---

## 📌 Overview

**MarketPulse AI** is an end-to-end Marketing Mix Modeling (MMM) and marketing budget optimization platform developed as an MBA Data Science & AI final-year project.

It helps users analyze marketing performance, understand relationships between marketing expenditure and revenue, forecast future performance, and explore data-driven budget allocation strategies.

The platform combines data analytics, machine learning, forecasting, and interactive dashboards in one application.

> **Note:** The built-in demonstration dataset is synthetic and intended for educational and demonstration purposes. It does not represent actual company performance.

---

## 🌐 Live Application

### Try MarketPulse AI Online

🚀 **[Launch MarketPulse AI](https://marketpulseaifinalrelease-iuhclpukvwxpdqj7hmhjy8.streamlit.app/)**

Explore the interactive dashboard and experience the application's available analytics features.

> The hosted app runs an earlier release. This release (ULTIMATE FINAL) has **not** been deployed there.
> The upgraded release described below (upgrade PARTs 01–06) has **not** been deployed anywhere either.

---

## 🆕 Upgrade (PARTs 01–06)

This release extends the ULTIMATE FINAL baseline in five parts. Nothing was removed.
The details are in `FINAL_CHANGELOG.md`, the status in `FINAL_PROJECT_STATUS.md` and the
test evidence in `FINAL_TEST_REPORT.md`.

**Navigation (sidebar)**

| # | Page | What it does |
|---|---|---|
| 00 | Upload Data | Upload, quality check, universal column-role mapping with evidence, analysis target(s), currency, cleaning, analysis |
| 01 | Statistics | Three tabs: Descriptive, Inferential (10 tests incl. the Z-test with a required known σ) and Predictive (12 classifiers, regression models) |
| 02 | Executive Overview | KPIs and overview charts |
| 03 | Marketing Performance | Channel and trend charts |
| 04 | Marketing Effectiveness Model | Driver response model (OLS, HC3 errors, chronological holdout, quality grade) |
| 05 | Channel Performance & Contribution | Contribution, response curves; ROI/ROAS only where channel-wise cost exists |
| 06 | Marketing Resource Optimizer | Spend or exposure allocation with feasibility diagnostics, remedies A–D, optional reserve, solution status |
| 07 | Marketing Scenario Simulator | Channel and total-budget scenarios, sensitivity (tornado) |
| 08 | Forecast | Revenue forecast with walk-forward evaluation |
| 09–15 | AI Marketing Analyst, Recommendations, Reports, Predictions, Anomalies & Drivers, Decision Intelligence, Prescriptive Analytics | Unchanged baseline features (chart labels completed) |

**Analytics modes.** Every upload is classified into one of three modes. The page header always states which one applies:

- **Spend-Based Marketing Analytics**: channel-wise advertising spend columns exist, so ROI, ROAS and budget optimisation are available.
- **Exposure-Based Marketing Analytics**: only exposure columns exist (impressions, GRPs, clicks and so on). Effects and exposure-mix optimisation are available. No monetary ROI or ROAS is ever calculated. Trade or promotion spend is a business variable, not channel spend.
- **General Business Analytics**: there are no marketing channels. The statistics and predictive features remain available.

**Currency.** The demo is INR. For uploads, the currency is detected from symbols or codes, or chosen by you. If it is unknown, no symbol is shown and INR or USD is never assumed.

**Optional model libraries.** XGBoost, LightGBM and CatBoost are optional. When they are missing, the Predictive tab lists them as *unavailable* and the app keeps working.

**Documentation index.**

| File | Contents |
|---|---|
| `INSTALLATION.md` | Installation and configuration |
| `WINDOWS_SETUP_GUIDE.md` | Windows setup, PowerShell scripts, one-command validator |
| `DOCKER_SETUP_GUIDE.md` | Docker / Compose, optional ML build argument |
| `USER_GUIDE.md` | User guide for the upgraded pages (incl. the Kaggle example) |
| `KNOWN_LIMITATIONS.md` | Every limitation with its verification status |
| `docs/ARCHITECTURE_UPGRADE.md` | Architecture of the new modules |
| `DATASET_COMPATIBILITY_REPORT.md` | Dataset compatibility (real Kaggle file validated in PART 06) |
| `MODEL_VALIDATION_REPORT.md` | Model validation |
| `SECURITY_AUDIT_REPORT.md` | Security audit (PART 05 + PART 06 local penetration-style tests) |
| `PERFORMANCE_REPORT.md` | Performance (PART 06 before/after) |
| `DEPLOYMENT_READINESS_REPORT.md` | Deployment readiness (Docker verification status) |
| `FINAL_TEST_REPORT.md` | Final test report |
| `FINAL_PROJECT_STATUS.md` | Final project status |
| `PART_STATUS.md` | Status of each part |

**Verification status at a glance** (details: `FINAL_PROJECT_STATUS.md`):

| Area | Status |
|---|---|
| Full test suite, clean Linux environment | Tested: passed (`FINAL_TEST_REPORT.md`) |
| Real Kaggle file `synthetic_mmm_weekly_india_SAT.csv` | Tested: passed (PART 06) |
| Docker with the **official** base image | **Blocked** (Docker Hub unreachable from the build environment) |
| Docker with a local stand-in base image, incl. optional ML | Tested with limitations |
| Windows | **Not tested on Windows**; scripts verified with PowerShell 7 on Linux |
| Production deployment | Not performed; not claimed production-ready |

---

## ✨ Key Features

| Feature                   | Description                                                                  |
| ------------------------- | ---------------------------------------------------------------------------- |
| 📊 Executive Dashboard    | View key performance indicators and dataset summaries.                       |
| 📂 Flexible Data Upload   | Upload supported CSV and Excel datasets.                                     |
| 🔍 Data Profiling         | Examine data types, missing values, and dataset structure.                   |
| 📈 Marketing Analytics    | Explore marketing performance and available trends.                          |
| 🧠 Marketing Mix Modeling | Estimate statistical relationships between marketing channels and revenue.   |
| 💰 ROI & ROAS Analysis    | Evaluate marketing efficiency when the required data is available.           |
| 🎯 Budget Optimization    | Explore budget allocation recommendations when a trained model is available. |
| 🔮 Revenue Forecasting    | Forecast future values when the dataset supports time-series analysis.       |
| 🧪 What-If Simulation     | Explore alternative marketing budget scenarios when supported.               |
| 🤖 AI Marketing Analyst   | Rule-based (offline) answers built from computed results, each labelled with its evidence type and source. |
| 💡 Recommendations        | Review data-driven findings and potential business actions.                  |
| 📄 Business Reports       | Generate reports based on available analytical results.                      |
| 📊 Advanced Analytics (Phase 8) | Now page 01 · Statistics (Descriptive / Inferential / Predictive tabs) and page 15 · Prescriptive Analytics (formerly pages 14–17): Descriptive Statistics, Inferential Statistics, Predictive Analytics and Prescriptive Analytics, all computed from **your active dataset** (never demo data unless you choose it). See `docs/PHASE_8_PART_6_FINAL_INTEGRATION.md` for the overview, workflow, data requirements, installation, testing and known limitations. |

---

## 📂 Flexible Dataset Support

MarketPulse AI follows a dataset-first approach.

It can profile uploaded CSV and supported Excel files without requiring every dataset to use the same column names.

### How It Works

1. **Upload Data** — Import a supported CSV or Excel file.
2. **Profile the Dataset** — Inspect rows, columns, data types, and missing values.
3. **Identify Column Roles** — Suggest possible dates, revenue measures, and marketing channels.
4. **Confirm Column Mapping** — Review and confirm ambiguous column assignments.
5. **Detect Available Capabilities** — Determine which analyses the dataset supports.
6. **Explore Insights** — Use the dashboard, KPIs, reports, and available models.

### Important

Not every dataset supports every analytical feature.

For example:

* Trend analysis requires a usable date column.
* Forecasting requires suitable date and numeric target data.
* Marketing Mix Modeling requires appropriate date, revenue, and marketing-channel data.
* Budget optimization requires a trained marketing mix model.

When required data is unavailable, the corresponding feature may be unavailable.

---

## 🧠 Machine Learning & Analytics

The project incorporates the following analytical capabilities:

* Marketing Mix Modeling
* Marketing Channel Contribution Analysis
* Revenue Forecasting
* Marketing ROI and ROAS Analysis
* Budget Optimization
* What-If Scenario Simulation
* Anomaly Detection and Driver Analysis
* Marketing KPI Analysis
* AI-Assisted Business Insights
* Decision Support and Business Reporting

### Model Interpretation

The built-in demonstration workflow uses an adstock and saturation-based Marketing Mix Modeling pipeline.

For supported uploaded datasets, the platform can use a Ridge regression approach with chronological training and holdout evaluation.

**Important:** Regression coefficients indicate statistical associations. They do not, by themselves, establish that marketing expenditure caused a change in revenue.

---

## 🛠️ Technology Stack

### Programming Language

* Python

### Data Analysis

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* Ridge Regression
* Marketing Mix Modeling

### Dashboard

* Streamlit

### API

* FastAPI
* Uvicorn

### Data Formats

* CSV
* Excel

### Deployment & Development

* Git
* GitHub
* Docker
* Docker Compose

---

## 🏗️ System Architecture

```text
              ┌────────────────────────┐
              │      Marketing Data    │
              │      CSV / Excel       │
              └────────────┬───────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │    Data Upload &       │
              │       Profiling        │
              └────────────┬───────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  Column Classification │
              │  & Standardization      │
              └────────────┬───────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │ Capability Detection   │
              └────────────┬───────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │ Analytics & ML Models  │
              │ MMM / Forecasting      │
              │ ROI / Budget Analysis  │
              └────────────┬───────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  Insights & Reports    │
              └────────────┬───────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │ Streamlit Dashboard    │
              │      FastAPI           │
              └────────────────────────┘
```

---

## ⚙️ Installation

**Verified environment:** Python 3.13.18 on Linux (full test suite). The dependency
ranges in `requirements.txt` declare a broader range; only 3.13.18 was verified.

```bash
unzip MarketPulse_AI_ULTIMATE_FINAL_RELEASE.zip
cd MarketPulse_AI_FINAL_RELEASE
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-optional.txt   # optional: XGBoost, LightGBM, CatBoost
```

The verified release ZIP is `MarketPulse_AI_FINAL_RELEASE_VERIFIED.zip`; it extracts to a
`MarketPulse_AI_FINAL_RELEASE/` folder. On Windows, use `scripts\windows\setup.ps1` (see `WINDOWS_SETUP_GUIDE.md`). LightGBM needs the OpenMP runtime (`libgomp1` on
Debian/Ubuntu). Full instructions: `INSTALLATION.md`.

## 🔐 Environment Setup

```bash
cp .env.example .env                # .env is git- and docker-ignored
```

| Variable | Default | Purpose |
|---|---|---|
| `MARKETPULSE_ENV` | `development` | `production` makes the API key mandatory; the API refuses to start without it. |
| `MARKETPULSE_API_KEY` | empty | Secret required in the `X-API-Key` header on protected API endpoints. |
| `CORS_ORIGINS` | localhost dev origins | Comma-separated allowed origins. Never `*` in production. |
| `ANTHROPIC_API_KEY` | empty | Currently unused. The AI Marketing Analyst is rule-based and offline; no code path calls an LLM, with or without this key. |
| `MARKETPULSE_DASHBOARD_PASSWORD_HASH` | empty | PBKDF2 hash of the dashboard password (generate with `python -m dashboard.components.auth`). Required in production; without it a production dashboard refuses to serve. In development, no sign-in is required unless a hash is set. |
| `MARKETPULSE_RATE_LIMIT` | on in production, off in development | `on`/`off` override for the API's per-IP rate limiter (HTTP 429 with `Retry-After`). |
| `MARKETPULSE_RATE_LIMIT_UPLOAD` / `_COMPUTE` / `_DEFAULT` | `10` / `30` / `120` | Requests per minute per client IP for upload, compute (optimise/simulate/ask/confirm/decisions run) and other endpoints. |
| `WEB_CONCURRENCY` | `1` | Must stay 1: the upload registry and rate limiter are in-process. Production refuses a larger value. |
| `MARKETPULSE_MAX_UPLOAD_MB` / `MARKETPULSE_MAX_ROWS` | `50` / `100000` | API upload size and row limits. |

Generate an API key: `python -c "import secrets; print(secrets.token_urlsafe(32))"`

## ▶️ Running Locally

**Dashboard**

```bash
streamlit run app.py                # http://localhost:8501
```

The demo data and model ship with the project, so the dashboard works immediately.

**API**

```bash
uvicorn api.main:app --reload       # http://127.0.0.1:8000, docs at /docs
```

In development mode without a key, the API logs a warning listing the protected
endpoints that are open. For any shared deployment:

```bash
export MARKETPULSE_ENV=production
export MARKETPULSE_API_KEY=<long random secret>
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Public endpoints: `/`, `/health`, `/docs`, `/openapi.json`, `/channels`,
`/model-performance`, `/attribution`, `/kpis` (demo dataset).
Protected endpoints: `POST /upload`, `POST /datasets/{id}/confirm`,
`POST /optimize-budget`, `POST /simulate-budget`, `POST /ask`,
`GET /datasets/{id}/decisions`, `POST /datasets/{id}/decisions/run`,
`GET /kpis?dataset_id=<uploaded id>`.

## 🧪 Testing

```bash
python -I -m pytest tests/ -v --tb=short
```

`pytest` is constrained to `<9.2` (pytest 9.2.0 rejects `pytest.approx` on nested
data used in `tests/test_phase8_part2_engine.py`). Latest verified result (PART 06, clean environment):
**3,522 collected, 3,522 passed, 0 failed, 0 skipped, 0 warnings** (`FINAL_TEST_REPORT.md`). On Windows use
`scripts\windows\run_tests.ps1`. (`FINAL_TESTING_REPORT.md` is the baseline's report.) `tools/benchmark.py <rows>` reproduces the
performance measurements in `FINAL_PERFORMANCE_REPORT.md`.

## 🚀 Deployment

**Docker Compose** (dashboard + API):

```bash
cp .env.example .env                # set MARKETPULSE_ENV=production and MARKETPULSE_API_KEY
docker compose up --build -d        # INSTALL_OPTIONAL_ML=true in .env adds XGBoost/LightGBM/CatBoost
```

Verification status: see `DOCKER_SETUP_GUIDE.md`. The official base image could not be pulled in the build
environment, so the containers were verified with a local stand-in base image.

`./data` and `./models` are bind-mounted; on first start the entrypoint creates the
demo artifacts only if they are missing and never overwrites existing data.
The entrypoint then drops root privileges: the app runs as the owner of `./data`
when that is a non-root host user, otherwise as uid/gid 10001. For HTTPS, use the
reverse-proxy setup in `deploy/` (see `deploy/README.md`). If the dashboard
password hash is placed in `.env`, every `$` must be written as `$$`
(Compose interpolation).

**Streamlit Community Cloud**: main file `app.py`, dependencies from
`requirements.txt`. Choose Python 3.13 in the app's advanced settings (the verified
version). Add `MARKETPULSE_DASHBOARD_PASSWORD_HASH` under *Secrets* to require
sign-in (recommended for any app that is not meant to be public).
`ANTHROPIC_API_KEY` is not needed: the AI Analyst is rule-based. The Streamlit app does not call the FastAPI backend, so API settings do
not apply there. Uploaded files on Streamlit Cloud live on ephemeral storage.

See `FINAL_DEPLOYMENT_VERIFICATION.md` for what was actually verified.

## 📋 Dataset Requirements

- Formats: `.csv` and `.xlsx` (legacy `.xls` is not supported; save as `.xlsx`).
- One header row; one record per row. Column names can be anything.
- Limits: the API accepts up to 50 MB per upload and analyses the first 100,000
  rows (configurable via `MARKETPULSE_MAX_UPLOAD_MB` / `MARKETPULSE_MAX_ROWS`).
  The dashboard uses Streamlit's default 200 MB upload limit; anomaly detection
  samples very large datasets down to 50,000 rows and says so in its output.
- Feature requirements: trends and forecasting need a parseable date column;
  MMM, ROI/ROAS and budget optimisation need a date, a revenue measure and at
  least one marketing-spend column. Missing capabilities are shown as unavailable
  with a reason, never filled with demo data.
- Medium- and low-confidence column roles must be confirmed on the Upload page
  before they are used.

## ⚠️ Known Limitations

- Model outputs are statistical associations, not proof of causation; forecasts
  and optimised budgets are estimates, not guarantees.
- The demo dataset is synthetic.
- The API's upload registry and rate limiter are in-memory and per-process; the
  API is pinned to a single worker (production refuses `WEB_CONCURRENCY` > 1).
- The dashboard sign-in is a single shared team password (no user accounts or roles).
- The AI Marketing Analyst is rule-based (keyword routing plus computed facts); it is
  not a language model.
- Docker was verified with a locally built substitute of `python:3.13-slim`
  (Docker Hub was unreachable from the build environment); see
  `FINAL_DEPLOYMENT_VERIFICATION.md`.
- Full list: `FINAL_KNOWN_LIMITATIONS.md`.

Upgrade limitations: the full list, each with its verification status, is in `KNOWN_LIMITATIONS.md`. The main ones:

- Docker was verified only with a local stand-in base image; the official `python:3.13-slim` image is
  **blocked** in the build environment.
- Windows was **not tested on Windows**. Run `scripts\windows\validate_windows.ps1`.
- The shared driver response model has no adstock: it is graded *weak* on the demo and *good* on the
  real Kaggle file.
- *Confirm & Analyze* takes about 255 s at 100,000 rows on 2 vCPUs (`PERFORMANCE_REPORT.md`).
- Verified only on Linux with Chrome.
