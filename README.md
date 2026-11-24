<div align="center">

# 📊 MarketPulse AI

### AI-Powered Marketing Mix Modeling & Budget Optimization

**Transform Marketing Data into Smarter Business Decisions.**

<p>
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Machine_Learning-Scikit--learn-orange?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="Machine Learning">
  <img src="https://img.shields.io/badge/Tests-3522_Passed-success?style=for-the-badge" alt="Tests">
  <img src="https://img.shields.io/badge/Status-Verified-blue?style=for-the-badge" alt="Status">
</p>

<p>
  <a href="https://marketpulse-ai-fhhgktfjqm4zuv2gryrvhy.streamlit.app/">
    <img src="https://img.shields.io/badge/🚀_LIVE_DEMO-Open_MarketPulse_AI-FF4B4B?style=for-the-badge" alt="Live Demo">
  </a>
  <a href="https://github.com/AshishMandalDSAI/MarketPulse-AI">
    <img src="https://img.shields.io/badge/GitHub-View_Repository-181717?style=for-the-badge&logo=github" alt="GitHub Repository">
  </a>
</p>

</div>

---

## 📌 Project Overview

**MarketPulse AI** is an end-to-end Marketing Mix Modeling (MMM), marketing analytics, revenue forecasting, and budget optimization platform developed as an MBA Data Science & AI final-year project.

The platform helps users transform raw business and marketing data into meaningful insights through data analysis, statistical modeling, machine learning, forecasting, and interactive dashboards.

It enables users to:

- Analyze marketing performance and business KPIs.
- Understand relationships between marketing expenditure and revenue.
- Identify marketing channel contributions and performance drivers.
- Forecast future revenue and business outcomes.
- Evaluate marketing ROI and ROAS when the required data is available.
- Explore data-driven marketing budget allocation strategies.
- Perform statistical and predictive analysis.
- Generate business insights and analytical reports.

The application follows a **dataset-first approach**, allowing users to upload different datasets without depending on fixed column names.

> **Demo Data Disclaimer:** The built-in demonstration dataset is synthetic and intended for educational and demonstration purposes. It does not represent actual company performance.

---

## 🌐 Live Application

### 🚀 Try MarketPulse AI

**[Launch MarketPulse AI – Live Demo](https://marketpulse-ai-fhhgktfjqm4zuv2gryrvhy.streamlit.app/)**

Explore the interactive application and its available analytics features.

**Deployment note:** The hosted application is a separate deployment. The upgraded ULTIMATE FINAL release and PART 01–06 changes described in this repository have not been deployed to the live application. Therefore, the live demo may not include every feature documented in this release.

---

## 🆕 Latest Upgrade — PART 01–06

The project has been extended through six implementation parts while preserving the existing baseline functionality.

The upgrade introduces improved data handling, universal column-role mapping, expanded statistical analysis, marketing effectiveness modeling, channel contribution analysis, resource optimization, scenario simulation, and forecasting improvements.

### Application Navigation

| Page | Module | Description |
|---|---|---|
| 00 | Upload Data | Dataset upload, quality checks, universal column-role mapping, analysis target selection, currency detection, cleaning, and analysis |
| 01 | Statistics | Descriptive, Inferential, and Predictive analytics |
| 02 | Executive Overview | Business KPIs and executive-level overview charts |
| 03 | Marketing Performance | Marketing channel and performance trend analysis |
| 04 | Marketing Effectiveness Model | OLS driver response model, HC3 errors, chronological holdout, and model quality grading |
| 05 | Channel Performance & Contribution | Channel contribution, response curves, and supported ROI/ROAS analysis |
| 06 | Marketing Resource Optimizer | Spend or exposure allocation, feasibility diagnostics, remedies A–D, and optional reserve |
| 07 | Marketing Scenario Simulator | Channel-level and total-budget scenarios with sensitivity analysis |
| 08 | Forecast | Revenue forecasting with walk-forward evaluation |
| 09–15 | Advanced Analytics | AI Marketing Analyst, Recommendations, Reports, Predictions, Anomalies & Drivers, Decision Intelligence, and Prescriptive Analytics |

### Analytics Modes

Every uploaded dataset is classified into one of three analytics modes.

**1. Spend-Based Marketing Analytics**

- Identifies channel-wise advertising expenditure.
- Supports marketing effectiveness analysis.
- Enables monetary ROI and ROAS calculations when the required data is available.
- Supports marketing budget optimization.

**2. Exposure-Based Marketing Analytics**

- Works with exposure variables such as impressions, GRPs, clicks, and similar metrics.
- Supports exposure-effect analysis and exposure-mix optimization.
- Does not calculate monetary ROI or ROAS without channel-wise monetary spend.
- Treats trade or promotion spend as a business variable, not automatically as channel advertising spend.

**3. General Business Analytics**

- Works with datasets that do not contain identifiable marketing channels.
- Retains supported statistical and predictive analytics features.
- Clearly identifies unavailable marketing-specific capabilities.

### Currency Detection

- The demonstration dataset uses INR.
- Uploaded datasets can have currency detected from symbols or currency codes.
- Users can select the currency manually.
- When currency is unknown, the application does not automatically assume INR or USD.

### Optional Machine Learning Libraries

XGBoost, LightGBM, and CatBoost are optional dependencies.

If these libraries are not installed, the Predictive Analytics page identifies them as unavailable while keeping the rest of the application operational.

### Upgrade Documentation

| Documentation | Purpose |
|---|---|
| `FINAL_CHANGELOG.md` | Complete upgrade change history |
| `FINAL_PROJECT_STATUS.md` | Final project status |
| `FINAL_TEST_REPORT.md` | Consolidated testing evidence |
| `PART_STATUS.md` | Implementation status of each part |
| `INSTALLATION.md` | Installation and configuration |
| `WINDOWS_SETUP_GUIDE.md` | Windows setup and PowerShell instructions |
| `DOCKER_SETUP_GUIDE.md` | Docker and Docker Compose setup |
| `USER_GUIDE.md` | Application usage instructions |
| `KNOWN_LIMITATIONS.md` | Known limitations and verification status |
| `docs/ARCHITECTURE_UPGRADE.md` | Architecture of the upgraded modules |
| `DATASET_COMPATIBILITY_REPORT.md` | Dataset compatibility verification |
| `MODEL_VALIDATION_REPORT.md` | Model validation results |
| `SECURITY_AUDIT_REPORT.md` | Security audit findings |
| `PERFORMANCE_REPORT.md` | Performance measurements |
| `DEPLOYMENT_READINESS_REPORT.md` | Deployment verification status |

### Verification Summary

| Area | Verification Status |
|---|---|
| Full test suite | 3,522 passed |
| Real Kaggle dataset | Tested successfully in PART 06 |
| Official Docker base image | Blocked: Docker Hub unreachable from build environment |
| Docker with local stand-in image | Tested with limitations |
| Windows environment | Not tested directly on Windows |
| Production deployment | Not performed |

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 📊 Executive Dashboard | Interactive business KPIs, summaries, and charts |
| 📂 Flexible Data Upload | Upload CSV and supported Excel datasets |
| 🔍 Data Profiling | Inspect data types, missing values, duplicates, and dataset structure |
| 🧩 Universal Column Mapping | Identify and confirm dates, revenue measures, and marketing variables |
| 📈 Marketing Analytics | Explore marketing performance and available trends |
| 🧠 Marketing Mix Modeling | Estimate statistical relationships between marketing channels and revenue |
| 💰 ROI & ROAS Analysis | Evaluate marketing efficiency when the required data exists |
| 🎯 Budget Optimization | Explore feasible marketing budget allocation strategies |
| 🔮 Revenue Forecasting | Forecast future business performance using supported time-series data |
| 🧪 What-If Simulation | Explore alternative budget and marketing scenarios |
| 📉 Anomaly Detection | Identify unusual patterns and potential performance drivers |
| 📊 Statistical Analysis | Descriptive, inferential, and predictive analytics |
| 🤖 AI Marketing Analyst | Offline, rule-based business insights based on computed results |
| 💡 Recommendations | Review data-driven findings and potential business actions |
| 📄 Business Reports | Generate reports from available analytical results |
| 🧠 Decision Intelligence | Support business decisions using analytical evidence |
| ⚙️ Prescriptive Analytics | Explore optimization-oriented business recommendations |

---

## 📂 Flexible Dataset Support

MarketPulse AI follows a dataset-first architecture.

Users can upload different datasets without requiring predefined column names.

### Supported Formats

- CSV (`.csv`)
- Excel (`.xlsx`)

Legacy Excel `.xls` files are not supported. Save them as `.xlsx` before uploading.

### How It Works

1. **Upload Data** — Import a supported CSV or Excel dataset.
2. **Profile Dataset** — Inspect rows, columns, data types, and missing values.
3. **Check Data Quality** — Identify missing values, duplicates, and potential data issues.
4. **Identify Column Roles** — Detect possible date, revenue, marketing spend, and exposure variables.
5. **Confirm Column Mapping** — Review and confirm ambiguous column assignments.
6. **Select Analysis Targets** — Confirm the target variables before analysis.
7. **Detect Capabilities** — Determine which analytical features are supported by the uploaded dataset.
8. **Explore Insights** — Use dashboards, KPIs, statistical analysis, models, and reports.

### Dataset Compatibility

Not every dataset supports every analytical feature.

| Analysis | Required Data |
|---|---|
| Basic Data Profiling | Supported CSV or Excel dataset |
| Descriptive Statistics | Suitable numeric or categorical variables |
| Inferential Statistics | Variables appropriate for the selected statistical test |
| Predictive Analytics | Suitable target and predictor variables |
| Trend Analysis | Parseable date column and relevant numeric variables |
| Revenue Forecasting | Suitable date and numeric target data |
| Marketing Mix Modeling | Appropriate date, revenue, and marketing-channel data |
| ROI / ROAS | Revenue and suitable channel-wise monetary spend |
| Budget Optimization | Suitable trained marketing effectiveness model |
| Scenario Simulation | Supported model and scenario inputs |

When the required data is unavailable, the application identifies the relevant feature as unavailable instead of silently substituting demonstration data.

Medium- and low-confidence column-role assignments must be confirmed before they are used.

---

## 🧠 Machine Learning & Analytics

The platform includes the following analytical capabilities:

- Marketing Mix Modeling (MMM)
- Marketing Channel Contribution Analysis
- Marketing Effectiveness Modeling
- Revenue Forecasting
- Marketing ROI and ROAS Analysis
- Marketing Budget Optimization
- What-If Scenario Simulation
- Anomaly Detection
- Driver Analysis
- Descriptive Statistics
- Inferential Statistics
- Predictive Analytics
- Prescriptive Analytics
- KPI Analysis
- Decision Intelligence
- Business Reporting

### Model Interpretation

The built-in demonstration workflow uses an adstock and saturation-based Marketing Mix Modeling pipeline.

For supported uploaded datasets, the platform can use Ridge regression with chronological training and holdout evaluation.

The upgraded marketing effectiveness model uses:

- Ordinary Least Squares (OLS)
- HC3 robust standard errors
- Chronological holdout evaluation
- Model quality grading

**Important:** Regression coefficients and model outputs represent statistical relationships. They do not independently establish that marketing expenditure caused a change in revenue. Forecasts and optimization results are estimates, not guarantees.

---

## 🛠️ Technology Stack

### Programming Language
- Python

### Data Analysis
- Pandas
- NumPy

### Machine Learning & Statistics
- Scikit-learn
- Ridge Regression
- Ordinary Least Squares (OLS)
- Statistical Modeling
- Marketing Mix Modeling
- XGBoost (optional)
- LightGBM (optional)
- CatBoost (optional)

### Dashboard & Visualization
- Streamlit
- Interactive Data Visualization

### Backend & API
- FastAPI
- Uvicorn

### Data Formats
- CSV
- Excel

### Development & Deployment
- Git
- GitHub
- Docker
- Docker Compose
- Pytest

---

## 🏗️ System Architecture

```text
              ┌────────────────────────┐
              │     Marketing Data     │
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
              │ Data Quality &         │
              │ Column Role Mapping    │
              └────────────┬───────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │ Capability Detection   │
              │ & Data Validation      │
              └────────────┬───────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │ Analytics & ML Models  │
              │                        │
              │ MMM / Statistics       │
              │ Forecasting            │
              │ ROI / Budget Analysis  │
              │ Scenario Simulation   │
              └────────────┬───────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │ Insights, Decisions &  │
              │ Business Reports       │
              └────────────┬───────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │    Application Layer  │
              │                        │
              │ Streamlit Dashboard   │
              │ FastAPI Backend       │
              └────────────────────────┘
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/AshishMandalDSAI/MarketPulse-AI.git
```

### 2. Navigate to the Project

```bash
cd MarketPulse-AI
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Environment

**Windows PowerShell:**

```powershell
.\venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Install Optional Machine Learning Libraries

```bash
pip install -r requirements-optional.txt
```

XGBoost, LightGBM, and CatBoost are optional.

### Verified Environment

The full upgraded test suite was verified on Python 3.13.18 in a Linux environment.

Although the dependency ranges declare a broader supported range, Python 3.13.18 is the specifically verified environment.

For additional setup instructions, refer to:

- `INSTALLATION.md`
- `WINDOWS_SETUP_GUIDE.md`
- `DOCKER_SETUP_GUIDE.md`

---

## 🔐 Environment Configuration

Create a local environment file from the example:

```bash
cp .env.example .env
```

The `.env` file is ignored by Git and Docker.

### Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `MARKETPULSE_ENV` | `development` | Application environment |
| `MARKETPULSE_API_KEY` | Empty | API authentication key |
| `CORS_ORIGINS` | Local development origins | Allowed API origins |
| `ANTHROPIC_API_KEY` | Empty | Currently unused; the AI Analyst operates offline |
| `MARKETPULSE_DASHBOARD_PASSWORD_HASH` | Empty | PBKDF2 dashboard password hash |
| `MARKETPULSE_RATE_LIMIT` | Environment-dependent | API rate-limiting control |
| `MARKETPULSE_RATE_LIMIT_UPLOAD` | `10` | Upload requests per minute |
| `MARKETPULSE_RATE_LIMIT_COMPUTE` | `30` | Compute requests per minute |
| `MARKETPULSE_RATE_LIMIT_DEFAULT` | `120` | Other API requests per minute |
| `WEB_CONCURRENCY` | `1` | API worker configuration |
| `MARKETPULSE_MAX_UPLOAD_MB` | `50` | Maximum API upload size |
| `MARKETPULSE_MAX_ROWS` | `100000` | Maximum API analysis rows |

### Generate an API Key

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

**Security notes:**

- Never commit `.env` or real API credentials to GitHub.
- Production API access requires an API key.
- Production dashboard access requires a configured password hash.
- Production CORS origins should be explicitly configured.
- The API is designed for a single worker because its upload registry and rate limiter are in-memory.

---

## ▶️ Running the Application Locally

### Start the Streamlit Dashboard

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

The demonstration dataset and model are included with the project, allowing the dashboard to run without uploading a dataset.

### Start the FastAPI Backend

```bash
uvicorn api.main:app --reload
```

API address:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

### Production API Configuration

Before running a shared production deployment, configure:

```bash
export MARKETPULSE_ENV=production
export MARKETPULSE_API_KEY=<your-long-random-secret>
```

Then start the API:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

In production, the API requires authentication for protected endpoints.

---

## 🧪 Testing & Verification

Testing is an important part of the MarketPulse AI development process.

### Run the Full Test Suite

```bash
python -I -m pytest tests/ -v --tb=short
```

### Latest Verified Test Results

| Metric | Result |
|---|---:|
| Tests Collected | 3,522 |
| Tests Passed | 3,522 |
| Tests Failed | 0 |
| Tests Skipped | 0 |
| Warnings | 0 |

**Overall Result: PASS**

These results are documented in `FINAL_TEST_REPORT.md`.

### Additional Verification

- Real Kaggle dataset compatibility testing
- Data validation and column mapping checks
- Statistical analysis verification
- Predictive model testing
- Marketing effectiveness model testing
- Budget optimization and feasibility checks
- Scenario simulation testing
- API and security checks
- Performance benchmarking

**Important:** The verified test results apply to the documented clean Linux test environment. They should not be interpreted as proof that every environment or deployment configuration has been independently tested.

For Windows testing, refer to:

```powershell
scripts\windows\run_tests.ps1
```

Windows validation instructions:

```powershell
scripts\windows\validate_windows.ps1
```

---

## 🚀 Deployment

### Docker & Docker Compose

The project includes Docker configuration for running the dashboard and API.

Create the environment file:

```bash
cp .env.example .env
```

Configure production variables and start:

```bash
docker compose up --build -d
```

Optional ML dependencies can be included using:

```text
INSTALL_OPTIONAL_ML=true
```

Refer to `DOCKER_SETUP_GUIDE.md` for complete instructions.

### Docker Verification Status

The official Docker base image could not be pulled in the build environment because Docker Hub was unreachable.

The container workflow was therefore tested using a locally built stand-in base image, including optional ML testing with documented limitations.

This is not claimed as a successful verification using the official base image.

### Streamlit Community Cloud

The application can be deployed through Streamlit Community Cloud.

Typical configuration:

- Main file: `app.py`
- Dependencies: `requirements.txt`
- Python version: 3.13

For deployments requiring dashboard authentication, configure:

```text
MARKETPULSE_DASHBOARD_PASSWORD_HASH
```

under the application's Secrets settings.

**Deployment note:** The Streamlit dashboard does not call the FastAPI backend. API configuration therefore does not automatically apply to the Streamlit deployment.

Uploaded files on Streamlit Community Cloud use ephemeral storage.

See `FINAL_DEPLOYMENT_VERIFICATION.md` for the actual verification details.

---

## 📋 Dataset Requirements & Limits

| Item | Details |
|---|---|
| Supported formats | CSV and XLSX |
| Legacy XLS | Not supported |
| API upload size | 50 MB by default |
| API row limit | 100,000 rows by default |
| Dashboard upload limit | Streamlit default 200 MB |
| Anomaly detection | Samples very large datasets down to 50,000 rows and reports this in its output |
| Date requirement | Required for trend analysis and forecasting |
| Revenue requirement | Required for revenue-based marketing analysis |
| Marketing spend | Required for monetary channel ROI/ROAS and spend-based optimization |

The API limits can be configured using:

```text
MARKETPULSE_MAX_UPLOAD_MB
MARKETPULSE_MAX_ROWS
```

The dashboard and API have different upload handling and storage behavior.

---

## ⚠️ Known Limitations

MarketPulse AI is an analytical and decision-support platform. Some limitations must be considered when interpreting results.

### Analytical Limitations

- Model outputs represent statistical associations, not proof of causation.
- Forecasts and optimized budgets are estimates, not guarantees.
- The demonstration dataset is synthetic.
- Some features require specific date, revenue, or marketing variables.
- The shared driver response model does not include adstock. Its quality grade differs between the demonstration dataset and the real Kaggle dataset.

### Application Limitations

- The API upload registry and rate limiter are in-memory and per-process.
- The API is restricted to a single worker.
- Dashboard authentication uses a shared team password rather than individual user accounts and roles.
- The AI Marketing Analyst is rule-based and works offline. It is not a language model.
- `ANTHROPIC_API_KEY` is currently unused.

### Deployment & Environment Limitations

- Docker verification used a local stand-in base image because the official image could not be downloaded in the build environment.
- Windows was not tested directly on a Windows machine; the scripts were verified with PowerShell 7 on Linux.
- Production deployment has not been performed or claimed.
- At 100,000 rows, the Confirm & Analyze workflow takes approximately 255 seconds on 2 vCPUs, as documented in `PERFORMANCE_REPORT.md`.

For the complete limitation list and verification status, refer to:

- `KNOWN_LIMITATIONS.md`
- `FINAL_KNOWN_LIMITATIONS.md`
- `FINAL_DEPLOYMENT_VERIFICATION.md`

---

## 📚 Project Documentation

The repository includes documentation covering installation, architecture, testing, deployment, security, performance, and application usage.

| Documentation | Description |
|---|---|
| `README.md` | Project overview and quick start |
| `INSTALLATION.md` | Installation and configuration |
| `WINDOWS_SETUP_GUIDE.md` | Windows setup instructions |
| `DOCKER_SETUP_GUIDE.md` | Docker and Compose instructions |
| `USER_GUIDE.md` | Application user guide |
| `FINAL_CHANGELOG.md` | Upgrade changes |
| `FINAL_PROJECT_STATUS.md` | Project status |
| `FINAL_TEST_REPORT.md` | Latest test results |
| `PART_STATUS.md` | Part-wise implementation status |
| `KNOWN_LIMITATIONS.md` | Upgrade limitations |
| `FINAL_KNOWN_LIMITATIONS.md` | Baseline limitations |
| `MODEL_VALIDATION_REPORT.md` | Model validation |
| `SECURITY_AUDIT_REPORT.md` | Security audit |
| `PERFORMANCE_REPORT.md` | Performance measurements |
| `DEPLOYMENT_READINESS_REPORT.md` | Deployment readiness |
| `FINAL_DEPLOYMENT_VERIFICATION.md` | Deployment verification |
| `DATASET_COMPATIBILITY_REPORT.md` | Dataset compatibility |
| `docs/ARCHITECTURE_UPGRADE.md` | Upgrade architecture |

---

## 🎓 Project Information

**Project Name:** MarketPulse AI

**Project Domain:** Marketing Analytics, Data Science, Machine Learning & Business Intelligence

**Academic Context:** MBA – Data Science & AI Final-Year Project

**Primary Objective:** To build an integrated platform for marketing performance analysis, Marketing Mix Modeling, revenue forecasting, and data-driven budget optimization.

---

## 👨‍💻 Author

**Ashish Mandal**

MBA – Data Science & Artificial Intelligence

### Connect & Explore

- **GitHub:** [AshishMandalDSAI](https://github.com/AshishMandalDSAI)
- **Project Repository:** [MarketPulse-AI](https://github.com/AshishMandalDSAI/MarketPulse-AI)
- **Live Application:** [Launch MarketPulse AI](https://marketpulse-ai-fhhgktfjqm4zuv2gryrvhy.streamlit.app/)

---

<div align="center">

### ⭐ If you find this project interesting, consider giving it a star!

**MarketPulse AI — Turning Marketing Data into Business Intelligence.**

</div>
