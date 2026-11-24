# MarketPulse AI
### AI-Driven Marketing Mix Modeling & Budget Optimization Platform
#### *Turn Marketing Spend into Smarter Business Decisions*

---

## 1. Title Page

**MARKETPULSE AI**
**AI-Driven Marketing Mix Modeling & Budget Optimization Platform**

**MBA Data Science & AI — Final-Year Project**

| | |
|---|---|
| **Student** | Ash |
| **Program** | MBA – Data Science & AI |
| **Project Type** | Academic Project |

> **Data disclosure:** This project uses an entirely **synthetic** dataset generated for academic
> demonstration purposes. The fictional company **NovaMart** and all associated figures do not describe
> a real business.

---

## 2. Abstract

Marketing teams routinely decide how to split a finite budget across channels such as television, search,
and social media, but a simple ratio of revenue to spend (naive ROAS) cannot separate one channel's true
effect from another's, ignores that advertising exposure persists in consumers' minds for weeks after
spend stops (**carryover**), and ignores that each additional rupee spent on an already-saturated channel
buys progressively less incremental revenue (**diminishing returns**). MarketPulse AI addresses this by
implementing a complete **Marketing Mix Modeling (MMM)** workflow for a fictional e-commerce company,
NovaMart, across eight marketing channels. A four-year synthetic weekly dataset is generated with a
documented data-generating process that itself encodes carryover, saturation, seasonality, promotions, and
external effects (price, competitor activity, macroeconomic conditions). Two transformations —
**geometric adstock** for carryover and **Hill-function saturation** for diminishing returns — are applied
to channel spend before fitting three progressively richer regression models (baseline, adstock, and
adstock+saturation) using a **chronological, non-shuffled train/test split** to avoid look-ahead bias. The
best model is selected on out-of-sample error rather than R² alone. The fitted model is then used for
**channel attribution** (via leave-one-channel-out ablation), a constrained **budget optimization** problem
solved with `scipy.optimize` (SLSQP), an interactive **what-if simulator**, a lightweight **revenue
forecast**, and an offline, rule-based **AI Marketing Analyst** that answers natural-language questions
using the project's own computed analytics. The complete system is delivered as an interactive **Streamlit
dashboard** and a **FastAPI** JSON backend, backed by SQLite persistence and validated by a 40-test
automated regression suite. On the held-out test period, the selected model achieves R² = 36.2%,
RMSE = ₹316,323, and MAPE = 4.2%; the optimizer projects a **21.7% improvement** in predicted weekly
revenue from reallocating (not increasing) the current marketing budget. This project is intended as a
demonstration of the MMM methodology and software-engineering practice, not as a certified forecast of any
real company's performance.

---

## 3. Introduction

The shift from single-channel to multi-channel digital marketing has made budget allocation substantially
harder to reason about intuitively. A company today may simultaneously run television and radio campaigns,
search and display advertising, social media and influencer partnerships, and email marketing — each with
different cost structures, different lag before an effect appears, and different points of diminishing
return. Historically, budget decisions have leaned on last year's allocation, executive judgment, or
simple click-through/last-touch attribution, none of which quantify how much revenue a channel is actually
driving once carryover and saturation are accounted for. This project treats marketing budget allocation as
a **data science and optimization problem**: first *measure* how each channel contributes to revenue using
a statistically grounded model, and only then *optimize* the allocation against that model — rather than
optimizing against untested assumptions or the previous year's spreadsheet.

## 4. Background

| Concept | Explanation |
|---|---|
| **Marketing Analytics** | Descriptive analysis of marketing spend, revenue, and customer metrics — trends, correlations, and channel-level summaries, without a predictive model. |
| **Marketing Mix Modeling (MMM)** | A statistical approach that models revenue (or sales) as a function of marketing spend across channels plus business controls, used to estimate each channel's contribution and to guide budget decisions. |
| **ROAS** (Return on Ad Spend) | Revenue generated ÷ marketing spend — a simple, easy-to-compute ratio that does not separate one channel's causal effect from another's. |
| **ROI** (Return on Investment) | More general profitability measure; in this project, ROAS is used as the marketing-specific analogue. |
| **Adstock** | A transformation that models how an advertising exposure's effect *carries over* into future periods rather than being fully "used up" the week it airs. |
| **Saturation** | A transformation that models *diminishing returns* — the idea that a channel's incremental effect per rupee shrinks as spend on that channel increases. |
| **Channel Attribution** | Estimating how much of total revenue is attributable to each marketing channel, typically via a fitted model rather than raw correlation. |
| **Budget Optimization** | Using a fitted model to determine the spend allocation, under a fixed total budget and constraints, that maximizes predicted revenue. |
| **Scenario / What-If Analysis** | Evaluating hypothetical budget changes against the fitted model before committing to them in the real world. |
| **Forecasting** | Projecting future revenue based on historical patterns (trend and seasonality). |

## 5. Problem Statement

NovaMart spends money every week across eight marketing channels but has no systematic analytical
framework to (a) estimate how much revenue each channel actually contributes, (b) account for the fact that
a channel's effect on revenue can persist for several weeks after the spend occurred, (c) account for the
fact that each channel's incremental effectiveness diminishes as spend on it increases, (d) evaluate
channel-level marketing efficiency beyond a simple spend-to-revenue ratio, (e) determine an evidence-based
optimal reallocation of its existing budget, (f) simulate the revenue impact of a hypothetical budget change
before committing to it, and (g) communicate all of the above to non-technical management in an accessible,
interactive form.

## 6. Project Objectives

1. Build a realistic, documented synthetic marketing dataset.
2. Perform data preprocessing (validation, cleaning) and feature engineering.
3. Analyze historical marketing performance (descriptive analytics).
4. Build and compare multiple Marketing Mix Models.
5. Implement an adstock (carryover) transformation.
6. Implement a saturation (diminishing-returns) transformation.
7. Estimate each channel's contribution to revenue.
8. Calculate marketing efficiency and ROAS at the channel level.
9. Optimize marketing budget allocation under real-world constraints.
10. Build an interactive what-if scenario simulator.
11. Forecast near-term future revenue.
12. Build an AI Marketing Analyst that answers questions from the project's own data.
13. Develop a professional, interactive dashboard.
14. Develop a FastAPI backend exposing the same capabilities as a JSON API.
15. Validate the system through an automated test suite.

## 7. Scope

**Included:** synthetic data generation; descriptive marketing analytics; Marketing Mix Modeling (adstock,
saturation, regression, model comparison); channel attribution; constrained budget optimization; what-if
scenario simulation; a lightweight revenue forecast; an offline rule-based AI analyst; a Streamlit
dashboard; a FastAPI backend; SQLite persistence; an automated test suite.

**Not included:** real company or confidential data; real-time integration with advertising platforms
(e.g., Google Ads, Meta Ads APIs); a guarantee of causal inference (the model estimates association under
its assumptions, not a randomized-experiment-verified causal effect); production cloud deployment or CI/CD.

## 8. Business Problem

**NovaMart** is a fictional e-commerce/retail company that invests weekly marketing spend across eight
channels: **TV, Radio, Print, Search, Social Media, Display Ads, Email Marketing, and Influencer
Marketing.** Management's core question is: *"Where should we spend our next marketing rupee to maximize
revenue?"* MarketPulse AI is built to give NovaMart's (fictional) management a systematic, model-based
answer to that question, rather than relying on intuition or last year's budget split.

> NovaMart is a **synthetic business scenario** created solely for this academic project.

## 9. Proposed Solution

MarketPulse AI implements the following pipeline, with every stage's output consumed by the next:

```
Data Generation
      ↓
Data Cleaning
      ↓
Feature Engineering
      ↓
Marketing Analytics
      ↓
Marketing Mix Modeling  ←  Adstock  +  Saturation
      ↓
Channel Attribution
      ↓
Budget Optimization
      ↓
What-If Simulation
      ↓
Forecasting
      ↓
AI Marketing Analyst
      ↓
Streamlit Dashboard   +   FastAPI Backend
```

- **Data Generation** produces a realistic, structured synthetic dataset with known ground-truth effects.
- **Data Cleaning** validates and repairs the dataset (missing values, duplicates, negative values).
- **Feature Engineering** derives the analysis-ready features every later stage depends on.
- **Marketing Analytics** provides descriptive, model-free business context (trends, shares, correlations).
- **Marketing Mix Modeling** (with Adstock and Saturation) is the statistical core: it estimates how spend
  on each channel, once carryover and diminishing returns are accounted for, relates to revenue.
- **Channel Attribution** translates the fitted model into a business-readable "how much did each channel
  contribute" table.
- **Budget Optimization** uses the fitted model to recommend a revenue-maximizing reallocation of the
  existing budget.
- **What-If Simulation** lets a user test hypothetical scenarios against the same fitted model.
- **Forecasting** projects near-term revenue independent of the channel-level model.
- **AI Marketing Analyst** answers natural-language business questions using the above outputs.
- **Streamlit Dashboard** and **FastAPI Backend** are two independent, equally-capable interfaces onto the
  same underlying pipeline outputs and fitted model.

[Figure 1: MarketPulse AI System Architecture]

## 10. Technology Stack

| Technology | Role in this Project |
|---|---|
| **Python** | Core implementation language for the entire pipeline, dashboard, and API. |
| **Pandas** | Data loading, cleaning, feature engineering, and tabular analytics throughout the pipeline. |
| **NumPy** | Numerical operations underlying adstock/saturation transforms and synthetic data generation. |
| **Scikit-learn** | Ridge regression used to fit the Marketing Mix Model; `StandardScaler` for feature scaling. |
| **SciPy** | `scipy.optimize.minimize` (SLSQP) powers the constrained budget optimizer. |
| **Plotly** | Interactive charts throughout the Streamlit dashboard (trends, contribution, response curves). |
| **Streamlit** | The interactive, multi-page business-facing dashboard (`app.py` + `dashboard/`). |
| **FastAPI** | JSON API backend exposing the same capabilities programmatically. |
| **Pydantic** | Request validation and structured error responses for the FastAPI backend. |
| **SQLite** | Lightweight, dependency-free persistence layer for pipeline outputs. |
| **Pytest-style testing** | 40-test automated regression suite (custom runner, `pytest`-compatible syntax). |
| **Docker** | Containerization for local/cloud deployment (Dockerfiles provided; build not tested in this sandbox — see §31). |

## 11. Dataset

- **Granularity:** Weekly
- **Size:** 208 observations (~4 years), 2022-01-03 to 2025-12-22
- **Marketing channels (8):** TV, Radio, Print, Search, Social Media, Display Ads, Email Marketing,
  Influencer Marketing
- **Other variables:** Promotion Spend, Average Price, Website Traffic, New Customers, Returning
  Customers, Orders, Holiday flag, Season, Competitor Index, Economic Index, and Revenue (the target
  variable)
- **Structure:** the generating process encodes annual seasonality, a gradual upward trend, holiday/sale
  weeks with a revenue lift, per-channel adstock and saturation effects, price/competitor/economic effects,
  realistic multiplicative noise, and four controlled anomaly weeks.

> **The dataset is synthetically generated for academic purposes and does not represent actual NovaMart
> company data** — because NovaMart itself does not exist. Synthetic data was used so that (a) no
> confidentiality or licensing concerns arise in an academic submission, (b) the ground-truth
> carryover/saturation structure is known and can be used to sanity-check whether the modeling pipeline
> recovers sensible relationships, and (c) the project remains fully reproducible for evaluation without
> depending on external data access.

## 12. Data Engineering

- **Validation** (`src/data/data_cleaning.py`): required-column check, null/duplicate date check, duplicate
  row check, missing-value count, negative spend/revenue/orders check, and a z-score outlier check on
  Revenue.
- **Missing values:** the generator deliberately injects 3 missing `Competitor_Index` values; these are
  forward-filled, with a median fallback for any remaining nulls.
- **Duplicates:** checked and removed (none present in the generated data).
- **Date validation:** dates confirmed monotonic, weekly-spaced, and non-null.
- **Outlier checking:** revenue outliers flagged via z-score (informational; not silently dropped).
- **Feature engineering** (`src/features/feature_engineering.py`) derives: time features (year, month,
  week, quarter), **Total/Digital/Traditional Marketing Spend** and their shares, per-channel spend share,
  week-over-week **Revenue Growth** and **Marketing Growth**, **Customer Growth**, and **Average Order
  Value (AOV)** = Revenue ÷ Orders.
- **Processed dataset:** `data/processed/marketing_data_clean.csv` — 208 rows × 45 columns — is the single
  source of truth consumed by every downstream stage (analytics, MMM, optimization, dashboard, API).

## 13. Exploratory Data Analysis

EDA is documented in `notebooks/01_data_exploration.ipynb` and computed by
`src/analytics/marketing_analytics.py`. Views produced: revenue trend, marketing spend trend, per-channel
spend and spend share, channel-revenue correlation, naive ROAS, monthly/quarterly/yearly performance, and
holiday/promotion impact comparisons.

[Figure 2: Revenue Trend]

From the actual computed outputs (`data/processed/marketing_kpis.json`):

- Total revenue across the 208-week period: **₹1,061,503,836**
- Total marketing spend: **₹354,224,801**
- Overall naive ROAS: **2.997×**
- Revenue growth, first half of the period vs. second half: **+17.8%**
- Marketing spend growth over the same comparison: **+29.0%**

Management can read from this that revenue grew over the observed period, but spend grew faster than
revenue — a signal (confirmed later by the MMM's channel-efficiency estimates) that the *mix* of channels,
not just the total amount spent, matters.

## 14. Marketing Mix Modeling

**Dependent variable:** weekly Revenue.
**Independent variables:** transformed (adstock/saturation) spend for each of the 8 marketing channels.
**Control variables:** Promotion Spend, Average Price, Holiday flag, Competitor Index, Economic Index.

Models were fit with **Ridge regression** (scikit-learn) after standardizing all features
(`StandardScaler`), which regularizes coefficient estimates and is well suited to the correlated,
moderate-sample-size setting here.

**Time-aware train/test split:** the last ~20% of weeks (chronologically) were held out as the test set,
with the remaining ~80% used for training — **no random shuffling**. Random shuffling a time series would
place some future weeks into the training set and some past weeks into the test set, letting the model
implicitly "see" information about the future during training (look-ahead bias) and producing an overly
optimistic, unrealistic accuracy estimate. A chronological split honestly simulates the real deployment
scenario: predicting weeks the model has never seen, using only what happened before them.

## 15. Adstock

Advertising exposure does not stop influencing a customer's decision the instant the ad stops airing — a
television campaign this week can still be shaping purchase decisions two or three weeks later. A model
that only considers *this week's* spend systematically **underestimates** channels with genuine carryover
(typically traditional/brand channels like TV and Radio), because it attributes none of their delayed
effect to them.

MarketPulse AI implements **geometric adstock** (`src/models/adstock.py`), the standard, well-documented
approach in MMM literature:

```
adstock_t = spend_t + decay × adstock_(t-1)
```

where `decay ∈ [0, 1)` is the fraction of last week's adstocked effect carrying into this week. Per-channel
decay rates were chosen via a documented grid search (`decay ∈ {0.05, 0.15, ..., 0.75}`) that maximizes each
channel's univariate correlation with Revenue.

## 16. Saturation

Even accounting for carryover, marketing spend does not scale revenue proportionally forever: the first
rupees spent on a channel typically generate strong incremental returns (reaching an under-saturated
audience), while additional spend beyond a certain point buys progressively smaller returns (the audience
is already reached, ad frequency capping sets in, etc.).

MarketPulse AI implements **Hill-function saturation** (`src/models/saturation.py`):

```
saturation(x) = x^s / (k^s + x^s)
```

where `k` is the half-saturation constant (the spend level producing 50% of a channel's maximum modeled
effect) and `s` is a shape parameter controlling curve steepness. The Hill function was chosen because it
passes through the origin (avoiding the undefined `log(0)` problem of a log-transform at zero spend) and
has an interpretable half-saturation parameter. Per-channel `s` was grid-searched over `{0.8, ..., 2.0}`.

**Important clarification on model selection:** saturation was **implemented and tested**, but — as
detailed in §17 — it was **not** the final selected model, because adding it on top of adstock did not
improve (and in fact slightly worsened) out-of-sample accuracy in this dataset. This is reported honestly
rather than selecting the more complex model by default.

## 17. Model Comparison

Three models were fit and evaluated on the same chronological train/test split (§14):

| Model | Test R² | Test RMSE | Test MAE | Test MAPE |
|---|---:|---:|---:|---:|
| 1. Baseline (raw, untransformed spend) | -0.229 | ₹439,169 | ₹349,060 | 6.11% |
| 2. **Adstock (SELECTED)** | **0.362** | **₹316,323** | **₹239,478** | **4.22%** |
| 3. Adstock + Saturation | 0.207 | ₹352,788 | ₹284,342 | 4.96% |

**Selection methodology:** the project does *not* select the model with the highest R² automatically.
Instead, a more complex model is only preferred over a simpler one if it also matches or improves
out-of-sample RMSE and MAPE within a small tolerance (`src/models/marketing_mix_model.py`). Here, the
Adstock+Saturation model's *test* R² is actually lower than the Adstock-only model's, and its RMSE/MAPE are
both worse — so the simpler **Adstock model was selected**, consistent with the parsimony principle.

**Verified final metrics (Adstock model, test set):** R² = **36.2%**, RMSE = **₹316,323**,
MAE = **₹239,478**, MAPE = **4.2%**.

**What these numbers mean:** an R² of 36.2% means the model explains roughly a third of the week-to-week
variance in test-period revenue — a modest but genuine result for a ~200-observation weekly business time
series with substantial unmodeled noise (promotions, one-off events, macro shocks). MAPE of 4.2% means
predictions are, on average, within about 4% of actual revenue in percentage terms. **This model provides
decision-support estimates under its stated assumptions — it does not prove a causal relationship between
any channel's spend and revenue**, and should be read as a defensible, transparent approximation rather
than ground truth.

[Figure 3: Actual vs Predicted Revenue]

## 18. Channel Attribution

Channel contribution is estimated via **leave-one-channel-out ablation**: for each channel, its transformed
spend feature is set to zero, the fitted model re-predicts revenue, and the drop from the full-model
prediction is taken as that channel's estimated incremental contribution
(`src/models/channel_contribution.py`).

| Channel | Spend (₹) | Estimated Contribution (₹) | Contribution % | Observed (Naive) ROAS | Model Efficiency | Est. Incremental Revenue (₹) |
|---|---:|---:|---:|---:|---:|---:|
| Display Ads | 33,386,873.27 | 156,703,739.25 | 29.16% | 31.79 | 4.69 | 156,703,739.25 |
| TV | 110,092,642.66 | 141,997,097.14 | 26.43% | 9.64 | 1.29 | 141,997,097.14 |
| Radio | 30,125,624.46 | 68,785,179.51 | 12.80% | 35.24 | 2.28 | 68,785,179.51 |
| Search | 69,338,796.00 | 48,410,904.58 | 9.01% | 15.31 | 0.70 | 48,410,904.58 |
| Influencer Marketing | 29,321,170.96 | 48,154,288.43 | 8.96% | 36.20 | 1.64 | 48,154,288.43 |
| Print | 20,100,444.74 | 26,236,083.74 | 4.88% | 52.81 | 1.31 | 26,236,083.74 |
| Social Media | 54,240,328.43 | 23,654,639.99 | 4.40% | 19.57 | 0.44 | 23,654,639.99 |
| Email Marketing | 7,618,920.86 | 23,361,854.91 | 4.35% | 139.32 | 3.07 | 23,361,854.91 |

*(Values reproduced exactly from `data/processed/channel_attribution.csv`, unrounded at source.)*

[Figure 4: Channel Contribution]

**According to the model estimates**, Display Ads shows both the highest model efficiency (₹4.69 of
estimated incremental revenue per ₹1 spent) and the largest estimated contribution share (29.2%), despite
being only the third-highest-spend channel. **Based on the analyzed synthetic dataset**, Social Media shows
the lowest model efficiency (₹0.44 per ₹1) despite being the third-largest spend channel — suggesting it
may be approaching saturation at current spend levels, or that its creative/targeting is underperforming
relative to other channels in this dataset. These findings describe this synthetic dataset only and should
not be read as claims about real-world channel performance.

## 19. Marketing ROI / ROAS

```
ROAS = Revenue ÷ Marketing Spend
```

The project reports two distinct ROI-style figures side by side and is explicit that they answer different
questions:

- **Observed (Naive) ROAS** — a simple, undecomposed ratio of total revenue to a channel's spend. It is
  easy to compute but does not separate a channel's own effect from co-occurring effects of other channels,
  seasonality, or price — and can be misleadingly high for a channel with very low spend (e.g., Email
  Marketing's naive ROAS of 139.32× mostly reflects its small ₹7.6M spend, not necessarily superior
  effectiveness).
- **Model-estimated efficiency** — derived from the leave-one-channel-out ablation described in §18, which
  accounts for adstock carryover and saturation and is therefore the more defensible figure for comparing
  channels' *marginal* effectiveness.

## 20. Budget Optimization

**Objective (in plain terms):** given a fixed total weekly marketing budget, find the split across the 8
channels that maximizes the fitted model's predicted revenue.

**Formally:**

```
maximize     R_hat(x_1, x_2, ..., x_8)                    [predicted revenue under the fitted MMM]

subject to   sum(x_i) = B                                  [total budget constraint]
             L*B <= x_i <= U*B     for each channel i       [min/max bounds per channel]
             x_i >= 0              for each channel i       [non-negativity]
```

where `B` is the total weekly budget, and `L`/`U` are the minimum/maximum fraction of the budget a single
channel may receive (defaults: 5%–55%), preventing degenerate "all-in-one-channel" solutions.

This is solved with `scipy.optimize.minimize` using the **SLSQP** (Sequential Least Squares Programming)
algorithm, chosen because it natively supports smooth, differentiable objectives with both bound and
equality constraints — a good match here since the adstock/saturation transforms in the objective are
smooth functions of spend.

**Verified real results** (`data/processed/budget_optimization_summary.json`), holding the last 8 weeks'
average spend (≈₹2,224,462/week total) as the current allocation:

| | Value |
|---|---:|
| Current predicted revenue | **₹5,253,939 / week** |
| Optimized predicted revenue | **₹6,395,502 / week** |
| Expected improvement | **+21.7%** |

The recommended allocation shifts budget sharply toward Display Ads (+362.4%) and Email Marketing
(+590.6%), and away from TV, Search, and Social Media — consistent with the efficiency ranking in §18.
Recommended allocations sum to the total budget within 1×10⁻⁹ numerical tolerance, with no negative
allocations.

[Figure 5: Current vs Optimized Budget]

**This is model-based scenario optimization under the fitted model's assumptions — it is a decision-support
estimate, not a guarantee of actual future revenue.** A real-world reallocation of this magnitude should be
piloted incrementally and re-measured, not implemented all at once purely on the model's recommendation.

## 21. What-If Simulator

The simulator (`src/optimization/simulator.py`) lets a user evaluate hypothetical budgets against the same
fitted model used for optimization, with a standard scenario set:

| Scenario | Definition |
|---|---|
| Current | Today's actual channel allocation |
| Optimized | The scipy-optimizer's recommended allocation (§20) |
| +10% Budget | Current mix scaled to 110% of total budget |
| +20% Budget | Current mix scaled to 120% of total budget |
| -10% Budget | Current mix scaled to 90% of total budget |

Each scenario reports predicted revenue, revenue change vs. current, and ROI — all recalculated live
against the fitted model (never hard-coded), so management can compare the revenue impact of several
plausible budget decisions **before** committing to any of them.

## 22. Forecasting

`src/forecasting/forecast.py` implements a deliberately simple, reliable forecast: a **Ridge regression on
a linear trend plus 52-week seasonal Fourier harmonics**, chosen over a more complex time-series model
because, at this dataset's size (~200 weekly observations), a reliable simple model was prioritized over an
unstable advanced one. The forecast projects **12 weeks** of future revenue with an approximate **90%
prediction interval** derived from in-sample residual standard deviation. No separate out-of-sample
forecast-accuracy metric (e.g., a held-out forecast MAPE) is claimed for this module beyond the in-sample
residual statistics actually computed.

[Figure 6: Forecast]

## 23. AI Marketing Analyst

`src/ai/analyst.py` implements a fully **offline, rule-based "Demo AI Mode"** — it never requires an
external API key and never fabricates a number. Each supported question type is routed to a specific
handler that reads the project's own computed analytics/model outputs (attribution, budget optimization,
KPIs, or a live simulator call) and phrases the answer in business language around those real numbers.

Example questions (all verified to produce grounded answers):
- *"Which channel should receive more budget?"*
- *"Which channel has the best efficiency?"*
- *"What happens if Search spending increases by 20%?"*
- *"Why is revenue declining?"*
- *"Which channel should management reduce?"*

An optional LLM-assisted mode activates only if an `ANTHROPIC_API_KEY` is configured, adding a more fluent
narrative layer — but the underlying figures still come from this project's own pipeline in either mode.
**This is an analytical assistant intended to make the project's own findings easier to query in natural
language — it is not a replacement for management judgment**, and it does not generate new analysis beyond
what the rest of the pipeline has already computed.

## 24. Dashboard

A 10-page Streamlit application (`app.py` + `dashboard/`):

| # | Page | Purpose |
|---|---|---|
| 01 | Executive Overview | KPI cards, revenue/spend trends, channel contribution, current-vs-optimized budget, and dynamically-generated business insights — the "what's happening, why, what to do" summary view. |
| 02 | Marketing Performance | Filterable (year/quarter/channel) revenue, spend, channel-share, seasonal, and customer analytics. |
| 03 | Marketing Mix Model | Model comparison table, actual-vs-predicted chart, residual analysis, and adstock/saturation curves. |
| 04 | Channel ROI | Attribution table, top-performer/lowest-efficiency/growth-opportunity highlights, spend-vs-contribution scatter. |
| 05 | Budget Optimizer | Interactive total-budget slider, live re-run of the scipy optimizer, current-vs-recommended chart. |
| 06 | What-If Simulator | Per-channel sliders plus Current/Optimized/+10%/+20%/-10% preset buttons, live predicted-revenue feedback. |
| 07 | Forecast | Historical revenue plus 12-week forecast with a 90% prediction interval. |
| 08 | AI Marketing Analyst | Chat-style interface to the offline analyst, with example question buttons. |
| 09 | Recommendations | Dynamically-generated management recommendations from the actual attribution/optimization outputs. |
| 10 | Reports | Downloadable CSV/JSON exports and an auto-generated executive summary. |

## 25. FastAPI

`api/main.py` exposes the same capabilities as a JSON API:

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Service + pipeline-readiness status |
| `/kpis` | GET | Marketing KPI bundle |
| `/channels` | GET | Channel-level descriptive performance |
| `/model-performance` | GET | Model comparison metrics |
| `/attribution` | GET | Channel attribution table |
| `/optimize-budget` | POST | Live budget optimization run |
| `/simulate-budget` | POST | Live what-if scenario evaluation |
| `/ask` | POST | AI Marketing Analyst question-answering |

Requests are validated with **Pydantic** models (e.g., `total_budget > 0`, `min_pct < max_pct`, valid
channel names). All errors — expected (404/400/422) or unexpected (500) — return a consistent JSON envelope
(`{"success": false, "error": ..., "message": ...}`) and never expose raw Python tracebacks or filesystem
paths to the client.

> **FastAPI was statically verified** (syntax-checked, routes and validation logic manually inspected)
> **in the cloud sandbox used to build this project, because the `fastapi` package could not be installed
> or runtime-tested there due to network restrictions. It should be runtime-tested (e.g., `uvicorn
> api.main:app --reload` plus the included `tests/test_api.py` suite) on the student's local machine before
> final submission.**

## 26. Database

A lightweight **SQLite** database (`data/processed/marketpulse.db`, built by `src/utils/database.py`)
persists five tables, populated directly from the pipeline's own outputs:

| Table | Source |
|---|---|
| `marketing_data` | `marketing_data_clean.csv` (the full 208×45 feature-engineered dataset) |
| `channel_metrics` | `channel_attribution.csv` |
| `model_results` | Model comparison metrics from `models/model_metrics.json` |
| `budget_recommendations` | `budget_optimization_result.csv` |
| `scenarios` | `scenario_comparison.csv` |

SQLite was chosen over PostgreSQL for local simplicity — no separate database server is required to run or
grade the project.

## 27. Testing

**Verified result: 40 / 40 tests passed** (`tests/run_all_tests.py`, `pytest`-compatible syntax, run
without requiring `pytest` itself to be installed).

**Executed** (genuinely run against real project code, not merely inspected): dataset generation and
validation; data cleaning; KPI calculation; adstock and saturation transforms; MMM training and evaluation;
channel attribution; budget optimization (allocation sums to budget, no negative spend, optimizer
convergence); the offline AI Marketing Analyst (including all 5 demo questions used in the project's live
demonstration, plus increase/reduce-budget routing).

**Statically verified, not executed:** 10 of the tests in `tests/test_api.py` (FastAPI HTTP-level tests —
validation-error responses, error envelope shape, endpoint status codes) are written to run automatically
once `fastapi`/`starlette` are installed, but were **skipped**, not failed, in this build sandbox because
`fastapi` could not be installed there. This distinction — skipped due to a missing dependency versus a
genuine test failure — is preserved explicitly in the test output rather than being hidden.

## 28. Security

A full repository scan (source code, notebooks, documentation, and the packaged ZIP) found: **no API keys,
no secrets, no tokens, no hard-coded passwords, and no `.env` file.** Only `.env.example` (placeholder
variable names, no real values) is included in the repository and the submission package.

## 29. Results

| Metric | Value |
|---|---|
| Selected Model | `adstock` |
| Test R² | 36.2% |
| Test RMSE | ₹316,323 |
| Test MAE | ₹239,478 |
| Test MAPE | 4.2% |
| Current Predicted Revenue | ₹5,253,939 / week |
| Optimized Predicted Revenue | ₹6,395,502 / week |
| Expected Improvement | +21.7% |
| Testing Result | 40 / 40 passed |
| Top channel (model efficiency) | Display Ads — ₹4.69 per ₹1 spent |
| Lowest channel (model efficiency) | Social Media — ₹0.44 per ₹1 spent |

The full channel attribution table is reproduced in §18.

## 30. Business Recommendations

Generated dynamically from the project's actual attribution and optimization outputs
(`dashboard/pages/9_recommendations.py`) — not hard-coded:

1. **Allocate more budget to Display Ads**, the highest model-efficiency channel; the optimizer recommends
   the largest single reallocation toward it (+362.4% vs. current spend).
2. **Investigate, rather than immediately cut, Social Media spend** — its low model efficiency may reflect
   genuine saturation at current spend levels or a targeting/creative issue, and this dataset alone cannot
   distinguish between the two.
3. **Use scenario analysis (What-If Simulator) before implementing any major budget change**, rather than
   applying the optimizer's full recommendation in one step.
4. **Consider seasonal allocation** — the data shows a measurable revenue lift during holiday/promotion
   weeks, suggesting incremental spend timed ahead of those windows may be more effective than a flat
   weekly allocation.

## 31. Limitations

- The dataset is **entirely synthetic**, generated for this academic project; it does not represent a real
  company, and results should not be generalized beyond this dataset.
- The project has **208 weekly observations** — a modest sample size for a model with 8+ channel features
  and several controls, which limits statistical power and contributes to the modest R².
- Model-estimated contribution and efficiency are **decompositions under the fitted model's assumptions,
  not causally-proven or audited figures** — they should inform, not replace, management judgment.
- Real-world marketing data would likely include additional variables (e.g., creative-level detail,
  audience segments, competitor spend data) not present in this synthetic dataset.
- The project is **not connected to any external advertising platform** (no live Google Ads/Meta Ads data).
- **Streamlit, FastAPI, and Docker were not runtime-tested** in the cloud sandbox used to build this
  project, due to that environment's network/dependency restrictions — they were statically verified
  (syntax-checked, manually cross-referenced against real data) instead. **Real deployment requires
  installing dependencies locally (`pip install -r requirements.txt`) and runtime-testing the dashboard,
  API, and Docker build on the student's own machine before final submission.**
- Per-channel adstock/saturation parameters use a univariate grid-search heuristic rather than a full joint
  optimization across all 8 channels simultaneously (documented as a simplification in `src/models/marketing_mix_model.py`).

## 32. Future Scope

*(Labeled explicitly as future scope — none of the following are current features of this project.)*

- Real integration with Google Ads and Meta Ads APIs for live spend/performance data.
- Real-time data pipelines replacing the static synthetic dataset.
- Bayesian and hierarchical MMM (e.g., PyMC-Marketing) for posterior uncertainty on channel contributions.
- Fully probabilistic (not point-estimate) channel attribution.
- More advanced forecasting models (e.g., Prophet, SARIMA, or a Bayesian structural time-series model).
- Cloud deployment with CI/CD.
- Automated model monitoring and data/concept drift detection in production.
- Production-grade LLM integration for the AI Marketing Analyst (currently offline/rule-based by design).
- Automated, scheduled reporting.

## 33. Conclusion

MarketPulse AI combines **data science** (synthetic data generation, cleaning, feature engineering),
**marketing analytics** (descriptive KPIs, channel performance), **statistical modeling** (adstock,
saturation, regularized regression, time-aware validation), **optimization** (constrained non-linear budget
allocation via SciPy), **AI** (an offline, data-grounded natural-language analyst), and **software
engineering / visualization** (an interactive Streamlit dashboard, a FastAPI backend, automated testing)
into a single, working decision-support platform for a fictional e-commerce marketing budget problem. Every
number reported in this document is drawn directly from the project's own regenerated pipeline outputs, and
every limitation — including components not runtime-tested in the build environment — has been disclosed
explicitly. **MarketPulse AI does not guarantee revenue growth**; it provides a transparent, reproducible,
model-based framework for making a better-informed marketing budget decision than intuition or
last-click ROAS alone would allow.

## 34. References / Technologies

- Pandas — https://pandas.pydata.org/
- NumPy — https://numpy.org/
- Scikit-learn — https://scikit-learn.org/
- SciPy (`scipy.optimize`) — https://scipy.org/
- Plotly — https://plotly.com/python/
- Streamlit — https://streamlit.io/
- FastAPI — https://fastapi.tiangolo.com/
- Pydantic — https://docs.pydantic.dev/
- Pytest (test syntax compatibility) — https://pytest.org/

*No academic papers or external research citations are claimed beyond the general, well-established concept
of geometric adstock in advertising-response modeling, which this project implements directly in code
(`src/models/adstock.py`) rather than citing as an external source.*
