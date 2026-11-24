> **SUPERSEDED (2026-11-22)** by the MarketPulse_AI_ULTIMATE_FINAL_RELEASE documents: `FINAL_RELEASE_STATUS.md` and `FINAL_CHANGELOG.md`. These notes describe the previous release (`MarketPulse_AI_FINAL_RELEASE_VERIFIED.zip`) and are kept for traceability.

# MarketPulse AI — Final Release Notes

**Release file:** `MarketPulse_AI_FINAL_RELEASE_VERIFIED.zip` (top folder `MarketPulse_AI_FINAL_RELEASE/`)
**Date:** 2026-11-22
**Previous release:** `MarketPulse_AI_FINAL_RELEASE.zip` (preserved unchanged)

## What MarketPulse AI is

A Marketing Mix Modeling and budget-optimisation platform: an 18-page Streamlit
dashboard plus a FastAPI backend. It profiles any uploaded CSV/XLSX dataset,
proposes column roles for the user to confirm, detects which analyses the data
supports, and runs only those: KPIs, trends, MMM with adstock/saturation,
ROI/ROAS, budget optimisation, what-if simulation, forecasting, anomaly and
driver analysis, decision intelligence, an AI analyst (offline mode or Claude),
and descriptive, inferential, predictive and prescriptive statistics. Results
are associations and estimates, not proof of causation or guaranteed outcomes.

## What changed in this release

This is a stabilisation release: no new features, no architecture change, no
dependency change. Highlights (full list with evidence in `FINAL_CHANGELOG.md`):

- **Security.** Fixed a critical upload path-traversal (arbitrary file write).
  Added a real production mode (`MARKETPULSE_ENV=production`): the API refuses to
  start without `MARKETPULSE_API_KEY`, and protected endpoints require it.
  Uploaded-dataset KPIs are now protected; the key comparison is constant-time;
  uploads are size-capped before buffering; error messages don't leak internals;
  host uploads are kept out of Docker images.
- **Upload → confirm flow.** MEDIUM/LOW column suggestions are no longer wiped
  before the user can see and accept them.
- **Dashboard.** Fixed first-visit crashes and double rendering on Executive
  Overview, Predictions and Anomalies & Drivers for uploaded datasets; the
  Predictions forecast chart now works with the forecasting engine's output.
- **Dataset-agnostic mapping.** Recognises `Meta_Ads`, `Google_Advertising`,
  `Marketing_Investment`, `Net_Sales` and similar; numeric ID columns are no
  longer suggested as dates or revenue.
- **API consistency.** `/simulate-budget` accepts the channel names returned by
  `/channels`.
- **Formats.** `.xls` is no longer advertised (it never worked without `xlrd`);
  users are told to save as `.xlsx`.
- **Warnings: 14 → 0.** The demo model was re-serialised with scikit-learn 1.9.1
  (numerically identical); infinity handling was fixed in three statistics paths.
- **Docs.** README now covers installation, environment, running, testing,
  deployment, dataset requirements and limitations. The stale "Phase 7 Part 8:
  Not Started" doc was corrected, the API reference was updated, and the earlier
  audit reports are marked superseded.

## Verification summary

| Area | Result |
|---|---|
| Full test suite (final code) | 2888 collected, 2888 passed, 0 failed, 0 skipped, 0 errors, 0 warnings |
| Live API (dev and prod modes) | 30/30 checks each; prod refuses to start without a key |
| Dashboard | 72/72 page renders (18 pages × demo + 3 uploaded datasets); live server healthy |
| Docker | Built and ran unmodified Dockerfiles on a **substitute** base image; official `python:3.13-slim` not reachable |
| Streamlit Cloud | Not verified (not reachable; nothing deployed or pushed) |

Details: `FINAL_TESTING_REPORT.md`, `FINAL_SECURITY_AUDIT_REPORT.md`,
`FINAL_DEPLOYMENT_VERIFICATION.md`, `FINAL_KNOWN_LIMITATIONS.md`,
`FINAL_RELEASE_STATUS.md`.

## Upgrade notes

- Local use needs no configuration; behaviour is unchanged except for the fixes above.
- **Any shared or internet-facing API deployment** should set
  `MARKETPULSE_ENV=production` and `MARKETPULSE_API_KEY` (see `.env.example`).
- `MARKETPULSE_ENV` must be `development` or `production`; any other value stops
  the API at startup on purpose.
- `.xls` uploads now get a clear 400 instead of a parse error.
- Clients calling `GET /kpis?dataset_id=<uploaded>` must send `X-API-Key` when a
  key is configured.
- The Streamlit Cloud app only picks up these fixes after its source repository
  is updated.
