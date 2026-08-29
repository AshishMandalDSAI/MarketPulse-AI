# MarketPulse AI — API Documentation

Base URL (local): `http://127.0.0.1:8000`

Run with: `uvicorn api.main:app --reload` (after `pip install -r requirements.txt`)
Interactive docs: `http://127.0.0.1:8000/docs`

## Endpoints

### `GET /health`
Returns service and pipeline-readiness status, e.g.:
```json
{"status": "healthy", "service": "MarketPulse AI API", "pipeline_ready": true,
 "pipeline_files": {"marketing_data_clean.csv": true, "channel_attribution.csv": true, "mmm_model.pkl": true}}
```
`status` is `"degraded"` (not a crash) if any required pipeline output is missing.

### `GET /kpis`
Returns the KPI bundle from `data/processed/marketing_kpis.json` (total revenue, spend, ROAS, growth,
holiday/promotion analysis).

### `GET /channels`
Returns `{"channels": [...]}` — one row per channel from `data/processed/channel_performance.csv`.

### `GET /model-performance`
Returns the model comparison (baseline / adstock / adstock+saturation) with R², RMSE, MAE, MAPE for each,
and which model was selected.

### `GET /attribution`
Returns `{"attribution": [...]}` — one row per channel from `data/processed/channel_attribution.csv`
(spend, model-estimated contribution, contribution %, naive ROAS, model efficiency).

### `POST /optimize-budget`
Body:
```json
{"total_budget": 2500000, "min_pct": 0.05, "max_pct": 0.55}
```
Validation: `total_budget > 0`; `0 <= min_pct < max_pct <= 1`; `min_pct * 8 channels` must not exceed 100%
of budget (else infeasible bounds are rejected with `422` before the optimizer ever runs).
Returns `{"summary": {...}, "comparison_table": [...]}` — a fresh optimization run against the trained MMM.
If the optimizer runs but does not report convergence, the response still returns `200` with an added
`"warning"` field rather than silently presenting an unreliable allocation as final.

### `POST /simulate-budget`
Body (either field, not both required):
```json
{"total_budget": 3000000}
```
or
```json
{"channel_allocation": {"TV": 400000, "Search": 600000, "...": "..."}}
```
Unknown channel names in `channel_allocation` return `400`. Providing neither field returns `422`.
Returns predicted revenue, revenue change, and ROI for the scenario.

### `POST /ask`
Body:
```json
{"question": "Which channel should receive more budget?"}
```
`question` must be 3–500 characters. Returns `{"question", "answer", "mode", "grounded_in_project_data"}`.
Works without any API key (Demo AI Mode); if `ANTHROPIC_API_KEY` is set, `mode` reflects LLM-assisted mode.

## Error handling

Every error response — expected (404/400/422) or unexpected (500) — uses the same JSON envelope, and the
API never returns raw Python tracebacks, file-system paths, or other internal details to the client:
```json
{"success": false, "error": "ValidationError", "message": "min_pct must be strictly less than max_pct"}
```
`error` is one of `HTTPException` (expected, e.g. missing pipeline output), `ValidationError` (bad request
body/params — Pydantic), or `InternalError` (unexpected server-side failure, logged server-side for
debugging but not exposed to the caller).

All successful responses are passed through a sanitizer that converts NumPy/Pandas scalar types to native
Python types and replaces non-finite floats (`NaN`/`Infinity`, which are invalid in strict JSON) with
`null`, so every response is guaranteed to be valid, portable JSON.
