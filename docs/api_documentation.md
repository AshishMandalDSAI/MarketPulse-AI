# MarketPulse AI — API Documentation

Base URL (local): `http://127.0.0.1:8000`

Run with: `uvicorn api.main:app --reload` (after `pip install -r requirements.txt`)
Interactive docs: `http://127.0.0.1:8000/docs`

## Authentication and security modes

Protected endpoints depend on `_require_api_key`, which compares the
`X-API-Key` header with `MARKETPULSE_API_KEY` (legacy alias `API_KEY`) using a
constant-time comparison. Behaviour depends on `MARKETPULSE_ENV`:

| `MARKETPULSE_ENV` | Key unset | Key set |
|---|---|---|
| `development` (default) | Protected endpoints accept requests; a startup warning lists them | Missing/wrong key → `401` |
| `production` | **API refuses to start** (`RuntimeError`) | Missing/wrong key → `401` |
| any other value | API refuses to start | API refuses to start |

**Protected:** `POST /upload`, `POST /datasets/{id}/confirm`,
`POST /optimize-budget`, `POST /simulate-budget`, `POST /ask`,
`GET /datasets/{id}/decisions`, `POST /datasets/{id}/decisions/run`, and
`GET /kpis?dataset_id=<uploaded id>`.

**Public:** `GET /`, `/health`, `/docs`, `/openapi.json`, `/channels`,
`/model-performance`, `/attribution`, and `GET /kpis` for the demo dataset
(no `dataset_id`, or `dataset_id=demo`). These expose only the bundled demo model.

## Dataset-first endpoints

These endpoints back the schema-agnostic upload flow. See
`docs/architecture.md` and `docs/methodology.md` for how profiling,
classification, and capability detection work.

### `POST /upload`
Multipart form upload, field name `file`. Accepts `.csv` and `.xlsx`
(read with pandas/`openpyxl`). Legacy `.xls` and any other extension return
`400` with guidance to save as `.xlsx`. The client filename is reduced to a
safe basename (directory parts and unusual characters removed) before it is
stored. Files over `MARKETPULSE_MAX_UPLOAD_MB` (default 50MB) return `413`;
empty files and files with no columns return `400`. Rows beyond
`MARKETPULSE_MAX_ROWS` (default 100,000) are truncated to the head rather than
rejected. A file that fails to parse returns `400` with a generic message (the
parser's details are logged server-side, not returned).

On success, a random 16-character hex `dataset_id` is minted, the raw
file is persisted to `data/uploads/<dataset_id>/raw/`, and the response
returns:
```json
{
  "success": true,
  "dataset_id": "a1b2c3d4e5f60718",
  "filename": "my_data.csv",
  "profile": { "n_rows": 500, "n_cols": 12, "columns": {"...": "..."}, "overall_missing_pct": 1.2 },
  "column_mapping": { "date": {"column": "Date", "confidence": 0.95, "confidence_level": "HIGH", "matched_via": "alias-exact", "needs_confirmation": false}, "...": "..." },
  "capabilities_if_defaults_accepted": { "marketing_mix_modeling": {"name": "Marketing Mix Modeling", "available": true, "reason": "..."}, "...": "..." }
}
```
`capabilities_if_defaults_accepted` is provisional — it reflects what
would be available if only the **HIGH-confidence** role guesses were
used (no MEDIUM/LOW guess is applied at this stage). `column_mapping` still
shows every MEDIUM/LOW suggestion with `needs_confirmation: true`, so the
client can accept them via `accept_suggested` on `/confirm`. The uploaded
dataframe and its raw column mapping are held in an in-memory registry
keyed by `dataset_id` until `/confirm` is called; nothing is analyzed or
persisted to `processed/`/`models/` yet.

### `POST /datasets/{dataset_id}/confirm`
Body (all fields optional):
```json
{
  "date_column": "OrderDate",
  "revenue_column": "TotalAmount",
  "channel_columns": ["GoogleSpend", "FacebookSpend"],
  "accept_suggested": ["date", "revenue", "channels"]
}
```
- `date_column` / `revenue_column`: an explicit user override for that
  role. Passing `null` explicitly clears the role (no date/revenue used);
  omitting the field leaves the auto-detected mapping as-is.
- `channel_columns`: an explicit list of columns to use as marketing
  spend/channels, replacing whatever was auto-detected.
- `accept_suggested`: role names (`"date"`, `"revenue"`, `"sales"`,
  `"cost"`, `"channels"`, etc.) whose MEDIUM/LOW-confidence auto-detected
  guess the user explicitly accepts **without** overriding the column —
  i.e. "use your suggestion as-is." Any role neither in `accept_suggested`
  nor given an explicit override, and whose confidence was below HIGH, is
  cleared and its column reclassified as a generic control/dimension
  column instead of being force-mapped.

Returns `404` if `dataset_id` was never uploaded (or the process
restarted since upload, since the pending-upload registry is in-memory
only). On success, runs the full dataset-first analysis pipeline and
persists all outputs under `data/uploads/<dataset_id>/`:
```json
{
  "success": true,
  "dataset_id": "a1b2c3d4e5f60718",
  "capabilities": { "...": "..." },
  "generic_kpis": [{"label": "Total Records", "value": "500", "note": "..."}],
  "report": ["Mapped 'OrderDate' -> date (confidence=1.0, via=user-override).", "..."],
  "date_column": "OrderDate",
  "revenue_column": "TotalAmount",
  "channel_columns": ["GoogleSpend", "FacebookSpend"]
}
```
`report` is a plain-English, ordered log of every mapping decision and
every analysis stage that ran or was skipped (and why).

### Existing analytics endpoints now accept `dataset_id`

`GET /kpis` accepts an optional `?dataset_id=...` query parameter.
Omitting it (or passing `dataset_id=demo`) returns the built-in demo
dataset's KPIs exactly as before — **fully backward compatible**.
Passing a confirmed upload's `dataset_id` instead returns that dataset's
generic KPIs. In both cases the response is `200` (never a bare 404 for
a missing marketing-KPI file) with:
```json
{
  "success": true,
  "dataset_id": "a1b2c3d4e5f60718",
  "generic_kpis": [{"label": "Total Records", "value": "500", "note": "..."}],
  "marketing_kpis_available": false,
  "marketing_kpis": {},
  "reason_unavailable": "This dataset does not have marketing-specific columns (date/revenue/channels) identified."
}
```
`generic_kpis` is always populated (once the dataset has been confirmed).
`marketing_kpis_available` reflects the dataset's `marketing_mix_modeling`
capability; when `false`, `marketing_kpis` is an empty object and
`reason_unavailable` explains why, instead of the endpoint failing.
Requesting KPIs for a `dataset_id` that hasn't been confirmed/analyzed
yet returns `404`.

## Legacy / demo-dataset endpoints

### `GET /health`
Returns service and pipeline-readiness status, e.g.:
```json
{"status": "healthy", "service": "MarketPulse AI API", "pipeline_ready": true,
 "pipeline_files": {"marketing_data_clean.csv": true, "channel_attribution.csv": true, "mmm_model.pkl": true}}
```
`status` is `"degraded"` (not a crash) if any required pipeline output is missing.

### `GET /kpis`
Documented under *Existing analytics endpoints now accept `dataset_id`* above. With no `dataset_id` (or
`dataset_id=demo`), `marketing_kpis` is the KPI bundle from `data/processed/marketing_kpis.json` (total
revenue, spend, ROAS, growth, holiday/promotion analysis) and `marketing_kpis_available` is `true` as long
as that file exists.

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
Channel names are matched case-insensitively and may use the canonical form
(`Social_Media`) or the display form returned by `GET /channels`
(`Social Media`). Unknown names, or the same channel given twice, return `400`.
Providing neither field returns `422`.
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
