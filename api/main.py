"""
MarketPulse AI - FastAPI Backend
====================================
Exposes the project's analytics, MMM, attribution, budget optimization,
simulation, and AI-analyst capabilities as a JSON API.

NOTE: This module requires `fastapi`, `pydantic`, and `uvicorn`
(see requirements.txt). Install and run with:

    pip install -r requirements.txt
    uvicorn api.main:app --reload

Then visit http://127.0.0.1:8000/docs for interactive API docs.
"""

import copy
import hmac
import json
import logging
import math
import os
import re
import sys
import threading
import traceback
import shutil
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Header, Depends
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ai.analyst import MarketingAnalyst
from src.optimization.simulator import BudgetSimulator, CHANNELS
from src.optimization.budget_optimizer import (
    load_bundle, get_recent_carry_in, optimize_budget, build_comparison_table,
)
from config import settings as app_settings
from src.data_engine.pipeline import profile_and_map, confirm_and_standardize
from src.data_engine.capability_engine import detect_capabilities
from src.data_engine.analysis_runner import run_full_analysis, validate_dataset_id, DATASET_ID_PATTERN
from src.kpi.generic_kpis import compute_generic_kpis

logger = logging.getLogger("marketpulse_api")
logging.basicConfig(level=logging.INFO)

DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
MARKETING_DATA_PATH = DATA_DIR / "marketing_data_clean.csv"

# In-memory registry of uploaded-but-not-yet-confirmed datasets: dataset_id -> {df, mapping}.
# Confirmed datasets are also persisted to disk (config.settings.dataset_processed_dir).
# Protected by _REGISTRY_LOCK for thread-safe access within a single-process deployment.
# NOTE: This lock does NOT protect across multiple worker processes. For multi-process
# deployments (e.g. gunicorn with >1 worker), use an external store (Redis, database).
_UPLOAD_REGISTRY = {}
_REGISTRY_LOCK = threading.Lock()


def _require_api_key(x_api_key: Optional[str] = Header(default=None)):
    """Protect write/compute and dataset-scoped endpoints.

    - API_KEY set: the X-API-Key header must match (constant-time comparison).
    - API_KEY unset in development: allowed (startup warning lists open endpoints).
    - API_KEY unset in production: rejected. The app normally refuses to start
      in this state; this branch is defence in depth.
    """
    expected = app_settings.API_KEY
    if not expected:
        if app_settings.is_production():
            raise HTTPException(
                status_code=503,
                detail="Server misconfigured: API authentication is required in production.",
            )
        return True
    if not x_api_key or not hmac.compare_digest(
        x_api_key.encode("utf-8"), expected.encode("utf-8")
    ):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key.")
    return True


# Endpoints that require X-API-Key whenever API_KEY is configured (always in
# production). Public endpoints: /, /health, /docs, /openapi.json, /channels,
# /model-performance, /attribution, and /kpis for the built-in demo dataset.
PROTECTED_ENDPOINTS = (
    "POST /optimize-budget", "POST /simulate-budget", "POST /ask", "POST /upload",
    "POST /datasets/{id}/confirm", "GET /kpis?dataset_id=<uploaded>",
    "GET /datasets/{id}/decisions", "POST /datasets/{id}/decisions/run",
)


# ---------------------------------------------------------------------------
# JSON safety: recursively convert NumPy/Pandas scalar types to native
# Python types and replace non-finite floats (NaN / Infinity — invalid in
# strict JSON) with None, so every response is guaranteed valid, portable
# JSON regardless of what the underlying model/optimizer produced.
# ---------------------------------------------------------------------------

def sanitize_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


class SafeJSONResponse(JSONResponse):
    """Default response class: sanitizes NumPy/NaN/Infinity before serializing."""
    def render(self, content) -> bytes:
        return json.dumps(sanitize_json(content), allow_nan=False).encode("utf-8")


# PART 06 security: the interactive docs and the OpenAPI schema are served only in development.
_DOCS_ENABLED = app_settings.APP_ENV != "production"
app = FastAPI(
    title="MarketPulse AI API",
    description="AI-Driven Marketing Mix Modeling & Budget Optimization Platform",
    version="1.0.0",
    default_response_class=SafeJSONResponse,
    docs_url="/docs" if _DOCS_ENABLED else None,
    redoc_url="/redoc" if _DOCS_ENABLED else None,
    openapi_url="/openapi.json" if _DOCS_ENABLED else None,
)

# CORS configuration:
# - Production: set CORS_ORIGINS env var to a comma-separated list of allowed origins
#   e.g. CORS_ORIGINS=https://app.example.com,https://admin.example.com
# - Development: defaults to allowing localhost origins on common dev ports.
# - WARNING: Do not use "*" in production. An empty or missing CORS_ORIGINS
#   defaults to the development allowlist below, NOT to permitting all origins.
_CORS_DEV_DEFAULTS = [
    "http://localhost:8501",   # Streamlit dashboard
    "http://localhost:3000",   # React dev server
    "http://localhost:8000",   # API self (for Swagger UI)
    "http://127.0.0.1:8501",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
]
_cors_env = os.environ.get("CORS_ORIGINS", "").strip()
_cors_origins = [o.strip() for o in _cors_env.split(",") if o.strip()] if _cors_env else _CORS_DEV_DEFAULTS

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Startup safety warning
# ---------------------------------------------------------------------------
class RateLimiter:
    """In-memory sliding-window limiter keyed by (client, endpoint class).

    Per process — correct because the API runs with exactly one worker (see
    check_security_configuration). For several API replicas, enforce limits at
    the reverse proxy / gateway instead (documented in deploy/).
    """

    MAX_TRACKED = 10_000

    def __init__(self):
        import collections
        import threading
        self._hits = collections.OrderedDict()
        self._lock = threading.Lock()

    def hit(self, key: tuple, limit: int, window: float = 60.0) -> tuple:
        import collections
        import time as _time
        now = _time.monotonic()
        with self._lock:
            q = self._hits.get(key)
            if q is None:
                q = collections.deque()
                self._hits[key] = q
                while len(self._hits) > self.MAX_TRACKED:
                    self._hits.popitem(last=False)
            else:
                self._hits.move_to_end(key)
            while q and now - q[0] >= window:
                q.popleft()
            if len(q) >= limit:
                return False, max(1, int(window - (now - q[0])) + 1)
            q.append(now)
            return True, 0

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


rate_limiter = RateLimiter()
_COMPUTE_PATHS = ("/optimize-budget", "/simulate-budget", "/ask")


def _rate_class(method: str, path: str) -> tuple:
    if path in ("/health", "/"):
        return None, 0
    if method == "POST" and path == "/upload":
        return "upload", app_settings.RATE_LIMIT_UPLOAD_PER_MIN
    if method == "POST" and (path in _COMPUTE_PATHS or path.endswith("/confirm") or path.endswith("/decisions/run")):
        return "compute", app_settings.RATE_LIMIT_COMPUTE_PER_MIN
    return "default", app_settings.RATE_LIMIT_DEFAULT_PER_MIN


_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}


@app.middleware("http")
async def security_headers_middleware(request, call_next):
    """PART 06: defence-in-depth headers on every API response, also when the API is reached
    directly rather than through the TLS proxy in deploy/ (which sets the same headers)."""
    response = await call_next(request)
    for k, v in _SECURITY_HEADERS.items():
        response.headers.setdefault(k, v)
    return response


@app.middleware("http")
async def rate_limit_middleware(request, call_next):
    if app_settings.RATE_LIMIT_ENABLED:
        cls, limit = _rate_class(request.method, request.url.path)
        if cls:
            client = request.client.host if request.client else "unknown"
            allowed, retry_after = rate_limiter.hit((client, cls), limit)
            if not allowed:
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    status_code=429,
                    content={"error": "rate_limited",
                             "message": f"Too many {cls} requests: limit is {limit} per minute. "
                                        f"Retry in {retry_after} s."},
                    headers={"Retry-After": str(retry_after)},
                )
    return await call_next(request)


def check_security_configuration() -> None:
    """Validate the security configuration at import/startup time.

    Raises RuntimeError (so uvicorn exits with a clear message) when the
    configuration would silently expose protected endpoints in production.
    """
    if app_settings.APP_ENV not in app_settings.VALID_APP_ENVS:
        raise RuntimeError(
            f"MARKETPULSE_ENV={app_settings.APP_ENV!r} is not valid. "
            f"Use one of: {', '.join(app_settings.VALID_APP_ENVS)}."
        )
    workers = os.environ.get("WEB_CONCURRENCY", "").strip()
    if workers.isdigit() and int(workers) > 1:
        msg = (f"WEB_CONCURRENCY={workers}: MarketPulse AI's API must run with exactly one worker "
               "(the upload registry and the rate limiter are kept in process memory). "
               "Scale with one API container per replica behind a gateway instead.")
        if app_settings.is_production():
            raise RuntimeError(msg)
        logger.warning(msg)
    if app_settings.is_production():
        if not app_settings.API_KEY:
            raise RuntimeError(
                "MARKETPULSE_ENV=production but no API key is configured. "
                "Set MARKETPULSE_API_KEY (or API_KEY) to a long random secret, "
                "e.g. `python -c \"import secrets; print(secrets.token_urlsafe(32))\"`. "
                "Refusing to start with unauthenticated protected endpoints."
            )
        if not os.environ.get("CORS_ORIGINS", "").strip():
            logger.warning(
                "Production mode: CORS_ORIGINS is not set, so only localhost "
                "origins are allowed. Set CORS_ORIGINS to your frontend domain(s)."
            )
        if "*" in _cors_origins:
            logger.warning("Production mode: CORS_ORIGINS contains '*'. Restrict it to explicit domains.")
        logger.info("Security: production mode, API key authentication enforced on protected endpoints; "
                    "rate limiting %s.", "on" if app_settings.RATE_LIMIT_ENABLED else "OFF")
    elif not app_settings.API_KEY:
        logger.warning(
            "Security: development mode and no API key set — protected endpoints "
            "(%s) accept unauthenticated requests. Fine for local use only. "
            "For any shared or public deployment set MARKETPULSE_ENV=production "
            "and MARKETPULSE_API_KEY.", ", ".join(PROTECTED_ENDPOINTS),
        )
    else:
        logger.info("Security: development mode, API key authentication enabled on protected endpoints.")


check_security_configuration()


# ---------------------------------------------------------------------------
# Global error handling — consistent envelope, never leak raw tracebacks,
# filesystem paths, or internal implementation details to API clients.
# ---------------------------------------------------------------------------

def _error_body(error_type: str, message: str) -> dict:
    return {"success": False, "error": error_type, "message": message}


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return SafeJSONResponse(
        status_code=exc.status_code,
        content=_error_body("HTTPException", str(exc.detail)),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Pydantic validation errors can contain deeply nested, non-JSON-safe
    # objects (e.g. ValueError instances) — reduce to plain strings only.
    messages = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", []) if p != "body")
        messages.append(f"{loc}: {err.get('msg')}" if loc else err.get("msg"))
    return SafeJSONResponse(
        status_code=422,
        content=_error_body("ValidationError", "; ".join(messages) or "Invalid request."),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled error on %s %s: %s", request.method, request.url.path, exc)
    logger.debug(traceback.format_exc())
    return SafeJSONResponse(
        status_code=500,
        content=_error_body(
            "InternalError",
            "An internal error occurred while processing this request. "
            "Please verify the analytics pipeline has been run and inputs are valid.",
        ),
    )


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class OptimizeBudgetRequest(BaseModel):
    total_budget: float = Field(..., gt=0, description="Total weekly marketing budget (INR)")
    min_pct: float = Field(0.05, ge=0, le=1)
    max_pct: float = Field(0.55, ge=0, le=1)

    @model_validator(mode="after")
    def check_bounds(self):
        if self.min_pct >= self.max_pct:
            raise ValueError("min_pct must be strictly less than max_pct")
        n_channels = 8
        if self.min_pct * n_channels > 1.0:
            raise ValueError(
                f"min_pct={self.min_pct} across {n_channels} channels exceeds 100% of budget — infeasible."
            )
        return self


class SimulateBudgetRequest(BaseModel):
    total_budget: Optional[float] = Field(None, gt=0)
    channel_allocation: Optional[dict] = Field(None, description="Explicit per-channel spend, e.g. {'TV': 500000}")

    @model_validator(mode="after")
    def check_one_provided(self):
        if self.total_budget is None and not self.channel_allocation:
            raise ValueError("Provide either total_budget or channel_allocation.")
        if self.channel_allocation:
            for k, v in self.channel_allocation.items():
                if not isinstance(v, (int, float)) or v < 0:
                    raise ValueError(f"channel_allocation['{k}'] must be a non-negative number.")
        return self


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)


class ConfirmDatasetRequest(BaseModel):
    date_column: Optional[str] = None
    revenue_column: Optional[str] = None
    channel_columns: Optional[list] = None
    accept_suggested: list = Field(default_factory=list)


def _read_json(name: str) -> dict:
    path = DATA_DIR / name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{name} not found — run the analytics pipeline first.")
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail=f"{name} exists but is not valid JSON — re-run the pipeline step that generates it.")


def _read_csv_as_records(name: str, directory: Path = DATA_DIR):
    import pandas as pd
    path = directory / name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{name} not found — run the relevant pipeline step first.")
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=500, detail=f"{name} exists but is empty — re-run the pipeline step that generates it.")
    return df.to_dict(orient="records")


def _load_trained_bundle():
    """Load the fitted MMM bundle, converting a missing/corrupt model file
    into a clean 404 instead of a raw pickle/traceback leaking to the client."""
    try:
        return load_bundle()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Trained model not found — run src/models/marketing_mix_model.py first.")
    except Exception:
        raise HTTPException(status_code=500, detail="Trained model file exists but could not be loaded — it may be corrupt. Re-run the training pipeline.")


def _load_marketing_data():
    import pandas as pd
    if not MARKETING_DATA_PATH.exists():
        raise HTTPException(status_code=404, detail="marketing_data_clean.csv not found — run the data pipeline first.")
    return pd.read_csv(MARKETING_DATA_PATH, parse_dates=["Date"])


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    pipeline_files = {
        "marketing_data_clean.csv": MARKETING_DATA_PATH.exists(),
        "channel_attribution.csv": (DATA_DIR / "channel_attribution.csv").exists(),
        "mmm_model.pkl": (MODELS_DIR / "mmm_model.pkl").exists(),
    }
    pipeline_ready = all(pipeline_files.values())
    return {
        "status": "healthy" if pipeline_ready else "degraded",
        "service": "MarketPulse AI API",
        "pipeline_ready": pipeline_ready,
        "pipeline_files": pipeline_files,
    }


@app.get("/channels")
def get_channels():
    return {"channels": _read_csv_as_records("channel_performance.csv")}


@app.get("/model-performance")
def get_model_performance():
    path = MODELS_DIR / "model_metrics.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="model_metrics.json not found — run marketing_mix_model.py first.")
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="model_metrics.json exists but is not valid JSON — re-run marketing_mix_model.py.")


@app.get("/attribution")
def get_attribution():
    return {"attribution": _read_csv_as_records("channel_attribution.csv")}


@app.post("/optimize-budget")
def optimize_budget_endpoint(req: OptimizeBudgetRequest, _auth: bool = Depends(_require_api_key)):
    bundle = _load_trained_bundle()
    df = _load_marketing_data()

    recent = df.tail(8)
    missing_spend_cols = [f"{ch}_Spend" for ch in CHANNELS if f"{ch}_Spend" not in df.columns]
    if missing_spend_cols:
        raise HTTPException(status_code=500, detail=f"Dataset is missing expected columns: {missing_spend_cols}")

    current_allocation = {ch: float(recent[f"{ch}_Spend"].mean()) for ch in CHANNELS}
    control_values = {c: float(recent[c].mean()) for c in bundle["control_cols"]}
    carry_in = get_recent_carry_in(df, bundle["channel_params"])

    try:
        result = optimize_budget(
            req.total_budget, bundle, carry_in, control_values, current_allocation,
            min_pct=req.min_pct, max_pct=req.max_pct,
        )
    except ValueError as e:  # infeasible / invalid inputs: the message is written for the user
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Budget optimization failed: %s", e)
        raise HTTPException(status_code=500, detail="Budget optimization failed to converge for the given inputs. Try different min_pct/max_pct bounds.")

    if not result.get("success", False):
        # Optimizer ran but did not converge — return it with a 200 so the
        # caller can see partial output, but flag it clearly rather than
        # silently presenting a possibly-unreliable allocation as final.
        return {"summary": result, "comparison_table": build_comparison_table(result).to_dict(orient="records"),
                "warning": "Optimizer did not report success — treat this allocation as indicative only."}

    table = build_comparison_table(result).to_dict(orient="records")
    return {"summary": result, "comparison_table": table}


@app.post("/simulate-budget")
def simulate_budget_endpoint(req: SimulateBudgetRequest, _auth: bool = Depends(_require_api_key)):
    try:
        sim = BudgetSimulator()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Trained model not found — run the model pipeline first.")

    if req.channel_allocation:
        # Accept both canonical names ("Social_Media") and the display names
        # returned by GET /channels ("Social Media"), case-insensitively.
        canonical = {c.lower(): c for c in CHANNELS}
        canonical.update({c.replace("_", " ").lower(): c for c in CHANNELS})
        allocation, unknown = {}, []
        for name, spend in req.channel_allocation.items():
            key = canonical.get(str(name).strip().lower())
            if key is None:
                unknown.append(name)
            elif key in allocation:
                raise HTTPException(status_code=400, detail=f"Channel {key!r} was given more than once in channel_allocation.")
            else:
                allocation[key] = spend
        if unknown:
            raise HTTPException(status_code=400, detail=f"Unknown channel(s) in channel_allocation: {sorted(map(str, unknown))}. Valid channels: {CHANNELS}")
        return sim.simulate_channel_allocation(allocation)

    return sim.simulate_total_budget(req.total_budget)


@app.post("/ask")
def ask_endpoint(req: AskRequest, _auth: bool = Depends(_require_api_key)):
    try:
        analyst = MarketingAnalyst()
    except Exception as e:
        logger.error("Analyst init failed: %s", e)
        raise HTTPException(status_code=500, detail="AI Marketing Analyst could not initialize. Ensure the analytics pipeline has been run.")
    return analyst.answer(req.question)


# ---------------------------------------------------------------------------
# Dataset-first endpoints: upload / confirm / dataset-scoped analytics
# ---------------------------------------------------------------------------

SUPPORTED_UPLOAD_SUFFIXES = (".csv", ".xlsx")
_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._ -]")


def _safe_upload_filename(name: Optional[str]) -> str:
    """Reduce a client-supplied filename to a safe basename.

    Strips any directory component (POSIX or Windows separators), replaces
    unusual characters, and never returns '', '.', '..' or a hidden file.
    """
    base = (name or "").replace("\\", "/").split("/")[-1]
    base = _UNSAFE_FILENAME_CHARS.sub("_", base).strip(" .")
    if not base:
        base = "upload.csv"
    suffix = Path(base).suffix.lower()
    stem = Path(base).stem.strip(" .") or "upload"
    return f"{stem[:100]}{suffix[:10]}"


def _validated_dataset_id(dataset_id: str) -> str:
    try:
        return validate_dataset_id(dataset_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...), _auth: bool = Depends(_require_api_key)):
    filename = _safe_upload_filename(file.filename)
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_UPLOAD_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail="Only .csv and .xlsx files are supported. Save legacy .xls files as .xlsx first.",
        )

    # Read at most limit+1 bytes so an oversized upload is rejected without
    # buffering the whole body in memory.
    limit = app_settings.MAX_UPLOAD_MB * 1024 * 1024
    raw_bytes = await file.read(limit + 1)
    if len(raw_bytes) > limit:
        raise HTTPException(status_code=413, detail=f"File exceeds the {app_settings.MAX_UPLOAD_MB}MB upload limit.")
    if not raw_bytes.strip():
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    dataset_id = uuid.uuid4().hex[:16]
    app_settings.ensure_dataset_dirs(dataset_id)
    raw_dir = (app_settings.dataset_upload_dir(dataset_id) / "raw").resolve()
    raw_path = (raw_dir / filename).resolve()
    if raw_path.parent != raw_dir:  # defence in depth; _safe_upload_filename already strips paths
        raise HTTPException(status_code=400, detail="Invalid file name.")
    raw_path.write_bytes(raw_bytes)

    try:
        if suffix == ".csv":
            df = pd.read_csv(raw_path)
        else:
            df = pd.read_excel(raw_path)
    except Exception as e:
        logger.info("Upload %s could not be parsed: %s", dataset_id, type(e).__name__)
        # PART 06: do not keep unparseable files on disk.
        shutil.rmtree(app_settings.dataset_upload_dir(dataset_id), ignore_errors=True)
        raise HTTPException(
            status_code=400,
            detail=f"Could not parse the uploaded {suffix} file. Check that it is a valid, "
                   f"non-empty {'CSV' if suffix == '.csv' else 'Excel workbook'} with a header row.",
        )
    if df.shape[1] == 0:
        shutil.rmtree(app_settings.dataset_upload_dir(dataset_id), ignore_errors=True)
        raise HTTPException(status_code=400, detail="The uploaded file has no columns.")

    # PART 06: rows beyond MARKETPULSE_MAX_ROWS are still dropped, but never silently.
    rows_received = int(len(df))
    warnings: list = []
    if rows_received > app_settings.MAX_ROWS:
        df = df.head(app_settings.MAX_ROWS)
        warnings.append(
            f"The file has {rows_received:,} rows; only the first {app_settings.MAX_ROWS:,} were used "
            f"(MARKETPULSE_MAX_ROWS). Rows after row {app_settings.MAX_ROWS:,} are not analysed.")

    profile, mapping = profile_and_map(df)
    with _REGISTRY_LOCK:
        _UPLOAD_REGISTRY[dataset_id] = {"df": df, "mapping": mapping}

    # Capabilities that WOULD apply if HIGH-confidence defaults are accepted
    # (no medium/low confidence roles auto-applied here — confirm-time only).
    # standardize() clears unconfirmed MEDIUM/LOW roles *in place*, so the
    # preview must run on a copy: the response and the registry must keep the
    # suggestions the user is being asked to confirm.
    provisional_dataset, _ = confirm_and_standardize(df, copy.deepcopy(mapping), confirmed_roles=set())
    provisional_caps = detect_capabilities(provisional_dataset)

    return {
        "success": True,
        "dataset_id": dataset_id,
        "filename": filename,
        "profile": profile,
        "column_mapping": mapping.to_dict(),
        "capabilities_if_defaults_accepted": provisional_caps.to_dict(),
        "rows_received": rows_received,
        "rows_used": int(len(df)),
        "rows_truncated": rows_received > int(len(df)),
        "warnings": warnings,
    }


@app.post("/datasets/{dataset_id}/confirm")
def confirm_dataset(dataset_id: str, req: ConfirmDatasetRequest, _auth: bool = Depends(_require_api_key)):
    dataset_id = _validated_dataset_id(dataset_id)

    with _REGISTRY_LOCK:
        entry = _UPLOAD_REGISTRY.get(dataset_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"No uploaded dataset found for dataset_id={dataset_id!r}. Upload it first via POST /upload.")

    # Work on a copy so each confirm call starts from the original suggestions
    # (standardize() and overrides mutate the mapping in place).
    df, mapping = entry["df"], copy.deepcopy(entry["mapping"])

    overrides = {}
    if req.date_column is not None or "date_column" in req.model_fields_set:
        overrides["date_column"] = req.date_column
    if req.revenue_column is not None or "revenue_column" in req.model_fields_set:
        overrides["revenue_column"] = req.revenue_column
    if req.channel_columns is not None:
        overrides["channel_columns"] = req.channel_columns

    confirmed_roles = set(req.accept_suggested or [])

    try:
        result = run_full_analysis(dataset_id, df=df, mapping=mapping, confirmed_roles=confirmed_roles, overrides=overrides or None)
    except Exception as e:
        logger.error("Dataset confirm/analysis failed for %s: %s", dataset_id, e)
        raise HTTPException(status_code=500, detail="Dataset analysis failed to complete. See server logs for details.")

    dataset = result["dataset"]
    return {
        "success": True,
        "dataset_id": dataset_id,
        "capabilities": result["capabilities"].to_dict(),
        "generic_kpis": [k.to_dict() for k in result["generic_kpis"]],
        "report": result["report"],
        "date_column": dataset.date_col,
        "revenue_column": dataset.revenue_col,
        "channel_columns": dataset.channels,
    }


@app.get("/kpis")
def get_kpis(dataset_id: Optional[str] = None, x_api_key: Optional[str] = Header(default=None)):
    if dataset_id and dataset_id != app_settings.DEMO_DATASET_ID:
        # KPIs of an uploaded dataset are user data: same protection as the
        # other dataset-scoped endpoints. Demo KPIs stay public.
        _require_api_key(x_api_key)
    if not dataset_id or dataset_id == app_settings.DEMO_DATASET_ID:
        marketing_kpis = {}
        try:
            marketing_kpis = _read_json("marketing_kpis.json")
            available = True
            reason = None
        except HTTPException:
            available = False
            reason = "marketing_kpis.json not found — run the analytics pipeline first."
        return {
            "success": True,
            "dataset_id": app_settings.DEMO_DATASET_ID,
            "marketing_kpis_available": available,
            "marketing_kpis": marketing_kpis,
            "reason_unavailable": reason,
            "generic_kpis": [],
        }

    dataset_id = _validated_dataset_id(dataset_id)
    processed_dir = app_settings.dataset_processed_dir(dataset_id)
    caps_path = processed_dir / "capabilities.json"
    generic_kpis_path = processed_dir / "generic_kpis.json"

    if not generic_kpis_path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id!r} has not been confirmed/analyzed yet.")

    try:
        with open(generic_kpis_path) as f:
            generic_kpis = json.load(f)
    except (json.JSONDecodeError, pd.errors.EmptyDataError):
        generic_kpis = []

    marketing_available = False
    reason = "This dataset does not have marketing-specific columns (date/revenue/channels) identified."
    if caps_path.exists():
        try:
            with open(caps_path) as f:
                caps = json.load(f)
            mmm_cap = caps.get("marketing_mix_modeling", {})
            marketing_available = bool(mmm_cap.get("available"))
            reason = None if marketing_available else mmm_cap.get("reason", reason)
        except json.JSONDecodeError:
            pass

    return {
        "success": True,
        "dataset_id": dataset_id,
        "generic_kpis": generic_kpis,
        "marketing_kpis_available": marketing_available,
        "marketing_kpis": {},
        "reason_unavailable": reason,
    }


# ---------------------------------------------------------------------------
# Decision Intelligence endpoints (Phase 7 Part 8)
# ---------------------------------------------------------------------------

@app.get("/datasets/{dataset_id}/decisions")
def get_decisions(dataset_id: str, _auth: bool = Depends(_require_api_key)):
    """Return the persisted decision intelligence report for a confirmed dataset.

    Distinguishes three states:
      - 200: a report (successful or failed) exists and is returned.
      - 404: dataset not confirmed, or no decision workflow has ever been run.
    """
    dataset_id = _validated_dataset_id(dataset_id)
    processed_dir = app_settings.dataset_processed_dir(dataset_id)

    # Dataset must have been confirmed (standardized data must exist)
    if not (processed_dir / "standardized_data.csv").exists():
        raise HTTPException(
            status_code=404,
            detail=f"Dataset {dataset_id!r} has not been confirmed yet. "
                   f"Upload and confirm via POST /upload and POST /datasets/{{id}}/confirm first.",
        )

    phase7_path = processed_dir / "phase7_decisions.json"
    if not phase7_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No decision report found for dataset {dataset_id!r}. "
                   f"Run the analysis pipeline or POST /datasets/{dataset_id}/decisions/run first.",
        )

    try:
        with open(phase7_path) as f:
            report_data = json.load(f)
    except (json.JSONDecodeError, OSError):
        raise HTTPException(
            status_code=500,
            detail="Decision report exists but could not be read — re-run the decision workflow.",
        )

    return {
        "success": True,
        "dataset_id": dataset_id,
        "decision_report": report_data,
    }


@app.post("/datasets/{dataset_id}/decisions/run")
def run_decisions(dataset_id: str, _auth: bool = Depends(_require_api_key)):
    """Run (or re-run) the decision intelligence workflow for a confirmed dataset.

    Uses persisted Phase 3–6 results — does not recompute upstream stages.
    """
    dataset_id = _validated_dataset_id(dataset_id)
    processed_dir = app_settings.dataset_processed_dir(dataset_id)

    if not (processed_dir / "standardized_data.csv").exists():
        raise HTTPException(
            status_code=404,
            detail=f"Dataset {dataset_id!r} has not been confirmed yet. "
                   f"Upload and confirm via POST /upload and POST /datasets/{{id}}/confirm first.",
        )

    # Load persisted upstream results (each may be absent)
    def _load_json(name):
        path = processed_dir / name
        if not path.exists():
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    analytics_dict = _load_json("dynamic_analytics.json")
    prediction_dict = _load_json("prediction_report.json")
    phase5_dict = _load_json("phase5_analysis.json")
    phase6_dict = _load_json("phase6_marketing.json")

    # Stale-result protection: remove any previous report before running
    phase7_path = processed_dir / "phase7_decisions.json"
    if phase7_path.exists():
        phase7_path.unlink()

    try:
        from src.decision.orchestrator import DecisionOrchestrator, OrchestrationRequest

        request = OrchestrationRequest(
            dataset_id=dataset_id,
            analytics_result=analytics_dict,
            prediction_report=prediction_dict,
            phase5_result=phase5_dict,
            marketing_result=phase6_dict,
        )
        orchestrator = DecisionOrchestrator()
        orch_report = orchestrator.run(request)
        report_data = orch_report.to_dict()

        with open(phase7_path, "w") as f:
            json.dump(report_data, f, indent=2, default=str)

        return {
            "success": True,
            "dataset_id": dataset_id,
            "status": orch_report.status,
            "decision_count": len(orch_report.decisions()),
            "decision_report": report_data,
        }
    except Exception as e:
        logger.error("Decision workflow failed for %s: %s", dataset_id, e)
        # Persist a failure marker
        try:
            with open(phase7_path, "w") as f:
                json.dump({
                    "dataset_id": dataset_id,
                    "status": "failed",
                    "error": str(e),
                }, f, indent=2)
        except Exception:
            pass
        raise HTTPException(
            status_code=500,
            detail="Decision intelligence workflow failed. "
                   "Ensure the analysis pipeline has been run and inputs are valid.",
        )


@app.get("/")
def root():
    return {
        "message": "MarketPulse AI API — see /docs for interactive documentation.",
        "endpoints": [
            "/health", "/kpis", "/channels", "/model-performance",
            "/attribution", "/optimize-budget (POST)", "/simulate-budget (POST)", "/ask (POST)",
        ],
    }
