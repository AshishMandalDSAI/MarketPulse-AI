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

import json
import logging
import math
import sys
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
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

logger = logging.getLogger("marketpulse_api")
logging.basicConfig(level=logging.INFO)

DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
MARKETING_DATA_PATH = DATA_DIR / "marketing_data_clean.csv"


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


app = FastAPI(
    title="MarketPulse AI API",
    description="AI-Driven Marketing Mix Modeling & Budget Optimization Platform",
    version="1.0.0",
    default_response_class=SafeJSONResponse,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.get("/kpis")
def get_kpis():
    return _read_json("marketing_kpis.json")


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
def optimize_budget_endpoint(req: OptimizeBudgetRequest):
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
def simulate_budget_endpoint(req: SimulateBudgetRequest):
    try:
        sim = BudgetSimulator()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Trained model not found — run the model pipeline first.")

    if req.channel_allocation:
        unknown = set(req.channel_allocation) - set(CHANNELS)
        if unknown:
            raise HTTPException(status_code=400, detail=f"Unknown channel(s) in channel_allocation: {sorted(unknown)}. Valid channels: {CHANNELS}")
        return sim.simulate_channel_allocation(req.channel_allocation)

    return sim.simulate_total_budget(req.total_budget)


@app.post("/ask")
def ask_endpoint(req: AskRequest):
    try:
        analyst = MarketingAnalyst()
    except Exception as e:
        logger.error("Analyst init failed: %s", e)
        raise HTTPException(status_code=500, detail="AI Marketing Analyst could not initialize. Ensure the analytics pipeline has been run.")
    return analyst.answer(req.question)


@app.get("/")
def root():
    return {
        "message": "MarketPulse AI API — see /docs for interactive documentation.",
        "endpoints": [
            "/health", "/kpis", "/channels", "/model-performance",
            "/attribution", "/optimize-budget (POST)", "/simulate-budget (POST)", "/ask (POST)",
        ],
    }
