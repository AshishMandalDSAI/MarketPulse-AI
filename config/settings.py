"""Centralized, environment-aware paths & limits for MarketPulse AI.

Everything here is computed relative to the project root or read from the
environment — never a hardcoded absolute personal path.
"""

import os
from pathlib import Path

# Project root = two levels up from this file (config/settings.py -> repo root)
BASE_DIR = Path(__file__).resolve().parents[1]

# --- Built-in / legacy demo dataset directories (backward compatible) ------
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

# --- Multi-dataset ("dataset-first") directories ----------------------------
UPLOADS_DIR = DATA_DIR / "uploads"

DEMO_DATASET_ID = "demo"


def dataset_upload_dir(dataset_id: str) -> Path:
    """Root directory for one uploaded dataset's artifacts."""
    return UPLOADS_DIR / dataset_id


def dataset_raw_path(dataset_id: str, filename: str) -> Path:
    return dataset_upload_dir(dataset_id) / "raw" / filename


def dataset_processed_dir(dataset_id: str) -> Path:
    return dataset_upload_dir(dataset_id) / "processed"


def dataset_models_dir(dataset_id: str) -> Path:
    return dataset_upload_dir(dataset_id) / "models"


def ensure_dataset_dirs(dataset_id: str) -> None:
    for d in (
        dataset_upload_dir(dataset_id) / "raw",
        dataset_processed_dir(dataset_id),
        dataset_models_dir(dataset_id),
    ):
        d.mkdir(parents=True, exist_ok=True)


# --- Runtime environment ----------------------------------------------------
# MARKETPULSE_ENV selects the security posture of the FastAPI backend:
#   development (default) — API_KEY is optional so the project runs locally
#                           with zero configuration; a startup warning lists
#                           the protected endpoints that are left open.
#   production            — API_KEY is REQUIRED. The API refuses to start
#                           without it, and protected endpoints reject any
#                           request without a valid X-API-Key header.
APP_ENV = (os.environ.get("MARKETPULSE_ENV") or "development").strip().lower()
VALID_APP_ENVS = ("development", "production")


def is_production() -> bool:
    return APP_ENV == "production"


# --- API rate limiting -------------------------------------------------------
# Per-client sliding-window limits (requests per minute). Enabled by default in
# production, disabled by default in development; MARKETPULSE_RATE_LIMIT=on|off
# overrides. Limits are per API process (the API runs with exactly one worker).
def _int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


_rl = (os.environ.get("MARKETPULSE_RATE_LIMIT") or "").strip().lower()
RATE_LIMIT_ENABLED = (_rl in ("1", "on", "true", "yes")) or (_rl == "" and APP_ENV == "production")
RATE_LIMIT_UPLOAD_PER_MIN = _int_env("MARKETPULSE_RATE_LIMIT_UPLOAD", 10)
RATE_LIMIT_COMPUTE_PER_MIN = _int_env("MARKETPULSE_RATE_LIMIT_COMPUTE", 30)
RATE_LIMIT_DEFAULT_PER_MIN = _int_env("MARKETPULSE_RATE_LIMIT_DEFAULT", 120)

# --- API auth ----------------------------------------------------------------
# Optional in development, mandatory in production (see APP_ENV above).
API_KEY = os.environ.get("MARKETPULSE_API_KEY") or os.environ.get("API_KEY") or None

# --- Performance / safety limits --------------------------------------------
MAX_ROWS = int(os.environ.get("MARKETPULSE_MAX_ROWS", "100000"))
MAX_UPLOAD_MB = int(os.environ.get("MARKETPULSE_MAX_UPLOAD_MB", "50"))
