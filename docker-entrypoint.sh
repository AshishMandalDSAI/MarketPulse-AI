#!/usr/bin/env bash
# MarketPulse AI — Docker container entrypoint
#
# 1. Privileges: the container starts as root only to prepare the data
#    volumes, then permanently drops to an unprivileged user before any
#    application code runs:
#      * if ./data is a host bind mount owned by a normal (non-root) host
#        user, the service runs as that same UID/GID, so host files keep
#        their ownership and stay editable on the host;
#      * if ./data and ./models are root-owned (fresh, Docker-created
#        directories), they are handed to the built-in `app` user (10001).
#    If the container is already started as a non-root user (compose `user:`,
#    Kubernetes runAsUser) no ownership is changed; the script only checks
#    that data/ and models/ are writable and fails with a clear message if not.
#    MARKETPULSE_RUN_AS_ROOT=1 restores the old behaviour (not recommended).
#
# 2. Initialization: creates the demo data/models on first run ONLY when
#    needed. Idempotent: skips initialization when data already exists (user
#    uploads, previous runs, or pre-populated host volumes).
#
#    Race-condition safe: uses flock(1) on a shared lock file so that when
#    multiple containers (dashboard + API) start against the same bind-mounted
#    volumes, only one runs the pipeline — the other waits for the lock and
#    then sees the sentinels already present.
#
#    Sentinel files checked:
#      data/processed/marketing_data_clean.csv  — demo dataset
#      models/mmm_model.pkl                     — trained MMM
#
#    If both exist, initialization is skipped entirely — existing user data
#    and uploads in data/uploads/ are never touched or overwritten.

set -e

DATA_SENTINEL="data/processed/marketing_data_clean.csv"
MODEL_SENTINEL="models/mmm_model.pkl"
LOCK_FILE="data/.init.lock"
DEFAULT_UID=10001
DEFAULT_GID=10001

# Ensure base directories exist before attempting the lock file
mkdir -p data/raw data/processed data/uploads models

if [ "$(id -u)" = "0" ] && [ "${MARKETPULSE_RUN_AS_ROOT:-0}" != "1" ]; then
    data_uid="$(stat -c %u data)"
    if [ "$data_uid" != "0" ]; then
        # Host bind mount owned by a real host user: run as that user.
        APP_UID="$data_uid"
        APP_GID="$(stat -c %g data)"
    else
        APP_UID="$DEFAULT_UID"
        APP_GID="$DEFAULT_GID"
    fi
    for d in data models; do
        if [ "$(stat -c %u "$d")" != "$APP_UID" ]; then
            echo "[MarketPulse] Giving $d/ to uid $APP_UID (it was owned by uid $(stat -c %u "$d"))."
            chown -R "$APP_UID:$APP_GID" "$d"
        fi
    done
    # Anything still root-owned (e.g. sub-folders created just above inside a
    # host-owned bind mount) goes to the runtime user; host-owned files are
    # left exactly as they are.
    find data models -user 0 -exec chown "$APP_UID:$APP_GID" {} +
    echo "[MarketPulse] Dropping root privileges: running as uid $APP_UID, gid $APP_GID."
    export APP_UID APP_GID
    exec python -c 'import os, sys
uid, gid = int(os.environ["APP_UID"]), int(os.environ["APP_GID"])
os.setgroups([]); os.setgid(gid); os.setuid(uid)
os.environ.setdefault("HOME", "/home/app")
if not os.access(os.environ["HOME"], os.W_OK):
    os.environ["HOME"] = "/tmp"
os.execvp(sys.argv[1], sys.argv[1:])' "$(command -v bash)" "$(readlink -f "$0")" "$@"
fi

if [ "$(id -u)" != "0" ]; then
    for d in data data/uploads models; do
        if [ ! -w "$d" ]; then
            echo "[MarketPulse] ERROR: $d/ is not writable by uid $(id -u)." >&2
            echo "[MarketPulse] Fix the host directory ownership (e.g. 'sudo chown -R $(id -u) ./data ./models')" >&2
            echo "[MarketPulse] or start the container as root so it can prepare the volume itself." >&2
            exit 1
        fi
    done
fi

# Use flock for exclusive locking so only one container initializes.
# The lock is held only during the sentinel check + pipeline run;
# once released, the other container re-checks sentinels and skips.
(
    flock -x 200

    if [ ! -f "$DATA_SENTINEL" ] || [ ! -f "$MODEL_SENTINEL" ]; then
        echo "[MarketPulse] Demo data/models not found — running initialization pipeline..."

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
        python src/utils/database.py 2>/dev/null || true  # database.py is optional

        echo "[MarketPulse] Initialization complete."
    else
        echo "[MarketPulse] Demo data and models already present — skipping initialization."
    fi

) 200>"$LOCK_FILE"

echo "[MarketPulse] Starting service as uid $(id -u): $@"
exec "$@"
