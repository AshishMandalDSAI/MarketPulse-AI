# MarketPulse AI - Dockerfile (Streamlit dashboard service)
FROM python:3.13-slim

WORKDIR /app

# System dependencies (kept minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-optional.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Optional gradient-boosting models (XGBoost, LightGBM, CatBoost) for Statistics -> Predictive.
# Off by default, so the default image is unchanged. Enable with:
#   docker compose build --build-arg INSTALL_OPTIONAL_ML=true      (or INSTALL_OPTIONAL_ML=true in .env)
# libgomp1 is the OpenMP runtime LightGBM needs. Without these packages the app still runs and
# shows each missing model as "unavailable" with the reason.
ARG INSTALL_OPTIONAL_ML=false
RUN if [ "$INSTALL_OPTIONAL_ML" = "true" ]; then \
        apt-get update && apt-get install -y --no-install-recommends libgomp1 \
        && rm -rf /var/lib/apt/lists/* \
        && pip install --no-cache-dir -r requirements-optional.txt; \
    fi

# Unprivileged runtime user. The entrypoint starts as root only to prepare the
# data/model volumes, then drops to this user (or to the owner of a host bind
# mount) before any application code runs.
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid 10001 --home-dir /home/app --create-home --shell /usr/sbin/nologin app

COPY . .

# Demo data/model pipeline runs at container startup (not build time) via
# the entrypoint script, so bind-mounted host volumes work correctly on
# both fresh installs (empty host dirs) and existing installs (user data
# already present). See docker-entrypoint.sh for details.

EXPOSE 8501

HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
