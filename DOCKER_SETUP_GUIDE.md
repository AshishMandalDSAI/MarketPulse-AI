# Docker setup guide

> **Verification status.**
>
> - **Not verified:** a build from the **official `python:3.13-slim`** image. Docker Hub
>   (`registry-1.docker.io`) is blocked by the network proxy of the environment where this release
>   was built (HTTP 403), and Debian package mirrors are unreachable there.
> - **Verified:** every step below, with a **local stand-in** for the base image (Python 3.13.18,
>   with an `apt-get` stub and an offline wheel directory). Details and evidence are in
>   `DEPLOYMENT_READINESS_REPORT.md`.
>
> Please run §5 on a machine with normal internet access to complete the verification.

## 1. What is in the box

| File | Purpose |
|---|---|
| `Dockerfile` | Dashboard image (Streamlit, port 8501). Build argument `INSTALL_OPTIONAL_ML` (default `false`). |
| `Dockerfile.api` | API image (FastAPI/uvicorn, port 8000, exactly one worker). |
| `docker-compose.yml` | Both services, `./data` and `./models` bind mounts, health checks, restart policy. |
| `deploy/docker-compose.proxy.yml`, `deploy/Caddyfile`, `deploy/nginx.conf` | Optional TLS reverse proxy with security headers. |
| `docker-entrypoint.sh` | Starts as root only to prepare the volumes, then drops to uid 10001 (or to the owner of the bind mount) before any app code runs. |
| `.dockerignore` | Keeps `.env`, secrets, venvs, caches and uploads out of the image. |

## 2. Configure (`.env`)

```bash
cp .env.example .env            # Windows: Copy-Item .env.example .env
```

| Variable | Set to |
|---|---|
| `MARKETPULSE_ENV` | `production` for anything other than a local trial |
| `MARKETPULSE_API_KEY` | A long random secret. Generate one with `python -c "import secrets; print(secrets.token_urlsafe(32))"` |
| `MARKETPULSE_DASHBOARD_PASSWORD_HASH` | The output of `python -m dashboard.components.auth`. **Write every `$` as `$$`** in `.env` for Compose. |
| `CORS_ORIGINS` | Your real front-end origin(s), never `*` |
| `INSTALL_OPTIONAL_ML` | `true` to include XGBoost, LightGBM and CatBoost (about 120 MB more); the default is `false` |

## 3. Build and run

```bash
docker compose build                                   # core image
docker compose build --build-arg INSTALL_OPTIONAL_ML=true dashboard   # or set it in .env
docker compose up -d
docker compose ps                                      # both services should become "healthy"
```

| URL | Expected |
|---|---|
| http://localhost:8501 | Dashboard sign-in, then the app |
| http://localhost:8501/_stcore/health | `ok` |
| http://localhost:8000/health | JSON, `"status": "healthy"` |
| http://localhost:8000/docs | **404 in production** (the API docs are disabled there by design since PART 06); available in development |

## 4. Optional ML models in Docker

- **Default image (`INSTALL_OPTIONAL_ML=false`):** XGBoost, LightGBM and CatBoost are *unavailable*.
  The Predictive tab lists each one with the reason and an install hint. The other 9 classifiers
  run normally. Nothing crashes.
- **`INSTALL_OPTIONAL_ML=true`:** the build installs `libgomp1` (the OpenMP runtime LightGBM needs)
  and `requirements-optional.txt`. All 12 classifiers are available.
- **Fallback is per library.** If a library is installed but cannot load (for example LightGBM
  without `libgomp1`), only that model is marked unavailable, with the exact error, and the
  others still run. This was tested in a container (`DEPLOYMENT_READINESS_REPORT.md` §4).

## 5. Verification checklist

Please run this on a machine with Docker Hub access:

```bash
docker compose build --no-cache --pull --build-arg INSTALL_OPTIONAL_ML=true
docker compose up -d && sleep 60 && docker compose ps
curl -s localhost:8000/health
curl -s -o /dev/null -w "%{http_code}\n" localhost:8501/_stcore/health
curl -s -o /dev/null -w "upload without key: %{http_code}\n" -F file=@tests/fixtures/part02/spend_weekly.csv localhost:8000/upload
curl -s -H "X-API-Key: $MARKETPULSE_API_KEY" -F file=@tests/fixtures/part02/spend_weekly.csv localhost:8000/upload | head -c 200
docker compose exec dashboard id                       # uid=10001(app)
docker compose restart && sleep 40 && docker compose ps     # healthy again; uploads under ./data persist
docker compose exec dashboard python tools/part06/optional_ml_check.py
```

Expected results:

- the upload without a key returns 401;
- both containers report healthy after the restart;
- `id` prints uid 10001;
- `optional_ml_check.py` prints `"availability"` with all three entries `true` when built with
  `INSTALL_OPTIONAL_ML=true`.

On Windows, use Docker Desktop (WSL 2) and the same commands in PowerShell, with `curl.exe`
instead of `curl`.

## 6. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Container restarts with `exec ./docker-entrypoint.sh: no such file or directory` | The script was converted to CRLF (Windows Git). `.gitattributes` prevents this; otherwise convert it back to LF. |
| Dashboard restarts with `Permission denied` on `data/` | The project was extracted from one of the PART 01–04 backup ZIPs with mode-0600 folders. Run `chmod -R u+rwX,go+rX .`. The final release ZIPs are fixed. |
| `$` in the password hash disappears | Write `$$` in `.env`. |
| LightGBM shows "cannot be loaded (libgomp.so.1…)" | Build with `INSTALL_OPTIONAL_ML=true` (it installs `libgomp1`), or ignore it: the other models still run. |
