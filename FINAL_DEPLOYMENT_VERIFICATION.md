# MarketPulse AI: Deployment Verification

**Release:** MarketPulse_AI_ULTIMATE_FINAL_RELEASE · **Date:** 2026-11-22

## Evidence labels

- **[FINAL]**: executed on a fresh extract of the release ZIP (the candidate build; its code is identical to the released ZIP, which differs only in report files; see `FINAL_RELEASE_STATUS.md`).
- **[CYCLE]**: executed earlier in this release cycle, on the same deployment files (Dockerfiles, entrypoint, compose) but an earlier application build.
- **[STATIC]**: established by reading the code or configuration.
- **[NOT VERIFIED]**: not checked.

## 1. Local Python (clean environment) [FINAL]

| Step | Result |
|---|---|
| Unzip to a new, empty directory | OK; `unzip -t`: no errors |
| `python3 -m venv venv && pip install -r requirements.txt` (Python 3.13.18) | OK in 14 s; 60 packages (`pip freeze` in `FINAL_DEPENDENCY_SECURITY_REPORT.md`) |
| `streamlit run app.py` | `/_stcore/health` → 200 `ok`. Chrome pass: 18/18 pages, 0 exceptions, 0 error banners, 0 console errors |
| `uvicorn api.main:app` (development) | `/health` 200, `/docs` 200, `/channels` 200, `POST /upload` 200; startup warning lists the open protected endpoints (expected without a key) |
| Production without `MARKETPULSE_API_KEY` | exits with code 1 (`RuntimeError`) |
| Production with a key | `POST /ask`: 401 without the key, 200 with it |
| Rate limit (production default) | uploads 1–10 return 200; the 11th returns **429** with `Retry-After: 60` |
| Key appears in logs | 0 occurrences |

## 2. Docker Compose [FINAL]

> **Base image caveat.** Docker Hub is unreachable from this environment, so `python:3.13-slim` is a **locally built substitute**: the host's Python 3.13.18, a minimal rootfs, stubbed apt, and a local wheel directory supplied through `PIP_NO_INDEX` / `PIP_FIND_LINKS` set **in the substitute image only**. The release Dockerfiles are unchanged standard `FROM python:3.13-slim` + `pip install -r requirements.txt`. A build against the **official** image is **[NOT VERIFIED]**; the commands are in `deploy/README.md` §2.

Configuration: production mode, a random API key, and the dashboard password hash in `.env` with `$` written as `$$`.

| Check | Result |
|---|---|
| `docker compose build` | OK (1 m 10 s) |
| `docker compose up -d` | both services **healthy** in ~10 s |
| Process identity (PID 1, both containers) | uid/gid **10001** (non-root) |
| Hash survives Compose interpolation | `pbkdf2_sha256$600000$…` intact inside the container |
| API | `/health` 200, `/docs` 200; `/ask` 401 without key, 200 with key |
| Dashboard sign-in in Chrome | before sign-in: password field, **0** navigation items; wrong password → "Incorrect password."; correct password → **18** navigation items and a Sign out button |
| Upload + confirm through the API | 200 / 200; upload folder on the host owned by 10001:10001 |
| `docker compose restart` | healthy again in 8 s; upload preserved; `GET /kpis?dataset_id=…` 200 |
| `docker compose down` | 1 s; no containers left |
| API container in production with an empty key | exits with code 1 (`RuntimeError`) |
| Secrets in logs | API key: 0 occurrences |

Earlier in the cycle, on the same entrypoint and compose files:

| Check | Result |
|---|---|
| Host-owned bind mount (`./data` owned by uid 1234) | app runs as 1234; no re-chown [CYCLE] |
| Fresh empty volume | demo artifacts initialised as uid 10001 in 15 s (15 files); restart is idempotent [CYCLE] |
| `MARKETPULSE_RUN_AS_ROOT=1` | runs as root (opt-out) [CYCLE]; covered by automated tests |
| Privilege-drop logic | real-process tests in `tests/test_ultimate_security.py` (root-owned volume handed to the app user; host ownership kept; opt-out) |

## 3. HTTPS reverse proxy

| Check | Result |
|---|---|
| `docker compose -f docker-compose.yml -f deploy/docker-compose.proxy.yml config` | valid (Compose v5.7.1) [FINAL] |
| Caddy and nginx configurations | security headers, body limits, WebSocket upgrade (nginx), `limit_req`, forwarded headers [STATIC] |
| Real certificate issuance, HTTPS end to end | **[NOT VERIFIED]** (no public DNS or ports here) |

## 4. Streamlit Community Cloud

The procedure is in `deploy/README.md` §5 (main file `app.py`, Python 3.13, the password hash in Secrets). It was **not executed** for this release, and no deployment was changed. The hosted link in README.md runs an earlier release.

## 5. Not done, by instruction

No push to GitHub, no change to any live deployment, and no external credentials were used.
